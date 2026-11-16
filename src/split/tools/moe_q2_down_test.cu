// Compare losslessly packed Q2 down against expanded Q4, including graph replay and BigJobs.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__global__ void check_q2_dot(unsigned* errors, int cases) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned) cases) return;
    alignas(4) unsigned char packed[18];
    block_q4_0 expanded[2];
    block_q8_1 y[2];
    unsigned rng = i * 747796405u + 2891336453u;
    const uint16_t scales[] = {0, 0x8000, 1, 0x8001, 0x2400, 0xb400, 0x7bff, 0xfbff};
    const uint16_t ds = scales[i % 8];
    packed[0] = ds & 255; packed[1] = ds >> 8;
    for (int k = 0; k < 16; ++k) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        packed[2 + k] = i < 256 ? i : rng & 255;
    }
    for (int h = 0; h < 2; ++h) {
        expanded[h].d = __ushort_as_half(ds);
        for (int k = 0; k < 16; ++k) {
            const int a = 32 * h + k, b = a + 16;
            const unsigned lo = (packed[2 + a / 4] >> (2 * (a % 4))) & 3;
            const unsigned hi = (packed[2 + b / 4] >> (2 * (b % 4))) & 3;
            expanded[h].qs[k] = (lo + 7) | ((hi + 7) << 4);
        }
        int sum = 0;
        for (int k = 0; k < 32; ++k) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            sum += y[h].qs[k] = (int) (rng % 255) - 127;
        }
        const float scale = .0001f * (1 + i % 1024);
        // The rounded activation sum is intentionally not recomputed from q8.
        y[h].ds = make_half2(__float2half(scale), __float2half(sum * scale + .0002f));
    }
    for (int h = 0; h < 2; ++h) {
        const float old = unit_dot<T_Q4_0, 32>((const char*) expanded, h, y);
        const float now = unit_dot_q2_exact((const char*) packed, h, y);
        if (__float_as_uint(old) != __float_as_uint(now)) atomicAdd(errors, 1u);
    }
}

void expand(const std::vector<unsigned char>& q2, std::vector<unsigned char>& q4) {
    q4.resize(q2.size() * 2);
    for (size_t block = 0; block < q2.size() / 18; ++block) {
        const auto* src = q2.data() + block * 18;
        for (int h = 0; h < 2; ++h) {
            auto* dst = q4.data() + (block * 2 + h) * 18;
            dst[0] = src[0]; dst[1] = src[1];
            for (int k = 0; k < 16; ++k) {
                const int a = 32 * h + k, b = a + 16;
                dst[2 + k] = (((src[2 + a / 4] >> (2 * (a % 4))) & 3) + 7) |
                             ((((src[2 + b / 4] >> (2 * (b % 4))) & 3) + 7) << 4);
            }
        }
    }
}

int run(int dev, int cols, int rows, bool timing) {
    constexpr int experts = 48, guard = 256;
    const size_t row2 = cols / 64 * 18, stride2 = row2 * rows;
    const size_t output_bytes = (size_t) experts * MAXT * rows * sizeof(float) + guard;
    std::vector<void*> mem;
    auto alloc = [&](size_t bytes) { void* p; ck(cudaMalloc(&p, bytes), "alloc"); mem.push_back(p); return p; };
    auto* d2 = (char*) alloc(experts * stride2);
    auto* d4 = (char*) alloc(experts * stride2 * 2);
    auto* gu = (float*) alloc((size_t) MAXT * TOPK * 2 * cols * sizeof(float));
    auto* jobs = (Job*) alloc(experts * sizeof(Job));
    auto* count = (int*) alloc(sizeof(int));
    auto* output = (float*) alloc(output_bytes);
    auto* ep2 = (const char**) alloc(experts * sizeof(char*));
    auto* ep4 = (const char**) alloc(experts * sizeof(char*));
    std::vector<const char*> hep2(experts), hep4(experts);
    for (int e = 0; e < experts; ++e) {
        hep2[e] = d2 + ((e * 5) % experts) * stride2;
        hep4[e] = d4 + ((e * 5) % experts) * stride2 * 2;
    }
    ck(cudaMemcpy(ep2, hep2.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "pointers2");
    ck(cudaMemcpy(ep4, hep4.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "pointers4");
    MoeW w2{nullptr, stride2, row2, rows, cols, ep2};
    MoeW w4{nullptr, stride2 * 2, row2 * 2, rows, cols, ep4};
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    std::vector<unsigned char> h2(experts * stride2), h4, ref(output_bytes), got(output_bytes);
    std::vector<float> hgu((size_t) MAXT * TOPK * 2 * cols);
    std::vector<Job> hj(experts);
    int T = 1, nj = 1; bool pre = true;
    auto launch = [&](bool packed) {
        const int nb = cols / 32, rpb = DN_THREADS / nb;
        const dim3 grid((rows + rpb - 1) / rpb, std::min(experts, T * TOPK));
        const size_t smem = sizeof(float) * T * DN_THREADS;
        if (pre) k_moe_h<<<(T * TOPK * nb + 7) / 8, 256, 0, st>>>(gu, cols, T * TOPK);
        if (packed) {
            if (pre) k_moe_down<T_Q4_0, true, true><<<grid, DN_THREADS, smem, st>>>(w2, jobs, count, gu, output);
            else k_moe_down<T_Q4_0, false, true><<<grid, DN_THREADS, smem, st>>>(w2, jobs, count, gu, output);
        } else {
            if (pre) k_moe_down<T_Q4_0, true><<<grid, DN_THREADS, smem, st>>>(w4, jobs, count, gu, output);
            else k_moe_down<T_Q4_0, false><<<grid, DN_THREADS, smem, st>>>(w4, jobs, count, gu, output);
        }
        ck(cudaGetLastError(), "down launch");
    };
    auto prepare = [&](int seed) {
        for (size_t i = 0; i < h2.size(); ++i) h2[i] = (i * 13 + (i >> 7) * 17 + seed * 23) & 255;
        for (size_t i = 0; i < h2.size(); i += 18) {
            const uint16_t scale = 0x2400 + ((i / 18) % 16) * 19;
            std::memcpy(h2.data() + i, &scale, 2);
        }
        expand(h2, h4);
        for (size_t i = 0; i < hgu.size(); ++i)
            hgu[i] = seed ? .013f * ((int) ((i * 7 + (i >> 6) * 11) % 511) - 255) : 0.f;
        ck(cudaMemcpyAsync(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice, st), "q2 upload");
        ck(cudaMemcpyAsync(d4, h4.data(), h4.size(), cudaMemcpyHostToDevice, st), "q4 upload");
        ck(cudaMemcpyAsync(gu, hgu.data(), hgu.size() * sizeof(float), cudaMemcpyHostToDevice, st), "input");
        ck(cudaStreamSynchronize(st), "prepare wait");
    };
    auto set_jobs = [&] {
        for (int j = 0; j < experts; ++j) {
            hj[j].w = (j * 7 + T) % experts;
            hj[j].n = timing ? T * TOPK / nj + (j < T * TOPK % nj) : 1 + j % T;
            for (int k = 0; k < hj[j].n; ++k) {
                hj[j].y[k] = (j * 3 + k * 7) % (T * TOPK);
                hj[j].out[k] = j * MAXT + k;
            }
        }
        ck(cudaMemcpyAsync(jobs, hj.data(), experts * sizeof(Job), cudaMemcpyHostToDevice, st), "jobs");
        ck(cudaMemcpyAsync(count, &nj, sizeof(nj), cudaMemcpyHostToDevice, st), "count");
        ck(cudaStreamSynchronize(st), "jobs wait");
    };
    int checks = 0;
    if (!timing) {
        for (int seed : {0, 1}) {
            prepare(seed);
            for (T = 1; T <= MAXT; ++T) for (int requested : {0, 1, 7, 48}) {
                nj = std::min(requested, T * TOPK); set_jobs();
                for (bool p : {false, true}) for (bool graph : {false, true}) {
                    pre = p;
                    for (bool packed : {false, true}) {
                        ck(cudaMemsetAsync(output, 0xB3, output_bytes, st), "output guards");
                        if (graph) {
                            cudaGraph_t g; cudaGraphExec_t ex;
                            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "capture"); launch(packed);
                            ck(cudaStreamEndCapture(st, &g), "capture end");
                            ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                            for (int k = 0; k < 3; ++k) ck(cudaGraphLaunch(ex, st), "replay");
                            ck(cudaStreamSynchronize(st), "replay wait"); cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
                        } else launch(packed);
                        ck(cudaStreamSynchronize(st), "result wait");
                        ck(cudaMemcpy((packed ? got : ref).data(), output, output_bytes, cudaMemcpyDeviceToHost), "result");
                    }
                    if (got != ref) throw std::runtime_error("down mismatch card=" + std::to_string(dev) + " cols=" +
                        std::to_string(cols) + " T=" + std::to_string(T) + " pre=" + std::to_string(pre));
                    ++checks;
                }
            }
        }
    } else {
        prepare(1); pre = true;
        cudaEvent_t a, b; ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event");
        for (int tokens : {1, 7, 16}) {
            T = tokens; nj = T == 1 ? 10 : 48; set_jobs();
            for (bool packed : {false, true, true, false}) {
                cudaGraph_t g; cudaGraphExec_t ex;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "capture");
                for (int i = 0; i < 20; ++i) launch(packed);
                ck(cudaStreamEndCapture(st, &g), "capture end");
                ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                for (int i = 0; i < 32; ++i) ck(cudaGraphLaunch(ex, st), "warm");
                ck(cudaEventRecord(a, st), "start");
                for (int i = 0; i < 8; ++i) ck(cudaGraphLaunch(ex, st), "timing");
                ck(cudaEventRecord(b, st), "end"); ck(cudaEventSynchronize(b), "wait");
                float ms; ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
                std::cout << "card " << dev << " T " << T << " packed " << packed << " us " << ms * 1000 / 160 << '\n';
                cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
            }
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    ck(cudaStreamDestroy(st), "stream destroy"); for (void* p : mem) ck(cudaFree(p), "free");
    return checks;
}
int run_big(int dev, int cols, int rows) {
    constexpr int experts = 3, guard = 256;
    const size_t row2 = cols / 64 * 18, stride2 = row2 * rows;
    const size_t output_bytes = (size_t) experts * PF_JOB * rows * sizeof(float) + guard;
    std::vector<void*> mem;
    auto alloc = [&](size_t bytes) { void* p; ck(cudaMalloc(&p, bytes), "big alloc"); mem.push_back(p); return p; };
    auto* d2 = (char*) alloc(experts * stride2);
    auto* d4 = (char*) alloc(experts * stride2 * 2);
    auto* gu = (float*) alloc((size_t) experts * PF_JOB * 2 * cols * sizeof(float));
    auto* jobs = (BigJob*) alloc(experts * sizeof(BigJob));
    auto* count = (int*) alloc(sizeof(int));
    auto* output = (float*) alloc(output_bytes);
    auto* ep2 = (const char**) alloc(experts * sizeof(char*));
    auto* ep4 = (const char**) alloc(experts * sizeof(char*));
    std::vector<const char*> hep2(experts), hep4(experts);
    for (int e = 0; e < experts; ++e) {
        hep2[e] = d2 + ((e * 2) % experts) * stride2;
        hep4[e] = d4 + ((e * 2) % experts) * stride2 * 2;
    }
    ck(cudaMemcpy(ep2, hep2.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "big pointers2");
    ck(cudaMemcpy(ep4, hep4.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "big pointers4");
    MoeW w2{nullptr, stride2, row2, rows, cols, ep2};
    MoeW w4{nullptr, stride2 * 2, row2 * 2, rows, cols, ep4};
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "big stream");
    std::vector<unsigned char> h2(experts * stride2), h4, ref(output_bytes), got(output_bytes);
    std::vector<float> hgu((size_t) experts * PF_JOB * 2 * cols);
    std::vector<BigJob> hj(experts);
    auto launch = [&](bool packed) {
        const int rpb = DN_THREADS / (cols / 32);
        const dim3 grid((rows + rpb - 1) / rpb, experts);
        if (packed) k_moe_down_big<T_Q4_0, true><<<grid, DN_THREADS, 0, st>>>(w2, jobs, count, gu, output);
        else k_moe_down_big<T_Q4_0><<<grid, DN_THREADS, 0, st>>>(w4, jobs, count, gu, output);
        ck(cudaGetLastError(), "big launch");
    };
    int checks = 0;
    for (int seed : {0, 1}) {
        for (size_t i = 0; i < h2.size(); ++i) h2[i] = (i * 13 + (i >> 7) * 17 + seed * 23) & 255;
        for (size_t i = 0; i < h2.size(); i += 18) {
            const uint16_t scale = 0x2400 + ((i / 18) % 16) * 19;
            std::memcpy(h2.data() + i, &scale, 2);
        }
        expand(h2, h4);
        for (size_t i = 0; i < hgu.size(); ++i)
            hgu[i] = seed ? .013f * ((int) ((i * 7 + (i >> 6) * 11) % 511) - 255) : 0.f;
        ck(cudaMemcpyAsync(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice, st), "big q2 upload");
        ck(cudaMemcpyAsync(d4, h4.data(), h4.size(), cudaMemcpyHostToDevice, st), "big q4 upload");
        ck(cudaMemcpyAsync(gu, hgu.data(), hgu.size() * sizeof(float), cudaMemcpyHostToDevice, st), "big input");
        for (int nj : {0, 1, 3}) for (int n : {0, 1, 15, 16, 17, 31, 32, PF_JOB - 1, PF_JOB}) {
            for (int j = 0; j < experts; ++j) {
                hj[j].w = (j + seed) % experts;
                hj[j].n = j % 2 ? std::min(PF_JOB, n + 1) : n;
                for (int k = 0; k < hj[j].n; ++k) {
                    hj[j].y[k] = (j * 13 + k * 7) % (experts * PF_JOB);
                    hj[j].out[k] = j * PF_JOB + k;
                }
            }
            ck(cudaMemcpyAsync(jobs, hj.data(), experts * sizeof(BigJob), cudaMemcpyHostToDevice, st), "big jobs");
            ck(cudaMemcpyAsync(count, &nj, sizeof(nj), cudaMemcpyHostToDevice, st), "big count");
            ck(cudaStreamSynchronize(st), "big prepare wait");
            for (bool graph : {false, true}) {
                for (bool packed : {false, true}) {
                    ck(cudaMemsetAsync(output, 0xB3, output_bytes, st), "big output guards");
                    if (graph) {
                        cudaGraph_t g; cudaGraphExec_t ex;
                        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "big capture"); launch(packed);
                        ck(cudaStreamEndCapture(st, &g), "big capture end");
                        ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "big instantiate");
                        for (int k = 0; k < 3; ++k) ck(cudaGraphLaunch(ex, st), "big replay");
                        ck(cudaStreamSynchronize(st), "big replay wait"); cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
                    } else launch(packed);
                    ck(cudaStreamSynchronize(st), "big result wait");
                    ck(cudaMemcpy((packed ? got : ref).data(), output, output_bytes, cudaMemcpyDeviceToHost), "big result");
                }
                if (got != ref) throw std::runtime_error("big mismatch card=" + std::to_string(dev) + " cols=" +
                    std::to_string(cols) + " rows=" + std::to_string(n));
                ++checks;
            }
        }
    }
    ck(cudaStreamDestroy(st), "big stream destroy"); for (void* p : mem) ck(cudaFree(p), "big free");
    return checks;
}

}

int main(int argc, char** argv) { try {
    std::cout.setf(std::ios::unitbuf);
    int checks = 0, big_checks = 0;
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        unsigned* errors; fnx::ck(cudaMalloc((void**) &errors, 4), "errors"); fnx::ck(cudaMemset(errors, 0, 4), "clear");
        constexpr int cases = 262144;
        fnx::check_q2_dot<<<cases / 128, 128>>>(errors, cases);
        unsigned bad; fnx::ck(cudaMemcpy(&bad, errors, 4, cudaMemcpyDeviceToHost), "error count"); cudaFree(errors);
        if (bad) throw std::runtime_error("Q2 dot differs from expanded Q4: " + std::to_string(bad));
        std::cout << "card " << dev << " PASS " << 2 * cases << " exact packed dot checks\n";
        for (int cols : {64, 256, 320}) {
            int n = fnx::run(dev, cols, 257, false); checks += n;
            big_checks += fnx::run_big(dev, cols, 257);
            std::cout << "card " << dev << " cols " << cols << " PASS " << n << " exact down/guard checks\n";
        }
        if (argc == 2 && !std::strcmp(argv[1], "--bench")) fnx::run(dev, 320, 2560, true);
    }
    std::cout << "PASS 1048576 exact dot checks and " << checks << " exact down/guard checks and " << big_checks << " prefill checks\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }

// Exact packed/expanded expert WMMA outputs, with padding, empty jobs, and graph replay.
#include "../src/moe_mma.cu"
#include <iostream>
#include <cstring>
namespace fnx {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
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
int run_mma(int dev, int cols, int rows) {
    constexpr int experts = 3, guard = 256;
    const size_t row2 = cols / 64 * 18, stride2 = row2 * rows;
    const size_t output_bytes = (size_t) experts * PF_JOB * rows * sizeof(float) + guard;
    std::vector<void*> mem;
    auto alloc = [&](size_t bytes) { void* p; ck(cudaMalloc(&p, bytes), "big alloc"); mem.push_back(p); return p; };
    auto* d2 = (char*) alloc(experts * stride2);
    auto* d4 = (char*) alloc(experts * stride2 * 2);
    auto* gu = (__half*) alloc((size_t) experts * PF_JOB * cols * sizeof(__half));
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
    std::vector<__half> hgu((size_t) experts * PF_JOB * cols);
    std::vector<BigJob> hj(experts);
    auto launch = [&](bool packed) {
        const dim3 grid((rows + MM_BM - 1) / MM_BM, experts);
        constexpr size_t smem = mm_smem<MM_BM>();
        if (packed) k_moe_mma<GGML_TYPE_Q2_0, MM_BM><<<grid, MM_THREADS, smem, st>>>(w2, jobs, count, gu, cols, output);
        else k_moe_mma<GGML_TYPE_Q4_0, MM_BM><<<grid, MM_THREADS, smem, st>>>(w4, jobs, count, gu, cols, output);
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
            hgu[i] = __float2half(seed ? .013f * ((int) ((i * 7 + (i >> 6) * 11) % 511) - 255) : 0.f);
        ck(cudaMemcpyAsync(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice, st), "big q2 upload");
        ck(cudaMemcpyAsync(d4, h4.data(), h4.size(), cudaMemcpyHostToDevice, st), "big q4 upload");
        ck(cudaMemcpyAsync(gu, hgu.data(), hgu.size() * sizeof(__half), cudaMemcpyHostToDevice, st), "big input");
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
int main() { try {
    std::cout.setf(std::ios::unitbuf);
    int checks = 0;
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        for (int cols : {64, 256, 320}) {
            const int n = fnx::run_mma(dev, cols, 257);
            checks += n;
            std::cout << "card " << dev << " cols " << cols << " PASS " << n << " exact MMA/guard checks\n";
        }
    }
    std::cout << "PASS " << checks << " exact packed Q2/Q4 MMA prefill checks\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }

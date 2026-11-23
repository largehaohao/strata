// Compare the decoded IQ weight reuse to the original kernel, including graph replay.
// Synthetic packed weights have valid finite scales and exercise every codeword.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
#include <memory>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <GgmlType TA>
void launch_test(int T, int cap, const MoeW& w, const Job* jobs, const int* njobs,
                 const block_q8_1* x, float* out, cudaStream_t stream, const CacheDownCopy* copy = nullptr,
                 const int* job_idx = nullptr, int job_cap = 48) {
    const dim3 grid((w.nrows + 3) / 4, job_cap);
    if constexpr (TA == T_IQ4_XS || TA == T_Q5_K) {
        if (cap >= 1) {
            const dim3 grid16((w.nrows + 7) / 8, job_cap);
            if (cap == 2) {
                switch (T) {
#define GH16_CASE(t) case t: k_moe_gu_hoist<TA, 16, t><<<grid16, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, job_idx); break
                    GH16_CASE(1); GH16_CASE(2); GH16_CASE(3); GH16_CASE(4);
                    GH16_CASE(5); GH16_CASE(6); GH16_CASE(7); GH16_CASE(8);
                    GH16_CASE(9); GH16_CASE(10); GH16_CASE(11); GH16_CASE(12);
                    GH16_CASE(13); GH16_CASE(14); GH16_CASE(15); GH16_CASE(16);
                    default: throw std::runtime_error("invalid test window");
#undef GH16_CASE
                }
                ck(cudaGetLastError(), "gate/up hoist 16-lane launch");
                return;
            }
            switch (T) {
#define GH32_CASE(t) case t: k_moe_gu_hoist<TA, 32, t><<<grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, job_idx); break
                GH32_CASE(1); GH32_CASE(2); GH32_CASE(3); GH32_CASE(4);
                GH32_CASE(5); GH32_CASE(6); GH32_CASE(7); GH32_CASE(8);
                GH32_CASE(9); GH32_CASE(10); GH32_CASE(11); GH32_CASE(12);
                GH32_CASE(13); GH32_CASE(14); GH32_CASE(15); GH32_CASE(16);
                default: throw std::runtime_error("invalid test window");
#undef GH32_CASE
            }
            ck(cudaGetLastError(), "gate/up hoist 32-lane launch");
            return;
        }
    }
    if (cap >= 2) {
        if constexpr (TA == T_IQ2_XXS || TA == T_IQ2_S || TA == T_IQ3_XXS || TA == T_IQ3_S) {
            const dim3 grid16((w.nrows + 7) / 8, job_cap);
            const dim3 fused_grid(grid16.x * grid16.y + (copy ? copy->blocks : 0));
#define GU16_CASE(t) case t: \
    if (copy) k_moe_gu_iq<TA, t, 16, true><<<fused_grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, *copy, job_idx); \
    else k_moe_gu_iq<TA, t, 16><<<grid16, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, CacheDownCopy{}, job_idx); break
            switch (T) {
                GU16_CASE(1); GU16_CASE(2); GU16_CASE(3); GU16_CASE(4);
                GU16_CASE(5); GU16_CASE(6); GU16_CASE(7); GU16_CASE(8);
                GU16_CASE(9); GU16_CASE(10); GU16_CASE(11); GU16_CASE(12);
                GU16_CASE(13); GU16_CASE(14); GU16_CASE(15); GU16_CASE(16);
                default: throw std::runtime_error("invalid test window");
            }
#undef GU16_CASE
            ck(cudaGetLastError(), "gate/up 16-lane launch");
            return;
        }
    }
#define GU_CASE(t) case t: \
    if (cap) k_moe_gu_iq<TA, t><<<grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, CacheDownCopy{}, job_idx); \
    else k_moe_gu<TA, 32, t><<<grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out, job_idx); \
    break
    switch (T) {
        GU_CASE(1); GU_CASE(2); GU_CASE(3); GU_CASE(4);
        GU_CASE(5); GU_CASE(6); GU_CASE(7); GU_CASE(8);
        GU_CASE(9); GU_CASE(10); GU_CASE(11); GU_CASE(12);
        GU_CASE(13); GU_CASE(14); GU_CASE(15); GU_CASE(16);
        default: throw std::runtime_error("invalid test window");
    }
#undef GU_CASE
    ck(cudaGetLastError(), "gate/up launch");
}

// The down-copy branch uses global expert IDs and nonzero pool slot offsets.
// Keep guard bytes around each slot and the complete allocation. Copy-list size
// intentionally differs from GU job count, including copies with zero GU jobs.
struct DownCopyTest {
    static constexpr int count = 48, base = 5, slot0 = 17;
    CacheDownCopy view;
    char *host = nullptr, *pool = nullptr;
    const char** table = nullptr;
    int *slots = nullptr, *list = nullptr;
    std::vector<unsigned char> expected, got;
    DownCopyTest(int T, bool coherent) {
        const size_t bytes = (1 + T % 3) * 4096 + 16;
        const size_t stride = bytes + 256;
#if FNX_HIP
        ck(hipHostMalloc((void**) &host, count * bytes, coherent ? hipHostMallocCoherent : hipHostMallocNonCoherent), "down host");
#else
        ck(cudaMallocHost((void**) &host, count * bytes), "down host");
#endif
        expected.resize(count * stride + 256); got.resize(expected.size());
        ck(cudaMalloc((void**) &pool, expected.size()), "down pool");
        ck(cudaMalloc((void**) &table, (count + base) * sizeof(char*)), "down table");
        ck(cudaMalloc((void**) &slots, (count + base) * sizeof(int)), "down slots");
        ck(cudaMalloc((void**) &list, (count + 1) * sizeof(int)), "down list");
        std::vector<const char*> ht(count + base);
        for (int j = 0; j < count; ++j) ht[j + base] = host + j * bytes;
        ck(cudaMemcpy(table, ht.data(), ht.size() * sizeof(char*), cudaMemcpyHostToDevice), "down table copy");
        constexpr int blocks[] = {1, 32, 64, 256};
        view = {table, slots, list, pool + 128, stride, 128, bytes, slot0, blocks[T % 4]};
    }
    void prepare(int rep, cudaStream_t stream) {
        for (size_t j = 0; j < count * view.bytes; ++j) host[j] = (j * 11 + (j >> 8) * 19 + rep * 13) & 255;
        constexpr int counts[] = {count, 0, 1, 7};
        const int n = counts[rep % 4];
        std::vector<int> hs(count + base, -1), hl(count + 1);
        std::fill(expected.begin(), expected.end(), 0xB3);
        hl[0] = n;
        for (int j = 0; j < count; ++j) hs[base + j] = slot0 + (j * 5 + rep * 7) % count;
        for (int j = 0; j < n; ++j) {
            const int e = (j * 7 + rep * 11) % count;
            hl[j + 1] = base + e;
            std::memcpy(expected.data() + 128 + (hs[base + e] - slot0) * view.slot_bytes + view.offset,
                        host + e * view.bytes, view.bytes);
        }
        ck(cudaMemcpy(slots, hs.data(), hs.size() * sizeof(int), cudaMemcpyHostToDevice), "down slots copy");
        ck(cudaMemcpy(list, hl.data(), hl.size() * sizeof(int), cudaMemcpyHostToDevice), "down list copy");
        ck(cudaMemsetAsync(pool, 0xB3, expected.size(), stream), "down guards");
    }
    void check() {
        ck(cudaMemcpy(got.data(), pool, got.size(), cudaMemcpyDeviceToHost), "down read");
        if (got != expected) throw std::runtime_error("fused down bytes or guards differ");
    }
    ~DownCopyTest() {
        cudaFreeHost(host); cudaFree(pool); cudaFree(table); cudaFree(slots); cudaFree(list);
    }
};

template <GgmlType TA>
int run_type(int dev) {
    constexpr int experts = 48, cols = 2560, rows = 641;
    constexpr size_t row_bytes = cols / qtraits<TA>::qk * qtraits<TA>::bs;
    const size_t weight_bytes = experts * rows * row_bytes;
    std::vector<unsigned char> weights(weight_bytes);
    for (size_t i = 0; i < weight_bytes; ++i) weights[i] = (i * 13 + (i >> 7) * 17) & 255;
    for (size_t i = 0; i < weight_bytes; i += qtraits<TA>::bs) {
        if constexpr (TA == T_Q5_K) {
            const uint16_t dm[2] = {0x2400, 0x2000};
            std::memcpy(weights.data() + i, dm, sizeof dm);
        } else {
            const uint16_t scale = 0x2400; // finite half: all IQ formats put d first
            std::memcpy(weights.data() + i, &scale, sizeof scale);
        }
    }
    std::vector<block_q8_1> x(MAXT * cols / 32);
    for (size_t i = 0; i < x.size(); ++i) {
        int sum = 0;
        for (int k = 0; k < 32; ++k) sum += x[i].qs[k] = (int)((i * 7 + k * 11) % 255) - 127;
        x[i].ds = make_half2(__float2half(0.02f), __float2half(sum * 0.02f));
    }
    char* dw = nullptr;
    block_q8_1* dx = nullptr;
    Job* dj = nullptr;
    int* dn = nullptr;
    float *baseline = nullptr, *candidate = nullptr;
    const size_t output_bytes = (size_t) experts * MAXT * rows * sizeof(float) + 256;
    int *dwl1 = nullptr, *dwlm = nullptr, *dn1 = nullptr, *dnm = nullptr;
    ck(cudaMalloc((void**)&dw, weight_bytes), "weights");
    ck(cudaMalloc((void**)&dx, x.size() * sizeof(block_q8_1)), "activations");
    ck(cudaMalloc((void**)&dj, experts * sizeof(Job)), "jobs");
    ck(cudaMalloc((void**)&dn, sizeof(int)), "count");
    ck(cudaMalloc((void**)&dwl1, experts * sizeof(int)), "wl1");
    ck(cudaMalloc((void**)&dwlm, experts * sizeof(int)), "wlm");
    ck(cudaMalloc((void**)&dn1, sizeof(int)), "n1");
    ck(cudaMalloc((void**)&dnm, sizeof(int)), "nm");
    ck(cudaMalloc((void**)&baseline, output_bytes), "baseline");
    ck(cudaMalloc((void**)&candidate, output_bytes), "candidate");
    ck(cudaMemcpy(dw, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "weights copy");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "activation copy");
    const MoeW w{dw, rows * row_bytes, row_bytes, rows, cols, nullptr};
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    std::vector<unsigned char> ref(output_bytes), got(output_bytes);
    int checks = 0;
    for (int T = 1; T <= MAXT; ++T) {
        for (int cap : {1, 2, 3, 4}) {
            if (cap >= 3 && TA != T_IQ2_XXS && TA != T_IQ2_S && TA != T_IQ3_XXS && TA != T_IQ3_S) continue;
            std::unique_ptr<DownCopyTest> dc;
            if (cap >= 3) dc = std::make_unique<DownCopyTest>(T, cap == 3);
            const CacheDownCopy* copy = dc ? &dc->view : nullptr;
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
            launch_test<TA>(T, cap, w, dj, dn, dx, candidate, stream, copy);
            ck(cudaStreamEndCapture(stream, &graph), "end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            constexpr int counts[] = {0, 1, 7, experts};
            for (int rep = 0; rep < 8; ++rep) {
                const int count = counts[rep % 4];
                if (dc) dc->prepare(rep, stream);
                std::vector<Job> jobs(experts);
                for (int j = 0; j < experts; ++j) {
                    jobs[j].w = (j * 7 + rep * 11) % experts;
                    jobs[j].n = 1 + (j + rep) % T;
                    for (int t = 0; t < jobs[j].n; ++t) {
                        jobs[j].y[t] = (t + rep) % T;
                        jobs[j].out[t] = j * MAXT + t;
                    }
                }
                ck(cudaMemcpy(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice), "job copy");
                ck(cudaMemcpy(dn, &count, sizeof(int), cudaMemcpyHostToDevice), "count copy");
                ck(cudaMemsetAsync(baseline, 0xA5, output_bytes, stream), "baseline guard");
                ck(cudaMemsetAsync(candidate, 0xA5, output_bytes, stream), "candidate guard");
                launch_test<TA>(T, 0, w, dj, dn, dx, baseline, stream);
                if ((rep / 4) % 2) ck(cudaGraphLaunch(exec, stream), "graph replay");
                else launch_test<TA>(T, cap, w, dj, dn, dx, candidate, stream, copy);
                ck(cudaStreamSynchronize(stream), "wait");
                if (dc) dc->check();
                ck(cudaMemcpy(ref.data(), baseline, output_bytes, cudaMemcpyDeviceToHost), "reference read");
                ck(cudaMemcpy(got.data(), candidate, output_bytes, cudaMemcpyDeviceToHost), "candidate read");
                if (ref != got) throw std::runtime_error("gate/up bit mismatch or guard overwrite: type " +
                    std::to_string(TA) + " T " + std::to_string(T) + " cap " + std::to_string(cap));
                ++checks;
            }
            ck(cudaGraphExecDestroy(exec), "destroy exec");
            ck(cudaGraphDestroy(graph), "destroy graph");
        }
        for (int wl_cap : {0, 1, 2}) {
            if (wl_cap == 2 && TA != T_IQ2_XXS && TA != T_IQ2_S && TA != T_IQ3_XXS && TA != T_IQ3_S && TA != T_IQ4_XS && TA != T_Q5_K) continue;
            if (wl_cap == 1 && TA != T_IQ4_XS && TA != T_Q5_K) continue;
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "wl capture");
            launch_test<TA>(1, wl_cap, w, dj, dn1, dx, candidate, stream, nullptr, dwl1, experts);
            launch_test<TA>(T, wl_cap, w, dj, dnm, dx, candidate, stream, nullptr, dwlm, experts);
            ck(cudaStreamEndCapture(stream, &graph), "wl end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "wl instantiate");
            constexpr int counts[] = {0, 1, 7, experts};
            for (int rep = 0; rep < 8; ++rep) {
                const int count = counts[rep % 4];
                std::vector<Job> jobs(experts);
                std::vector<int> hwl1, hwlm;
                for (int j = 0; j < experts; ++j) {
                    jobs[j].w = (j * 7 + rep * 11) % experts;
                    jobs[j].n = (rep % 4 == 2) ? 1 : (1 + (j + rep) % T);
                    for (int t = 0; t < jobs[j].n; ++t) {
                        jobs[j].y[t] = (t + rep) % T;
                        jobs[j].out[t] = j * MAXT + t;
                    }
                    if (j < count) {
                        if (jobs[j].n == 1) hwl1.push_back(j);
                        else hwlm.push_back(j);
                    }
                }
                const int n1 = (int) hwl1.size(), nm = (int) hwlm.size();
                ck(cudaMemcpy(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice), "wl job copy");
                ck(cudaMemcpy(dn, &count, sizeof(int), cudaMemcpyHostToDevice), "wl count copy");
                ck(cudaMemcpy(dn1, &n1, sizeof(int), cudaMemcpyHostToDevice), "n1 copy");
                ck(cudaMemcpy(dnm, &nm, sizeof(int), cudaMemcpyHostToDevice), "nm copy");
                if (n1) ck(cudaMemcpy(dwl1, hwl1.data(), n1 * sizeof(int), cudaMemcpyHostToDevice), "wl1 copy");
                if (nm) ck(cudaMemcpy(dwlm, hwlm.data(), nm * sizeof(int), cudaMemcpyHostToDevice), "wlm copy");
                ck(cudaMemsetAsync(baseline, 0xA5, output_bytes, stream), "wl baseline guard");
                ck(cudaMemsetAsync(candidate, 0xA5, output_bytes, stream), "wl candidate guard");
                launch_test<TA>(T, 0, w, dj, dn, dx, baseline, stream);
                if ((rep / 4) % 2) ck(cudaGraphLaunch(exec, stream), "wl graph replay");
                else {
                    launch_test<TA>(1, wl_cap, w, dj, dn1, dx, candidate, stream, nullptr, dwl1, experts);
                    launch_test<TA>(T, wl_cap, w, dj, dnm, dx, candidate, stream, nullptr, dwlm, experts);
                }
                ck(cudaStreamSynchronize(stream), "wl wait");
                ck(cudaMemcpy(ref.data(), baseline, output_bytes, cudaMemcpyDeviceToHost), "wl ref read");
                ck(cudaMemcpy(got.data(), candidate, output_bytes, cudaMemcpyDeviceToHost), "wl got read");
                if (ref != got) throw std::runtime_error("gate/up job_idx mismatch or guard overwrite: type " +
                    std::to_string(TA) + " T " + std::to_string(T) + " cap " + std::to_string(wl_cap));
                ++checks;
            }
            ck(cudaGraphExecDestroy(exec), "wl destroy exec");
            ck(cudaGraphDestroy(graph), "wl destroy graph");
        }
    }
    // Same live jobs for each timing arm; events bracket 40 ordered launches.
    cudaEvent_t start, end;
    ck(cudaEventCreate(&start), "start"); ck(cudaEventCreate(&end), "end");
    for (int count : {10, 24, 48}) {
        ck(cudaMemcpy(dn, &count, sizeof count, cudaMemcpyHostToDevice), "timing count");
        std::vector<Job> jobs(experts);
        for (int j = 0; j < count; ++j) {
            jobs[j].w = j;
            jobs[j].n = 60 / count + (j < 60 % count); // 6 tokens x top-10 picks
            for (int t = 0; t < jobs[j].n; ++t) jobs[j].y[t] = (j + t) % 6, jobs[j].out[t] = j * MAXT + t;
        }
        ck(cudaMemcpy(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice), "timing jobs");
        for (int cap : {0, 1, 2, 2, 1, 0}) {
            for (int i = 0; i < 500; ++i) launch_test<TA>(6, cap, w, dj, dn, dx, candidate, stream);
            ck(cudaEventRecord(start, stream), "record start");
            for (int i = 0; i < 40; ++i) launch_test<TA>(6, cap, w, dj, dn, dx, candidate, stream);
            ck(cudaEventRecord(end, stream), "record end");
            ck(cudaEventSynchronize(end), "event wait");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, start, end), "elapsed");
            std::cout << "card " << dev << " type " << TA << " jobs " << count << " reuse " << cap << " us " << ms * 1000 / 40 << '\n';
        }
    }
    ck(cudaEventDestroy(start), "free event"); ck(cudaEventDestroy(end), "free event");
    ck(cudaStreamDestroy(stream), "free stream");
    for (void* p : {(void*)dw, (void*)dx, (void*)dj, (void*)dn, (void*)dwl1, (void*)dwlm, (void*)dn1, (void*)dnm, (void*)baseline, (void*)candidate})
        ck(cudaFree(p), "free");
    return checks;
}
} // namespace fnx

int main() {
    int count = 0;
    fnx::ck(cudaGetDeviceCount(&count), "devices");
    if (count < 2) throw std::runtime_error("two physical devices required");
    int checks = 0;
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        checks += fnx::run_type<fnx::T_IQ2_XXS>(dev);
        checks += fnx::run_type<fnx::T_IQ2_S>(dev);
        checks += fnx::run_type<fnx::T_IQ3_XXS>(dev);
        checks += fnx::run_type<fnx::T_IQ3_S>(dev);
        checks += fnx::run_type<fnx::T_IQ4_XS>(dev);
        checks += fnx::run_type<fnx::T_Q5_K>(dev);
    }
    std::cout << "MoE gate/up: " << checks << " exact direct/graph checks passed (2048 with down copies and guards)\n";
}

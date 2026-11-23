// Compare fused, precomputed, hoisted weight-decode, and cnt-split worklist MoE down kernels for every decode window.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
#include <string>
#include <vector>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <GgmlType TA, bool PACKED_Q2>
void launch_down(int T, int variant, bool split, bool pre, const MoeW& w, const Job* dj, const int* dn,
                 const int* dwl1, const int* dwlm, const int* dn1, const int* dnm, const float* dg, float* out,
                 int cols, int grid_jobs, int nm_cap, cudaStream_t stream) {
    const int rpb = DN_THREADS / (cols / 32);
    const dim3 grid_all((w.nrows + rpb - 1) / rpb, grid_jobs);
    const dim3 grid_m((w.nrows + rpb - 1) / rpb, nm_cap);
    const size_t shared = (size_t) T * DN_THREADS * sizeof(float);
    if (pre) k_moe_h<<<(T * TOPK * (cols / 32) + 7) / 8, 256, 0, stream>>>(dg, cols, T * TOPK);
    const bool hoist = !PACKED_Q2 && (variant == 1);
    if (split) {
        if (pre) k_moe_down1<TA, true, PACKED_Q2><<<grid_all, DN_THREADS, 0, stream>>>(w, dj, dn1, dg, out, dwl1);
        else     k_moe_down1<TA, false, PACKED_Q2><<<grid_all, DN_THREADS, 0, stream>>>(w, dj, dn1, dg, out, dwl1);
        if constexpr (!PACKED_Q2 && DownUnit<TA>::supported) {
            if (hoist) {
                if (pre) k_moe_down<TA, true, PACKED_Q2, true><<<grid_m, DN_THREADS, shared, stream>>>(w, dj, dnm, dg, out, dwlm);
                else     k_moe_down<TA, false, PACKED_Q2, true><<<grid_m, DN_THREADS, shared, stream>>>(w, dj, dnm, dg, out, dwlm);
                return;
            }
        }
        if (pre) k_moe_down<TA, true, PACKED_Q2, false><<<grid_m, DN_THREADS, shared, stream>>>(w, dj, dnm, dg, out, dwlm);
        else     k_moe_down<TA, false, PACKED_Q2, false><<<grid_m, DN_THREADS, shared, stream>>>(w, dj, dnm, dg, out, dwlm);
        return;
    }
    if constexpr (!PACKED_Q2 && DownUnit<TA>::supported) {
        if (hoist) {
            if (pre) k_moe_down<TA, true, PACKED_Q2, true><<<grid_all, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
            else     k_moe_down<TA, false, PACKED_Q2, true><<<grid_all, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
            return;
        }
    }
    if (pre) k_moe_down<TA, true, PACKED_Q2, false><<<grid_all, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
    else     k_moe_down<TA, false, PACKED_Q2, false><<<grid_all, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
}

template <GgmlType TA, bool PACKED_Q2 = false>
int run_type(int dev, bool bench) {
    constexpr int experts = 48, rows = 2560, cols = 320, pairs = MAXT * TOPK;
    constexpr size_t row_bytes = PACKED_Q2 ? ((size_t) cols / 64 * 18) : ((size_t) cols / qtraits<TA>::qk * qtraits<TA>::bs);
    constexpr size_t blk_bytes = PACKED_Q2 ? 18 : qtraits<TA>::bs;
    std::vector<unsigned char> weights((size_t) experts * rows * row_bytes);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (i * 13 + i / 71) % 256;
    for (size_t i = 0; i < weights.size(); i += blk_bytes) {
        if constexpr (!PACKED_Q2 && TA == T_Q5_1) {
            const uint16_t dm[2] = {0x2400, 0x2000};
            std::memcpy(weights.data() + i, dm, sizeof dm);
        } else {
            const uint16_t scale = 0x2400;
            std::memcpy(weights.data() + i, &scale, sizeof scale);
        }
    }
    std::vector<float> gu((size_t) pairs * cols * 2);
    for (size_t i = 0; i < gu.size(); ++i) gu[i] = ((int) (i * 23 % 107) - 53) * .03f;
    char* dw = nullptr;
    float *dg = nullptr, *out = nullptr;
    Job* dj = nullptr;
    int *dn = nullptr, *dwl1 = nullptr, *dwlm = nullptr, *dn_wl = nullptr;
    const size_t output_bytes = (size_t) pairs * rows * sizeof(float) + 256;
    ck(cudaMalloc((void**) &dw, weights.size()), "weights");
    ck(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice), "weights copy");
    ck(cudaMalloc((void**) &dg, gu.size() * sizeof(float)), "activations");
    ck(cudaMemcpy(dg, gu.data(), gu.size() * sizeof(float), cudaMemcpyHostToDevice), "activations copy");
    ck(cudaMalloc((void**) &out, output_bytes), "output");
    ck(cudaMalloc((void**) &dj, (size_t) pairs * sizeof(Job)), "jobs");
    ck(cudaMalloc((void**) &dn, sizeof(int)), "count");
    ck(cudaMalloc((void**) &dwl1, (size_t) pairs * sizeof(int)), "wl1");
    ck(cudaMalloc((void**) &dwlm, (size_t) pairs * sizeof(int)), "wlm");
    ck(cudaMalloc((void**) &dn_wl, 2 * sizeof(int)), "n_wl");
    const MoeW w{dw, rows * row_bytes, row_bytes, rows, cols, nullptr};
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ck(cudaDeviceSynchronize(), "initial uploads");
    std::vector<unsigned char> ref(output_bytes), got(output_bytes);
    int checks = 0;

    struct CaseCfg {
        int hoist;
        bool split;
        bool pre;
        bool use_graph;
    };
    std::vector<CaseCfg> cfgs;
    if constexpr (PACKED_Q2) {
        cfgs = {{0, false, true, true}, {0, true, false, true}, {0, true, true, true}};
    } else {
        cfgs = {{0, false, true, true}, {1, false, false, false}, {1, false, true, true},
                {1, true, false, true}, {1, true, true, true}, {0, true, false, false}};
    }

    for (int T = 1; T <= MAXT; ++T) {
        const int cap = T * TOPK;
        for (const CaseCfg& cfg : cfgs) {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (cfg.use_graph) {
                ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
                launch_down<TA, PACKED_Q2>(T, cfg.hoist, cfg.split, cfg.pre, w, dj, dn, dwl1, dwlm, dn_wl, dn_wl + 1,
                                           dg, out, cols, cap, std::max(1, cap / 2), stream);
                ck(cudaStreamEndCapture(stream, &graph), "capture end");
                ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            }
            for (int rep = 0; rep < (cfg.use_graph ? 8 : 4); ++rep) {
                const int pat = rep % 4;
                std::vector<Job> jobs(cap);
                int count = 0, cursor = 0;
                if (pat == 0) {
                    count = 0;
                } else if (pat == 1) {
                    count = cap;
                    for (int j = 0; j < count; ++j) {
                        jobs[j].w = (j * 7 + rep * 11) % experts;
                        jobs[j].n = 1;
                        jobs[j].y[0] = jobs[j].out[0] = (cursor++ * 37 + rep * 13) % cap;
                    }
                } else if (pat == 2) {
                    const int n_each = T >= 2 ? 2 : 1;
                    count = cap / n_each;
                    for (int j = 0; j < count; ++j) {
                        jobs[j].w = (j * 5 + rep * 9) % experts;
                        jobs[j].n = n_each;
                        for (int t = 0; t < n_each; ++t)
                            jobs[j].y[t] = jobs[j].out[t] = (cursor++ * 37 + rep * 13) % cap;
                    }
                } else {
                    int left = cap;
                    while (left > 0 && count < cap) {
                        const int want = (count % 3 == 0) ? 1 : std::min(T, 3);
                        const int n_each = std::min(left, want);
                        jobs[count].w = (count * 7 + rep * 11) % experts;
                        jobs[count].n = n_each;
                        for (int t = 0; t < n_each; ++t)
                            jobs[count].y[t] = jobs[count].out[t] = (cursor++ * 37 + rep * 13) % cap;
                        left -= n_each;
                        ++count;
                    }
                }
                std::vector<int> hwl1, hwlm;
                for (int j = 0; j < count; ++j) {
                    if (jobs[j].n == 1) hwl1.push_back(j);
                    else hwlm.push_back(j);
                }
                const int h_nwl[2] = {(int) hwl1.size(), (int) hwlm.size()};
                ck(cudaMemcpyAsync(dj, jobs.data(), jobs.size() * sizeof(Job), cudaMemcpyHostToDevice, stream), "jobs");
                ck(cudaMemcpyAsync(dn, &count, sizeof count, cudaMemcpyHostToDevice, stream), "count");
                ck(cudaMemcpyAsync(dn_wl, h_nwl, sizeof h_nwl, cudaMemcpyHostToDevice, stream), "n_wl");
                if (!hwl1.empty()) ck(cudaMemcpyAsync(dwl1, hwl1.data(), hwl1.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "wl1");
                if (!hwlm.empty()) ck(cudaMemcpyAsync(dwlm, hwlm.data(), hwlm.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "wlm");

                ck(cudaMemsetAsync(out, 0xA5, output_bytes, stream), "reference guard");
                launch_down<TA, PACKED_Q2>(T, 0, false, false, w, dj, dn, dwl1, dwlm, dn_wl, dn_wl + 1,
                                           dg, out, cols, cap, std::max(1, cap / 2), stream);
                ck(cudaStreamSynchronize(stream), "reference wait");
                ck(cudaMemcpy(ref.data(), out, output_bytes, cudaMemcpyDeviceToHost), "reference read");

                ck(cudaMemsetAsync(out, 0xA5, output_bytes, stream), "candidate guard");
                if (cfg.use_graph && rep >= 4) ck(cudaGraphLaunch(exec, stream), "replay");
                else launch_down<TA, PACKED_Q2>(T, cfg.hoist, cfg.split, cfg.pre, w, dj, dn, dwl1, dwlm, dn_wl, dn_wl + 1,
                                                dg, out, cols, cap, std::max(1, cap / 2), stream);
                ck(cudaStreamSynchronize(stream), "candidate wait");
                ck(cudaMemcpy(got.data(), out, output_bytes, cudaMemcpyDeviceToHost), "candidate read");
                if (ref != got)
                    throw std::runtime_error("MoE down mismatch: card " + std::to_string(dev) + " type " + std::to_string(TA) +
                                             " packed_q2 " + std::to_string(PACKED_Q2) + " T " + std::to_string(T) +
                                             " hoist " + std::to_string(cfg.hoist) + " split " + std::to_string(cfg.split) +
                                             " pre " + std::to_string(cfg.pre) + " rep " + std::to_string(rep));
                ++checks;
            }
            if (cfg.use_graph) {
                ck(cudaGraphExecDestroy(exec), "destroy exec");
                ck(cudaGraphDestroy(graph), "destroy graph");
            }
        }
    }
    if (bench && !PACKED_Q2 && dev == 0) {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "event a");
        ck(cudaEventCreate(&b), "event b");
        for (int T : {1, 4, 6, 8, 16}) {
            const int cap = T * TOPK;
            std::vector<Job> jobs(cap);
            int count = 0, cursor = 0, left = cap;
            // Realistic decode workload (~60% cnt==1 jobs, remainder cnt==2..min(T,4)).
            const int target1 = (cap * 6) / 10;
            while (left > 0 && count < cap) {
                const int want = (cursor < target1 || T == 1) ? 1 : std::min(T, 2 + (count % 3));
                const int n_each = std::min(left, want);
                jobs[count].w = (count * 7) % experts;
                jobs[count].n = n_each;
                for (int t = 0; t < n_each; ++t) jobs[count].y[t] = jobs[count].out[t] = cursor++;
                left -= n_each;
                ++count;
            }
            std::vector<int> hwl1, hwlm;
            for (int j = 0; j < count; ++j) {
                if (jobs[j].n == 1) hwl1.push_back(j);
                else hwlm.push_back(j);
            }
            const int h_nwl[2] = {(int) hwl1.size(), (int) hwlm.size()};
            ck(cudaMemcpyAsync(dj, jobs.data(), jobs.size() * sizeof(Job), cudaMemcpyHostToDevice, stream), "bench jobs");
            ck(cudaMemcpyAsync(dn, &count, sizeof count, cudaMemcpyHostToDevice, stream), "bench count");
            ck(cudaMemcpyAsync(dn_wl, h_nwl, sizeof h_nwl, cudaMemcpyHostToDevice, stream), "bench n_wl");
            if (!hwl1.empty()) ck(cudaMemcpyAsync(dwl1, hwl1.data(), hwl1.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "bench wl1");
            if (!hwlm.empty()) ck(cudaMemcpyAsync(dwlm, hwlm.data(), hwlm.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "bench wlm");
            ck(cudaStreamSynchronize(stream), "bench upload sync");
            const bool pre = T >= 9;
            struct Arm { int hoist; bool split; const char* name; };
            const Arm arms[] = {{0, false, "base"}, {1, false, "hoist"}, {0, true, "split_wl"}, {1, true, "both"}};
            std::cout << "bench card " << dev << " type " << TA << " T " << T
                      << " jobs " << count << " (n1=" << h_nwl[0] << ",nm=" << h_nwl[1] << "):";
            double sum_us[4] = {};
            for (int arm_i : {0, 1, 2, 3, 3, 2, 1, 0}) {
                const Arm& arm = arms[arm_i];
                cudaGraph_t bg;
                cudaGraphExec_t be;
                ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "bench capture");
                launch_down<TA, PACKED_Q2>(T, arm.hoist, arm.split, pre, w, dj, dn, dwl1, dwlm, dn_wl, dn_wl + 1,
                                           dg, out, cols, cap, std::max(1, cap / 2), stream);
                ck(cudaStreamEndCapture(stream, &bg), "bench end");
                ck(cudaGraphInstantiate(&be, bg, nullptr, nullptr, 0), "bench instantiate");
                for (int i = 0; i < 200; ++i) ck(cudaGraphLaunch(be, stream), "bench warm");
                ck(cudaEventRecord(a, stream), "bench a");
                for (int i = 0; i < 100; ++i) ck(cudaGraphLaunch(be, stream), "bench run");
                ck(cudaEventRecord(b, stream), "bench b");
                ck(cudaEventSynchronize(b), "bench wait");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b), "bench elapsed");
                sum_us[arm_i] += (ms * 1000.0 / 100.0) * 0.5;
                ck(cudaGraphExecDestroy(be), "bench destroy exec");
                ck(cudaGraphDestroy(bg), "bench destroy graph");
            }
            for (int i = 0; i < 4; ++i) std::cout << " " << arms[i].name << "=" << sum_us[i] << "us";
            std::cout << "\n";
        }
        ck(cudaEventDestroy(a), "destroy a");
        ck(cudaEventDestroy(b), "destroy b");
    }
    ck(cudaStreamDestroy(stream), "free stream");
    for (void* p : {(void*) dw, (void*) dg, (void*) out, (void*) dj, (void*) dn, (void*) dwl1, (void*) dwlm, (void*) dn_wl})
        ck(cudaFree(p), "free");
    return checks;
}
} // namespace fnx

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--bench"))
            throw std::runtime_error("usage: split_moe_down_test [--bench]");
        const bool bench = argc == 2;
        int count = 0;
        fnx::ck(cudaGetDeviceCount(&count), "devices");
        if (count < 2) throw std::runtime_error("two physical devices required");
        int checks = 0;
        for (int dev = 0; dev < 2; ++dev) {
            fnx::ck(cudaSetDevice(dev), "device");
            checks += fnx::run_type<fnx::T_IQ4_NL>(dev, bench);
            checks += fnx::run_type<fnx::T_Q4_0>(dev, bench);
            checks += fnx::run_type<fnx::T_Q5_0>(dev, bench);
            checks += fnx::run_type<fnx::T_Q5_1>(dev, bench);
            checks += fnx::run_type<fnx::T_Q8_0>(dev, bench);
            checks += fnx::run_type<fnx::T_Q4_0, true>(dev, bench);
        }
        std::cout << "MoE down prequant/hoist/split: " << checks << " exact direct/graph comparisons passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

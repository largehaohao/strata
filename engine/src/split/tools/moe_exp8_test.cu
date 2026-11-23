// Verify 0-ULP parity of fused EXP8 (gate+up+SwiGLU+q8_1) across all 9 weight formats,
// LDS staging modes (0/1/2), NT_MAX=1/4 chunking, job_idx split worklists, CacheDownCopy,
// CacheGateUpCopy, direct + HIP Graph replay, and end-to-end down integration on dual GPUs.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <GgmlType TA>
void fill_test_weights(std::vector<unsigned char>& dst, int experts, int rows, size_t row_bytes, int seed = 0) {
    const size_t raw_bytes = (size_t) rows * row_bytes;
    for (int e = 0; e < experts; ++e) {
        unsigned char* at = dst.data() + (size_t) e * raw_bytes;
        for (size_t i = 0; i < raw_bytes; ++i)
            at[i] = (i * 13 + (i >> 7) * 17 + e * 31 + seed * 11) & 255;
        for (size_t i = 0; i < raw_bytes; i += qtraits<TA>::bs) {
            if constexpr (TA == T_Q4_K || TA == T_Q5_K) {
                const uint16_t dm[2] = {(uint16_t) (0x2400 + ((seed + e) & 3) * 17), 0x2000};
                std::memcpy(at + i, dm, sizeof dm);
            } else if constexpr (TA == T_Q6_K) {
                const uint16_t scale = 0x2400 + ((seed + e) & 3) * 17;
                std::memcpy(at + i + qtraits<TA>::bs - 2, &scale, sizeof scale);
            } else {
                const uint16_t scale = 0x2400 + ((seed + e) & 3) * 17;
                std::memcpy(at + i, &scale, sizeof scale);
            }
        }
    }
}

template <GgmlType TA>
void launch_baseline_gu(int T, int cap, const MoeW& w, const Job* jobs, const int* njobs,
                        const block_q8_1* x, float* gu, cudaStream_t s) {
    if constexpr (TA == T_Q4_K) {
        gu_q4k_launch(T, w, jobs, njobs, cap, x, gu, s);
    } else {
        const dim3 grid((w.nrows + MMVQ_THREADS / 32 - 1) / (MMVQ_THREADS / 32), cap);
        gu_launch<TA, 32>(T, grid, s, w, jobs, njobs, x, gu);
    }
}

template <GgmlType TA>
void launch_exp8_test(int T, int cap, int lds_mode, const MoeW& w, const Job* jobs, const int* njobs,
                      const block_q8_1* x, cudaStream_t s, const CacheDownCopy* copy = nullptr,
                      const CacheGateUpCopy* gu_copy = nullptr, const int* job_idx = nullptr,
                      block_q8_1* hq_out = nullptr) {
    const int nxb = (w.nrows / 2) / 32;
    const bool stage_x = (lds_mode == 1);
    const bool stage_grid = exp8_is_iq4<TA>() && (lds_mode == 1 || lds_mode == 2);
    const int nt = (T == 1) ? 1 : 4;
    const size_t smem = stage_x ? (size_t) (T < nt ? T : nt) * (w.ncols / QK8_1) * sizeof(block_q8_1) : 0;
    const dim3 grid(nxb, cap);
    if constexpr (TA == T_Q4_K) {
        if (nt == 1) {
            if (stage_x) k_moe_gu_exp8_q4k<1, true><<<grid, Q4K_THREADS, smem, s>>>(w, jobs, njobs, x, job_idx, hq_out);
            else         k_moe_gu_exp8_q4k<1, false><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, x, job_idx, hq_out);
        } else {
            if (stage_x) k_moe_gu_exp8_q4k<4, true><<<grid, Q4K_THREADS, smem, s>>>(w, jobs, njobs, x, job_idx, hq_out);
            else         k_moe_gu_exp8_q4k<4, false><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, x, job_idx, hq_out);
        }
    } else if constexpr (exp8_is_iq4<TA>()) {
        const dim3 fused_grid(nxb * cap + (copy ? copy->blocks : 0));
        const CacheDownCopy empty_dn{};
        const CacheGateUpCopy empty_gu{};
#define E8_IQ_CALL(NTVAL, SX, SG) \
    if (gu_copy && copy) k_moe_gu_exp8<TA, NTVAL, SX, SG, true, true><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, *gu_copy, job_idx, hq_out); \
    else if (gu_copy)    k_moe_gu_exp8<TA, NTVAL, SX, SG, false, true><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, *gu_copy, job_idx, hq_out); \
    else if (copy)       k_moe_gu_exp8<TA, NTVAL, SX, SG, true, false><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, empty_gu, job_idx, hq_out); \
    else                 k_moe_gu_exp8<TA, NTVAL, SX, SG, false, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, empty_gu, job_idx, hq_out)
        if (nt == 1) {
            if (stage_x) { if (stage_grid) { E8_IQ_CALL(1, true, true); } else { E8_IQ_CALL(1, true, false); } }
            else         { if (stage_grid) { E8_IQ_CALL(1, false, true); } else { E8_IQ_CALL(1, false, false); } }
        } else {
            if (stage_x) { if (stage_grid) { E8_IQ_CALL(4, true, true); } else { E8_IQ_CALL(4, true, false); } }
            else         { if (stage_grid) { E8_IQ_CALL(4, false, true); } else { E8_IQ_CALL(4, false, false); } }
        }
#undef E8_IQ_CALL
    } else {
        if (nt == 1) {
            if (stage_x) k_moe_gu_exp8<TA, 1, true, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, {}, {}, job_idx, hq_out);
            else         k_moe_gu_exp8<TA, 1, false, false><<<grid, 256, 0, s>>>(w, jobs, njobs, x, {}, {}, job_idx, hq_out);
        } else {
            if (stage_x) k_moe_gu_exp8<TA, 4, true, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, {}, {}, job_idx, hq_out);
            else         k_moe_gu_exp8<TA, 4, false, false><<<grid, 256, 0, s>>>(w, jobs, njobs, x, {}, {}, job_idx, hq_out);
        }
    }
    ck(cudaGetLastError(), "exp8 launch");
}

struct Exp8DownCopy {
    static constexpr int count = 48, base = 3 * NEXP, slot0 = 17, guard = 256;
    CacheDownCopy view;
    char *host = nullptr, *pool = nullptr;
    const char** table = nullptr;
    int *slots = nullptr, *list = nullptr;
    std::vector<unsigned char> expected, got;
    Exp8DownCopy(int T, bool coherent) {
        const size_t bytes = (1 + T % 3) * 4096 + 16;
        const size_t stride = bytes + guard;
        const size_t pool_bytes = count * stride + guard * 2;
#if FNX_HIP
        ck(hipHostMalloc((void**) &host, count * bytes, coherent ? hipHostMallocCoherent : hipHostMallocNonCoherent), "dn host");
#else
        ck(cudaMallocHost((void**) &host, count * bytes), "dn host");
#endif
        expected.resize(pool_bytes);
        got.resize(pool_bytes);
        ck(cudaMalloc((void**) &pool, pool_bytes), "dn pool");
        ck(cudaMalloc((void**) &table, (base + count) * sizeof(char*)), "dn table");
        ck(cudaMalloc((void**) &slots, (base + count) * sizeof(int)), "dn slots");
        ck(cudaMalloc((void**) &list, (count + 1) * sizeof(int)), "dn list");
        std::vector<const char*> ht(base + count, nullptr);
        for (int j = 0; j < count; ++j) ht[base + j] = host + (size_t) j * bytes;
        ck(cudaMemcpy(table, ht.data(), ht.size() * sizeof(char*), cudaMemcpyHostToDevice), "dn table");
        constexpr int blocks[] = {1, 16, 64, 128};
        view = {table, slots, list, pool + guard, stride, 128, bytes, slot0, blocks[T % 4]};
    }
    void prepare(const std::vector<int>& miss_e, int seed, cudaStream_t s) {
        for (size_t i = 0; i < count * view.bytes; ++i) host[i] = (i * 19 + i / 31 + seed * 7) & 255;
        std::vector<int> hs(base + count, -1), hl(count + 1);
        std::fill(expected.begin(), expected.end(), 0xB3);
        hl[0] = (int) miss_e.size();
        for (int j = 0; j < count; ++j) hs[base + j] = slot0 + (j * 5 + seed * 7) % count;
        for (int j = 0; j < hl[0]; ++j) {
            const int e = miss_e[j];
            hl[1 + j] = base + e;
            std::memcpy(expected.data() + guard + (size_t) (hs[base + e] - slot0) * view.slot_bytes + view.offset,
                        host + (size_t) e * view.bytes, view.bytes);
        }
        ck(cudaMemcpyAsync(slots, hs.data(), hs.size() * sizeof(int), cudaMemcpyHostToDevice, s), "dn slots");
        ck(cudaMemcpyAsync(list, hl.data(), hl.size() * sizeof(int), cudaMemcpyHostToDevice, s), "dn list");
        ck(cudaMemsetAsync(pool, 0xB3, expected.size(), s), "dn guards");
    }
    void check() {
        ck(cudaMemcpy(got.data(), pool, got.size(), cudaMemcpyDeviceToHost), "dn pool read");
        if (got != expected) throw std::runtime_error("EXP8 CacheDownCopy bytes or guards differ");
    }
    ~Exp8DownCopy() {
        cudaFreeHost(host); cudaFree(pool); cudaFree(table); cudaFree(slots); cudaFree(list);
    }
};

__global__ void k_copy_g_moe_hq(block_q8_1* __restrict__ dst, int npairs) {
    const int total = npairs * DN_MAXB;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dst[idx] = (&g_moe_hq[0][0])[idx];
}

template <GgmlType TA>
int run_type(int dev) {
    constexpr int experts = 48, cols = N, ffs = 320, rows = 2 * ffs, pairs = MAXT * TOPK, guard = 256;
    constexpr size_t row_bytes = cols / qtraits<TA>::qk * qtraits<TA>::bs;
    const size_t weight_bytes = (size_t) experts * rows * row_bytes;
    std::vector<unsigned char> weights(weight_bytes);
    fill_test_weights<TA>(weights, experts, rows, row_bytes, 3);

    std::vector<block_q8_1> x(MAXT * cols / 32);
    for (size_t i = 0; i < x.size(); ++i) {
        int sum = 0;
        for (int k = 0; k < 32; ++k) sum += x[i].qs[k] = (int) ((i * 7 + k * 11) % 255) - 127;
        x[i].ds = make_half2(__float2half(0.02f), __float2half(sum * 0.02f));
    }

    char* dw = nullptr;
    block_q8_1 *dx = nullptr, *dhq = nullptr;
    float *dgu = nullptr;
    Job* dj = nullptr;
    int *dn = nullptr, *dwl1 = nullptr, *dwlm = nullptr, *dn1 = nullptr, *dnm = nullptr;
    const size_t gu_bytes = (size_t) pairs * rows * sizeof(float);
    const size_t hq_payload = (size_t) pairs * DN_MAXB * sizeof(block_q8_1);
    const size_t hq_bytes = hq_payload + 2 * guard;
    ck(cudaMalloc((void**) &dw, weight_bytes), "weights");
    ck(cudaMalloc((void**) &dx, x.size() * sizeof(block_q8_1)), "x");
    ck(cudaMalloc((void**) &dgu, gu_bytes), "gu");
    ck(cudaMalloc((void**) &dhq, hq_bytes), "hq");
    ck(cudaMalloc((void**) &dj, experts * sizeof(Job)), "jobs");
    ck(cudaMalloc((void**) &dn, sizeof(int)), "n");
    ck(cudaMalloc((void**) &dwl1, experts * sizeof(int)), "wl1");
    ck(cudaMalloc((void**) &dwlm, experts * sizeof(int)), "wlm");
    ck(cudaMalloc((void**) &dn1, sizeof(int)), "n1");
    ck(cudaMalloc((void**) &dnm, sizeof(int)), "nm");
    ck(cudaMemcpy(dw, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "weights copy");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "x copy");

    const MoeW w{dw, (size_t) rows * row_bytes, row_bytes, rows, cols, nullptr};
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    std::vector<unsigned char> ref_hq(hq_bytes), got_hq(hq_bytes);
    int checks = 0;

    auto verify_hq = [&](int active_pairs, const char* label, int T, int mode, int rep) {
        ck(cudaMemcpy(got_hq.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost), "cand hq read");
        for (size_t i = 0; i < (size_t) guard; ++i)
            if (got_hq[i] != 0xA5)
                throw std::runtime_error(std::string(label) + " leading guard overwrite");
        const size_t active_end = guard + (size_t) active_pairs * DN_MAXB * sizeof(block_q8_1);
        for (size_t i = guard; i < active_end; ++i)
            if (got_hq[i] != ref_hq[i])
                throw std::runtime_error(std::string(label) + " 0-ULP mismatch: card " + std::to_string(dev) +
                                         " type " + std::to_string(TA) + " T " + std::to_string(T) +
                                         " mode " + std::to_string(mode) + " rep " + std::to_string(rep));
        for (size_t i = active_end; i < hq_bytes; ++i)
            if (got_hq[i] != 0xA5)
                throw std::runtime_error(std::string(label) + " tail/guard overwrite");
    };

    std::vector<int> lds_modes = {0, 1};
    if constexpr (exp8_is_iq4<TA>()) lds_modes.push_back(2);

    for (int T = 1; T <= MAXT; ++T) {
        const int max_pairs = T * TOPK;
        for (int lds : lds_modes) {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "exp8 capture");
            launch_exp8_test<TA>(T, experts, lds, w, dj, dn, dx, stream, nullptr, nullptr, nullptr,
                                 reinterpret_cast<block_q8_1*>(reinterpret_cast<unsigned char*>(dhq) + guard));
            ck(cudaStreamEndCapture(stream, &graph), "exp8 end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "exp8 inst");
            constexpr int counts[] = {0, 1, 7, experts};
            for (int rep = 0; rep < 8; ++rep) {
                const int want_count = counts[rep % 4];
                std::vector<Job> jobs(experts);
                int count = 0, cursor = 0, left = max_pairs;
                while (count < want_count && left > 0) {
                    const int nj = (count == 0) ? std::min(left, T)
                                 : (want_count == experts && T >= 3) ? std::min(left, 1 + ((count + rep) % 3))
                                 : std::min(left, 1 + ((count + rep) % T));
                    jobs[count].w = (count * 7 + rep * 11) % experts;
                    jobs[count].n = nj;
                    for (int t = 0; t < nj; ++t) {
                        jobs[count].y[t] = (t + rep) % T;
                        jobs[count].out[t] = cursor++;
                    }
                    left -= nj;
                    ++count;
                }
                const int active_pairs = cursor;
                ck(cudaMemcpyAsync(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice, stream), "jobs");
                ck(cudaMemcpyAsync(dn, &count, sizeof(int), cudaMemcpyHostToDevice, stream), "count");

                ck(cudaMemsetAsync(dgu, 0, gu_bytes, stream), "clear gu");
                launch_baseline_gu<TA>(T, experts, w, dj, dn, dx, dgu, stream);
                ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "ref hq guard");
                if (active_pairs > 0) {
                    const int nh = active_pairs * DN_MAXB;
                    k_moe_h<<<(nh + 7) / 8, 256, 0, stream>>>(dgu, ffs, active_pairs);
                    auto* hq_dst_ref = reinterpret_cast<block_q8_1*>(reinterpret_cast<unsigned char*>(dhq) + guard);
                    k_copy_g_moe_hq<<<(active_pairs * DN_MAXB + 255) / 256, 256, 0, stream>>>(hq_dst_ref, active_pairs);
                    ck(cudaGetLastError(), "copy g_moe_hq");
                }
                ck(cudaStreamSynchronize(stream), "ref sync");
                ck(cudaMemcpy(ref_hq.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost), "ref read");

                ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "cand hq guard");
                if ((rep / 4) % 2) ck(cudaGraphLaunch(exec, stream), "exp8 replay");
                else launch_exp8_test<TA>(T, experts, lds, w, dj, dn, dx, stream, nullptr, nullptr, nullptr,
                                          reinterpret_cast<block_q8_1*>(reinterpret_cast<unsigned char*>(dhq) + guard));
                ck(cudaStreamSynchronize(stream), "cand sync");
                verify_hq(active_pairs, "exp8", T, lds, rep);
                ++checks;
            }
            ck(cudaGraphExecDestroy(exec), "destroy exec");
            ck(cudaGraphDestroy(graph), "destroy graph");
        }

        // Split worklist (NT_MAX=1 on wl1 + NT_MAX=4 on wlm) parity with LDS=1.
        {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            auto* hq_dst = reinterpret_cast<block_q8_1*>(reinterpret_cast<unsigned char*>(dhq) + guard);
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "wl capture");
            launch_exp8_test<TA>(1, experts, 1, w, dj, dn1, dx, stream, nullptr, nullptr, dwl1, hq_dst);
            launch_exp8_test<TA>(T, experts, 1, w, dj, dnm, dx, stream, nullptr, nullptr, dwlm, hq_dst);
            ck(cudaStreamEndCapture(stream, &graph), "wl end");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "wl inst");
            constexpr int counts[] = {0, 1, 7, 16};
            for (int rep = 0; rep < 8; ++rep) {
                const int want_count = counts[rep % 4];
                std::vector<Job> jobs(experts);
                std::vector<int> hwl1, hwlm;
                int count = 0, cursor = 0, left = max_pairs;
                while (count < want_count && left > 0) {
                    const int want_n = (rep % 4 == 2) ? 1 : (1 + (count + rep) % T);
                    const int nj = std::min(left, want_n);
                    jobs[count].w = (count * 7 + rep * 11) % experts;
                    jobs[count].n = nj;
                    for (int t = 0; t < nj; ++t) {
                        jobs[count].y[t] = (t + rep) % T;
                        jobs[count].out[t] = cursor++;
                    }
                    if (nj == 1) hwl1.push_back(count);
                    else hwlm.push_back(count);
                    left -= nj;
                    ++count;
                }
                const int active_pairs = cursor;
                const int n1 = (int) hwl1.size(), nm = (int) hwlm.size();
                if (count) ck(cudaMemcpyAsync(dj, jobs.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream), "wl jobs");
                ck(cudaMemcpyAsync(dn, &count, sizeof(int), cudaMemcpyHostToDevice, stream), "wl count");
                ck(cudaMemcpyAsync(dn1, &n1, sizeof(int), cudaMemcpyHostToDevice, stream), "wl n1");
                ck(cudaMemcpyAsync(dnm, &nm, sizeof(int), cudaMemcpyHostToDevice, stream), "wl nm");
                if (n1) ck(cudaMemcpyAsync(dwl1, hwl1.data(), n1 * sizeof(int), cudaMemcpyHostToDevice, stream), "wl1");
                if (nm) ck(cudaMemcpyAsync(dwlm, hwlm.data(), nm * sizeof(int), cudaMemcpyHostToDevice, stream), "wlm");

                ck(cudaMemsetAsync(dgu, 0, gu_bytes, stream), "wl ref gu");
                launch_baseline_gu<TA>(T, experts, w, dj, dn, dx, dgu, stream);
                ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "wl ref hq");
                if (active_pairs > 0) {
                    k_moe_h<<<(active_pairs * DN_MAXB + 7) / 8, 256, 0, stream>>>(dgu, ffs, active_pairs);
                    k_copy_g_moe_hq<<<(active_pairs * DN_MAXB + 255) / 256, 256, 0, stream>>>(hq_dst, active_pairs);
                    ck(cudaGetLastError(), "wl copy ref hq");
                }
                ck(cudaStreamSynchronize(stream), "wl ref sync");
                ck(cudaMemcpy(ref_hq.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost), "wl ref read");

                ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "wl cand hq");
                if ((rep / 4) % 2) ck(cudaGraphLaunch(exec, stream), "wl replay");
                else {
                    launch_exp8_test<TA>(1, experts, 1, w, dj, dn1, dx, stream, nullptr, nullptr, dwl1, hq_dst);
                    launch_exp8_test<TA>(T, experts, 1, w, dj, dnm, dx, stream, nullptr, nullptr, dwlm, hq_dst);
                }
                ck(cudaStreamSynchronize(stream), "wl cand sync");
                verify_hq(active_pairs, "exp8_wl", T, 1, rep);
                ++checks;
            }
            ck(cudaGraphExecDestroy(exec), "wl destroy exec");
            ck(cudaGraphDestroy(graph), "wl destroy graph");
        }
        (void) max_pairs;
    }

    // CacheDownCopy and CacheGateUpCopy parity (all 4 IQ formats, including jb.n > 4 multi-chunk cold misses).
    if constexpr (exp8_is_iq4<TA>()) {
        constexpr int base = Exp8DownCopy::base;
        const size_t host_stride = ((size_t) rows * row_bytes + 15) / 16 * 16;
        const size_t slot_stride = host_stride + guard;
        const size_t pool_bytes = experts * slot_stride + 2 * guard;
        char* gu_host = nullptr;
        ck(hipHostMalloc((void**) &gu_host, (size_t) experts * host_stride, hipHostMallocCoherent), "gu host");
        char* gu_pool = nullptr;
        const char** gu_ep = nullptr;
        const char** gu_ht = nullptr;
        unsigned* miss_bits = nullptr;
        int* gu_list = nullptr;
        ck(cudaMalloc((void**) &gu_pool, pool_bytes), "gu pool");
        ck(cudaMalloc((void**) &gu_ep, experts * sizeof(char*)), "gu ep");
        ck(cudaMalloc((void**) &gu_ht, (base + experts) * sizeof(char*)), "gu ht");
        ck(cudaMalloc((void**) &miss_bits, ((experts + 31) / 32) * sizeof(unsigned)), "miss bits");
        ck(cudaMalloc((void**) &gu_list, (experts + 1) * sizeof(int)), "gu list");

        std::vector<const char*> hep(experts), hht(base + experts, nullptr);
        for (int e = 0; e < experts; ++e) {
            hep[e] = gu_pool + guard + (size_t) ((e * 5) % experts) * slot_stride;
            hht[base + e] = gu_host + (size_t) e * host_stride;
        }
        ck(cudaMemcpy(gu_ep, hep.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "gu ep");
        ck(cudaMemcpy(gu_ht, hht.data(), hht.size() * sizeof(char*), cudaMemcpyHostToDevice), "gu ht");
        const MoeW w_ep{nullptr, (size_t) rows * row_bytes, row_bytes, rows, cols, gu_ep};
        std::vector<unsigned char> initial(pool_bytes), expected(pool_bytes), got_pool(pool_bytes);
        auto* hq_dst = reinterpret_cast<block_q8_1*>(reinterpret_cast<unsigned char*>(dhq) + guard);

        for (int T : {1, 4, 8, 16}) {
            for (bool with_dn : {true, false}) {
                for (int lds : {1, 2}) {
                    Exp8DownCopy dc(T, true);
                    const CacheGateUpCopy gu_copy{gu_ht, with_dn ? dc.list : gu_list, 3, miss_bits};
                    cudaGraph_t graph;
                    cudaGraphExec_t exec;
                    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "cache capture");
                    launch_exp8_test<TA>(T, 16, lds, w_ep, dj, dn, dx, stream, with_dn ? &dc.view : nullptr,
                                         &gu_copy, nullptr, hq_dst);
                    ck(cudaStreamEndCapture(stream, &graph), "cache end");
                    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cache inst");

                    for (int rep = 0; rep < 4; ++rep) {
                        const int nj = (rep == 0) ? 0 : (rep == 1 ? 1 : 8);
                        std::vector<unsigned char> host_flat((size_t) experts * rows * row_bytes);
                        fill_test_weights<TA>(host_flat, experts, rows, row_bytes, rep + T);
                        for (int e = 0; e < experts; ++e)
                            std::memcpy(gu_host + (size_t) e * host_stride,
                                        host_flat.data() + (size_t) e * rows * row_bytes,
                                        (size_t) rows * row_bytes);
                        ck(cudaMemcpyAsync(dw, host_flat.data(), weight_bytes, cudaMemcpyHostToDevice, stream), "ref flat w");

                        std::vector<Job> jobs(experts);
                        std::vector<int> miss_e;
                        int cursor = 0;
                        std::vector<unsigned> bitmap((experts + 31) / 32, 0);
                        for (int j = 0; j < nj; ++j) {
                            jobs[j].w = (j * 5 + rep * 3) % experts;
                            // Deduplicate expert ids so each missed expert has one job.
                            bool dup = false;
                            for (int q = 0; q < j; ++q) if (jobs[q].w == jobs[j].w) dup = true;
                            if (dup) jobs[j].w = (jobs[j].w + j + 1) % experts;
                            jobs[j].n = (T == 1) ? 1 : (j == 0 ? T : (1 + (j + rep) % T));
                            for (int t = 0; t < jobs[j].n; ++t) {
                                jobs[j].y[t] = (t + rep) % T;
                                jobs[j].out[t] = cursor++;
                            }
                            if ((j + rep) % 2 == 0) {
                                miss_e.push_back(jobs[j].w);
                                bitmap[jobs[j].w / 32] |= 1u << (jobs[j].w % 32);
                            }
                        }
                        const int active_pairs = cursor;
                        std::fill(initial.begin(), initial.end(), 0xCC);
                        for (int e = 0; e < experts; ++e)
                            std::memcpy(initial.data() + guard + (size_t) ((e * 5) % experts) * slot_stride,
                                        gu_host + (size_t) e * host_stride, (size_t) rows * row_bytes);
                        expected = initial;
                        for (int me : miss_e)
                            std::memset(initial.data() + guard + (size_t) ((me * 5) % experts) * slot_stride,
                                        0xA5, (size_t) rows * row_bytes);

                        if (with_dn) dc.prepare(miss_e, rep + T, stream);
                        std::vector<int> hl(1 + miss_e.size());
                        hl[0] = (int) miss_e.size();
                        for (size_t i = 0; i < miss_e.size(); ++i) hl[1 + i] = base + miss_e[i];
                        ck(cudaMemcpyAsync(gu_list, hl.data(), hl.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "gu list");
                        ck(cudaMemcpyAsync(miss_bits, bitmap.data(), bitmap.size() * sizeof(unsigned), cudaMemcpyHostToDevice, stream), "bitmap");
                        ck(cudaMemcpyAsync(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice, stream), "cache jobs");
                        ck(cudaMemcpyAsync(dn, &nj, sizeof(int), cudaMemcpyHostToDevice, stream), "cache nj");

                        ck(cudaMemsetAsync(dgu, 0, gu_bytes, stream), "cache ref gu");
                        launch_baseline_gu<TA>(T, 16, w, dj, dn, dx, dgu, stream);
                        ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "cache ref hq");
                        if (active_pairs > 0) {
                            k_moe_h<<<(active_pairs * DN_MAXB + 7) / 8, 256, 0, stream>>>(dgu, ffs, active_pairs);
                            k_copy_g_moe_hq<<<(active_pairs * DN_MAXB + 255) / 256, 256, 0, stream>>>(hq_dst, active_pairs);
                            ck(cudaGetLastError(), "cache ref copy");
                        }
                        ck(cudaStreamSynchronize(stream), "cache ref sync");
                        ck(cudaMemcpy(ref_hq.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost), "cache ref read");

                        ck(cudaMemcpyAsync(gu_pool, initial.data(), pool_bytes, cudaMemcpyHostToDevice, stream), "poison cold pool");
                        ck(cudaMemsetAsync(dhq, 0xA5, hq_bytes, stream), "cache cand hq");
                        if (rep >= 2) ck(cudaGraphLaunch(exec, stream), "cache replay");
                        else launch_exp8_test<TA>(T, 16, lds, w_ep, dj, dn, dx, stream, with_dn ? &dc.view : nullptr,
                                                  &gu_copy, nullptr, hq_dst);
                        ck(cudaStreamSynchronize(stream), "cache cand sync");
                        if (with_dn) dc.check();
                        ck(cudaMemcpy(got_pool.data(), gu_pool, pool_bytes, cudaMemcpyDeviceToHost), "gu pool read");
                        if (got_pool != expected)
                            throw std::runtime_error("EXP8 CacheGateUpCopy pool/guard mismatch");
                        verify_hq(active_pairs, "exp8_cache", T, lds, rep);
                        ++checks;
                    }
                    ck(cudaGraphExecDestroy(exec), "cache exec destroy");
                    ck(cudaGraphDestroy(graph), "cache graph destroy");
                }
            }
        }
        cudaFreeHost(gu_host);
        for (void* p : {(void*) gu_pool, (void*) gu_ep, (void*) gu_ht, (void*) miss_bits, (void*) gu_list})
            ck(cudaFree(p), "cache free");
    }

    // End-to-end FFN check through global g_moe_hq -> k_moe_down<T_IQ4_NL, PRE=true> and PACKED_Q2.
    {
        constexpr int dn_cols = ffs, dn_rows = N;
        constexpr size_t iq4_rb = dn_cols / qtraits<T_IQ4_NL>::qk * qtraits<T_IQ4_NL>::bs;
        constexpr size_t q2_rb = (size_t) dn_cols / 64 * 18;
        std::vector<unsigned char> h_iq4((size_t) experts * dn_rows * iq4_rb), h_q2((size_t) experts * dn_rows * q2_rb);
        fill_test_weights<T_IQ4_NL>(h_iq4, experts, dn_rows, iq4_rb, 5);
        for (size_t i = 0; i < h_q2.size(); ++i) h_q2[i] = (i * 17 + i / 59) & 255;
        for (size_t i = 0; i < h_q2.size(); i += 18) { uint16_t sc = 0x2400; std::memcpy(h_q2.data() + i, &sc, 2); }
        char *d_iq4 = nullptr, *d_q2 = nullptr;
        float* d_part = nullptr;
        Job* dj_dn = nullptr;
        const size_t part_bytes = (size_t) pairs * dn_rows * sizeof(float) + 2 * guard;
        ck(cudaMalloc((void**) &d_iq4, h_iq4.size()), "d_iq4");
        ck(cudaMalloc((void**) &d_q2, h_q2.size()), "d_q2");
        ck(cudaMalloc((void**) &d_part, part_bytes), "d_part");
        ck(cudaMalloc((void**) &dj_dn, experts * sizeof(Job)), "dj_dn");
        ck(cudaMemcpy(d_iq4, h_iq4.data(), h_iq4.size(), cudaMemcpyHostToDevice), "h_iq4 copy");
        ck(cudaMemcpy(d_q2, h_q2.data(), h_q2.size(), cudaMemcpyHostToDevice), "h_q2 copy");
        const MoeW w_iq4{d_iq4, (size_t) dn_rows * iq4_rb, iq4_rb, dn_rows, dn_cols, nullptr};
        const MoeW w_q2{d_q2, (size_t) dn_rows * q2_rb, q2_rb, dn_rows, dn_cols, nullptr};
        std::vector<unsigned char> ref_p(part_bytes), got_p(part_bytes);
        for (int T : {1, 4, 6, 8, 16}) {
            const int count = std::min(experts, T * TOPK);
            std::vector<Job> jg(count), jd(count);
            int cursor = 0;
            int nj_actual = count;
            for (int j = 0; j < count; ++j) {
                jg[j].w = jd[j].w = (j * 7 + T) % experts;
                const int nj = (T == 1) ? 1 : ((j == 0 && T >= 5) ? T : (1 + ((j + T) % 2)));
                const int left = T * TOPK - cursor;
                const int n_take = std::min(nj, std::max(1, left));
                jg[j].n = jd[j].n = n_take;
                for (int t = 0; t < n_take; ++t) {
                    const int pair_id = cursor++;
                    jg[j].y[t] = (t + T) % T;
                    jg[j].out[t] = jd[j].y[t] = jd[j].out[t] = pair_id;
                }
                if (cursor >= T * TOPK) {
                    nj_actual = j + 1;
                    break;
                }
            }
            ck(cudaMemcpyAsync(dn, &nj_actual, sizeof(int), cudaMemcpyHostToDevice, stream), "e2e nj");
            ck(cudaMemcpyAsync(dj, jg.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream), "e2e jg");
            ck(cudaMemcpyAsync(dj_dn, jd.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream), "e2e jd");
            const int rpb = DN_THREADS / (dn_cols / DN_U);
            const dim3 grid_dn((dn_rows + rpb - 1) / rpb, count);
            const size_t smem_dn = (size_t) T * DN_THREADS * sizeof(float);
            auto* part_mid = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(d_part) + guard);
            for (bool packed_q2 : {false, true}) {
                ck(cudaMemsetAsync(dgu, 0, gu_bytes, stream), "e2e clr gu");
                launch_baseline_gu<TA>(T, count, w, dj, dn, dx, dgu, stream);
                ck(cudaMemsetAsync(d_part, 0xA5, part_bytes, stream), "e2e ref part");
                if (packed_q2) k_moe_down<T_Q4_0, false, true><<<grid_dn, DN_THREADS, smem_dn, stream>>>(w_q2, dj_dn, dn, dgu, part_mid);
                else           k_moe_down<T_IQ4_NL, false, false><<<grid_dn, DN_THREADS, smem_dn, stream>>>(w_iq4, dj_dn, dn, dgu, part_mid);
                ck(cudaStreamSynchronize(stream), "e2e ref wait");
                ck(cudaMemcpy(ref_p.data(), d_part, part_bytes, cudaMemcpyDeviceToHost), "e2e ref read");

                ck(cudaMemsetAsync(d_part, 0xA5, part_bytes, stream), "e2e cand part");
                launch_exp8_test<TA>(T, count, 1, w, dj, dn, dx, stream, nullptr, nullptr, nullptr, nullptr);
                if (packed_q2) k_moe_down<T_Q4_0, true, true><<<grid_dn, DN_THREADS, smem_dn, stream>>>(w_q2, dj_dn, dn, nullptr, part_mid);
                else           k_moe_down<T_IQ4_NL, true, false, true><<<grid_dn, DN_THREADS, smem_dn, stream>>>(w_iq4, dj_dn, dn, nullptr, part_mid);
                ck(cudaStreamSynchronize(stream), "e2e cand wait");
                ck(cudaMemcpy(got_p.data(), d_part, part_bytes, cudaMemcpyDeviceToHost), "e2e cand read");
                if (ref_p != got_p)
                    throw std::runtime_error("EXP8 E2E FFN 0-ULP mismatch: card " + std::to_string(dev) +
                                             " type " + std::to_string(TA) + " T " + std::to_string(T) +
                                             " packed_q2 " + std::to_string(packed_q2));
                ++checks;
            }
        }
        for (void* p : {(void*) d_iq4, (void*) d_q2, (void*) d_part, (void*) dj_dn}) ck(cudaFree(p), "e2e free");
    }

    ck(cudaStreamDestroy(stream), "free stream");
    for (void* p : {(void*) dw, (void*) dx, (void*) dgu, (void*) dhq, (void*) dj, (void*) dn,
                    (void*) dwl1, (void*) dwlm, (void*) dn1, (void*) dnm})
        ck(cudaFree(p), "free");
    return checks;
}

template <GgmlType TA>
void bench_type() {
    constexpr int experts = 48, cols = N, ffs = 320, rows = 2 * ffs, pairs = MAXT * TOPK;
    constexpr size_t gu_rb = cols / qtraits<TA>::qk * qtraits<TA>::bs;
    constexpr size_t dn_rb = ffs / qtraits<T_IQ4_NL>::qk * qtraits<T_IQ4_NL>::bs;
    std::vector<unsigned char> w_gu((size_t) experts * rows * gu_rb);
    std::vector<unsigned char> w_dn((size_t) experts * cols * dn_rb);
    fill_test_weights<TA>(w_gu, experts, rows, gu_rb, 1);
    fill_test_weights<T_IQ4_NL>(w_dn, experts, cols, dn_rb, 2);

    std::vector<block_q8_1> x(MAXT * cols / 32);
    for (size_t i = 0; i < x.size(); ++i) {
        int sum = 0;
        for (int k = 0; k < 32; ++k) sum += x[i].qs[k] = (int) ((i * 7 + k * 11) % 255) - 127;
        x[i].ds = make_half2(__float2half(0.02f), __float2half(sum * 0.02f));
    }

    char *d_wgu = nullptr, *d_wdn = nullptr;
    block_q8_1* dx = nullptr;
    float *d_gu = nullptr, *d_part = nullptr;
    Job *dj_gu = nullptr, *dj_dn = nullptr;
    int *dn = nullptr, *dwl1 = nullptr, *dwlm = nullptr, *dn_wl = nullptr;
    ck(cudaMalloc((void**) &d_wgu, w_gu.size()), "b wgu");
    ck(cudaMalloc((void**) &d_wdn, w_dn.size()), "b wdn");
    ck(cudaMalloc((void**) &dx, x.size() * sizeof(block_q8_1)), "b x");
    ck(cudaMalloc((void**) &d_gu, (size_t) pairs * rows * sizeof(float)), "b gu");
    ck(cudaMalloc((void**) &d_part, (size_t) pairs * cols * sizeof(float)), "b part");
    ck(cudaMalloc((void**) &dj_gu, pairs * sizeof(Job)), "b jgu");
    ck(cudaMalloc((void**) &dj_dn, pairs * sizeof(Job)), "b jdn");
    ck(cudaMalloc((void**) &dn, sizeof(int)), "b n");
    ck(cudaMalloc((void**) &dwl1, pairs * sizeof(int)), "b wl1");
    ck(cudaMalloc((void**) &dwlm, pairs * sizeof(int)), "b wlm");
    ck(cudaMalloc((void**) &dn_wl, 2 * sizeof(int)), "b nwl");
    ck(cudaMemcpy(d_wgu, w_gu.data(), w_gu.size(), cudaMemcpyHostToDevice), "b wgu copy");
    ck(cudaMemcpy(d_wdn, w_dn.data(), w_dn.size(), cudaMemcpyHostToDevice), "b wdn copy");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "b x copy");

    const MoeW wg{d_wgu, (size_t) rows * gu_rb, gu_rb, rows, cols, nullptr};
    const MoeW wd{d_wdn, (size_t) cols * dn_rb, dn_rb, cols, ffs, nullptr};
    cudaStream_t s;
    cudaEvent_t a, b;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "b stream");
    ck(cudaEventCreate(&a), "b ea");
    ck(cudaEventCreate(&b), "b eb");

    for (int T : {1, 4, 6, 8, 16}) {
        const int cap = T * TOPK;
        std::vector<Job> jg(cap), jd(cap);
        int count = 0, cursor = 0, left = cap;
        const int target1 = (cap * 6) / 10;
        while (left > 0 && count < cap) {
            const int want = (cursor < target1 || T == 1) ? 1 : std::min(T, 2 + (count % 3));
            const int n_each = std::min(left, want);
            jg[count].w = jd[count].w = (count * 7) % experts;
            jg[count].n = jd[count].n = n_each;
            for (int t = 0; t < n_each; ++t) {
                jg[count].y[t] = (cursor + t) % T;
                jg[count].out[t] = jd[count].y[t] = jd[count].out[t] = cursor;
                ++cursor;
            }
            left -= n_each;
            ++count;
        }
        std::vector<int> hwl1, hwlm;
        for (int j = 0; j < count; ++j) {
            if (jg[j].n == 1) hwl1.push_back(j);
            else hwlm.push_back(j);
        }
        const int h_nwl[2] = {(int) hwl1.size(), (int) hwlm.size()};
        ck(cudaMemcpyAsync(dj_gu, jg.data(), count * sizeof(Job), cudaMemcpyHostToDevice, s), "b jg");
        ck(cudaMemcpyAsync(dj_dn, jd.data(), count * sizeof(Job), cudaMemcpyHostToDevice, s), "b jd");
        ck(cudaMemcpyAsync(dn, &count, sizeof(int), cudaMemcpyHostToDevice, s), "b count");
        ck(cudaMemcpyAsync(dn_wl, h_nwl, sizeof(h_nwl), cudaMemcpyHostToDevice, s), "b nwl");
        if (!hwl1.empty()) ck(cudaMemcpyAsync(dwl1, hwl1.data(), hwl1.size() * sizeof(int), cudaMemcpyHostToDevice, s), "b wl1");
        if (!hwlm.empty()) ck(cudaMemcpyAsync(dwlm, hwlm.data(), hwlm.size() * sizeof(int), cudaMemcpyHostToDevice, s), "b wlm");
        ck(cudaStreamSynchronize(s), "b up wait");

        const int rpb = DN_THREADS / (ffs / DN_U), gx_dn = (cols + rpb - 1) / rpb;
        const size_t smem_dn = (size_t) T * DN_THREADS * sizeof(float);
        auto run_arm = [&](int arm) {
            // arm 0=base, 1=exp8_lds0, 2=exp8_lds1, 3=exp8_lds2, 4=exp8_lds1+p5, 5=exp8_lds2+p5
            if (arm == 0) {
                if constexpr (exp8_is_iq4<TA>()) {
                    const dim3 g16((rows + 7) / 8, cap);
#define B_IQ(t) case t: k_moe_gu_iq<TA, t, 16><<<g16, MMVQ_THREADS, 0, s>>>(wg, dj_gu, dn, dx, d_gu); break
                    switch (T) { B_IQ(1); B_IQ(4); B_IQ(6); B_IQ(8); B_IQ(16); default: break; }
#undef B_IQ
                } else {
                    launch_baseline_gu<TA>(T, cap, wg, dj_gu, dn, dx, d_gu, s);
                }
                if (T >= 9) {
                    k_moe_h<<<(cap * DN_MAXB + 7) / 8, 256, 0, s>>>(d_gu, ffs, cap);
                    k_moe_down<T_IQ4_NL, true, false, false><<<dim3(gx_dn, cap), DN_THREADS, smem_dn, s>>>(wd, dj_dn, dn, d_gu, d_part);
                } else {
                    k_moe_down<T_IQ4_NL, false, false, false><<<dim3(gx_dn, cap), DN_THREADS, smem_dn, s>>>(wd, dj_dn, dn, d_gu, d_part);
                }
                return;
            }
            if (arm == 1 || arm == 2 || arm == 3) {
                const int lds = arm - 1;
                launch_exp8_test<TA>(T, cap, lds, wg, dj_gu, dn, dx, s, nullptr, nullptr, nullptr, nullptr);
                k_moe_down<T_IQ4_NL, true, false, false><<<dim3(gx_dn, cap), DN_THREADS, smem_dn, s>>>(wd, dj_dn, dn, nullptr, d_part);
                return;
            }
            const int lds = (arm == 5) ? 2 : 1;
            const bool split = (T >= 4);
            if (split) {
                launch_exp8_test<TA>(1, cap, lds, wg, dj_gu, dn_wl, dx, s, nullptr, nullptr, dwl1, nullptr);
                launch_exp8_test<TA>(T, std::max(1, cap / 2), lds, wg, dj_gu, dn_wl + 1, dx, s, nullptr, nullptr, dwlm, nullptr);
                k_moe_down1<T_IQ4_NL, true, false><<<dim3(gx_dn, cap), DN_THREADS, 0, s>>>(wd, dj_dn, dn_wl, nullptr, d_part, dwl1);
                k_moe_down<T_IQ4_NL, true, false, true><<<dim3(gx_dn, std::max(1, cap / 2)), DN_THREADS, smem_dn, s>>>(wd, dj_dn, dn_wl + 1, nullptr, d_part, dwlm);
            } else {
                launch_exp8_test<TA>(T, cap, lds, wg, dj_gu, dn, dx, s, nullptr, nullptr, nullptr, nullptr);
                k_moe_down<T_IQ4_NL, true, false, false><<<dim3(gx_dn, cap), DN_THREADS, smem_dn, s>>>(wd, dj_dn, dn, nullptr, d_part);
            }
        };

        std::vector<int> arms = {0, 1, 2, 4};
        if constexpr (exp8_is_iq4<TA>()) arms = {0, 1, 2, 3, 4, 5};
        const char* names[] = {"base", "lds0", "lds1", "lds2", "full1", "full2"};
        std::vector<int> order = arms;
        order.insert(order.end(), arms.rbegin(), arms.rend());
        double sum_us[6] = {};
        for (int arm : order) {
            cudaGraph_t bg;
            cudaGraphExec_t be;
            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "b cap");
            run_arm(arm);
            ck(cudaStreamEndCapture(s, &bg), "b end");
            ck(cudaGraphInstantiate(&be, bg, nullptr, nullptr, 0), "b inst");
            for (int i = 0; i < 200; ++i) ck(cudaGraphLaunch(be, s), "b warm");
            ck(cudaEventRecord(a, s), "b rec a");
            for (int i = 0; i < 100; ++i) ck(cudaGraphLaunch(be, s), "b run");
            ck(cudaEventRecord(b, s), "b rec b");
            ck(cudaEventSynchronize(b), "b wait");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, a, b), "b elapsed");
            sum_us[arm] += (ms * 1000.0 / 100.0) * 0.5;
            ck(cudaGraphExecDestroy(be), "b exec d");
            ck(cudaGraphDestroy(bg), "b graph d");
        }
        std::cout << "bench card 0 type " << TA << " T " << T << " jobs " << count
                  << " (n1=" << h_nwl[0] << ",nm=" << h_nwl[1] << "):";
        for (int arm : arms) std::cout << " " << names[arm] << "=" << sum_us[arm] << "us";
        std::cout << "\n";
    }

    ck(cudaEventDestroy(a), "b ea d");
    ck(cudaEventDestroy(b), "b eb d");
    ck(cudaStreamDestroy(s), "b stream d");
    for (void* p : {(void*) d_wgu, (void*) d_wdn, (void*) dx, (void*) d_gu, (void*) d_part,
                    (void*) dj_gu, (void*) dj_dn, (void*) dn, (void*) dwl1, (void*) dwlm, (void*) dn_wl})
        ck(cudaFree(p), "b free");
}
} // namespace fnx

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--bench"))
            throw std::runtime_error("usage: split_moe_exp8_test [--bench]");
        const bool bench = argc == 2;
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
            checks += fnx::run_type<fnx::T_Q4_K>(dev);
            checks += fnx::run_type<fnx::T_Q5_K>(dev);
            checks += fnx::run_type<fnx::T_Q6_K>(dev);
            checks += fnx::run_type<fnx::T_Q8_0>(dev);
        }
        std::cout << "MoE EXP8: " << checks << " exact direct/graph/cache/e2e checks passed\n";
        if (bench) {
            fnx::ck(cudaSetDevice(0), "device 0 bench");
            fnx::bench_type<fnx::T_IQ2_XXS>();
            fnx::bench_type<fnx::T_IQ2_S>();
            fnx::bench_type<fnx::T_IQ3_XXS>();
            fnx::bench_type<fnx::T_IQ3_S>();
            fnx::bench_type<fnx::T_IQ4_XS>();
            fnx::bench_type<fnx::T_Q4_K>();
            fnx::bench_type<fnx::T_Q5_K>();
            fnx::bench_type<fnx::T_Q8_0>();
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

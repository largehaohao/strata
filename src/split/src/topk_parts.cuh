// Included after topk.cuh inside fnx. Host standard headers come from kernels.cu.
#pragma once
namespace {
inline int topk_parts_mode() {
    static const int parts = [] {
        const auto p = env_int("FNX_TOPK_PARTS", 0);
        if (p != 0 && p != 2 && p != 4 && p != 8 && p != 16 && p != 32)
            throw std::runtime_error("FNX_TOPK_PARTS: expected0,2,4,8,16,32");
#if !FNX_HIP || FNX_NCARD != 2 || FNX_27B
        if (p) throw std::runtime_error("FNX_TOPK_PARTS requires two-card HIP Flash-Next");
#endif
        return (int)p;
    }();
    return parts;
}
}

#if FNX_HIP && FNX_NCARD == 2 && !FNX_27B
// Exact per-partition candidates followed by an exact score-bit/token-id merge.
namespace {
constexpr int PART_THREADS = 256, PART_CAP = 256;
__global__ void __launch_bounds__(PART_THREADS) k_topk_parts(float* __restrict__ logits, int full_n, int v0, int parts, const int* __restrict__ pen_tok,
                                                           const int* __restrict__ pen_cnt, const int* __restrict__ pen_n, int pen_stride,
                                                           const float* __restrict__ pen_par, float* __restrict__ out_v,
                                                           int* __restrict__ out_i, unsigned* err) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int part = blockIdx.y;
    const int width = (full_n + parts - 1) / parts;
    const int start = part * width;
    const int n = min(width, full_n - start);
    float* L = logits + (size_t) row * full_n + start;
    v0 += start;
    const int out_row = row * parts + part;
    // a NaN or inf among the head's logits (before the penalties, which a request sets): err bit 16
    bool bad = false;
    for (int j = tid; j < n; j += PART_THREADS) bad = bad || !isfinite(L[j]);
    if (bad) atomicOr(err, 16u);
    __syncthreads();   // every logit checked before any penalty changes one
    // the row's penalties on this card's tokens (each token once in the list)
    const float repeat = pen_par[row * 3], freq = pen_par[row * 3 + 1], present = pen_par[row * 3 + 2];   // the row's penalty parameters
    for (int i = tid; i < pen_n[row]; i += PART_THREADS) {
        const int tok = pen_tok[row * pen_stride + i] - v0, cnt = pen_cnt[row * pen_stride + i];
        if (tok < 0 || tok >= n || cnt <= 0) continue;
        float l = L[tok];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[tok] = l - (float) cnt * freq - present;
    }
    __syncthreads();
    // the TOPK_N-th highest key: four passes of 8 bits
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = TOPK_N;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += PART_THREADS) hist[b] = 0;
        __syncthreads();
        for (int jb = 0; jb < n; jb += PART_THREADS) {
            const int j = jb + tid;
            const unsigned k = j < n ? tk_key(L[j]) : 0u;
            const bool in = j < n && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
#endif
            if (in && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned) __popc(peers));
        }
        __syncthreads();
        if (tid == 0) {
            unsigned acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want) break;
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned) b << shift);
            want_s = want - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // everything at or above the threshold, then sorted: (key desc, id asc) as one 64-bit key, descending
    __shared__ unsigned long long cand[PART_CAP];
    __shared__ unsigned n_cand;
    for (int i = tid; i < PART_CAP; i += PART_THREADS) cand[i] = 0ull;
    if (tid == 0) n_cand = 0;
    __syncthreads();
    for (int j = tid; j < n; j += PART_THREADS) {
        const unsigned k = tk_key(L[j]);
        if (k >= prefix) {
            const unsigned s = atomicAdd(&n_cand, 1u);
            if (s < PART_CAP) cand[s] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
    }
    __syncthreads();
    const bool wide = n_cand > PART_CAP;
    __syncthreads();   // every thread has read n_cand before the wide pass resets it
    if (wide) {   // block-uniform. The first TOPK_N of the sorted set are every key above the threshold
        // (TOPK_N - want of them) and the `want` lowest ids at it: those exactly, the ties taken in id order a chunk at a
        // time (a block-wide rank from each warp's ballot), stopping once `want` are in
        __shared__ unsigned warp_n[PART_THREADS / 32];
        const int warp = tid >> 5;
        for (int i = tid; i < PART_CAP; i += PART_THREADS) cand[i] = 0ull;
        if (tid == 0) n_cand = 0;
        __syncthreads();
        for (int j = tid; j < n; j += PART_THREADS) {
            const unsigned k = tk_key(L[j]);
            if (k > prefix) cand[atomicAdd(&n_cand, 1u)] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
        const unsigned base = TOPK_N - want;
        unsigned taken = 0;
        for (int jb = 0; jb < n && taken < want; jb += PART_THREADS) {
            const int j = jb + tid;
            const bool eq = j < n && tk_key(L[j]) == prefix;
            const unsigned bal = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) warp_n[warp] = (unsigned) __popc(bal);
            __syncthreads();
            unsigned before = (unsigned) __popc(bal & ((1u << lane) - 1u)), total = 0;
            for (int w = 0; w < PART_THREADS / 32; ++w) {
                before += w < warp ? warp_n[w] : 0u;
                total += warp_n[w];
            }
            if (eq && taken + before < want) cand[base + taken + before] = ((unsigned long long) prefix << 32) | (0xffffffffu - (unsigned) j);
            taken += total;
            __syncthreads();   // warp_n read by every thread before the next chunk writes it
        }
        __syncthreads();
    }
    for (int size = 2; size <= PART_CAP; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < PART_CAP) {
                const int o = tid ^ stride;
                if (o > tid) {
                    const bool desc = (tid & size) == 0;
                    const unsigned long long a = cand[tid], b = cand[o];
                    if (desc ? a < b : a > b) {
                        cand[tid] = b;
                        cand[o] = a;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int) (0xffffffffu - (unsigned) (c & 0xffffffffu));
        out_i[out_row * TOPK_N + tid] = v0 + j;
        out_v[out_row * TOPK_N + tid] = L[j];
    }
}


template<int PARTS>
__global__ void __launch_bounds__(256) k_topk_merge_parts(const float* values, const int* ids, float* out_v, int* out_i) {
    constexpr int COUNT = PARTS * TOPK_N;
    __shared__ unsigned long long keys[COUNT];
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int j = tid; j < COUNT; j += 256) {
        const size_t at = (size_t) row * COUNT + j;
        keys[j] = ((unsigned long long) tk_key(values[at]) << 32) | (0xffffffffu - (unsigned) ids[at]);
    }
    __syncthreads();
    for (int size = 2; size <= COUNT; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < COUNT; j += 256) {
                const int other = j ^ stride;
                if (other > j) {
                    const unsigned long long a = keys[j], b = keys[other];
                    if (((j & size) == 0) ? a < b : a > b) {
                        keys[j] = b; keys[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < TOPK_N) {
        const unsigned long long key = keys[tid];
        const unsigned ordered = (unsigned) (key >> 32);
        const unsigned bits = (ordered & 0x80000000u) ? ordered & 0x7fffffffu : ~ordered;
        out_v[row * TOPK_N + tid] = __uint_as_float(bits);
        out_i[row * TOPK_N + tid] = (int) (0xffffffffu - (unsigned) key);
    }
}
}  // namespace
namespace {
inline void topk_parts_check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
struct TopKPartsScratch {float* values = nullptr; int* ids = nullptr;};
using TopKPartsKey = std::pair<int, std::uintptr_t>;
struct TopKPartsRegistry {
    std::mutex mutex;
    std::map<TopKPartsKey, std::shared_ptr<TopKPartsScratch>> streams;
};
inline TopKPartsRegistry& topk_parts_registry() {static TopKPartsRegistry r;return r;}
inline std::shared_ptr<TopKPartsScratch> topk_parts_scratch(cudaStream_t stream, int parts) {
    int device;
    topk_parts_check(cudaGetDevice(&device), "top-k device");
    const TopKPartsKey key{device, reinterpret_cast<std::uintptr_t>(stream)};
    auto& registry = topk_parts_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.streams.find(key);
    if (found != registry.streams.end()) return found->second;
    cudaStreamCaptureStatus status;
    topk_parts_check(cudaStreamIsCapturing(stream, &status), "top-k capture status");
    if (status != cudaStreamCaptureStatusNone) return {};
    hipDeviceProp_t prop{};
    topk_parts_check(hipGetDeviceProperties(&prop, device), "top-k properties");
    if (std::string(prop.gcnArchName).find("gfx1100") != 0)
        throw std::runtime_error("FNX_TOPK_PARTS requires gfx1100");
    auto scratch = std::make_shared<TopKPartsScratch>();
    const size_t bytes = (size_t) MAXT * parts * TOPK_N * sizeof(float);
    topk_parts_check(cudaMalloc(reinterpret_cast<void**>(&scratch->values), bytes), "top-k part values");
    const auto error = cudaMalloc(reinterpret_cast<void**>(&scratch->ids), bytes);
    if (error != cudaSuccess) {(void)cudaFree(scratch->values);topk_parts_check(error, "top-k part ids");}
    registry.streams.emplace(key, scratch);
    own([key, scratch] {
        auto& registry = topk_parts_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.streams.erase(key);
        (void)cudaSetDevice(key.first);
        (void)cudaFree(scratch->ids);
        (void)cudaFree(scratch->values);
    });
    return scratch;
}
inline bool topk_parts(float* logits, int n, int T, int v0, const int* pen_tok, const int* pen_cnt,
    const int* pen_n, int pen_stride, const float* pen_par, float* out_v, int* out_i, unsigned* err, void* stream) {
    const int parts = topk_parts_mode();
    if (!parts || T < 1 || T > MAXT || n - (parts - 1) * ((n + parts - 1) / parts) < TOPK_N) return false;
    auto scratch = topk_parts_scratch((cudaStream_t)stream, parts);
    if (!scratch) return false;
    k_topk_parts<<<dim3(T, parts), PART_THREADS, 0, (cudaStream_t)stream>>>(logits, n, v0, parts, pen_tok, pen_cnt,
        pen_n, pen_stride, pen_par, scratch->values, scratch->ids, err);
    switch (parts) {
        case 2: k_topk_merge_parts<2><<<T, 256, 0, (cudaStream_t)stream>>>(scratch->values, scratch->ids, out_v, out_i); break;
        case 4: k_topk_merge_parts<4><<<T, 256, 0, (cudaStream_t)stream>>>(scratch->values, scratch->ids, out_v, out_i); break;
        case 8: k_topk_merge_parts<8><<<T, 256, 0, (cudaStream_t)stream>>>(scratch->values, scratch->ids, out_v, out_i); break;
        case 16: k_topk_merge_parts<16><<<T, 256, 0, (cudaStream_t)stream>>>(scratch->values, scratch->ids, out_v, out_i); break;
        case 32: k_topk_merge_parts<32><<<T, 256, 0, (cudaStream_t)stream>>>(scratch->values, scratch->ids, out_v, out_i); break;
    }
    return true;
}
}

#else
namespace {
inline bool topk_parts(float*, int, int, int, const int*, const int*, const int*, int, const float*, float*, int*, unsigned*, void*) {
    (void)topk_parts_mode();return false;
}
}
#endif

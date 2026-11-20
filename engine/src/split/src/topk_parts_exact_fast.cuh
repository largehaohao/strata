// Exact radix early-stop and a smaller sort for already selected sets.
#pragma once
template<int LIMIT>
__device__ __forceinline__ void topk_sort_shared(unsigned long long* cand) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= LIMIT; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < LIMIT) {
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
 }

template<bool EARLY>
__global__ void __launch_bounds__(PART_THREADS) k_topk_parts_exact_fast(float* __restrict__ logits, int full_n, int v0, int parts, const int* __restrict__ pen_tok,
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
    __shared__ unsigned prefix_s, want_s, done_s;
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
            done_s = hist[b] == want_s;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        const bool selected_all = done_s != 0;
        __syncthreads();
        if constexpr (EARLY) { if (selected_all) break; }
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
    // Exactly 64 chosen keys occupy the first 64 slots, including the
    // full-tie fallback. Larger unresolved tie sets keep the full sort.
    if (wide || n_cand <= TOPK_N) topk_sort_shared<TOPK_N>(cand);
    else topk_sort_shared<PART_CAP>(cand);
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int) (0xffffffffu - (unsigned) (c & 0xffffffffu));
        out_i[out_row * TOPK_N + tid] = v0 + j;
        out_v[out_row * TOPK_N + tid] = L[j];
    }
}

template<int PARTS>
__global__ void __launch_bounds__(256) k_topk_merge_tree(const float* values, const int* ids, float* out_v, int* out_i) {
    constexpr int COUNT = PARTS * TOPK_N, LANES = 256;
    constexpr int PER_THREAD = (COUNT + LANES - 1) / LANES;
    __shared__ unsigned long long keys[COUNT];
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int j = tid; j < COUNT; j += LANES) {
        const size_t at = (size_t) row * COUNT + j;
        keys[j] = ((unsigned long long) tk_key(values[at]) << 32) | (0xffffffffu - (unsigned) ids[at]);
    }
    __syncthreads();
    for (int lists = PARTS; lists > 1; lists >>= 1) {
        unsigned long long held[PER_THREAD];
        int destination[PER_THREAD];
#pragma unroll
        for (int slot = 0; slot < PER_THREAD; ++slot) {
            const int j = tid + slot * LANES;
            destination[slot] = -1;
            held[slot] = 0;
            if (j < lists * TOPK_N) {
                const int list = j / TOPK_N, own = j % TOPK_N;
                const unsigned long long key = keys[j];
                const int other = (list ^ 1) * TOPK_N;
                int lo = 0, hi = TOPK_N;
                while (lo < hi) {
                    const int mid = (lo + hi) / 2;
                    if (keys[other + mid] > key) lo = mid + 1;
                    else hi = mid;
                }
                const int rank = own + lo;
                held[slot] = key;
                if (rank < TOPK_N) destination[slot] = (list / 2) * TOPK_N + rank;
            }
        }
        __syncthreads();
#pragma unroll
        for (int slot = 0; slot < PER_THREAD; ++slot)
            if (destination[slot] >= 0) keys[destination[slot]] = held[slot];
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

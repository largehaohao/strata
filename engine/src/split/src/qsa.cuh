// QSA: the attention layers' block selection (included by kernels.cu), as llama.cpp's qwen4exp graph builds it.
// Blocks of QSA_POOL positions in order; a block's key is the mean of its members' raw indexer keys, RMS-normed and
// rotated to its first member's position, kept in f16 (the cache type). A row at position p sees the complete
// blocks [0, (p + 1) / 4) and always attends its tail, the last (p + 1) % 4 positions; it scores the visible blocks
// with its indexer query (4 heads, each RMS-normed and rotated to p): sum over heads of relu(q_h . k) / sqrt(128),
// and keeps the QSA_TOP highest (the lower block on a tie; llama.cpp's CUDA top-k takes tied blocks in atomic order,
// so any choice among exact ties is one it can make). With QSA_TOP or fewer visible blocks every block is kept: the
// attention is dense. Every card computes the same selection from the same inputs.
namespace {

constexpr int QSA_POOL = 4, QSA_TOP = QSA_TOP_K / QSA_POOL, QSA_DIM = QSA_KEY, QSA_ROT = NROT;
static_assert(QSA_DIM % 16 == 0 && QSA_ROT <= QSA_DIM && QSA_TOP_K % QSA_POOL == 0,
              "the score tiles take QSA_DIM in sixteens, the rotary must fit a key, and top_k must divide into whole pools");
constexpr int QSA_ROWS = QSA_HEADS * QSA_DIM + QSA_DIM;   // the projection's rows: q (QSA_HEADS heads), then k

constexpr int QSA_MAXSEL = QSA_TOP * QSA_POOL + QSA_POOL - 1;

__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((unsigned) h << 16); }

// y[t][r] = W_r . x[t] for the QSA_ROWS rows (r < QSA_HEADS * QSA_DIM: wq row r, else wk row r - that many), bf16 weights,
// f32 activations; a warp per row
template <int T>
__global__ void __launch_bounds__(256) k_idx_proj(const uint16_t* __restrict__ wq, const uint16_t* __restrict__ wk,
                                                 const float* __restrict__ x, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = (wq ? 0 : QSA_HEADS * QSA_DIM) + blockIdx.x * 8 + warp;
    if (r >= QSA_ROWS) return;
    const uint16_t* w = r < QSA_HEADS * QSA_DIM ? wq + (size_t) r * N : wk + (size_t) (r - QSA_HEADS * QSA_DIM) * N;
    float acc[T] = {};
#pragma unroll 4
    for (int c = lane * 2; c < N; c += 64) {
        const unsigned wv = *reinterpret_cast<const unsigned*>(w + c);
        const float w0 = bf16f((uint16_t) (wv & 0xffffu)), w1 = bf16f((uint16_t) (wv >> 16));
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float2 xv = *reinterpret_cast<const float2*>(x + (size_t) t * N + c);
            acc[t] = fmaf(w0, xv.x, fmaf(w1, xv.y, acc[t]));
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * QSA_ROWS + r] = v;
    }
}

// x (QSA_DIM values, one per thread): RMS norm with w, then NEOX rotary on the first QSA_ROT dims at pos
__device__ __forceinline__ float idx_norm_rope(float v, const float* w, int pos, float eps, const Rope& base, float* sh, float* buf) {
    const int i = threadIdx.x;
    const float ms = block_sum(v * v, sh) / (float) QSA_DIM;
    const float y = v * rsqrtf(ms + eps) * w[i];
    buf[i] = y;
    __syncthreads();
    float out = y;
    if (i < QSA_ROT) {
        const int k = i % (QSA_ROT / 2);
        const float theta = (float) rope_pos(base, pos, k) * rope_unit_freq(base, k, QSA_ROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        sn *= base.mscale; cs *= base.mscale;
        const float x0 = buf[k], x1 = buf[k + QSA_ROT / 2];
        out = i < QSA_ROT / 2 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    __syncthreads();
    return out;
}

// block (row t, part): part < 4: indexer query head `part` normed and rotated into q; part 4: the raw key into the ring
__global__ void __launch_bounds__(QSA_DIM) k_idx_prep(const float* __restrict__ y, const float* __restrict__ qn, __half* ring, int ring_rows,
                                                      float* __restrict__ q, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, part = blockIdx.y, i = threadIdx.x, p = *pos0p + t;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] =
            idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else {
        ring[(size_t) (p % ring_rows) * QSA_DIM + i] = __float2half_rn(yt[QSA_HEADS * QSA_DIM + i]);
    }
}

// a row whose position completes a block pools it: the mean of the members' raw keys (f16), normed, rotated to the
// first member, stored f16
__global__ void __launch_bounds__(QSA_DIM) k_idx_pool(const __half* ring, int ring_rows, const float* __restrict__ kn, __half* pooled, KvPages g,
                                                      const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, i = threadIdx.x, p = *pos0p + t;
    if ((p + 1) % QSA_POOL != 0) return;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    float v = 0.f;
#pragma unroll
    for (int m = 0; m < QSA_POOL; ++m) v += __half2float(ring[(size_t) ((p - (QSA_POOL - 1) + m) % ring_rows) * QSA_DIM + i]);
    v *= 1.0f / (float) QSA_POOL;
    const int j = (p + 1) / QSA_POOL - 1;
    pooled[kvp_block(g, j) + i] = __float2half_rn(idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf));
}

// k_idx_prep and k_idx_pool in one launch: parts 0 .. QSA_HEADS as k_idx_prep's, part QSA_HEADS + 1 k_idx_pool's, which
// takes the window's own raw keys from y (rounded to f16 as the ring holds them) and the earlier ones from the ring
__global__ void __launch_bounds__(QSA_DIM) k_idx_prep_pool(const float* __restrict__ y, const float* __restrict__ qn,
                                                           const float* __restrict__ kn, __half* ring, int ring_rows, float* __restrict__ q,
                                                           __half* pooled, KvPages g, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, part = blockIdx.y + (q ? 0 : QSA_HEADS), i = threadIdx.x, pos0 = *pos0p, p = pos0 + t;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] = idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else if (part == QSA_HEADS) {
        ring[(size_t) (p % ring_rows) * QSA_DIM + i] = __float2half_rn(yt[QSA_HEADS * QSA_DIM + i]);
    } else {
        if ((p + 1) % QSA_POOL != 0) return;
        float v = 0.f;
#pragma unroll
        for (int m = 0; m < QSA_POOL; ++m) {
            const int pm = p - (QSA_POOL - 1) + m;
            v += __half2float(pm >= pos0 ? __float2half_rn(y[(size_t) (pm - pos0) * QSA_ROWS + QSA_HEADS * QSA_DIM + i])
                                         : ring[(size_t) (pm % ring_rows) * QSA_DIM + i]);
        }
        v *= 1.0f / (float) QSA_POOL;
        const int j = (p + 1) / QSA_POOL - 1;
        pooled[kvp_block(g, j) + i] = __float2half_rn(idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf));
    }
}

// scores[t][j] for the visible blocks j of rows with more than QSA_TOP of them: a thread per block
__global__ void __launch_bounds__(256) k_idx_score(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                   float* __restrict__ scores, int max_pools, const int* pos0p, LaneRows lr) {
    const int t = blockIdx.y, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL;
    if (nv <= QSA_TOP) return;
    g = row_pages(g, lr, t);
    __shared__ float qs[QSA_HEADS * QSA_DIM];
    for (int k = threadIdx.x; k < QSA_HEADS * QSA_DIM; k += blockDim.x) qs[k] = q[(size_t) t * QSA_HEADS * QSA_DIM + k];
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nv) return;
    const uint4* kr = reinterpret_cast<const uint4*>(pooled + kvp_block(g, j));
    float d[QSA_HEADS] = {};
#pragma unroll 4
    for (int c = 0; c < QSA_DIM / 8; ++c) {
        const uint4 u = kr[c];
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 kv = __half22float2(h2[k]);
            const int dim = c * 8 + 2 * k;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) d[h] = fmaf(qs[h * QSA_DIM + dim], kv.x, fmaf(qs[h * QSA_DIM + dim + 1], kv.y, d[h]));
        }
    }
    const float w = rsqrtf((float) QSA_DIM);
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(d[h], 0.f) * w;
    scores[(size_t) t * max_pools + j] = s;
}

// Same per-key FP32 expressions; each CTA reuses its query across multiple key tiles.
__global__ void __launch_bounds__(256) k_idx_score_persistent(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                   float* __restrict__ scores, int max_pools, const int* pos0p, LaneRows lr) {
    const int t = blockIdx.y, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL;
    if (nv <= QSA_TOP || blockIdx.x * blockDim.x >= (unsigned) nv) return;
    g = row_pages(g, lr, t);
    __shared__ float qs[QSA_HEADS * QSA_DIM];
    for (int k = threadIdx.x; k < QSA_HEADS * QSA_DIM; k += blockDim.x) qs[k] = q[(size_t) t * QSA_HEADS * QSA_DIM + k];
    __syncthreads();
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nv; j += blockDim.x * gridDim.x) {
    const uint4* kr = reinterpret_cast<const uint4*>(pooled + kvp_block(g, j));
    float d[QSA_HEADS] = {};
#pragma unroll 4
    for (int c = 0; c < QSA_DIM / 8; ++c) {
        const uint4 u = kr[c];
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 kv = __half22float2(h2[k]);
            const int dim = c * 8 + 2 * k;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) d[h] = fmaf(qs[h * QSA_DIM + dim], kv.x, fmaf(qs[h * QSA_DIM + dim + 1], kv.y, d[h]));
        }
    }
    const float w = rsqrtf((float) QSA_DIM);
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(d[h], 0.f) * w;
    scores[(size_t) t * max_pools + j] = s;
    }
}

// the same for a prompt's rows on the tensor cores: a block takes 16 rows (their 4 heads: 64 rows of q, in f16 high and
// low halves) against 64 blocks' keys (f16 already), f32 accumulation; each key is read once per 16 rows instead of once
// per row. Columns past a row's visible blocks get scores too (the select reads only the visible ones)
constexpr int QSC_ROWS = 16, QSC_KEYS = 64, QSC_LD = QSA_DIM + 8, QSC_SLD = 16 + 4;
__global__ void __launch_bounds__(256) k_idx_score_tc(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                      float* __restrict__ scores, int max_pools, const int* pos0p, int T) {
    using namespace nvcuda;
    const int t0 = blockIdx.y * QSC_ROWS, j0 = blockIdx.x * QSC_KEYS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tl = min(T, t0 + QSC_ROWS) - 1;   // the block's last row sees the most blocks
    const int nv_max = (*pos0p + tl + 1) / QSA_POOL;
    if (nv_max <= QSA_TOP || j0 >= nv_max) return;
    __shared__ __align__(32) __half qh[QSC_ROWS * QSA_HEADS][QSC_LD];
    __shared__ __align__(32) __half ql[QSC_ROWS * QSA_HEADS][QSC_LD];
    __shared__ __align__(32) float st[8][16][QSC_SLD];
    for (int e = tid; e < QSC_ROWS * QSA_HEADS * QSA_DIM; e += 256) {
        const int r = e / QSA_DIM, d = e % QSA_DIM;
        const float x = t0 + r / QSA_HEADS < T ? q[(size_t) t0 * QSA_HEADS * QSA_DIM + e] : 0.f;
        const __half h = __float2half_rn(x);
        qh[r][d] = h;
        ql[r][d] = __float2half_rn(x - __half2float(h));
    }
    __syncthreads();
    const float w = rsqrtf((float) QSA_DIM);
    // 16 output tiles (4 of rows x 4 of keys); warp w takes row tile w / 2, key tiles 2 (w % 2) and 2 (w % 2) + 1
    const int rt = warp / 2;
    for (int kk = 0; kk < 2; ++kk) {
        const int kt = (warp % 2) * 2 + kk, jt = j0 + kt * 16;
        if (jt >= nv_max) break;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
        wmma::fill_fragment(fs, 0.f);
#pragma unroll
        for (int d0 = 0; d0 < QSA_DIM; d0 += 16) {
            wmma::load_matrix_sync(fb, pooled + kvp_block(g, jt) + d0, QSA_DIM);   // 16 blocks never cross a page (48)
            wmma::load_matrix_sync(fa, &qh[rt * 16][d0], QSC_LD);
            wmma::mma_sync(fs, fa, fb, fs);
            wmma::load_matrix_sync(fa, &ql[rt * 16][d0], QSC_LD);
            wmma::mma_sync(fs, fa, fb, fs);
        }
        wmma::store_matrix_sync(&st[warp][0][0], fs, QSC_SLD, wmma::mem_row_major);
        __syncwarp();
        // the tile's 4 rows x 16 keys: the heads' relus summed in k_idx_score's order
        for (int e = lane; e < 4 * 16; e += 32) {
            const int r = e / 16, c = e % 16, t = t0 + rt * 4 + r, j = jt + c;
            if (t >= T || j >= max_pools) continue;
            float s = 0.f;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(st[warp][r * QSA_HEADS + h][c], 0.f) * w;
            scores[(size_t) t * max_pools + j] = s;
        }
        __syncwarp();
    }
}

// row t's attended positions on this card (block j owned by card kv_card(j)), as local indices kv_local_block(j) * 4 + p % 4, ascending,
// into cells[t] (n_cells[t] of them), or n_cells[t] = -1 when every visible block is kept (dense). A block per row: a radix select of the QSA_TOP-th highest score (scores >= 0: their bits
// order as unsigned), then the blocks above it and the first ties by index, in index order, then the tail.
constexpr int QSA_SEL_THREADS = 1024;
struct QsaCutoff {
    unsigned bits, ties;
};
__device__ __forceinline__ QsaCutoff qsa_cutoff(const unsigned* key, int nv) {
    const int tid = threadIdx.x;
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = QSA_TOP;   // find the want-th largest among keys matching prefix on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += QSA_SEL_THREADS) hist[b] = 0;
        __syncthreads();
        // a warp's keys of one bin counted with one atomic (scores share their high bits: per key, the first pass's
        // atomics all hit two or three bins and queue)
        for (int j0 = 0; j0 < nv; j0 += QSA_SEL_THREADS) {
            const int j = j0 + tid;
            const unsigned k = j < nv ? key[j] : 0u;
            const bool in = j < nv && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
#endif
            if (in && (tid & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned) __popc(peers));
        }
        __syncthreads();
        if (tid < 32) {   // the highest bin b > 0 where the count from 255 down reaches want, else 0; acc: the count above b
            const int lane = tid, hi = 255 - 8 * lane;   // lane l: bins hi .. hi - 7
            unsigned own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) own += hist[hi - i];
            unsigned before = own;   // inclusive scan over the lanes (higher bins first), then exclusive
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, before, o);
                if (lane >= o) before += v;
            }
            before -= own;
            const unsigned hit = __ballot_sync(0xffffffffu, before + own >= want);
            const int first = hit ? __ffs(hit) - 1 : 31;
            if (lane == first) {
                unsigned acc = before;
                int b = hi - 7;
                for (int i = 0; i < 8; ++i) {
                    const int bb = hi - i;
                    if (bb == 0 || acc + hist[bb] >= want) {
                        b = bb;
                        break;
                    }
                    acc += hist[bb];
                }
                prefix_s = prefix | ((unsigned) b << shift);
                want_s = want - acc;
            }
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    return {prefix, want};
}

__global__ void __launch_bounds__(QSA_SEL_THREADS) k_idx_select(const float* __restrict__ scores, int* __restrict__ cells,
                                                                int* __restrict__ n_cells, int max_pools, const int* pos0p,
                                                                int card, LaneRows lr) {
    const int t = blockIdx.x, tid = threadIdx.x, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL, n_tail = (p + 1) % QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);
    const QsaCutoff cut = qsa_cutoff(key, nv);
    const unsigned thr = cut.bits, want = cut.ties;
    const int chunk = (nv + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS, j0 = min(nv, tid * chunk), j1 = min(nv, j0 + chunk);
    int n_gt = 0, n_eq = 0;
    for (int j = j0; j < j1; ++j) {
        const unsigned k = key[j];
        n_gt += k > thr;
        n_eq += k == thr;
    }
    __shared__ int s_gt[QSA_SEL_THREADS], s_eq[QSA_SEL_THREADS];
    s_gt[tid] = n_gt;
    s_eq[tid] = n_eq;
    __syncthreads();
    // exclusive scans (Hillis-Steele over the block)
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0, e = tid >= o ? s_eq[tid - o] : 0;
        __syncthreads();
        s_gt[tid] += a;
        s_eq[tid] += e;
        __syncthreads();
    }
    // which blocks of the chunk are taken (ties: the first `want` by index), then how many of them this card owns
    const int eq_start = s_eq[tid] - n_eq;
    int own = 0;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            own += take && kv_card(j) == card;
        }
    }
    __syncthreads();
    s_gt[tid] = own;
    __syncthreads();
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0;
        __syncthreads();
        s_gt[tid] += a;
        __syncthreads();
    }
    int out = s_gt[tid] - own;   // owned selected blocks before this chunk
    int* row = cells + (size_t) t * QSA_MAXSEL;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            if (take && kv_card(j) == card) {
#pragma unroll
                for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(j) * QSA_POOL + m;
                ++out;
            }
        }
    }
    if (tid == QSA_SEL_THREADS - 1) {   // the last chunk's thread: `out` is the card's selected-block count
        int n = out * QSA_POOL;
        const int jt = p / QSA_POOL;   // the tail's block (incomplete)
        if (kv_card(jt) == card)
            for (int m = 0; m < n_tail; ++m) row[n++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = n;
    }
}

#if FNX_HIP && FNX_NCARD == 2 && !FNX_27B
constexpr int QSA_PAIR_MAX_POOLS = QSA_SEL_THREADS * 64;

__device__ __forceinline__ unsigned qsa_wave_sum_prefix(unsigned v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned before = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v += before;
    }
    return v;
}

// Exclusive sum in chronological chunk order. The second barrier publishes every
// wave's offset. Each invocation supplies its own shared storage.
__device__ __forceinline__ unsigned qsa_block_offset(unsigned count, unsigned* waves) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const unsigned inclusive = qsa_wave_sum_prefix(count);
    if (lane == 31) waves[wave] = inclusive;
    __syncthreads();
    if (wave == 0) {
        const unsigned total = waves[lane];
        waves[lane] = qsa_wave_sum_prefix(total) - total;
    }
    __syncthreads();
    return waves[wave] + inclusive - count;
}

// Sort the CTA's 1024 register keys. Steps within a wave use shuffles.
__device__ __forceinline__ unsigned qsa_sort_key(unsigned key, unsigned* scratch) {
    const int tid = threadIdx.x;
    for (int span = 2; span <= QSA_SEL_THREADS; span <<= 1) {
        for (int step = span / 2; step; step >>= 1) {
            unsigned peer;
            if (step < 32) peer = __shfl_xor_sync(0xffffffffu, key, step);
            else {
                scratch[tid] = key;
                __syncthreads();
                peer = scratch[tid ^ step];
                __syncthreads();
            }
            const bool low = ((tid & span) == 0) == ((tid & step) == 0);
            key = low ? min(key, peer) : max(key, peer);
        }
    }
    return key;
}

__device__ __forceinline__ unsigned qsa_count_greater(const unsigned* sorted, unsigned cutoff) {
    int lo = 0, hi = QSA_SEL_THREADS;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (sorted[mid] <= cutoff) lo = mid + 1;
        else hi = mid;
    }
    return QSA_SEL_THREADS - lo;
}

// Sampling only picks a pivot. Count all greater/equal scores before using
// it. If the bounded winner buffer cannot certify the cutoff, use full radix.
__device__ __forceinline__ QsaCutoff qsa_sample_cutoff(const unsigned* scores, int visible) {
    if (visible < 4096) return qsa_cutoff(scores, visible);
    const int tid = threadIdx.x, lane = tid & 31;
    __shared__ unsigned scratch[2 * QSA_SEL_THREADS], waves[32];
    __shared__ unsigned pivot, greater, equal, keep_ties;
    const unsigned sample = qsa_sort_key(scores[tid * visible / QSA_SEL_THREADS], scratch);
    const int rank = (2 * QSA_TOP * QSA_SEL_THREADS + visible - 1) / visible;
    if (tid == QSA_SEL_THREADS - rank) pivot = sample;
    if (tid == 0) greater = 0;
    __syncthreads();
    unsigned own_equal = 0;
    for (int base = 0; base < visible; base += QSA_SEL_THREADS) {
        const int j = base + tid;
        const unsigned value = j < visible ? scores[j] : 0u;
        const bool above = j < visible && value > pivot;
        own_equal += j < visible && value == pivot;
        const unsigned votes = __ballot_sync(0xffffffffu, above);
        unsigned start = 0;
        if (lane == 0 && votes) start = atomicAdd(&greater, (unsigned)__popc(votes));
        start = __shfl_sync(0xffffffffu, start, 0);
        const unsigned slot = start + __popc(votes & ((1u << lane) - 1));
        if (above && slot < 2 * QSA_SEL_THREADS) scratch[slot] = value;
    }
    const unsigned before = qsa_block_offset(own_equal, waves);
    if (tid == QSA_SEL_THREADS - 1) equal = before + own_equal;
    __syncthreads();
    if (greater < QSA_TOP && greater + equal >= QSA_TOP) return {pivot, QSA_TOP - greater};
    if (greater < QSA_TOP || greater > 2 * QSA_SEL_THREADS) return qsa_cutoff(scores, visible);

    // All buffered keys are strictly above a nonnegative pivot, so zero
    // padding cannot enter the top 512 of these one or two sorted arrays.
    const bool two = greater > QSA_SEL_THREADS;
    unsigned key = tid < greater ? scratch[tid] : 0u;
    unsigned second = tid + QSA_SEL_THREADS < greater ? scratch[tid + QSA_SEL_THREADS] : 0u;
    __syncthreads();
    key = qsa_sort_key(key, scratch);
    if (two) second = qsa_sort_key(second, scratch + QSA_SEL_THREADS);
    scratch[tid] = key;
    if (two) scratch[tid + QSA_SEL_THREADS] = second;
    __syncthreads();
    if (tid == 0) {
        unsigned cutoff = scratch[QSA_SEL_THREADS - QSA_TOP];
        const unsigned* other = scratch + QSA_SEL_THREADS;
        if (two) {
            // Choose how many of the top 512 come from each sorted half.
            int lo = 0, hi = QSA_TOP;
            while (lo < hi) {
                const int n = (lo + hi) / 2, m = QSA_TOP - n;
                if (scratch[QSA_SEL_THREADS - n - 1] > other[QSA_SEL_THREADS - m]) lo = n + 1;
                else hi = n;
            }
            const int m = QSA_TOP - lo;
            cutoff = min(lo ? scratch[QSA_SEL_THREADS - lo] : ~0u, m ? other[QSA_SEL_THREADS - m] : ~0u);
        }
        pivot = cutoff;
        keep_ties = QSA_TOP - qsa_count_greater(scratch, cutoff) - (two ? qsa_count_greater(other, cutoff) : 0);
    }
    __syncthreads();
    return {pivot, keep_ties};
}

// Both position partitions select from the same global scores. Resolve the
// cutoff and its ties once, then compact the two ordered lists independently.
template<bool Sampled = false, bool Coalesced = false, bool QuerySplit = false>
__global__ void __launch_bounds__(QSA_SEL_THREADS) k_idx_select_pair(const float* __restrict__ scores, int* __restrict__ cells,
                                                                     int* __restrict__ n_cells, int max_pools, const int* pos0p,
                                                                     int stride, LaneRows lr, int card = 0, int group = 4) {
    if constexpr (QuerySplit) {
        if ((blockIdx.x / group) % 2 != card) return;
    }
    const int t = blockIdx.x, tid = threadIdx.x, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = n_cells[stride + t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);
    const QsaCutoff cut = Sampled ? qsa_sample_cutoff(key, nv) : qsa_cutoff(key, nv);
    using Mask = unsigned long long;
    Mask greater = 0, equal = 0;
    int first;
    if constexpr (Coalesced) {
        // Adjacent lanes read adjacent scores. Transpose just the predicates
        // into chronological 64-key masks, keeping the cutoff and tie order.
        __shared__ unsigned above_bits[QSA_PAIR_MAX_POOLS / 32];
        __shared__ unsigned equal_bits[QSA_PAIR_MAX_POOLS / 32];
        const int lane = tid & 31;
        for (int base = 0; base < nv; base += QSA_SEL_THREADS) {
            const int j = base + tid;
            const unsigned k = j < nv ? key[j] : 0;
            const unsigned above = __ballot_sync(0xffffffffu, j < nv && k > cut.bits);
            const unsigned ties = __ballot_sync(0xffffffffu, j < nv && k == cut.bits);
            if (lane == 0 && j < nv) {
                above_bits[j / 32] = above;
                equal_bits[j / 32] = ties;
            }
        }
        __syncthreads();
        first = tid * 64;
        if (first < nv) {
            greater = above_bits[2 * tid];
            equal = equal_bits[2 * tid];
        }
        if (first + 32 < nv) {
            greater |= (Mask) above_bits[2 * tid + 1] << 32;
            equal |= (Mask) equal_bits[2 * tid + 1] << 32;
        }
    } else {
        const int chunk = (nv + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS;
        first = tid * chunk;
        const int length = max(0, min(chunk, nv - first));
        for (int i = 0; i < length; ++i) {
            const unsigned k = key[first + i];
            const Mask bit = 1ull << i;
            if (k > cut.bits) greater |= bit;
            if (k == cut.bits) equal |= bit;
        }
    }
    __shared__ unsigned tie_waves[32], selected_waves[32];
    const int ties = __popcll(equal);
    const int take = (int) cut.ties - (int) qsa_block_offset(ties, tie_waves);
    Mask selected = greater;
    if (take >= ties) selected |= equal;
    else {
        for (int i = 0; i < take; ++i) {
            selected |= equal & -equal;
            equal &= equal - 1;
        }
    }
    constexpr Mask even = 0x5555555555555555ull;
    const int count0 = __popcll(selected & ((first & 1) ? ~even : even));
    // At most QSA_TOP (512) blocks in total, so packed sums cannot carry.
    const unsigned counts = count0 | (unsigned)(__popcll(selected) - count0) << 16;
    const unsigned offsets = qsa_block_offset(counts, selected_waves);
    int out[2] = {(int)(offsets & 0xffffu), (int)(offsets >> 16)};
    while (selected) {
        const int j = first + __ffsll(selected) - 1, owner = kv_card(j);
        int* dest = cells + ((size_t) owner * stride + t) * QSA_MAXSEL + out[owner]++ * QSA_POOL;
#pragma unroll
        for (int i = 0; i < QSA_POOL; ++i) dest[i] = kv_local_block(j) * QSA_POOL + i;
        selected &= selected - 1;
    }
    if (tid == QSA_SEL_THREADS - 1) {
        const int tail = (p + 1) % QSA_POOL;
#pragma unroll
        for (int owner = 0; owner < 2; ++owner) {
            const int row = owner * stride + t;
            int n = out[owner] * QSA_POOL;
            if (kv_card(nv) == owner)
                for (int i = 0; i < tail; ++i) cells[(size_t) row * QSA_MAXSEL + n++] = kv_local_block(nv) * QSA_POOL + i;
            n_cells[row] = n;
        }
    }
}
#endif

}  // namespace

// The routed experts in three launches (included by kernels.cu, after k_mmvq's helpers):
//   router + route: the logits, top-10 per token, their weights, and the job list (one job per distinct expert);
//   gate|up: every job's stacked [gate; up] rows for its tokens;
//   down: each block makes its job's silu(gate) * up rows as q8_1 in shared memory, then the down rows.
// Output rows: pair t*TOPK+k. The shared expert runs beside them as dense launches into rows T*TOPK+t; the weighted
// sum happens in the ring.
namespace {

// The router and the routing in one launch. logits rows of NEXP + 1: the router's, then the shared expert's gate,
// a warp per row. The last block to finish selects each token's top 10 by logit (softmax is monotonic; ties: the lower
// index), a warp per token, the weights the selected softmax renormalized (exp(l - l_max) over the ten), and builds
// the jobs; with an ExpertCache (ec.n_slots) its first warp then plans the layer's slots (cache_plan).
constexpr int RR_THREADS = 256, RR_WPR = 2, RR_ROWS = RR_THREADS / 32 / RR_WPR;   // two warps per logit row

// one warp: the layer's jobs whose experts are in a slot keep it (ref 2: not to be taken); each other one takes the
// slot of the layer's pool that the pool's CLOCK hand reaches first with ref 0 (a ref 1 it passes drops to 0), its old
// expert pointing back at RAM. The hand goes round the pool's slots past its lent ones. list: how many took a slot,
// then their ids
__device__ void cache_plan(const Job* jobs, int nj, int layer, const ExpertCache& ec, int* list) {
    const int lane = threadIdx.x & 31, base = layer * NEXP, p = ec.pool_of[layer], lent = ec.lent[p];
    unsigned missed = 0;
    // an arena's pools hold every expert until it lends, and then only its own pool's experts can be missing
    if (ec.arena_full && (p != ec.arena_pool || lent == 0)) {
        if (ec.miss_bits && lane < (NEXP + 31) / 32) ec.miss_bits[lane] = 0;
        if (lane == 0) list[0] = 0;
        return;
    }
    const int s0 = ec.s0[p] + lent, n = ec.n[p] - lent;
    for (int j = lane; j < nj; j += 32)
        if (const int s = ec.slot_of[base + jobs[j].w]; s >= 0) {
            if (ec.age) ec.age[s] = min(ec.max_age, (int) ec.age[s] + 1);
            ec.ref[s] = 2;
        }
    __syncwarp();
    int hand = ec.hand[p] - s0, k = 0;   // within the slots not lent
    if (hand < 0 || hand >= n) hand = 0;
    // the misses found 32 jobs at a time (a load each, side by side), then taken in job order
    for (int j0 = 0; j0 < nj; j0 += 32) {
        unsigned miss = __ballot_sync(0xffffffffu, j0 + lane < nj && ec.slot_of[base + jobs[j0 + lane].w] == -1);
        while (miss) {
            const int g = base + jobs[j0 + __ffs(miss) - 1].w;
            if (ec.miss_bits && lane == (g - base) / 32) missed |= 1u << ((g - base) % 32);
            miss &= miss - 1;
            int victim = -1;
            while (victim < 0) {
                const int idx = s0 + (hand + lane) % n;
                const unsigned char r = ec.ref[idx];
                const unsigned free = __ballot_sync(0xffffffffu, r == 0);
                const int f = free ? __ffs(free) - 1 : 32;
                if (lane < f && r == 1) {
                    if (ec.age && ec.age[idx] > 1) --ec.age[idx];
                    else {
                        ec.ref[idx] = 0;
                        if (ec.age) ec.age[idx] = 0;
                    }
                }
                __syncwarp();  // the next scan can wrap to another lane's ref/age update
                if (free) victim = s0 + (hand + f) % n;
                hand = (hand + (free ? f + 1 : 32)) % n;
            }
            if (lane == 0) {
                if (const int o = ec.owner[victim]; o >= 0) {
                    ec.slot_of[o] = -1;
                    ec.ep_gu[o] = ec.host_gu[o];
                    ec.ep_dn[o] = ec.host_dn[o];
                }
                ec.owner[victim] = g;
                ec.slot_of[g] = victim;
                ec.ref[victim] = 2;
                if (ec.age) ec.age[victim] = 1;
                char* at = ec.pool[p] + (size_t) (victim - ec.s0[p]) * ec.slot_bytes[p];
                ec.ep_gu[g] = at;
                ec.ep_dn[g] = at + ec.dn_off[p];
                list[1 + k] = g;
            }
            ++k;
            __syncwarp();
        }
    }
    for (int j = lane; j < nj; j += 32)
        if (const int s = ec.slot_of[base + jobs[j].w]; s >= 0) ec.ref[s] = 1;
    if (ec.miss_bits && lane < (NEXP + 31) / 32) ec.miss_bits[lane] = missed;
    if (lane == 0) {
        ec.hand[p] = s0 + hand;
        list[0] = k;
    }
}

template <int T, bool BF16 = false>
__global__ void __launch_bounds__(RR_THREADS) k_router_route(const void* __restrict__ W, const float* __restrict__ x,
                                                            float* __restrict__ logits, int* __restrict__ topi,
                                                            float* __restrict__ topw, Job* __restrict__ gu, Job* __restrict__ dn,
                                                            int* __restrict__ njobs, unsigned* __restrict__ counter,
                                                            const ExpertCache ec, int layer, int* list, const ShexpUp su, unsigned* errors,
                                                            int* __restrict__ wl1 = nullptr, int* __restrict__ wlm = nullptr,
                                                            int* __restrict__ n_wl = nullptr) {
    static_assert(RR_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the router's launch");
    constexpr int NC = N / 4 / 32 / RR_WPR, L = NEXP + 1;   // float4 chunks per lane
    const int nrr = (int) gridDim.x - su.blocks;              // the router's own blocks; the shared expert's after them
    if ((int) blockIdx.x >= nrr) {
        shexp_up_block<T>(su, blockIdx.x - nrr);
        return;
    }
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31, rw = warp / RR_WPR, half = warp % RR_WPR;
    const int row = blockIdx.x * RR_ROWS + rw;
    __shared__ float rsum[T][RR_THREADS / 32];
    if (row < L) {
        const size_t first4 = (size_t) row * (N / 4) + half * NC * 32;
        float4 wv[NC];
#pragma unroll
        for (int i = 0; i < NC; ++i) wv[i] = load_weight4<BF16>(W, first4 + i * 32 + lane);
        // unrolled over the tokens: each pair's x loads and sums run beside the others' (rolled, the pairs waited on
        // each other: one wave of blocks, so latency set the time). Each logit's summation order is unchanged
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const bool two = t + 1 < T;
            const float4* x4 = reinterpret_cast<const float4*>(x + (size_t) t * N) + half * NC * 32;
            const float4* y4 = two ? x4 + N / 4 : x4;
            float a = 0.f, c = 0.f;
#pragma unroll
            for (int i = 0; i < NC; ++i) {
                const float4 b = x4[i * 32 + lane], d = y4[i * 32 + lane];
                a += wv[i].x * b.x + wv[i].y * b.y + wv[i].z * b.z + wv[i].w * b.w;
                c += wv[i].x * d.x + wv[i].y * d.y + wv[i].z * d.z + wv[i].w * d.w;
            }
            a = wsum(a);
            c = wsum(c);
            if (lane == 0) {
                rsum[t][warp] = a;
                if (two) rsum[t + 1][warp] = c;
            }
        }
    }
    __syncthreads();
    for (int q = e; q < T * RR_ROWS; q += RR_THREADS) {
        const int t = q / RR_ROWS, r = q % RR_ROWS, lrow = blockIdx.x * RR_ROWS + r;
        if (lrow < L) logits[(size_t) t * L + lrow] = rsum[t][r * RR_WPR] + rsum[t][r * RR_WPR + 1];
    }
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (e == 0) last = atomicAdd(counter, 1u) == (unsigned) nrr - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    // top 10 per token: warp w takes tokens w, w + 8, ...; lane l holds experts l + 32 i
    __shared__ int ex[MAXT * TOPK];
    static_assert(T * TOPK < RR_THREADS, "a thread per pick and one more (the job count) below");
#pragma unroll 1
    for (int t = warp; t < T; t += RR_THREADS / 32) {
        float v[NEXP / 32];
#pragma unroll
        for (int i = 0; i < NEXP / 32; ++i) v[i] = __ldcg(logits + (size_t) t * L + lane + 32 * i);
        unsigned taken = 0;
        float top0 = 0.f, z = 0.f, wk = 0.f;
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY;
            int bi = NEXP;
#pragma unroll
            for (int i = 0; i < NEXP / 32; ++i)
                if (!(taken >> i & 1u) && v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (bi == NEXP) {
                // No ordered logit remains. Lane zero has more candidates than TOPK.
                bi = 32 * __shfl_sync(0xffffffffu, __ffs(~taken) - 1, 0);
                if (lane == 0 && errors) atomicOr(errors, 32u);
            }
            if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
            if (k == 0) top0 = bv;
            const float ek = __expf(bv - top0);
            z += ek;
            if (lane == k) {
                wk = ek;
                topi[t * TOPK + k] = bi;
                ex[t * TOPK + k] = bi;
            }
        }
        if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
    }
    __syncthreads();
    // jobs: one per distinct expert, numbered in the order of each expert's first pick; a pair takes the next slot of
    // its expert's job (the order of a job's rows does not matter)
    constexpr int P = T * TOPK;
    __shared__ int firstpair[NEXP], slots[NEXP], jobof[NEXP];
    __shared__ unsigned char first[MAXT * TOPK];
    for (int x2 = e; x2 < NEXP; x2 += RR_THREADS) {
        firstpair[x2] = P;
        slots[x2] = 0;
    }
    __syncthreads();
    if (e < P) atomicMin(&firstpair[ex[e]], e);
    __syncthreads();
    if (e < P) first[e] = firstpair[ex[e]] == e;
    __syncthreads();
    if (e < P && first[e]) {
        int j = 0;
        for (int q = 0; q < e; ++q) j += first[q];
        jobof[ex[e]] = j;
        gu[j].w = dn[j].w = ex[e];
    }
    __syncthreads();
    if (e < P) {
        const int j = jobof[ex[e]], s = atomicAdd(&slots[ex[e]], 1);
        gu[j].y[s] = e / TOPK;
        gu[j].out[s] = dn[j].y[s] = dn[j].out[s] = e;
    }
    __syncthreads();
    if (e < P && first[e]) gu[jobof[ex[e]]].n = dn[jobof[ex[e]]].n = slots[ex[e]];
    __shared__ int nj;
    if (e == P) {
        int n = 0;
        for (int q = 0; q < P; ++q) n += first[q];
        njobs[0] = nj = n;
        *counter = 0u;
    }
    if (wl1) {
        static_assert(2 * P <= NEXP, "worklist scan reuses dead firstpair buffer");
        int* is1 = firstpair;
        int* ism = firstpair + P;
        if (e < P) {
            const int c = first[e] ? slots[ex[e]] : 0;
            is1[e] = c == 1 ? 1 : 0;
            ism[e] = c >= 2 ? 1 : 0;
        }
        __syncthreads();
        for (int d = 1; d < P; d <<= 1) {
            int v1 = 0, vm = 0;
            if (e < P && e >= d) {
                v1 = is1[e - d];
                vm = ism[e - d];
            }
            __syncthreads();
            if (e < P && e >= d) {
                is1[e] += v1;
                ism[e] += vm;
            }
            __syncthreads();
        }
        if (e < P && first[e]) {
            const int c = slots[ex[e]];
            const int j = jobof[ex[e]];
            if (c == 1) wl1[is1[e] - 1] = j;
            else wlm[ism[e] - 1] = j;
        }
        if (e == P - 1) {
            n_wl[0] = is1[P - 1];
            n_wl[1] = ism[P - 1];
        }
    }
    if (!ec.n_slots) return;
    __syncthreads();
    if (warp == 0) cache_plan(gu, nj, layer, ec, list);
}

// rows of one job: tmp[j] = W row . y[ys[j]] for j < n <= TT, G lanes per row
template <GgmlType type, int G, int TT>
__device__ __forceinline__ void job_rows(const char* wr, bool live, int ncols, int n, const block_q8_1* y, const int* ys,
                                         int y_stride, float* tmp) {
    constexpr int qk = qtraits<type>::qk, qi = qtraits<type>::qi, vdr = vdr_of<type>();
    constexpr int per_block = qi / vdr;
    static_assert(G >= per_block && G % per_block == 0, "G must cover a block");
    constexpr int bpi = G / per_block;
    const int gl = threadIdx.x % G, bpr = ncols / qk;
    if (live)
#pragma unroll 8
        for (int kbx = gl / per_block; kbx < bpr; kbx += bpi) {
            const int kby = kbx * (qk / QK8_1), kqs = vdr * (gl % per_block);
#pragma unroll
            for (int j = 0; j < TT; ++j)
                if (j < n) tmp[j] += vdot<type>(wr, y + (size_t) ys[j] * y_stride + kby, kbx, kqs);
        }
}

// TT: the most rows a job has in this launch (the window's T): the loop unrolls that far, not to MAXT
template <int G, int TT = MAXT>
__device__ __forceinline__ void job_store(const float* tmp, int n, bool live, int row, const int* outs, float* out, int out_stride) {
    const int gl = threadIdx.x % G;
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        if (j >= n) break;
        float v = tmp[j];
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (live && gl == 0) out[(size_t) outs[j] * out_stride + row] = v;
    }
}

// one thread's dot of U columns of a weight row (its unit `u`) with one q8_1 activation row
template <GgmlType type, int U>
__device__ __forceinline__ float unit_dot(const char* wrow, int u, const block_q8_1* y) {
    constexpr int qk = qtraits<type>::qk, qi = qtraits<type>::qi, vdr = vdr_of<type>();
    float s = 0.f;
    if constexpr (U % qk == 0) {
#pragma unroll
        for (int b = 0; b < U / qk; ++b) {
            const int kbx = u * (U / qk) + b;
#pragma unroll
            for (int i = 0; i < qi / vdr; ++i) s += vdot<type>(wrow, y + kbx * (qk / QK8_1), kbx, vdr * i);
        }
    } else {
        __trap();   // a type whose blocks are wider than the unit: the launcher refuses it
    }
    return s;
}

#include "moe_w.cuh"

// gate|up: rows [0, 2 * ffs) of each job [0, *njobs)
template <GgmlType TA, int G, int TT>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                       const block_q8_1* __restrict__ y, float* __restrict__ out,
                                                       const int* __restrict__ job_idx = nullptr) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji_real];
    __syncthreads();
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
    float tmp[TT] = {};
    job_rows<TA, G, TT>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp);
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

#include "moe_iq_reuse.cuh"
#include "moe_iq_cache.cuh"
#include "moe_q2_down.cuh"
#include "moe_hoist.cuh"

// down: one block per (job, rows_per_block rows). The block first makes its tokens' h = silu(gate) * up (gu rows
// jb.y[j] of [gate (ffs) | up (ffs)]) as q8_1 in shared memory; thread (row, u) then dots its 32-column unit u of the
// row with each token's h, and the block sums a row's units into out row jb.out[j]
constexpr int DN_THREADS = 256, DN_U = 32, DN_MAXB = FF_CARD_MAX / 32 > 8 ? FF_CARD_MAX / 32 : 8;   // DN_MAXB: ffs / 32 on any card

// every pair's h for k_moe_down, made once by k_moe_h (each of the ~70 down blocks of a job made its own). One MoE runs at
// a time on a card (its window's stream), so one buffer per card
__device__ block_q8_1 g_moe_hq[MAXT * TOPK][DN_MAXB];

// h = silu(gate) * up of pair row e of gu ([gate (ffs) | up (ffs)]) as q8_1, a warp per (pair, 32 columns)
__global__ void k_moe_h(const float* __restrict__ gu, int ffs, int npairs) {
    const int nb = ffs / DN_U, q = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (q >= npairs * nb) return;
    const int e = q / nb, b = q % nb;
    const float* g = gu + (size_t) e * 2 * ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
    const float amax = wmax(fabsf(xi)), sm = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& h = g_moe_hq[e][b];
    h.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (lane == 0) h.ds = make_half2(__float2half(d), __float2half(sm));
}

template <GgmlType TA, bool PACKED_Q2, bool HOIST>
__device__ __forceinline__ void moe_down_rows(const char* wr, bool live, int n, int u, const block_q8_1 (*hq)[DN_MAXB],
                                              float (*part)[DN_THREADS]) {
    if (!live) return;
    if constexpr (PACKED_Q2) {
        const Q2Half decoded = q2_load_half(wr, u);
        for (int j = 0; j < n; ++j) part[j][threadIdx.x] = q2_half_dot(decoded, hq[j] + u);
    } else if constexpr (HOIST && DownUnit<TA>::supported) {
        const auto decoded = DownUnit<TA>::load(wr, u);
        for (int j = 0; j < n; ++j) part[j][threadIdx.x] = DownUnit<TA>::dot(decoded, hq[j] + u);
    } else {
        for (int j = 0; j < n; ++j) part[j][threadIdx.x] = unit_dot<TA, DN_U>(wr, u, hq[j]);
    }
}

// PRE: the pairs' h made by k_moe_h (past 8 rows: kbench 16 rows 82.9 -> 70.7 us); else each block makes its own (the
// extra launch costs more than it saves: 5 rows 30.7 -> 31.7)
template <GgmlType TA, bool PRE, bool PACKED_Q2 = false, bool HOIST = false>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                        const float* __restrict__ gu, float* __restrict__ out,
                                                        const int* __restrict__ job_idx = nullptr) {
    static_assert(!PACKED_Q2 || TA == T_Q4_0, "packed Q2 retains Q4_0 arithmetic");
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    __shared__ block_q8_1 hq[MAXT][DN_MAXB];
    extern __shared__ float dn_part[];   // [the window's T][DN_THREADS] (launched with that much): MAXT's would halve the
    float(*part)[DN_THREADS] = reinterpret_cast<float(*)[DN_THREADS]>(dn_part);   // blocks an SM holds
    if (threadIdx.x == 0) jb = jobs[ji_real];
    __syncthreads();
    const int ffs = w.ncols, nb = ffs / DN_U, n = jb.n;
    if constexpr (PRE) {
        static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
        for (int q = threadIdx.x; q < n * nb * 9; q += DN_THREADS) {
            const int jb9 = q / 9, j = jb9 / nb, b = jb9 % nb;
            reinterpret_cast<int*>(&hq[j][b])[q % 9] = reinterpret_cast<const int*>(&g_moe_hq[jb.y[j]][b])[q % 9];
        }
    } else {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(fabsf(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
            if (lane == 0) hq[j][b].ds = make_half2(__float2half(d), __float2half(sm));
        }
    }
    __syncthreads();
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    moe_down_rows<TA, PACKED_Q2, HOIST>(moe_expert(w, jb.w) + (size_t) row * w.row_a, live, n, u, hq, part);
    __syncthreads();
    for (int e = threadIdx.x; e < n * rpb; e += DN_THREADS) {
        const int j = e / rpb, rr = e % rpb, orow = blockIdx.x * rpb + rr;
        if (orow >= w.nrows) continue;
        float s = 0.f;
        for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
        out[(size_t) jb.out[j] * w.nrows + orow] = s;
    }
}

template <GgmlType TA, bool PRE, bool PACKED_Q2>
__device__ __forceinline__ void moe_down1_body(MoeW w, int jb_w, int jb_y, int jb_out, const float* __restrict__ gu,
                                               float* __restrict__ out, block_q8_1* hq, float* part) {
    const int ffs = w.ncols, nb = ffs / DN_U;
    if constexpr (PRE) {
        static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
        for (int q = threadIdx.x; q < nb * 9; q += DN_THREADS) {
            const int b = q / 9;
            reinterpret_cast<int*>(&hq[b])[q % 9] = reinterpret_cast<const int*>(&g_moe_hq[jb_y][b])[q % 9];
        }
    } else {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int b = warp; b < nb; b += DN_THREADS / 32) {
            const float* g = gu + (size_t) jb_y * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(fabsf(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[b].qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
            if (lane == 0) hq[b].ds = make_half2(__float2half(d), __float2half(sm));
        }
    }
    __syncthreads();
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    if (r < rpb && row < w.nrows) {
        const char* wr = moe_expert(w, jb_w) + (size_t) row * w.row_a;
        if constexpr (PACKED_Q2) part[threadIdx.x] = unit_dot_q2_exact(wr, u, hq);
        else part[threadIdx.x] = unit_dot<TA, DN_U>(wr, u, hq);
    }
    __syncthreads();
    if (threadIdx.x < rpb) {
        const int rr = threadIdx.x, orow = blockIdx.x * rpb + rr;
        if (orow < w.nrows) {
            float s = 0.f;
            for (int k = 0; k < nb; ++k) s += part[rr * nb + k];
            out[(size_t) jb_out * w.nrows + orow] = s;
        }
    }
}

template <GgmlType TA, bool PRE, bool PACKED_Q2 = false>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down1(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                         const float* __restrict__ gu, float* __restrict__ out,
                                                         const int* __restrict__ job_idx = nullptr) {
    static_assert(!PACKED_Q2 || TA == T_Q4_0, "packed Q2 retains Q4_0 arithmetic");
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ int jb_w, jb_y, jb_out;
    __shared__ block_q8_1 hq[DN_MAXB];
    __shared__ float part[DN_THREADS];
    if (threadIdx.x == 0) {
        const Job& jb = jobs[ji_real];
        jb_w = jb.w;
        jb_y = jb.y[0];
        jb_out = jb.out[0];
    }
    __syncthreads();
    moe_down1_body<TA, PRE, PACKED_Q2>(w, jb_w, jb_y, jb_out, gu, out, hq, part);
}

// FNX_SH_FUSE_IQ: the shared expert inside the routed down launch for the IQ cache layers (whose gate|up launch cannot
// carry it). k_moe_h_sh makes the pairs' h as k_moe_h does and, after them, the shared expert's swiglu as
// k_swiglu_stacked does (q8_1 and its split twin); k_moe_down_sh runs the shared down rows as q8s blocks (one row a
// warp, as q8s_gemv runs 2560 rows) in its first sd_rows rows of blocks and k_moe_down<TA, true>'s jobs after them.
// Every value is made by the same expressions in the same order, so the same bits.
struct ShexpDownHq {
    Q8Segs w;
    int rows = 0, ffs = 0, nqb = 0;
    const block_q8_1* y = nullptr;
    const int8_t* tq = nullptr;
    const float* td = nullptr;
};

__global__ void k_moe_h_sh(const float* __restrict__ gu, int ffs, int npairs, const float* __restrict__ sh_gu, int sh_ffs,
                           block_q8_1* __restrict__ sh_hq, int8_t* __restrict__ sh_tq, float* __restrict__ sh_td, int T) {
    const int nb = ffs / DN_U, q = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (q < npairs * nb) {
        const int e = q / nb, b = q % nb;
        const float* g = gu + (size_t) e * 2 * ffs;
        const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
        const float amax = wmax(fabsf(xi)), sm = wsum(xi);
        const float d = amax / 127.0f;
        block_q8_1& h = g_moe_hq[e][b];
        h.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        if (lane == 0) h.ds = make_half2(__float2half(d), __float2half(sm));
        return;
    }
    const int snb = sh_ffs / 32, r = q - npairs * nb;
    if (r >= T * snb) return;
    const int t = r / snb, b = r % snb;
    const float* g = sh_gu + (size_t) t * 2 * sh_ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[sh_ffs + b * 32 + lane];
    const float amax = wmax(fabsf(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& y = sh_hq[(size_t) t * snb + b];
    const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    y.qs[lane] = qv;
    if (lane == 0) y.ds = make_half2(__float2half(d), __float2half(sum));
    if (sh_tq) {
        sh_tq[(size_t) t * sh_ffs + b * 32 + lane] = qv;
        if (lane == 0) sh_td[(size_t) t * snb + b] = __half2float(__float2half(d));
    }
}

template <GgmlType TA, bool PACKED_Q2, int TT, bool HOIST = false>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down_sh(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                           float* __restrict__ out, const ShexpDownHq sd, int sd_rows,
                                                           const int* __restrict__ job_idx = nullptr) {
    static_assert(DN_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the down launch");
    static_assert(!PACKED_Q2 || TA == T_Q4_0, "packed Q2 retains Q4_0 arithmetic");
    if ((int) blockIdx.y < sd_rows) {
        const int bx = blockIdx.y * gridDim.x + blockIdx.x;
        if (bx < sd.nqb) {
            if (sd.tq) q8s_block<TT, 1, true, false>(sd.w, sd.rows, sd.ffs, sd.y, sd.tq, sd.td, bx);
            else q8s_block<TT, 1, false, false>(sd.w, sd.rows, sd.ffs, sd.y, nullptr, nullptr, bx);
        }
        return;
    }
    const int ji = blockIdx.y - sd_rows;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    __shared__ block_q8_1 hq[MAXT][DN_MAXB];
    extern __shared__ float dn_part[];
    float(*part)[DN_THREADS] = reinterpret_cast<float(*)[DN_THREADS]>(dn_part);
    if (threadIdx.x == 0) jb = jobs[ji_real];
    __syncthreads();
    const int ffs = w.ncols, nb = ffs / DN_U, n = jb.n;
    static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
    for (int q = threadIdx.x; q < n * nb * 9; q += DN_THREADS) {
        const int jb9 = q / 9, j = jb9 / nb, b = jb9 % nb;
        reinterpret_cast<int*>(&hq[j][b])[q % 9] = reinterpret_cast<const int*>(&g_moe_hq[jb.y[j]][b])[q % 9];
    }
    __syncthreads();
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    moe_down_rows<TA, PACKED_Q2, HOIST>(moe_expert(w, jb.w) + (size_t) row * w.row_a, live, n, u, hq, part);
    __syncthreads();
    for (int e = threadIdx.x; e < n * rpb; e += DN_THREADS) {
        const int j = e / rpb, rr = e % rpb, orow = blockIdx.x * rpb + rr;
        if (orow >= w.nrows) continue;
        float s = 0.f;
        for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
        out[(size_t) jb.out[j] * w.nrows + orow] = s;
    }
}

template <GgmlType TA, bool PACKED_Q2, int TT>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down1_sh(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                            float* __restrict__ out, const ShexpDownHq sd, int sd_rows,
                                                            const int* __restrict__ job_idx = nullptr) {
    static_assert(DN_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the down launch");
    static_assert(!PACKED_Q2 || TA == T_Q4_0, "packed Q2 retains Q4_0 arithmetic");
    if ((int) blockIdx.y < sd_rows) {
        const int bx = blockIdx.y * gridDim.x + blockIdx.x;
        if (bx < sd.nqb) {
            if (sd.tq) q8s_block<TT, 1, true, false>(sd.w, sd.rows, sd.ffs, sd.y, sd.tq, sd.td, bx);
            else q8s_block<TT, 1, false, false>(sd.w, sd.rows, sd.ffs, sd.y, nullptr, nullptr, bx);
        }
        return;
    }
    const int ji = blockIdx.y - sd_rows;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ int jb_w, jb_y, jb_out;
    __shared__ block_q8_1 hq[DN_MAXB];
    __shared__ float part[DN_THREADS];
    if (threadIdx.x == 0) {
        const Job& jb = jobs[ji_real];
        jb_w = jb.w;
        jb_y = jb.y[0];
        jb_out = jb.out[0];
    }
    __syncthreads();
    moe_down1_body<TA, true, PACKED_Q2>(w, jb_w, jb_y, jb_out, nullptr, out, hq, part);
}

template <GgmlType type> constexpr int per_block_of() { return qtraits<type>::qi / vdr_of<type>(); }
template <GgmlType type> constexpr bool unit_ok(int U) { return U % qtraits<type>::qk == 0; }

// lanes per row: enough to keep every block of a row busy (as mmvq picks them)
template <GgmlType TA> int moe_g(int ncols) {
    constexpr int pa = per_block_of<TA>();
    const int lanes = (ncols / qtraits<TA>::qk) * pa;
    int G = pa;
    while (G < lanes && G < 32) G *= 2;
    return G;
}

// k_moe_gu for BigJobs: the job's pairs MAXT at a time through the row
template <GgmlType TA, int G>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                           const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    load_bigjob(jb, jobs + ji);
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int nb = min(MAXT, jb.n - b0);
        float tmp[MAXT] = {};
        job_rows<TA, G, MAXT>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, nb, y, jb.y + b0, w.ncols / 32, tmp);
        job_store<G>(tmp, nb, live, row, jb.out + b0, out, w.nrows);
    }
}

template <bool On, typename T> struct MoeHoistSlot { struct Type {}; };
template <typename T> struct MoeHoistSlot<true, T> { using Type = T; };

// k_moe_down for BigJobs: per MAXT pairs, their h into shared memory, the dots, the sums
template <GgmlType TA, bool PACKED_Q2 = false, bool HOIST = false>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                            const float* __restrict__ gu, float* __restrict__ out) {
    static_assert(!PACKED_Q2 || TA == T_Q4_0, "packed Q2 retains Q4_0 arithmetic");
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    __shared__ block_q8_1 hq[MAXT][DN_MAXB];
    __shared__ float part[MAXT][DN_THREADS];
    load_bigjob(jb, jobs + ji);
    const int ffs = w.ncols, nb = ffs / DN_U, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    const char* wr = moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
    Q2Half decoded{};
    if constexpr (PACKED_Q2) {
        if (live) decoded = q2_load_half(wr, u);
    }
    typename MoeHoistSlot<HOIST && !PACKED_Q2 && DownUnit<TA>::supported, DownUnit<TA>>::Type decoded_du{};
    if constexpr (HOIST && !PACKED_Q2 && DownUnit<TA>::supported) {
        if (live) decoded_du = DownUnit<TA>::load(wr, u);
    }
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int n = min(MAXT, jb.n - b0);
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[b0 + j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(fabsf(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
            if (lane == 0) hq[j][b].ds = make_half2(__float2half(d), __float2half(sm));
        }
        __syncthreads();
        if (live) {
            if constexpr (PACKED_Q2) {
                for (int j = 0; j < n; ++j) part[j][threadIdx.x] = q2_half_dot(decoded, hq[j] + u);
            } else if constexpr (HOIST && DownUnit<TA>::supported) {
                for (int j = 0; j < n; ++j) part[j][threadIdx.x] = DownUnit<TA>::dot(decoded_du, hq[j] + u);
            } else {
                for (int j = 0; j < n; ++j) part[j][threadIdx.x] = unit_dot<TA, DN_U>(wr, u, hq[j]);
            }
        }
        __syncthreads();
        for (int e = threadIdx.x; e < n * rpb; e += DN_THREADS) {
            const int j = e / rpb, rr = e % rpb, orow = blockIdx.x * rpb + rr;
            if (orow >= w.nrows) continue;
            float s = 0.f;
            for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
            out[(size_t) jb.out[b0 + j] * w.nrows + orow] = s;
        }
        __syncthreads();
    }
}

template <GgmlType TA, bool PACKED_Q2 = false>
void moe_launch_big(bool down, const MoeW& w, const BigJob* jobs, const int* njobs, int njobs_cap, const void* in, float* out,
                    cudaStream_t s) {
    if constexpr (PACKED_Q2) {
        if (!down || w.ncols % 64 || w.row_a != (size_t) w.ncols / 64 * 18)
            throw std::runtime_error("packed Q2_0 requires down rows of whole 64-value blocks");
    }
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U);
        if (!PACKED_Q2 && DownUnit<TA>::supported && moe_hoist_enabled()) {
            k_moe_down_big<TA, PACKED_Q2, true><<<dim3((w.nrows + rpb - 1) / rpb, njobs_cap), DN_THREADS, 0, s>>>(
                w, jobs, njobs, (const float*) in, out);
        } else {
            k_moe_down_big<TA, PACKED_Q2, false><<<dim3((w.nrows + rpb - 1) / rpb, njobs_cap), DN_THREADS, 0, s>>>(
                w, jobs, njobs, (const float*) in, out);
        }
        return;
    }
    const int G = moe_g<TA>(w.ncols);
#define FNX_MB(g)                                                                                                        \
    {                                                                                                                    \
        const dim3 grid((w.nrows + MMVQ_THREADS / g - 1) / (MMVQ_THREADS / g), njobs_cap);                              \
        k_moe_gu_big<TA, g><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out);                 \
        return;                                                                                                          \
    }
    if constexpr (per_block_of<TA>() <= 4) { if (G == 4) FNX_MB(4) }
    if constexpr (per_block_of<TA>() <= 8) { if (G == 8) FNX_MB(8) }
    if constexpr (per_block_of<TA>() <= 16) { if (G == 16) FNX_MB(16) }
    FNX_MB(32)
#undef FNX_MB
}

// the gate|up kernel for a window of T rows (a job holds at most T)
template <GgmlType TA, int G>
void gu_launch(int T, dim3 grid, cudaStream_t s, const MoeW& w, const Job* jobs, const int* njobs, const block_q8_1* y, float* out,
               const int* job_idx = nullptr) {
    switch (T) {
        case 1: k_moe_gu<TA, G, 1><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 2: k_moe_gu<TA, G, 2><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 3: k_moe_gu<TA, G, 3><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 4: k_moe_gu<TA, G, 4><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 5: k_moe_gu<TA, G, 5><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 6: k_moe_gu<TA, G, 6><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 7: k_moe_gu<TA, G, 7><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 8: k_moe_gu<TA, G, 8><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 9: k_moe_gu<TA, G, 9><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 10: k_moe_gu<TA, G, 10><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 11: k_moe_gu<TA, G, 11><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 12: k_moe_gu<TA, G, 12><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 13: k_moe_gu<TA, G, 13><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 14: k_moe_gu<TA, G, 14><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 15: k_moe_gu<TA, G, 15><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        case 16: k_moe_gu<TA, G, 16><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out, job_idx); break;
        default: throw std::runtime_error("moe gate|up: T out of range");
    }
}

// gate|up with the shared expert's swiglu and down as extra rows of blocks (the other expert types' version of k_moe_gu_q4k's sd rows): 256-thread blocks (the shared
// expert's q8s blocks are Q8S_THREADS wide), G lanes a row as k_moe_gu has them, so every row sums its terms in the same order and the answers are the same bits
template <GgmlType TA, int G, int TT, bool HOIST = false>
__global__ void __launch_bounds__(Q8S_THREADS) k_moe_gu_sd(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                         const block_q8_1* __restrict__ y, float* __restrict__ out, const ShexpDown sd, int sd_rows,
                                                         const int* __restrict__ job_idx = nullptr) {
    if ((int) blockIdx.y < sd_rows) {
        shexp_down_blocks<TT>(sd, blockIdx.y * gridDim.x + blockIdx.x, sd_rows * gridDim.x);
        return;
    }
    const int ji = blockIdx.y - sd_rows;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji_real];
    __syncthreads();
    const int row = blockIdx.x * (Q8S_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
    float tmp[TT] = {};
    if constexpr (HOIST && (TA == T_IQ4_XS || TA == T_Q5_K)) {
        moe_gu_hoist_rows<TA, G, TT>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp);
    } else {
        job_rows<TA, G, TT>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp);
    }
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

template <GgmlType TA>
void gu_sd_launch(int T, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const block_q8_1* y, float* out, cudaStream_t s, const ShexpDown& sd,
                  const int* job_idx = nullptr) {
    if (moe_g<TA>(w.ncols) != 32) throw std::runtime_error("moe gate|up: only 32 lanes a row are compiled (rows of N columns)");
    const int gx = (w.nrows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32), sd_blocks = (sd.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    const int sd_rows = sd.rows ? (sd_blocks + gx - 1) / gx : 0;
    const dim3 grid(gx, njobs_max + sd_rows);
    if constexpr (TA == T_IQ4_XS || TA == T_Q5_K) {
        if (T >= 2 && moe_hoist_enabled()) {
#define FNX_GSH(t) case t: k_moe_gu_sd<TA, 32, t, true><<<grid, Q8S_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows, job_idx); break;
            switch (T) {
                FNX_GSH(2) FNX_GSH(3) FNX_GSH(4) FNX_GSH(5) FNX_GSH(6) FNX_GSH(7) FNX_GSH(8)
                FNX_GSH(9) FNX_GSH(10) FNX_GSH(11) FNX_GSH(12) FNX_GSH(13) FNX_GSH(14) FNX_GSH(15) FNX_GSH(16)
                default: throw std::runtime_error("moe gate|up (shared expert): T out of range");
            }
#undef FNX_GSH
            return;
        }
    }
#define FNX_GS(t) case t: k_moe_gu_sd<TA, 32, t><<<grid, Q8S_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows, job_idx); break;
    switch (T) {
        FNX_GS(1) FNX_GS(2) FNX_GS(3) FNX_GS(4) FNX_GS(5) FNX_GS(6) FNX_GS(7) FNX_GS(8)
        FNX_GS(9) FNX_GS(10) FNX_GS(11) FNX_GS(12) FNX_GS(13) FNX_GS(14) FNX_GS(15) FNX_GS(16)
        default: throw std::runtime_error("moe gate|up (shared expert): T out of range");
    }
#undef FNX_GS
}

inline int moe_pre_rows_env() {
    static const int pre_rows = [] {
        const long long v = env_int("FNX_MOE_PRE_ROWS", 9);
        if (v < 1 || v > MAXT + 1) throw std::runtime_error("FNX_MOE_PRE_ROWS: expected 1..17");
        return (int) v;
    }();
    return pre_rows;
}

template <GgmlType TA, bool PACKED_Q2, bool PRE, bool HOIST>
void moe_down_dispatch(int gx, int njobs_max, size_t smem, cudaStream_t s, const MoeW& w, const Job* jobs,
                       const int* njobs, const float* in, float* out, const int* job_idx, const int* wl1,
                       const int* wlm, const int* n_wl) {
    if (wl1) {
        k_moe_down1<TA, PRE, PACKED_Q2><<<dim3(gx, njobs_max), DN_THREADS, 0, s>>>(w, jobs, n_wl, in, out, wl1);
        k_moe_down<TA, PRE, PACKED_Q2, HOIST><<<dim3(gx, njobs_max > 1 ? njobs_max / 2 : 1), DN_THREADS, smem, s>>>(
            w, jobs, n_wl + 1, in, out, wlm);
        return;
    }
    k_moe_down<TA, PRE, PACKED_Q2, HOIST><<<dim3(gx, njobs_max), DN_THREADS, smem, s>>>(w, jobs, njobs, in, out, job_idx);
}

template <GgmlType TA, bool PACKED_Q2 = false>
void moe_launch(bool down, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const void* in, float* out,
                cudaStream_t s, int T, const CacheDownCopy* copy = nullptr, const CacheGateUpCopy* gu_copy = nullptr,
                const int* job_idx = nullptr, const int* wl1 = nullptr, const int* wlm = nullptr, const int* n_wl = nullptr,
                bool hq_ready = false) {
    if constexpr (PACKED_Q2) {
        if (!down || w.ncols % 64 || w.row_a != (size_t) w.ncols / 64 * 18)
            throw std::runtime_error("packed Q2_0 requires down rows of whole 64-value blocks");
    }
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U), nh = njobs_max * (w.ncols / DN_U);   // njobs_max: the window's pairs
        const int gx = (w.nrows + rpb - 1) / rpb;
        const size_t smem = sizeof(float) * T * DN_THREADS;
        const bool pre = hq_ready || (T >= moe_pre_rows_env());
        if (pre && !hq_ready) {
            if (njobs_max > MAXT * TOPK) throw std::runtime_error("moe_down: more pairs than g_moe_hq holds");
            k_moe_h<<<(nh + 7) / 8, 256, 0, s>>>((const float*) in, w.ncols, njobs_max);
        }
        const bool hoist = !PACKED_Q2 && DownUnit<TA>::supported && moe_hoist_enabled() && T >= 2;
        if constexpr (!PACKED_Q2 && DownUnit<TA>::supported) {
            if (hoist) {
                if (pre) moe_down_dispatch<TA, PACKED_Q2, true, true>(gx, njobs_max, smem, s, w, jobs, njobs, (const float*) in, out, job_idx, wl1, wlm, n_wl);
                else     moe_down_dispatch<TA, PACKED_Q2, false, true>(gx, njobs_max, smem, s, w, jobs, njobs, (const float*) in, out, job_idx, wl1, wlm, n_wl);
                return;
            }
        }
        if (pre) moe_down_dispatch<TA, PACKED_Q2, true, false>(gx, njobs_max, smem, s, w, jobs, njobs, (const float*) in, out, job_idx, wl1, wlm, n_wl);
        else     moe_down_dispatch<TA, PACKED_Q2, false, false>(gx, njobs_max, smem, s, w, jobs, njobs, (const float*) in, out, job_idx, wl1, wlm, n_wl);
        return;
    }
    // gate|up rows have N columns, where every type's lanes per row come to 32: only that variant is compiled (the
    // others, never used, were 192 of the file's kernels). Q4_K's gate|up has a kernel of its own (moe_q4k.cuh)
    if constexpr (TA == T_Q4_K) {
        throw std::runtime_error("moe gate|up: Q4_K runs gu_q4k_launch");
    } else {
        if (moe_g<TA>(w.ncols) != 32) throw std::runtime_error("moe gate|up: only 32 lanes a row are compiled (rows of N columns)");
        const dim3 grid((w.nrows + MMVQ_THREADS / 32 - 1) / (MMVQ_THREADS / 32), njobs_max);
        if constexpr (TA == T_IQ2_XXS || TA == T_IQ2_S || TA == T_IQ3_XXS || TA == T_IQ3_S) {
            static const long long reuse = [] {
                const long long value = env_int("FNX_IQ_REUSE", 0);
                if constexpr (TA == T_IQ2_XXS) {
                    const long long xxs = env_int("FNX_IQ2_XXS_REUSE", 0);
                    if (xxs != 0 && xxs != 1) throw std::runtime_error("FNX_IQ2_XXS_REUSE: expected 0 or 1");
                    return xxs ? value : 0LL;
                }
                return value;
            }();
            if (copy && !reuse) throw std::runtime_error("down copy requires FNX_IQ_REUSE=1");
            if (reuse != 0 && reuse != 1) throw std::runtime_error("FNX_IQ_REUSE: expected 0 or 1");
            if (reuse) {
                static const long long group = env_int("FNX_IQ_GROUP", 32);
                static const long long fast_signs = env_int("FNX_IQ_FAST_SIGNS", 0);
                if (fast_signs != 0 && fast_signs != 1)
                    throw std::runtime_error("FNX_IQ_FAST_SIGNS: expected 0 or 1");
                if (group != 16 && group != 32) throw std::runtime_error("FNX_IQ_GROUP: expected 16 or 32");
                const dim3 grid16((w.nrows + 7) / 8, njobs_max);
                if (copy && (group != 16 || copy->blocks < 1 || copy->blocks > 256 ||
                             !copy->host || !copy->slots || !copy->list || !copy->pool ||
                             !copy->bytes || copy->bytes % 16 || copy->offset % 16 || copy->slot_bytes % 16 ||
                             copy->offset > copy->slot_bytes || copy->bytes > copy->slot_bytes - copy->offset))
                    throw std::runtime_error("invalid IQ down copy descriptor/group");
                const dim3 fused_grid(grid16.x * grid16.y + (copy ? copy->blocks : 0));
                if (gu_copy && (!gu_copy->host || (!copy && !job_idx) || (copy && gu_copy->list != copy->list) ||
                                group != 16 || gu_copy->layer < 0 || gu_copy->layer >= NL || !w.ep || w.ncols != N ||
                                w.row_a != (size_t) N / qtraits<TA>::qk * qtraits<TA>::bs))
                    throw std::runtime_error("invalid IQ gate/up copy descriptor/shape");
#define FNX_IQR(t) case t: \
    if (gu_copy && copy && fast_signs) k_moe_gu_iq_cache<TA, t, 16, true, true><<<fused_grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, *copy, *gu_copy, job_idx); \
    else if (gu_copy && copy) k_moe_gu_iq_cache<TA, t, 16, true><<<fused_grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, *copy, *gu_copy, job_idx); \
    else if (gu_copy && fast_signs) k_moe_gu_iq_cache<TA, t, 16, false, true><<<grid16, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, *gu_copy, job_idx); \
    else if (gu_copy) k_moe_gu_iq_cache<TA, t, 16><<<grid16, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, *gu_copy, job_idx); \
    else if (copy && fast_signs) k_moe_gu_iq<TA, t, 16, true, true><<<fused_grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, *copy, job_idx); \
    else if (copy) k_moe_gu_iq<TA, t, 16, true><<<fused_grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, *copy, job_idx); \
    else if (group == 16 && fast_signs) k_moe_gu_iq<TA, t, 16, false, true><<<grid16, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, job_idx); \
    else if (group == 16) k_moe_gu_iq<TA, t, 16><<<grid16, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, job_idx); \
    else if (fast_signs) k_moe_gu_iq<TA, t, 32, false, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, job_idx); \
    else k_moe_gu_iq<TA, t><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, CacheDownCopy{}, job_idx); \
    break
                switch (T) {
                    FNX_IQR(1); FNX_IQR(2); FNX_IQR(3); FNX_IQR(4);
                    FNX_IQR(5); FNX_IQR(6); FNX_IQR(7); FNX_IQR(8);
                    FNX_IQR(9); FNX_IQR(10); FNX_IQR(11); FNX_IQR(12);
                    FNX_IQR(13); FNX_IQR(14); FNX_IQR(15); FNX_IQR(16);
                    default: throw std::runtime_error("moe gate|up: T out of range");
                }
#undef FNX_IQR
                return;
            }
        }
        if constexpr (TA == T_IQ4_XS || TA == T_Q5_K) {
            if (T >= 2 && moe_hoist_enabled()) {
#define FNX_GH(t) case t: k_moe_gu_hoist<TA, 32, t><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out, job_idx); break;
                switch (T) {
                    FNX_GH(2) FNX_GH(3) FNX_GH(4) FNX_GH(5) FNX_GH(6) FNX_GH(7) FNX_GH(8)
                    FNX_GH(9) FNX_GH(10) FNX_GH(11) FNX_GH(12) FNX_GH(13) FNX_GH(14) FNX_GH(15) FNX_GH(16)
                    default: throw std::runtime_error("moe gate|up (hoist): T out of range");
                }
#undef FNX_GH
                return;
            }
        }
        gu_launch<TA, 32>(T, grid, s, w, jobs, njobs, (const block_q8_1*) in, out, job_idx);
    }
}

// FNX_SH_FUSE_IQ: the routed down (k_moe_h's h always made first, as FNX_MOE_PRE_ROWS <= T has it) with the shared
// expert's swiglu in the same h launch and its down rows in the down launch
template <GgmlType TA, bool PACKED_Q2, bool HOIST>
void moe_down_sh_dispatch(int gx, int njobs_max, int sd_rows, size_t smem, cudaStream_t s, int T,
                          const MoeW& w, const Job* jobs, const int* njobs, float* out, const ShexpDownHq& sd,
                          const int* wl1, const int* wlm, const int* n_wl) {
    if (wl1) {
        const dim3 grid1(gx, njobs_max + sd_rows);
#define FNX_DSH1(t) case t: k_moe_down1_sh<TA, PACKED_Q2, t><<<grid1, DN_THREADS, 0, s>>>(w, jobs, n_wl, out, sd, sd_rows, wl1); break;
        switch (T) {
            FNX_DSH1(1) FNX_DSH1(2) FNX_DSH1(3) FNX_DSH1(4) FNX_DSH1(5) FNX_DSH1(6) FNX_DSH1(7) FNX_DSH1(8)
            FNX_DSH1(9) FNX_DSH1(10) FNX_DSH1(11) FNX_DSH1(12) FNX_DSH1(13) FNX_DSH1(14) FNX_DSH1(15) FNX_DSH1(16)
            default: throw std::runtime_error("moe_down (shared expert): T out of range");
        }
#undef FNX_DSH1
        k_moe_down<TA, true, PACKED_Q2, HOIST><<<dim3(gx, njobs_max > 1 ? njobs_max / 2 : 1), DN_THREADS, smem, s>>>(
            w, jobs, n_wl + 1, nullptr, out, wlm);
        return;
    }
    const dim3 grid(gx, njobs_max + sd_rows);
#define FNX_DSH(t) case t: k_moe_down_sh<TA, PACKED_Q2, t, HOIST><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, out, sd, sd_rows); break;
    switch (T) {
        FNX_DSH(1) FNX_DSH(2) FNX_DSH(3) FNX_DSH(4) FNX_DSH(5) FNX_DSH(6) FNX_DSH(7) FNX_DSH(8)
        FNX_DSH(9) FNX_DSH(10) FNX_DSH(11) FNX_DSH(12) FNX_DSH(13) FNX_DSH(14) FNX_DSH(15) FNX_DSH(16)
        default: throw std::runtime_error("moe_down (shared expert): T out of range");
    }
#undef FNX_DSH
}

template <GgmlType TA, bool PACKED_Q2 = false>
void moe_down_sh_launch(const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const float* gu, float* out,
                        cudaStream_t s, int T, const ShexpDownHq& sd, const float* sh_gu, block_q8_1* sh_hq,
                        const int* wl1 = nullptr, const int* wlm = nullptr, const int* n_wl = nullptr, bool hq_ready = false) {
    if constexpr (!unit_ok<TA>(DN_U)) {
        throw std::runtime_error("moe_down (shared expert): shape or type does not fit the units");
    } else {
        if (w.ncols / DN_U > DN_MAXB || w.ncols % DN_U) throw std::runtime_error("moe_down (shared expert): shape does not fit the units");
        if (njobs_max > MAXT * TOPK) throw std::runtime_error("moe_down (shared expert): more pairs than g_moe_hq holds");
        if (T < 1 || T > MAXT || sd.ffs % 32 || sd.ffs > SH_MAXF) throw std::runtime_error("moe_down (shared expert): rows or FF slice");
        const int nb = w.ncols / DN_U, rpb = DN_THREADS / nb, gx = (w.nrows + rpb - 1) / rpb;
        const int nh = njobs_max * nb + T * (sd.ffs / 32);
        if (!hq_ready) {
            k_moe_h_sh<<<(nh + 7) / 8, 256, 0, s>>>(gu, w.ncols, njobs_max, sh_gu, sd.ffs, sh_hq, const_cast<int8_t*>(sd.tq),
                                                   const_cast<float*>(sd.td), T);
        }
        const int sd_rows = (sd.nqb + gx - 1) / gx;
        const size_t smem = sizeof(float) * T * DN_THREADS;
        const bool hoist = !PACKED_Q2 && DownUnit<TA>::supported && moe_hoist_enabled() && T >= 2;
        if constexpr (!PACKED_Q2 && DownUnit<TA>::supported) {
            if (hoist) {
                moe_down_sh_dispatch<TA, PACKED_Q2, true>(gx, njobs_max, sd_rows, smem, s, T, w, jobs, njobs, out, sd, wl1, wlm, n_wl);
                return;
            }
        }
        moe_down_sh_dispatch<TA, PACKED_Q2, false>(gx, njobs_max, sd_rows, smem, s, T, w, jobs, njobs, out, sd, wl1, wlm, n_wl);
    }
}

}  // namespace

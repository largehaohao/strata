// Fused decode MoE gate|up + SwiGLU + q8_1 kernel (EXP_MODE=8) writing directly into g_moe_hq,
// with opt-in LDS staging modes:
//   FNX_MOE_EXP8=0|1 (default 0): enable fused kernel and force k_moe_down<PRE=true> for T=1..MAXT
//   FNX_MOE_LDS=0|1|2 (default 0):
//     0 = no x_q8_1 or IQ grid LDS staging (fusion-only ablation)
//     1 = stage IQ grid codebook + chunked x_q8_1 (min(T, 4) rows) in LDS (<= 20.4 KiB / block)
//     2 = stage IQ grid codebook only in LDS (<= 9.4 KiB / block)
// Preserves 0-ULP bitwise identity with k_moe_gu<32>/k_moe_gu_q4k + k_moe_h across all 9 formats.
#pragma once

inline bool moe_exp8_enabled() {
    static const bool on = [] {
        const long long v = env_int("FNX_MOE_EXP8", 0);
        if (v != 0 && v != 1) throw std::runtime_error("FNX_MOE_EXP8: expected 0 or 1");
        return v == 1;
    }();
    return on;
}

inline int moe_lds_mode() {
    static const int mode = [] {
        const long long v = env_int("FNX_MOE_LDS", 0);
        if (v < 0 || v > 2) throw std::runtime_error("FNX_MOE_LDS: expected 0, 1, or 2");
        if (v > 0 && !moe_exp8_enabled()) throw std::runtime_error("FNX_MOE_LDS requires FNX_MOE_EXP8=1");
        return (int) v;
    }();
    return mode;
}

inline bool moe_gu_can_exp8_type(uint32_t t) {
    switch (t) {
        case T_IQ2_XXS:
        case T_IQ2_S:
        case T_IQ3_XXS:
        case T_IQ3_S:
        case T_IQ4_XS:
        case T_Q4_K:
        case T_Q5_K:
        case T_Q6_K:
        case T_Q8_0:
            return true;
        default:
            return false;
    }
}

template <GgmlType TA>
constexpr bool exp8_is_iq4() {
    return TA == T_IQ2_XXS || TA == T_IQ2_S || TA == T_IQ3_XXS || TA == T_IQ3_S;
}

// IQ codebook LDS layout (uint32_t words for aligned bank-conflict-free shared reads).
template <GgmlType TA>
struct Exp8Grid {
    static constexpr bool has_grid = false;
    static constexpr int words = 1;
    __device__ __forceinline__ static const uint32_t* src() { return nullptr; }
};

template <>
struct Exp8Grid<T_IQ2_XXS> {
    static constexpr bool has_grid = true;
    static constexpr int words = 512;   // 256 * uint64_t = 2048 B
    __device__ __forceinline__ static const uint32_t* src() { return reinterpret_cast<const uint32_t*>(iq2xxs_grid); }
};

template <>
struct Exp8Grid<T_IQ2_S> {
    static constexpr bool has_grid = true;
    static constexpr int words = 2048;  // 1024 * uint64_t = 8192 B
    __device__ __forceinline__ static const uint32_t* src() { return reinterpret_cast<const uint32_t*>(iq2s_grid); }
};

template <>
struct Exp8Grid<T_IQ3_XXS> {
    static constexpr bool has_grid = true;
    static constexpr int words = 256;   // 256 * uint32_t = 1024 B
    __device__ __forceinline__ static const uint32_t* src() { return iq3xxs_grid; }
};

template <>
struct Exp8Grid<T_IQ3_S> {
    static constexpr bool has_grid = true;
    static constexpr int words = 512;   // 512 * uint32_t = 2048 B
    __device__ __forceinline__ static const uint32_t* src() { return iq3s_grid; }
};

template <bool StageGrid, GgmlType TA>
struct Exp8GridSlot {
    uint32_t dummy;
    __device__ __forceinline__ const uint32_t* ptr() const { return nullptr; }
    __device__ __forceinline__ void load(int) {}
};

template <GgmlType TA>
struct Exp8GridSlot<true, TA> {
    static_assert(Exp8Grid<TA>::has_grid, "LDS grid staging requires an IQ codebook type");
    alignas(16) uint32_t data[Exp8Grid<TA>::words];
    __device__ __forceinline__ const uint32_t* ptr() const { return data; }
    __device__ __forceinline__ void load(int tid) {
        const uint32_t* s = Exp8Grid<TA>::src();
#pragma unroll
        for (int i = tid; i < Exp8Grid<TA>::words; i += 256) data[i] = s[i];
    }
};

template <bool GuCopy, GgmlType TA>
struct Exp8StageSlot {
    uint32_t dummy;
};

template <GgmlType TA>
struct Exp8StageSlot<true, TA> {
    static constexpr size_t packed_row = N / qtraits<TA>::qk * qtraits<TA>::bs;
    static_assert((8 * packed_row) % 16 == 0, "aligned 8-row staging group");
    alignas(16) unsigned char data[8 * packed_row];
};

// Hoisted per-(kbx, iqs) weight decoder and dot traits for all non-Q4_K EXP8 formats.
template <GgmlType TA>
struct SplitGu;

template <>
struct SplitGu<T_IQ2_XXS> {
    using W = IqDot32;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t* sgrid) {
        const block_iq2_xxs* b = reinterpret_cast<const block_iq2_xxs*>(row) + kbx;
        const int packed = get_int_b2(b->qs, iqs);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const uint32_t aux = get_int_b2(b->qs, iqs + 1);
        W r;
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const uint2 grid = STAGE_GRID ? reinterpret_cast<const uint2*>(sgrid)[qs[k / 2]]
                                          : reinterpret_cast<const uint2*>(iq2xxs_grid)[qs[k / 2]];
            const unsigned v7 = (aux >> (7 * k / 2)) & 127u;
            const unsigned signs = v7 | ((__popc(v7) & 1u) << 7);
            r.grid[k] = iq_signed_nibble((int) grid.x, signs);
            r.grid[k + 1] = iq_signed_nibble((int) grid.y, signs >> 4);
        }
        r.d = __half2float(b->d);
        r.scale0 = (aux >> 27) | 1;
        r.scale1 = 0;
        return r;
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return iq_dot32<T_IQ2_XXS>(w, x_row[kby + iqs / 2]);
    }
};

template <>
struct SplitGu<T_IQ2_S> {
    using W = IqDot32;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t* sgrid) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + kbx;
        const int packed = get_int_b2(b->qs, iqs / 2);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const int qh = b->qh[iqs / 2];
        const int signs_packed = get_int_b2(b->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&signs_packed);
        W r;
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const int gidx = qs[k / 2] | ((qh << (8 - k)) & 0x300);
            const int* grid = STAGE_GRID ? reinterpret_cast<const int*>(sgrid + 2 * gidx)
                                         : reinterpret_cast<const int*>(iq2s_grid + gidx);
            r.grid[k] = iq_signed_nibble(grid[0], signs[k / 2]);
            r.grid[k + 1] = iq_signed_nibble(grid[1], signs[k / 2] >> 4);
        }
        r.d = __half2float(b->d);
        r.scale0 = b->scales[iqs / 2] & 15;
        r.scale1 = b->scales[iqs / 2] >> 4;
        return r;
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return iq_dot32<T_IQ2_S>(w, x_row[kby + iqs / 2]);
    }
};

template <>
struct SplitGu<T_IQ3_XXS> {
    using W = IqDot32;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t* sgrid) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + kbx;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const uint32_t aux = get_int_b2(b->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const uint32_t signs = unpack_ksigns(aux >> (7 * k / 2));
            const uint32_t g0 = STAGE_GRID ? sgrid[qs[k]] : iq3xxs_grid[qs[k]];
            const uint32_t g1 = STAGE_GRID ? sgrid[qs[i_add1(k)]] : iq3xxs_grid[qs[k + 1]];
            r.grid[k] = iq_signed_nibble((int) g0, signs);
            r.grid[k + 1] = iq_signed_nibble((int) g1, signs >> 4);
        }
        r.d = __half2float(b->d);
        r.scale0 = aux >> 28;
        r.scale1 = 0;
        return r;
    }
    __device__ __forceinline__ static constexpr int i_add1(int k) { return k + 1; }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return iq_dot32<T_IQ3_XXS>(w, x_row[kby + iqs / 2]);
    }
};

template <>
struct SplitGu<T_IQ3_S> {
    using W = IqDot32;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t* sgrid) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + kbx;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const int qh = b->qh[iqs / 2];
        const int signs_packed = get_int_b2(b->signs, iqs / 2);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&signs_packed);
        W r;
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const int i0 = qs[k] | ((qh << (8 - k)) & 0x100);
            const int i1 = qs[k + 1] | ((qh << (7 - k)) & 0x100);
            const uint32_t g0 = STAGE_GRID ? sgrid[i0] : iq3s_grid[i0];
            const uint32_t g1 = STAGE_GRID ? sgrid[i1] : iq3s_grid[i1];
            r.grid[k] = iq_signed_nibble((int) g0, signs[k / 2]);
            r.grid[k + 1] = iq_signed_nibble((int) g1, signs[k / 2] >> 4);
        }
        r.d = __half2float(b->d);
        r.scale0 = 1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 4)) & 15);
        r.scale1 = 0;
        return r;
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return iq_dot32<T_IQ3_S>(w, x_row[kby + iqs / 2]);
    }
};

template <>
struct SplitGu<T_IQ4_XS> {
    using W = Iq4XsDot32;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t*) {
        return iq4xs_load32(row, kbx, iqs);
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return iq4xs_dot32(w, x_row[kby + iqs / 4]);
    }
};

template <>
struct SplitGu<T_Q5_K> {
    using W = Q5KDot16;
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t*) {
        return q5k_load16(row, kbx, iqs);
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        return q5k_dot16(w, x_row + kby, iqs);
    }
};

template <>
struct SplitGu<T_Q6_K> {
    struct W {
        int vi[2];
        int sc[2];
        float d;
    };
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t*) {
        const block_q6_K* bq6_K = reinterpret_cast<const block_q6_K*>(row) + kbx;
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        const int vl = get_int_b2(bq6_K->ql, iqs);
        const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
        const int8_t* scales = bq6_K->scales + scale_offset;
        W r;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            r.sc[i] = scales[4 * i];
            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
#if FNX_HIP
            r.vi[i] = (int) (((uint32_t) (vil | vih) + 0x60606060u) ^ 0x80808080u);
#else
            r.vi[i] = qd_vsubss4((vil | vih), 0x20202020);
#endif
        }
        r.d = __half2float(bq6_K->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        const block_q8_1* bq8_1 = x_row + kby;
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        const int q8_word = iqs % QI8_1;
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const block_q8_1* bx = bq8_1 + bq8_offset + 2 * i;
            const int u = get_int_b4(bx->qs, q8_word);
            const float d8 = __low2float(bx->ds);
            sumf += d8 * (qd_dp4a(w.vi[i], u, 0) * w.sc[i]);
        }
        return w.d * sumf;
    }
};

template <>
struct SplitGu<T_Q8_0> {
    struct W {
        int v[2];
        float d;
    };
    template <bool STAGE_GRID>
    __device__ __forceinline__ static W load(const char* row, int kbx, int iqs, const uint32_t*) {
        const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row) + kbx;
        W r;
        r.v[0] = get_int_b2(b->qs, iqs + 0);
        r.v[1] = get_int_b2(b->qs, iqs + 1);
        r.d = __half2float(b->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const W& w, const block_q8_1* x_row, int kby, int iqs) {
        const block_q8_1& x = x_row[kby];
        int sumi = 0;
        sumi = qd_dp4a(w.v[0], get_int_b4(x.qs, iqs + 0), sumi);
        sumi = qd_dp4a(w.v[1], get_int_b4(x.qs, iqs + 1), sumi);
        return (w.d * __half2float(__low2half(x.ds))) * ((float) sumi);
    }
};

// Direct g_moe_hq SwiGLU + q8_1 epilogue shared by k_moe_gu_exp8 and k_moe_gu_exp8_q4k.
template <int NT_MAX>
__device__ __forceinline__ void exp8_swiglu_epilogue(const float (&res)[NT_MAX][64], int warp, int lane, int cn,
                                                     int bx, const Job& jb, int c0, block_q8_1* hq_dst) {
    if (warp >= cn) return;
    const float gg = res[warp][lane];
    const float uu = res[warp][32 + lane];
    const float xi = silu_(gg) * uu;
    const float amax = wmax(fabsf(xi)), sm = wsum(xi);
    const float d = amax / 127.0f;
    const int pair = jb.out[c0 + warp];
    block_q8_1& h = hq_dst[(size_t) pair * DN_MAXB + bx];
    h.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (lane == 0) h.ds = make_half2(__float2half(d), __float2half(sm));
}

template <int NT_MAX>
__device__ __forceinline__ void exp8_stage_x(block_q8_1* sx, const block_q8_1* y, const Job& jb,
                                             int c0, int cn, int xb, int tid) {
    static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
    const int words = cn * xb * 9;
    int* s_words = reinterpret_cast<int*>(sx);
    for (int q = tid; q < words; q += 256) {
        const int b = q / 9, j = b / xb, bi = b % xb;
        s_words[q] = reinterpret_cast<const int*>(y + (size_t) jb.y[c0 + j] * xb + bi)[q % 9];
    }
}

template <GgmlType TA, int NT_MAX, bool STAGE_X, bool STAGE_GRID, bool COPY = false, bool GU_COPY = false>
__global__ void __launch_bounds__(256) k_moe_gu_exp8(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, CacheDownCopy copy = {}, CacheGateUpCopy cache = {},
    const int* __restrict__ job_idx = nullptr, block_q8_1* __restrict__ hq_out = nullptr) {
    static_assert(TA != T_Q4_K, "Q4_K uses k_moe_gu_exp8_q4k");
    static_assert(NT_MAX == 1 || NT_MAX == 4, "chunk limit");
    static_assert(!COPY || exp8_is_iq4<TA>(), "CacheDownCopy requires IQ4 reuse type");
    static_assert(!GU_COPY || exp8_is_iq4<TA>(), "CacheGateUpCopy requires IQ4 reuse type");

    const int ffs = w.nrows / 2, nxb = ffs / 32;
    int bx = blockIdx.x, ji = blockIdx.y;
    if constexpr (COPY) {
        if ((int) blockIdx.x < copy.blocks) {
            copy_down_experts(copy);
            return;
        }
        const int block = blockIdx.x - copy.blocks;
        bx = block % nxb;
        ji = block / nxb;
    }
    if (ji >= *njobs) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    __shared__ bool missing;
    __shared__ float res[NT_MAX][64];
    __shared__ Exp8GridSlot<STAGE_GRID, TA> sgrid;
    __shared__ Exp8StageSlot<GU_COPY, TA> staged;
    extern __shared__ int exp8_sx[];

    if (tid == 0) {
        jb = jobs[ji_real];
        if constexpr (GU_COPY) {
            missing = false;
            const int g = cache.layer * NEXP + jb.w;
            if (cache.miss_bits) missing = (cache.miss_bits[jb.w / 32] >> (jb.w % 32)) & 1u;
            else for (int i = 0, n = cache.list[0]; i < n; ++i)
                if (cache.list[1 + i] == g) { missing = true; break; }
        }
    }
    if constexpr (STAGE_GRID) sgrid.load(tid);
    __syncthreads();

    constexpr int qk = qtraits<TA>::qk, vdr = vdr_of<TA>(), ipb = qtraits<TA>::qi / vdr;
    static_assert(32 % ipb == 0, "EXP8 requires warp-divisible IQ/K sub-blocks");
    const int bpr = w.ncols / qk, xb = w.ncols / QK8_1, r0 = bx * 32;
    const int iqs = vdr * (lane % ipb), kb0 = lane / ipb;
    const char* exp_base = moe_expert(w, jb.w);
    block_q8_1* hq_dst = hq_out ? hq_out : &g_moe_hq[0][0];
    block_q8_1* sx = STAGE_X ? reinterpret_cast<block_q8_1*>(exp8_sx) : nullptr;

#pragma unroll 1
    for (int c0 = 0; c0 < jb.n; c0 += NT_MAX) {
        const int cn = min(NT_MAX, jb.n - c0);
        if constexpr (STAGE_X) {
            exp8_stage_x<NT_MAX>(sx, y, jb, c0, cn, xb, tid);
            __syncthreads();
        }
        if constexpr (GU_COPY) {
            // 8 passes of 8 contiguous rows so missing cache rows fit in 5.2-8.6 KiB LDS.
#pragma unroll 1
            for (int p = 0; p < 8; ++p) {
                const int local_row = p * 8 + warp;
                const int global_row = (local_row < 32 ? 0 : ffs) + r0 + (local_row & 31);
                const char* wr = exp_base + (size_t) global_row * w.row_a;
                if (c0 == 0 && missing) {
                    const int row0 = (p < 4 ? 0 : ffs) + r0 + (p & 3) * 8;
                    const size_t offset = (size_t) row0 * w.row_a;
                    const size_t bytes = (size_t) min(8, w.nrows - row0) * w.row_a;
                    const int g = cache.layer * NEXP + jb.w;
                    const char* src = cache.host[g] + offset;
                    char* dst = const_cast<char*>(exp_base) + offset;
                    for (size_t i = tid; i < bytes / 16; i += 256) {
                        const uint4 chunk = reinterpret_cast<const uint4*>(src)[i];
                        reinterpret_cast<uint4*>(staged.data)[i] = chunk;
                        reinterpret_cast<uint4*>(dst)[i] = chunk;
                    }
                    for (size_t i = bytes / 16 * 16 + tid; i < bytes; i += 256)
                        dst[i] = staged.data[i] = src[i];
                    __syncthreads();
                    wr = reinterpret_cast<const char*>(staged.data) + warp * w.row_a;
                }
                float acc[NT_MAX] = {};
#pragma unroll 8
                for (int kbx = kb0; kbx < bpr; kbx += 32 / ipb) {
                    const auto rw = SplitGu<TA>::template load<STAGE_GRID>(wr, kbx, iqs, sgrid.ptr());
                    const int kby = kbx * (qk / QK8_1);
#pragma unroll
                    for (int j = 0; j < NT_MAX; ++j) {
                        if (j >= cn) break;
                        const block_q8_1* x_row = STAGE_X ? (sx + j * xb) : (y + (size_t) jb.y[c0 + j] * xb);
                        acc[j] += SplitGu<TA>::dot(rw, x_row, kby, iqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT_MAX; ++j) {
                    if (j >= cn) break;
                    const float s = wsum(acc[j]);
                    if (lane == 0) res[j][local_row] = s;
                }
                if (c0 == 0 && missing) __syncthreads();
            }
        } else if constexpr (NT_MAX == 1) {
            // 1 row/warp with rolled pass loop keeps VGPRs low and I-cache compact on NT=1.
#pragma unroll 1
            for (int p = 0; p < 8; ++p) {
                const int local_row = p * 8 + warp;
                const int global_row = (local_row < 32 ? 0 : ffs) + r0 + (local_row & 31);
                const char* wr = exp_base + (size_t) global_row * w.row_a;
                float acc = 0.f;
                const block_q8_1* x_row = STAGE_X ? sx : (y + (size_t) jb.y[c0] * xb);
#pragma unroll 8
                for (int kbx = kb0; kbx < bpr; kbx += 32 / ipb) {
                    const auto rw = SplitGu<TA>::template load<STAGE_GRID>(wr, kbx, iqs, sgrid.ptr());
                    acc += SplitGu<TA>::dot(rw, x_row, kbx * (qk / QK8_1), iqs);
                }
                const float s = wsum(acc);
                if (lane == 0) res[0][local_row] = s;
            }
        } else {
            // 4 passes of 16 rows (2 rows/warp) cut chunk-barrier overhead and double x reuse.
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int i0 = p * 16 + warp;
                const int i1 = i0 + 8;
                const int gr0 = (i0 < 32 ? 0 : ffs) + r0 + (i0 & 31);
                const int gr1 = (i1 < 32 ? 0 : ffs) + r0 + (i1 & 31);
                const char* w0 = exp_base + (size_t) gr0 * w.row_a;
                const char* w1 = exp_base + (size_t) gr1 * w.row_a;
                float a0[NT_MAX] = {};
                float a1[NT_MAX] = {};
#pragma unroll 8
                for (int kbx = kb0; kbx < bpr; kbx += 32 / ipb) {
                    const auto rw0 = SplitGu<TA>::template load<STAGE_GRID>(w0, kbx, iqs, sgrid.ptr());
                    const auto rw1 = SplitGu<TA>::template load<STAGE_GRID>(w1, kbx, iqs, sgrid.ptr());
                    const int kby = kbx * (qk / QK8_1);
#pragma unroll
                    for (int j = 0; j < NT_MAX; ++j) {
                        if (j >= cn) break;
                        const block_q8_1* x_row = STAGE_X ? (sx + j * xb) : (y + (size_t) jb.y[c0 + j] * xb);
                        a0[j] += SplitGu<TA>::dot(rw0, x_row, kby, iqs);
                        a1[j] += SplitGu<TA>::dot(rw1, x_row, kby, iqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT_MAX; ++j) {
                    if (j >= cn) break;
                    const float s0 = wsum(a0[j]);
                    const float s1 = wsum(a1[j]);
                    if (lane == 0) {
                        res[j][i0] = s0;
                        res[j][i1] = s1;
                    }
                }
            }
        }
        __syncthreads();
        exp8_swiglu_epilogue<NT_MAX>(res, warp, lane, cn, bx, jb, c0, hq_dst);
        if (c0 + NT_MAX < jb.n) {
            if constexpr (GU_COPY) {
                if (c0 == 0 && missing) __threadfence();
            }
            __syncthreads();
        }
    }
}

// Q4_K uses Q4K_LPR=8 lanes/row (16-byte uint4 loads and 64-value super-groups) so gate|up sums match k_moe_gu_q4k bit-for-bit.
template <int NT_MAX, bool STAGE_X>
__global__ void __launch_bounds__(Q4K_THREADS) k_moe_gu_exp8_q4k(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, const int* __restrict__ job_idx = nullptr,
    block_q8_1* __restrict__ hq_out = nullptr) {
    static_assert(NT_MAX == 1 || NT_MAX == 4, "chunk limit");
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row_in_blk = tid / Q4K_LPR, lane8 = tid % Q4K_LPR;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    __shared__ float res[NT_MAX][64];
    extern __shared__ int exp8_sx[];
    if (tid == 0) jb = jobs[ji_real];
    __syncthreads();

    const int bx = blockIdx.x, ffs = w.nrows / 2, r0 = bx * 32, xb = w.ncols / QK8_1;
    const char* exp_base = moe_expert(w, jb.w);
    const char* wg = exp_base + (size_t) (r0 + row_in_blk) * w.row_a;
    const char* wu = exp_base + (size_t) (ffs + r0 + row_in_blk) * w.row_a;
    block_q8_1* hq_dst = hq_out ? hq_out : &g_moe_hq[0][0];
    block_q8_1* sx = STAGE_X ? reinterpret_cast<block_q8_1*>(exp8_sx) : nullptr;
    const int sx_y[4] = {0, 1, 2, 3};

#pragma unroll 1
    for (int c0 = 0; c0 < jb.n; c0 += NT_MAX) {
        const int cn = min(NT_MAX, jb.n - c0);
        if constexpr (STAGE_X) {
            exp8_stage_x<NT_MAX>(sx, y, jb, c0, cn, xb, tid);
            __syncthreads();
        }
        const block_q8_1* x_ptr = STAGE_X ? sx : y;
        const int* y_idx = STAGE_X ? sx_y : (jb.y + c0);
        {
            float tg[NT_MAX] = {};
            q4k_row<NT_MAX>(wg, lane8, cn, x_ptr, y_idx, xb, tg);
#pragma unroll
            for (int j = 0; j < NT_MAX; ++j) {
                if (j >= cn) break;
                float v = tg[j];
#pragma unroll
                for (int o = Q4K_LPR / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane8 == 0) res[j][row_in_blk] = v;
            }
        }
        asm volatile("" ::: "memory");
        {
            float tu[NT_MAX] = {};
            q4k_row<NT_MAX>(wu, lane8, cn, x_ptr, y_idx, xb, tu);
#pragma unroll
            for (int j = 0; j < NT_MAX; ++j) {
                if (j >= cn) break;
                float v = tu[j];
#pragma unroll
                for (int o = Q4K_LPR / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane8 == 0) res[j][32 + row_in_blk] = v;
            }
        }
        __syncthreads();
        exp8_swiglu_epilogue<NT_MAX>(res, warp, lane, cn, bx, jb, c0, hq_dst);
        if (c0 + NT_MAX < jb.n) __syncthreads();
    }
}

template <GgmlType TA, int NT_MAX, bool STAGE_X>
void moe_exp8_launch_iq(const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const block_q8_1* x,
                        cudaStream_t s, size_t smem, bool stage_grid, const CacheDownCopy* copy,
                        const CacheGateUpCopy* gu_copy, const int* job_idx) {
    const int nxb = (w.nrows / 2) / 32;
    const dim3 grid(nxb, njobs_max);
    const dim3 fused_grid(nxb * njobs_max + (copy ? copy->blocks : 0));
    const CacheDownCopy empty_dn{};
    const CacheGateUpCopy empty_gu{};
    if (stage_grid) {
        if (gu_copy && copy) k_moe_gu_exp8<TA, NT_MAX, STAGE_X, true, true, true><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, *gu_copy, job_idx);
        else if (gu_copy)    k_moe_gu_exp8<TA, NT_MAX, STAGE_X, true, false, true><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, *gu_copy, job_idx);
        else if (copy)       k_moe_gu_exp8<TA, NT_MAX, STAGE_X, true, true, false><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, empty_gu, job_idx);
        else                 k_moe_gu_exp8<TA, NT_MAX, STAGE_X, true, false, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, empty_gu, job_idx);
    } else {
        if (gu_copy && copy) k_moe_gu_exp8<TA, NT_MAX, STAGE_X, false, true, true><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, *gu_copy, job_idx);
        else if (gu_copy)    k_moe_gu_exp8<TA, NT_MAX, STAGE_X, false, false, true><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, *gu_copy, job_idx);
        else if (copy)       k_moe_gu_exp8<TA, NT_MAX, STAGE_X, false, true, false><<<fused_grid, 256, smem, s>>>(w, jobs, njobs, x, *copy, empty_gu, job_idx);
        else                 k_moe_gu_exp8<TA, NT_MAX, STAGE_X, false, false, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, empty_dn, empty_gu, job_idx);
    }
}

template <GgmlType TA>
void moe_exp8_launch_typed(const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const block_q8_1* x,
                           cudaStream_t s, int T, int lds_mode, const CacheDownCopy* copy,
                           const CacheGateUpCopy* gu_copy, const int* job_idx) {
    if (w.row_a != (size_t) N / qtraits<TA>::qk * qtraits<TA>::bs)
        throw std::runtime_error("moe_gate_up (exp8): invalid row stride");
    const int nxb = (w.nrows / 2) / 32;
    const bool stage_x = (lds_mode == 1);
    const bool stage_grid = exp8_is_iq4<TA>() && (lds_mode == 1 || lds_mode == 2);
    const int nt = (T == 1) ? 1 : 4;
    const size_t smem = stage_x ? (size_t) (T < nt ? T : nt) * (w.ncols / QK8_1) * sizeof(block_q8_1) : 0;
    if constexpr (TA == T_Q4_K) {
        const dim3 grid(nxb, njobs_max);
        if (nt == 1) {
            if (stage_x) k_moe_gu_exp8_q4k<1, true><<<grid, Q4K_THREADS, smem, s>>>(w, jobs, njobs, x, job_idx);
            else         k_moe_gu_exp8_q4k<1, false><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, x, job_idx);
        } else {
            if (stage_x) k_moe_gu_exp8_q4k<4, true><<<grid, Q4K_THREADS, smem, s>>>(w, jobs, njobs, x, job_idx);
            else         k_moe_gu_exp8_q4k<4, false><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, x, job_idx);
        }
    } else if constexpr (exp8_is_iq4<TA>()) {
        if (nt == 1) {
            if (stage_x) moe_exp8_launch_iq<TA, 1, true>(w, jobs, njobs, njobs_max, x, s, smem, stage_grid, copy, gu_copy, job_idx);
            else         moe_exp8_launch_iq<TA, 1, false>(w, jobs, njobs, njobs_max, x, s, 0, stage_grid, copy, gu_copy, job_idx);
        } else {
            if (stage_x) moe_exp8_launch_iq<TA, 4, true>(w, jobs, njobs, njobs_max, x, s, smem, stage_grid, copy, gu_copy, job_idx);
            else         moe_exp8_launch_iq<TA, 4, false>(w, jobs, njobs, njobs_max, x, s, 0, stage_grid, copy, gu_copy, job_idx);
        }
    } else {
        const dim3 grid(nxb, njobs_max);
        if (nt == 1) {
            if (stage_x) k_moe_gu_exp8<TA, 1, true, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, {}, {}, job_idx);
            else         k_moe_gu_exp8<TA, 1, false, false><<<grid, 256, 0, s>>>(w, jobs, njobs, x, {}, {}, job_idx);
        } else {
            if (stage_x) k_moe_gu_exp8<TA, 4, true, false><<<grid, 256, smem, s>>>(w, jobs, njobs, x, {}, {}, job_idx);
            else         k_moe_gu_exp8<TA, 4, false, false><<<grid, 256, 0, s>>>(w, jobs, njobs, x, {}, {}, job_idx);
        }
    }
}

void moe_exp8_dispatch(uint32_t type, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max,
                       const block_q8_1* x, cudaStream_t s, int T, const CacheDownCopy* copy = nullptr,
                       const CacheGateUpCopy* gu_copy = nullptr, const int* job_idx = nullptr) {
    if (!moe_gu_can_exp8_type(type))
        throw std::runtime_error("moe_gate_up (exp8): unsupported weight type " + std::to_string(type));
    if (w.ncols != N || (w.nrows % 64) != 0 || (w.nrows / 2) / 32 > DN_MAXB)
        throw std::runtime_error("moe_gate_up (exp8): invalid matrix shape");
    if (T < 1 || T > MAXT || njobs_max < 1 || njobs_max > MAXT * TOPK)
        throw std::runtime_error("moe_gate_up (exp8): window or job bound out of range");
    const bool is_iq = type == T_IQ2_XXS || type == T_IQ2_S || type == T_IQ3_XXS || type == T_IQ3_S;
    if ((copy || gu_copy) && !is_iq)
        throw std::runtime_error("moe_gate_up (exp8): cache copy requires IQ2/IQ3 weight type");
    if (copy && (copy->blocks < 1 || copy->blocks > 256 || !copy->host || !copy->slots || !copy->list ||
                 !copy->pool || !copy->bytes || copy->bytes % 16 || copy->offset % 16 || copy->slot_bytes % 16 ||
                 copy->offset > copy->slot_bytes || copy->bytes > copy->slot_bytes - copy->offset))
        throw std::runtime_error("invalid EXP8 IQ down copy descriptor");
    if (gu_copy && (!gu_copy->host || (!gu_copy->list && !gu_copy->miss_bits) || (!copy && !job_idx) ||
                    (copy && gu_copy->list != copy->list) ||
                    gu_copy->layer < 0 || gu_copy->layer >= NL || !w.ep))
        throw std::runtime_error("invalid EXP8 IQ gate/up copy descriptor");
    const int lds_mode = moe_lds_mode();
    switch (type) {
        case T_IQ2_XXS: moe_exp8_launch_typed<T_IQ2_XXS>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_IQ2_S:   moe_exp8_launch_typed<T_IQ2_S>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_IQ3_XXS: moe_exp8_launch_typed<T_IQ3_XXS>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_IQ3_S:   moe_exp8_launch_typed<T_IQ3_S>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_IQ4_XS:  moe_exp8_launch_typed<T_IQ4_XS>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_Q4_K:    moe_exp8_launch_typed<T_Q4_K>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_Q5_K:    moe_exp8_launch_typed<T_Q5_K>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_Q6_K:    moe_exp8_launch_typed<T_Q6_K>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        case T_Q8_0:    moe_exp8_launch_typed<T_Q8_0>(w, jobs, njobs, njobs_max, x, s, T, lds_mode, copy, gu_copy, job_idx); break;
        default: throw std::runtime_error("moe_gate_up (exp8): unhandled weight type");
    }
}

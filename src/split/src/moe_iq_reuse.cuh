// Reuse one decoded 32-value IQ weight group for every row of its expert job.
// The integer expressions follow ggml-cuda/vecdotq.cuh, including its divisions
// before scaling. This is not a conversion to a different quantization format.
#pragma once

struct IqDot32 {
    int grid[8];
    float d;
    int scale0, scale1;
};

// The ggml HIP header supplies byte-by-byte implementations after kernels.cu
// removes the compatibility macros. Use the tree's packed operations here.
// IQ grid magnitudes fit positive signed bytes; applying their signs cannot
// saturate, so wrapping and signed-saturating subtraction give the same values.
__device__ __forceinline__ int iq_sign_mask(int bits) {
#if FNX_HIP
    return ::strata::hip_compat::vcmpne4(bits, 0);
#else
    return __vcmpne4(bits, 0);
#endif
}

__device__ __forceinline__ int iq_signed_grid(int grid, int mask) {
#if FNX_HIP
    return ::strata::hip_compat::vsub4(grid ^ mask, mask);
#else
    return __vsub4(grid ^ mask, mask);
#endif
}

template <ggml_type TA>
__device__ __forceinline__ IqDot32 iq_decode32(const char* row, int kbx, int iqs) {
    IqDot32 r;
    if constexpr (TA == GGML_TYPE_IQ2_S) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + kbx;
        const int packed = get_int_b2(b->qs, iqs / 2);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const int qh = b->qh[iqs / 2];
        const int signs_packed = get_int_b2(b->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&signs_packed);
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const int* grid = reinterpret_cast<const int*>(iq2s_grid + (qs[k / 2] | ((qh << (8 - k)) & 0x300)));
            const int s0 = iq_sign_mask(((signs[k / 2] & 0x03) << 7) | ((signs[k / 2] & 0x0C) << 21));
            const int s1 = iq_sign_mask(((signs[k / 2] & 0x30) << 3) | ((signs[k / 2] & 0xC0) << 17));
            r.grid[k] = iq_signed_grid(grid[0], s0);
            r.grid[k + 1] = iq_signed_grid(grid[1], s1);
        }
        r.d = __half2float(b->d);
        r.scale0 = b->scales[iqs / 2] & 15;
        r.scale1 = b->scales[iqs / 2] >> 4;
    } else if constexpr (TA == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + kbx;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const uint32_t aux = get_int_b2(b->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const uint32_t signs = unpack_ksigns(aux >> (7 * k / 2));
            const int s0 = iq_sign_mask(signs & 0x08040201);
            const int s1 = iq_sign_mask(signs & 0x80402010);
            r.grid[k] = iq_signed_grid(iq3xxs_grid[qs[k]], s0);
            r.grid[k + 1] = iq_signed_grid(iq3xxs_grid[qs[k + 1]], s1);
        }
        r.d = __half2float(b->d);
        r.scale0 = aux >> 28;
        r.scale1 = 0;
    } else {
        static_assert(TA == GGML_TYPE_IQ3_S, "IQ reuse type");
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + kbx;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed);
        const int qh = b->qh[iqs / 2];
        const int signs_packed = get_int_b2(b->signs, iqs / 2);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&signs_packed);
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            const int g0 = iq3s_grid[qs[k] | ((qh << (8 - k)) & 0x100)];
            const int g1 = iq3s_grid[qs[k + 1] | ((qh << (7 - k)) & 0x100)];
            const int s0 = iq_sign_mask(((signs[k / 2] & 0x03) << 7) | ((signs[k / 2] & 0x0C) << 21));
            const int s1 = iq_sign_mask(((signs[k / 2] & 0x30) << 3) | ((signs[k / 2] & 0xC0) << 17));
            r.grid[k] = iq_signed_grid(g0, s0);
            r.grid[k + 1] = iq_signed_grid(g1, s1);
        }
        r.d = __half2float(b->d);
        r.scale0 = 1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 4)) & 15);
        r.scale1 = 0;
    }
    return r;
}

template <ggml_type TA>
__device__ __forceinline__ float iq_dot32(const IqDot32& w, const block_q8_1& x) {
    int sum = 0;
    if constexpr (TA == GGML_TYPE_IQ2_S) {
        int a = 0, b = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int q = get_int_b4(x.qs, k);
            if (k < 4) a = ggml_cuda_dp4a(w.grid[k], q, a);
            else b = ggml_cuda_dp4a(w.grid[k], q, b);
        }
        sum = (a * w.scale0 + b * w.scale1 + (a + b) / 2) / 4;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) sum = ggml_cuda_dp4a(w.grid[k], get_int_b4(x.qs, k), sum);
        if constexpr (TA == GGML_TYPE_IQ3_XXS) sum = (w.scale0 * sum + sum / 2) / 2;
        else sum *= w.scale0;
    }
    const float d = w.d * __low2float(x.ds);
    return d * sum;
}

template <ggml_type TA, int TT, int G = 32>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_iq(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    static_assert(G == 16 || G == 32, "IQ lanes per row");
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const char* wr = moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
    // A 16-lane row keeps both halves of the original 32-lane sums separately.
    // Join them before the remaining shuffles to preserve the reduction order.
    constexpr int PARTS = 32 / G;
    float tmp[PARTS][TT] = {};
    if constexpr (TA == GGML_TYPE_IQ2_S || TA == GGML_TYPE_IQ3_XXS || TA == GGML_TYPE_IQ3_S) {
        constexpr int qk = ggml_cuda_type_traits<TA>::qk, vdr = vdr_of<TA>();
        constexpr int per_block = ggml_cuda_type_traits<TA>::qi / vdr;
        static_assert(G >= per_block && G % per_block == 0, "lanes per weight block");
        const int gl = threadIdx.x % G, kqs = vdr * (gl % per_block);
        if (live)
#pragma unroll
            for (int p = 0; p < PARTS; ++p) {
#pragma unroll 8
                for (int kbx = (gl + p * G) / per_block; kbx < w.ncols / qk; kbx += 32 / per_block) {
                    const IqDot32 decoded = iq_decode32<TA>(wr, kbx, kqs);
#pragma unroll
                    for (int j = 0; j < TT; ++j)
                        if (j < jb.n) tmp[p][j] += iq_dot32<TA>(decoded, y[(size_t) jb.y[j] * (w.ncols / 32) + kbx * (qk / 32) + kqs / 2]);
                }
            }
    } else {
        static_assert(G == 32, "fallback keeps its original lane group");
        job_rows<TA, G, TT>(wr, live, w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp[0]);
    }
    if constexpr (PARTS == 2) {
#pragma unroll
        for (int j = 0; j < TT; ++j) tmp[0][j] += tmp[1][j];
    }
    job_store<G, TT>(tmp[0], jb.n, live, row, jb.out, out, w.nrows);
}

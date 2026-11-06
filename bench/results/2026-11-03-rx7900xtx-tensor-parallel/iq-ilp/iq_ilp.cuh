template <ggml_type TA, int A>
__device__ __forceinline__ float iq_dot32_ilp(const IqDot32& w, const block_q8_1& x) {
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
        int partial[A] = {};
#pragma unroll
        for (int k = 0; k < 8; ++k) partial[k % A] = ggml_cuda_dp4a(w.grid[k], get_int_b4(x.qs, k), partial[k % A]);
#pragma unroll
        for (int k = 0; k < A; ++k) sum += partial[k];
        if constexpr (TA == GGML_TYPE_IQ3_XXS) sum = (w.scale0 * sum + sum / 2) / 2;
        else sum *= w.scale0;
    }
    const float d = w.d * __low2float(x.ds);
    return d * sum;
}

template <ggml_type TA, int TT, int G = 32, bool COPY = false, int A = 2>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_iq_ilp(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, float* __restrict__ out, CacheDownCopy copy = {}) {
    int row_block = blockIdx.x, ji = blockIdx.y;
    if constexpr (COPY) {
        // Prefix blocks read RAM while the remaining blocks compute with the
        // already resident gate/up weights. No grid barrier: kernel completion
        // orders every copied down byte before the subsequent down launch.
        if ((int) blockIdx.x < copy.blocks) {
            const size_t per = copy.bytes / 16;
            for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
                 i < (size_t) copy.list[0] * per; i += (size_t) copy.blocks * blockDim.x) {
                const int g = copy.list[1 + i / per];
                char* dst = copy.pool + (size_t) (copy.slots[g] - copy.slot0) * copy.slot_bytes + copy.offset;
                reinterpret_cast<uint4*>(dst)[i % per] = reinterpret_cast<const uint4*>(copy.host[g])[i % per];
            }
            return;
        }
        const int rows = (w.nrows + MMVQ_THREADS / G - 1) / (MMVQ_THREADS / G);
        const int block = blockIdx.x - copy.blocks;
        row_block = block % rows;
        ji = block / rows;
    }
    if (ji >= *njobs) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    static_assert(G == 16 || G == 32, "IQ lanes per row");
    const int row = row_block * (MMVQ_THREADS / G) + threadIdx.x / G;
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
                        if (j < jb.n) tmp[p][j] += iq_dot32_ilp<TA, A>(decoded, y[(size_t) jb.y[j] * (w.ncols / 32) + kbx * (qk / 32) + kqs / 2]);
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

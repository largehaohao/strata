// The IQ gate/up block fills its own cold rows before using them. Hot jobs
// compute immediately while cold jobs read RAM. Kernel completion publishes all
// copied gate/up and down bytes for subsequent layers/windows. Keep the IQ dot
// and reduction order identical to k_moe_gu_iq; moe_cache_gu_test checks both.
template <ggml_type TA, int TT, int G = 32, bool COPY = false, bool PACKED_SIGNS = false>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_iq_cache(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, float* __restrict__ out, CacheDownCopy copy = {}, CacheGateUpCopy cache = {}) {
    int row_block = blockIdx.x, ji = blockIdx.y;
    if constexpr (COPY) {
        // Prefix blocks read down weights while the remaining blocks fill and
        // compute GU rows. Kernel completion orders every copied byte before
        // the subsequent down launch; no grid barrier is needed.
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
    __shared__ bool missing;
    if (threadIdx.x == 0) {
        jb = jobs[ji];
        missing = false;
        const int g = cache.layer * NEXP + jb.w;
        for (int i = 0, n = cache.list[0]; i < n; ++i)
            if (cache.list[1 + i] == g) { missing = true; break; }
    }
    __syncthreads();
    constexpr int group_rows = MMVQ_THREADS / G;
    constexpr size_t packed_row = N / ggml_cuda_type_traits<TA>::qk * ggml_cuda_type_traits<TA>::bs;
    static_assert(G == 16 && group_rows * packed_row % 16 == 0, "aligned raw row groups");
    __shared__ __align__(16) unsigned char staged[group_rows * packed_row];
    if (missing) {
        const int g = cache.layer * NEXP + jb.w;
        const int row0 = row_block * group_rows;
        const size_t offset = (size_t) row0 * w.row_a;
        const size_t bytes = (size_t) min(group_rows, w.nrows - row0) * w.row_a;
        const char* src = cache.host[g] + offset;
        char* dst = const_cast<char*>(moe_expert(w, jb.w)) + offset;
        for (size_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) {
            const uint4 chunk = reinterpret_cast<const uint4*>(src)[i];
            reinterpret_cast<uint4*>(staged)[i] = chunk;
            reinterpret_cast<uint4*>(dst)[i] = chunk;
        }
        for (size_t i = bytes / 16 * 16 + threadIdx.x; i < bytes; i += blockDim.x)
            dst[i] = staged[i] = src[i];
        __syncthreads();
    }
    static_assert(G == 16 || G == 32, "IQ lanes per row");
    const int row = row_block * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const char* wr = missing ? reinterpret_cast<const char*>(staged) + (threadIdx.x / G) * w.row_a
                             : moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
    // A 16-lane row keeps both halves of the original 32-lane sums separately.
    // Join them before the remaining shuffles to preserve the reduction order.
    constexpr int PARTS = 32 / G;
    float tmp[PARTS][TT] = {};
    if constexpr (TA == GGML_TYPE_IQ2_XXS || TA == GGML_TYPE_IQ2_S || TA == GGML_TYPE_IQ3_XXS || TA == GGML_TYPE_IQ3_S) {
        constexpr int qk = ggml_cuda_type_traits<TA>::qk, vdr = vdr_of<TA>();
        constexpr int per_block = ggml_cuda_type_traits<TA>::qi / vdr;
        static_assert(G >= per_block && G % per_block == 0, "lanes per weight block");
        const int gl = threadIdx.x % G, kqs = vdr * (gl % per_block);
        if (live)
#pragma unroll
            for (int p = 0; p < PARTS; ++p) {
#pragma unroll 8
                for (int kbx = (gl + p * G) / per_block; kbx < w.ncols / qk; kbx += 32 / per_block) {
                    const IqDot32 decoded = iq_decode32<TA, PACKED_SIGNS>(wr, kbx, kqs);
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

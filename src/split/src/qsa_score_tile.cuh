// Long-prompt QSA scores with a 4-query by 256-key tile. The two FP16
// parts of each query and all FP32 WMMA additions stay in the reference order.
// Fewer query rows reduce LDS from 44 KiB to 18.5 KiB. Every query still
// scores all visible keys; selection and chronological ties are unchanged.
// Included beside qsa.cuh inside the kernels' anonymous namespace.
#pragma once

template<bool QuerySplit = false>
__global__ void __launch_bounds__(256) k_idx_score_tc_wide(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                      float* __restrict__ scores, int max_pools, const int* pos0p, int T,
                                                      int card = 0, unsigned* generation = nullptr) {
    if constexpr (QuerySplit) {
        // Increment exactly once, including when block zero belongs to the
        // peer. Subsequent kernels on this stream observe the completed tick.
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            const unsigned old = *generation;
            *generation = old == 0xfffffffeu ? 1u : old + 1u;
        }
        if (blockIdx.y % 2 != card) return;
    }
    constexpr int QSC_ROWS = 4, QSC_KEYS = 256, PerWave = 2, KeyWaves = 8;
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
    // One 16-head query tile against 16 key tiles; each wave owns two key tiles.
    const int rt = warp / KeyWaves;
    for (int kk = 0; kk < PerWave; ++kk) {
        const int kt = (warp % KeyWaves) * PerWave + kk, jt = j0 + kt * 16;
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

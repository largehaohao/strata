// Prompt HC kernels, included inside the prompt kernel namespace.
#pragma once

__device__ __forceinline__ float pf_sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float pf_wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float pf_block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = pf_wsum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < nw; ++w) r += sh[w];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- hyper-connection read, batched
// row t: R' = R + bo * 2 sigmoid(inj / HC) per stream (written back), rs[t][c] = 1 / sqrt(mean R'^2 + eps), and
// xn = R' * w_norm * rs as f16 (the down mix's input); with w_inj (bf16 rows of D) also inject_out[t][r] = w_inj[r] . xn
// from the same f32 xn while it is in registers (k_hcb_mixinj then needs no second pass over R)
constexpr int HCB_THREADS = 256, HCB_PER = N / HCB_THREADS;
static_assert(N % HCB_THREADS == 0, "stream elements per thread");
__global__ void __launch_bounds__(HCB_THREADS) k_hcb_prep(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                                          const float* __restrict__ wn, float* __restrict__ rs, __half* __restrict__ xh,
                                                          float eps, const uint16_t* __restrict__ w_inj, float* __restrict__ inject_out) {
    const int t = blockIdx.x, i0 = threadIdx.x;
    __shared__ float sh[32];
    __shared__ float shi[HC][HCB_THREADS / 32];
    float acc[HC] = {};
    for (int c = 0; c < HC; ++c) {
        float v[HCB_PER];
        float ss = 0.f;
        const float g = bo ? 2.0f * pf_sigm(inj[t * HC + c] / (float) HC) : 0.f;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            float r = R[(size_t) t * D + c * N + i];
            if (bo) {
                r = fmaf(bo[(size_t) t * N + i], g, r);
                R[(size_t) t * D + c * N + i] = r;
            }
            v[k] = r;
            ss += r * r;
        }
        const float r_s = rsqrtf(pf_block_sum(ss, sh) / (float) N + eps);
        if (i0 == 0) rs[t * HC + c] = r_s;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            const float xn = v[k] * wn[c * N + i] * r_s;
            xh[(size_t) t * D + c * N + i] = __float2half(xn);
            if (w_inj)
#pragma unroll
                for (int r = 0; r < HC; ++r) acc[r] = fmaf(__uint_as_float((unsigned) w_inj[(size_t) r * D + c * N + i] << 16), xn, acc[r]);
        }
    }
    if (!w_inj) return;
    const int lane = i0 & 31, warp = i0 >> 5;
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        const float s = pf_wsum(acc[r]);
        if (lane == 0) shi[r][warp] = s;
    }
    __syncthreads();
    if (i0 < HC) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < HCB_THREADS / 32; ++w) s += shi[i0][w];
        inject_out[t * HC + i0] = s;
    }
}

// Keep the four residual streams in registers and reduce their norms together.
// Each sum and inject dot product retains the reference kernel's exact order.
__global__ void __launch_bounds__(HCB_THREADS) k_hcb_prep_streams(float* __restrict__ R, const float* __restrict__ bo,
    const float* __restrict__ inj, const float* __restrict__ wn, float* __restrict__ rs, __half* __restrict__ xh,
    float eps, const uint16_t* __restrict__ w_inj, float* __restrict__ inject_out) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float partial[HC][HCB_THREADS / 32];
    float v[HC][HCB_PER], residual[HCB_PER], acc[HC] = {};
    if (bo) {
#pragma unroll
        for (int k=0;k<HCB_PER;++k) residual[k] = bo[(size_t)t*N + tid + k*HCB_THREADS];
    }
#pragma unroll
    for (int c=0;c<HC;++c) {
        const float gate = bo ? 2.f * pf_sigm(inj[t*HC+c] / (float)HC) : 0.f;
        float sum = 0.f;
#pragma unroll
        for (int k=0;k<HCB_PER;++k) {
            const size_t at = (size_t)t*D + c*N + tid + k*HCB_THREADS;
            float r = R[at];
            if (bo) { r = fmaf(residual[k], gate, r); R[at] = r; }
            v[c][k] = r;
            sum += r*r;
        }
        const float part = pf_wsum(sum);
        if (lane==0) partial[c][warp] = part;
    }
    __syncthreads();
#pragma unroll
    for (int c=0;c<HC;++c) {
        float sum = 0.f;
#pragma unroll
        for (int w=0;w<HCB_THREADS/32;++w) sum += partial[c][w];
        const float scale = rsqrtf(sum / (float)N + eps);
        if (tid==0) rs[t*HC+c] = scale;
#pragma unroll
        for (int k=0;k<HCB_PER;++k) {
            const int i = tid + k*HCB_THREADS;
            const float xn = v[c][k] * wn[c*N+i] * scale;
            xh[(size_t)t*D + c*N+i] = __float2half(xn);
            if (w_inj)
#pragma unroll
                for (int r=0;r<HC;++r)
                    acc[r] = fmaf(__uint_as_float((unsigned)w_inj[(size_t)r*D+c*N+i] << 16), xn, acc[r]);
        }
    }
    if (!w_inj) return;
    __syncthreads(); // all norm reads complete before the inject reduction reuses partial
#pragma unroll
    for (int r=0;r<HC;++r) {
        const float sum = pf_wsum(acc[r]);
        if (lane==0) partial[r][warp] = sum;
    }
    __syncthreads();
    if (tid<HC) {
        float sum=0.f;
#pragma unroll
        for (int w=0;w<HCB_THREADS/32;++w) sum += partial[tid][w];
        inject_out[t*HC+tid] = sum;
    }
}

// per row t: mixed[t][i] = mean_c xn_c[i] * sigmoid(up_c[i]), then (with w_inj: bf16 rows of D) inject[t][r] =
// w_inj[r] . xn[t]: warps 4 r .. 4 r + 3 take a quarter of D each, their sums added in order. xn = R wn rs,
// recomputed with k_hcb_prep's operations (so its values) rather than kept as a copy
constexpr int HCM_THREADS = 512, HCM_WPR = HCM_THREADS / 32 / HC;   // warps per inject row
__global__ void __launch_bounds__(HCM_THREADS) k_hcb_mixinj(const float* __restrict__ R, const float* __restrict__ wn,
                                                            const float* __restrict__ rs, const float* __restrict__ up,
                                                            float* __restrict__ mixed, __half* __restrict__ mixh,
                                                            const uint16_t* __restrict__ w_inj,
                                                            float* __restrict__ inj) {
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* Rt = R + (size_t) t * D;
    float r_s[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) r_s[c] = rs[t * HC + c];
    for (int i = tid; i < N; i += HCM_THREADS) {
        float g[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) g[c] = pf_sigm(up[(size_t) t * D + c * N + i]) * (Rt[c * N + i] * wn[c * N + i] * r_s[c]);
        const float m = (g[0] + g[1] + g[2] + g[3]) / (float) HC;
        mixed[(size_t) t * N + i] = m;
        mixh[(size_t) t * N + i] = __float2half(m);
    }
    if (!w_inj) return;
    __shared__ float part[HC][HCM_WPR];
    const int r = warp / HCM_WPR, q = warp % HCM_WPR;
    constexpr int QLEN = D / HCM_WPR;
    float acc = 0.f;
    for (int i = q * QLEN + lane; i < (q + 1) * QLEN; i += 32)
        acc += __uint_as_float((unsigned) w_inj[(size_t) r * D + i] << 16) * (Rt[i] * wn[i] * r_s[i / N]);
    acc = pf_wsum(acc);
    if (lane == 0) part[r][q] = acc;
    __syncthreads();
    if (tid < HC) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < HCM_WPR; ++k) s += part[tid][k];
        inj[t * HC + tid] = s;
    }
}

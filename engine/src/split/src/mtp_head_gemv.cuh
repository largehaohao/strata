#pragma once
__global__ void __launch_bounds__(128) k_draft_head_full(
        const char* __restrict__ w, size_t row_bytes, int rows, int cols,
        const block_q8_1* __restrict__ y, float* __restrict__ out) {
    constexpr int G=8;
    constexpr int per=qtraits<T_Q4_0>::qi/vdr_of<T_Q4_0>();
    const int lane=threadIdx.x%G, row=blockIdx.x*16+threadIdx.x/G;
    float sum=0.f;
    if(row<rows) for(int b=lane/per;b<cols/32;b+=G/per)
        sum+=vdot<T_Q4_0>(w+(size_t)row*row_bytes,y+b,b,vdr_of<T_Q4_0>()*(lane%per));
#pragma unroll
    for(int d=G/2;d;d>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
    if(!lane && row<rows)out[row]=sum;
}
// Speculative Q4 head only. Surviving rows complete
// the original G=8 accumulator in its original order. Main logits never use it.
template<int KEEP, int QUARTERS>
__global__ void __launch_bounds__(128) k_draft_head_prune(
        const char* __restrict__ w, size_t row_bytes, int rows, int cols,
        const block_q8_1* __restrict__ y, float* __restrict__ out) {
    constexpr int G = 8, R = 16;
    constexpr int per = qtraits<T_Q4_0>::qi / vdr_of<T_Q4_0>();
    constexpr int step = G / per;
    const int lane = threadIdx.x % G, local = threadIdx.x / G;
    const int row = blockIdx.x * R + local, nb = cols / 32;
    const int stop = (nb * QUARTERS / 4 / step) * step;
    const char* wr = w + (size_t)(row < rows ? row : 0) * row_bytes;
    float sum = 0.f;
    if (row < rows)
        for (int b = lane / per; b < stop; b += step)
            sum += vdot<T_Q4_0>(wr, y + b, b, vdr_of<T_Q4_0>() * (lane % per));
    float score = sum;
#pragma unroll
    for (int d = G / 2; d; d >>= 1) score += __shfl_xor_sync(0xffffffffu, score, d);
    __shared__ float partial[R];
    if (!lane) partial[local] = row < rows ? score : -3.4028234663852886e38f;
    __syncthreads();
    int rank = 0;
    const float mine = partial[local];
#pragma unroll
    for (int i = 0; i < R; ++i)
        rank += partial[i] > mine || (partial[i] == mine && i < local);
    const bool keep = row < rows && rank < KEEP;
    if constexpr (KEEP == 4) {
        // Preserve the eight independent accumulators for each surviving row.
        // With unordered scores the original rank rule can create duplicate ranks.
        if (!__syncthreads_or(row < rows && !isfinite(score))) {
            __shared__ float sums[KEEP][G];
            __shared__ int selected[KEEP];
            if (keep) {
                sums[rank][lane] = sum;
                if (!lane) selected[rank] = row;
            } else if (!lane && row < rows) {
                out[row] = -3.4028234663852886e38f;
            }
            __syncthreads();
            if (threadIdx.x >= KEEP * G) return;
            const int group = threadIdx.x / G;
            const bool valid = group < rows - (int)blockIdx.x * R;
            const int target = valid ? selected[group] : 0;
            sum = valid ? sums[group][lane] : 0.f;
            if (valid)
                for (int b = stop + lane / per; b < nb; b += step)
                    sum += vdot<T_Q4_0>(w + (size_t)target * row_bytes, y + b, b, vdr_of<T_Q4_0>() * (lane % per));
#pragma unroll
            for (int d = G / 2; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
            if (!lane && valid) out[target] = sum;
            return;
        }
    }
    if (keep)
        for (int b = stop + lane / per; b < nb; b += step)
            sum += vdot<T_Q4_0>(wr, y + b, b, vdr_of<T_Q4_0>() * (lane % per));
#pragma unroll
    for (int d = G / 2; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane && row < rows) out[row] = keep ? sum : -3.4028234663852886e38f;
}

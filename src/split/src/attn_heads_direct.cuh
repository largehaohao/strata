// RDNA3 prompt attention: keep Q/K fragments in registers and stage transposed
// int8 V codes. The eight 32-dimension score partials, FP16 rounding and softmax
// order are the same as k_pf_attn_head. Only the storage and wave ownership of
// the output change; each wave owns one KV scale group.
#if FNX_HIP
__global__ void __launch_bounds__(256) k_pf_attn_head_direct(const float* __restrict__ qg,
    const __half* __restrict__ kc, const __half* __restrict__ vc, KvPages g,
    const int* __restrict__ pos0p, const int* __restrict__ cells, const int* __restrict__ n_cells,
    int owner, int C, int r0, float* __restrict__ part) {
#if FNX_WMMA_GFX11
    using H = nvcuda::wmma::fnx_h16;
    using F = nvcuda::wmma::fnx_f8;
    static_assert(HD == 256 && KV8_GROUP == 32 && ATT_QPK == 12, "direct prompt geometry");
    __shared__ float scores[8][ATT_QPK][33];
    __shared__ __align__(16) __half prob[16][40];
    __shared__ __align__(16) int8_t value[8][32][48]; // [wave][dimension][position], 16-byte aligned rows
    __shared__ float vscale[8][32];
    __shared__ float mx[16], sum[16], alpha[16];
    const int t = r0 + blockIdx.x, card = blockIdx.y, tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, col = lane & 15, half = lane >> 4;
    if (cells) cells += (size_t) card * C * QSA_MAXSEL;
    if (n_cells) n_cells += card * C;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    H q[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int i = 0; i < 16; ++i)
            q[k][i] = col < ATT_QPK ? (_Float16) qg[((size_t) t * ATT_OWN + col) * 2 * HD + wave * 32 + k * 16 + i] : (_Float16) 0;
    F output[2] = {};
    if (tid < 16) { mx[tid] = -INFINITY; sum[tid] = 0.f; }
    const float scale = rsqrtf((float) HD);
    for (int c0 = 0; c0 < ctx; c0 += 32) {
        const int np = min(32, ctx - c0);
        uint4 bytes[2] = {};
        float vd = 0.f;
        if (lane < np) {
            const int pos = list ? list[c0 + lane] : c0 + lane;
            const size_t at = kvq_row(g, pos) + card * HD + wave * 32;
            const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(vc) + at);
            bytes[0] = src[0]; bytes[1] = src[1];
            vd = __half2float(vc[kvq_scale(g, pos) + card * (HD / KV8_GROUP) + wave]);
        }
        vscale[wave][lane] = vd;
        const uint32_t* vb = reinterpret_cast<const uint32_t*>(bytes);
#pragma unroll
        for (int d = 0; d < 32; ++d) value[wave][d][lane] = (int8_t) (vb[d / 4] >> (8 * (d % 4)));
        // Exactly the reference's two K steps per wave and two position tiles.
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const int p = tile * 16 + col;
            F dot = {};
            size_t at = 0;
            float kd = 0.f;
            if (p < np) {
                const int pos = list ? list[c0 + p] : c0 + p;
                at = kvq_row(g, pos) + card * HD + wave * 32;
                kd = __half2float(kc[kvq_scale(g, pos) + card * (HD / KV8_GROUP) + wave]);
            }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                uint4 raw = {};
                if (p < np) raw = *reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(kc) + at + k * 16);
                const uint32_t* words = reinterpret_cast<const uint32_t*>(&raw);
                H key;
#pragma unroll
                for (int i = 0; i < 16; ++i) key[i] = (_Float16) ((int8_t) (words[i / 4] >> (8 * (i % 4))) * kd);
                dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q[k], key, dot);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (2 * i + half < ATT_QPK) scores[wave][2 * i + half][p] = dot[i];
        }
        __syncthreads();
        const int r = tid / 16, c = tid % 16;
        if (r < ATT_QPK) {
            float sv[2], maximum = -INFINITY;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < 8; ++w) s += scores[w][r][u * 16 + c];
                sv[u] = c + 16 * u < np ? s * scale : -INFINITY;
                maximum = fmaxf(maximum, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, off, 16));
            const float old = mx[r], next = fmaxf(old, maximum);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const float p = sv[u] == -INFINITY ? 0.f : __expf(sv[u] - next);
                ps += p;
                prob[r][u * 16 + c] = __float2half_rn(p);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 16);
            if (c == 0) {
                const float a = old == -INFINITY ? 0.f : __expf(old - next);
                alpha[r] = a; sum[r] = sum[r] * a + ps; mx[r] = next;
            }
        }
        __syncthreads();
        F pv[2] = {};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            H p = {};
            if (col < ATT_QPK) p = nvcuda::wmma::fnx_ld_ab(prob[col] + k * 16);
#pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const uint4 raw = *reinterpret_cast<const uint4*>(&value[wave][tile * 16 + col][k * 16]);
                const uint32_t* words = reinterpret_cast<const uint32_t*>(&raw);
                H v;
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    v[i] = (_Float16) ((int8_t) (words[i / 4] >> (8 * (i % 4))) * vscale[wave][k * 16 + i]);
                pv[tile] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p, v, pv[tile]);
            }
        }
#pragma unroll
        for (int tile = 0; tile < 2; ++tile)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (2 * i + half < ATT_QPK) output[tile][i] = fmaf(output[tile][i], alpha[2 * i + half], pv[tile][i]);
        // Each wave owns its V codes/scales. Cross-wave scratch is read again
        // only after the next iteration's barrier, before its softmax writes.
    }
    __syncthreads(); // also orders max/sum initialization for an empty partition
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int h = 2 * i + half;
        if (h >= ATT_QPK) continue;
        float* dst = part + (((size_t) card * C + t) * ATT_OWN + h) * AP_W;
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) dst[wave * 32 + tile * 16 + col] = output[tile][i];
        if (wave == 0 && col == 0) { dst[HD] = mx[h]; dst[HD + 1] = sum[h]; }
    }
#else
    __builtin_trap();
#endif
}
#endif

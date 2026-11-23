// The prompt path's weight tiles (moe_mma.cuh) for the 4-bit codebook and i-quant routed experts of the IQ3_S set: Q4_0 (what the Q2_0
// experts are repacked to at load, tensor_convert.hpp), IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS and IQ2_S. Each value is ggml's own dequantization
// (ggml-cuda/dequantize.cuh dequantize_*: the f32 product d * scale * grid * sign) rounded once to f16, so the f16 operands are the ones
// llama.cpp's cuBLAS path would have made. A thread takes one 32-value sub-block of a row per k-step (MM_TPR = 2 of them side by side
// in the step's 64 columns) like the Q5_1 and Q8_0 tiles; the 256-value types find theirs at (k0 % 256) / 32 + u in the block.
// First version: the raw bytes are read with 16- and 32-bit loads (the blocks of 110, 98 and 82 bytes are only 2-byte aligned) and the
// grids come from ggml's tables in global memory (cached); not yet tuned for speed.

__device__ __forceinline__ uint32_t mq_ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t mq_ld32(const uint8_t* p) { return mq_ld16(p) | (mq_ld16(p + 2) << 16); }
__device__ __forceinline__ float mq_half(uint32_t h) { return __half2float(__ushort_as_half((uint16_t) h)); }
// 32 f32 values to f16 pairs and into the tile row
__device__ __forceinline__ void mq_put32(__half* dst, const float (&f)[32]) {
    __half2 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    mm_store8(dst, y);
    mm_store8(dst + 16, y + 8);
}
__device__ __forceinline__ void mq_zero32(__half* dst) {
    float z[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) z[i] = 0.f;
    mq_put32(dst, z);
}

// Unsloth Dynamic v3.1 keeps IQ2_XXS gate/up experts and a Q5_0 down layer.
// Read their original blocks; the prompt operands are ggml's dequantized values
// rounded once to FP16, as for the other expert formats below.
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
__device__ __forceinline__ void mq_iq_packed4(__half2*, uint32_t, __half2, __half2, uint32_t, int);

__device__ __forceinline__ uint32_t mm_spread4_fast(uint32_t t) {
    return (((t & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}

__device__ __forceinline__ uint32_t mm_iq4nl_u8_4(uint32_t x) {
    constexpr uint32_t KV0 = 0x3F2D1801u, KV1 = 0x766A5D4Fu;
    constexpr uint32_t KV2 = 0xA6998D81u, KV3 = 0xF1D9C5B5u;
    const uint32_t p = x & 0x07070707u;
    const uint32_t lo8 = __builtin_amdgcn_perm(KV1, KV0, p);
    const uint32_t hi8 = __builtin_amdgcn_perm(KV3, KV2, p);
    return __builtin_amdgcn_perm(hi8, lo8, ((x >> 1) & 0x04040404u) | 0x03020100u);
}

__device__ __forceinline__ void mm_hgrid_init_ssign(uint2* ssign) {
    if (threadIdx.x < 16) {
        const uint32_t i = threadIdx.x;
        ssign[i] = make_uint2(((i & 1u) << 15) | (((i >> 1) & 1u) << 31),
                              (((i >> 2) & 1u) << 15) | (((i >> 3) & 1u) << 31));
    }
}

__device__ __forceinline__ void mq_iq_hgrid4(__half2* out, uint2 g, __half2 sh, __half2 dh, uint2 sg) {
    const __half2 a = __hmul2(__hmul2(__builtin_bit_cast(__half2, g.x), sh), dh);
    const __half2 b = __hmul2(__hmul2(__builtin_bit_cast(__half2, g.y), sh), dh);
    out[0] = __builtin_bit_cast(__half2, __builtin_bit_cast(uint32_t, a) ^ sg.x);
    out[1] = __builtin_bit_cast(__half2, __builtin_bit_cast(uint32_t, b) ^ sg.y);
}

template <> struct MmHgridInfo<T_IQ3_XXS> {
    static constexpr bool has_grid = true;
    static constexpr size_t grid_bytes = 256 * sizeof(uint2);
    __device__ __forceinline__ static void init(void* sgrid_raw, uint2* ssign) {
        mm_hgrid_init_ssign(ssign);
        uint2* sg = reinterpret_cast<uint2*>(sgrid_raw);
        const __half2 k = mm_k1024();
        for (int i = threadIdx.x; i < 256; i += MM_THREADS) {
            __half2 a, b;
            mm_bytes_h2(iq3xxs_grid[i], k, a, b);
            sg[i] = make_uint2(__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b));
        }
    }
};
template <> struct MmHgridInfo<T_IQ3_S> {
    static constexpr bool has_grid = true;
    static constexpr size_t grid_bytes = 512 * sizeof(uint2);
    __device__ __forceinline__ static void init(void* sgrid_raw, uint2* ssign) {
        mm_hgrid_init_ssign(ssign);
        uint2* sg = reinterpret_cast<uint2*>(sgrid_raw);
        const __half2 k = mm_k1024();
        for (int i = threadIdx.x; i < 512; i += MM_THREADS) {
            __half2 a, b;
            mm_bytes_h2(iq3s_grid[i], k, a, b);
            sg[i] = make_uint2(__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b));
        }
    }
};
template <> struct MmHgridInfo<T_IQ2_XXS> {
    static constexpr bool has_grid = true;
    static constexpr size_t grid_bytes = 256 * sizeof(uint4);
    __device__ __forceinline__ static void init(void* sgrid_raw, uint2* ssign) {
        mm_hgrid_init_ssign(ssign);
        uint4* sg = reinterpret_cast<uint4*>(sgrid_raw);
        const __half2 k = mm_k1024();
        for (int i = threadIdx.x; i < 256; i += MM_THREADS) {
            const uint64_t v = iq2xxs_grid[i];
            __half2 a, b, c, d;
            mm_bytes_h2((uint32_t) v, k, a, b);
            mm_bytes_h2((uint32_t) (v >> 32), k, c, d);
            sg[i] = make_uint4(__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b),
                               __builtin_bit_cast(uint32_t, c), __builtin_bit_cast(uint32_t, d));
        }
    }
};
#endif
template <> struct MmTile<T_IQ2_XXS> {
    uint32_t codes, aux;
    uint16_t dh;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        item = u * 32 < kc;
        live = item && row0 + r < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) (row0 + r) * row_a + (size_t) (k0 / 256) * 66;
            const int ib = (k0 % 256) / 32 + u;
            dh = (uint16_t) mq_ld16(b);
            codes = mq_ld32(b + 2 + ib * 8);
            aux = mq_ld32(b + 6 + ib * 8);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        __half* dst = sw + threadIdx.x / MM_TPR * MM_LD + threadIdx.x % MM_TPR * 32;
        if (!live) { mq_zero32(dst); return; }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        const __half2 d = __half2half2(__ushort_as_half(dh));
        const __half2 scale = __float2half2_rn((float) (1 + 2 * (aux >> 28)) * 0.125f);
        __half2 h[16];
#else
        const float d = mq_half(dh) * (0.5f + (aux >> 28)) * 0.25f;
        float f[32];
#endif
#pragma unroll
        for (int l = 0; l < 4; ++l) {
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
            const uint64_t grid = iq2xxs_grid[(codes >> (8 * l)) & 255];
#else
            const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + ((codes >> (8 * l)) & 255));
#endif
            const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
            mq_iq_packed4(h + 4 * l, (uint32_t) grid, scale, d, signs, 0);
            mq_iq_packed4(h + 4 * l + 2, (uint32_t) (grid >> 32), scale, d, signs, 4);
#else
#pragma unroll
            for (int j = 0; j < 8; ++j)
                f[8 * l + j] = d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
#endif
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        mm_store8(dst, h);
        mm_store8(dst + 16, h + 8);
#else
        mq_put32(dst, f);
#endif
    }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
    __device__ __forceinline__ void put_hgrid(int, int, __half* sw, const void* sgrid, const uint2* ssign) const {
        if (!item) return;
        __half* dst = sw + threadIdx.x / MM_TPR * MM_LD + threadIdx.x % MM_TPR * 32;
        if (!live) { mq_zero32(dst); return; }
        const uint4* sg_tbl = reinterpret_cast<const uint4*>(sgrid);
        const __half2 dh2 = __half2half2(__ushort_as_half(dh));
        const __half2 sh = __float2half2_rn((float) (1 + 2 * (aux >> 28)) * 0.125f);
        __half2 h[16];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint4 gv = sg_tbl[(codes >> (8 * l)) & 255];
            const uint32_t s7 = (aux >> (7 * l)) & 127u;
            const uint32_t sb = s7 | ((__popc(s7) & 1u) << 7);
            const uint2 s0 = ssign[sb & 15u], s1 = ssign[sb >> 4];
            mq_iq_hgrid4(h + 4 * l, make_uint2(gv.x, gv.y), sh, dh2, s0);
            mq_iq_hgrid4(h + 4 * l + 2, make_uint2(gv.z, gv.w), sh, dh2, s1);
        }
        mm_store8(dst, h);
        mm_store8(dst + 16, h + 8);
    }
#endif
};

template <> struct MmTile<T_Q5_0> {
    uint16_t dh;
    uint32_t high, q[4];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        item = u * 32 < kc;
        live = item && row0 + r < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) (row0 + r) * row_a + (size_t) (k0 / 32 + u) * 22;
            dh = (uint16_t) mq_ld16(b);
            high = mq_ld32(b + 2);
#pragma unroll
            for (int l = 0; l < 4; ++l) q[l] = mq_ld32(b + 6 + 4 * l);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        __half* dst = sw + threadIdx.x / MM_TPR * MM_LD + threadIdx.x % MM_TPR * 32;
        if (!live) { mq_zero32(dst); return; }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        const __half2 d = __half2half2(__ushort_as_half(dh));
        const __half2 bias = __half2half2(__ushort_as_half(0x6410));   // 1040.0h = 1024 + 16
        const __half2 zero = __float2half2_rn(0.f);
        __half2 lo[8], hi[8];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t a = (q[w] & 0x0F0F0F0Fu) | mm_spread4_fast(high >> (4 * w));
            const uint32_t b = ((q[w] >> 4) & 0x0F0F0F0Fu) | mm_spread4_fast(high >> (16 + 4 * w));
            __half2 x0, x1;
            mm_bytes_h2(a, bias, x0, x1);
            lo[2 * w] = __hfma2(x0, d, zero);
            lo[2 * w + 1] = __hfma2(x1, d, zero);
            mm_bytes_h2(b, bias, x0, x1);
            hi[2 * w] = __hfma2(x0, d, zero);
            hi[2 * w + 1] = __hfma2(x1, d, zero);
        }
        mm_store8(dst, lo);
        mm_store8(dst + 16, hi);
#else
        float f[32];
        const float d = mq_half(dh);
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            const unsigned byte = (q[l / 4] >> (8 * (l % 4))) & 255;
            f[l] = d * ((int) ((byte & 15) | (((high >> l) & 1) << 4)) - 16);
            f[l + 16] = d * ((int) ((byte >> 4) | (((high >> (l + 16)) & 1) << 4)) - 16);
        }
        mq_put32(dst, f);
#endif
    }
};

// Q4_0 (value (nibble - 8) * d) and IQ4_NL (kvalues_iq4nl[nibble] * d): 18 bytes, low nibbles are values 0..15, high 16..31
template <bool NL>
struct MmTile4 {
    uint16_t v[9];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 32 + u) * 18;
#pragma unroll
            for (int l = 0; l < 9; ++l) v[l] = (uint16_t) mq_ld16(b + 2 * l);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        if constexpr (!NL) {
            // Every (nibble - 8) is exact in half. A single half FMA rounds
            // the exact d*q product once, with the reference's canonical +0.
            const __half2 d = __half2half2(__ushort_as_half(v[0]));
            const __half2 bias = __half2half2(__ushort_as_half(0x6408));
            const __half2 zero = __float2half2_rn(0.f);
            __half2 lo[8], hi[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t q = (uint32_t)v[1 + 2 * w] | ((uint32_t)v[2 + 2 * w] << 16);
                __half2 a, b;
                mm_bytes_h2(q & 0x0f0f0f0fu, bias, a, b);
                lo[2 * w] = __hfma2(a, d, zero);
                lo[2 * w + 1] = __hfma2(b, d, zero);
                mm_bytes_h2((q >> 4) & 0x0f0f0f0fu, bias, a, b);
                hi[2 * w] = __hfma2(a, d, zero);
                hi[2 * w + 1] = __hfma2(b, d, zero);
            }
            mm_store8(sw + r * MM_LD + u * 32, lo);
            mm_store8(sw + r * MM_LD + u * 32 + 16, hi);
            return;
        } else {
            const __half2 d = __half2half2(__ushort_as_half(v[0]));
            const __half2 k1152 = __half2half2(__ushort_as_half(0x6480));   // 1024 + 128
            __half2 lo[8], hi[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t q = (uint32_t) v[1 + 2 * w] | ((uint32_t) v[2 + 2 * w] << 16);
                __half2 a, b;
                mm_bytes_h2(mm_iq4nl_u8_4(q), k1152, a, b);
                lo[2 * w] = __hmul2(a, d);
                lo[2 * w + 1] = __hmul2(b, d);
                mm_bytes_h2(mm_iq4nl_u8_4(q >> 4), k1152, a, b);
                hi[2 * w] = __hmul2(a, d);
                hi[2 * w + 1] = __hmul2(b, d);
            }
            mm_store8(sw + r * MM_LD + u * 32, lo);
            mm_store8(sw + r * MM_LD + u * 32 + 16, hi);
            return;
        }
#else
        const float d = mq_half(v[0]);
        float f[32];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t q = (uint32_t) v[1 + 2 * w] | ((uint32_t) v[2 + 2 * w] << 16);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int lo = (q >> (8 * k)) & 0xF, hi = (q >> (8 * k + 4)) & 0xF;
                f[4 * w + k] = d * (float) (NL ? (int) kvalues_iq4nl[lo] : lo - 8);
                f[16 + 4 * w + k] = d * (float) (NL ? (int) kvalues_iq4nl[hi] : hi - 8);
            }
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
#endif
    }
};
template <> struct MmTile<T_Q4_0> : MmTile4<false> {};
template <> struct MmTile<T_IQ4_NL> : MmTile4<true> {};

// Packed Q2_0 tile: 32-value half from 18-byte 64-value block, (code - 1) * d.
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
template <> struct MmTile<T_Q2_0> {
    uint32_t w[2];
    uint16_t dh;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const int half = k0 / 32 + u;
            const auto* b = reinterpret_cast<const uint8_t*>(we) + (size_t) row * row_a + (size_t) (half / 2) * 18;
            const auto* q = b + 2 + (half % 2) * 8;
            dh = (uint16_t) mq_ld16(b);
            w[0] = mq_ld32(q);
            w[1] = mq_ld32(q + 4);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        __half* dst = sw + r * MM_LD + u * 32;
        if (!live) {
            mq_zero32(dst);
            return;
        }
        const __half2 d = __half2half2(__ushort_as_half(dh));
        const __half2 k256 = __half2half2(__ushort_as_half(0x1C00));   // 1/256
        const __half2 m5 = __half2half2(__ushort_as_half(0xC500));     // -5.0
        const __half2 zero = __float2half2_rn(0.f);
        __half2 out[16];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t x = w[h];
            const uint32_t A = (x & 0x03030303u) | 0x64646464u;
            const uint32_t B = ((x >> 2) & 0x03030303u) | 0x64646464u;
            const uint32_t C = ((x >> 4) & 0x03030303u) | 0x64646464u;
            const uint32_t D = ((x >> 6) & 0x03030303u) | 0x64646464u;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t sel = 0x0Cu | ((uint32_t) b << 8) | (0x0Cu << 16) | ((uint32_t) (4 + b) << 24);
                const __half2 p0 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(B, A, sel));
                const __half2 p1 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(D, C, sel));
                out[8 * h + 2 * b] = __hfma2(__hfma2(p0, k256, m5), d, zero);
                out[8 * h + 2 * b + 1] = __hfma2(__hfma2(p1, k256, m5), d, zero);
            }
        }
        mm_store8(dst, out);
        mm_store8(dst + 16, out + 8);
    }
};
#else
// Expand only the fetched 32-value half of a packed Q2_0 block into the
// original Q4_0 tile registers. Inherited put() and WMMA arithmetic are unchanged.
template <> struct MmTile<T_Q2_0> : MmTile4<false> {
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const int half = k0 / 32 + u;
            const auto* b = reinterpret_cast<const uint8_t*>(we) + (size_t) row * row_a + (size_t) (half / 2) * 18;
            const auto* q = b + 2 + (half % 2) * 8;
            v[0] = (uint16_t) mq_ld16(b);
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const int shift = 4 * (w % 2);
                const unsigned lo = (q[w / 2] >> shift) & 15u;
                const unsigned hi = (q[4 + w / 2] >> shift) & 15u;
                v[1 + w] = (uint16_t) (((lo & 3u) + 7u) | (((hi & 3u) + 7u) << 4) |
                    (((lo >> 2) + 7u) << 8) | (((hi >> 2) + 7u) << 12));
            }
        }
    }
};
#endif

// IQ4_XS: 136 bytes = d (f16), scales_h (u16), scales_l[4], qs[128]; sub-block ib has 16 qs bytes at 8 + 16 ib and a 6-bit scale
template <> struct MmTile<T_IQ4_XS> {
    uint32_t hd[2], q[4];
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 136;
            hd[0] = mq_ld32(b);
            hd[1] = mq_ld32(b + 4);
#pragma unroll
            for (int l = 0; l < 4; ++l) q[l] = mq_ld32(b + 8 + 16 * ib + 4 * l);
        }
    }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        __half* dst = sw + r * MM_LD + u * 32;
        if (!live) {
            mq_zero32(dst);
            return;
        }
        const uint32_t sh = hd[0] >> 16, sl = (hd[1] >> (8 * (ib / 2))) & 0xFF;
        const int ls = (int) (((sl >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4));
        const float d = mq_half(hd[0] & 0xFFFF) * (float) (ls - 32);
        const __half2 k1152 = __half2half2(__ushort_as_half(0x6480));
        __half2 lo[8], hi[8];
        auto scale_pair = [&](__half2 v) -> __half2 {
            const float2 f = __half22float2(v);
            return __floats2half2_rn(f.x * d, f.y * d);
        };
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            __half2 a, b;
            mm_bytes_h2(mm_iq4nl_u8_4(q[l]), k1152, a, b);
            lo[2 * l] = scale_pair(a);
            lo[2 * l + 1] = scale_pair(b);
            mm_bytes_h2(mm_iq4nl_u8_4(q[l] >> 4), k1152, a, b);
            hi[2 * l] = scale_pair(a);
            hi[2 * l + 1] = scale_pair(b);
        }
        mm_store8(dst, lo);
        mm_store8(dst + 16, hi);
    }
#else
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const uint32_t sh = hd[0] >> 16, sl = (hd[1] >> (8 * (ib / 2))) & 0xFF;
        const int ls = (int) (((sl >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4));
        const float d = mq_half(hd[0] & 0xFFFF) * (float) (ls - 32);
        float f[32];
#pragma unroll
        for (int l = 0; l < 4; ++l)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (q[l] >> (8 * k)) & 0xFF;
                f[4 * l + k] = d * (float) kvalues_iq4nl[byte & 0xF];
                f[16 + 4 * l + k] = d * (float) kvalues_iq4nl[byte >> 4];
            }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
#endif
};

__device__ __forceinline__ void mq_signed4(float* y, uint32_t grid, float d, uint32_t signs, int s0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) y[j] = d * (float) ((grid >> (8 * j)) & 0xFF) * ((signs & kmask_iq2xs[s0 + j]) ? -1.f : 1.f);
}

#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
// Apply the strata branch's direct-sign idea to TP's original weight values.
// Grid * odd scale is <=465 (IQ3_S), <=1922 (IQ3_XXS), or <=1333 (IQ2_S).
// These integers, including XXS's /4 and IQ2's /8, are exact in FP16.
// Multiply this factor by the original half d only once: the same rounding
// as the old exact FP32 product followed by conversion. Do not round d*scale.
__device__ __forceinline__ void mq_iq_packed4(__half2* out, uint32_t grid, __half2 scale, __half2 d,
                                               uint32_t signs, int s0) {
    __half2 a, b;
    mm_bytes_h2(grid, mm_k1024(), a, b);
    a = __hmul2(__hmul2(a, scale), d);
    b = __hmul2(__hmul2(b, scale), d);
    const uint32_t sa = ((signs >> s0) & 1u) << 15 | ((signs >> (s0 + 1)) & 1u) << 31;
    const uint32_t sb = ((signs >> (s0 + 2)) & 1u) << 15 | ((signs >> (s0 + 3)) & 1u) << 31;
    out[0] = __builtin_bit_cast(__half2, __builtin_bit_cast(uint32_t, a) ^ sa);
    out[1] = __builtin_bit_cast(__half2, __builtin_bit_cast(uint32_t, b) ^ sb);
}
#endif

// IQ3_S: 110 bytes = d, qs[64], qh[8], signs[32], scales[4]; sub-block ib: qs[8 ib..], qh[ib], signs[4 ib..], a 4-bit scale (1 + 2 s) d
template <> struct MmTile<T_IQ3_S> {
    uint32_t dq, qs[2], sg, qh, sc;   // d (low 16 bits of dq), 8 qs bytes, 4 sign bytes, qh byte, scales byte
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 110;
            dq = mq_ld16(b);
            qs[0] = mq_ld32(b + 2 + 8 * ib);
            qs[1] = mq_ld32(b + 6 + 8 * ib);
            qh = b[66 + ib];
            sg = mq_ld32(b + 74 + 4 * ib);
            sc = b[106 + ib / 2];
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const int scale = 1 + 2 * ((sc >> (4 * (ib % 2))) & 0xF);
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        const __half2 dh = __half2half2(__ushort_as_half((uint16_t) dq));
        const __half2 sh = __float2half2_rn((float) scale);
        __half2 h[16];
#else
        const float d = mq_half(dq) * (float) scale;
        float f[32];
#endif
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF, q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint32_t g1 = iq3s_grid[q0 | ((qh << (8 - 2 * il)) & 256)], g2 = iq3s_grid[q1 | ((qh << (7 - 2 * il)) & 256)];
            const uint32_t sb = (sg >> (8 * il)) & 0xFF;
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
            mq_iq_packed4(h + 4 * il, g1, sh, dh, sb, 0);
            mq_iq_packed4(h + 4 * il + 2, g2, sh, dh, sb, 4);
#else
            mq_signed4(f + 8 * il, g1, d, sb, 0);
            mq_signed4(f + 8 * il + 4, g2, d, sb, 4);
#endif
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        mm_store8(sw + r * MM_LD + u * 32, h);
        mm_store8(sw + r * MM_LD + u * 32 + 16, h + 8);
#else
        mq_put32(sw + r * MM_LD + u * 32, f);
#endif
    }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
    __device__ __forceinline__ void put_hgrid(int, int, __half* sw, const void* sgrid, const uint2* ssign) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const int scale = 1 + 2 * ((sc >> (4 * (ib % 2))) & 0xF);
        const uint2* sg_tbl = reinterpret_cast<const uint2*>(sgrid);
        const __half2 dh2 = __half2half2(__ushort_as_half((uint16_t) dq));
        const __half2 sh = __float2half2_rn((float) scale);
        __half2 h[16];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF;
            const uint32_t q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint2 g1 = sg_tbl[q0 | ((qh << (8 - 2 * il)) & 256)];
            const uint2 g2 = sg_tbl[q1 | ((qh << (7 - 2 * il)) & 256)];
            const uint32_t sb = (sg >> (8 * il)) & 0xFFu;
            mq_iq_hgrid4(h + 4 * il, g1, sh, dh2, ssign[sb & 15u]);
            mq_iq_hgrid4(h + 4 * il + 2, g2, sh, dh2, ssign[sb >> 4]);
        }
        mm_store8(sw + r * MM_LD + u * 32, h);
        mm_store8(sw + r * MM_LD + u * 32 + 16, h + 8);
    }
#endif
};

// IQ3_XXS: 98 bytes = d, qs[96]: 64 grid indices (8 per sub-block) then per sub-block a u32 of four 7-bit sign codes and a 4-bit scale at the top
template <> struct MmTile<T_IQ3_XXS> {
    uint32_t dq, qs[2], aux;
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 98;
            dq = mq_ld16(b);
            qs[0] = mq_ld32(b + 2 + 8 * ib);
            qs[1] = mq_ld32(b + 6 + 8 * ib);
            aux = mq_ld32(b + 66 + 4 * ib);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        const __half2 dh = __half2half2(__ushort_as_half((uint16_t) dq));
        const __half2 sh = __float2half2_rn((float) (1 + 2 * (aux >> 28)) * 0.25f);
        __half2 h[16];
#else
        const float d = mq_half(dq) * (0.5f + (float) (aux >> 28)) * 0.5f;
        float f[32];
#endif
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF, q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint32_t sb = ksigns_iq2xs[(aux >> (7 * il)) & 127];
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
            mq_iq_packed4(h + 4 * il, iq3xxs_grid[q0], sh, dh, sb, 0);
            mq_iq_packed4(h + 4 * il + 2, iq3xxs_grid[q1], sh, dh, sb, 4);
#else
            mq_signed4(f + 8 * il, iq3xxs_grid[q0], d, sb, 0);
            mq_signed4(f + 8 * il + 4, iq3xxs_grid[q1], d, sb, 4);
#endif
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        mm_store8(sw + r * MM_LD + u * 32, h);
        mm_store8(sw + r * MM_LD + u * 32 + 16, h + 8);
#else
        mq_put32(sw + r * MM_LD + u * 32, f);
#endif
    }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
    __device__ __forceinline__ void put_hgrid(int, int, __half* sw, const void* sgrid, const uint2* ssign) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const uint2* sg_tbl = reinterpret_cast<const uint2*>(sgrid);
        const __half2 dh2 = __half2half2(__ushort_as_half((uint16_t) dq));
        const __half2 sh = __float2half2_rn((float) (1 + 2 * (aux >> 28)) * 0.25f);
        __half2 h[16];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF;
            const uint32_t q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint32_t s7 = (aux >> (7 * il)) & 127u;
            const uint32_t sb = s7 | ((__popc(s7) & 1u) << 7);
            mq_iq_hgrid4(h + 4 * il, sg_tbl[q0], sh, dh2, ssign[sb & 15u]);
            mq_iq_hgrid4(h + 4 * il + 2, sg_tbl[q1], sh, dh2, ssign[sb >> 4]);
        }
        mm_store8(sw + r * MM_LD + u * 32, h);
        mm_store8(sw + r * MM_LD + u * 32 + 16, h + 8);
    }
#endif
};

// IQ2_S: 82 bytes = d, qs[32] (grid index low bytes, 4 per sub-block), signs[32] (4 per sub-block), qh[8], scales[8]; 10-bit grid index of 8 values
template <> struct MmTile<T_IQ2_S> {
    uint32_t dq, qs, sg, qh, sc;
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 82;
            dq = mq_ld16(b);
            qs = mq_ld32(b + 2 + 4 * ib);
            sg = mq_ld32(b + 34 + 4 * ib);
            qh = b[66 + ib];
            sc = b[74 + ib];
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        const __half2 dh = __half2half2(__ushort_as_half((uint16_t) dq));
        __half2 h[16];
#else
        float f[32];
#endif
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t grid = iq2s_grid[((qs >> (8 * il)) & 0xFF) | ((qh << (8 - 2 * il)) & 0x300)];
            const uint32_t sb = (sg >> (8 * il)) & 0xFF;
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
            const __half2 sh = __float2half2_rn((float) (1 + 2 * ((sc >> (4 * (il / 2))) & 0xF)) * 0.125f);
            mq_iq_packed4(h + 4 * il, (uint32_t) grid, sh, dh, sb, 0);
            mq_iq_packed4(h + 4 * il + 2, (uint32_t) (grid >> 32), sh, dh, sb, 4);
#else
            const float d = mq_half(dq) * (0.5f + (float) ((sc >> (4 * (il / 2))) & 0xF)) * 0.25f;
            mq_signed4(f + 8 * il, (uint32_t) grid, d, sb, 0);
            mq_signed4(f + 8 * il + 4, (uint32_t) (grid >> 32), d, sb, 4);
#endif
        }
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
        mm_store8(sw + r * MM_LD + u * 32, h);
        mm_store8(sw + r * MM_LD + u * 32 + 16, h + 8);
#else
        mq_put32(sw + r * MM_LD + u * 32, f);
#endif
    }
};

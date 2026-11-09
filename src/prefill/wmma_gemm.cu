// src/prefill/wmma_gemm.cu - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100).
//
// Computes Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA intrinsics:
//   * X is T x K row-major (leading dim K), fp16 or bf16
//   * W is N x K row-major (leading dim K), fp16 or bf16
//   * Y is T x N row-major with leading dimension ldy >= N, fp32
//
// Hardware: AMD RDNA3 (gfx1100, e.g. RX 7900 XTX)
//   * v_wmma_f32_16x16x16_f16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
//   * v_wmma_f32_16x16x16_bf16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
//   * Wave32 doubled input fragment: lane t (lane_lo = t & 15) holds row lane_lo of A (16 elements along K)
//     and column lane_lo of B (16 elements along K). Lanes 16..31 duplicate lanes 0..15.
//   * Wave32 C output mapping: lane t holds column lane_lo of 16x16 output tile,
//     with 8 elements alternating rows: row m = 2*i + lane_hi (lane_hi = t >> 4).
//   * Tile variants:
//       1. gemm_wmma_64x64_4w: 4 waves (128 threads), 64T x 64N x 16K tile, double-buffered LDS for W.
//       2. gemm_wmma_16x16_1w: 1 wave (32 threads), 16T x 16N x 16K tile, zero LDS/barriers, for small T/N.

#include "wmma_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdlib>
#include <type_traits>
#include "strata/kernels/gfx_arch.hpp"

#include <atomic>
#include <cstring>

// STRATA_WMMA_GFX11 is defined by the BUILD (CMakeLists.txt, from CMAKE_HIP_ARCHITECTURES), not inferred
// from compiler macros: measured on this toolchain the HOST pass of a HIP compile does not define
// __gfx1100__ but does define __HIP_DEVICE_COMPILE__, so a compiler-macro guard here silently selected the
// "return false" stub at the bottom of this file for the very symbol the engine links - the WMMA path then
// never ran, while the same file compiled with hipcc (as the probes do) took the real branch.  One
// build-defined macro is uniform across the host and device passes.
#if defined(STRATA_WMMA_GFX11)

using v8fp32 = float __attribute__((ext_vector_type(8)));

// Does the CURRENT device run the gfx11 WMMA kernels of this file?  Exactly the targets whose intrinsics are selected
// below (gfx1100 / 1101 / 1102 / 1103 / 1150 / 1151): a gfx11 part outside the list (gfx1152) has no device code
// here, so it takes the hipBLASLt / hipBLAS paths instead of running a kernel that is not there.  Cached per device
// (hipGetDeviceProperties is not cheap, and a mixed-GPU box has more than one answer).
static bool strata_wmma_gfx11_device() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        hipDeviceProp_t prop{};
        const bool ok = hipGetDeviceProperties(&prop, dev) == hipSuccess && strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName);
        c = ok ? 1 : 2;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
}

// The same answer for host code that decides whether to pad an activation's row stride (prefill.cpp, STRATA_PF_PAD).
bool strata_pf_padded_gemm_available() { return strata_wmma_gfx11_device(); }

template <typename ElemT>
struct WmmaTraits;

template <>
struct WmmaTraits<_Float16> {
    using vec_t = _Float16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        (void) a; (void) b; (void) c;
        __builtin_trap();   // a device pass without the gfx11 intrinsic: the runtime gate never launches it, and a stray launch must not return zeros
#endif
    }
};

template <>
struct WmmaTraits<__bf16> {
    using vec_t = __bf16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
        (void) a; (void) b; (void) c;
        __builtin_trap();   // see above
#endif
    }
};

// ===========================================================================
// Variant 1: 16x16_1w - Single-wave kernel for small T / N.
// Zero LDS, zero barrier synchronization, full register residency.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_16x16_1w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 16;
    const int n_tile = blockIdx.x * 16;
    if (m_tile >= T || n_tile >= N) return;

    const int lane = threadIdx.x;   // 0..31
    const int lane_lo = lane & 15;  // 0..15
    const int lane_hi = lane >> 4;  // 0 or 1

    v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const int m_row = m_tile + lane_lo;
    const int n_row = n_tile + lane_lo;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        vec_t a_frag, b_frag;

        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        if (n_row < N) {
            __builtin_memcpy(&b_frag, W + (int64_t)n_row * K + k_tile, sizeof(b_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = 0;
        }

        c_acc = WmmaTraits<ElemT>::mma(a_frag, b_frag, c_acc);
    }

    const int out_n = n_tile + lane_lo;
    if (out_n < N) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_tile + 2 * i + lane_hi;
            if (out_m < T) {
                float* dst = Y + (int64_t)out_m * ldy + out_n;
                if (beta == 0.0f) {
                    *dst = c_acc[i];
                } else {
                    *dst = beta * (*dst) + c_acc[i];
                }
            }
        }
    }
}

// ===========================================================================
// Variant 2: 64x64_4w - 4 waves per block (128 threads), 64T x 64N tile.
// Double-buffered LDS tile for W (4 KB LDS total), cooperative 128-bit global loads.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_64x64_4w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 64;
    const int n_tile = blockIdx.x * 64;
    if (m_tile >= T || n_tile >= N) return;

    const int tid = threadIdx.x;   // 0..127
    const int wave_id = tid >> 5;  // 0..3
    const int lane = tid & 31;     // 0..31
    const int lane_lo = lane & 15; // 0..15
    const int lane_hi = lane >> 4; // 0 or 1

    // 4 accumulators per wave covering 4 x 16 N-subtiles
    v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // Double-buffered LDS tile: 64 rows of N x 16 elements of K (2 KB per buffer)
    alignas(16) __shared__ ElemT b_lds[2][64][16];

    // Thread mapping for cooperative loading of W into LDS (128 threads load 64x16 elements)
    // Each thread loads 8 halfs (16 bytes = uint4)
    const int row_in_tile = tid >> 1;     // 0..63
    const int k_sub = (tid & 1) << 3;     // 0 or 8
    const int actual_n = n_tile + row_in_tile;

    auto load_w_into_lds = [&](int buf, int k_tile) {
        const int actual_k = k_tile + k_sub;
        uint4 val = {0, 0, 0, 0};
        if (actual_n < N && actual_k < K) {
            const void* ptr = reinterpret_cast<const void*>(W + (int64_t)actual_n * K + actual_k);
            val = *reinterpret_cast<const uint4*>(ptr);
        }
        *reinterpret_cast<uint4*>(&b_lds[buf][row_in_tile][k_sub]) = val;
    };

    // Pre-fill buffer 0
    load_w_into_lds(0, 0);
    __syncthreads();

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    int cur_buf = 0;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        const int next_buf = 1 - cur_buf;
        const int k_next = k_tile + 16;

        // Prefetch next K-tile of W into LDS
        if (k_next < K) {
            load_w_into_lds(next_buf, k_next);
        }

        // Load A fragment from X for current wave's 16 M-rows
        vec_t a_frag;
        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        // Read B fragments from LDS and compute WMMA
        vec_t b_frag0, b_frag1, b_frag2, b_frag3;
        __builtin_memcpy(&b_frag0, &b_lds[cur_buf][0 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag1, &b_lds[cur_buf][16 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag2, &b_lds[cur_buf][32 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag3, &b_lds[cur_buf][48 + lane_lo][0], sizeof(vec_t));

        c_acc0 = WmmaTraits<ElemT>::mma(a_frag, b_frag0, c_acc0);
        c_acc1 = WmmaTraits<ElemT>::mma(a_frag, b_frag1, c_acc1);
        c_acc2 = WmmaTraits<ElemT>::mma(a_frag, b_frag2, c_acc2);
        c_acc3 = WmmaTraits<ElemT>::mma(a_frag, b_frag3, c_acc3);

        __syncthreads();
        cur_buf = next_buf;
    }

    // Store C to Y (coalesced 64-byte writes per wave half)
    const int m_tile_wave = m_tile + wave_id * 16;
    auto store_acc = [&](const v8fp32& acc, int n_base) {
        const int out_n = n_base + lane_lo;
        if (out_n < N) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_m = m_tile_wave + 2 * i + lane_hi;
                if (out_m < T) {
                    float* dst = Y + (int64_t)out_m * ldy + out_n;
                    if (beta == 0.0f) {
                        *dst = acc[i];
                    } else {
                        *dst = beta * (*dst) + acc[i];
                    }
                }
            }
        }
    };

    store_acc(c_acc0, n_tile + 0);
    store_acc(c_acc1, n_tile + 16);
    store_acc(c_acc2, n_tile + 32);
    store_acc(c_acc3, n_tile + 48);
}

template <typename ElemT>
static inline bool strata_wmma_gemm_dispatch(const uint16_t* X, const uint16_t* W, float* Y,
                                             int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                             void* stream) {
    if (!X || !W || !Y) return false;
    if (T <= 0 || N <= 0 || K <= 0) return false;
    // Runtime gate (review): run only on gfx11 devices, whatever this build compiled for.  The build-time
    // STRATA_WMMA_GFX11 macro controls whether the intrinsics compile; this check controls whether they run.
    if (!strata_wmma_gfx11_device()) return false;
    if (K % 16 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);

    if (T >= 32 && N >= 32) {
        dim3 block(128);
        dim3 grid((uint32_t)((N + 63) / 64), (uint32_t)((T + 63) / 64), 1);
        gemm_wmma_64x64_4w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    } else {
        dim3 block(32);
        dim3 grid((uint32_t)((N + 15) / 16), (uint32_t)((T + 15) / 16), 1);
        gemm_wmma_16x16_1w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    }
    return true;
}

bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream) {
    return strata_wmma_gemm_dispatch<_Float16>(X, W, Y, T, N, K, ldy, beta, stream);
}

bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream) {
    return strata_wmma_gemm_dispatch<__bf16>(X, W, Y, T, N, K, ldy, beta, stream);
}

// ===========================================================================
// S23 (opt-in STRATA_PF_GEMM=1): the prompt projections' FP16 GEMM on gfx11 (gfx1151: 30-33 TFLOPS on the 16K-row
// projection shapes where the tuned hipBLASLt reaches 25-27; s23/gemm_probe4.hip).  Block 128 x BN (BN 256 or 128),
// BK 32, 8 waves of 64 x BN/4, LDS double buffer with the next stage prefetched into named registers (arrays there
// went to scratch), grouped tile order (8 row tiles sweep all column tiles: the weights stream from DRAM once per
// group).  FP32 accumulation, K in another order than hipBLASLt: rounding-level, quality-gated.
// ===========================================================================
namespace pfg {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM = 128, BK = 32, LDK = BK + 8, GM = 8;
__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(h16, w);
}

// HC down has 320 output columns and K=10240. For a cached prompt increment
// the large prompt tile leaves most CUs idle. One wave per 16x16 output tile
// preserves the native kernel's sequential FP32 WMMA accumulation along K.
__global__ void __launch_bounds__(128) kernel_hc_down_increment(const _Float16* __restrict__ x,
                                                                const _Float16* __restrict__ w,
                                                                float* __restrict__ y, int rows,
                                                                int ldx, int ldw, int ldy) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int n = 320, k = 10240;
    const int tile = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const int first_row = tile / (n / 16) * 16;
    if (first_row >= rows) return;
    const int col = tile % (n / 16) * 16 + lane % 16;
    const _Float16* xr = x + (size_t) min(first_row + lane % 16, rows - 1) * ldx;
    const _Float16* wr = w + (size_t) col * ldw;
    f8 acc = {};
    for (int offset = 0; offset < k; offset += 64) {
        h16 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = frag(xr + offset + 16 * i);
            b[i] = frag(wr + offset + 16 * i);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[i], acc);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = first_row + 2 * i + lane / 16;
        if (row < rows) y[(size_t) row * ldy + col] = acc[i];
    }
#endif
}

template<bool Quantized, bool Packed = false>
__device__ __forceinline__ h16 gsu_weight_fragment(const void* weights, const uint16_t* scales, size_t index) {
    if constexpr (!Quantized) return frag(static_cast<const _Float16*>(weights) + index);
    else if constexpr (Packed) {
        const uint4 raw = *reinterpret_cast<const uint4*>(static_cast<const signed char*>(weights) + index);
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
        const __half2 d = __half2half2(__ushort_as_half(scales[index / 32]));
        // Place biased bytes under the exponent for 1024, then subtract 1152.
        // Both the signed integers and the block scale are exact half values.
        const __half2 bias = __half2half2(__ushort_as_half(0x6480));
        const __half2 zero = __float2half2_rn(0.f);
        using U8 = unsigned int __attribute__((ext_vector_type(8)));
        U8 bits;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = words[i] ^ 0x80808080u;
            const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, w, 0x04010400u);
            const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, w, 0x04030402u);
            const __half2 x = __hsub2(__builtin_bit_cast(__half2, lo), bias);
            const __half2 y = __hsub2(__builtin_bit_cast(__half2, hi), bias);
            bits[2 * i] = __builtin_bit_cast(unsigned int, __hfma2(x, d, zero));
            bits[2 * i + 1] = __builtin_bit_cast(unsigned int, __hfma2(y, d, zero));
        }
        return __builtin_bit_cast(h16, bits);
    }
    else {
        using Bytes = signed char __attribute__((ext_vector_type(16)));
        Bytes q;
        __builtin_memcpy(&q, static_cast<const signed char*>(weights) + index, sizeof(q));
        const float d = (float)__builtin_bit_cast(_Float16, scales[index / 32]);
        h16 out;
#pragma unroll
        for (int i = 0; i < 16; ++i) out[i] = (_Float16)(d * (float)q[i]);
        return out;
    }
}

// rocBLAS 5.2 on gfx1100 uses three FP32 accumulators for these small-row
// projections. Each 128-column tile rotates its 16-element segments along K.
template<int k, int stagger_mask, bool Quantized = false, bool Packed = false>
__global__ void __launch_bounds__(96) kernel_gsu3(const _Float16* __restrict__ x, const void* __restrict__ w,
                                                  float* __restrict__ y, int rows, int ldy,
                                                  const uint16_t* scales = nullptr, int first_column = 0) {
    constexpr int segments = k / 16;
    const int group = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.y * 4 + lane / 8, col = blockIdx.x * 8 + lane % 8;
    __shared__ float partial[3][32];
    float sum = 0.f;
    if (row < rows) {
        const int count = (segments + 2 - group) / 3;
        const int start = (((first_column + col) / 128) & stagger_mask) * 8;
        for (int pass = 0; pass < 2; ++pass) {
            const int begin = pass == 0 ? start : 0, end = pass == 0 ? count : start;
            for (int step = begin; step < end; ++step) {
                const int offset = 16 * (group + 3 * step);
                const h16 a = frag(x + (size_t) row * k + offset);
                const h16 b = gsu_weight_fragment<Quantized,Packed>(w, scales, (size_t) col * k + offset);
#pragma unroll
                for (int i = 0; i < 16; ++i) sum = fmaf((float) a[i], (float) b[i], sum);
            }
        }
    }
    partial[group][lane] = sum;
    __syncthreads();
    if (group == 0 && row < rows)
        y[(size_t) row * ldy + col] = (partial[0][lane] + partial[1][lane]) + partial[2][lane];
}

// Two output rows share each decoded weight fragment. Each row retains the
// original three partial chains, segment rotation and final addition order.
template <int K, int StaggerMask>
__global__ void __launch_bounds__(96) kernel_gsu3_rowpair(const _Float16* __restrict__ x, const void* __restrict__ w,
                                                        float* __restrict__ y, int rows, int ldy,
                                                        const uint16_t* scales, int first_column) {
    constexpr int segments = K / 16;
    const int group = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.y * 8 + lane / 8, col = blockIdx.x * 8 + lane % 8;
    __shared__ float partial[3][64];
    float sum0 = 0.f, sum1 = 0.f;
    if (row < rows) {
        const int count = (segments + 2 - group) / 3;
        const int start = (((first_column + col) / 128) & StaggerMask) * 8;
        for (int pass = 0; pass < 2; ++pass) {
            const int begin = pass == 0 ? start : 0, end = pass == 0 ? count : start;
            for (int step = begin; step < end; ++step) {
                const int offset = 16 * (group + 3 * step);
                const h16 a0 = frag(x + (size_t) row * K + offset);
                h16 a1 = {};
                if (row + 4 < rows) a1 = frag(x + (size_t) (row + 4) * K + offset);
                const h16 b = gsu_weight_fragment<true, true>(w, scales, (size_t) col * K + offset);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    sum0 = fmaf((float) a0[i], (float) b[i], sum0);
                    sum1 = fmaf((float) a1[i], (float) b[i], sum1);
                }
            }
        }
    }
    partial[group][lane] = sum0;
    partial[group][32 + lane] = sum1;
    __syncthreads();
    if (group == 0) {
        if (row < rows) y[(size_t) row * ldy + col] = (partial[0][lane] + partial[1][lane]) + partial[2][lane];
        if (row + 4 < rows)
            y[(size_t) (row + 4) * ldy + col] = (partial[0][32 + lane] + partial[1][32 + lane]) + partial[2][32 + lane];
    }
}

template <int BN, int WN>
__global__ void __launch_bounds__(256) kernel(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                              float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                              int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int TN = WN / 16, NB = BN / 64;
    __shared__ __align__(16) _Float16 sA[2][BM][LDK];
    __shared__ __align__(16) _Float16 sB[2][BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 2, sq = tid & 3;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    f8 acc[4][TN];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 ra0 = ldA(0, sr), ra1 = ldA(0, sr + 64), rb0 = ldB(0, sr), rb1 = ldB(0, sr + 64), rb2 = rb0, rb3 = rb0;
    if constexpr (NB > 2) { rb2 = ldB(0, sr + 128); rb3 = ldB(0, sr + 192); }
    int buf = 0;
    *reinterpret_cast<uint4*>(&sA[0][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[0][sr + 64][8 * sq]) = ra1;
    *reinterpret_cast<uint4*>(&sB[0][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[0][sr + 64][8 * sq]) = rb1;
    if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[0][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[0][sr + 192][8 * sq]) = rb3; }
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            ra0 = ldA(k0 + BK, sr); ra1 = ldA(k0 + BK, sr + 64); rb0 = ldB(k0 + BK, sr); rb1 = ldB(k0 + BK, sr + 64);
            if constexpr (NB > 2) { rb2 = ldB(k0 + BK, sr + 128); rb3 = ldB(k0 + BK, sr + 192); }
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            h16 a[4], bb[TN];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = frag(&sA[buf][64 * wm + 16 * i + l16][ks]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bb[j] = frag(&sB[buf][WN * wn + 16 * j + l16][ks]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bb[j], acc[i][j]);
        }
        if (more) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[nb][sr + 64][8 * sq]) = ra1;
            *reinterpret_cast<uint4*>(&sB[nb][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[nb][sr + 64][8 * sq]) = rb1;
            if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[nb][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[nb][sr + 192][8 * sq]) = rb3; }
            __syncthreads();
            buf = nb;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
// The 128 x 256 kernel with a 64-k LDS tile (single buffer, the next tile in 12 named staging registers, 2 barriers
// per tile: half the loop trips and barriers per k of the BK 32 double buffer). Each accumulator takes the same
// WMMAs in the same k order as `kernel`: the same bits (s23/gemm_probe11.hip, memcmp). gfx1151, M 16384, both
// operands padded: 38.0 -> 41.3 TFLOPS (N 2560 K 6144), 38.9 -> 40.9 (N 12288 K 2560), 40.3 -> 40.4 (N 10240).
// K a multiple of 64.
__global__ void __launch_bounds__(256) kernel_bk64(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                   float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                                   int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 256, WN = 64, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) _Float16 sA[BM][LDK64];
    __shared__ __align__(16) _Float16 sB[BN][LDK64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;            // 32 rows per pass, 8 x 8 halves per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                      a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]),
                      b3 = frag(&sB[br + 48][ks]);
#define PFG_ROW(i, a)                                                                                                 \
    acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[i][0]);                                         \
    acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[i][1]);                                         \
    acc[i][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[i][2]);                                         \
    acc[i][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[i][3]);
            PFG_ROW(0, a0) PFG_ROW(1, a1) PFG_ROW(2, a2) PFG_ROW(3, a3)
#undef PFG_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    float* y = Y + (size_t) m * ldy + n;
                    *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}
// A 128 x 96 tile uses 31.5 KiB LDS and four waves, with the same ordered
// FP32 WMMA dot products as kernel_bk64. Test both burst and interleaved
// prefetching independently; changing the schedule never changes K order.
// Scheduling idea: AMD's RDNA FP16 GEMM walkthrough (2026-10-16), steps 7/8.
// Unlike that example, these accumulators and outputs remain FP32.
template <int I, int End, class Fn>
__device__ __forceinline__ void schedule_steps(Fn& fn) {
    fn(std::integral_constant<int,I>{});
    if constexpr(I+1<End)schedule_steps<I+1,End>(fn);
}
template <bool Interleave>
__global__ void __launch_bounds__(128) kernel_bk64_n96(const _Float16* __restrict__ X,
    const _Float16* __restrict__ W, float* __restrict__ Y, int M, int N, int K,
    int ldy, int accumulate, int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 96, LDK = 72;
    __shared__ __align__(16) _Float16 sA[BM][LDK], sB[BN][LDK];
    const int tid=threadIdx.x, lane=tid&31, wave=tid>>5, l16=lane&15, hi=lane>>4;
    const int wm=wave&1, wn=wave>>1;
    const int nm=(M+BM-1)/BM, nn=(N+BN-1)/BN, block=blockIdx.x;
    const int group=block/(GM*nn), first=group*GM, size=min(GM,nm-first);
    const int m0=(first+(block%(GM*nn))%size)*BM, n0=((block%(GM*nn))/size)*BN;
    const int sr=tid>>3, col=8*(tid&7);
    // This experimental kernel requires each operand to fit a 32-bit byte
    // range. The descriptor supplies the uniform base; only a byte offset
    // lives in a vector register. Invalid trailing rows read zero.
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b128)
    const auto xr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(X),0,
        (unsigned)((size_t)M*ldx*2),0x31004000u);
    const auto wr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(W),0,
        (unsigned)((size_t)N*ldw*2),0x31004000u);
    const unsigned xo=(unsigned)(m0+sr)*(unsigned)ldx*2u+(unsigned)col*2u;
    const unsigned wo=(unsigned)(n0+sr)*(unsigned)ldw*2u+(unsigned)col*2u;
    auto ldA = [&](int r,int k) -> uint4 {
        const auto v=__builtin_amdgcn_raw_buffer_load_b128(xr,xo+(unsigned)r*ldx*2u,(unsigned)k*2u,0);
        return __builtin_bit_cast(uint4,v);
    };
    auto ldB = [&](int r,int k) -> uint4 {
        const auto v=__builtin_amdgcn_raw_buffer_load_b128(wr,wo+(unsigned)r*ldw*2u,(unsigned)k*2u,0);
        return __builtin_bit_cast(uint4,v);
    };
#else
    auto ldA = [&](int r,int k) -> uint4 {
        return *reinterpret_cast<const uint4*>(X+(size_t)min(m0+sr+r,M-1)*ldx+k+col);
    };
    auto ldB = [&](int r,int k) -> uint4 {
        return *reinterpret_cast<const uint4*>(W+(size_t)min(n0+sr+r,N-1)*ldw+k+col);
    };
#endif
    uint4 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5;
    auto load = [&](auto piece,int k) {
        constexpr int i=decltype(piece)::value;
        if constexpr(i==0)a0=ldA(0,k); if constexpr(i==1)b0=ldB(0,k);
        if constexpr(i==2)a1=ldA(16,k);if constexpr(i==3)b1=ldB(16,k);
        if constexpr(i==4)a2=ldA(32,k);if constexpr(i==5)b2=ldB(32,k);
        if constexpr(i==6)a3=ldA(48,k);if constexpr(i==7)b3=ldB(48,k);
        if constexpr(i==8)a4=ldA(64,k);if constexpr(i==9)b4=ldB(64,k);
        if constexpr(i==10)a5=ldA(80,k);if constexpr(i==11)b5=ldB(80,k);
        if constexpr(i==12)a6=ldA(96,k);if constexpr(i==13)a7=ldA(112,k);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][col])=a0;
        *reinterpret_cast<uint4*>(&sA[sr+16][col])=a1;
        *reinterpret_cast<uint4*>(&sA[sr+32][col])=a2;
        *reinterpret_cast<uint4*>(&sA[sr+48][col])=a3;
        *reinterpret_cast<uint4*>(&sA[sr+64][col])=a4;
        *reinterpret_cast<uint4*>(&sA[sr+80][col])=a5;
        *reinterpret_cast<uint4*>(&sA[sr+96][col])=a6;
        *reinterpret_cast<uint4*>(&sA[sr+112][col])=a7;
        *reinterpret_cast<uint4*>(&sB[sr][col])=b0;
        *reinterpret_cast<uint4*>(&sB[sr+16][col])=b1;
        *reinterpret_cast<uint4*>(&sB[sr+32][col])=b2;
        *reinterpret_cast<uint4*>(&sB[sr+48][col])=b3;
        *reinterpret_cast<uint4*>(&sB[sr+64][col])=b4;
        *reinterpret_cast<uint4*>(&sB[sr+80][col])=b5;
    };
    f8 acc[4][3];
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<3;++j)acc[i][j]=f8{0,0,0,0,0,0,0,0};
    auto first_load=[&](auto piece){load(piece,0);};
    schedule_steps<0,14>(first_load);
    store();__syncthreads();
    const int ar=64*wm+l16,br=48*wn+l16;
    auto consume=[&](int k0,auto fetch_tag) {
        constexpr bool fetch=decltype(fetch_tag)::value;
        if constexpr(fetch && !Interleave) {
            auto next_load=[&](auto piece){load(piece,k0+64);};
            schedule_steps<0,14>(next_load);
        }
        h16 av[2],bv[3];
        av[0]=frag(&sA[ar][0]);bv[0]=frag(&sB[br][0]);
        auto multiply=[&](auto index) {
            constexpr int step=decltype(index)::value;
            constexpr int slice=step/12,m=(step%12)/3,n=step%3;
            if constexpr(fetch && Interleave) {
                if constexpr(step<=12)load(std::integral_constant<int,step>{},k0+64);
                if constexpr(step==14)load(std::integral_constant<int,13>{},k0+64);
            }
            // Read only the fragments needed by the next instruction. A uses
            // two alternating registers; the three B fragments survive a slice.
            if constexpr(step+1<48) {
                constexpr int ns=(step+1)/12,nm=((step+1)%12)/3,nn=(step+1)%3;
                if constexpr(ns!=slice || nm!=m)av[nm%2]=frag(&sA[ar+16*nm][16*ns]);
                if constexpr(nm==0)bv[nn]=frag(&sB[br+16*nn][16*ns]);
            }
            if constexpr(Interleave)__builtin_amdgcn_s_setprio(1);
            acc[m][n]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av[m%2],bv[n],acc[m][n]);
            if constexpr(Interleave)__builtin_amdgcn_s_setprio(0);
        };
        schedule_steps<0,48>(multiply);
    };
    for(int k0=0;k0+64<K;k0+=64) {
        consume(k0,std::true_type{});
        __syncthreads();store();__syncthreads();
    }
    consume(K-64,std::false_type{});
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<3;++j) {
            const int n=n0+48*wn+16*j+l16;
            if(n>=N)continue;
#pragma unroll
            for(int e=0;e<8;++e) {
                const int m=m0+64*wm+16*i+2*e+hi;
                if(m<M) {
                    float* y=Y+(size_t)m*ldy+n;
                    *y=accumulate ? *y+acc[i][j][e] : acc[i][j][e];
                }
            }
        }
#endif
}

// HC up projection and stream mixing with the same FP32 WMMA and gate arithmetic.
__global__ void __launch_bounds__(256) kernel_hcup_mix(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                   float* __restrict__ Y, uint16_t* __restrict__ Yh, const float* __restrict__ R,
                                                   const float* __restrict__ norm, const float* __restrict__ rms, int M, int N, int K) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 256, WN = 64, BK64 = 64, LDK64 = BK64 + 8;
    union Scratch {
        struct { _Float16 a[BM][LDK64], b[BN][LDK64]; } product;
        float2 mix[4][32][64];
    };
    __shared__ __align__(16) Scratch scratch;
    auto& sA = scratch.product.a;
    auto& sB = scratch.product.b;
    const int width = N / 4;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = width / 64, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * 64;
    const int sr = tid >> 3, sq = tid & 7;            // 32 rows per pass, 8 x 8 halves per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * K + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) ((r / 64) * width + n0 + r % 64) * K + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                      a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]),
                      b3 = frag(&sB[br + 48][ks]);
#define PFG_ROW(i, a)                                                                                                 \
    acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[i][0]);                                         \
    acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[i][1]);                                         \
    acc[i][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[i][2]);                                         \
    acc[i][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[i][3]);
            PFG_ROW(0, a0) PFG_ROW(1, a1) PFG_ROW(2, a2) PFG_ROW(3, a3)
#undef PFG_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }
    // The four wave pairs now cover the four HC streams for the SAME
    // 64 channels. Reuse the dead GEMM tile to sum those streams in order;
    // the 10,240-column up result never makes a round trip through VRAM.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                const int d = n0 + 16 * j + l16;
                float xn = 0.f, gate = 0.f;
                if (m < M) {
                    xn = R[(size_t)m * N + wn * width + d] * norm[wn * width + d] * rms[m * 4 + wn];
                    gate = 1.f / (1.f + __expf(-acc[i][j][e]));
                }
                scratch.mix[wn][16 * wm + 2 * e + hi][16 * j + l16] = make_float2(xn,gate);
            }
        __syncthreads();
        for (int at = tid; at < 32 * 64; at += 256) {
            const int row = at / 64, d = at % 64;
            const int m = m0 + (row / 16) * 64 + i * 16 + row % 16;
            if (m < M) {
                // Keep gate and normalized residual separate until this
                // expression, preserving the original mixer's FMA contraction.
                float g[4];
#pragma unroll
                for (int c=0;c<4;++c) {
                    const float2 pair = scratch.mix[c][row][d];
                    g[c] = pair.y * pair.x;
                }
                const float value = (g[0] + g[1] + g[2] + g[3]) / 4.f;
                Y[(size_t)m * width + n0 + d] = value;
                reinterpret_cast<_Float16*>(Yh)[(size_t)m * width + n0 + d] = (_Float16)value;
            }
        }
    }
#endif
}

// The native gr_upmix tile, adapted to split's FP16 images and mixer order.
// Each lane owns all four streams of its output, so the epilogue stays in
// registers. The 64 complete weight rows are staged once; one barrier.
__global__ void __launch_bounds__(512) kernel_hcup_register(const _Float16* __restrict__ X,
    const _Float16* __restrict__ W, float* __restrict__ Y, uint16_t* __restrict__ Yh,
    const float* __restrict__ R, const float* __restrict__ norm, const float* __restrict__ rms, int M) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int WIDTH = 2560, RANK = 320, COLS = 16, ROWS = 256, LD = RANK + 8;
    __shared__ __align__(16) _Float16 weights[4 * COLS][LD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col = lane & 15, half = lane >> 4;
    const int t0 = blockIdx.y * ROWS + wave * 16, d0 = blockIdx.x * COLS;
    for (int i = tid; i < 4 * COLS * (RANK / 8); i += 512) {
        const int row = i / (RANK / 8), k = i % (RANK / 8) * 8;
        *reinterpret_cast<uint4*>(&weights[row][k]) = *reinterpret_cast<const uint4*>(
            W + (size_t) (row / COLS * WIDTH + d0 + row % COLS) * RANK + k);
    }
    __syncthreads();
    if (t0 >= M) return;
    const _Float16* x = X + (size_t) min(t0 + col, M - 1) * RANK;
    f8 acc[4] = {};
#pragma unroll
    for (int k = 0; k < RANK; k += 16) {
        const h16 a = frag(x + k);
#pragma unroll
        for (int c = 0; c < 4; ++c)
            acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, frag(&weights[c * COLS + col][k]), acc[c]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int t = t0 + 2 * i + half, d = d0 + col;
        if (t >= M) continue;
        float g[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float xn = R[(size_t) t * (4 * WIDTH) + c * WIDTH + d] * norm[c * WIDTH + d] * rms[t * 4 + c];
            const float gate = 1.f / (1.f + __expf(-acc[c][i]));
            g[c] = gate * xn;
        }
        const float mixed = (g[0] + g[1] + g[2] + g[3]) / 4.f;
        Y[(size_t) t * WIDTH + d] = mixed;
        reinterpret_cast<_Float16*>(Yh)[(size_t) t * WIDTH + d] = (_Float16) mixed;
    }
#endif
}

// Q8 weights enter the same register-mixer tile without a full FP16 image.
// Keep the original 20 ascending WMMA steps and four-stream epilogue order.
__global__ void __launch_bounds__(512) kernel_hcup_q8_register(const _Float16* __restrict__ X,
    const int8_t* __restrict__ Q, const uint16_t* __restrict__ scales,
    float* __restrict__ Y, uint16_t* __restrict__ Yh,
    const float* __restrict__ R, const float* __restrict__ norm, const float* __restrict__ rms, int M) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int WIDTH = 2560, RANK = 320, COLS = 16, ROWS = 256, LD = RANK + 8;
    __shared__ __align__(16) _Float16 weights[4 * COLS][LD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col = lane & 15, half = lane >> 4;
    const int t0 = blockIdx.y * ROWS + wave * 16, d0 = blockIdx.x * COLS;
    for (int i = tid; i < 4 * COLS * (RANK / 16); i += 512) {
        const int row = i / (RANK / 16), k = i % (RANK / 16) * 16;
        const size_t at = (size_t) (row / COLS * WIDTH + d0 + row % COLS) * RANK + k;
        const h16 value = gsu_weight_fragment<true, false>(Q, scales, at);
        __builtin_memcpy(&weights[row][k], &value, sizeof(value));
    }
    __syncthreads();
    if (t0 >= M) return;
    const _Float16* x = X + (size_t) min(t0 + col, M - 1) * RANK;
    f8 acc[4] = {};
#pragma unroll
    for (int k = 0; k < RANK; k += 16) {
        const h16 a = frag(x + k);
#pragma unroll
        for (int c = 0; c < 4; ++c)
            acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, frag(&weights[c * COLS + col][k]), acc[c]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int t = t0 + 2 * i + half, d = d0 + col;
        if (t >= M) continue;
        float g[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float xn = R[(size_t) t * (4 * WIDTH) + c * WIDTH + d] * norm[c * WIDTH + d] * rms[t * 4 + c];
            const float gate = 1.f / (1.f + __expf(-acc[c][i]));
            g[c] = gate * xn;
        }
        const float mixed = (g[0] + g[1] + g[2] + g[3]) / 4.f;
        Y[(size_t) t * WIDTH + d] = mixed;
        reinterpret_cast<_Float16*>(Yh)[(size_t) t * WIDTH + d] = (_Float16) mixed;
    }
#endif
}

// S23 (STRATA_PF_HCDOWN=1): the hyper-connection read's down (N nd = 320) and inject (N ni = 4) projections of xn16
// (BF16, T x 10240, token stride ldx: 10240 + 64 avoids the 4 KB-multiple stride) as ONE GEMM - weight rows
// [w_down; w_inject] read from the two tensors, the epilogue splitting the columns into lo (T x nd) and inj (T x ni),
// so xn16 is read once.  128 x 128 blocks, the BK 64 single-buffer scheme of kernel_bk64.  s23/hcdown_probe.hip,
// 16K tokens: 3.60 / 3.77 ms (hipBLASLt in the engine: down 6.78 + inject 1.81).  FP32 accumulation in another k
// order than hipBLASLt: rounding-level.
typedef __bf16 hb16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ hb16 bfrag(const uint16_t* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hb16, w);
}
__global__ void __launch_bounds__(256) kernel_hcdown(const uint16_t* __restrict__ X, int ldx,
                                                     const uint16_t* __restrict__ Wd, const uint16_t* __restrict__ Wi,
                                                     int nd, int ni, float* __restrict__ lo, float* __restrict__ inj,
                                                     int M, int K) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 128, WN = 32, TN = 2, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) uint16_t sA[BM][LDK64];
    __shared__ __align__(16) uint16_t sB[BN][LDK64];
    const int N = nd + ni;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        const int n = min(n0 + r, N - 1);
        const uint16_t* row = n < nd ? Wd + (size_t) n * K : Wi + (size_t) (n - nd) * K;
        return *reinterpret_cast<const uint4*>(row + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
    };
    f8 acc[4][TN];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const hb16 a0 = bfrag(&sA[ar][ks]), a1 = bfrag(&sA[ar + 16][ks]), a2 = bfrag(&sA[ar + 32][ks]),
                       a3 = bfrag(&sA[ar + 48][ks]);
            const hb16 b0 = bfrag(&sB[br][ks]), b1 = bfrag(&sB[br + 16][ks]);
            acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc[0][0]);
            acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc[1][0]);
            acc[2][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b0, acc[2][0]);
            acc[3][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b0, acc[3][0]);
            acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc[0][1]);
            acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc[1][1]);
            acc[2][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b1, acc[2][1]);
            acc[3][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b1, acc[3][1]);
        }
        if (more) {
            __syncthreads();
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    if (n < nd) lo[(size_t) m * nd + n] = acc[i][j][e];
                    else inj[(size_t) m * ni + (n - nd)] = acc[i][j][e];
                }
            }
        }
#endif
}

// The native HC down tile with FP16 operands and the original ascending K order.
// BK64 reuses each loaded operand over four WMMAs; the old narrow tile uses BK32.
__global__ void __launch_bounds__(256) kernel_narrow_bk64(const _Float16* __restrict__ X, int ldx,
    const _Float16* __restrict__ W, int ldw, float* __restrict__ Y, int M, int N, int K, int ldy) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 128, WN = 32, TN = 2, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) _Float16 sA[BM][LDK64];
    __shared__ __align__(16) _Float16 sB[BN][LDK64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        const int n = min(n0 + r, N - 1);
        const _Float16* row = W + (size_t) n * ldw;
        return *reinterpret_cast<const uint4*>(row + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
    };
    f8 acc[4][TN];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                       a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]);
            acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc[0][0]);
            acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, acc[1][0]);
            acc[2][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b0, acc[2][0]);
            acc[3][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b0, acc[3][0]);
            acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, acc[0][1]);
            acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc[1][1]);
            acc[2][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b1, acc[2][1]);
            acc[3][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b1, acc[3][1]);
        }
        if (more) {
            __syncthreads();
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    Y[(size_t) m * ldy + n] = acc[i][j][e];
                }
            }
        }
#endif
}

// S23/S (STRATA_HCD_EXACT=1): the HC down projection (N 320, K 10240, BF16 in, FP32 out) bitwise equal to hipBLASLt
// solution 1176 / 1177 (MT32x96 / MT96x96, StaggerU 32, stride 256 B): every output is the k tiles of 32 accumulated in
// order, 16 per WMMA, starting at k tile 4 * (token tile of 96 % 32) and wrapping (hcd_bit / hcd_perm probes: 0 of
// 5.2 M outputs differ at T 16384).  96 x 160 blocks (two per token tile), 6 waves of 32 x 80.
typedef uint32_t hx_u4 __attribute__((ext_vector_type(4)));
constexpr int HX_N = 320, HX_BN = 160, HX_BM = 96, HX_BK = 64, HX_LDK = HX_BK + 8, HX_NT = 192, HX_K = 10240;
__device__ __forceinline__ hb16 hx_frag(const uint16_t* p) {
    const hx_u4 a = *reinterpret_cast<const hx_u4*>(p), b = *reinterpret_cast<const hx_u4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hb16, w);
}
__global__ void __launch_bounds__(HX_NT) kernel_hcd_exact(const uint16_t* __restrict__ X, int ldx,
                                                          const uint16_t* __restrict__ W, float* __restrict__ Y, int M) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int K = HX_K, BN = HX_BN, BMT = HX_BM, BK = HX_BK, NT = HX_NT;
    __shared__ __align__(16) uint16_t sA[BMT][HX_LDK];
    __shared__ __align__(16) uint16_t sB[BN][HX_LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave % 3, wn = wave / 3, nbase = BN * (blockIdx.x & 1);
    const int tt = blockIdx.x >> 1, m0 = tt * BMT;
    constexpr int nslab = K / BK;
    const int rot = (2 * (tt & 31)) % nslab;   // 4 k tiles of 32 = 2 slabs of 64
    hx_u4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6;
#define HX_LDA(v, i) { const int idx = tid + NT * (i), r = idx >> 3, q = idx & 7; v = *reinterpret_cast<const hx_u4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * q); }
#define HX_LDB(v, i) { const int idx = min(tid + NT * (i), BN * 8 - 1), r = idx >> 3, q = idx & 7; v = *reinterpret_cast<const hx_u4*>(W + (size_t) (nbase + r) * K + k0 + 8 * q); }
#define HX_STA(v, i) { const int idx = tid + NT * (i); *reinterpret_cast<hx_u4*>(&sA[idx >> 3][8 * (idx & 7)]) = v; }
#define HX_STB(v, i) { const int idx = tid + NT * (i); if (idx < BN * 8) *reinterpret_cast<hx_u4*>(&sB[idx >> 3][8 * (idx & 7)]) = v; }
#define HX_LOAD(k0v) { const int k0 = (k0v); HX_LDA(ra0, 0) HX_LDA(ra1, 1) HX_LDA(ra2, 2) HX_LDA(ra3, 3) HX_LDB(rb0, 0) HX_LDB(rb1, 1) HX_LDB(rb2, 2) HX_LDB(rb3, 3) HX_LDB(rb4, 4) HX_LDB(rb5, 5) HX_LDB(rb6, 6) }
#define HX_STORE() { HX_STA(ra0, 0) HX_STA(ra1, 1) HX_STA(ra2, 2) HX_STA(ra3, 3) HX_STB(rb0, 0) HX_STB(rb1, 1) HX_STB(rb2, 2) HX_STB(rb3, 3) HX_STB(rb4, 4) HX_STB(rb5, 5) HX_STB(rb6, 6) }
    f8 acc[2][5];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    HX_LOAD(rot * BK);
    HX_STORE();
    __syncthreads();
    const int ar = 32 * wm + l16, br = 80 * wn + l16;
    for (int s = 0; s < nslab; ++s) {
        const bool more = s + 1 < nslab;
        if (more) HX_LOAD(((s + 1 + rot) % nslab) * BK);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const hb16 a0 = hx_frag(&sA[ar][ks]), a1 = hx_frag(&sA[ar + 16][ks]);
            const hb16 b0 = hx_frag(&sB[br][ks]), b1 = hx_frag(&sB[br + 16][ks]), b2 = hx_frag(&sB[br + 32][ks]),
                       b3 = hx_frag(&sB[br + 48][ks]), b4 = hx_frag(&sB[br + 64][ks]);
#define HX_W2(j, bj) acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, bj, acc[0][j]); acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, bj, acc[1][j]);
            HX_W2(0, b0) HX_W2(1, b1) HX_W2(2, b2) HX_W2(3, b3) HX_W2(4, b4)
#undef HX_W2
        }
        if (more) {
            __syncthreads();
            HX_STORE();
            __syncthreads();
        }
    }
#undef HX_LDA
#undef HX_LDB
#undef HX_STA
#undef HX_STB
#undef HX_LOAD
#undef HX_STORE
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int n = nbase + 80 * wn + 16 * j + l16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 32 * wm + 16 * i + 2 * e + hi;
                if (m < M) Y[(size_t) m * HX_N + n] = acc[i][j][e];
            }
        }
#endif
}
// Q8 HC up / shared down for medium incremental chunks. Weight expansion
// rounds to FP16 before the same twenty ascending WMMA steps as the wide path.
template<int TileRows>
__global__ void __launch_bounds__(128) kernel_q8_k320(const _Float16* x, const int8_t* q,
        const uint16_t* scales, float* y, int rows, int ldy) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int m0 = blockIdx.y * TileRows, n = blockIdx.x * 64 + wave * 16 + (lane & 15);
    f8 acc[TileRows / 16]{};
    for (int k = 0; k < 320; k += 16) {
        const h16 weight = gsu_weight_fragment<true, false>(q, scales, (size_t) n * 320 + k);
#pragma unroll
        for (int i = 0; i < TileRows / 16; ++i) {
            const int m = min(m0 + i * 16 + (lane & 15), rows - 1);
            const h16 input = frag(x + (size_t) m * 320 + k);
            acc[i] = WmmaTraits<_Float16>::mma(input, weight, acc[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < TileRows / 16; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int m = m0 + i * 16 + e * 2 + (lane >> 4);
            if (m < rows) y[(size_t) m * ldy + n] = acc[i][e];
        }
}
}  // namespace pfg

bool strata_pf_q8_k320_f16(const uint16_t* x, const int8_t* q, const uint16_t* scales, float* y,
                           int rows, int n, int k, int ldy, void* stream) {
    if (!x || !q || !scales || !y || k != 320 || rows < 64 || rows > 256 ||
        (n != 2560 && n != 10240) || ldy < n ||
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(q)) & 15) || !strata_wmma_gfx11_device()) return false;
    auto st = static_cast<hipStream_t>(stream);
    if (rows <= 96) pfg::kernel_q8_k320<32><<<dim3(n / 64, (rows + 31) / 32), 128, 0, st>>>(
        (const _Float16*) x, q, scales, y, rows, ldy);
    else pfg::kernel_q8_k320<64><<<dim3(n / 64, (rows + 63) / 64), 128, 0, st>>>(
        (const _Float16*) x, q, scales, y, rows, ldy);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_hcdown_bf16(const uint16_t* X, int64_t ldx, const uint16_t* Wd, const uint16_t* Wi, int64_t nd,
                           int64_t ni, float* lo, float* inj, int64_t T, int64_t K, void* stream) {
    if (!X || !Wd || !Wi || !lo || !inj || T < 64 || nd < 1 || ni < 0 || K % 64 != 0 || ldx < K || ldx % 8 != 0) return false;
    if (T > (1LL << 30) || K > (1LL << 30) || ldx > (1LL << 30)) return false;
    if (!strata_wmma_gfx11_device()) return false;
    const int64_t mt = (T + pfg::BM - 1) / pfg::BM;
    const unsigned grid = (unsigned) (mt * ((nd + ni + 127) / 128));
    pfg::kernel_hcdown<<<grid, 256, 0, static_cast<hipStream_t>(stream)>>>(X, (int) ldx, Wd, Wi, (int) nd, (int) ni,
                                                                            lo, inj, (int) T, (int) K);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_hcdown_exact_bf16(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N,
                                 int64_t K, void* stream) {
    if (!X || !W || !Y || T < 1 || N != pfg::HX_N || K != pfg::HX_K || ldx < K || ldx % 8 != 0 || ldx > (1LL << 30) ||
        T > (1LL << 30))
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    const unsigned grid = (unsigned) (2 * ((T + pfg::HX_BM - 1) / pfg::HX_BM));
    pfg::kernel_hcd_exact<<<grid, pfg::HX_NT, 0, static_cast<hipStream_t>(stream)>>>(X, (int) ldx, W, Y, (int) T);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, void* stream) {
    return strata_pf_gemm_f16_ld(X, K, W, K, Y, T, N, K, ldy, beta, stream);
}

bool strata_pf_hcup_q8_f16(const uint16_t* X, const int8_t* Q, const uint16_t* scales,
                           const float* R, const float* norm, const float* rms, float* Y, uint16_t* Yh,
                           int64_t T, int64_t N, int64_t K, void* stream) {
    if (!X || !Q || !scales || !R || !norm || !rms || !Y || !Yh || T < 64 || T > (1LL << 28) ||
        N != 10240 || K != 320 || ((reinterpret_cast<uintptr_t>(X) | reinterpret_cast<uintptr_t>(Q)) & 15))
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    pfg::kernel_hcup_q8_register<<<dim3(2560 / 16, (T + 255) / 256), 512, 0, static_cast<hipStream_t>(stream)>>>(
        (const _Float16*) X, Q, scales, Y, Yh, R, norm, rms, (int) T);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_hcup_mix_f16(const uint16_t* X, const uint16_t* W, const float* R, const float* norm, const float* rms,
                            float* Y, uint16_t* Yh, int64_t T, int64_t N, int64_t K, void* stream) {
    if (!X || !W || !R || !norm || !rms || !Y || !Yh || T < 64 || T > (1LL << 30) || N != 10240 || K != 320)
        return false;
    if (!strata_wmma_gfx11_device()) return false;
    static const bool registers = [] { const char* v = std::getenv("STRATA_PF_HC_UP_REGISTER"); return v && v[0] == '1'; }();
    if (registers) {
        pfg::kernel_hcup_register<<<dim3(2560 / 16, (T + 255) / 256), 512, 0, static_cast<hipStream_t>(stream)>>>(
            (const _Float16*) X, (const _Float16*) W, Y, Yh, R, norm, rms, (int) T);
        return hipGetLastError() == hipSuccess;
    }
    const unsigned grid = (unsigned)(((T + pfg::BM - 1) / pfg::BM) * (N / 4 / 64));
    pfg::kernel_hcup_mix<<<grid,256,0,static_cast<hipStream_t>(stream)>>>((const _Float16*)X,(const _Float16*)W,
        Y,Yh,R,norm,rms,(int)T,(int)N,(int)K);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_gemm_narrow_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                               int64_t ldy, void* stream) {
    return strata_pf_gemm_narrow_f16_ld(X, K, W, K, Y, T, N, K, ldy, stream);
}

template<int k, int mask>
static bool launch_gsu3(const uint16_t* x, const uint16_t* w, float* y, int rows, int n, int ldy, void* stream) {
    if (!x || !w || !y || rows < 16 || rows >= 64 || ldy < n ||
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 15) || !strata_wmma_gfx11_device())
        return false;
    pfg::kernel_gsu3<k, mask><<<dim3(n / 8, (rows + 3) / 4), 96, 0, static_cast<hipStream_t>(stream)>>>(
        (const _Float16*) x, (const _Float16*) w, y, rows, ldy);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_hcdown_gsu3_f16(const uint16_t* x, const uint16_t* w, float* y, int rows, int n, int k, int ldy, void* stream) {
    return n == 320 && k == 10240 && launch_gsu3<10240, 15>(x, w, y, rows, n, ldy, stream);
}

template <int K, int Mask>
static bool q8_gsu_selected(const uint16_t* x, const int8_t* q, const uint16_t* d, float* y,
                            int rows, int n, int ldy, int first_column, void* stream) {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_PF_Q8_PAIR");
        return v && v[0] >= '1' && v[0] <= '2' && v[1] == 0 ? v[0] - '0' : 0;
    }();
    auto st = static_cast<hipStream_t>(stream);
    // Pair rows in measured shapes with enough blocks for the card.
    const bool pair_rows = mode == 2 && (n >= 2048 || (K == 10240 && rows >= 32));
    if (pair_rows)
        pfg::kernel_gsu3_rowpair<K, Mask><<<dim3(n / 8, (rows + 7) / 8), 96, 0, st>>>(
            (const _Float16*) x, q, y, rows, ldy, d, first_column);
    else if (mode)
        pfg::kernel_gsu3<K, Mask, true, true><<<dim3(n / 8, (rows + 3) / 4), 96, 0, st>>>(
            (const _Float16*) x, q, y, rows, ldy, d, first_column);
    else
        pfg::kernel_gsu3<K, Mask, true><<<dim3(n / 8, (rows + 3) / 4), 96, 0, st>>>(
            (const _Float16*) x, q, y, rows, ldy, d, first_column);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_q8_gsu3_f16(const uint16_t* x, const int8_t* q, const uint16_t* d, float* y,
                           int rows, int n, int k, int ldy, int reference_n, int first_column, void* stream) {
    if (!x || !q || !d || !y || rows < 16 || rows >= 64 || n < 1 || n % 8 || ldy < n ||
        first_column < 0 || reference_n < n || first_column > reference_n - n ||
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(q)) & 15) || !strata_wmma_gfx11_device()) return false;
    if (k == 10240 && reference_n == 320)
        return q8_gsu_selected<10240, 15>(x, q, d, y, rows, n, ldy, first_column, stream);
    if (k == 2560 && (reference_n == 512 || reference_n == 640 || reference_n == 1664 || reference_n == 5760 || reference_n == 6528))
        return q8_gsu_selected<2560, 3>(x, q, d, y, rows, n, ldy, first_column, stream);
    if (k == 3072 && reference_n == 2560)
        return q8_gsu_selected<3072, 7>(x, q, d, y, rows, n, ldy, first_column, stream);
    return false;
}

bool strata_pf_wide_gsu3_f16(const uint16_t* x, const uint16_t* w, float* y, int rows, int n, int k, int ldy, void* stream) {
    if (k == 2560 && (n == 512 || n == 640 || n == 1664 || n == 5760 || n == 6528))
        return launch_gsu3<2560, 3>(x, w, y, rows, n, ldy, stream);
    if (k == 3072 && n == 2560) return launch_gsu3<3072, 7>(x, w, y, rows, n, ldy, stream);
    return false;
}

bool strata_pf_gemm_narrow_f16_ld(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y,
                                  int64_t T, int64_t N, int64_t K, int64_t ldy, void* stream) {
    if (!X || !W || !Y || T < 64 || N < 1 || N >= 512 || K < pfg::BK || K % pfg::BK != 0 || ldy < N ||
        T > (1LL << 30) || K > (1LL << 30) || ldy > (1LL << 30) || ldx < K || ldw < K || ldx % 8 || ldw % 8 ||
        ldx > (1LL << 30) || ldw > (1LL << 30)) return false;
    if (!strata_wmma_gfx11_device()) return false;
    static const bool hc_increment = [] {
        const char* v = std::getenv("STRATA_PF_HC_DOWN_INCREMENT");
        return v && std::strcmp(v, "1") == 0;
    }();
    if (hc_increment && T <= 512 && N == 320 && K == 10240) {
        const unsigned blocks = (unsigned) (((T + 15) / 16) * (N / 16) / 4);
        pfg::kernel_hc_down_increment<<<blocks, 128, 0, static_cast<hipStream_t>(stream)>>>(
            (const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) ldx, (int) ldw, (int) ldy);
        return hipGetLastError() == hipSuccess;
    }
    const unsigned grid = (unsigned) (((T + pfg::BM - 1) / pfg::BM) * ((N + 127) / 128));
    static const bool bk64 = [] { const char* v = std::getenv("STRATA_PF_NARROW_BK64"); return v && v[0] == '1'; }();
    if (bk64 && K % 64 == 0)
        pfg::kernel_narrow_bk64<<<grid, 256, 0, static_cast<hipStream_t>(stream)>>>((const _Float16*) X, (int) ldx,
            (const _Float16*) W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy);
    else
        pfg::kernel<128, 32><<<grid, 256, 0, static_cast<hipStream_t>(stream)>>>((const _Float16*) X, (const _Float16*) W,
            Y, (int) T, (int) N, (int) K, (int) ldy, 0, (int) ldx, (int) ldw);
    return hipGetLastError() == hipSuccess;
}

bool strata_pf_gemm_f16_ld(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t T,
                           int64_t N, int64_t K, int64_t ldy, float beta, void* stream) {
    if (!X || !W || !Y || T < 64 || N < 512 || K % pfg::BK != 0 || K < pfg::BK) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldx < K || ldw < K || ldx % 8 != 0 || ldw % 8 != 0 || ldx > (1LL << 30) || ldw > (1LL << 30)) return false;
    if (ldy < N || T > (1LL << 30) || N > (1LL << 30) || K > (1LL << 30)) return false;
    if (!strata_wmma_gfx11_device()) return false;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int acc = beta == 1.0f ? 1 : 0;
    const int64_t mt = (T + pfg::BM - 1) / pfg::BM;
    // STRATA_PF_BK64=0: the BK 32 kernel for the wide shapes too (the A/B; the same bits)
    static const bool bk64 = [] { const char* v = std::getenv("STRATA_PF_BK64"); return !v || v[0] != '0'; }();
    static const bool n96 = [] { const char* v = std::getenv("STRATA_PF_GEMM96"); return v && std::strcmp(v,"1")==0; }();
    const uint64_t grid96 = (uint64_t)mt * ((N+95)/96);
    if (n96 && N >= 1024 && bk64 && K % 64 == 0 && grid96 <= UINT32_MAX &&
        (uint64_t)T*ldx*2 <= UINT32_MAX && (uint64_t)N*ldw*2 <= UINT32_MAX) {
        // The burst schedule wins the representative split-engine shapes;
        // keep the interleaved schedule as an independent benchmark control.
        pfg::kernel_bk64_n96<false><<<(unsigned)grid96,128,0,s>>>((const _Float16*)X,(const _Float16*)W,Y,
            (int)T,(int)N,(int)K,(int)ldy,acc,(int)ldx,(int)ldw);
    } else if (N >= 1024 && bk64 && K % 64 == 0) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel_bk64<<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                              (int) ldy, acc, (int) ldx, (int) ldw);
    } else if (N >= 1024) {
        const unsigned grid = (unsigned) (mt * ((N + 255) / 256));
        pfg::kernel<256, 64><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    } else {
        const unsigned grid = (unsigned) (mt * ((N + 127) / 128));
        pfg::kernel<128, 32><<<grid, 256, 0, s>>>((const _Float16*) X, (const _Float16*) W, Y, (int) T, (int) N, (int) K,
                                                  (int) ldy, acc, (int) ldx, (int) ldw);
    }
    return hipGetLastError() == hipSuccess;
}

#else
// Non-HIP / Non-GFX11 compilation fallback
bool strata_pf_hcup_q8_f16(const uint16_t*, const int8_t*, const uint16_t*, const float*, const float*, const float*,
                           float*, uint16_t*, int64_t, int64_t, int64_t, void*) {
    return false;
}
bool strata_pf_q8_k320_f16(const uint16_t*, const int8_t*, const uint16_t*, float*, int, int, int, int, void*) {
    return false;
}
bool strata_pf_q8_gsu3_f16(const uint16_t*, const int8_t*, const uint16_t*, float*, int, int, int, int, int, int, void*) {
    return false;
}
bool strata_pf_hcdown_gsu3_f16(const uint16_t*, const uint16_t*, float*, int, int, int, int, void*) {
    return false;
}
bool strata_pf_wide_gsu3_f16(const uint16_t*, const uint16_t*, float*, int, int, int, int, void*) {
    return false;
}
bool strata_pf_hcup_mix_f16(const uint16_t*, const uint16_t*, const float*, const float*, const float*, float*, uint16_t*,
                            int64_t, int64_t, int64_t, void*) {
    return false;
}
bool strata_pf_gemm_narrow_f16_ld(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t,
                                  int64_t, void*) {
    return false;
}
bool strata_pf_gemm_narrow_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, void*) {
    return false;
}
bool strata_pf_gemm_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float, void*) {
    return false;
}
bool strata_pf_gemm_f16_ld(const uint16_t*, int64_t, const uint16_t*, int64_t, float*, int64_t, int64_t, int64_t,
                           int64_t, float, void*) {
    return false;
}
// No padded product in this build (no gfx11 target): prefill.cpp does not pad an activation's row stride.
bool strata_pf_padded_gemm_available() { return false; }
bool strata_pf_hcdown_bf16(const uint16_t*, int64_t, const uint16_t*, const uint16_t*, int64_t, int64_t, float*, float*,
                           int64_t, int64_t, void*) {
    return false;
}
bool strata_pf_hcdown_exact_bf16(const uint16_t*, int64_t, const uint16_t*, float*, int64_t, int64_t, int64_t, void*) {
    return false;
}
bool strata_wmma_gemm_f16(const uint16_t*, const uint16_t*, float*,
                          int64_t, int64_t, int64_t, int64_t, float,
                          void*) {
    return false;
}

bool strata_wmma_gemm_bf16(const uint16_t*, const uint16_t*, float*,
                           int64_t, int64_t, int64_t, int64_t, float,
                           void*) {
    return false;
}
#endif

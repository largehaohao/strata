// Force-included (after the tree's hip_compat/cuda_runtime.h) into every src/split source of the HIP build: the CUDA runtime and
// cuBLAS names src/split uses that the shared shim does not rename.
#pragma once
#include <hip/hip_runtime.h>
#include <cuda_runtime.h>   // the tree's shim (include/strata/hip_compat): its names are redefined below, so it must come first

#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaGetErrorName hipGetErrorName
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaGraphExecUpdateResultInfo fnx_graph_update_info
struct fnx_graph_update_info {
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
};
inline hipError_t cudaGraphExecUpdate(hipGraphExec_t exec, hipGraph_t g, fnx_graph_update_info* info) {
    return hipGraphExecUpdate(exec, g, &info->errorNode, &info->result);
}

// cuBLAS: the f16 products with tensor-op math are hipBLAS's default on a gfx12 card
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_DEFAULT_MATH

// __ldcg: a load that skips the non-coherent caches (data another block of the same launch wrote, behind the hc read's grid barrier):
// a relaxed agent-scope atomic load of 4 / 8 bytes
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "__ldcg: 4 or 8 bytes");
    if constexpr (sizeof(T) == 4) {
        const unsigned u = __hip_atomic_load(reinterpret_cast<const unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 4);
        return r;
    } else {
        const unsigned long long u = __hip_atomic_load(reinterpret_cast<const unsigned long long*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 8);
        return r;
    }
}

// HIP: a plain cudaMemset goes through the legacy stream, which HIP refuses while any stream of the process is capturing (a buffer first
// needed inside a graph capture, as the window graphs' scratch is); the zeroing runs on a stream of its own per card and thread instead
inline hipError_t fnx_zero_dev(void* p, size_t bytes) {
    static thread_local hipStream_t st[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        (void) hipGetLastError();
        return hipMemset(p, 0, bytes);
    }
    if (!st[dev]) {
        const hipError_t e = hipStreamCreateWithFlags(&st[dev], hipStreamNonBlocking);
        if (e != hipSuccess) return e;
    }
    const hipError_t e = hipMemsetAsync(p, 0, bytes, st[dev]);
    return e != hipSuccess ? e : hipStreamSynchronize(st[dev]);
}
#define FNX_ZERO_DEV(p, bytes) fnx_zero_dev((p), (bytes))

// The synchronous copies and sets (cudaMemcpy, cudaMemset) are the legacy-stream kind, which HIP refuses while ANY stream of the process is
// capturing: the engine captures one card's window graph while the next card's host side still uploads its row tables. They run on a private
// non-blocking stream per card and thread, then wait for it.
inline hipStream_t fnx_side_stream() {
    static thread_local hipStream_t st[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        (void) hipGetLastError();
        return nullptr;
    }
    if (!st[dev] && hipStreamCreateWithFlags(&st[dev], hipStreamNonBlocking) != hipSuccess) {
        (void) hipGetLastError();
        st[dev] = nullptr;
    }
    return st[dev];
}
inline hipError_t fnx_memcpy(void* d, const void* s, size_t n, hipMemcpyKind k) {
    hipStream_t st = fnx_side_stream();
    if (!st) return hipMemcpy(d, s, n, k);
    const hipError_t e = hipMemcpyAsync(d, s, n, k, st);
    return e != hipSuccess ? e : hipStreamSynchronize(st);
}
inline hipError_t fnx_memset(void* p, int v, size_t n) {
    hipStream_t st = fnx_side_stream();
    if (!st) return hipMemset(p, v, n);
    const hipError_t e = hipMemsetAsync(p, v, n, st);
    return e != hipSuccess ? e : hipStreamSynchronize(st);
}
#undef cudaMemcpy
#undef cudaMemset
#define cudaMemcpy fnx_memcpy
#define cudaMemset fnx_memset

// Wave32 register-only butterfly shuffle on RDNA (gfx10+): DPP row_xmask for offsets 1/2/4/8
// and v_permlanex16_b32 for offset 16, eliminating ds_bpermute LDS round-trips while preserving
// each call site's exact reduction order (bitwise identical to __shfl_xor).
template <typename T>
__device__ __forceinline__ T fnx_shfl_xor_sync(uint32_t mask, T value, int lane_mask, int width = 32) {
    ::strata::hip_compat::require_full_wave_mask(mask);
#if defined(__HIP_DEVICE_COMPILE__) && !defined(STRATA_HIP_GFX906) && !defined(FNX_HIP_NO_DPP)
    if constexpr (sizeof(T) == 4) {
        if (__builtin_constant_p(lane_mask) && __builtin_constant_p(width) && lane_mask > 0 && lane_mask < width) {
            int bits;
            __builtin_memcpy(&bits, &value, 4);
            if (lane_mask == 1) {
                bits = __builtin_amdgcn_update_dpp(0, bits, 0x161, 0xf, 0xf, false);
                T out;
                __builtin_memcpy(&out, &bits, 4);
                return out;
            } else if (lane_mask == 2) {
                bits = __builtin_amdgcn_update_dpp(0, bits, 0x162, 0xf, 0xf, false);
                T out;
                __builtin_memcpy(&out, &bits, 4);
                return out;
            } else if (lane_mask == 4) {
                bits = __builtin_amdgcn_update_dpp(0, bits, 0x164, 0xf, 0xf, false);
                T out;
                __builtin_memcpy(&out, &bits, 4);
                return out;
            } else if (lane_mask == 8) {
                bits = __builtin_amdgcn_update_dpp(0, bits, 0x168, 0xf, 0xf, false);
                T out;
                __builtin_memcpy(&out, &bits, 4);
                return out;
            } else if (lane_mask == 16 && width == 32) {
                bits = __builtin_amdgcn_permlanex16(bits, bits, 0x76543210u, 0xfedcba98u, false, false);
                T out;
                __builtin_memcpy(&out, &bits, 4);
                return out;
            }
        }
    }
#endif
    return __shfl_xor(value, lane_mask, width);
}
#undef __shfl_xor_sync
#define __shfl_xor_sync(...) (fnx_shfl_xor_sync(__VA_ARGS__))

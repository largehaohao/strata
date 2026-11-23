// The prompt path's routed experts on the tensor cores (moe_mma.cuh), in an object of their own so a change to them
// recompiles in seconds.
#include "env.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "qdot.cuh"   // block layouts and codebook tables (strata-quant-common.h)

namespace fnx {

#if FNX_HIP && defined(FNX_MOE_MMA_DISPATCH_WIDE) && !defined(FNX_MOE_MMA_WIDE)
void moe_gate_up_mma_wide(const DW&, size_t, const BigJob*, const int*, int, const uint16_t*, float*, void*);
void moe_down_mma_wide(const DW&, size_t, const BigJob*, const int*, int, const uint16_t*, float*, void*);
static bool moe_wide_requested(int type) {
    static const bool on = [] {
        const char* v = std::getenv("FNX_PF_MOE_WIDE");
        return v && v[0] == '1' && v[1] == 0;
    }();
    return on && (type == T_IQ3_S || type == T_IQ3_XXS || type == T_IQ2_S || type == T_IQ2_XXS ||
                  type == T_IQ4_NL || type == T_IQ4_XS || type == T_Q5_0 || type == T_Q2_0 || type == T_Q4_0);
}
#endif

namespace {

void ck_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }

#include "moe_w.cuh"
#include "moe_mma.cuh"
#include "moe_mma_iq.cuh"

bool trim_token_tiles() {
#if FNX_HIP && defined(FNX_MOE_MMA_WIDE)
    static const bool enabled = [] {
        const char* value = std::getenv("FNX_PF_MOE_TRIM_TILES");
        return value && value[0] == '1' && value[1] == 0;
    }();
    return enabled;
#else
    return false;
#endif
}

#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
bool pf_moe_hgrid_enabled() {
    static const bool enabled = env_flag("FNX_PF_MOE_HGRID");
    return enabled;
}
#endif

template <int TYPE>
void mma_launch(const MoeW& w, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, float* out, cudaStream_t s) {
#if defined(FNX_MOE_MMA_WIDE) && FNX_HIP
    if (pf_moe_hgrid_enabled()) {
        if constexpr (MmHgridInfo<TYPE>::has_grid && mm_smem<MM_BM, TYPE, true>() <= 48 * 1024) {
            constexpr size_t sm = mm_smem<MM_BM, TYPE, true>();
            k_moe_mma<TYPE, MM_BM, false, true><<<dim3((w.nrows + MM_BM - 1) / MM_BM, njobs_cap), MM_THREADS, sm, s>>>(
                w, jobs, njobs, x, w.ncols, out, nullptr, trim_token_tiles());
            return;
        }
    }
#endif
    constexpr size_t sm = mm_smem<MM_BM>();
    static_assert(sm <= 48 * 1024, "moe_mma: shared memory past the default limit");
    k_moe_mma<TYPE, MM_BM><<<dim3((w.nrows + MM_BM - 1) / MM_BM, njobs_cap), MM_THREADS, sm, s>>>(
        w, jobs, njobs, x, w.ncols, out, nullptr, trim_token_tiles());
}

void moe_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, float* out, void* st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    if (w.ncols % 32) throw std::runtime_error("moe_mma: columns not in units of 32");
    cudaStream_t s = (cudaStream_t) st;
    switch (A.type) {
        case T_Q4_K: mma_launch<T_Q4_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_K: mma_launch<T_Q5_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_1: mma_launch<T_Q5_1>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_0: mma_launch<T_Q5_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ2_XXS: mma_launch<T_IQ2_XXS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q8_0: mma_launch<T_Q8_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q4_0: mma_launch<T_Q4_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q2_0:
            if (w.ncols % 64 || w.row_a != (size_t) w.ncols / 64 * 18)
                throw std::runtime_error("moe_mma: packed Q2_0 requires whole 64-value blocks");
            mma_launch<T_Q2_0>(w, jobs, njobs, njobs_cap, x, out, s);
            break;
        case T_IQ4_NL: mma_launch<T_IQ4_NL>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_XS: mma_launch<T_IQ4_XS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_S: mma_launch<T_IQ3_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_XXS: mma_launch<T_IQ3_XXS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ2_S: mma_launch<T_IQ2_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        default: throw std::runtime_error("moe_mma: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch("moe_mma");
}
}  // namespace
void moe_gate_up_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* xh, float* gu, void* st) {
#if FNX_HIP && defined(FNX_MOE_MMA_DISPATCH_WIDE) && !defined(FNX_MOE_MMA_WIDE)
    if (moe_wide_requested(A.type)) { moe_gate_up_mma_wide(A, sa, jobs, njobs, njobs_cap, xh, gu, st); return; }
#endif
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(xh), gu, st);
}
void swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, void* st) {
    k_swiglu_h16<<<(unsigned) std::min<size_t>(4096, ((size_t) P * ffs + 255) / 256), 256, 0, (cudaStream_t) st>>>(gu, ffs, P, reinterpret_cast<__half*>(h16));
    ck_launch("swiglu_h16");
}
void moe_down_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, float* part, void* st) {
#if FNX_HIP && defined(FNX_MOE_MMA_DISPATCH_WIDE) && !defined(FNX_MOE_MMA_WIDE)
    if (moe_wide_requested(A.type)) { moe_down_mma_wide(A, sa, jobs, njobs, njobs_cap, h16, part, st); return; }
#endif
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(h16), part, st);
}

}  // namespace fnx

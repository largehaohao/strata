// A separate instantiation keeps the original prompt kernel available for
// exact comparison and flag-off serving. Both retain FP16 inputs / FP32 sums.
#define FNX_MOE_MMA_WIDE 1
#define moe_gate_up_mma moe_gate_up_mma_wide
#define moe_down_mma moe_down_mma_wide
#define swiglu_h16 swiglu_h16_wide
#include "moe_mma.cu"

namespace fnx {
namespace {
template<int TYPE>
void fused_gate_launch(const MoeW& w, const BigJob* jobs, const int* count, int capacity,
                       const uint16_t* x, uint16_t* h, void* stream) {
    k_moe_mma<TYPE,MM_BM,true><<<dim3(w.nrows/MM_BM,capacity),MM_THREADS,mm_smem<MM_BM>(),(cudaStream_t)stream>>>(
        w,jobs,count,reinterpret_cast<const __half*>(x),w.ncols,nullptr,reinterpret_cast<__half*>(h),trim_token_tiles());
}
}
bool moe_gate_up_swiglu_supported(const DW& a) {
    if(a.n_out<=0 || a.n_out%MM_BM || a.n_in<=0 || a.n_in%32) return false;
    switch(a.type) {
        case T_IQ3_S:return a.n_in%256==0 && a.row_bytes==(size_t)a.n_in/256*110;
        case T_IQ3_XXS:return a.n_in%256==0 && a.row_bytes==(size_t)a.n_in/256*98;
        case T_IQ2_S:return a.n_in%256==0 && a.row_bytes==(size_t)a.n_in/256*82;
        case T_IQ4_XS:return a.n_in%256==0 && a.row_bytes==(size_t)a.n_in/256*136;
        case T_IQ4_NL:case T_Q4_0:return a.row_bytes==(size_t)a.n_in/32*18;
        case T_Q2_0:return a.n_in%64==0 && a.row_bytes==(size_t)a.n_in/64*18;
        default:return false;
    }
}
bool moe_gate_up_swiglu_mma(const DW& a, size_t stride, const BigJob* jobs, const int* count, int capacity,
                           const uint16_t* x, uint16_t* h, void* stream) {
    if(!moe_gate_up_swiglu_supported(a)) return false;
    const MoeW w{(const char*)a.p,stride,a.row_bytes,a.n_out,a.n_in,a.ep};
    switch(a.type) {
        case T_IQ3_S: fused_gate_launch<GGML_TYPE_IQ3_S>(w,jobs,count,capacity,x,h,stream); break;
        case T_IQ3_XXS: fused_gate_launch<GGML_TYPE_IQ3_XXS>(w,jobs,count,capacity,x,h,stream); break;
        case T_IQ2_S: fused_gate_launch<GGML_TYPE_IQ2_S>(w,jobs,count,capacity,x,h,stream); break;
        case T_IQ4_XS: fused_gate_launch<GGML_TYPE_IQ4_XS>(w,jobs,count,capacity,x,h,stream); break;
        case T_IQ4_NL: fused_gate_launch<GGML_TYPE_IQ4_NL>(w,jobs,count,capacity,x,h,stream); break;
        case T_Q4_0: fused_gate_launch<GGML_TYPE_Q4_0>(w,jobs,count,capacity,x,h,stream); break;
        case T_Q2_0:fused_gate_launch<GGML_TYPE_Q2_0>(w,jobs,count,capacity,x,h,stream); break;
        default:return false;
    }
    ck_launch("fused gate/up SwiGLU");
    return true;
}
}

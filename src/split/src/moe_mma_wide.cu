// A separate instantiation keeps the original prompt kernel available for
// exact comparison and flag-off serving. Both retain FP16 inputs / FP32 sums.
#define FNX_MOE_MMA_WIDE 1
#define moe_gate_up_mma moe_gate_up_mma_wide
#define moe_down_mma moe_down_mma_wide
#define swiglu_h16 swiglu_h16_wide
#include "moe_mma.cu"

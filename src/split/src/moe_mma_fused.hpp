#pragma once
#include <cstddef>
#include <cstdint>
namespace fnx {
struct DW;
struct BigJob;
bool moe_gate_up_swiglu_supported(const DW&);
// False leaves the output untouched and requests the ordinary two-step path.
bool moe_gate_up_swiglu_mma(const DW&, size_t, const BigJob*, const int*, int,
                          const uint16_t*, uint16_t*, void*);
}

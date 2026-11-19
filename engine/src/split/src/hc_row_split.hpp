#pragma once
// Optional two-card HIP HC-down row partition. Allocate after ordinary buffers.
namespace fnx {
void hc_row_split_prepare(int card, bool peer, int lanes, unsigned* errors);
}

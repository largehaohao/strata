// Compact only exact BF16 values. Never round a model weight.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace fnx {
inline bool pack_lossless_bf16(const float* src, size_t n, std::vector<uint16_t>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, src + i, sizeof(bits));
        if (bits & 0xffffu) { out.clear(); return false; }
        out[i] = (uint16_t) (bits >> 16);
    }
    return true;
}
} // namespace fnx

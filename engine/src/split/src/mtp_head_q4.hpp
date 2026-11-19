// Optional draft-only Q8_0 -> Q4_0 head conversion. The main output head and
// the order/content of the draft vocabulary are unchanged.
#pragma once

#include "tensor_convert.hpp"
#include <algorithm>
#include <limits>

namespace fnx {

// ggml's quantize_row_q4_0_ref: the first signed maximum, and rounded products
// before the 8.5 offset. Keep this host conversion independent of contraction.
inline void mtp_head_q4_block(const float* x, uint8_t* out) {
    float amax = 0.f, value = 0.f;
    for (int j = 0; j < 32; ++j) {
        if (!std::isfinite(x[j])) throw std::runtime_error("mtp Q4 head: non-finite weight");
        if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); value = x[j]; }
    }
    const float d = value / -8.f, id = d ? 1.f / d : 0.f;
    const uint16_t h = conv::f2h(d);
    if ((h & 0x7c00u) == 0x7c00u) throw std::runtime_error("mtp Q4 head: scale exceeds FP16");
    std::memcpy(out, &h, 2);
    for (int j = 0; j < 16; ++j) {
        volatile float lo = x[j] * id, hi = x[j + 16] * id;
        out[2 + j] = (uint8_t) (std::min(15, (int) (lo + 8.5f)) | (std::min(15, (int) (hi + 8.5f)) << 4));
    }
}

inline std::vector<uint8_t> mtp_head_q4_rows(const Tensor& tensor, const std::vector<int>& rows) {
    if (tensor.type != T_Q8_0 || !tensor.data || tensor.ne[0] <= 0 || tensor.ne[0] % 32 || tensor.nrows() <= 0)
        throw std::runtime_error("mtp Q4 head: expected available Q8_0 rows with whole blocks");
    const size_t nb = (size_t) tensor.ne[0] / 32;
    if (nb > std::numeric_limits<size_t>::max() / 34 ||
        (size_t) tensor.nrows() > tensor.nbytes / (nb * 34) ||
        rows.size() > std::numeric_limits<size_t>::max() / (nb * 18))
        throw std::runtime_error("mtp Q4 head: invalid tensor size");
    std::vector<uint8_t> out(rows.size() * nb * 18);
    for (size_t r = 0; r < rows.size(); ++r) {
        if (rows[r] < 0 || rows[r] >= tensor.nrows()) throw std::runtime_error("mtp Q4 head: row out of range");
        for (size_t b = 0; b < nb; ++b) {
            float x[32];
            conv::dequantize(T_Q8_0, tensor.data + ((size_t) rows[r] * nb + b) * 34, 32, x);
            mtp_head_q4_block(x, out.data() + (r * nb + b) * 18);
        }
    }
    return out;
}

}  // namespace fnx

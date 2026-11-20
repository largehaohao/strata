// Result buffers shared by the sampled verify and draft paths with their host readers.
#pragma once
#include <cstddef>
#include <type_traits>
#include "env.hpp"
#include "kernels.hpp"
namespace fnx {
struct VerifySamplePacket {
    // Right-align the active scores and left-align IDs so T rows and both errors copy together.
    float values[MAXT * TOPK_N];
    unsigned top_error, window_error;
    int ids[MAXT * TOPK_N];
};
static_assert(offsetof(VerifySamplePacket, ids) == MAXT * TOPK_N * sizeof(float) + 2 * sizeof(unsigned),
              "verify packet layout");
struct alignas(64) DraftSamplePacket {
    float values[TOPK_N];
    int ids[TOPK_N];
    float best_v[MAXT];
    int best_i[MAXT];
    unsigned top_error, error;
};
struct alignas(64) DraftSampleInput {
    int ctl[2];
    unsigned reserved[14];
    float emb[MAXT * N];
};
static_assert(std::is_standard_layout<DraftSamplePacket>::value, "sample packet layout");
static_assert(offsetof(DraftSampleInput, emb) == 64, "cache-line input header");
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(unsigned) == 4, "GPU word sizes");
inline size_t draft_sample_input_bytes(int T) { return offsetof(DraftSampleInput, emb) + (size_t)T * N * sizeof(float); }
inline int draft_sample_io_mode() {
    static const int mode = [] {
        const auto n = env_int("FNX_DRAFT_SAMPLE_IO", 0);
        if (n < 0 || n > 2) throw std::runtime_error("FNX_DRAFT_SAMPLE_IO: expected 0, 1 or 2");
#if !FNX_HIP || FNX_NCARD != 2 || FNX_27B
        if (n) throw std::runtime_error("FNX_DRAFT_SAMPLE_IO requires two-card HIP Flash-Next");
#endif
        return (int)n;
    }();
    return mode;
}
}  // namespace fnx

// Optional two-card Flash-Next attention layout. Each card owns one KV head and
// twelve query heads, retaining the original position partitions and merge order.
#pragma once
#include "model.hpp"
#include "env.hpp"
namespace fnx {
inline bool attn_head_split() {
    static const bool enabled = [] {
        const long long v = env_int("FNX_ATTN_HEAD_SPLIT", 0);
        if (v != 0 && v != 1) throw std::runtime_error("FNX_ATTN_HEAD_SPLIT: expected 0 or 1");
#if !(FNX_HIP && FNX_NCARD == 2 && !FNX_27B)
        if (v) throw std::runtime_error("FNX_ATTN_HEAD_SPLIT requires the two-card Flash-Next HIP build");
#endif
        return v != 0;
    }();
    return enabled;
}
inline int attn_projected_q_heads() { return attn_head_split() ? NQH / NCARD : NQH; }
inline int attn_projected_kv_heads() { return attn_head_split() ? 1 : NKVH; }
inline int attn_position_partitions() { return attn_head_split() ? 2 : 1; }
inline void attn_head_check_runtime(int lanes, bool batched_drafts) {
    if (attn_head_split() && (lanes != 1 || batched_drafts))
        throw std::runtime_error("FNX_ATTN_HEAD_SPLIT requires one lane and FNX_BATCH_DRAFTS=0");
}
}  // namespace fnx

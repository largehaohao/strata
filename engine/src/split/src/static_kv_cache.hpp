// Host-only plan for lending an unused K/V suffix to the static decode cache.
// The cache keeps its original slot count when K/V occupies the full capacity.
#pragma once
#include "static_cache_layout.hpp"

namespace fnx {
struct StaticKvCachePlan {
    StaticCacheLayout cache;
    int pool = -1;
    int extra_slots = 0;
    size_t prefix_bytes = 0;
    size_t region_bytes = 0;
};

inline StaticKvCachePlan static_kv_cache_plan(const StaticCacheLayout& original, size_t full_kv, size_t base_kv, int pool_hint = -1) {
    StaticKvCachePlan plan;
    plan.cache = original;
    if (!full_kv && !base_kv) return plan;
    if (!base_kv || base_kv > full_kv || base_kv % 256 || full_kv % 256 || original.pools.empty() ||
        pool_hint < -1 || pool_hint >= (int) original.pools.size())
        throw std::runtime_error("static K/V cache: invalid page region");
    // The pool serving the most layers usually has the most slots. It can use
    // the additional capacity without reducing either pool's original budget.
    for (int p = 0; p < (int) original.pools.size(); ++p)
        if (plan.pool < 0 || original.pools[p].slots > original.pools[plan.pool].slots) plan.pool = p;
    if (pool_hint >= 0) plan.pool = pool_hint;
    auto& pool = plan.cache.pools[plan.pool];
    const size_t stride = pool.stride(), spare = full_kv - base_kv;
    if (!stride || pool.slots <= 0) throw std::runtime_error("static K/V cache: invalid cache pool");
    const size_t extra = spare / stride + (spare % stride != 0);
    if (extra > (size_t) (std::numeric_limits<int>::max() - original.slots))
        throw std::runtime_error("static K/V cache: too many borrowed slots");
    plan.extra_slots = (int) extra;
    pool.slots += plan.extra_slots;
    if ((size_t) pool.slots > (std::numeric_limits<size_t>::max() - base_kv) / stride)
        throw std::runtime_error("static K/V cache: region size overflow");
    plan.prefix_bytes = base_kv;
    plan.region_bytes = base_kv + (size_t) pool.slots * stride;
    plan.cache.slots = 0;
    plan.cache.bytes = 0;
    for (auto& p : plan.cache.pools) {
        p.start = plan.cache.slots;
        plan.cache.slots += p.slots;
        const size_t bytes = (size_t) p.slots * p.stride();
        if (bytes > std::numeric_limits<size_t>::max() - plan.cache.bytes)
            throw std::runtime_error("static K/V cache: total size overflow");
        plan.cache.bytes += bytes;
    }
    return plan;
}
}  // namespace fnx

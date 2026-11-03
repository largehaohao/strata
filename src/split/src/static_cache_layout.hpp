// Host-only sizing for the static expert cache. Grouping changes slot padding,
// not the weight bytes, placement or total cache memory budget.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fnx {
struct StaticCacheShape { size_t gu, dn; };
struct StaticCachePool {
    size_t gu = 0, dn = 0;
    int start = 0, slots = 0, layers = 0;
    size_t stride() const { return gu + dn; }
};
struct StaticCacheLayout {
    std::vector<int> pool_of;
    std::vector<StaticCachePool> pools;
    int slots = 0;
    size_t bytes = 0;
};

inline StaticCacheLayout static_cache_layout(const std::vector<StaticCacheShape>& shapes, int budget_slots, int groups,
                                             int min_slots = 160) {
    if (shapes.empty() || shapes.size() > 1024 || min_slots < 32 || min_slots % 32 ||
        budget_slots < min_slots || budget_slots > 4096 || budget_slots % 32 || (groups != 1 && groups != 2))
        throw std::runtime_error("static cache layout: invalid shapes, slot count or group count");
    size_t gu = 0, dn = 0;
    std::vector<size_t> sizes;
    for (const auto& s : shapes) {
        if (!s.gu || !s.dn || s.gu % 16 || s.dn % 16 || s.gu > std::numeric_limits<size_t>::max() - s.dn)
            throw std::runtime_error("static cache layout: invalid expert strides");
        gu = std::max(gu, s.gu); dn = std::max(dn, s.dn);
        sizes.push_back(s.gu + s.dn);
    }
    if (gu > std::numeric_limits<size_t>::max() - dn || gu + dn > std::numeric_limits<size_t>::max() / budget_slots)
        throw std::runtime_error("static cache layout: size overflow");
    const size_t budget = (gu + dn) * budget_slots;
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    const bool split = groups == 2 && sizes.size() > 1;
    const size_t cut = sizes[(sizes.size() - 1) / 2];
    StaticCacheLayout layout;
    layout.pools.resize(split ? 2 : 1);
    for (const auto& s : shapes) {
        const int p = split && s.gu + s.dn > cut;
        layout.pool_of.push_back(p);
        auto& pool = layout.pools[p];
        pool.gu = std::max(pool.gu, s.gu); pool.dn = std::max(pool.dn, s.dn); ++pool.layers;
    }
    if (!split) {
        layout.pools[0].slots = budget_slots;
    } else {
        // Every pool can hold a complete MAXT * TOPK window. Divide the rest
        // by layer count; compare integer ratios to avoid rounding drift.
        size_t left = budget;
        for (auto& p : layout.pools) {
            const size_t bytes = p.stride() * min_slots;
            if (bytes > left) throw std::runtime_error("static cache layout: budget too small for two full windows");
            p.slots = min_slots; left -= bytes;
        }
        int total_slots = 2 * min_slots;
        // Also bound metadata growth when one synthetic/outlier shape is huge.
        // At most twice the original slot count can be allocated.
        while (total_slots + 32 <= 2 * budget_slots) {
            int chosen = -1;
            for (int p = 0; p < 2; ++p) {
                const auto& a = layout.pools[p];
                if (a.stride() * 32 > left) continue;
                if (chosen < 0 || a.slots * layout.pools[chosen].layers < layout.pools[chosen].slots * a.layers)
                    chosen = p;
            }
            if (chosen < 0) break;
            auto& p = layout.pools[chosen];
            p.slots += 32; total_slots += 32; left -= p.stride() * 32;
        }
    }
    for (auto& p : layout.pools) {
        p.start = layout.slots;
        layout.slots += p.slots; layout.bytes += p.stride() * p.slots;
    }
    return layout;
}
} // namespace fnx

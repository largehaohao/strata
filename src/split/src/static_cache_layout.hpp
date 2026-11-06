// Host-only sizing for the static expert cache. Grouping changes slot padding,
// not the weight bytes, placement or total cache memory budget.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
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
                                             int min_slots = 160, bool min_padding = false, int pool1_slots = 0) {
    if (shapes.empty() || shapes.size() > 1024 || min_slots < 32 || min_slots % 32 ||
        budget_slots < min_slots || budget_slots > 4096 || budget_slots % 32 || (groups != 1 && groups != 2))
        throw std::runtime_error("static cache layout: invalid shapes, slot count or group count");
    if (pool1_slots && (groups != 2 || pool1_slots < min_slots || pool1_slots > 4096 || pool1_slots % 32))
        throw std::runtime_error("static cache layout: invalid pool 1 slot count");
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
    std::vector<int> partition;
    for (const auto& s : shapes) partition.push_back(split && s.gu + s.dn > cut);
    if (min_padding && split) {
        // A median of total bytes can mix a wide GU/small DN with a narrow
        // GU/wide DN, wasting both dimensions. Compare cuts on GU, DN and
        // total size using the actual independently padded pool dimensions.
        // This is O(n^2), bounded by the number of model layers; it reads no
        // benchmark routes. Preserve the original partition on equal cost.
        using Cost = std::pair<size_t, size_t>;
        auto cost = [&](const std::vector<int>& assignment, Cost& out) {
            StaticCachePool p[2];
            for (size_t i = 0; i < shapes.size(); ++i) {
                auto& pool = p[assignment[i]];
                pool.gu = std::max(pool.gu, shapes[i].gu);
                pool.dn = std::max(pool.dn, shapes[i].dn);
                ++pool.layers;
            }
            if (!p[0].layers || !p[1].layers) return false;
            size_t left = budget;
            for (const auto& pool : p) {
                const size_t bytes = pool.stride() * min_slots;
                if (bytes > left) return false;
                left -= bytes;
            }
            // Exact weighted average as quotient + remainder; multiplying
            // a potentially large stride by all layers would overflow.
            size_t whole = 0, remainder = 0;
            for (const auto& pool : p) {
                whole += (pool.stride() / shapes.size()) * pool.layers;
                remainder += (pool.stride() % shapes.size()) * pool.layers;
            }
            out = {whole + remainder / shapes.size(), remainder % shapes.size()};
            return true;
        };
        Cost best{std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        cost(partition, best);
        std::vector<int> candidate(shapes.size());
        for (int axis = 0; axis < 3; ++axis) {
            auto key = [&](const StaticCacheShape& s) { return axis == 0 ? s.gu : axis == 1 ? s.dn : s.gu + s.dn; };
            std::vector<size_t> cuts;
            for (const auto& s : shapes) cuts.push_back(key(s));
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            for (size_t boundary : cuts) {
                for (size_t i = 0; i < shapes.size(); ++i) candidate[i] = key(shapes[i]) > boundary;
                Cost next;
                if (cost(candidate, next) && next < best) {
                    best = next;
                    partition = candidate;
                }
            }
        }
    }
    StaticCacheLayout layout;
    layout.pools.resize(split ? 2 : 1);
    layout.pool_of = partition;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto& s = shapes[i];
        const int p = partition[i];
        auto& pool = layout.pools[p];
        pool.gu = std::max(pool.gu, s.gu); pool.dn = std::max(pool.dn, s.dn); ++pool.layers;
    }
    if (!split) {
        if (pool1_slots) throw std::runtime_error("static cache layout: pool 1 requires two nonempty groups");
        layout.pools[0].slots = budget_slots;
    } else if (pool1_slots) {
        // Repartition the original byte budget. Extra K/V capacity is added
        // later and must not be charged to both pools here.
        auto& p0 = layout.pools[0];
        auto& p1 = layout.pools[1];
        if ((size_t)pool1_slots > budget / p1.stride() || pool1_slots > 2 * budget_slots - min_slots)
            throw std::runtime_error("static cache layout: pool 1 exceeds the cache budget");
        const size_t remaining = budget - (size_t)pool1_slots * p1.stride();
        const size_t cap0 = (2 * budget_slots - pool1_slots) / 32;
        p0.slots = (int)(std::min(remaining / p0.stride() / 32, cap0) * 32);
        if (p0.slots < min_slots)
            throw std::runtime_error("static cache layout: pool 0 cannot hold a complete window");
        p1.slots = pool1_slots;
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

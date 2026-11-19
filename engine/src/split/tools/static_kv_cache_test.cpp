#include "static_kv_cache.hpp"
#include <iostream>

int main() {
    try {
        int checks = 0;
        for (int groups : {1, 2}) for (int slots : {320, 1024, 4096}) {
            std::vector<fnx::StaticCacheShape> shapes(40, {4096, 2048});
            shapes.insert(shapes.end(), 8, {8192, 4096});
            const auto original = fnx::static_cache_layout(shapes, slots, groups);
            const auto off = fnx::static_kv_cache_plan(original, 0, 0);
            if (off.pool != -1 || off.cache.bytes != original.bytes || off.cache.slots != original.slots)
                throw std::runtime_error("disabled layout changed");
            ++checks;
            for (size_t full : {256u, 4096u, 40960u, 327680u}) for (size_t base : {size_t(256), full}) for (int pool_hint = -1; pool_hint < (int) original.pools.size(); ++pool_hint) {
                const auto plan = fnx::static_kv_cache_plan(original, full, base, pool_hint);
                if (pool_hint >= 0 && plan.pool != pool_hint) throw std::runtime_error("pool choice ignored");
                const auto& pool = plan.cache.pools[plan.pool];
                const int kept = original.pools[plan.pool].slots;
                const size_t stride = pool.stride();
                if (plan.cache.slots != original.slots + plan.extra_slots ||
                    plan.cache.bytes + base < original.bytes + full ||
                    plan.cache.bytes + base >= original.bytes + full + stride)
                    throw std::runtime_error("shared allocation budget");
                for (size_t used = base; used <= full; used += 256) {
                    const size_t lent = (used - base) / stride + ((used - base) % stride != 0);
                    const size_t first_active = base + lent * stride;
                    if (first_active < used || (size_t) pool.slots - lent < (size_t) kept ||
                        plan.region_bytes - first_active < (size_t) kept * stride)
                        throw std::runtime_error("K/V overlaps active slots or consumes base cache");
                    ++checks;
                }
                for (size_t p = 0; p < plan.cache.pools.size(); ++p) {
                    if ((int) p != plan.pool && plan.cache.pools[p].slots != original.pools[p].slots)
                        throw std::runtime_error("another cache pool changed");
                    if (p && plan.cache.pools[p].start != plan.cache.pools[p-1].start + plan.cache.pools[p-1].slots)
                        throw std::runtime_error("slot metadata overlaps");
                }
            }
            for (auto bytes : {std::make_pair(size_t(0), size_t(256)), {256, 512}, {512, 1}, {513, 256}}) {
                bool failed = false;
                try { fnx::static_kv_cache_plan(original, bytes.first, bytes.second); }
                catch (const std::runtime_error&) { failed = true; }
                if (!failed) throw std::runtime_error("invalid page sizes accepted");
                ++checks;
            }
        }
        std::cout << "PASS " << checks << " shared K/V cache layout checks\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

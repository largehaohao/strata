#include "static_cache_layout.hpp"
#include "static_kv_cache.hpp"
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <limits>
using namespace fnx;
static void need(bool b) { if (!b) throw std::runtime_error("cache layout assertion failed"); }
static void invalid(const std::vector<StaticCacheShape>& s, int slots, int groups, int minimum = 160) {
    try { (void) static_cache_layout(s, slots, groups, minimum); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid layout was accepted");
}
static void check(const std::vector<StaticCacheShape>& shapes, int budget_slots, int groups, bool padding = false) {
    const auto a = static_cache_layout(shapes, budget_slots, groups, 160, padding);
    need(a.pools.size() >= 1 && a.pools.size() <= (size_t) groups);
    need(a.pool_of.size() == shapes.size());
    size_t gu = 0, dn = 0, bytes = 0; int slots = 0, layers = 0;
    for (const auto& s : shapes) { gu = std::max(gu, s.gu); dn = std::max(dn, s.dn); }
    for (const auto& p : a.pools) {
        need(p.start == slots && p.slots >= 160 && p.slots % 32 == 0 && p.layers > 0);
        slots += p.slots; layers += p.layers; bytes += p.slots * p.stride();
    }
    need(slots == a.slots && bytes == a.bytes && bytes <= (gu + dn) * budget_slots);
    need(slots <= 2 * budget_slots && layers == (int) shapes.size());
    for (size_t l = 0; l < shapes.size(); ++l) {
        need(a.pool_of[l] >= 0 && a.pool_of[l] < (int) a.pools.size());
        const auto& p = a.pools[a.pool_of[l]];
        need(shapes[l].gu <= p.gu && shapes[l].dn <= p.dn);
    }
    if (groups == 1) need(a.slots == budget_slots && a.bytes == (gu + dn) * budget_slots);
    auto reversed = shapes; std::reverse(reversed.begin(), reversed.end());
    const auto b = static_cache_layout(reversed, budget_slots, groups, 160, padding);
    need(a.slots == b.slots && a.bytes == b.bytes && a.pools.size() == b.pools.size());
    for (size_t l = 0; l < shapes.size(); ++l) need(a.pool_of[l] == b.pool_of[shapes.size() - l - 1]);
    if (padding) {
        const auto old = static_cache_layout(shapes, budget_slots, groups);
        size_t before = 0, after = 0;
        for (const auto& p : old.pools) before += p.stride() * p.layers;
        for (const auto& p : a.pools) after += p.stride() * p.layers;
        need(after <= before);
    }
}
int main() {
    const std::vector<StaticCacheShape> four{{256,128},{320,160},{384,192},{448,224}};
    check(four, 1024, 1); check(four, 352, 2);
    const auto plan = static_cache_layout(four, 352, 2);
    need(plan.pool_of == std::vector<int>({0,0,1,1}));
    need(plan.pools[0].gu == 320 && plan.pools[0].dn == 160);
    need(plan.pools[1].gu == 448 && plan.pools[1].dn == 224);
    need(plan.pools[1].start == plan.pools[0].slots);
    const auto uniform = static_cache_layout({{256,128},{256,128}}, 160, 2);
    need(uniform.pools.size() == 1 && uniform.slots == 160);
    // Independent GU/DN maxima can exceed every individual layer's total size.
    check({{1024,16},{16,1024},{2048,16},{16,2048}}, 512, 2);
    check({{16,16},{size_t(1) << 40,16}}, 1024, 2); // bounded slot growth
    // The real Q2-down model: a median of total bytes mixes opposing GU/DN
    // dimensions. The padding policy recovers slots within the same budget.
    std::vector<StaticCacheShape> packed;
    auto add = [&](int n, size_t gu, size_t dn) { packed.insert(packed.end(), n, {gu, dn}); };
    add(5,524800,230400); add(15,524800,460800); add(3,627200,230400);
    add(14,627200,460800); add(1,704000,230400); add(9,704000,460800); add(1,870400,460800);
    const auto old_packed = static_cache_layout(packed, 1024, 2);
    const auto new_packed = static_cache_layout(packed, 1024, 2, 160, true);
    need(old_packed.slots == 1088 && new_packed.slots == 1184);
    need(new_packed.pools[0].layers == 37 && new_packed.pools[0].gu == 627200 && new_packed.pools[0].dn == 460800);
    const auto kv = static_kv_cache_plan(new_packed, size_t(172) * 1517568, size_t(16) * 1517568);
    need(kv.extra_slots == 218 && kv.cache.slots == 1402 && kv.cache.bytes == 1595417600);
    check(packed, 1024, 2, true);
    for (auto& s : packed) s.dn = 460800;
    const auto old_expanded = static_cache_layout(packed, 1024, 2);
    const auto new_expanded = static_cache_layout(packed, 1024, 2, 160, true);
    need(old_expanded.pool_of == new_expanded.pool_of && old_expanded.bytes == new_expanded.bytes && old_expanded.slots == new_expanded.slots);
    const size_t large = (std::numeric_limits<size_t>::max() / 4096) & ~size_t(31);
    const auto edge = static_cache_layout({{large,16},{16,large},{large/2,large/2}}, 1024, 2, 160, true);
    need(edge.bytes <= 2 * large * 1024 && edge.slots <= 2048);
    invalid({}, 1024, 1); invalid(four, 0, 1); invalid(four, 159, 1); invalid(four, 161, 1);
    invalid(four, 4128, 1); invalid(four, 1024, 0); invalid(four, 1024, 3);
    invalid(four, 1024, 1, 0); invalid(four, 1024, 1, 161); invalid(four, 160, 2);
    invalid({{0,16}}, 160, 1); invalid({{16,0}}, 160, 1); invalid({{17,16}}, 160, 1);
    const size_t mx = std::numeric_limits<size_t>::max() & ~size_t(15);
    invalid({{mx,16}}, 160, 1); invalid({{mx / 16 * 16,16}}, 160, 2);
    invalid({{mx/256/16*16,16}}, 4096, 1); // multiplication overflow
    uint32_t seed = 37;
    for (int n = 1; n <= 48; ++n) for (int cap : {352,512,1024,4096}) {
        std::vector<StaticCacheShape> s;
        for (int i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u; const size_t g = 16 * (1 + seed % 128);
            seed = seed * 1664525u + 1013904223u; s.push_back({g,16 * (1 + seed % 128)});
        }
        check(s, cap, 1); check(s, cap, 2);
        check(s, cap, 1, true); check(s, cap, 2, true);
    }
    std::cout << "static cache layout: allocation bounds, pool assignment, 768 generated layouts and invalid inputs passed\n";
}

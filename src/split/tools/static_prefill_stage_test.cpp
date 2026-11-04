#include "static_prefill_stage.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

static void check_at(bool ok, int line) {
    if (!ok) throw std::runtime_error("static prefill staging check failed at line " + std::to_string(line));
}
#define check(ok) check_at((ok), __LINE__)

int main() {
    size_t checked = 0;
    // Independent byte-labelled source banks. Cover every placement of eight
    // experts, unequal matrix sizes, partial stages and noncontiguous sources.
    for (unsigned mask = 0; mask < 256; ++mask)
        for (size_t gs : {size_t(16), size_t(80)})
            for (size_t ds : {size_t(32), size_t(112)})
                for (bool gaps : {false, true}) {
                    const size_t step = gaps ? 2 : 1;
                    std::vector<char> resident(8), g(8 * gs * step), d(8 * ds * step);
                    std::vector<const char*> gp(8), dp(8);
                    size_t cold = 0, hot = 0;
                    for (int e = 0; e < 8; ++e) resident[e] = (mask >> e) & 1;
                    const size_t nc = std::count(resident.begin(), resident.end(), 0);
                    for (int e = 0; e < 8; ++e) {
                        const size_t slot = resident[e] ? nc + hot++ : cold++;
                        char* a = g.data() + slot * gs * step;
                        char* b = d.data() + slot * ds * step;
                        for (size_t i = 0; i < gs; ++i) a[i] = char(e * 17 + i);
                        for (size_t i = 0; i < ds; ++i) b[i] = char(e * 31 + i);
                        gp[e] = a, dp[e] = b;
                    }
                    for (size_t cap : {size_t(0), gs + ds - 1, gs + ds, 3 * (gs + ds) + 7, 8 * (gs + ds)}) {
                        std::vector<char> stage(cap + 64, char(0x5a));
                        char* at = stage.data() + 32;
                        auto plan = fnx::static_prefill_stage(resident, gp, dp, gs, ds, at, cap);
                        for (const auto& copy : plan.copies) {
                            check(copy.dst >= at && copy.dst + copy.n <= at + cap);
                            std::memcpy(copy.dst, copy.src, copy.n);
                        }
                        const size_t moved = std::min(nc, cap / (gs + ds));
                        check(plan.staged == moved * (gs + ds));
                        check(plan.in_place == (nc - moved) * (gs + ds));
                        if (!gaps) check(plan.copies.size() == (moved ? 2 : 0));
                        for (int e = 0; e < 8; ++e) {
                            check(std::memcmp(plan.gu[e], gp[e], gs) == 0);
                            check(std::memcmp(plan.down[e], dp[e], ds) == 0);
                            if (resident[e]) check(plan.gu[e] == gp[e] && plan.down[e] == dp[e]);
                        }
                        for (size_t i = 0; i < 32; ++i) check(stage[i] == char(0x5a));
                        for (size_t i = cap + 32; i < stage.size(); ++i) check(stage[i] == char(0x5a));
                        ++checked;
                    }
                }
    bool rejected = false;
    try { fnx::static_prefill_stage({2}, {"g"}, {"d"}, 1, 1, nullptr, 0); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected);
    rejected = false;
    try { fnx::static_prefill_stage({0}, {"g"}, {"d"}, 1, 1, nullptr, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected);
    rejected = false;
    try { fnx::static_prefill_stage({0}, {}, {"d"}, 1, 1, nullptr, 0); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected);
    std::cout << "static prefill staging: " << checked << " byte/capacity cases passed\n";
}

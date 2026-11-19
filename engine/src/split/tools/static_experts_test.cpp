#include "static_experts.hpp"
#include <cstdint>
#include <iostream>

void check(bool ok) { if (!ok) throw std::runtime_error("static placement check failed"); }

int main() {
    // Reconstruct uniquely labelled expert bytes through the two compact banks.
    // Includes empty banks and every hot/cold assignment of eight experts.
    for (unsigned mask = 0; mask < 256; ++mask) {
        std::vector<char> resident(8);
        for (int e = 0; e < 8; ++e) resident[e] = (mask >> e) & 1;
        const fnx::StaticExpertPlan p(resident);
        check(p.hot + p.cold == resident.size());
        std::vector<uint32_t> hot(p.hot * 19), cold(p.cold * 19);
        for (size_t e = 0; e < 8; ++e) {
            auto& bank = resident[e] ? hot : cold;
            for (size_t i = 0; i < 19; ++i) bank.at(p.slot[e] * 19 + i) = e * 19 + i;
        }
        for (size_t e = 0; e < 8; ++e) {
            const auto& bank = resident[e] ? hot : cold;
            for (size_t i = 0; i < 19; ++i) check(bank.at(p.slot[e] * 19 + i) == e * 19 + i);
        }
    }
    const fnx::StaticExpertPlan empty({});
    check(empty.hot == 0 && empty.cold == 0 && empty.slot.empty());
    bool rejected = false;
    try { fnx::StaticExpertPlan bad({1, 2}); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected);
    std::cout << "static expert placement: 256 partitions passed\n";
}

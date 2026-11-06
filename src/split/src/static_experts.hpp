// Compact, immutable expert placement. Hot experts have no duplicate host copy;
// cold experts stay in mapped host memory. The kernels use the usual DW::ep table.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fnx {
struct StaticExpertPlan {
    std::vector<size_t> slot;
    size_t hot = 0, cold = 0;
    explicit StaticExpertPlan(const std::vector<char>& resident) {
        slot.reserve(resident.size());
        for (char r : resident) {
            if (r != 0 && r != 1) throw std::runtime_error("static experts: residency must be 0 or 1");
            slot.push_back(r ? hot++ : cold++);
        }
    }
};
} // namespace fnx

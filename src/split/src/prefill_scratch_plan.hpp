// Host-side ownership checks for a temporary tail of the static KV/cache arena.
// GPU synchronization and cache invalidation remain the Engine's responsibility.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace fnx {
struct PrefillScratchPlan {
    int begin = 0, end = 0;
    explicit operator bool() const { return begin >= 0 && end > begin; }
    int pages() const { return end - begin; }
};

template <size_t Cards>
inline PrefillScratchPlan plan_prefill_scratch(int base, int cap, int needed_pages, int used_top,
    size_t bytes, const std::array<size_t,Cards>& page_bytes) {
    if (!bytes || base < 0 || cap <= base || needed_pages < 0 || used_top < 0 || !Cards) return {};
    size_t count = 0;
    for (size_t page : page_bytes) {
        if (!page || page % 256) return {};
        const size_t n = bytes / page + (bytes % page != 0);
        if (n > (size_t)(cap - base)) return {};
        count = std::max(count, n);
    }
    const int first = cap - (int) count;
    if (first < std::max({base, needed_pages, used_top})) return {};
    return {first, cap};
}

// All IDs must be free before any are removed. The Engine keeps this list in
// descending order so its ordinary allocator takes low page IDs from back().
inline bool reserve_prefill_pages(const PrefillScratchPlan& plan, std::vector<int>& free) {
    if (!plan || !std::is_sorted(free.begin(), free.end(), std::greater<int>()) ||
        std::adjacent_find(free.begin(), free.end()) != free.end()) return false;
    for (int pg=plan.begin;pg<plan.end;++pg)
        if (!std::binary_search(free.begin(),free.end(),pg,std::greater<int>())) return false;
    free.erase(std::remove_if(free.begin(),free.end(),[&](int pg) {
        return pg>=plan.begin && pg<plan.end;
    }),free.end());
    return true;
}

// Explicit return only, after every GPU is idle. A bad/double return must not
// partially change the list or make one physical page available twice.
inline void return_prefill_pages(const PrefillScratchPlan& plan, std::vector<int>& free) {
    if (!plan) throw std::logic_error("prefill scratch: invalid return");
    for (int pg=plan.begin;pg<plan.end;++pg)
        if (std::find(free.begin(),free.end(),pg)!=free.end())
            throw std::logic_error("prefill scratch: page already free");
    std::vector<int> next=free;
    next.reserve(next.size()+(size_t)plan.pages());
    for (int pg=plan.begin;pg<plan.end;++pg) next.push_back(pg);
    std::sort(next.begin(),next.end(),std::greater<int>());
    free.swap(next);
}
} // namespace fnx

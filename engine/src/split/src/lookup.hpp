// Context-copy proposals. The first token must be the already sampled d1;
// no image cell, or token after one, is offered. Search0 preserves the nearest
// trigram rule; a positive limit keeps the longest usable of those occurrences.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fnx {
// Owned by one generate() call. Positions only refer to its append-only token vector.
class LookupIndex {
    std::vector<int> heads_, previous_;
    size_t bucket(const int* t) const {
        uint32_t h = (uint32_t)t[0] * 0x9e3779b1u + (uint32_t)t[1];
        h = (h ^ (h >> 16)) * 0x85ebca6bu + (uint32_t)t[2];
        return (h ^ (h >> 16)) & (heads_.size() - 1);
    }
public:
    int first(const std::vector<int>& ctx, size_t end) {
        if (heads_.empty() || end > heads_.size() * 2) {
            size_t capacity = 16;
            while (capacity < end) capacity *= 2;
            heads_.assign(capacity, -1);
            previous_.clear();
            previous_.reserve(capacity);
        }
        for (size_t i = previous_.size(); i < end; ++i) {
            int& head = heads_[bucket(ctx.data() + i)];
            previous_.push_back(head);
            head = (int)i;
        }
        return heads_[bucket(ctx.data() + end)];
    }
    int previous(int i) const { return previous_[i]; }
};

inline std::vector<int> lookup_drafts(const std::vector<int>& ctx, int lk, int d1, int search, int image_token, LookupIndex* index = nullptr) {
    constexpr int LOOKUP_N = 3, LOOKUP_MIN = 3;
    std::vector<int> r;
    if (lk == 0 || (int) ctx.size() <= LOOKUP_N) return r;
    const size_t n0 = ctx.size() - LOOKUP_N;
    if (search < 0 || search > 64) throw std::runtime_error("lookup search must be in 0..64");
    const int limit = std::max(1, search);
    int matched = 0;
    for (int pos = index ? index->first(ctx, n0) : (int)n0 - 1; pos >= 0;
         pos = index ? index->previous(pos) : pos - 1) {
        if (!std::equal(ctx.begin() + pos, ctx.begin() + pos + LOOKUP_N, ctx.begin() + n0)) continue;
        if (d1 < 0 || ctx[pos + LOOKUP_N] == d1) {
            int extra = 0;
            while (LOOKUP_N + extra < lk && extra < pos && ctx[pos - 1 - extra] == ctx[n0 - 1 - extra]) ++extra;
            const int k = std::min(lk, std::max(LOOKUP_MIN, LOOKUP_N + extra));
            const auto first = ctx.begin() + pos + LOOKUP_N;
            auto last = first + std::min<std::ptrdiff_t>(k, ctx.end() - first);
            last = std::find_if(first, last, [image_token](int t) { return t >= image_token; });
            if ((size_t)(last - first) > r.size()) r.assign(first, last);
            if ((int)r.size() == lk) break;
        }
        if (++matched >= limit) break;
    }
    return r;
}

}  // namespace fnx

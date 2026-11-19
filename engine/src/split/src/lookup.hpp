// Context-copy proposals. The first token must be the already sampled d1;
// no image cell, or token after one, is offered. Search0 preserves the nearest
// trigram rule; a positive limit keeps the longest usable of those occurrences.
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fnx {
inline std::vector<int> lookup_drafts(const std::vector<int>& ctx, int lk, int d1, int search, int image_token) {
    constexpr int LOOKUP_N = 3, LOOKUP_MIN = 3;
    std::vector<int> r;
    if (lk == 0 || (int) ctx.size() <= LOOKUP_N) return r;
    const size_t n0 = ctx.size() - LOOKUP_N;
    // Opt-in: inspect a bounded number of matching trigrams. Prefer the
    // longest usable context copy; ties keep the most recent occurrence.
    // The first token remains d1, so its actual q and rejection rule are unchanged.
    if (search < 0 || search > 64) throw std::runtime_error("lookup search must be in0..64");
    if (search) {
        int matched = 0;
        for (size_t pos = n0; pos-- > 0;) {
            if (!std::equal(ctx.begin() + pos, ctx.begin() + pos + LOOKUP_N, ctx.begin() + n0)) continue;
            ++matched;
            if (pos + LOOKUP_N < ctx.size() && (d1 < 0 || ctx[pos + LOOKUP_N] == d1)) {
                int m = LOOKUP_N;
                while (m < lk && (size_t)(m - LOOKUP_N) < pos && ctx[pos - 1 - (m - LOOKUP_N)] == ctx[n0 - 1 - (m - LOOKUP_N)]) ++m;
                const int k = std::min(lk, std::max(LOOKUP_MIN, m));
                const auto first = ctx.begin() + pos + LOOKUP_N;
                auto last = ctx.begin() + std::min(ctx.size(), pos + LOOKUP_N + k);
                last = std::find_if(first, last, [image_token](int t) { return t >= image_token; });
                if ((size_t)(last - first) > r.size()) r.assign(first, last);
                if ((int)r.size() == lk) break;
            }
            if (matched >= search) break;
        }
        return r;
    }
    for (size_t s = n0; s-- > 0;)
        if (std::equal(ctx.begin() + s, ctx.begin() + s + LOOKUP_N, ctx.begin() + n0)) {
            if (s + LOOKUP_N < ctx.size() && (d1 < 0 || ctx[s + LOOKUP_N] == d1)) {
                int m = LOOKUP_N;
                while (m < lk && (size_t) (m - LOOKUP_N) < s && ctx[s - 1 - (m - LOOKUP_N)] == ctx[n0 - 1 - (m - LOOKUP_N)]) ++m;
                const int k = std::min(lk, std::max(LOOKUP_MIN, m));
                r.assign(ctx.begin() + s + LOOKUP_N, ctx.begin() + std::min(ctx.size(), s + LOOKUP_N + k));
            }
            break;
        }
    // an image cell is never a draft (nor anything after it)
    r.erase(std::find_if(r.begin(), r.end(), [image_token](int t) { return t >= image_token; }), r.end());
    return r;
}

}  // namespace fnx

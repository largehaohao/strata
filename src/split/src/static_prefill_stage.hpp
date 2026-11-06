// Prefill's temporary tables for immutable hot/cold experts. Copy bytes only;
// decode keeps the original tables and its independent cache throughout.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fnx {
struct PrefillCacheBorrow {
    size_t phase_bytes;    // rounded up to a slot boundary
    size_t addend_offset;
    int slots;
};

// One prefix owns the phase view, residuals and addends. Leave at least
// one slot outside the loan, as for residual-only borrowing.
inline PrefillCacheBorrow prefill_cache_borrow(size_t phase, int chunks, size_t stride, size_t addend,
                                               size_t slot_bytes, int capacity) {
    const size_t limit = std::numeric_limits<size_t>::max();
    if (chunks < 0 || !slot_bytes || capacity < 2 || (chunks && stride > limit / (size_t) chunks))
        throw std::runtime_error("prefill cache borrow: invalid shape");
    const size_t phase_slots = phase / slot_bytes + (phase % slot_bytes != 0);
    if (phase_slots > limit / slot_bytes) throw std::runtime_error("prefill cache borrow: phase overflow");
    const size_t aligned = phase_slots * slot_bytes, residual = (size_t) chunks * stride;
    if (residual > limit - aligned || addend > limit - aligned - residual)
        throw std::runtime_error("prefill cache borrow: size overflow");
    const size_t total = aligned + residual + addend;
    const size_t slots = total / slot_bytes + (total % slot_bytes != 0);
    if (slots >= (size_t) capacity) throw std::runtime_error("prefill cache borrow exceeds independent pool");
    return {aligned, aligned + residual, (int) slots};
}

struct PrefillStageCopy {
    const char* src;
    char* dst;
    size_t n;
};
struct StaticPrefillStage {
    std::vector<const char*> gu, down;
    std::vector<PrefillStageCopy> copies;
    size_t staged = 0, in_place = 0;
};

struct RoutedPrefillTransfers {
    std::vector<PrefillStageCopy> copies;
    std::vector<int> experts;
    size_t bytes = 0;
};

// The strata branch's actual-route uploads, adapted to immutable TP tables.
// Planning does not mark an expert ready: the caller does that only after all
// copies and the consumer's readiness event have been queued successfully.
inline RoutedPrefillTransfers routed_prefill_transfers(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        const std::vector<unsigned char>& ready, size_t gs, size_t ds) {
    if (!src_gu || !src_dn || !dst_gu || !dst_dn || !counts || !gs || !ds)
        throw std::runtime_error("routed prefill DMA: invalid tables or strides");
    RoutedPrefillTransfers out;
    for (size_t e = 0; e < ready.size(); ++e) {
        if (counts[e] < 0) throw std::runtime_error("routed prefill DMA: negative route count");
        if (counts[e] && !ready[e] && (src_gu[e] != dst_gu[e] || src_dn[e] != dst_dn[e]))
            out.experts.push_back((int) e);
    }
    for (int part = 0; part < 2; ++part) {
        const size_t first = out.copies.size(), bytes = part ? ds : gs;
        for (int e : out.experts) {
            const char* src = part ? src_dn[e] : src_gu[e];
            char* dst = const_cast<char*>(part ? dst_dn[e] : dst_gu[e]);
            if (src == dst) continue;
            if (!src || !dst) throw std::runtime_error("routed prefill DMA: missing expert");
            if (out.copies.size() > first && out.copies.back().src + out.copies.back().n == src &&
                out.copies.back().dst + out.copies.back().n == dst)
                out.copies.back().n += bytes;
            else out.copies.push_back({src, dst, bytes});
            out.bytes += bytes;
        }
    }
    return out;
}

inline StaticPrefillStage static_prefill_stage(const std::vector<char>& resident,
        const std::vector<const char*>& gu, const std::vector<const char*>& down,
        size_t gs, size_t ds, char* stage, size_t capacity) {
    if (gu.size() != resident.size() || down.size() != resident.size() || !gs || !ds ||
        gs > std::numeric_limits<size_t>::max() - ds)
        throw std::runtime_error("static prefill staging: invalid expert layout");
    size_t cold = 0;
    for (size_t e = 0; e < resident.size(); ++e) {
        if ((resident[e] != 0 && resident[e] != 1) || !gu[e] || !down[e])
            throw std::runtime_error("static prefill staging: invalid expert pointer or residency");
        cold += !resident[e];
    }
    const size_t fit = std::min(cold, capacity / (gs + ds));
    if (fit && !stage) throw std::runtime_error("static prefill staging: missing stage");
    if (cold > std::numeric_limits<size_t>::max() / (gs + ds))
        throw std::runtime_error("static prefill staging: byte count overflow");
    StaticPrefillStage out{gu, down, {}, fit * (gs + ds), (cold - fit) * (gs + ds)};
    auto add = [&](const char* src, char* dst, size_t n, bool same_bank) {
        if (same_bank && !out.copies.empty() && out.copies.back().src + out.copies.back().n == src &&
            out.copies.back().dst + out.copies.back().n == dst)
            out.copies.back().n += n;
        else out.copies.push_back({src, dst, n});
    };
    // Separate GU and down walks coalesce compact cold banks into two DMA
    // transfers per layer instead of submitting two transfers per expert.
    for (int part = 0; part < 2; ++part) {
        size_t k = 0;
        for (size_t e = 0; e < resident.size() && k < fit; ++e) {
            if (resident[e]) continue;
            const size_t stride = part ? ds : gs;
            char* dst = stage + (part ? fit * gs : 0) + k * stride;
            add(part ? down[e] : gu[e], dst, stride, k != 0);
            (part ? out.down : out.gu)[e] = dst;
            ++k;
        }
    }
    return out;
}
} // namespace fnx

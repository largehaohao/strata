// A decode-only cache beside immutable static expert placement. The model's
// original tables remain available to prefill; only decode uses these tables.
#pragma once

#include "engine_parts.hpp"
#include <algorithm>
#include <cstdio>

namespace fnx {

inline ExpertCache static_decode_cache(const Model& model, const CardW& weights, int slots) {
    ExpertCache ec;
    ec.n_slots = ec.n[0] = slots;
    ec.n_pools = 1;
    size_t gu_max = 0, dn_max = 0;
    for (const LayerW& layer : weights.L) {
        if (!layer.gu_e.ep || !layer.down_e.ep || layer.gu_e_stride % 16 || layer.down_e_stride % 16)
            throw std::runtime_error("static decode cache: expected aligned static expert tables");
        gu_max = std::max(gu_max, layer.gu_e_stride);
        dn_max = std::max(dn_max, layer.down_e_stride);
    }
    ec.dn_off[0] = gu_max;
    ec.slot_bytes[0] = gu_max + dn_max;
    ec.pool[0] = dalloc<char>((size_t) slots * ec.slot_bytes[0]);
    ec.ep_gu = dalloc<const char*>(NL * NEXP);
    ec.ep_dn = dalloc<const char*>(NL * NEXP);
    auto host_gu = dalloc<const char*>(NL * NEXP);
    auto host_dn = dalloc<const char*>(NL * NEXP);
    ec.host_gu = host_gu;
    ec.host_dn = host_dn;
    ec.slot_of = dalloc<int>(NL * NEXP);
    ec.owner = dalloc<int>(slots);
    ec.ref = dalloc<unsigned char>(slots);
    ec.hand = dalloc<int>(EC_POOLS);
    ec.lent = dalloc<int>(EC_POOLS);
    std::vector<int> places(NL * NEXP, -1), owners(slots, -1);
    for (int l = 0; l < NL; ++l) {
        if (model.resident[l].size() != NEXP) throw std::runtime_error("static decode cache: residency shape");
        for (int e = 0; e < NEXP; ++e)
            if (model.resident[l][e]) places[l * NEXP + e] = -2;   // fixed VRAM: never takes a cache slot
        const size_t bytes = NEXP * sizeof(char*);
        ck(cudaMemcpy(ec.ep_gu + l * NEXP, weights.L[l].gu_e.ep, bytes, cudaMemcpyDeviceToDevice), "cache gate/up table");
        ck(cudaMemcpy(ec.ep_dn + l * NEXP, weights.L[l].down_e.ep, bytes, cudaMemcpyDeviceToDevice), "cache down table");
    }
    ck(cudaMemcpy(host_gu, ec.ep_gu, NL * NEXP * sizeof(char*), cudaMemcpyDeviceToDevice), "cache gate/up backing");
    ck(cudaMemcpy(host_dn, ec.ep_dn, NL * NEXP * sizeof(char*), cudaMemcpyDeviceToDevice), "cache down backing");
    ck(cudaMemcpy(ec.slot_of, places.data(), places.size() * sizeof(int), cudaMemcpyHostToDevice), "cache initial places");
    ck(cudaMemcpy(ec.owner, owners.data(), owners.size() * sizeof(int), cudaMemcpyHostToDevice), "cache initial owners");
    std::fprintf(stderr, "fnx: card %d: static decode cache %d slots, %zu MiB, fixed hot experts bypass it\n",
                 weights.dev, slots, ((size_t) slots * ec.slot_bytes[0]) >> 20);
    return ec;
}

} // namespace fnx

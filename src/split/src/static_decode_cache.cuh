// A decode-only cache beside immutable static expert placement. The model's
// original tables remain available to prefill; only decode uses these tables.
#pragma once

#include "engine_parts.hpp"
#include "static_cache_layout.hpp"
#include <algorithm>
#include <cstdio>

namespace fnx {

inline ExpertCache static_decode_cache(const Model& model, const CardW& weights, int slots, int groups = 1) {
    std::vector<StaticCacheShape> shapes;
    for (const LayerW& layer : weights.L) {
        if (!layer.gu_e.ep || !layer.down_e.ep)
            throw std::runtime_error("static decode cache: expected static expert tables");
        shapes.push_back({layer.gu_e_stride, layer.down_e_stride});
    }
    const auto layout = static_cache_layout(shapes, slots, groups, MAXT * TOPK);
    ExpertCache ec;
    ec.n_slots = layout.slots;
    ec.n_pools = (int) layout.pools.size();
    for (int p = 0; p < ec.n_pools; ++p) {
        const auto& pool = layout.pools[p];
        ec.s0[p] = pool.start; ec.n[p] = pool.slots;
        ec.dn_off[p] = pool.gu; ec.slot_bytes[p] = pool.stride();
        ec.pool[p] = dalloc<char>((size_t) pool.slots * pool.stride());
    }
    for (int l = 0; l < NL; ++l) ec.pool_of[l] = layout.pool_of[l];
    ec.ep_gu = dalloc<const char*>(NL * NEXP);
    ec.ep_dn = dalloc<const char*>(NL * NEXP);
    auto host_gu = dalloc<const char*>(NL * NEXP);
    auto host_dn = dalloc<const char*>(NL * NEXP);
    ec.host_gu = host_gu;
    ec.host_dn = host_dn;
    ec.slot_of = dalloc<int>(NL * NEXP);
    ec.owner = dalloc<int>(ec.n_slots);
    ec.ref = dalloc<unsigned char>(ec.n_slots);
    ec.hand = dalloc<int>(EC_POOLS);
    ec.lent = dalloc<int>(EC_POOLS);
    std::vector<int> places(NL * NEXP, -1), owners(ec.n_slots, -1);
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
    ck(cudaMemcpy(ec.hand, ec.s0, EC_POOLS * sizeof(int), cudaMemcpyHostToDevice), "cache initial hands");
    std::fprintf(stderr, "fnx: card %d: static decode cache %d slots, %zu MiB in %d pools, fixed hot experts bypass it\n",
                 weights.dev, ec.n_slots, layout.bytes >> 20, ec.n_pools);
    if (groups == 2)
        for (int p = 0; p < ec.n_pools; ++p)
            std::fprintf(stderr, "fnx: cache pool %d: %d layers, %d slots, gate/up %zu + down %zu bytes\n",
                         p, layout.pools[p].layers, ec.n[p], ec.dn_off[p], layout.pools[p].dn);
    return ec;
}

} // namespace fnx

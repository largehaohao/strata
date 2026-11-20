// Fill only the immutable cold experts used by this chunk. Later chunks in
// the same layer reuse them; the caller clears ready at the next layer.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace fnx {
// With cache-backed sources an in-place consumer can differ from the source.
// The original model pointer is never a writable staging destination.
__device__ __forceinline__ bool stage_copy_needed(const char* src, const char* dst,
                                                   const char* const* model, int expert) {
    return src != dst && (!model || dst != model[expert]);
}

// Resolve tables once after this pass has borrowed its slots and reserved K/V pages.
__global__ void k_static_prefill_tables(const int* slots, const char* const* cache_gu, const char* const* cache_dn,
        const char* const* stage_gu, const char* const* stage_dn, const char** gu, const char** dn, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    gu[i] = slots[i] >= 0 ? cache_gu[i] : stage_gu[i];
    dn[i] = slots[i] >= 0 ? cache_dn[i] : stage_dn[i];
}

__global__ void k_static_prefill_copy(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        const unsigned char* ready, size_t gs, size_t ds, const char* const* model_gu, const char* const* model_dn) {
    const int e = blockIdx.y;
    if (!counts[e] || ready[e]) return;
    const size_t g = gs / sizeof(uint4), d = ds / sizeof(uint4);
    const bool copy_g = stage_copy_needed(src_gu[e], dst_gu[e], model_gu, e),
               copy_d = stage_copy_needed(src_dn[e], dst_dn[e], model_dn, e);
    if (!copy_g && !copy_d) return;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < g + d;
         i += (size_t) gridDim.x * blockDim.x) {
        if (i < g) {
            if (copy_g) reinterpret_cast<uint4*>(const_cast<char*>(dst_gu[e]))[i] =
                reinterpret_cast<const uint4*>(src_gu[e])[i];
        } else if (copy_d) {
            reinterpret_cast<uint4*>(const_cast<char*>(dst_dn[e]))[i - g] =
                reinterpret_cast<const uint4*>(src_dn[e])[i - g];
        }
    }
}

// A separate launch is the device-wide completion boundary. Marking an expert
// from one copy block could make a later block skip bytes it has not copied yet.
__global__ void k_static_prefill_ready(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        unsigned char* ready, size_t gs, size_t ds, int experts, unsigned long long* moved,
        const int* direct_slots, const char* const* model_gu, const char* const* model_dn) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts || !counts[e] || ready[e]) return;
    const size_t n = (stage_copy_needed(src_gu[e], dst_gu[e], model_gu, e) ? gs : 0) +
                     (stage_copy_needed(src_dn[e], dst_dn[e], model_dn, e) ? ds : 0);
    if (direct_slots && direct_slots[e] >= 0) {
        ready[e] = 1;
        if (moved) atomicAdd(moved + 1, (unsigned long long)(gs + ds));
        return;
    }
    if (!n) return;
    ready[e] = 1;
    if (moved) atomicAdd(moved, (unsigned long long) n);
}

inline void static_prefill_copy(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        unsigned char* ready, size_t gs, size_t ds, int experts, unsigned long long* moved, cudaStream_t stream,
        int blocks_per_expert = 8, const int* direct_slots = nullptr,
        const char* const* model_gu = nullptr, const char* const* model_dn = nullptr) {
    if (!gs || !ds || gs % sizeof(uint4) || ds % sizeof(uint4) || experts <= 0 ||
        blocks_per_expert <= 0 || blocks_per_expert > 32)
        throw std::runtime_error("selective prefill staging requires aligned expert strides");
    k_static_prefill_copy<<<dim3(blocks_per_expert, experts), 256, 0, stream>>>(src_gu, src_dn, dst_gu, dst_dn, counts, ready, gs, ds, model_gu, model_dn);
    k_static_prefill_ready<<<(experts + 255) / 256, 256, 0, stream>>>(src_gu, src_dn, dst_gu, dst_dn, counts,
                                                                 ready, gs, ds, experts, moved, direct_slots, model_gu, model_dn);
}
} // namespace fnx

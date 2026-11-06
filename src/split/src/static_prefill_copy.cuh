// Fill only the immutable cold experts used by this chunk. Later chunks in
// the same layer reuse them; the caller clears ready at the next layer.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace fnx {
__global__ void k_static_prefill_copy(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        const unsigned char* ready, size_t gs, size_t ds) {
    const int e = blockIdx.y;
    if (!counts[e] || ready[e]) return;
    const size_t g = gs / sizeof(uint4), d = ds / sizeof(uint4);
    const bool copy_g = src_gu[e] != dst_gu[e], copy_d = src_dn[e] != dst_dn[e];
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
        unsigned char* ready, size_t gs, size_t ds, int experts, unsigned long long* moved) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts || !counts[e] || ready[e]) return;
    const size_t n = (src_gu[e] != dst_gu[e] ? gs : 0) + (src_dn[e] != dst_dn[e] ? ds : 0);
    if (!n) return;
    ready[e] = 1;
    if (moved) atomicAdd(moved, (unsigned long long) n);
}

inline void static_prefill_copy(const char* const* src_gu, const char* const* src_dn,
        const char* const* dst_gu, const char* const* dst_dn, const int* counts,
        unsigned char* ready, size_t gs, size_t ds, int experts, unsigned long long* moved, cudaStream_t stream,
        int blocks_per_expert = 8) {
    if (!gs || !ds || gs % sizeof(uint4) || ds % sizeof(uint4) || experts <= 0 ||
        blocks_per_expert <= 0 || blocks_per_expert > 32)
        throw std::runtime_error("selective prefill staging requires aligned expert strides");
    k_static_prefill_copy<<<dim3(blocks_per_expert, experts), 256, 0, stream>>>(src_gu, src_dn, dst_gu, dst_dn, counts, ready, gs, ds);
    k_static_prefill_ready<<<(experts + 255) / 256, 256, 0, stream>>>(src_gu, src_dn, dst_gu, dst_dn, counts,
                                                                 ready, gs, ds, experts, moved);
}
} // namespace fnx

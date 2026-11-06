// Decode-only staging of the routed matrices. Coalesced reads bring mapped-host
// experts into VRAM once, before the quantized dot kernels reuse their weights.
// A single scratch bank is reused for gate/up and down, in stream order.
#pragma once

namespace fnx {
namespace {
constexpr size_t MOE_STAGE_CHUNK = 16 * 1024;
__global__ void k_moe_stage(DW w, size_t stride, const Job* jobs, const int* njobs,
                            char* scratch, const char** ep, const unsigned char* hot) {
    const int j = blockIdx.y;
    if (j >= *njobs) return;
    const int e = jobs[j].w;
    const char* src = w.ep ? w.ep[e] : (const char*) w.p + (size_t) e * stride;
    if (hot && hot[e]) {
        if (blockIdx.x == 0 && threadIdx.x == 0) ep[e] = src;
        return;
    }
    char* dst = scratch + (size_t) j * stride;
    if (blockIdx.x == 0 && threadIdx.x == 0) ep[e] = dst;
    const size_t start = (size_t) blockIdx.x * MOE_STAGE_CHUNK;
    const size_t end = min(start + MOE_STAGE_CHUNK, stride);
    for (size_t off = start + (size_t) threadIdx.x * sizeof(uint4); off < end;
         off += (size_t) blockDim.x * sizeof(uint4))
        *reinterpret_cast<uint4*>(dst + off) = *reinterpret_cast<const uint4*>(src + off);
}

DW stage_moe(const DW& w, size_t stride, const Job* jobs, const int* njobs, int max_jobs,
             char* scratch, const char** ep, cudaStream_t stream, const unsigned char* hot = nullptr) {
    if (!scratch) return w;
    k_moe_stage<<<dim3((stride + MOE_STAGE_CHUNK - 1) / MOE_STAGE_CHUNK, max_jobs), 128, 0, stream>>>(
        w, stride, jobs, njobs, scratch, ep, hot);
    ck(cudaGetLastError(), "moe staging");
    DW staged = w;
    staged.p = nullptr;
    staged.ep = ep;
    return staged;
}
} // namespace
} // namespace fnx

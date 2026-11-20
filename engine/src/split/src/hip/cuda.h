// HIP stand-in for the CUDA driver API calls src/split makes: stream memory operations (the prompt path's copy-engine ring and
// attention flags). The sources include <cuda.h>; the HIP build puts this directory first on the include path.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../ring_wait.cuh"

typedef hipError_t CUresult;
typedef hipStream_t CUstream;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS hipSuccess
#define CU_STREAM_WAIT_VALUE_GEQ hipStreamWaitValueGte

inline CUresult cuGetErrorString(CUresult e, const char** s) {
    *s = hipGetErrorString(e);
    return hipSuccess;
}
// a 32-bit write / wait on the stream (host memory the card maps). Two implementations, chosen once by FNX_HIP_MEMOP:
//   kernel (default): a one-thread kernel writes the flag (after a system fence) or spins on it, in stream order: 20 us for a cross-card round trip on
//                     two R9700 (phaseA-tests/sfnx/RESULTS-HIP.md), repeatable in the engine.
//   native (FNX_HIP_MEMOP=native): hipStreamWriteValue32 / hipStreamWaitValue32, the runtime's own stream memory operations: 13 us, but not reliable
//                     in the engine's multi-stream use (answers varied from start to start).
// A timed-out software wait marks the pass failed before the stream continues.
#include <cstdlib>
#include <cstring>
static __global__ void fnx_memop_write(volatile unsigned* p, unsigned v) {
    __threadfence_system();   // everything this stream wrote before (the copy engine's / kernels' stores into host memory) first
    *p = v;
    __threadfence_system();
}
static __global__ void fnx_memop_wait(volatile unsigned* p, unsigned v, unsigned* err, unsigned long long limit) {
    const auto t0 = fnx::ring_clock();
    while ((int) (*p - v) < 0) {   // wrapping >=
        if (fnx::ring_expired(t0, limit)) { atomicOr(err, 1u); break; }
    }
    __threadfence_system();
}
inline bool fnx_native_memops() {
    // default: kernels (see the comment above). With the runtime's native stream memory ops the prompt path's text varied from start to start on 4 cards
    // (3 distinct outputs of 5 at 26K tokens); with kernel flags it did not (5 of 5 identical, same speed)
    static const bool native = [] { const char* e = std::getenv("FNX_HIP_MEMOP"); return e && !std::strcmp(e, "native"); }();
    return native;
}
inline CUresult cuStreamWriteValue32(CUstream s, CUdeviceptr p, uint32_t v, unsigned flags) {
    if (fnx_native_memops()) return hipStreamWriteValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), flags);
    fnx_memop_write<<<1, 1, 0, s>>>(reinterpret_cast<volatile unsigned*>(p), v);
    return hipGetLastError();
}
inline CUresult fnx_stream_wait(CUstream s, CUdeviceptr p, uint32_t v, unsigned* err, unsigned long long limit) {
    if (fnx_native_memops()) return hipStreamWaitValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), CU_STREAM_WAIT_VALUE_GEQ, 0xffffffffu);
    fnx_memop_wait<<<1, 1, 0, s>>>(reinterpret_cast<volatile unsigned*>(p), v, err, limit);
    return hipGetLastError();
}
inline CUresult cuInit(unsigned) { return hipInit(0); }

#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

namespace fnx {
inline unsigned long long ring_wait_ticks(int dev) {
#if FNX_HIP
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0)
        throw std::runtime_error("ring wait: unavailable GPU wall clock");
    return static_cast<unsigned long long>(khz) * 10000;  // ten seconds
#else
    return 4000000000ULL;
#endif
}
__device__ __forceinline__ unsigned long long ring_clock() {
#if FNX_HIP
    return wall_clock64();
#else
    return clock64();
#endif
}
__device__ __forceinline__ bool ring_expired(unsigned long long start, unsigned long long limit) {
#if FNX_HIP
    __builtin_amdgcn_s_sleep(1);
#endif
    return ring_clock() - start > limit;
}
}

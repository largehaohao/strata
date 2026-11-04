// Four weights in the same order as an FP32 float4 load. BF16 is storage only;
// expansion restores the exact original FP32 bits before arithmetic.
#pragma once
template<bool BF16>
__device__ __forceinline__ float4 load_weight4(const void* weights, size_t index4) {
    if constexpr (BF16) {
        const uint2 bits = __ldg(reinterpret_cast<const uint2*>(weights) + index4);
        return make_float4(__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
                           __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u));
    } else return __ldg(reinterpret_cast<const float4*>(weights) + index4);
}

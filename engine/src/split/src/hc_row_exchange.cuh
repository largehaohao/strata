// HC-down row partition. The original GEMM is unchanged.
// Only already-rounded lo16 values cross the link; no partial sums are added.
constexpr int HCS_WORDS_PER_BLOCK = 1024;
struct HcsMail {
    unsigned *data, *ready, *generation, *errors;
    int capacity_words, capacity_blocks;
    unsigned long long timeout_ticks;
};
__device__ __forceinline__ __half hcs_lo_value(float down) {
    float x = down / (float) HC;
    return __float2half(x / (1.0f + __expf(-x)));
}
__global__ void hcs_tick(unsigned* generation) {
    if (!threadIdx.x) {
        unsigned n = *generation + 1u;
        *generation = n > 0xfffffffeu || n == 0u ? 1u : n;
    }
}
__global__ void hcs_lo(const float* down, int n, __half* lo) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
        lo[e] = hcs_lo_value(down[e]);
}
template<bool FUSED>
__global__ void hcs_push(const float* down, __half* lo, int words, HcsMail own, HcsMail peer) {
    const unsigned generation = *own.generation;
    const size_t bank = generation & 1u;
    volatile unsigned* dst = peer.data + bank * peer.capacity_words;
    unsigned* local = reinterpret_cast<unsigned*>(lo);
    const int first = blockIdx.x * HCS_WORDS_PER_BLOCK;
    for (int e = first + threadIdx.x; e < min(first + HCS_WORDS_PER_BLOCK, words); e += blockDim.x) {
        unsigned value;
        if constexpr (FUSED) {
            const __half a = hcs_lo_value(down[2 * e]), b = hcs_lo_value(down[2 * e + 1]);
            value = (unsigned)__half_as_ushort(a) | (unsigned)__half_as_ushort(b) << 16;
            local[e] = value;
        } else value = local[e];
        dst[e] = value;
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) {
        reinterpret_cast<volatile unsigned*>(peer.ready)[bank * peer.capacity_blocks + blockIdx.x] = generation;
        __threadfence_system();
    }
}
__global__ void hcs_pull(__half* lo, int words, HcsMail own) {
    const unsigned generation = *own.generation;
    const size_t bank = generation & 1u;
    __shared__ int failed;
    if (!threadIdx.x) {
        failed = 0;
        const auto begin = wall_clock64();
        while (reinterpret_cast<volatile const unsigned*>(own.ready)[bank * own.capacity_blocks + blockIdx.x] != generation) {
            if (*reinterpret_cast<volatile const unsigned*>(own.errors)) { failed = 1; break; }
            if (wall_clock64() - begin > own.timeout_ticks) {
                atomicOr(own.errors, 256u); failed = 1; break;
            }
            __nanosleep(64);
        }
        __threadfence_system();
    }
    __syncthreads();
    if (failed) return;
    const volatile unsigned* src = own.data + bank * own.capacity_words;
    unsigned* dst = reinterpret_cast<unsigned*>(lo);
    const int first = blockIdx.x * HCS_WORDS_PER_BLOCK;
    for (int e = first + threadIdx.x; e < min(first + HCS_WORDS_PER_BLOCK, words); e += blockDim.x)
        dst[e] = src[e];
    // A slot unused by shorter chunks must not retain a ready tag across wrap.
    // The peer cannot reuse this bank until it receives our next generation,
    // whose push is queued after this pull (including this clear) completes.
    __syncthreads();
    if (!threadIdx.x) {
        reinterpret_cast<volatile unsigned*>(own.ready)[bank * own.capacity_blocks + blockIdx.x] = 0;
        __threadfence_system();
    }
}

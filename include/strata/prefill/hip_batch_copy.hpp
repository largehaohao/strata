#pragma once

// Opt-in scatter H2D enqueue for Linux HIP 7+ headers. STRATA_PREFILL_HIP_BATCH_COPY=1 enables the attempt.
// Resolve the newer runtime entry point lazily: neither an older HIP runtime nor flag-off serving acquires a
// link-time dependency on hipMemcpyBatchAsync. Other platforms/headers keep individual cudaMemcpyAsync calls.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>

#if (defined(STRATA_USE_HIP) || defined(__HIPCC__)) && defined(__linux__) && \
    defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
#include <dlfcn.h>
#define STRATA_PREFILL_CAN_HIP_BATCH_COPY 1
#else
#define STRATA_PREFILL_CAN_HIP_BATCH_COPY 0
#endif

namespace strata::prefill::hip_batch_copy {

inline bool requested() {
#if STRATA_PREFILL_CAN_HIP_BATCH_COPY
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PREFILL_HIP_BATCH_COPY");
        return v && v[0] == '1' && v[1] == 0;
    }();
    return on;
#else
    return false;
#endif
}

#if STRATA_PREFILL_CAN_HIP_BATCH_COPY
inline auto entry_point() {
    using Fn = decltype(&hipMemcpyBatchAsync);
    static const Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hipMemcpyBatchAsync"));
    return fn;
}
#endif

// The arrays are consumed during this call; the pointed-to pinned source/destination spans must remain valid
// until the stream reaches the caller's readiness event. Only non-overlapping copy spans belong in one batch.
// Success means all copies were queued, not completed. No synchronization or consumer work is inserted here.
// On a missing/failed batch API, replay ALL copies on the SAME stream. A failure index describes an attempted
// operation, not a reliably completed prefix; replaying a partial batch is safe before the caller queues readers.
// failed_index is count on success/no identified failing copy, otherwise the first failed fallback copy's index.
// used_batch is true only when the batch call itself succeeded. Optional outputs are initialized on every call.
inline cudaError_t enqueue(void** dsts, void** srcs, size_t* sizes, size_t count, cudaStream_t stream,
                           size_t* failed_index = nullptr, bool* used_batch = nullptr) {
    if (failed_index) *failed_index = count;
    if (used_batch) *used_batch = false;
    if (count == 0) return cudaSuccess;
    if (!dsts || !srcs || !sizes) {
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        return hipErrorInvalidValue;
#elif defined(STRATA_GPU_METAL)
        return strata::gpu::kErrorInvalidValue;
#else
        return cudaErrorInvalidValue;
#endif
    }
#if STRATA_PREFILL_CAN_HIP_BATCH_COPY
    if (requested()) {
        if (const auto batch = entry_point()) {
            // Preserve a pre-existing runtime error rather than letting a retry clear an unrelated fault.
            const hipError_t prior = hipPeekAtLastError();
            if (prior != hipSuccess) return prior;
            size_t batch_failed = count;
            const hipError_t result = batch(dsts, srcs, sizes, count, nullptr, nullptr, 0, &batch_failed, stream);
            if (result == hipSuccess) {
                if (used_batch) *used_batch = true;
                return cudaSuccess;
            }
            // The failed enqueue may set HIP's last-error slot. Its error is superseded only if every
            // individual retry queues successfully. Never trust batch_failed as a completed-prefix boundary.
            (void) batch_failed;
            (void) hipGetLastError();
        }
    }
#endif
    for (size_t i = 0; i < count; ++i) {
        const cudaError_t result = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, stream);
        if (result != cudaSuccess) {
            if (failed_index) *failed_index = i;
            return result;
        }
    }
    return cudaSuccess;
}

}  // namespace strata::prefill::hip_batch_copy

#undef STRATA_PREFILL_CAN_HIP_BATCH_COPY

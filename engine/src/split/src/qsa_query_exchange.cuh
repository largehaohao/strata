// Included inside the kernel anonymous namespace on two-card HIP.
// Separate finite push and bounded pull kernels avoid producer starvation.
// Both peers own queries on each call. Alternating mailbox sets prevent a
// fast peer overwriting an unread payload; the nonzero generation cycle
// has even length so the sets alternate across wrap as well.
// Same query partition and generation protocol; lossless position-list packing.
// The selector emits complete four-position blocks, followed by at most one
// partial block. Send the local block ID and reconstruct its original positions.
static_assert(QSA_POOL == 4, "four consecutive positions per selected block");
constexpr int QSX_BLOCKS = (QSA_MAXSEL + 2 * (QSA_POOL - 1)) / QSA_POOL;
constexpr int QSX_WORDS = ((1 + (QSX_BLOCKS + 1) / 2 + 15) / 16) * 16;
struct QsxMail {
    int* data;
    unsigned* ready;
    unsigned* generation;
    unsigned* errors;
    int capacity;
    unsigned long long timeout_ticks;
};
__global__ void k_qsx_push(const int* cells, const int* counts, int T, int stride,
                          int card, int group, QsxMail own, QsxMail peer) {
    const int t = blockIdx.x;
    if (t >= T || (t / group) % 2 != card) return;
    const unsigned generation = *own.generation;
    const int n0 = counts[t], n1 = counts[stride + t];
    const bool dense = n0 == -1 && n1 == -1;
    if (t >= peer.capacity) {
        if (!threadIdx.x) atomicOr(own.errors, 64u);
        return;
    }
    const size_t slot = (size_t)(generation & 1u) * peer.capacity + t;
    volatile unsigned* dst = reinterpret_cast<unsigned*>(peer.data) + slot * QSX_WORDS;
    if (threadIdx.x == 0) dst[0] = dense ? 0xffffffffu : (unsigned)n0 | (unsigned)n1 << 16;
    const bool valid = !dense && n0 >= 0 && n0 <= QSA_MAXSEL && n1 >= 0 && n1 <= QSA_MAXSEL - n0;
    if (!dense && !valid && !threadIdx.x) atomicOr(own.errors, 64u);
    const int b0 = valid ? (n0 + 3) / 4 : 0, b1 = valid ? (n1 + 3) / 4 : 0, blocks = b0 + b1;
    for (int i = threadIdx.x; i < (blocks + 1) / 2; i += blockDim.x) {
        unsigned word = 0;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int block = 2 * i + half;
            if (block >= blocks) continue;
            const size_t from = block < b0 ? (size_t)t * QSA_MAXSEL + 4 * block
                                          : ((size_t)stride + t) * QSA_MAXSEL + 4 * (block - b0);
            const int first = cells[from];
            if (first < 0 || first % 4 || (unsigned)first / 4 > 0xffffu) {
                atomicOr(own.errors, 64u);
                continue;
            }
            word |= (unsigned)(first / 4) << (16 * half);
        }
        dst[1 + i] = word;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // Publish even on failure so a waiting peer receives the original error.
        const unsigned error = *reinterpret_cast<volatile const unsigned*>(own.errors);
        if (error) { dst[0] = 0xfffe0000u | error; __threadfence_system(); }
        reinterpret_cast<volatile unsigned*>(peer.ready)[slot] = generation;
        __threadfence_system();
    }
}
__global__ void k_qsx_pull(int* cells, int* counts, int T, int stride,
                          int card, int group, QsxMail own) {
    const int t = blockIdx.x;
    if (t >= T || (t / group) % 2 == card) return;
    const unsigned generation = *own.generation;
    const size_t slot = (size_t)(generation & 1u) * own.capacity + t;
    __shared__ int n0, n1, failed;
    if (threadIdx.x == 0) {
        failed = 0;
        const auto begin = wall_clock64();
        while (reinterpret_cast<volatile const unsigned*>(own.ready)[slot] != generation) {
            if (*reinterpret_cast<volatile const unsigned*>(own.errors)) { failed = 1; break; }
            if (wall_clock64() - begin > own.timeout_ticks) {
                atomicOr(own.errors, 128u); failed = 1; break;
            }
            __nanosleep(64);
        }
        if (!failed) {
            const volatile unsigned* src = reinterpret_cast<const unsigned*>(own.data) + slot * QSX_WORDS;
            const unsigned sizes = src[0];
            n0 = sizes == 0xffffffffu ? -1 : (int)(sizes & 0xffffu);
            n1 = sizes == 0xffffffffu ? -1 : (int)(sizes >> 16);
            if (!((n0 == -1 && n1 == -1) || (n0 >= 0 && n1 >= 0 && n0 + n1 <= QSA_MAXSEL))) {
                atomicOr(own.errors, sizes >> 16 == 0xfffeu ? sizes & 0xffffu : 64u); failed = 1;
            }
        }
    }
    __syncthreads();
    if (failed) {
        if (!threadIdx.x) counts[t] = counts[stride + t] = 0;
        return;
    }
    const volatile unsigned* src = reinterpret_cast<const unsigned*>(own.data) + slot * QSX_WORDS;
    if (threadIdx.x == 0) counts[t] = n0;
    if (threadIdx.x == 1) counts[stride + t] = n1;
    const int n = n0 < 0 ? 0 : n0 + n1, b0 = n0 < 0 ? 0 : (n0 + 3) / 4;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const bool first = i < n0;
        const int local = first ? i : i - n0;
        const int block = (first ? 0 : b0) + local / 4;
        const unsigned packed = src[1 + block / 2];
        const int id = (packed >> (16 * (block & 1))) & 0xffffu;
        const size_t to = first ? (size_t)t * QSA_MAXSEL + local
                                : ((size_t)stride + t) * QSA_MAXSEL + local;
        cells[to] = 4 * id + local % 4;
    }
    // A shorter batch may leave this row untouched until generation wraps.
    __syncthreads();
    if (threadIdx.x == 0) {
        reinterpret_cast<volatile unsigned*>(own.ready)[slot] = 0;
        __threadfence_system();
    }
}

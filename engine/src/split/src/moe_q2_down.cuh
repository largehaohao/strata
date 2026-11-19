// A packed Q2_0 half-block has exactly the Q4_0 values code + 7. Keep
// Q4_0's two partial dots and its Q8_1 sum correction, including rounding.
__device__ __forceinline__ unsigned q2_spread_byte(unsigned x) {
    x = (x | (x << 12)) & 0x000f000fu;
    return (x | (x << 6)) & 0x03030303u;
}

struct Q2Half { int v[QI4_0]; float d; };

__device__ __forceinline__ Q2Half q2_load_half(const char* row, int half_block) {
    const auto* b = reinterpret_cast<const unsigned char*>(row) + (half_block / 2) * 18;
    const auto* q = b + 2 + (half_block % 2) * 8;
    Q2Half decoded;
#pragma unroll
    for (int i = 0; i < QI4_0; ++i) {
        const unsigned lo = q2_spread_byte(q[i]) + 0x07070707u;
        const unsigned hi = q2_spread_byte(q[i + 4]) + 0x07070707u;
        decoded.v[i] = (int) (lo | (hi << 4));
    }
    decoded.d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(b)));
    return decoded;
}

__device__ __forceinline__ float q2_half_dot(const Q2Half& decoded, const block_q8_1* y) {
    float sum = 0.f;
#pragma unroll
    for (int part = 0; part < QI4_0 / VDR_Q4_0; ++part) {
        int u[2 * VDR_Q4_0];
        constexpr int width = VDR_Q4_0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            u[2 * i] = get_int_b4(y->qs, part * width + i);
            u[2 * i + 1] = get_int_b4(y->qs, part * width + i + QI4_0);
        }
        sum += vec_dot_q4_0_q8_1_impl<width>(decoded.v + part * width, u, decoded.d, y->ds);
    }
    return sum;
}

__device__ __forceinline__ float unit_dot_q2_exact(const char* row, int u, const block_q8_1* y) {
    return q2_half_dot(q2_load_half(row, u), y + u);
}

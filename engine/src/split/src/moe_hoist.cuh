// Hoisted weight-decode helpers and kernels for decode MoE gate|up (IQ4_XS, Q5_K)
// and non-Q2 down projections (Q4_0, Q5_0, Q5_1, IQ4_NL, Q8_0).
// Every expression preserves the exact per-part rounding and reduction order of qdot.cuh.
#pragma once

inline bool moe_hoist_enabled() {
    static const bool on = [] {
        const long long v = env_int("FNX_MOE_HOIST", 0);
        if (v != 0 && v != 1) throw std::runtime_error("FNX_MOE_HOIST: expected 0 or 1");
        return v == 1;
    }();
    return on;
}

struct Iq4XsDot32 {
    int2 v[4];
    int ls;
    float d;
};

__device__ __forceinline__ Iq4XsDot32 iq4xs_load32(const char* row, int kbx, int iqs) {
    const block_iq4_xs* bq4 = reinterpret_cast<const block_iq4_xs*>(row) + kbx;
    Iq4XsDot32 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
    }
    r.ls = (((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4)) - 32;
    r.d = __half2float(bq4->d);
    return r;
}

__device__ __forceinline__ float iq4xs_dot32(const Iq4XsDot32& w, const block_q8_1& x) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int u0 = get_int_b4(x.qs, j + 0);
        const int u1 = get_int_b4(x.qs, j + 4);
        sumi = qd_dp4a(w.v[j].x, u0, sumi);
        sumi = qd_dp4a(w.v[j].y, u1, sumi);
    }
    sumi *= w.ls;
    const float d = w.d * __low2float(x.ds);
    return d * sumi;
}

struct Q5KDot16 {
    int v0[2];
    int v1[2];
    half2 dm;
    uint8_t sc[2];
    uint8_t m[2];
};

__device__ __forceinline__ Q5KDot16 q5k_load16(const char* row, int kbx, int iqs) {
    const block_q5_K* bq5_K = reinterpret_cast<const block_q5_K*>(row) + kbx;
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = reinterpret_cast<const int*>(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5_K->qh + 4 * ((iqs / 2) % 4));
    const int vl0 = ql[0];
    const int vl1 = ql[4];
    const int vh0 = qh[0] >> bq8_offset;
    const int vh1 = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5_K->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    const uint16_t a0 = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    const uint16_t a1 = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    Q5KDot16 r;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        r.v0[i] = ((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
        r.v1[i] = ((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
    }
    r.dm = bq5_K->dm;
    r.sc[0] = (uint8_t) (a0 & 0xFF);
    r.sc[1] = (uint8_t) (a0 >> 8);
    r.m[0] = (uint8_t) (a1 & 0xFF);
    r.m[1] = (uint8_t) (a1 >> 8);
    return r;
}

__device__ __forceinline__ float q5k_dot16(const Q5KDot16& w, const block_q8_1* bq8_1, int iqs) {
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        const float d8 = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        const int u0 = q8[0];
        const int u1 = q8[4];
        const int dot1 = qd_dp4a(w.v0[i], u0, qd_dp4a(w.v1[i], u1, 0));
        const int dot2 = qd_dp4a(0x01010101, u0, qd_dp4a(0x01010101, u1, 0));
        sumf_d += d8 * (dot1 * w.sc[i]);
        sumf_m += d8 * (dot2 * w.m[i]);
    }
    const float2 dm5f = __half22float2(w.dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

template <GgmlType TA, int G, int TT>
__device__ __forceinline__ void moe_gu_hoist_rows(const char* wr, bool live, int ncols, int n, const block_q8_1* y,
                                                  const int* ys, int yb_stride, float* tmp) {
    static_assert(TA == T_IQ4_XS || TA == T_Q5_K, "moe_gu_hoist_rows supports T_IQ4_XS and T_Q5_K");
    static_assert(G == 16 || G == 32, "hoist lanes per row");
    constexpr int qk = qtraits<TA>::qk, vdr = vdr_of<TA>();
    constexpr int per_block = qtraits<TA>::qi / vdr;
    static_assert(G >= per_block && G % per_block == 0, "lanes per weight block");
    constexpr int PARTS = 32 / G;
    const int gl = threadIdx.x % G, kqs = vdr * (gl % per_block), bpr = ncols / qk;
    float acc[PARTS][TT] = {};
    if (live) {
#pragma unroll
        for (int p = 0; p < PARTS; ++p) {
#pragma unroll 8
            for (int kbx = (gl + p * G) / per_block; kbx < bpr; kbx += 32 / per_block) {
                if constexpr (TA == T_IQ4_XS) {
                    const Iq4XsDot32 decoded = iq4xs_load32(wr, kbx, kqs);
#pragma unroll
                    for (int j = 0; j < TT; ++j)
                        if (j < n) acc[p][j] += iq4xs_dot32(decoded, y[(size_t) ys[j] * yb_stride + kbx * (qk / QK8_1) + kqs / vdr]);
                } else {
                    const Q5KDot16 decoded = q5k_load16(wr, kbx, kqs);
#pragma unroll
                    for (int j = 0; j < TT; ++j)
                        if (j < n) acc[p][j] += q5k_dot16(decoded, y + (size_t) ys[j] * yb_stride + kbx * (qk / QK8_1), kqs);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        if constexpr (PARTS == 2) tmp[j] += acc[0][j] + acc[1][j];
        else tmp[j] += acc[0][j];
    }
}

template <GgmlType TA, int G, int TT>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_hoist(
    MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
    const block_q8_1* __restrict__ y, float* __restrict__ out, const int* __restrict__ job_idx = nullptr) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    const int ji_real = job_idx ? job_idx[ji] : ji;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji_real];
    __syncthreads();
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const char* wr = moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
    float tmp[TT] = {};
    moe_gu_hoist_rows<TA, G, TT>(wr, live, w.ncols, jb.n, y, jb.y, w.ncols / QK8_1, tmp);
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

template <GgmlType TA>
struct DownUnit {
    static constexpr bool supported = false;
};

template <>
struct DownUnit<T_Q4_0> {
    static constexpr bool supported = true;
    int vi0[QI4_0];
    int vi1[QI4_0];
    float d;
    __device__ __forceinline__ static DownUnit load(const char* wrow, int u) {
        const block_q4_0* b = reinterpret_cast<const block_q4_0*>(wrow) + u;
        DownUnit r;
#pragma unroll
        for (int idx = 0; idx < QI4_0; ++idx) {
            const int v = get_int_b2(b->qs, idx);
            r.vi0[idx] = (v >> 0) & 0x0F0F0F0F;
            r.vi1[idx] = (v >> 4) & 0x0F0F0F0F;
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const DownUnit& w, const block_q8_1* y) {
        const float2 ds8f = __half22float2(y->ds);
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < QI4_0 / VDR_Q4_0; ++p) {
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_Q4_0; ++i) {
                const int idx = p * VDR_Q4_0 + i;
                const int u0 = get_int_b4(y->qs, idx);
                const int u1 = get_int_b4(y->qs, idx + QI4_0);
                sumi = qd_dp4a(w.vi0[idx], u0, sumi);
                sumi = qd_dp4a(w.vi1[idx], u1, sumi);
            }
            sum += w.d * (sumi * ds8f.x - (8 * VDR_Q4_0 / QI4_0) * ds8f.y);
        }
        return sum;
    }
};

template <>
struct DownUnit<T_Q5_0> {
    static constexpr bool supported = true;
    int vi0[QI5_0];
    int vi1[QI5_0];
    float d;
    __device__ __forceinline__ static DownUnit load(const char* wrow, int u) {
        const block_q5_0* b = reinterpret_cast<const block_q5_0*>(wrow) + u;
        const int qh = get_int_b2(b->qh, 0);
        DownUnit r;
#pragma unroll
        for (int idx = 0; idx < QI5_0; ++idx) {
            const int vl = get_int_b2(b->qs, idx);
            const int vh = qh >> (4 * idx);
            int v0 = (vl >> 0) & 0x0F0F0F0F;
            v0 |= (vh << 4) & 0x00000010;
            v0 |= (vh << 11) & 0x00001000;
            v0 |= (vh << 18) & 0x00100000;
            v0 |= (vh << 25) & 0x10000000;
            r.vi0[idx] = v0;

            int v1 = (vl >> 4) & 0x0F0F0F0F;
            v1 |= (vh >> 12) & 0x00000010;
            v1 |= (vh >> 5) & 0x00001000;
            v1 |= (vh << 2) & 0x00100000;
            v1 |= (vh << 9) & 0x10000000;
            r.vi1[idx] = v1;
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const DownUnit& w, const block_q8_1* y) {
        const float2 ds8f = __half22float2(y->ds);
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < QI5_0 / VDR_Q5_0; ++p) {
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_Q5_0; ++i) {
                const int idx = p * VDR_Q5_0 + i;
                const int u0 = get_int_b4(y->qs, idx);
                const int u1 = get_int_b4(y->qs, idx + QI5_0);
                sumi = qd_dp4a(w.vi0[idx], u0, sumi);
                sumi = qd_dp4a(w.vi1[idx], u1, sumi);
            }
            sum += w.d * (sumi * ds8f.x - (16 * VDR_Q5_0 / QI5_0) * ds8f.y);
        }
        return sum;
    }
};

template <>
struct DownUnit<T_Q5_1> {
    static constexpr bool supported = true;
    int vi0[QI5_1];
    int vi1[QI5_1];
    half2 dm;
    __device__ __forceinline__ static DownUnit load(const char* wrow, int u) {
        const block_q5_1* b = reinterpret_cast<const block_q5_1*>(wrow) + u;
        const int qh = get_int_b4(b->qh, 0);
        DownUnit r;
#pragma unroll
        for (int idx = 0; idx < QI5_1; ++idx) {
            const int vl = get_int_b4(b->qs, idx);
            const int vh = qh >> (4 * idx);
            int v0 = (vl >> 0) & 0x0F0F0F0F;
            v0 |= (vh << 4) & 0x00000010;
            v0 |= (vh << 11) & 0x00001000;
            v0 |= (vh << 18) & 0x00100000;
            v0 |= (vh << 25) & 0x10000000;
            r.vi0[idx] = v0;

            int v1 = (vl >> 4) & 0x0F0F0F0F;
            v1 |= (vh >> 12) & 0x00000010;
            v1 |= (vh >> 5) & 0x00001000;
            v1 |= (vh << 2) & 0x00100000;
            v1 |= (vh << 9) & 0x10000000;
            r.vi1[idx] = v1;
        }
        r.dm = b->dm;
        return r;
    }
    __device__ __forceinline__ static float dot(const DownUnit& w, const block_q8_1* y) {
#ifdef FNX_QD_FAST_FP16
        const float2 tmp = __half22float2(__hmul2(w.dm, y->ds));
        const float d5d8 = tmp.x;
        const float m5s8 = tmp.y;
#else
        const float2 dm5f = __half22float2(w.dm);
        const float2 ds8f = __half22float2(y->ds);
        const float d5d8 = dm5f.x * ds8f.x;
        const float m5s8 = dm5f.y * ds8f.y;
#endif
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < QI5_1 / VDR_Q5_1; ++p) {
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_Q5_1; ++i) {
                const int idx = p * VDR_Q5_1 + i;
                const int u0 = get_int_b4(y->qs, idx);
                const int u1 = get_int_b4(y->qs, idx + QI5_1);
                sumi = qd_dp4a(w.vi0[idx], u0, sumi);
                sumi = qd_dp4a(w.vi1[idx], u1, sumi);
            }
            sum += sumi * d5d8 + m5s8 / (QI5_1 / VDR_Q5_1);
        }
        return sum;
    }
};

template <>
struct DownUnit<T_IQ4_NL> {
    static constexpr bool supported = true;
    int2 v[QI4_NL];
    float d;
    __device__ __forceinline__ static DownUnit load(const char* wrow, int u) {
        const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(wrow) + u;
        DownUnit r;
#pragma unroll
        for (int idx = 0; idx < QI4_NL; ++idx) {
            r.v[idx] = get_int_from_table_16(get_int_b2(b->qs, idx), kvalues_iq4nl);
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const DownUnit& w, const block_q8_1* y) {
        const float d = w.d * __low2float(y->ds);
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < QI4_NL / VDR_IQ4_NL; ++p) {
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_IQ4_NL; ++i) {
                const int idx = p * VDR_IQ4_NL + i;
                const int u0 = get_int_b4(y->qs, idx);
                const int u1 = get_int_b4(y->qs, idx + 4);
                sumi = qd_dp4a(w.v[idx].x, u0, sumi);
                sumi = qd_dp4a(w.v[idx].y, u1, sumi);
            }
            sum += d * sumi;
        }
        return sum;
    }
};

template <>
struct DownUnit<T_Q8_0> {
    static constexpr bool supported = true;
    int v[QI8_0];
    float d;
    __device__ __forceinline__ static DownUnit load(const char* wrow, int u) {
        const block_q8_0* b = reinterpret_cast<const block_q8_0*>(wrow) + u;
        DownUnit r;
#pragma unroll
        for (int idx = 0; idx < QI8_0; ++idx) {
            r.v[idx] = get_int_b2(b->qs, idx);
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ __forceinline__ static float dot(const DownUnit& w, const block_q8_1* y) {
        const float d8 = w.d * __half2float(__low2half(y->ds));
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < QI8_0 / VDR_Q8_0; ++p) {
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_Q8_0; ++i) {
                const int idx = p * VDR_Q8_0 + i;
                sumi = qd_dp4a(w.v[idx], get_int_b4(y->qs, idx), sumi);
            }
            sum += d8 * ((float) sumi);
        }
        return sum;
    }
};

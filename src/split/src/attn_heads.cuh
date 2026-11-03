// Included after attn_prompt.cuh. Two virtual position partitions, both local:
// cache head slot p holds this card's single KV head for virtual partition p.
// The computation and reduction order match the original kernels in kernels.cu
// and attn_prompt.cuh; only the head and page addresses differ.
__global__ void __launch_bounds__(HD) k_attn_part_head(const float* qg, const __half* kc, const __half* vc, KvPages g, float* acc_out,
                                                  AttPart* mz_out, int nq, int q0, int nkv, int kv0, const int* pos0p,
                                                  const int* __restrict__ cells, const int* __restrict__ n_cells, int owner, int T, LaneRows lr, int selection_stride) {
    // t: the window's row; its position and pages: its lane's (lr), or the sequence's *pos0p + t
    const int s = blockIdx.x, t = blockIdx.y, kvl = 0, card = blockIdx.z, i = threadIdx.x, lane = i & 31, warp = i >> 5;
    g = row_pages(g, lr, t);
    acc_out += (size_t) card * T * nq * ATT_SPLITS * HD;
    mz_out += (size_t) card * T * nq * ATT_SPLITS;
    if (cells) cells += (size_t) card * selection_stride * QSA_MAXSEL;
    if (n_cells) n_cells += card * selection_stride;
    // the attended positions (local indices on this card): the selection's list, or every owned one up to the row's
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(row_pos(lr, pos0p, t), card);
    // the local q heads of KV head kv0 + kvl: global heads [(kv0 + kvl) * QPK, + QPK) within [q0, q0 + nq)
    const int g0 = max(q0, (kv0 + kvl) * ATT_QPK), g1 = min(q0 + nq, (kv0 + kvl + 1) * ATT_QPK);
    const int h0 = g0 - q0, nh = g1 - g0;
    const int len = ((ctx + ATT_SPLITS - 1) / ATT_SPLITS + ATT_CH - 1) / ATT_CH * ATT_CH;
    const int p0 = s * len, p1 = min(ctx, p0 + len);
    __shared__ float q[ATT_MAXH][HD];
    __shared__ float kk[ATT_CH][HD + 1];
    __shared__ float e[ATT_MAXH][ATT_CH];
    __shared__ float scale[ATT_MAXH];
    for (int k = i; k < nh * HD; k += HD) q[k / HD][k % HD] = qg[((size_t) t * nq + h0 + k / HD) * 2 * HD + k % HD];
    constexpr int HPW = (ATT_MAXH + HD / 32 - 1) / (HD / 32);   // heads per warp: warp w keeps heads w, w + 8
    float m[HPW], z[HPW];
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        m[k] = -INFINITY;
        z[k] = 0.f;
    }
    float acc[ATT_MAXH] = {};
    __syncthreads();
#pragma unroll 1
    for (int c0 = p0; c0 < p1; c0 += ATT_CH) {
        const int np = min(ATT_CH, p1 - c0);
        // the chunk's K rows into shared memory and this thread's dimension of its V rows into registers
        float vr[ATT_CH];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp) {
            const int pos = pp < np ? (list ? list[c0 + pp] : c0 + pp) : 0;
            if (g.q8) {
                const size_t r = kvq_row(g, pos) + (size_t) card * HD + i, sc = kvq_scale(g, pos) + ((size_t) card * HD + i) / KV8_GROUP;
                kk[pp][i] = pp < np ? reinterpret_cast<const int8_t*>(kc)[r] * __half2float(kc[sc]) : 0.f;
                vr[pp] = pp < np ? reinterpret_cast<const int8_t*>(vc)[r] * __half2float(vc[sc]) : 0.f;
                continue;
            }
            const size_t r = kvp_row(g, pos) + (size_t) card * HD + i;   // nkv == NKVH: a row holds both heads
            kk[pp][i] = pp < np ? __half2float(kc[r]) : 0.f;
            vr[pp] = pp < np ? __half2float(vc[r]) : 0.f;
        }
        __syncthreads();
        // head warp + 8 k: lane = position; its score, then the head's online-softmax step
#pragma unroll
        for (int k = 0; k < HPW; ++k) {
            const int hh = warp + k * (HD / 32);
            if (hh >= nh) break;
            float sacc = 0.f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d) sacc = fmaf(q[hh][d], kk[lane][d], sacc);
            const float sv = lane < np ? sacc * rsqrtf((float) HD) : -INFINITY;
            const float mn = fmaxf(m[k], wmax(sv));
            const float w = sv == -INFINITY ? 0.f : __expf(sv - mn);
            const float sc = m[k] == -INFINITY ? 0.f : __expf(m[k] - mn);
            z[k] = z[k] * sc + wsum(w);
            m[k] = mn;
            e[hh][lane] = w;
            if (lane == 0) scale[hh] = sc;
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < ATT_MAXH; ++hh)
            if (hh < nh) acc[hh] *= scale[hh];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp)
#pragma unroll
            for (int hh = 0; hh < ATT_MAXH; ++hh)
                if (hh < nh) acc[hh] = fmaf(e[hh][pp], vr[pp], acc[hh]);
    }
#pragma unroll
    for (int hh = 0; hh < ATT_MAXH; ++hh) {
        if (hh >= nh) break;
        const size_t slot = ((size_t) t * nq + h0 + hh) * ATT_SPLITS + s;
        acc_out[slot * HD + i] = acc[hh];
    }
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        const int hh = warp + k * (HD / 32);
        if (hh < nh && lane == 0) mz_out[((size_t) t * nq + h0 + hh) * ATT_SPLITS + s] = AttPart{m[k], z[k]};
    }
}

__global__ void __launch_bounds__(HD) k_attn_local_head(const float* acc_in, const AttPart* mz_in, float* part, int T) {
    acc_in += (size_t) blockIdx.z * T * ATT_OWN * ATT_SPLITS * HD;
    mz_in += (size_t) blockIdx.z * T * ATT_OWN * ATT_SPLITS;
    part += (size_t) blockIdx.z * T * ATT_OWN * (HD + 2);
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t) t * ATT_OWN + h) * ATT_SPLITS;
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) mx = fmaxf(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = __expf(p.m - mx);
        z = fmaf(w, p.z, z);
        acc = fmaf(w, acc_in[(base + s) * HD + i], acc);
    }
    float* o = part + ((size_t) t * ATT_OWN + h) * (HD + 2);
    o[i] = acc;
    if (i == 0) {
        o[HD] = mx;
        o[HD + 1] = z;
    }
}


__global__ void k_attn_merge_head(const float* qg, const float* part, float* out, int T, int first) {
    const int hl = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t off = ((size_t)t * ATT_OWN + hl) * (HD + 2);
    const size_t block = (size_t) T * ATT_OWN * (HD + 2);
    const float* a0 = part + first * block + off;
    const float* a1 = part + (1 - first) * block + off;
    float m = a0[HD], z = a0[HD + 1], a = a0[i];
    att_merge(m, z, a, a1[HD], a1[HD + 1], a1[i]);
    out[((size_t)t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t)t * ATT_OWN + hl) * 2 * HD + HD + i]);
}
__global__ void __launch_bounds__(AP_THREADS) k_pf_attn_head(const float* __restrict__ qg, const __half* __restrict__ kc,
                                                        const __half* __restrict__ vc, KvPages g, const int* __restrict__ pos0p,
                                                        const int* __restrict__ cells, const int* __restrict__ n_cells, int owner,
                                                        int C, int r0, float* __restrict__ part) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char ap_smem[];
    __half* qh = reinterpret_cast<__half*>(ap_smem);           // [16][AP_LD]
    __half* kv = qh + 16 * AP_LD;                               // [APC][AP_LD]: K, then V
    __half* ph = kv + APC * AP_LD;                              // [16][AP_PLD]
    float* st = reinterpret_cast<float*>(ph + 16 * AP_PLD);     // [16][AP_LDF]: S partials, then the P V tile
    float* sm = st + 16 * AP_LDF;                               // [16]: running max
    float* sz = sm + 16;                                        // [16]: running sum
    float* ssc = sz + 16;                                       // [16]: this tile's rescale
    const int t = r0 + blockIdx.x, kvh = 0, card = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    if (cells) cells += (size_t)card * C * QSA_MAXSEL;
    if (n_cells) n_cells += card * C;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    // q rows (heads kvh * QPK + j) as f16; rows past QPK are zero
    for (int e = tid; e < 16 * HD; e += AP_THREADS) {
        const int r = e / HD, d = e % HD;
        const float x = r < ATT_QPK ? qg[((size_t) t * ATT_OWN + kvh * ATT_QPK + r) * 2 * HD + d] : 0.f;
        qh[r * AP_LD + d] = __float2half_rn(x);
    }
    if (tid < 16) {
        sm[tid] = -INFINITY;
        sz[tid] = 0.f;
    }
    // this thread's O elements: row orow, columns ocol + 16 j
    const int orow = tid / 16, ocol = tid % 16;
    float o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = 0.f;
    const float scale = rsqrtf((float) HD);
    // a tile's K or V rows, 8 halves a load, gathered into registers ahead of their use: V during S and the softmax,
    // the next tile's K during P V (each a load of the position's index, then of the row)
    constexpr int AP_GL = APC * HD / 8 / AP_THREADS;
    static_assert(APC * HD / 8 % AP_THREADS == 0, "a tile's loads spread evenly");
    auto gather = [&](const __half* base, int c, int nr, uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = make_uint4(0, 0, 0, 0);
            if (pp < nr) {
                const int pos = list ? list[c + pp] : c + pp;
                if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves
                    const size_t at = kvq_row(g, pos) + (size_t) card * HD + d8 * 8;
                    const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(base) + at);
                    const float d = __half2float(base[kvq_scale(g, pos) + ((size_t) card * HD + d8 * 8) / KV8_GROUP]);
                    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                    __align__(16) __half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn(qb[k] * d);
                    r[i] = *reinterpret_cast<const uint4*>(h);
                } else {
                    r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + (size_t) card * HD + d8 * 8);
                }
            }
        }
    };
    auto place = [&](const uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<uint4*>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, min(APC, ctx), kr);
    for (int c0 = 0; c0 < ctx; c0 += APC) {
        const int np = min(APC, ctx - c0);
        __syncthreads();   // the last tile's V and P reads are done
        place(kr);
        gather(vc, c0, np, vr);
        __syncthreads();
        // S partials: warp w takes dimensions [32 w, 32 w + 32) (two k-steps) for each 16-position tile
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[AP_NPT];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) wmma::load_matrix_sync(fa[ks], qh + warp * 32 + ks * 16, AP_LD);
#pragma unroll
            for (int pt = 0; pt < AP_NPT; ++pt) {
                wmma::fill_fragment(fs[pt], 0.f);
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    wmma::load_matrix_sync(fb, kv + pt * 16 * AP_LD + warp * 32 + ks * 16, AP_LD);
                    wmma::mma_sync(fs[pt], fa[ks], fb, fs[pt]);
                }
                // the eight warps' partials side by side in st: warp w at columns [APC w, APC w + APC)
                wmma::store_matrix_sync(st + warp * APC + pt * 16, fs[pt], AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // the scores: thread (row r, positions c + 16 u) sums the eight partials; the row's online-softmax step (16
        // lanes a row)
        {
            const int r = tid / 16, c = tid % 16;
            float sv[AP_NPT], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                float sacc = 0.f;
#pragma unroll
                for (int w = 0; w < AP_THREADS / 32; ++w) sacc += st[r * AP_LDF + w * APC + u * 16 + c];
                sv[u] = c + 16 * u < np ? sacc * scale : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 16));
            const float mo = sm[r], mn = fmaxf(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                const float p = sv[u] == -INFINITY ? 0.f : __expf(sv[u] - mn);
                ps += p;
                ph[r * AP_PLD + u * 16 + c] = __float2half_rn(p);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 16);
            __syncwarp();
            if (c == 0) {
                const float sc = mo == -INFINITY ? 0.f : __expf(mo - mn);
                ssc[r] = sc;
                sz[r] = sz[r] * sc + ps;
                sm[r] = mn;
            }
        }
        // V rows of the tile into the K buffer (its S reads are done: the barrier orders them)
        __syncthreads();
        place(vr);
        __syncthreads();
        if (c0 + APC < ctx) gather(kc, c0 + APC, min(APC, ctx - c0 - APC), kr);
        // P V: warp w takes O columns [32 w, 32 w + 32) (two 16-column tiles), the tile's positions in k-steps of 16
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp[AP_NPT];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo;
#pragma unroll
            for (int ks = 0; ks < AP_NPT; ++ks) wmma::load_matrix_sync(fp[ks], ph + ks * 16, AP_PLD);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const int d0 = warp * 32 + ct * 16;
                wmma::fill_fragment(fo, 0.f);
#pragma unroll
                for (int ks = 0; ks < AP_NPT; ++ks) {
                    wmma::load_matrix_sync(fv, kv + ks * 16 * AP_LD + d0, AP_LD);
                    wmma::mma_sync(fo, fp[ks], fv, fo);
                }
                wmma::store_matrix_sync(st + d0, fo, AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        const float sc = ssc[orow];
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = fmaf(o[j], sc, st[orow * AP_LDF + ocol + 16 * j]);
    }
    __syncthreads();
    if (orow < ATT_QPK) {
        const int h = kvh * ATT_QPK + orow, k = card, hl = h % ATT_OWN;
        float* dst = part + (((size_t) k * C + t) * ATT_OWN + hl) * AP_W;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[ocol + 16 * j] = o[j];
        if (ocol == 0) {
            dst[HD] = sm[orow];
            dst[HD + 1] = sz[orow];
        }
    }
}

__global__ void __launch_bounds__(HD) k_pf_attn_merge_head(const float* __restrict__ own, const float* __restrict__ recv,
                                                      const float* __restrict__ qg, float* __restrict__ out, int C, int card) {
    const int t = blockIdx.x, hl = blockIdx.y, i = threadIdx.x;
    const size_t at = ((size_t) t * ATT_OWN + hl) * AP_W;
    float m = own[at + HD], z = own[at + HD + 1], a = own[at + i];
#pragma unroll
    for (int k = 0; k < NCARD - 1; ++k) {
        const float* src = recv + (size_t) k * C * ATT_OWN * AP_W + at;
        att_merge(m, z, a, src[HD], src[HD + 1], src[i]);
    }
    const int hg = hl;
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * ATT_OWN + hg) * 2 * HD + HD + i]);
}

__global__ void k_kv_append_head(const float* k, const float* v, __half* kc, __half* vc, KvPages g, int card, const int* pos0p) {
    const int t = blockIdx.y, p = *pos0p + t, j = p / 4;
    const int partition = kv_card(j);
    const int r = kv_local_block(j) * 4 + p % 4;
    if (g.q8) {
        // a warp's 32 elements are one group: d = max |x| / 127, q = round(x / d) (llama.cpp's q8_0)
        static_assert(KV8_GROUP == 32, "a group per warp");
        int8_t *kq = reinterpret_cast<int8_t*>(kc), *vq = reinterpret_cast<int8_t*>(vc);
        const size_t at = kvq_row(g, r) + partition * HD, sc = kvq_scale(g, r) + partition * HD / KV8_GROUP;
        for (int e = threadIdx.x; e < HD; e += blockDim.x) {
            const float x[2] = {k[(size_t) t * HD + e], v[(size_t) t * HD + e]};
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const float d = wmax(fabsf(x[w])) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
                (w ? vq : kq)[at + e] = (int8_t) roundf(x[w] * id);
                if ((e & 31) == 0) (w ? vc : kc)[sc + e / KV8_GROUP] = __float2half_rn(d);
            }
        }
        return;
    }
    const size_t at = kvp_row(g, r) + partition * HD;
    for (int e = threadIdx.x; e < HD; e += blockDim.x) {
        kc[at + e] = __float2half_rn(k[(size_t) t * HD + e]);
        vc[at + e] = __float2half_rn(v[(size_t) t * HD + e]);
    }
}

__global__ void __launch_bounds__(HD) k_attn_prep_append_head(float* qg, float* k, const float* v, const float* qn, const float* kn, __half* kc,
                                                         __half* vc, KvPages g, int card, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x, p = *pos0p + t, j = p / 4;
    __shared__ float sh[32];
    if (h < ATT_OWN) {
        norm_rope(qg + ((size_t) t * ATT_OWN + h) * 2 * HD, qn, p, eps, base, sh);
        return;
    }
    const bool is_k = h < ATT_OWN + 1;
    const int kh = h - ATT_OWN - (is_k ? 0 : 1), e = kh * HD + i;
    float x;
    if (is_k) {
        float* kr = k + ((size_t) t * 1 + kh) * HD;
        norm_rope(kr, kn, p, eps, base, sh);
        __syncthreads();
        x = kr[i];
    } else {
        x = v[(size_t) t * 1 * HD + e];
    }
    const int ce = kv_card(j) * HD + i;
    const int r = kv_local_block(j) * 4 + p % 4;
    __half* cache = is_k ? kc : vc;
    if (g.q8) {
        static_assert(KV8_GROUP == 32 && HD % 32 == 0, "a group per warp");
        const float d = wmax(fabsf(x)) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        reinterpret_cast<int8_t*>(cache)[kvq_row(g, r) + ce] = (int8_t) roundf(x * id);
        if ((e & 31) == 0) cache[kvq_scale(g, r) + ce / KV8_GROUP] = __float2half_rn(d);
        return;
    }
    cache[kvp_row(g, r) + ce] = __float2half_rn(x);
}

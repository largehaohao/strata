#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
#include <memory>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(s) + cudaGetErrorString(e));
}
struct Memory {
    std::vector<void*> values;
    template<class T> T* get(size_t n) {
        void* p; ck(cudaMalloc(&p, n * sizeof(T)), "alloc"); values.push_back(p); return (T*)p;
    }
    ~Memory() { for (void* p : values) cudaFree(p); }
};
__global__ void fill_kv(__half* k0, __half* k1, __half* v0, __half* v1,
                        __half* hk0, __half* hk1, __half* hv0, __half* hv1, KvPages g, int count) {
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)count * NKVH * HD) return;
    const int p = e / (NKVH * HD), h = e / HD % NKVH, d = e % HD;
    const int card = kv_card(p / 4), row = kv_local_block(p / 4) * 4 + p % 4;
    __half *kr = card ? k1 : k0, *vr = card ? v1 : v0;
    __half *kh = h ? hk1 : hk0, *vh = h ? hv1 : hv0;
    const int8_t k = (int)((e * 13 + e / 41) % 255) - 127;
    const int8_t v = (int)((e * 19 + e / 29) % 255) - 127;
    const __half ks = __float2half(0.00390625f * (1 + (p + h + d / 32) % 3));
    const __half vs = __float2half(0.001953125f * (1 + (p + h + d / 32) % 5));
    if (g.q8) {
        const size_t a = kvq_row(g, row), s = kvq_scale(g, row);
        ((int8_t*)kr)[a + h * HD + d] = ((int8_t*)kh)[a + card * HD + d] = k;
        ((int8_t*)vr)[a + h * HD + d] = ((int8_t*)vh)[a + card * HD + d] = v;
        if (d % 32 == 0) {
            kr[s + (h * HD + d) / 32] = kh[s + (card * HD + d) / 32] = ks;
            vr[s + (h * HD + d) / 32] = vh[s + (card * HD + d) / 32] = vs;
        }
    } else {
        const size_t a = kvp_row(g, row);
        kr[a + h * HD + d] = kh[a + card * HD + d] = __float2half(k * __half2float(ks));
        vr[a + h * HD + d] = vh[a + card * HD + d] = __float2half(v * __half2float(vs));
    }
}
__global__ void fill_query(float* q, float* q0, float* q1, int T, int seed) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= T * NQH * 2 * HD) return;
    const int h = e / (2 * HD) % NQH, t = e / (NQH * 2 * HD), d = e % (2 * HD);
    const float v = seed == 2 ? 0.f : ((int)((e * 17 + e / 31 + seed * 23) % 511) - 255) / 256.f;
    q[e] = v;
    (h < ATT_OWN ? q0 : q1)[((size_t)t * ATT_OWN + h % ATT_OWN) * 2 * HD + d] = v;
}
__global__ void prepare_mail(const float* part, unsigned long long* mail, int owner, int T, unsigned tag) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int W = HD + 2;
    if (e >= T * ATT_OWN * W) return;
    const int t = e / (ATT_OWN * W), h = e / W % ATT_OWN, d = e % W;
    const float v = part[((size_t)t * NQH + owner * ATT_OWN + h) * W + d];
    mail[(tag & 1) * (size_t)ATT_ROWS * ATT_OWN * W + e] = __float_as_uint(v) | ((unsigned long long)tag << 32);
}
int run(int dev, bool q8) {
    constexpr int count = 32768, pages = (count + KVP_POS - 1) / KVP_POS;
    constexpr int pad = 64;
    Memory mem;
    std::vector<int> pt(pages);
    for (int i = 0; i < pages; ++i) pt[i] = pages - 1 - i;
    auto* dpt = mem.get<int>(pages);
    ck(cudaMemcpy(dpt, pt.data(), pages * sizeof(int), cudaMemcpyHostToDevice), "pt");
    KvPages g{dpt, kvp_rows_halves(0, q8) + 64, kvp_rows(0), q8};
    __half *k[2], *v[2], *hk[2], *hv[2];
    for (int c = 0; c < 2; ++c) {
        k[c] = mem.get<__half>(pages * g.page); v[c] = mem.get<__half>(pages * g.page);
        hk[c] = mem.get<__half>(pages * g.page); hv[c] = mem.get<__half>(pages * g.page);
    }
    fill_kv<<<(count * NKVH * HD + 255) / 256, 256>>>(k[0], k[1], v[0], v[1], hk[0], hk[1], hv[0], hv[1], g, count);
    auto* q = mem.get<float>(MAXT * NQH * 2 * HD);
    float* hq[2] = {mem.get<float>(MAXT * ATT_OWN * 2 * HD), mem.get<float>(MAXT * ATT_OWN * 2 * HD)};
    auto* pos = mem.get<int>(1); auto* tag = mem.get<unsigned>(1); auto* err = mem.get<unsigned>(1);
    auto* cells = mem.get<int>(2 * MAXT * QSA_MAXSEL); auto* ncells = mem.get<int>(2 * MAXT);
    float *acc[2], *part[2]; AttPart* mz[2];
    for (int c = 0; c < 2; ++c) {
        acc[c] = mem.get<float>(MAXT * NQH * ATT_SPLITS * HD);
        mz[c] = mem.get<AttPart>(MAXT * NQH * ATT_SPLITS);
        part[c] = mem.get<float>(MAXT * NQH * (HD + 2));
    }
    auto* ha = mem.get<float>(2 * MAXT * ATT_OWN * ATT_SPLITS * HD);
    auto* hm = mem.get<AttPart>(2 * MAXT * ATT_OWN * ATT_SPLITS);
    auto* hp = mem.get<float>(2 * MAXT * ATT_OWN * (HD + 2));
    const size_t out_n = 2 * pad + MAXT * ATT_OWN * HD, out_bytes = out_n * sizeof(float);
    auto* out = mem.get<float>(out_n);
    const size_t mail_n = 2 * ATT_ROWS * ATT_OWN * (HD + 2);
    auto* mail = mem.get<unsigned long long>(mail_n); auto* dummy = mem.get<unsigned long long>(mail_n);
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    ck(cudaMemset(err, 0, 4), "err"); ck(cudaDeviceSynchronize(), "init");
    cudaEvent_t a,b; ck(cudaEventCreate(&a),"event"); ck(cudaEventCreate(&b),"event");
    int checks = 0;
    for (int start : {0, 3, 4, 193, 2048, 32699}) for (int T = 1; T <= MAXT; ++T) for (int seed : {0, 1, 2}) {
        ck(cudaMemcpyAsync(pos, &start, 4, cudaMemcpyHostToDevice, st), "pos");
        const unsigned seq = checks + 5, tagval = seq * 256u + 4u;
        ck(cudaMemcpyAsync(tag, &seq, 4, cudaMemcpyHostToDevice, st), "tag");
        fill_query<<<(T * NQH * 2 * HD + 255) / 256, 256, 0, st>>>(q, hq[0], hq[1], T, seed);
        std::vector<int> hc(2 * T * QSA_MAXSEL), hn(2 * T);
        for (int c = 0; c < 2; ++c) for (int t = 0; t < T; ++t) {
            const int p = start + t, block = p / 4;
            int ctx = ((block + 1 - c) / 2) * 4 + (block % 2 == c ? p % 4 + 1 : 0);
            const int n = seed == 1 ? std::min(ctx / 3, QSA_MAXSEL) : -1;
            hn[c * T + t] = n;
            for (int i = 0; i < n; ++i) hc[((size_t)c * T + t) * QSA_MAXSEL + i] = i * 3;
        }
        ck(cudaMemcpyAsync(cells, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice, st), "cells");
        ck(cudaMemcpyAsync(ncells, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice, st), "ncells");
        auto refpart = [&](int c) {
            k_attn_part<<<dim3(ATT_SPLITS, T, NKVH), HD, 0, st>>>(q,k[c],v[c],g,acc[c],mz[c],NQH,0,NKVH,0,pos,cells+(size_t)c*T*QSA_MAXSEL,ncells+c*T,c,{});
            k_attn_local<<<dim3(NQH,T),HD,0,st>>>(acc[c],mz[c],part[c]);
        };
        for (int c = 0; c < 2; ++c) refpart(c);
        for (int owner = 0; owner < 2; ++owner) for (bool onehop : {false, true}) {
            AttRing ring; ring.card=owner; ring.in=mail; ring.peer_in[1-owner]=dummy; ring.err=err; ring.next_in=dummy;
            prepare_mail<<<(T*ATT_OWN*(HD+2)+255)/256,256,0,st>>>(part[1-owner],mail,owner,T,tagval);
            auto ref = [&]() {
                if (onehop) k_att_merge_a2a<<<dim3(ATT_OWN,T),HD,0,st>>>(ring,q,part[owner],out+pad,T,tag,3);
                else k_att_merge<<<dim3(ATT_OWN,T),HD,0,st>>>(ring,q,part[owner],out+pad,T,tag,3);
            };
            auto test = [&]() {
                k_attn_part_head<<<dim3(ATT_SPLITS,T,2),HD,0,st>>>(hq[owner],hk[owner],hv[owner],g,ha,hm,ATT_OWN,owner*ATT_OWN,1,owner,pos,cells,ncells,owner,T,{},T);
                k_attn_local_head<<<dim3(ATT_OWN,T,2),HD,0,st>>>(ha,hm,hp,T);
                k_attn_merge_head<<<dim3(ATT_OWN,T),HD,0,st>>>(hq[owner],hp,out+pad,T,onehop ? 0 : 1-owner);
            };
            std::vector<unsigned char> expected(out_bytes), actual(out_bytes);
            ck(cudaMemsetAsync(out,0xA5,out_bytes,st),"guard"); ref(); ck(cudaStreamSynchronize(st),"ref");
            ck(cudaMemcpy(expected.data(),out,out_bytes,cudaMemcpyDeviceToHost),"ref result");
            cudaGraph_t graph; cudaGraphExec_t ex;
            ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture"); test();
            ck(cudaStreamEndCapture(st,&graph),"capture end"); ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"graph");
            for (bool replay : {false,true}) {
                ck(cudaMemsetAsync(out,0xA5,out_bytes,st),"guard"); if(replay)ck(cudaGraphLaunch(ex,st),"replay");else test();
                ck(cudaStreamSynchronize(st),"test"); ck(cudaMemcpy(actual.data(),out,out_bytes,cudaMemcpyDeviceToHost),"result");
                if(actual!=expected) {
                    size_t nbad=0, first=0; float maxdiff=0;
                    for(size_t i=0;i<out_n;++i) if(std::memcmp(actual.data()+4*i,expected.data()+4*i,4)) {
                        if(!nbad) first=i; ++nbad;
                        maxdiff=std::max(maxdiff,std::fabs(((float*)actual.data())[i]-((float*)expected.data())[i]));
                    }
                    std::cerr<<"mismatch dev "<<dev<<" q8 "<<q8<<" pos "<<start<<" T "<<T<<" seed "<<seed<<" owner "<<owner<<" replay "<<replay<<" bad "<<nbad<<" first "<<first<<" max "<<maxdiff<<"\n";
                    throw std::runtime_error("attention head mismatch");
                }
                ++checks;
            }
            cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
            // The prompt path uses WMMA and its own merge order (owner, then peer).
            for (int c=0;c<2;++c) k_pf_attn<<<dim3(T,NKVH),AP_THREADS,AP_SMEM,st>>>(q,k[c],v[c],g,pos,cells+(size_t)c*T*QSA_MAXSEL,ncells+c*T,c,T,0,part[c]);
            const size_t blk=(size_t)T*ATT_OWN*(HD+2);
            ck(cudaMemsetAsync(out,0xA5,out_bytes,st),"pf guard");
            k_pf_attn_merge<<<dim3(T,ATT_OWN),HD,0,st>>>(part[owner]+owner*blk,part[1-owner]+owner*blk,q,out+pad,T,owner);
            ck(cudaStreamSynchronize(st),"pf reference");ck(cudaMemcpy(expected.data(),out,out_bytes,cudaMemcpyDeviceToHost),"pf reference copy");
            auto pftest=[&](){
                k_pf_attn_head<<<dim3(T,2),AP_THREADS,AP_SMEM,st>>>(hq[owner],hk[owner],hv[owner],g,pos,cells,ncells,owner,T,0,hp);
                k_pf_attn_merge_head<<<dim3(T,ATT_OWN),HD,0,st>>>(hp+owner*blk,hp+(1-owner)*blk,hq[owner],out+pad,T,owner);
            };
            ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"pf capture");pftest();ck(cudaStreamEndCapture(st,&graph),"pf capture end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"pf graph");
            for(bool replay:{false,true}){
                ck(cudaMemsetAsync(out,0xA5,out_bytes,st),"pf guard");if(replay)ck(cudaGraphLaunch(ex,st),"pf replay");else pftest();
                ck(cudaStreamSynchronize(st),"pf wait");ck(cudaMemcpy(actual.data(),out,out_bytes,cudaMemcpyDeviceToHost),"pf result");
                if(actual!=expected)throw std::runtime_error("prefill attention mismatch dev="+std::to_string(dev)+" T="+std::to_string(T)+" pos="+std::to_string(start)+" owner="+std::to_string(owner));
                ++checks;
            }
            cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
            for (int c=0;c<2;++c) refpart(c); // Restore decode layout for the next owner/timing.

        }
    }
    unsigned errors;ck(cudaMemcpy(&errors,err,4,cudaMemcpyDeviceToHost),"errors");
    if(errors)throw std::runtime_error("ring reference error");
    cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(st);
    return checks;
}
}

// Frozen original append kernels, omitted by FNX_KPART=99 above.
namespace fnx {
__global__ void k_kv_append(const float* k, const float* v, __half* kc, __half* vc, KvPages g, int card, const int* pos0p) {
    const int t = blockIdx.y, p = *pos0p + t, j = p / 4;
    if (kv_card(j) != card) return;
    const int r = kv_local_block(j) * 4 + p % 4;
    if (g.q8) {
        // a warp's 32 elements are one group: d = max |x| / 127, q = round(x / d) (llama.cpp's q8_0)
        static_assert(KV8_GROUP == 32, "a group per warp");
        int8_t *kq = reinterpret_cast<int8_t*>(kc), *vq = reinterpret_cast<int8_t*>(vc);
        const size_t at = kvq_row(g, r), sc = kvq_scale(g, r);
        for (int e = threadIdx.x; e < NKVH * HD; e += blockDim.x) {
            const float x[2] = {k[(size_t) t * NKVH * HD + e], v[(size_t) t * NKVH * HD + e]};
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const float d = wmax(fabsf(x[w])) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
                (w ? vq : kq)[at + e] = (int8_t) roundf(x[w] * id);
                if ((e & 31) == 0) (w ? vc : kc)[sc + e / KV8_GROUP] = __float2half_rn(d);
            }
        }
        return;
    }
    const size_t at = kvp_row(g, r);
    for (int e = threadIdx.x; e < NKVH * HD; e += blockDim.x) {
        kc[at + e] = __float2half_rn(k[(size_t) t * NKVH * HD + e]);
        vc[at + e] = __float2half_rn(v[(size_t) t * NKVH * HD + e]);
    }
}

// k_attn_prep and k_kv_append in one launch: block (h, row t), HD threads; h < NQH: q head h normed and rotated; then
// the NKVH k heads, normed, rotated and appended; then the NKVH v heads, appended (a row whose position is another
// card's appends nothing). The same values as the two kernels
__global__ void __launch_bounds__(HD) k_attn_prep_append(float* qg, float* k, const float* v, const float* qn, const float* kn, __half* kc,
                                                         __half* vc, KvPages g, int card, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x, p = *pos0p + t, j = p / 4;
    __shared__ float sh[32];
    if (h < NQH) {
        norm_rope(qg + ((size_t) t * NQH + h) * 2 * HD, qn, p, eps, base, sh);
        return;
    }
    const bool is_k = h < NQH + NKVH;
    const int kh = h - NQH - (is_k ? 0 : NKVH), e = kh * HD + i;
    float x;
    if (is_k) {
        float* kr = k + ((size_t) t * NKVH + kh) * HD;
        norm_rope(kr, kn, p, eps, base, sh);
        __syncthreads();
        x = kr[i];
    } else {
        x = v[(size_t) t * NKVH * HD + e];
    }
    if (kv_card(j) != card) return;
    const int r = kv_local_block(j) * 4 + p % 4;
    __half* cache = is_k ? kc : vc;
    if (g.q8) {
        static_assert(KV8_GROUP == 32 && HD % 32 == 0, "a group per warp");
        const float d = wmax(fabsf(x)) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        reinterpret_cast<int8_t*>(cache)[kvq_row(g, r) + e] = (int8_t) roundf(x * id);
        if ((e & 31) == 0) cache[kvq_scale(g, r) + e / KV8_GROUP] = __float2half_rn(d);
        return;
    }
    cache[kvp_row(g, r) + e] = __float2half_rn(x);
}

int append_test(int dev, bool q8) {
    Memory mem;
    const std::vector<int> table{2,0};
    int* pt=mem.get<int>(2);ck(cudaMemcpy(pt,table.data(),8,cudaMemcpyHostToDevice),"table");
    KvPages g{pt,kvp_rows_halves(0,q8)+64,kvp_rows(0),q8};
    const size_t cache_bytes=3*g.page*2;
    __half *rk[2],*rv[2],*hk[2],*hv[2];float *q[2],*k[2],*v[2],*qh[2],*kh[2],*vh[2];
    for(int c=0;c<2;++c){
        rk[c]=mem.get<__half>(cache_bytes/2);rv[c]=mem.get<__half>(cache_bytes/2);
        hk[c]=mem.get<__half>(cache_bytes/2);hv[c]=mem.get<__half>(cache_bytes/2);
        q[c]=mem.get<float>(MAXT*NQH*2*HD);k[c]=mem.get<float>(MAXT*NKVH*HD);v[c]=mem.get<float>(MAXT*NKVH*HD);
        qh[c]=mem.get<float>(MAXT*ATT_OWN*2*HD);kh[c]=mem.get<float>(MAXT*HD);vh[c]=mem.get<float>(MAXT*HD);
    }
    std::vector<float> weights(HD);for(int i=0;i<HD;++i)weights[i]=0.5f+(i%11)/8.f;
    float* norm=mem.get<float>(HD);ck(cudaMemcpy(norm,weights.data(),HD*4,cudaMemcpyHostToDevice),"norm");
    int* pos=mem.get<int>(1);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
    int checks=0;
    for(int T=1;T<=MAXT;++T)for(int p0:{0,3,191})for(int seed:{0,1})for(bool fused:{false,true}){
        std::vector<float> iq(T*NQH*2*HD),ik(T*NKVH*HD),iv(ik.size());
        for(size_t i=0;i<iq.size();++i)iq[i]=seed?0.f:((int)((i*19+i/37)%511)-255)/64.f;
        for(size_t i=0;i<ik.size();++i){ik[i]=seed?0.f:((int)((i*13+i/29)%511)-255)/128.f;iv[i]=seed?0.f:((int)((i*23+i/31)%511)-255)/32.f;}
        ck(cudaMemcpyAsync(pos,&p0,4,cudaMemcpyHostToDevice,st),"pos");
        std::vector<unsigned char> refk[2],refv[2];std::vector<float> refq[2];
        for(int c=0;c<2;++c){
            ck(cudaMemcpyAsync(q[c],iq.data(),iq.size()*4,cudaMemcpyHostToDevice,st),"q");
            ck(cudaMemcpyAsync(k[c],ik.data(),ik.size()*4,cudaMemcpyHostToDevice,st),"k");
            ck(cudaMemcpyAsync(v[c],iv.data(),iv.size()*4,cudaMemcpyHostToDevice,st),"v");
            ck(cudaMemsetAsync(rk[c],0xA5,cache_bytes,st),"guard");ck(cudaMemsetAsync(rv[c],0xA5,cache_bytes,st),"guard");
            if(fused)k_attn_prep_append<<<dim3(NQH+2*NKVH,T),HD,0,st>>>(q[c],k[c],v[c],norm,norm,rk[c],rv[c],g,c,pos,1e-6f,Rope{});
            else k_kv_append<<<dim3(1,T),256,0,st>>>(k[c],v[c],rk[c],rv[c],g,c,pos);
            ck(cudaStreamSynchronize(st),"ref wait");refk[c].resize(cache_bytes);refv[c].resize(cache_bytes);refq[c].resize(iq.size());
            ck(cudaMemcpy(refk[c].data(),rk[c],cache_bytes,cudaMemcpyDeviceToHost),"rk");
            ck(cudaMemcpy(refv[c].data(),rv[c],cache_bytes,cudaMemcpyDeviceToHost),"rv");
            ck(cudaMemcpy(refq[c].data(),q[c],iq.size()*4,cudaMemcpyDeviceToHost),"rq");
        }
        for(int c=0;c<2;++c){
            std::vector<unsigned char> ek(cache_bytes,0xA5),ev(ek),ak(cache_bytes),av(cache_bytes);
            std::vector<float> inq(T*ATT_OWN*2*HD),ink(T*HD),inv(ink.size()),eq(inq.size()),aq(inq.size());
            for(int t=0;t<T;++t){
                std::memcpy(inq.data()+t*ATT_OWN*2*HD,iq.data()+((size_t)t*NQH+c*ATT_OWN)*2*HD,ATT_OWN*2*HD*4);
                std::memcpy(eq.data()+t*ATT_OWN*2*HD,refq[c].data()+((size_t)t*NQH+c*ATT_OWN)*2*HD,ATT_OWN*2*HD*4);
                std::memcpy(ink.data()+t*HD,ik.data()+((size_t)t*NKVH+c)*HD,HD*4);
                std::memcpy(inv.data()+t*HD,iv.data()+((size_t)t*NKVH+c)*HD,HD*4);
                const int p=p0+t,part=p/4%2,row=p/8*4+p%4;
                const size_t page=table[row/g.rows]*g.page*2, local=row%g.rows;
                if(q8){
                    const size_t r=page+local*NKVH*HD,scale=page+(size_t)g.rows*NKVH*HD+local*NKVH*HD/KV8_GROUP*2;
                    std::memcpy(ek.data()+r+part*HD,refk[part].data()+r+c*HD,HD);std::memcpy(ev.data()+r+part*HD,refv[part].data()+r+c*HD,HD);
                    std::memcpy(ek.data()+scale+part*HD/KV8_GROUP*2,refk[part].data()+scale+c*HD/KV8_GROUP*2,HD/KV8_GROUP*2);
                    std::memcpy(ev.data()+scale+part*HD/KV8_GROUP*2,refv[part].data()+scale+c*HD/KV8_GROUP*2,HD/KV8_GROUP*2);
                }else{
                    const size_t r=page+local*NKVH*HD*2;
                    std::memcpy(ek.data()+r+part*HD*2,refk[part].data()+r+c*HD*2,HD*2);std::memcpy(ev.data()+r+part*HD*2,refv[part].data()+r+c*HD*2,HD*2);
                }
            }
            auto call=[&](){if(fused)k_attn_prep_append_head<<<dim3(ATT_OWN+2,T),HD,0,st>>>(qh[c],kh[c],vh[c],norm,norm,hk[c],hv[c],g,c,pos,1e-6f,Rope{});
                else k_kv_append_head<<<dim3(1,T),256,0,st>>>(kh[c],vh[c],hk[c],hv[c],g,c,pos);};
            cudaGraph_t graph;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");call();
            ck(cudaStreamEndCapture(st,&graph),"capture end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"graph");
            for(bool replay:{false,true}){
                ck(cudaMemcpyAsync(qh[c],inq.data(),inq.size()*4,cudaMemcpyHostToDevice,st),"qh");ck(cudaMemcpyAsync(kh[c],ink.data(),ink.size()*4,cudaMemcpyHostToDevice,st),"kh");ck(cudaMemcpyAsync(vh[c],inv.data(),inv.size()*4,cudaMemcpyHostToDevice,st),"vh");
                ck(cudaMemsetAsync(hk[c],0xA5,cache_bytes,st),"guard");ck(cudaMemsetAsync(hv[c],0xA5,cache_bytes,st),"guard");
                if(replay)ck(cudaGraphLaunch(ex,st),"replay");else call();ck(cudaStreamSynchronize(st),"wait");
                ck(cudaMemcpy(ak.data(),hk[c],cache_bytes,cudaMemcpyDeviceToHost),"ak");ck(cudaMemcpy(av.data(),hv[c],cache_bytes,cudaMemcpyDeviceToHost),"av");ck(cudaMemcpy(aq.data(),qh[c],aq.size()*4,cudaMemcpyDeviceToHost),"aq");
                if(ak!=ek||av!=ev||std::memcmp(aq.data(),eq.data(),aq.size()*4))throw std::runtime_error("append mismatch T="+std::to_string(T)+" p="+std::to_string(p0)+" fused="+std::to_string(fused));
                ++checks;
            }
            cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
        }
    }
    cudaStreamDestroy(st);return checks;
}
}
int main() {
    try {
        int checks = 0;
        for (int dev = 0; dev < 2; ++dev) {
            fnx::ck(cudaSetDevice(dev), "device");
            for (bool q8 : {false, true}) {
                checks += fnx::run(dev, q8);
                checks += fnx::append_test(dev, q8);
            }
        }
        std::cout << "PASS " << checks << " exact head attention and append checks\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

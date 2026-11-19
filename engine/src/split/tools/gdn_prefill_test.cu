// Compare both production recurrent layouts, including state carried between
// chunks. The narrow layout is the existing reference; no model files needed.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <random>
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffers {
    std::vector<void*> ptrs;
    template <class T> T* get(size_t n) {
        T* p; check(cudaMalloc((void**)&p, n * sizeof(T))); ptrs.push_back(p); return p;
    }
    ~Buffers() { for (void* p : ptrs) (void)cudaFree(p); }
};

int run_gdn_case(int C, int nv, int seed, cudaStream_t stream) {
    constexpr int pad = 32;
    constexpr float guard = -7891.25f;
    const int nk = std::min(nv, 8), ch = (2 * nk + nv) * HS;
    const size_t ns = (size_t)nv * HS * HS, ny = (size_t)C * nv * HS;
    std::mt19937 random(713 + C + nv + seed);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    std::vector<float> xc((size_t)C * ch), ab((size_t)C * 2 * nv), initial(ns + 2 * pad, guard);
    std::vector<float> dt(nv, .125f), decay(nv, -.1f);
    std::vector<int> map(nv);
    for (int h=0;h<nv;++h) map[h] = (h * 3 + 1) % nk;
    for (int t=0;t<C;++t) for (int h=0;h<2*nk+nv;++h) for (int d=0;d<HS;++d)
        xc[((size_t)t * (2*nk+nv) + h) * HS + d] = uniform(random) * (h < 2*nk ? .125f : 1.f);
    for (float& v : ab) v = uniform(random) * 3.f;
    for (size_t i=0;i<ns;++i) initial[pad+i] = seed ? uniform(random) * .25f : 0.f;
    Buffers mem;
    auto *x=mem.get<float>(xc.size()), *a=mem.get<float>(ab.size());
    auto *d=mem.get<float>(nv), *g=mem.get<float>(nv);
    auto *m=mem.get<int>(nv);
    const size_t nprep=(size_t)((C+GCH-1)/GCH)*nv*GCH_PREP_FLOATS;
    auto *prep=mem.get<float>(nprep+2*pad), *compact=mem.get<float>(nprep+2*pad);
    float* state[2] = {mem.get<float>(initial.size()),mem.get<float>(initial.size())};
    float* output[2] = {mem.get<float>(ny+2*pad),mem.get<float>(ny+2*pad)};
    auto copy = [&](void* dst,const void* src,size_t n) { check(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,stream)); };
    copy(x,xc.data(),xc.size()*4); copy(a,ab.data(),ab.size()*4);
    copy(d,dt.data(),nv*4); copy(g,decay.data(),nv*4); copy(m,map.data(),nv*4);
    for (auto* s:state) copy(s,initial.data(),initial.size()*4);
    std::vector<float> pref(nprep+2*pad,guard), pgot(pref.size());
    auto prep_call = [&](bool small) {
        if(small) k_gdn_ch_prep_compact<<<dim3((C+GCH-1)/GCH,nv),GCH_THREADS,0,stream>>>(x,a,d,g,m,nk,nv,C,compact+pad);
        else k_gdn_ch_prep<<<dim3((C+GCH-1)/GCH,nv),GCH_THREADS,0,stream>>>(x,a,d,g,m,nk,nv,C,prep+pad);
    };
    for(bool graph:{false,true}) {
        std::fill(pref.begin(),pref.end(),guard);
        copy(prep,pref.data(),pref.size()*4); copy(compact,pref.data(),pref.size()*4);
        if(graph) {
            cudaGraph_t gr; cudaGraphExec_t ex;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed)); prep_call(false);prep_call(true);
            check(cudaStreamEndCapture(stream,&gr)); check(cudaGraphInstantiate(&ex,gr,nullptr,nullptr,0));
            for(int k=0;k<3;++k) check(cudaGraphLaunch(ex,stream));
            check(cudaStreamSynchronize(stream));check(cudaGraphExecDestroy(ex));check(cudaGraphDestroy(gr));
        } else {prep_call(false);prep_call(true);}
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(pref.data(),prep,pref.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(pgot.data(),compact,pgot.size()*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(pref.data(),pgot.data(),pref.size()*4)) throw std::runtime_error("GDN compact prep changed bits");
        for(int i=0;i<pad;++i) if(pgot[i]!=guard || pgot[pgot.size()-1-i]!=guard)
            throw std::runtime_error("GDN compact prep guard overwritten");
    }
    auto call = [&](int which) {
        if (which == 0)
            k_gdn_ch_rec<GCB,false><<<nv*GCH_REC_BLOCKS_PER_HEAD,GCR_THREADS,GCH_REC_SMEM,stream>>>(x,prep+pad,state[0]+pad,output[0]+pad,nk,nv,m,C);
        else
            k_gdn_ch_rec<32,true><<<nv*(HS/32),256,GCH_REC_WIDE_SMEM,stream>>>(x,compact+pad,state[1]+pad,output[1]+pad,nk,nv,m,C);
    };
    std::vector<float> expected(ny+2*pad,guard),actual(expected.size()),se(initial.size()),sa(initial.size());
    for (int pass=0;pass<2;++pass) {
        std::fill(expected.begin(),expected.end(),guard);
        for (auto* y:output) copy(y,expected.data(),expected.size()*4);
        call(0);call(1);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(expected.data(),output[0],expected.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(),output[1],actual.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(se.data(),state[0],se.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(sa.data(),state[1],sa.size()*4,cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(),expected.data(),actual.size()*4) || std::memcmp(sa.data(),se.data(),sa.size()*4))
            throw std::runtime_error("GDN layout changed output/state bits C="+std::to_string(C)+" nv="+std::to_string(nv)+" pass="+std::to_string(pass));
        for (size_t i=pad;i<pad+ny;++i) if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite recurrent output");
        for (size_t i=pad;i<pad+ns;++i) if (!std::isfinite(sa[i])) throw std::runtime_error("nonfinite recurrent state");
        for (int i=0;i<pad;++i)
            if (actual[i]!=guard || actual[pad+ny+i]!=guard || sa[i]!=guard || sa[pad+ns+i]!=guard)
                throw std::runtime_error("recurrent output/state guard overwritten");
    }
    if (C>=1024 && nv==24 && seed==1) {
        cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
        float times[2];
        for (int which=0;which<2;++which) {
            check(cudaEventRecord(start,stream));
            for (int i=0;i<5;++i) call(which);
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            check(cudaEventElapsedTime(&times[which],start,end));
        }
        std::printf("GDN C=%d nv=%d: narrow %.3f ms wide %.3f ms\n",C,nv,times[0]/5,times[1]/5);
        float prep_ms[2]={};
        for(int round=0;round<3;++round) for(int i=0;i<2;++i) {
            const int which=(i+round)%2;
            for(int j=0;j<3;++j) prep_call(which);
            check(cudaEventRecord(start,stream));
            for(int j=0;j<10;++j) prep_call(which);
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float dt;check(cudaEventElapsedTime(&dt,start,end));prep_ms[which]+=dt/30;
        }
        std::printf("GDN prep C=%d nv=%d: original %.3f ms compact %.3f ms\n",C,nv,prep_ms[0],prep_ms[1]);
        check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    }
    return 4;
}

int run_row_case(int C, int nv, int parity, cudaStream_t stream) {
    const int nk = std::min(nv, 8), ch = (2 * nk + nv) * HS;
    constexpr int pad = 32;
    constexpr float guard = -7891.25f;
    std::mt19937 random(317 + C + nv + parity);
    std::uniform_real_distribution<float> uniform(-3.f, 3.f);
    std::vector<float> q((size_t)C * (ch + nv * HS)), hist(6 * ch), weights(4 * ch), norm(HS);
    for (auto* v : {&q, &hist, &weights, &norm}) for (float& f : *v) f = uniform(random);
    std::vector<float> cx((size_t)C * ch + 2 * pad, guard), ny((size_t)C * nv * HS + 2 * pad, guard);
    for (size_t i = pad; i < ny.size() - pad; ++i) ny[i] = uniform(random);
    // Include exact zero rows and tiny nonzero inputs in both implementations.
    if (C > 1) {
        std::fill(q.begin() + ch + nv * HS, q.begin() + 2 * (ch + nv * HS), 0.f);
        for (int i = 0; i < nv * HS; ++i) ny[pad + nv * HS + i] = (i & 1) ? 1e-20f : -1e-20f;
    }
    int ctl[4] = {0, 0, 0, parity};
    Buffers mem;
    auto *dq = mem.get<float>(q.size()), *dh = mem.get<float>(hist.size()), *dw = mem.get<float>(weights.size());
    auto *dn = mem.get<float>(norm.size()); auto *dc = mem.get<int>(4);
    float* conv[2] = {mem.get<float>(cx.size()), mem.get<float>(cx.size())};
    float* y[2] = {mem.get<float>(ny.size()), mem.get<float>(ny.size())};
    auto copy = [&](void* dst, const void* src, size_t n) { check(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream)); };
    copy(dq,q.data(),q.size()*4); copy(dh,hist.data(),hist.size()*4); copy(dw,weights.data(),weights.size()*4);
    copy(dn,norm.data(),norm.size()*4); copy(dc,ctl,sizeof(ctl));
    auto conv_call = [&](int which) {
        if (which) k_gdn_pf_conv_rows<<<(C*(2*nk+nv)+GDN_PF_ROW_WAVES-1)/GDN_PF_ROW_WAVES,
            GDN_PF_ROW_THREADS,0,stream>>>(dq,dh,dw,dc,conv[which]+pad,nk,nv,C);
        else k_gdn_pf_conv<<<dim3(C,2*nk+nv),HS,0,stream>>>(dq,dh,dw,dc,conv[which]+pad,nk,nv);
    };
    auto norm_call = [&](int which) {
        if (which) k_gdn_pf_norm_rows<<<(C*nv+GDN_PF_ROW_WAVES-1)/GDN_PF_ROW_WAVES,
            GDN_PF_ROW_THREADS,0,stream>>>(dq,dn,y[which]+pad,nk,nv,1e-5f,C);
        else k_gdn_pf_norm_own<<<dim3(C,nv),HS,0,stream>>>(dq,dn,y[which]+pad,nk,nv,1e-5f);
    };
    std::vector<float> cr(cx.size()), ca(cx.size()), nr(ny.size()), na(ny.size());
    for (bool graph : {false, true}) {
        for (int which = 0; which < 2; ++which) {
            copy(conv[which],cx.data(),cx.size()*4); copy(y[which],ny.data(),ny.size()*4);
            if (graph) {
                cudaGraph_t g; cudaGraphExec_t ex;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
                conv_call(which); norm_call(which);
                check(cudaStreamEndCapture(stream,&g));
                check(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
                for (int k=0;k<3;++k) check(cudaGraphLaunch(ex,stream));
                check(cudaStreamSynchronize(stream));
                check(cudaGraphExecDestroy(ex)); check(cudaGraphDestroy(g));
            } else { conv_call(which); norm_call(which); }
        }
        check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(cr.data(),conv[0],cr.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(ca.data(),conv[1],ca.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(nr.data(),y[0],nr.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(na.data(),y[1],na.size()*4,cudaMemcpyDeviceToHost));
        if (std::memcmp(cr.data(),ca.data(),cr.size()*4) || std::memcmp(nr.data(),na.data(),nr.size()*4)) {
            for (size_t i=0;i<cr.size();++i) if (std::memcmp(&cr[i],&ca[i],4)) {
                std::fprintf(stderr,"conv at %zu reference %.9g rows %.9g parity %d graph %d\n",i,cr[i],ca[i],parity,graph); break;
            }
            for (size_t i=0;i<nr.size();++i) if (std::memcmp(&nr[i],&na[i],4)) {
                std::fprintf(stderr,"norm at %zu reference %.9g rows %.9g parity %d graph %d\n",i,nr[i],na[i],parity,graph); break;
            }
            throw std::runtime_error("GDN row bits changed C="+std::to_string(C)+" nv="+std::to_string(nv));
        }
        for (int i=0;i<pad;++i)
            if (ca[i]!=guard || ca[ca.size()-1-i]!=guard || na[i]!=guard || na[na.size()-1-i]!=guard)
                throw std::runtime_error("GDN row guard overwritten");
    }
    if (C >= 1024 && nv == 24 && parity == 1) {
        cudaEvent_t a,b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
        for (bool norm_only : {false,true}) {
            float ms[2] = {};
            for (int round=0;round<3;++round) for (int i=0;i<2;++i) {
                const int which=(round+i)%2;
                auto call = [&] { if (norm_only) norm_call(which); else conv_call(which); };
                for (int j=0;j<3;++j) call();
                check(cudaEventRecord(a,stream));
                for (int j=0;j<10;++j) call();
                check(cudaEventRecord(b,stream)); check(cudaEventSynchronize(b));
                float dt; check(cudaEventElapsedTime(&dt,a,b)); ms[which]+=dt/30;
            }
            std::printf("GDN %s C=%d nv=%d: block %.3f ms rows %.3f ms\n",norm_only?"norm":"conv",C,nv,ms[0],ms[1]);
        }
        check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
    }
    return 4;
}
}

int main() {
    try {
        int count=0;
        for (int dev=0;dev<2;++dev) {
            fnx::check(cudaSetDevice(dev));
            fnx::check(cudaFuncSetAttribute(fnx::k_gdn_ch_rec<fnx::GCB,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)fnx::GCH_REC_SMEM));
            fnx::check(cudaFuncSetAttribute(fnx::k_gdn_ch_rec<32,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)fnx::GCH_REC_WIDE_SMEM));
            cudaStream_t stream;fnx::check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for (int C:{1,3,16,31,32,33,63,127,256,1024,4096,8192})
                for (int nv:{1,4,24}) for (int seed:{0,1}) count+=fnx::run_gdn_case(C,nv,seed,stream);
            for (int C:{1,2,3,17,31,32,33,127,1024,4096})
                for (int nv:{1,7,24}) for (int parity:{0,1}) count+=fnx::run_row_case(C,nv,parity,stream);
            fnx::check(cudaStreamDestroy(stream));
        }
        std::cout << "PASS " << count << " exact GDN output, carried-state and guard checks on two cards\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

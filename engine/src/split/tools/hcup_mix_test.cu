#include "kernels.hpp"
#include "wmma_gemm.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace fnx {
namespace {
#include "hc_prefill.cuh"
}
void check_up(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct UpMemory {
    std::vector<void*> ptrs;
    template<class T>T* get(size_t n){T* p;check_up(cudaMalloc((void**)&p,n*sizeof(T)));ptrs.push_back(p);return p;}
    ~UpMemory(){for(void* p:ptrs)(void)cudaFree(p);}
};
void run_up_case(int C,int seed,cudaStream_t stream,bool quantized=false){
    constexpr int pad=32;
    std::mt19937 random(813+C+seed);std::uniform_real_distribution<float> random_value(-1.f,1.f);
    std::vector<__half> x((size_t)C*LR),w((size_t)D*LR);
    std::vector<int8_t> q;
    std::vector<uint16_t> scales;
    std::vector<float> residual((size_t)C*D),norm(D),rs((size_t)C*HC);
    for(auto& v:x)v=__float2half(random_value(random));
    for(auto& v:w)v=__float2half(random_value(random)*.125f);
    if(quantized){
        q.resize(w.size());scales.resize(w.size()/32);
        const uint16_t edges[]={0,1,0x03ff,0x0400,0x0800,0x1000,0x3000,0x3c00};
        for(size_t i=0;i<scales.size();++i)
            scales[i]=seed==2?edges[i%8]:__half_as_ushort(__float2half((random()%1001)*.00001f));
        for(size_t i=0;i<w.size();++i){
            q[i]=(int8_t)(random()&255);
            w[i]=__float2half(__half2float(__ushort_as_half(scales[i/32]))*(float)q[i]);
        }
    }
    for(auto& v:residual)v=seed==0?0.f:random_value(random)*(seed==2?64.f:1.f);
    if(seed==3)for(size_t i=0;i<residual.size();++i)if((i%D)/N!=0)residual[i]=0.f;
    for(auto& v:norm)v=random_value(random);
    for(auto& v:rs)v=1.f+random_value(random)*.5f;
    UpMemory mem;
    auto* dx=mem.get<uint16_t>(x.size());auto* dw=mem.get<uint16_t>(w.size());
    auto* dq=quantized?mem.get<int8_t>(q.size()):nullptr;
    auto* ds=quantized?mem.get<uint16_t>(scales.size()):nullptr;
    auto* r=mem.get<float>(residual.size());auto* wn=mem.get<float>(norm.size());auto* rms=mem.get<float>(rs.size());
    auto* up=mem.get<float>((size_t)C*D);
    const size_t count=(size_t)C*N+2*pad;
    float* y[2]={mem.get<float>(count),mem.get<float>(count)};
    uint16_t* h[2]={mem.get<uint16_t>(count),mem.get<uint16_t>(count)};
    auto copy=[&](void* dst,const void* src,size_t n){check_up(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,stream));};
    copy(dx,x.data(),x.size()*2);copy(dw,w.data(),w.size()*2);copy(r,residual.data(),residual.size()*4);
    copy(wn,norm.data(),norm.size()*4);copy(rms,rs.data(),rs.size()*4);
    if(quantized){copy(dq,q.data(),q.size());copy(ds,scales.data(),scales.size()*2);}
    for(int i=0;i<2;++i){check_up(cudaMemsetAsync(y[i],0xA5,count*4,stream));check_up(cudaMemsetAsync(h[i],0xA5,count*2,stream));}
    auto call=[&](int i){
        if(i==0){
            if(quantized&&C>=3072){
                if(!strata_pf_hcup_mix_f16(dx,dw,r,wn,rms,y[0]+pad,h[0]+pad,C,D,LR,stream))throw std::runtime_error("reference fused dispatch");
                return;
            }
            const bool q8_product=quantized&&strata_pf_q8_k320_f16(dx,dq,ds,up,C,D,LR,D,stream);
            if(!q8_product&&!strata_pf_gemm_f16(dx,dw,up,C,D,LR,D,0,stream))throw std::runtime_error("reference GEMM dispatch");
            k_hcb_mixinj<<<C,HCM_THREADS,0,stream>>>(r,wn,rms,up,y[0]+pad,(__half*)(h[0]+pad),nullptr,nullptr);
        }else{
            const bool launched=quantized
                ?strata_pf_hcup_q8_f16(dx,dq,ds,r,wn,rms,y[1]+pad,h[1]+pad,C,D,LR,stream)
                :strata_pf_hcup_mix_f16(dx,dw,r,wn,rms,y[1]+pad,h[1]+pad,C,D,LR,stream);
            if(!launched)throw std::runtime_error("fused HC up dispatch");
        }
    };
    call(0);call(1);check_up(cudaGetLastError());check_up(cudaStreamSynchronize(stream));
    if(quantized){
        cudaGraph_t graph;cudaGraphExec_t exec;
        check_up(cudaMemsetAsync(y[1],0xA5,count*4,stream));check_up(cudaMemsetAsync(h[1],0xA5,count*2,stream));
        check_up(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));call(1);check_up(cudaStreamEndCapture(stream,&graph));
        check_up(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int repeat=0;repeat<3;++repeat)check_up(cudaGraphLaunch(exec,stream));
        check_up(cudaStreamSynchronize(stream));check_up(cudaGraphExecDestroy(exec));check_up(cudaGraphDestroy(graph));
    }
    std::vector<float> expected(count),actual(count);std::vector<uint16_t> eh(count),ah(count);
    check_up(cudaMemcpy(expected.data(),y[0],count*4,cudaMemcpyDeviceToHost));check_up(cudaMemcpy(actual.data(),y[1],count*4,cudaMemcpyDeviceToHost));
    check_up(cudaMemcpy(eh.data(),h[0],count*2,cudaMemcpyDeviceToHost));check_up(cudaMemcpy(ah.data(),h[1],count*2,cudaMemcpyDeviceToHost));
    if(std::memcmp(expected.data(),actual.data(),count*4)||eh!=ah){
        size_t differences=0;float worst=0;
        for(size_t i=pad;i<count-pad;++i)if(std::memcmp(&actual[i],&expected[i],4)){++differences;worst=std::max(worst,std::abs(actual[i]-expected[i]));}
        std::fprintf(stderr,"HC up C=%d seed=%d Q8=%d differences=%zu max_abs=%.9g\n",C,seed,quantized,differences,worst);
        throw std::runtime_error("fused HC up differs from native GEMM + original mixer");
    }
    for(size_t i=pad;i<count-pad;++i)if(!std::isfinite(actual[i]))throw std::runtime_error("nonfinite mixed output");
    for(int i=0;i<pad;++i){
        unsigned a,b;std::memcpy(&a,&actual[i],4);std::memcpy(&b,&actual[count-pad+i],4);
        if(a!=0xA5A5A5A5u||b!=0xA5A5A5A5u||ah[i]!=0xA5A5u||ah[count-pad+i]!=0xA5A5u)throw std::runtime_error("mixed output guards");
    }
    if(seed==1&&!quantized){
        cudaEvent_t a,b;check_up(cudaEventCreate(&a));check_up(cudaEventCreate(&b));float ms[2];
        for(int i=0;i<2;++i){check_up(cudaEventRecord(a,stream));for(int n=0;n<5;++n)call(i);check_up(cudaEventRecord(b,stream));check_up(cudaEventSynchronize(b));check_up(cudaEventElapsedTime(&ms[i],a,b));}
        std::printf("HC up C=%d: separate %.3f ms fused %.3f ms\n",C,ms[0]/5,ms[1]/5);
        check_up(cudaEventDestroy(a));check_up(cudaEventDestroy(b));
    }
}
void check_q8_rejections(cudaStream_t stream){
    UpMemory mem;auto* p=mem.get<uint16_t>(32);
    check_up(cudaMemsetAsync(p,0xA5,64,stream));
    for(int invalid=0;invalid<14;++invalid){
        const uint16_t* x=p;const int8_t* q=(const int8_t*)p;const uint16_t* d=p;
        const float *r=(const float*)p,*norm=r,*rms=r;float* y=(float*)p;uint16_t* h=p;
        int64_t rows=64,n=10240,k=320;
        switch(invalid){
            case 0:rows=63;break;case 1:rows=(1LL<<28)+1;break;
            case 2:n=2560;break;case 3:k=256;break;
            case 4:x=nullptr;break;case 5:q=nullptr;break;case 6:d=nullptr;break;
            case 7:r=nullptr;break;case 8:norm=nullptr;break;case 9:rms=nullptr;break;
            case 10:y=nullptr;break;case 11:h=nullptr;break;
            case 12:x=p+1;break;case 13:q=(const int8_t*)p+1;break;
        }
        if(strata_pf_hcup_q8_f16(x,q,d,r,norm,rms,y,h,rows,n,k,stream))throw std::runtime_error("unsupported Q8 HC up dispatch");
    }
    check_up(cudaStreamSynchronize(stream));uint16_t actual[32];
    check_up(cudaMemcpy(actual,p,sizeof(actual),cudaMemcpyDeviceToHost));
    for(auto v:actual)if(v!=0xA5A5u)throw std::runtime_error("rejected Q8 HC up touched output");
}
}
int main(){
    try{
        setenv("STRATA_PF_HC_UP_REGISTER","1",1);setenv("STRATA_PF_GEMM96","1",1);
        int count=0,quantized=0;
        for(int dev=0;dev<2;++dev){
            fnx::check_up(cudaSetDevice(dev));cudaStream_t stream;fnx::check_up(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for(int C:{64,65,129,511,1024,2048,3072,4014,4096,8192})for(int seed:{0,3,1,2}){fnx::run_up_case(C,seed,stream);++count;}
            for(int C:{64,65,96,128,255,256,257,511,512,1024,2048,3072,4095,4096})
                for(int seed:{0,3,1,2}){fnx::run_up_case(C,seed,stream,true);++quantized;}
            fnx::check_q8_rejections(stream);
            fnx::check_up(cudaStreamDestroy(stream));
        }
        std::printf("PASS %d FP16 and %d Q8 exact fused HC up cases, Q8 graph replay and 28 rejection checks on two cards\n",count,quantized);
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}

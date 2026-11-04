#include "kernels.hpp"
#include "wmma_gemm.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
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
void run_up_case(int C,int seed,cudaStream_t stream){
    constexpr int pad=32;
    std::mt19937 random(813+C+seed);std::uniform_real_distribution<float> random_value(-1.f,1.f);
    std::vector<__half> x((size_t)C*LR),w((size_t)D*LR);
    std::vector<float> residual((size_t)C*D),norm(D),rs((size_t)C*HC);
    for(auto& v:x)v=__float2half(random_value(random));
    for(auto& v:w)v=__float2half(random_value(random)*.125f);
    for(auto& v:residual)v=seed==0?0.f:random_value(random)*(seed==2?64.f:1.f);
    if(seed==3)for(size_t i=0;i<residual.size();++i)if((i%D)/N!=0)residual[i]=0.f;
    for(auto& v:norm)v=random_value(random);
    for(auto& v:rs)v=1.f+random_value(random)*.5f;
    UpMemory mem;
    auto* dx=mem.get<uint16_t>(x.size());auto* dw=mem.get<uint16_t>(w.size());
    auto* r=mem.get<float>(residual.size());auto* wn=mem.get<float>(norm.size());auto* rms=mem.get<float>(rs.size());
    auto* up=mem.get<float>((size_t)C*D);
    const size_t count=(size_t)C*N+2*pad;
    float* y[2]={mem.get<float>(count),mem.get<float>(count)};
    uint16_t* h[2]={mem.get<uint16_t>(count),mem.get<uint16_t>(count)};
    auto copy=[&](void* dst,const void* src,size_t n){check_up(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,stream));};
    copy(dx,x.data(),x.size()*2);copy(dw,w.data(),w.size()*2);copy(r,residual.data(),residual.size()*4);
    copy(wn,norm.data(),norm.size()*4);copy(rms,rs.data(),rs.size()*4);
    for(int i=0;i<2;++i){check_up(cudaMemsetAsync(y[i],0xA5,count*4,stream));check_up(cudaMemsetAsync(h[i],0xA5,count*2,stream));}
    auto call=[&](int i){
        if(i==0){
            if(!strata_pf_gemm_f16(dx,dw,up,C,D,LR,D,0,stream))throw std::runtime_error("reference GEMM dispatch");
            k_hcb_mixinj<<<C,HCM_THREADS,0,stream>>>(r,wn,rms,up,y[0]+pad,(__half*)(h[0]+pad),nullptr,nullptr);
        }else if(!strata_pf_hcup_mix_f16(dx,dw,r,wn,rms,y[1]+pad,h[1]+pad,C,D,LR,stream))throw std::runtime_error("fused HC up dispatch");
    };
    call(0);call(1);check_up(cudaGetLastError());check_up(cudaStreamSynchronize(stream));
    std::vector<float> expected(count),actual(count);std::vector<uint16_t> eh(count),ah(count);
    check_up(cudaMemcpy(expected.data(),y[0],count*4,cudaMemcpyDeviceToHost));check_up(cudaMemcpy(actual.data(),y[1],count*4,cudaMemcpyDeviceToHost));
    check_up(cudaMemcpy(eh.data(),h[0],count*2,cudaMemcpyDeviceToHost));check_up(cudaMemcpy(ah.data(),h[1],count*2,cudaMemcpyDeviceToHost));
    if(std::memcmp(expected.data(),actual.data(),count*4)||eh!=ah){
        size_t differences=0;float worst=0;
        for(size_t i=pad;i<count-pad;++i)if(std::memcmp(&actual[i],&expected[i],4)){++differences;worst=std::max(worst,std::abs(actual[i]-expected[i]));}
        std::fprintf(stderr,"HC up C=%d seed=%d differences=%zu max_abs=%.9g\n",C,seed,differences,worst);
        throw std::runtime_error("fused HC up differs from native GEMM + original mixer");
    }
    for(size_t i=pad;i<count-pad;++i)if(!std::isfinite(actual[i]))throw std::runtime_error("nonfinite mixed output");
    for(int i=0;i<pad;++i){
        unsigned a,b;std::memcpy(&a,&actual[i],4);std::memcpy(&b,&actual[count-pad+i],4);
        if(a!=0xA5A5A5A5u||b!=0xA5A5A5A5u||ah[i]!=0xA5A5u||ah[count-pad+i]!=0xA5A5u)throw std::runtime_error("mixed output guards");
    }
    if(seed==1){
        cudaEvent_t a,b;check_up(cudaEventCreate(&a));check_up(cudaEventCreate(&b));float ms[2];
        for(int i=0;i<2;++i){check_up(cudaEventRecord(a,stream));for(int n=0;n<5;++n)call(i);check_up(cudaEventRecord(b,stream));check_up(cudaEventSynchronize(b));check_up(cudaEventElapsedTime(&ms[i],a,b));}
        std::printf("HC up C=%d: separate %.3f ms fused %.3f ms\n",C,ms[0]/5,ms[1]/5);
        check_up(cudaEventDestroy(a));check_up(cudaEventDestroy(b));
    }
}
}
int main(){
    try{
        int count=0;
        for(int dev=0;dev<2;++dev){
            fnx::check_up(cudaSetDevice(dev));cudaStream_t stream;fnx::check_up(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for(int C:{64,65,129,511,1024,2048,3072,4014,4096,8192})for(int seed:{0,3,1,2}){fnx::run_up_case(C,seed,stream);++count;}
            fnx::check_up(cudaStreamDestroy(stream));
        }
        std::printf("PASS %d exact fused HC up, FP16 activation and guard cases on two cards\n",count);
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}

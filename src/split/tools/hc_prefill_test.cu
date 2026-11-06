#include "kernels.hpp"
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
void check_hc(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct HcMemory {
    std::vector<void*> pointers;
    template<class T> T* get(size_t count) {
        T* p;check_hc(cudaMalloc((void**)&p,count*sizeof(T)));pointers.push_back(p);return p;
    }
    ~HcMemory(){for(void* p:pointers)(void)cudaFree(p);}
};
void run_hc_case(int C, bool residual, bool inject, int seed, cudaStream_t stream) {
    constexpr int guard=32;
    std::mt19937 random(123+seed+C);
    std::uniform_real_distribution<float> values(-1.f,1.f);
    std::vector<float> R((size_t)C*D+2*guard,-9871.5f),bo((size_t)C*N),inj((size_t)C*HC),norm(D);
    std::vector<uint16_t> wi((size_t)HC*D);
    for(size_t i=guard;i<R.size()-guard;++i) R[i]=seed==0?0.f:values(random)*(seed==2?64.f:1.f);
    for(auto& v:bo)v=values(random);
    for(auto& v:inj)v=values(random)*8.f;
    for(auto& v:norm)v=values(random);
    for(auto& v:wi){float f=values(random);unsigned bits;std::memcpy(&bits,&f,4);v=(uint16_t)(bits>>16);}
    HcMemory mem;
    float *r[2]={mem.get<float>(R.size()),mem.get<float>(R.size())};
    float *rs[2]={mem.get<float>((size_t)C*HC+2*guard),mem.get<float>((size_t)C*HC+2*guard)};
    float *out[2]={mem.get<float>((size_t)C*HC+2*guard),mem.get<float>((size_t)C*HC+2*guard)};
    __half *xh[2]={mem.get<__half>((size_t)C*D+2*guard),mem.get<__half>((size_t)C*D+2*guard)};
    auto *b=mem.get<float>(bo.size()),*in=mem.get<float>(inj.size()),*wn=mem.get<float>(norm.size());
    auto *w=mem.get<uint16_t>(wi.size());
    auto copy=[&](void* dst,const void* src,size_t bytes){check_hc(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream));};
    copy(b,bo.data(),bo.size()*4);copy(in,inj.data(),inj.size()*4);copy(wn,norm.data(),norm.size()*4);copy(w,wi.data(),wi.size()*2);
    for(int i=0;i<2;++i){
        copy(r[i],R.data(),R.size()*4);
        check_hc(cudaMemsetAsync(rs[i],0xA5,((size_t)C*HC+2*guard)*4,stream));
        check_hc(cudaMemsetAsync(out[i],0xA5,((size_t)C*HC+2*guard)*4,stream));
        check_hc(cudaMemsetAsync(xh[i],0xA5,((size_t)C*D+2*guard)*2,stream));
    }
    auto call=[&](int i){
        if(i==0) k_hcb_prep<<<C,HCB_THREADS,0,stream>>>(r[i]+guard,residual?b:nullptr,in,wn,rs[i]+guard,xh[i]+guard,1e-6f,inject?w:nullptr,out[i]+guard);
        else k_hcb_prep_streams<<<C,HCB_THREADS,0,stream>>>(r[i]+guard,residual?b:nullptr,in,wn,rs[i]+guard,xh[i]+guard,1e-6f,inject?w:nullptr,out[i]+guard);
    };
    call(0);call(1);check_hc(cudaGetLastError());check_hc(cudaStreamSynchronize(stream));
    auto equal=[&](const void* a,const void* b,size_t bytes,const char* what){
        std::vector<unsigned char> x(bytes),y(bytes);
        check_hc(cudaMemcpy(x.data(),a,bytes,cudaMemcpyDeviceToHost));check_hc(cudaMemcpy(y.data(),b,bytes,cudaMemcpyDeviceToHost));
        if(x!=y)throw std::runtime_error(std::string("HC mismatch ")+what+" C="+std::to_string(C)+" seed="+std::to_string(seed));
    };
    equal(r[0],r[1],R.size()*4,"residual/guards");
    equal(rs[0],rs[1],((size_t)C*HC+2*guard)*4,"norm/guards");
    equal(xh[0],xh[1],((size_t)C*D+2*guard)*2,"activation/guards");
    equal(out[0],out[1],((size_t)C*HC+2*guard)*4,"inject/guards");
    if(C>=1024 && residual && inject && seed==1){
        cudaEvent_t start,end;check_hc(cudaEventCreate(&start));check_hc(cudaEventCreate(&end));float ms[2];
        for(int i=0;i<2;++i){
            check_hc(cudaEventRecord(start,stream));for(int j=0;j<5;++j)call(i);
            check_hc(cudaEventRecord(end,stream));check_hc(cudaEventSynchronize(end));check_hc(cudaEventElapsedTime(&ms[i],start,end));
        }
        std::printf("HC C=%d: reference %.3f ms stream norms %.3f ms\n",C,ms[0]/5,ms[1]/5);
        check_hc(cudaEventDestroy(start));check_hc(cudaEventDestroy(end));
    }
}
}
int main(){
    try{
        int count=0;
        for(int dev=0;dev<2;++dev){
            fnx::check_hc(cudaSetDevice(dev));cudaStream_t stream;fnx::check_hc(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for(int C:{1,3,17,64,256,1024,4096})for(bool residual:{false,true})for(bool inject:{false,true})for(int seed:{0,1,2}){
                fnx::run_hc_case(C,residual,inject,seed,stream);++count;
            }
            fnx::check_hc(cudaStreamDestroy(stream));
        }
        std::printf("PASS %d exact HC residual, norm, activation, inject and guard cases on two cards\n",count);
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}

// Independent dispatch of the existing wide kernel and both 96-column schedules.
#include "../../prefill/wmma_gemm.cu"
namespace pfg {
#include "prefill_skinny_probes.cuh"
}
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static void check(hipError_t e) {
    if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));
}
struct Memory {
    std::vector<void*> owned;
    template<class T> T* get(size_t n) {
        T* p;check(hipMalloc((void**)&p,n*sizeof(T)));owned.push_back(p);return p;
    }
    ~Memory(){for(void* p:owned)(void)hipFree(p);}
};
struct Shape {int m,n,k;};

static int run(Shape shape,int seed,int padding,bool micro,hipStream_t stream) {
    const int M=shape.m,N=shape.n,K=shape.k,ld=K+padding,ldy=N+13;
    constexpr int tail=32;
    constexpr float guard=-8891.25f;
    std::mt19937 rng(1927+M+N+seed);
    std::uniform_real_distribution<float> random(-1.f,1.f);
    std::vector<_Float16> x((size_t)M*ld),w((size_t)N*ld);
    auto value=[&] {return (_Float16)(seed==2?64.f:seed==0?(int(rng()%17)-8)/16.f:random(rng));};
    for(int r=0;r<M;++r)for(int c=0;c<K;++c)x[(size_t)r*ld+c]=value();
    for(int r=0;r<N;++r)for(int c=0;c<K;++c)w[(size_t)r*ld+c]=value();
    Memory memory;auto* dx=memory.get<_Float16>(x.size());auto* dw=memory.get<_Float16>(w.size());
    std::vector<float> initial((size_t)M*ldy+tail,guard);
    for(int r=0;r<M;++r)for(int c=0;c<N;++c)initial[(size_t)r*ldy+c]=.125f*(int((r+c)%17)-8);
    float* y[5]={memory.get<float>(initial.size()),memory.get<float>(initial.size()),memory.get<float>(initial.size()),memory.get<float>(initial.size()),memory.get<float>(initial.size())};
    check(hipMemcpyAsync(dx,x.data(),x.size()*2,hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(dw,w.data(),w.size()*2,hipMemcpyHostToDevice,stream));
    auto launch=[&](int mode,int accumulate) {
        const unsigned mt=(M+127)/128;
        if(mode==0) {
            if(N<512 && accumulate==0)pfg::kernel_narrow_bk64<<<mt*((N+127)/128),256,0,stream>>>(dx,ld,dw,ld,y[mode],M,N,K,ldy);
            else pfg::kernel_bk64<<<mt*((N+255)/256),256,0,stream>>>(dx,dw,y[mode],M,N,K,ldy,accumulate,ld,ld);
        }
        if(mode==1)pfg::kernel_bk64_n96<false><<<mt*((N+95)/96),128,0,stream>>>(dx,dw,y[mode],M,N,K,ldy,accumulate,ld,ld);
        if(mode==2)pfg::kernel_bk64_n96<true><<<mt*((N+95)/96),128,0,stream>>>(dx,dw,y[mode],M,N,K,ldy,accumulate,ld,ld);
        if(mode==3)pfg::kernel_narrow96<<<((M+95)/96)*((N+95)/96),128,0,stream>>>(dx,dw,y[mode],M,N,K,ldy,accumulate,ld,ld);
        if(mode==4)pfg::kernel_hcdown320<<<((M+63)/64)*((N+319)/320),320,0,stream>>>(dx,dw,y[mode],M,N,K,ldy,accumulate,ld,ld);
        check(hipGetLastError());
    };
    std::vector<float> reference(initial.size()),actual(initial.size());
    int comparisons=0;
    for(int accumulate:{0,1})for(bool graph:{false,true}) {
        for(int mode=0;mode<5;++mode) {
            check(hipMemcpyAsync(y[mode],initial.data(),initial.size()*4,hipMemcpyHostToDevice,stream));
            if(graph) {
                hipGraph_t g;hipGraphExec_t ex;
                check(hipStreamBeginCapture(stream,hipStreamCaptureModeRelaxed));launch(mode,accumulate);
                check(hipStreamEndCapture(stream,&g));check(hipGraphInstantiate(&ex,g,nullptr,nullptr,0));
                for(int i=0;i<3;++i)check(hipGraphLaunch(ex,stream));
                check(hipStreamSynchronize(stream));check(hipGraphExecDestroy(ex));check(hipGraphDestroy(g));
            } else launch(mode,accumulate);
            check(hipStreamSynchronize(stream));
            if(mode==0)check(hipMemcpy(reference.data(),y[mode],reference.size()*4,hipMemcpyDeviceToHost));
            else {
                check(hipMemcpy(actual.data(),y[mode],actual.size()*4,hipMemcpyDeviceToHost));
                if(std::memcmp(reference.data(),actual.data(),actual.size()*4)) {
                    std::fprintf(stderr,"changed bits M=%d N=%d K=%d mode=%d seed=%d padding=%d beta=%d graph=%d\n",
                        M,N,K,mode,seed,padding,accumulate,graph);
                    throw std::runtime_error("GEMM schedule mismatch");
                }
                ++comparisons;
            }
        }
        for(int r=0;r<M;++r)for(int c=N;c<ldy;++c)
            if(reference[(size_t)r*ldy+c]!=guard)throw std::runtime_error("row guard overwritten");
        for(size_t i=(size_t)M*ldy;i<reference.size();++i)
            if(reference[i]!=guard)throw std::runtime_error("end guard overwritten");
    }
    if(micro) {
        hipEvent_t a,b;check(hipEventCreate(&a));check(hipEventCreate(&b));float ms[5]={};
        for(int round=0;round<5;++round)for(int i=0;i<5;++i) {
            const int mode=(i+round)%5;
            for(int j=0;j<3;++j)launch(mode,0);
            check(hipEventRecord(a,stream));
            for(int j=0;j<10;++j)launch(mode,0);
            check(hipEventRecord(b,stream));check(hipEventSynchronize(b));
            float elapsed;check(hipEventElapsedTime(&elapsed,a,b));ms[mode]+=elapsed/50;
        }
        std::printf("micro M=%d N=%d K=%d old_ms=%.6f n96_ms=%.6f interleave_ms=%.6f narrow96_ms=%.6f full320_ms=%.6f\n",M,N,K,ms[0],ms[1],ms[2],ms[3],ms[4]);
        check(hipEventDestroy(a));check(hipEventDestroy(b));
    }
    return comparisons;
}

int main(){try {
    setvbuf(stdout,nullptr,_IOLBF,0);int cases=0;
    for(int dev=0;dev<2;++dev) {
        check(hipSetDevice(dev));
        if(!strata_pf_padded_gemm_available())throw std::runtime_error("gfx11 required");
        const void* kernels[]={reinterpret_cast<const void*>(pfg::kernel_bk64),
            reinterpret_cast<const void*>(pfg::kernel_bk64_n96<false>),reinterpret_cast<const void*>(pfg::kernel_bk64_n96<true>),
            reinterpret_cast<const void*>(pfg::kernel_narrow96),reinterpret_cast<const void*>(pfg::kernel_hcdown320)};
        for(int mode=0;mode<5;++mode) {
            hipFuncAttributes a{};check(hipFuncGetAttributes(&a,kernels[mode]));
            std::printf("card %d mode %d registers=%d shared=%zu local=%zu\n",dev,mode,a.numRegs,a.sharedSizeBytes,a.localSizeBytes);
        }
        hipStream_t stream;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        const Shape edges[]={{64,1024,64},{65,1031,192},{129,1664,2560},{257,2560,320},{65,319,64},{129,320,192}};
        for(auto shape:edges)for(int seed:{0,1,2})for(int pad:{0,64})cases+=run(shape,seed,pad,false,stream);
        const Shape large[]={{4096,2560,3072},{4096,6528,2560},{4096,1664,2560},{16384,2560,2560},
            {2048,320,10240},{4096,320,10240},{8192,320,10240}};
        for(auto shape:large)cases+=run(shape,1,0,true,stream);
        check(hipStreamDestroy(stream));std::printf("card %d exact schedules PASS\n",dev);
    }
    std::printf("PASS %d exact schedule, FP32 range, padding, beta and graph comparisons\n",cases);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

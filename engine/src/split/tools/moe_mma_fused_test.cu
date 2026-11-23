// Independent wide GEMM + SwiGLU versus a fused, identically ordered epilogue.
#include "../src/moe_mma.cu"
#include "moe_mma_fused.hpp"
#include <cstring>
#include <iostream>

namespace fnx {
void moe_gate_up_mma_wide(const DW&,size_t,const BigJob*,const int*,int,const uint16_t*,float*,void*);
static void ck(cudaError_t e,const char* what) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
struct Format {int type,block,bytes;const char* name;};
static int run(const Format& fmt,int cols,int ffs,bool micro) {
    const int capacity=micro?512:3, experts=micro?256:3, rows=2*ffs;
    const size_t rb=(size_t)cols/fmt.block*fmt.bytes, stride=rb*rows;
    const size_t pairs=(size_t)capacity*PF_JOB, output_bytes=pairs*ffs*2, guard=256;
    std::vector<void*> allocations;
    auto alloc=[&](size_t bytes){void* p;ck(cudaMalloc(&p,bytes),"alloc");allocations.push_back(p);return p;};
    auto* weights=(char*)alloc(experts*stride);
    auto* input=(uint16_t*)alloc(pairs*cols*2);
    auto* jobs=(BigJob*)alloc(capacity*sizeof(BigJob));auto* count=(int*)alloc(sizeof(int));
    auto* gu=(float*)alloc(pairs*rows*4);
    auto* ref=(uint16_t*)alloc(output_bytes);
    auto* result=(unsigned char*)alloc(output_bytes+2*guard);
    auto* ep=(const char**)alloc(experts*sizeof(char*));
    std::vector<const char*> pointers(experts);
    for(int e=0;e<experts;++e)pointers[e]=weights+(size_t)((e*5)%experts)*stride;
    ck(cudaMemcpy(ep,pointers.data(),experts*sizeof(char*),cudaMemcpyHostToDevice),"ep");
    DW w{};w.p=weights;w.ep=ep;w.type=fmt.type;w.n_out=rows;w.n_in=cols;w.row_bytes=rb;
    std::vector<unsigned char> raw(experts*stride), expected(output_bytes+2*guard), got(expected.size()), tmp(output_bytes);
    std::vector<__half> x(pairs*cols);std::vector<BigJob> hj(capacity);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
    auto launch=[&](bool fused){
        if(fused) {
            if(!moe_gate_up_swiglu_mma(w,stride,jobs,count,capacity,input,(uint16_t*)(result+guard),st))
                throw std::runtime_error("unexpected fused fallback");
        } else {
            if(fmt.type==T_IQ4_XS) moe_gate_up_mma(w,stride,jobs,count,capacity,input,gu,st);
            else moe_gate_up_mma_wide(w,stride,jobs,count,capacity,input,gu,st);
            swiglu_h16(gu,ffs,(int)pairs,ref,st);
        }
    };
    const std::vector<int> seeds=micro?std::vector<int>{1}:std::vector<int>{0,1};
    const std::vector<int> counts=micro?std::vector<int>{capacity}:std::vector<int>{0,1,3};
    std::vector<int> sizes;
    if(micro) {
        sizes={PF_JOB};
    } else {
        for(int len:{0,1,15,16,17,31,32,33,63,64,65,PF_JOB-1,PF_JOB}) {
            if(len<=PF_JOB && std::find(sizes.begin(),sizes.end(),len)==sizes.end()) sizes.push_back(len);
        }
    }
    int checked=0;
    for(int seed:seeds) {
        for(size_t i=0;i<raw.size();++i)raw[i]=(i*13+(i>>7)*17+seed*23)&255;
        for(size_t i=0;i<raw.size();i+=fmt.bytes) {
            const uint16_t edge[]={0,1,3,0x200,0x3ff,0x400,0x1400,0x2400,0x3c00,0x5400,0x5800};
            const uint16_t magnitude=micro?(uint16_t)(0x2400+(i/fmt.bytes)%16*19):edge[(i/fmt.bytes)%11];
            const uint16_t d=magnitude|((i/fmt.bytes)%7==0?0x8000:0);
            std::memcpy(raw.data()+i,&d,2);
        }
        for(size_t i=0;i<x.size();++i)x[i]=__float2half(seed?.013f*((int)((i*7+(i>>6)*11)%511)-255):0.f);
        ck(cudaMemcpyAsync(weights,raw.data(),raw.size(),cudaMemcpyHostToDevice,st),"weights");
        ck(cudaMemcpyAsync(input,x.data(),x.size()*2,cudaMemcpyHostToDevice,st),"input");
        // Exercise both contiguous expert storage and shuffled pointer tables.
        w.ep=seed?ep:nullptr;
        for(int nj:counts)for(int n:sizes) {
            for(int j=0;j<capacity;++j) {
                hj[j].w=(j+seed)%experts;hj[j].n=j%2?std::min(PF_JOB,n+1):n;
                for(int k=0;k<hj[j].n;++k) {
                    hj[j].y[k]=(j*13+k*7)%(capacity*PF_JOB);
                    hj[j].out[k]=j*PF_JOB+k;
                }
            }
            ck(cudaMemcpyAsync(jobs,hj.data(),capacity*sizeof(BigJob),cudaMemcpyHostToDevice,st),"jobs");
            ck(cudaMemcpyAsync(count,&nj,sizeof(nj),cudaMemcpyHostToDevice,st),"count");
            ck(cudaMemsetAsync(gu,0,pairs*rows*4,st),"reference init");
            launch(false);ck(cudaStreamSynchronize(st),"reference");
            ck(cudaMemcpy(tmp.data(),ref,output_bytes,cudaMemcpyDeviceToHost),"read reference");
            std::fill(expected.begin(),expected.end(),0xB3);
            for(int j=0;j<nj;++j)for(int k=0;k<hj[j].n;++k) {
                const size_t off=(size_t)hj[j].out[k]*ffs*2;
                std::memcpy(expected.data()+guard+off,tmp.data()+off,ffs*2);
            }
            for(bool graph:{false,true}) {
                ck(cudaMemsetAsync(result,0xB3,got.size(),st),"guards");
                if(graph) {
                    cudaGraph_t g;cudaGraphExec_t ex;
                    ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");launch(true);
                    ck(cudaStreamEndCapture(st,&g),"capture end");
                    ck(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0),"graph instantiate");
                    for(int i=0;i<3;++i)ck(cudaGraphLaunch(ex,st),"replay");
                    ck(cudaStreamSynchronize(st),"graph done");
                    ck(cudaGraphExecDestroy(ex),"graph free");ck(cudaGraphDestroy(g),"graph free");
                } else launch(true);
                ck(cudaStreamSynchronize(st),"fused done");
                ck(cudaMemcpy(got.data(),result,got.size(),cudaMemcpyDeviceToHost),"read fused");
                if(got!=expected) {
                    size_t at=0;while(at<got.size()&&got[at]==expected[at])++at;
                    throw std::runtime_error(std::string("fused mismatch ")+fmt.name+" K="+std::to_string(cols)+
                        " FF="+std::to_string(ffs)+" n="+std::to_string(n)+" nj="+std::to_string(nj)+
                        " seed="+std::to_string(seed)+" byte="+std::to_string(at));
                }
                ++checked;
            }
        }
    }
    if(micro) {
        cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");float ms[2]={};
        for(int round=0;round<3;++round)for(int i=0;i<2;++i) {
            const bool fused=(i+round)%2;
            for(int j=0;j<3;++j)launch(fused);
            ck(cudaEventRecord(a,st),"start");for(int j=0;j<8;++j)launch(fused);
            ck(cudaEventRecord(b,st),"end");ck(cudaEventSynchronize(b),"timed");float elapsed;
            ck(cudaEventElapsedTime(&elapsed,a,b),"elapsed");ms[fused]+=elapsed/24;
        }
        std::cout<<"micro "<<fmt.name<<" K="<<cols<<" FF="<<ffs<<" jobs="<<capacity
                 <<" separate_ms="<<ms[0]<<" fused_ms="<<ms[1]<<'\n';
        ck(cudaEventDestroy(a),"free event");ck(cudaEventDestroy(b),"free event");
    }
    DW unsupported=w;unsupported.n_out=rows-1;
    if(moe_gate_up_swiglu_mma(unsupported,stride,jobs,count,capacity,input,(uint16_t*)(result+guard),st))
        throw std::runtime_error("missing shape fallback");
    unsupported=w;unsupported.type=T_Q8_0;
    if(moe_gate_up_swiglu_mma(unsupported,stride,jobs,count,capacity,input,(uint16_t*)(result+guard),st))
        throw std::runtime_error("missing type fallback");
    ck(cudaStreamDestroy(st),"free stream");for(void* p:allocations)ck(cudaFree(p),"free");
    return checked;
}
}
int main(){try{
    std::cout.setf(std::ios::unitbuf);int checked=0;
    const fnx::Format formats[]={{fnx::T_IQ3_S,256,110,"IQ3_S"},{fnx::T_IQ3_XXS,256,98,"IQ3_XXS"},
        {fnx::T_IQ2_S,256,82,"IQ2_S"},{fnx::T_IQ4_NL,32,18,"IQ4_NL"},{fnx::T_Q4_0,32,18,"Q4_0"},{fnx::T_Q2_0,64,18,"Q2_0"},
        {fnx::T_IQ4_XS,256,136,"IQ4_XS"},{fnx::T_IQ2_XXS,256,66,"IQ2_XXS"}};
    for(int dev=0;dev<2;++dev) {
        fnx::ck(cudaSetDevice(dev),"device");
        for(const auto& f:formats) {
            for(int ffs:{64,128,320})checked+=fnx::run(f,f.block==256?256:320,ffs,false);
            std::cout<<"card "<<dev<<' '<<f.name<<" exact split-K/graph/guards PASS\n";
        }
        for(int i:{0,1,2,6})checked+=fnx::run(formats[i],2560,320,true);
    }
    std::cout<<"PASS "<<checked<<" exact fused gate/up SwiGLU comparisons\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

// Compare production draft-head kernels with the full G8 dot product.
// Test variant numbers are internal to this harness, not FNX_MTP_HEAD_MODE.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
#include <random>
namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
int run(int dev, int rows, int cols, bool bench) {
    const int nb = cols / 32, pad = 64;
    const size_t bytes = (rows + 2*pad) * sizeof(float), rb = nb * sizeof(block_q4_0);
    std::vector<block_q4_0> w((size_t)rows * nb);
    std::vector<block_q8_1> y(nb);
    std::mt19937 rng(120015 + rows + cols);
    for (auto& b : w) {
        b.d = __ushort_as_half((rng() % 2 ? 0x8000 : 0) | (0x2400 + rng() % 0x1800));
        for (auto& q : b.qs) q = rng();
    }
    Job job{}; job.n = 1;
    std::vector<void*> mem;
    auto alloc = [&](size_t n) { void* p; ck(cudaMalloc(&p,n), "alloc"); mem.push_back(p); return p; };
    const size_t wb=w.size()*sizeof(w[0]);
    auto* dw = (char*)alloc(3*wb);
    auto* dy = (block_q8_1*)alloc(y.size()*sizeof(y[0]));
    auto* dj = (Job*)alloc(sizeof(Job));
    auto* out = (float*)alloc(bytes);
    for(int i=0;i<3;++i)ck(cudaMemcpy(dw+i*wb,w.data(),wb,cudaMemcpyHostToDevice),"w");
    ck(cudaMemcpy(dj,&job,sizeof(job),cudaMemcpyHostToDevice),"job");
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
    cudaEvent_t a,b; ck(cudaEventCreate(&a),"event"); ck(cudaEventCreate(&b),"event");
    int weight_copy=0;
    auto call = [&](int mode) {
        const char* weights=dw+(weight_copy++%3)*wb;
        const dim3 grid((rows+15)/16);
        switch(mode) {
        case 0: k_mmvq<T_Q4_0,8><<<grid,128,0,st>>>(weights,0,rb,rows,cols,dj,nullptr,1,dy,nb,out+pad,rows); break;
        case 1: k_draft_head_prune<2,2><<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        case 2: k_draft_head_prune<4,2><<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        case 3: k_draft_head_prune<4,1><<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        case 4: k_draft_head_prune<2,3><<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        case 5: k_draft_head_prune<16,2><<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        case 6: k_draft_head_full<<<grid,128,0,st>>>(weights,rb,rows,cols,dy,out+pad); break;
        }
    };
    int checks = 0;
    for (int rep=0;rep<3;++rep) {
        for (auto& v:y) {
            v.ds=make_half2(__ushort_as_half(0x2400+rng()%0x1800),__ushort_as_half(0x2800+rng()%0x1800));
            for (auto& q:v.qs) q=rep==2?0:int(rng()%255)-127;
            if(rep==2) v.ds=make_half2(__float2half(0.f),__float2half(0.f));
        }
        ck(cudaMemcpyAsync(dy,y.data(),y.size()*sizeof(y[0]),cudaMemcpyHostToDevice,st),"y");
        std::vector<uint32_t> ref(bytes/4), got(bytes/4);
        ck(cudaMemsetAsync(out,0xa5,bytes,st),"guard"); call(0); ck(cudaStreamSynchronize(st),"ref sync");
        ck(cudaMemcpy(ref.data(),out,bytes,cudaMemcpyDeviceToHost),"ref");
        for (int mode=1;mode<=6;++mode) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture"); call(mode);
            ck(cudaStreamEndCapture(st,&graph),"capture end"); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
            for(bool replay:{false,true}) {
                ck(cudaMemsetAsync(out,0xa5,bytes,st),"guard");
                if(replay) ck(cudaGraphLaunch(exec,st),"replay"); else call(mode);
                ck(cudaStreamSynchronize(st),"sync"); ck(cudaMemcpy(got.data(),out,bytes,cudaMemcpyDeviceToHost),"got");
                for(int i=0;i<rows+2*pad;++i) {
                    const bool active=i>=pad && i<pad+rows;
                    if(!active || mode>=5 || got[i]!=0xff7fffffu)
                        if(got[i]!=ref[i]) throw std::runtime_error("surviving score or guard differs");
                }
                const int keep=mode==1||mode==4?2:mode>=5?16:4;
                for(int r0=0;r0<rows;r0+=16) {
                    int kept=0;
                    for(int r=r0;r<std::min(r0+16,rows);++r) {
                        bool selected=got[pad+r]!=0xff7fffffu; kept+=selected;
                        if(rep==2 && selected!=(r-r0<keep)) throw std::runtime_error("zero tie order");
                    }
                    if(kept!=std::min(keep,rows-r0)) throw std::runtime_error("incorrect number of survivors");
                }
                ++checks;
            }
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
        }
        if(bench && rep==0 && rows==53150 && cols==2560) {
            for(int mode:{0,1,2,3,4,5,6,6,5,4,3,2,1,0}) {
                for(int i=0;i<50;++i)call(mode);
                ck(cudaEventRecord(a,st),"start"); for(int i=0;i<80;++i)call(mode); ck(cudaEventRecord(b,st),"stop");
                ck(cudaEventSynchronize(b),"time sync"); float ms; ck(cudaEventElapsedTime(&ms,a,b),"elapsed");
                std::cout<<"card "<<dev<<" mode "<<mode<<" us "<<ms*12.5<<std::endl;
            }
        }
    }
    cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(st);for(void* p:mem)cudaFree(p);return checks;
}
}
int main(int argc,char** argv){try{
    if(argc>2 || (argc==2 && std::string(argv[1])!="--bench"))throw std::runtime_error("usage: split_mtp_head_gemv_test [--bench]");
    const bool bench=argc==2;int checks=0;for(int dev=0;dev<2;++dev){fnx::ck(cudaSetDevice(dev),"device");for(auto shape:{std::pair<int,int>{17,64},{257,320},{53150,2560}})checks+=fnx::run(dev,shape.first,shape.second,bench);}std::cout<<"PASS "<<checks<<" selected exact scores, survivor counts, tie and guard checks\n";}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}

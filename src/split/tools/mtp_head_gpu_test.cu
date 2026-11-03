// Exercise the production Q4 head dispatcher against ggml's 32-lane reader.
#define FNX_KPART 1
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; }
void ck(cudaError_t e, const char* s) { if(e!=cudaSuccess) throw std::runtime_error(std::string(s)+cudaGetErrorString(e)); }
struct TestMem {
    std::vector<void*> all;
    template<class T> T* alloc(size_t n) { void* p;ck(cudaMalloc(&p,n*sizeof(T)),"alloc");all.push_back(p);return (T*)p; }
    ~TestMem(){for(void* p:all)cudaFree(p);}
};
int run(int rows, int cols) {
    constexpr int pad=64;
    const int nb=cols/32,stride=rows+17;
    const size_t count=(size_t)MAXT*stride+2*pad;
    std::vector<uint8_t> weights((size_t)rows*nb*18);
    for(size_t b=0;b<(size_t)rows*nb;++b){
        const uint16_t d=(uint16_t)((b%2?0x8000:0)|0x2400|(b%13)*31);
        std::memcpy(weights.data()+b*18,&d,2);
        for(int j=0;j<16;++j)weights[b*18+2+j]=(b*31+j*19)%256;
    }
    TestMem mem;auto* w=mem.alloc<uint8_t>(weights.size());auto* x=mem.alloc<block_q8_1>(MAXT*nb);
    auto* jobs=mem.alloc<Job>(1);auto* out=mem.alloc<float>(count);auto* ref=mem.alloc<float>(count);
    ck(cudaMemcpy(w,weights.data(),weights.size(),cudaMemcpyHostToDevice),"weights");
    DW dw;dw.p=w;dw.type=T_Q4_0;dw.n_in=cols;dw.n_out=rows;dw.row_bytes=nb*18;
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
    std::vector<block_q8_1> input(MAXT*nb);std::vector<float> a(count),b(count);
    int checks=0;
    for(int T=1;T<=MAXT;++T){
        Job job{};job.n=T;for(int i=0;i<T;++i){job.y[i]=(i*7)%MAXT;job.out[i]=T-1-i;}
        ck(cudaMemcpyAsync(jobs,&job,sizeof job,cudaMemcpyHostToDevice,st),"job");
        ck(cudaStreamSynchronize(st),"job ready");
        auto launch=[&]{mmvq(dw,0,jobs,nullptr,1,x,out+pad,stride,st);};
        cudaGraph_t graph;cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");launch();
        ck(cudaStreamEndCapture(st,&graph),"end capture");ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
        for(int rep=0;rep<3;++rep){
            for(size_t k=0;k<input.size();++k){
                float sum=0;const float d=__half2float(__ushort_as_half(0x2800+(k%11)*27));
                for(int j=0;j<32;++j){int v=rep==2?0:(int)((k*7+j*13+rep*11)%255)-127;input[k].qs[j]=v;sum+=v*d;}
                input[k].ds=make_half2(__float2half(d),__float2half(sum));
            }
            ck(cudaMemcpyAsync(x,input.data(),input.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice,st),"inputs");
            ck(cudaMemsetAsync(ref,0xA5,count*4,st),"ref guard");
            k_mmvq<GGML_TYPE_Q4_0,32><<<(rows+3)/4,128,0,st>>>((const char*)w,0,nb*18,rows,cols,jobs,nullptr,1,x,nb,ref+pad,stride);
            for(bool replay:{false,true}){
                ck(cudaMemsetAsync(out,0xA5,count*4,st),"out guard");if(replay)ck(cudaGraphLaunch(exec,st),"replay");else launch();
                ck(cudaStreamSynchronize(st),"wait");
                ck(cudaMemcpy(a.data(),ref,count*4,cudaMemcpyDeviceToHost),"ref");
                ck(cudaMemcpy(b.data(),out,count*4,cudaMemcpyDeviceToHost),"out");
                for(size_t i=0;i<count;++i){
                    const bool live=i>=pad && (i-pad)/stride<(size_t)T && (i-pad)%stride<(size_t)rows;
                    if(!live){if(std::memcmp(&a[i],&b[i],4))throw std::runtime_error("output guard overwritten");}
                    else if(!std::isfinite(b[i]) || std::fabs(a[i]-b[i])>2e-5f*std::max(1.f,std::fabs(a[i])) || (rep==2&&b[i]!=0))
                        throw std::runtime_error("Q4 head product mismatch");
                }
                ++checks;
            }
        }
        ck(cudaGraphExecDestroy(exec),"destroy exec");ck(cudaGraphDestroy(graph),"destroy graph");
    }
    ck(cudaStreamDestroy(st),"destroy stream");return checks;
}
}
int main(){try{int total=0;for(int dev=0;dev<2;++dev){fnx::ck(cudaSetDevice(dev),"device");int n=0;
    for(auto shape:{std::make_pair(8197,2560),std::make_pair(257,320)})n+=fnx::run(shape.first,shape.second);
    total+=n;std::cout<<"card "<<dev<<": "<<n<<" Q4 direct/graph checks\n";}
    std::cout<<"PASS "<<total<<" Q4 head products and guards\n";
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}

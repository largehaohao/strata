#define FNX_KPART 99
#include "../src/split/src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; }
void ck(cudaError_t e, const char* s) { if(e != cudaSuccess) throw std::runtime_error(std::string(s) + cudaGetErrorString(e)); }
struct Memory {
    std::vector<void*> all;
    template<class T> T* put(const std::vector<T>& v) {
        void* p; ck(cudaMalloc(&p,v.size()*sizeof(T)),"alloc"); all.push_back(p);
        ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"upload"); return (T*)p;
    }
    ~Memory(){for(void* p:all)cudaFree(p);}
};
template<int G> void q4(const char* w,int rows,int cols,const Job* job,const block_q8_1* x,float* out,cudaStream_t st) {
    k_mmvq<GGML_TYPE_Q4_0,G><<<(rows+128/G-1)/(128/G),128,0,st>>>(w,0,cols/32*18,rows,cols,job,nullptr,1,x,cols/32,out,rows);
}
void run(int dev,int rows) {
    constexpr int cols=2560,nb=cols/32,pad=64;
    std::vector<int8_t> w8((size_t)rows*cols),x(cols);
    std::vector<uint16_t> d8((size_t)rows*nb,0x2400);
    std::vector<unsigned char> w4((size_t)rows*nb*18);
    std::vector<float> xd(nb,.03125f);
    std::vector<block_q8_1> packed(nb);
    for(int r=0;r<rows;++r)for(int b=0;b<nb;++b) {
        const size_t at=((size_t)r*nb+b)*18; w4[at]=0;w4[at+1]=0x24;
        for(int j=0;j<16;++j) {
            const int a=(r*13+b*7+j*3)%16, c=(r*17+b*5+j*11)%16;
            w4[at+2+j]=a|(c<<4);w8[(size_t)r*cols+b*32+j]=a-8;w8[(size_t)r*cols+b*32+j+16]=c-8;
        }
    }
    Memory mem;
    auto* dw8=mem.put(w8);auto* dd8=mem.put(d8);auto* dw4=mem.put(w4);
    auto* dx=mem.put(x);auto* ddx=mem.put(xd);auto* dq=mem.put(packed);
    auto* out=mem.put(std::vector<float>(rows+pad));
    Job job{}; job.n=1; job.y[0]=0;job.out[0]=0;
    auto* jobs=mem.put(std::vector<Job>{job});
    Q8Segs g{};g.q[0]=dw8;g.d[0]=dd8;g.out[0]=out;g.end[0]=g.stride[0]=rows;
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
    auto call=[&](int mode){
        if(mode==0) k_q8s<1,2,true,false><<<(rows+15)/16,256,0,st>>>(g,rows,cols,nullptr,dx,ddx);
        else if(mode==8)q4<8>((const char*)dw4,rows,cols,jobs,dq,out,st);
        else if(mode==16)q4<16>((const char*)dw4,rows,cols,jobs,dq,out,st);
        else q4<32>((const char*)dw4,rows,cols,jobs,dq,out,st);
    };
    cudaGraph_t graph[4];cudaGraphExec_t exec[4];const int modes[]={0,8,16,32};
    for(int m=0;m<4;++m) {
        ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");call(modes[m]);
        ck(cudaStreamEndCapture(st,&graph[m]),"end capture");ck(cudaGraphInstantiate(&exec[m],graph[m],nullptr,nullptr,0),"instantiate");
    }
    std::vector<unsigned char> ref((rows+pad)*4),got(ref.size());
    for(int rep=0;rep<3;++rep) {
        for(int b=0;b<nb;++b){int sum=0;for(int j=0;j<32;++j){const int v=rep==2?0:((b*11+j*3+rep*7)%255)-127;x[b*32+j]=v;packed[b].qs[j]=v;sum+=v;}
            packed[b].ds=make_half2(__float2half(.03125f),__float2half(sum*.03125f));}
        ck(cudaMemcpyAsync(dx,x.data(),x.size(),cudaMemcpyHostToDevice,st),"x");
        ck(cudaMemcpyAsync(dq,packed.data(),packed.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice,st),"packed x");
        ck(cudaMemsetAsync(out,0xA5,ref.size(),st),"guard");call(0);ck(cudaStreamSynchronize(st),"ref wait");
        ck(cudaMemcpy(ref.data(),out,ref.size(),cudaMemcpyDeviceToHost),"ref");
        for(int m=0;m<4;++m)for(int replay=0;replay<2;++replay){
            ck(cudaMemsetAsync(out,0xA5,ref.size(),st),"guard");if(replay)ck(cudaGraphLaunch(exec[m],st),"graph");else call(modes[m]);
            ck(cudaStreamSynchronize(st),"wait");ck(cudaMemcpy(got.data(),out,got.size(),cudaMemcpyDeviceToHost),"got");
            if(got!=ref)throw std::runtime_error("head mismatch "+std::to_string(modes[m]));
        }
    }
    cudaEvent_t start,end;ck(cudaEventCreate(&start),"event");ck(cudaEventCreate(&end),"event");
    for(int m:{0,1,2,3,3,2,1,0}){
        for(int i=0;i<200;++i)call(modes[m]);
        ck(cudaEventRecord(start,st),"start");for(int i=0;i<100;++i)call(modes[m]);ck(cudaEventRecord(end,st),"end");
        ck(cudaEventSynchronize(end),"time wait");float ms;ck(cudaEventElapsedTime(&ms,start,end),"time");
        std::cout<<"card "<<dev<<" rows "<<rows<<" mode "<<modes[m]<<" us "<<ms*10<<" exact direct/graph\n";
    }
    for(int m=0;m<4;++m){cudaGraphExecDestroy(exec[m]);cudaGraphDestroy(graph[m]);}
    cudaEventDestroy(start);cudaEventDestroy(end);cudaStreamDestroy(st);
}
}
int main(){try{for(int dev=0;dev<2;++dev){fnx::ck(cudaSetDevice(dev),"device");for(int rows:{53150,53149})fnx::run(dev,rows);}}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}

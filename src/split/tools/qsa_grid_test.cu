// Exact local score-grid results, paged KV guards and graph replay on both cards.
#define FNX_KPART 99
#include "../src/kernels.cu"

#include <array>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> twins; return twins; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
namespace {
constexpr int pools=65536, max_rows=16, max_stride=16, pad=13;
constexpr size_t cells_size=2*pad+(size_t)2*max_stride*QSA_MAXSEL, counts_size=2*pad+2*max_stride;
struct Memory {
    int dev; struct Block {unsigned char* p;size_t bytes;};std::vector<Block> blocks;
    explicit Memory(int device):dev(device){}
    template<class V> V* make(size_t n) {
        unsigned char* p=nullptr; const size_t bytes=n*sizeof(V);
        ck(cudaMalloc((void**)&p,bytes+256),"allocation");
        ck(cudaMemset(p,0xa5,bytes+256),"allocation guards");blocks.push_back({p,bytes});
        return reinterpret_cast<V*>(p+128);
    }
    void guards(){
        ck(cudaSetDevice(dev),"guard device"); unsigned char h[256];
        for(auto b:blocks){
            ck(cudaMemcpy(h,b.p,128,cudaMemcpyDeviceToHost),"front guard");
            ck(cudaMemcpy(h+128,b.p+128+b.bytes,128,cudaMemcpyDeviceToHost),"back guard");
            for(auto v:h)if(v!=0xa5)throw std::runtime_error("QSA shard guard overwritten");
        }
    }
    ~Memory(){cudaSetDevice(dev);for(auto b:blocks)cudaFree(b.p);}
};
__global__ void initialize_keys(__half* keys,KvPages pages){
    const int j=blockIdx.x,d=threadIdx.x;
    unsigned v=(j*QSA_DIM+d)*1664525u+1013904223u;v^=v>>11;
    keys[kvp_block(pages,j)+d]=__float2half_rn((int(v%65537)-32768)*0x1p-12f);
}
__global__ void compare_ints(const int* a,const int* b,size_t n,unsigned* errors){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x)
        if(a[i]!=b[i])atomicAdd(errors,1u);
}
__global__ void compare_scores(const float* ref,const float* got,int T,unsigned* errors){
    const int j=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(j>=pools||t>=T)return;
    const size_t i=(size_t)t*pools+j;
    if(__float_as_uint(ref[i])!=__float_as_uint(got[i]))atomicAdd(errors,1u);
}
struct State {
    int dev;Memory mem;cudaStream_t st;
    __half* keys;float *query,*ref_score,*score;int *ref_cells,*cells,*ref_counts,*counts,*pos;
    KvPages pages;unsigned* check;
    explicit State(int device):dev(device),mem(device){
        ck(cudaSetDevice(dev),"state device");ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
        constexpr int np=(pools+KVP_BLOCKS-1)/KVP_BLOCKS;
        const size_t ps=kvp_page(dev,true);auto* table=mem.make<int>(np);
        std::vector<int> hp(np);std::iota(hp.begin(),hp.end(),0);std::mt19937 rng(51739);std::shuffle(hp.begin(),hp.end(),rng);
        ck(cudaMemcpy(table,hp.data(),np*sizeof(int),cudaMemcpyHostToDevice),"page table");
        pages={table,ps,kvp_rows(dev),1};keys=mem.make<__half>(np*ps);
        query=mem.make<float>((size_t)max_rows*512);
        ref_score=mem.make<float>((size_t)max_rows*pools);score=mem.make<float>((size_t)max_rows*pools);
        ref_cells=mem.make<int>(cells_size);cells=mem.make<int>(cells_size);
        ref_counts=mem.make<int>(counts_size);counts=mem.make<int>(counts_size);pos=mem.make<int>(1);
        check=mem.make<unsigned>(3);
        initialize_keys<<<pools,QSA_DIM,0,st>>>(keys,pages);ck(cudaStreamSynchronize(st),"key initialization");
    }
    ~State(){cudaSetDevice(dev);cudaStreamDestroy(st);}
};
struct Shape {int budget,T,stride;};
void launch(State& s,State&,Shape sh,bool candidate){
    float* score=candidate?s.score:s.ref_score;
    int* cells=(candidate?s.cells:s.ref_cells)+pad;int* counts=(candidate?s.counts:s.ref_counts)+pad;
    const int blocks=std::min((pools+255)/256,std::max(1,(sh.budget+sh.T-1)/sh.T));
    if(candidate)k_idx_score_persistent<<<dim3(blocks,sh.T),256,0,s.st>>>(s.query,s.keys,s.pages,score,pools,s.pos,{});
    else k_idx_score<<<dim3((pools+255)/256,sh.T),256,0,s.st>>>(s.query,s.keys,s.pages,score,pools,s.pos,{});
    k_idx_select_pair<true,true><<<sh.T,QSA_SEL_THREADS,0,s.st>>>(score,cells,counts,pools,s.pos,sh.stride,{});
    ck(cudaGetLastError(),"QSA persistent launch");
}
struct Graphs {
    cudaGraph_t graph[2]{};cudaGraphExec_t exec[2]{};
    Graphs(std::array<std::unique_ptr<State>,2>& s,Shape sh,bool candidate,int repetitions){
        for(int c=0;c<2;++c){ck(cudaSetDevice(c),"capture device");
            ck(cudaStreamBeginCapture(s[c]->st,cudaStreamCaptureModeThreadLocal),"capture");
            for(int i=0;i<repetitions;++i)launch(*s[c],*s[1-c],sh,candidate);
            ck(cudaStreamEndCapture(s[c]->st,&graph[c]),"end capture");
            ck(cudaGraphInstantiate(&exec[c],graph[c],nullptr,nullptr,0),"instantiate");
        }
    }
    ~Graphs(){for(int c=0;c<2;++c){cudaSetDevice(c);cudaGraphExecDestroy(exec[c]);cudaGraphDestroy(graph[c]);}}
};
void sync(std::array<std::unique_ptr<State>,2>& s){for(int c=0;c<2;++c){ck(cudaSetDevice(c),"sync device");ck(cudaStreamSynchronize(s[c]->st),"synchronize");}}
void run(){
    std::array<std::unique_ptr<State>,2> s;for(int c=0;c<2;++c)s[c]=std::make_unique<State>(c);
    std::vector<Shape> shapes;
    for(int budget:{64,128,256,512})for(int T:{1,2,4,8,16})shapes.push_back({budget,T,16});
    size_t checks=0;
    for(int pattern=0;pattern<3;++pattern){
        std::mt19937 rng(7329+pattern);std::vector<float> q((size_t)max_rows*512);
        for(auto& v:q)v=pattern==0?0.f:(int(rng()%65537)-32768)*(pattern==1?0x1p-28f:0x1p-13f);
        for(int c=0;c<2;++c){ck(cudaSetDevice(c),"query device");ck(cudaMemcpy(s[c]->query,q.data(),q.size()*4,cudaMemcpyHostToDevice),"queries");}
        for(Shape sh:shapes){
            Graphs graphs(s,sh,true,1);
            for(int position:{0,2046,2051,32761,74401,204793,262144-sh.T}){
                for(int c=0;c<2;++c){auto& a=*s[c];ck(cudaSetDevice(c),"reference device");
                    ck(cudaMemcpy(a.pos,&position,4,cudaMemcpyHostToDevice),"position");
                    ck(cudaMemsetAsync(a.ref_cells,0x5a,cells_size*4,a.st),"reference cells");
                    ck(cudaMemsetAsync(a.ref_counts,0x5a,counts_size*4,a.st),"reference counts");
                    ck(cudaMemsetAsync(a.ref_score,0x5a,(size_t)sh.T*pools*4,a.st),"reference scores");
                    launch(a,*s[1-c],sh,false);
                }
                sync(s);
                for(int replay=0;replay<3;++replay){
                    for(int c=0;c<2;++c){auto& a=*s[c];ck(cudaSetDevice(c),"candidate device");
                        ck(cudaMemsetAsync(a.cells,0x5a,cells_size*4,a.st),"candidate cells");
                        ck(cudaMemsetAsync(a.counts,0x5a,counts_size*4,a.st),"candidate counts");
                        ck(cudaMemsetAsync(a.score,0x5a,(size_t)sh.T*pools*4,a.st),"candidate scores");
                        ck(cudaMemsetAsync(a.check,0,12,a.st),"comparison counters");
                        if(replay)ck(cudaGraphLaunch(graphs.exec[c],a.st),"replay");else launch(a,*s[1-c],sh,true);
                        compare_ints<<<1024,256,0,a.st>>>(a.ref_cells,a.cells,cells_size,a.check);
                        compare_ints<<<64,256,0,a.st>>>(a.ref_counts,a.counts,counts_size,a.check+1);
                        compare_scores<<<dim3((pools+255)/256,sh.T),256,0,a.st>>>(a.ref_score,a.score,sh.T,a.check+2);
                    }
                    sync(s);
                    for(int c=0;c<2;++c){ck(cudaSetDevice(c),"result device");unsigned e[3];ck(cudaMemcpy(e,s[c]->check,12,cudaMemcpyDeviceToHost),"results");
                        if(e[0]||e[1]||e[2]){
                            std::fprintf(stderr,"QSA_GRID_FAIL card=%d pattern=%d budget=%d T=%d pos=%d replay=%d cells=%u counts=%u scores=%u\n",c,pattern,sh.budget,sh.T,position,replay,e[0],e[1],e[2]);
                            throw std::runtime_error("QSA score grid mismatch");
                        }++checks;
                    }
                }
            }
            std::printf("QSA_GRID_CHECK pattern=%d budget=%d T=%d checks=%zu\n",pattern,sh.budget,sh.T,checks);std::fflush(stdout);
        }
        for(auto& a:s)a->mem.guards();
    }
    std::printf("QSA_GRID passed %zu exact score/list/count/guard/direct/graph checks\n",checks);std::fflush(stdout);

}
}
}
int main(){try{fnx::run();}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

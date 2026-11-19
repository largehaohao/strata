// Exact two-card query ownership, packed-list exchange, guards and graph replay.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include "../src/hip/peer.h"
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
constexpr int pools=65536, max_rows=512, max_stride=4099, pad=13;
constexpr size_t cells_size=2*pad+(size_t)2*max_stride*QSA_MAXSEL, counts_size=2*pad+2*max_stride;
struct Memory {
    int dev; struct Block {unsigned char* p;size_t bytes;};std::vector<Block> blocks;
    explicit Memory(int device):dev(device){}
    template<class V> V* make(size_t n,bool fine=false) {
        unsigned char* p=nullptr; const size_t bytes=n*sizeof(V);
        if(fine) ck(hipExtMallocWithFlags((void**)&p,bytes+256,hipDeviceMallocFinegrained),"fine allocation");
        else ck(cudaMalloc((void**)&p,bytes+256),"allocation");
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
__global__ void compare_scores(const float* ref,const float* got,int T,const int* pos,int group,int card,bool split,unsigned* errors){
    const int j=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    const int visible=(*pos+t+1)/QSA_POOL; const size_t i=(size_t)t*pools+j;
    if(j>=pools)return;
    if(!split||(t/group)%2==card){
        if(visible>QSA_TOP&&j<visible&&__float_as_uint(ref[i])!=__float_as_uint(got[i]))atomicAdd(errors,1u);
    }else if(__float_as_uint(got[i])!=0x5a5a5a5au)atomicAdd(errors,1u);
}
__global__ void flush_cache(unsigned* data,size_t n){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x)
        data[i]=data[i]*1664525u+1013904223u;
}
struct State {
    int dev;Memory mem;cudaStream_t st;cudaEvent_t begin,end;
    __half* keys;float *query,*ref_score,*score;int *ref_cells,*cells,*ref_counts,*counts,*pos;
    KvPages pages;QsxMail mail;unsigned *check,*flush;
    explicit State(int device):dev(device),mem(device){
        ck(cudaSetDevice(dev),"state device");ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
        ck(cudaEventCreate(&begin),"event");ck(cudaEventCreate(&end),"event");
        constexpr int np=(pools+KVP_BLOCKS-1)/KVP_BLOCKS;
        const size_t ps=kvp_page(dev,true);auto* table=mem.make<int>(np);
        std::vector<int> hp(np);std::iota(hp.begin(),hp.end(),0);std::mt19937 rng(51739);std::shuffle(hp.begin(),hp.end(),rng);
        ck(cudaMemcpy(table,hp.data(),np*sizeof(int),cudaMemcpyHostToDevice),"page table");
        pages={table,ps,kvp_rows(dev),1};keys=mem.make<__half>(np*ps);
        query=mem.make<float>((size_t)max_rows*512);
        ref_score=mem.make<float>((size_t)max_rows*pools);score=mem.make<float>((size_t)max_rows*pools);
        ref_cells=mem.make<int>(cells_size);cells=mem.make<int>(cells_size);
        ref_counts=mem.make<int>(counts_size);counts=mem.make<int>(counts_size);pos=mem.make<int>(1);
        mail.data=mem.make<int>((size_t)2*max_rows*QSX_WORDS,true);
        mail.ready=mem.make<unsigned>(2*max_rows,true);mail.generation=mem.make<unsigned>(1);mail.errors=mem.make<unsigned>(1);
        mail.capacity=max_rows;int rate=0;ck(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,dev),"wall clock rate");
        mail.timeout_ticks=(unsigned long long)rate*1000;
        check=mem.make<unsigned>(3);flush=mem.make<unsigned>(32*1024*1024);
        ck(cudaMemset(mail.ready,0,2*max_rows*sizeof(unsigned)),"unpublished flags");
        ck(cudaMemset(mail.errors,0,sizeof(unsigned)),"mail errors");
        unsigned generation=0xfffffffcu;ck(cudaMemcpy(mail.generation,&generation,4,cudaMemcpyHostToDevice),"wrap seed");
        initialize_keys<<<pools,QSA_DIM,0,st>>>(keys,pages);ck(cudaStreamSynchronize(st),"key initialization");
    }
    ~State(){cudaSetDevice(dev);cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(st);}
};
struct Shape {int kind,T,stride;int group()const{return kind==0?1:kind==1?4:16;}bool split()const{return T>group();}};
void launch(State& s,State& peer,Shape sh,bool candidate){
    const bool split=candidate&&sh.split();const int T=sh.T;
    float* score=candidate?s.score:s.ref_score;
    int* cells=(candidate?s.cells:s.ref_cells)+pad;int* counts=(candidate?s.counts:s.ref_counts)+pad;
    if(split)k_idx_score_tc_wide<true><<<dim3((pools+255)/256,(T+3)/4),256,0,s.st>>>(
        s.query,s.keys,s.pages,score,pools,s.pos,T,s.dev,s.mail.generation);
    else k_idx_score_tc_wide<<<dim3((pools+255)/256,(T+3)/4),256,0,s.st>>>(
        s.query,s.keys,s.pages,score,pools,s.pos,T);
    if(split){
        k_idx_select_pair<true,true,true><<<T,QSA_SEL_THREADS,0,s.st>>>(score,cells,counts,pools,s.pos,sh.stride,{},s.dev,sh.group());
        k_qsx_push<<<T,256,0,s.st>>>(cells,counts,T,sh.stride,s.dev,sh.group(),s.mail,peer.mail);
        k_qsx_pull<<<T,256,0,s.st>>>(cells,counts,T,sh.stride,s.dev,sh.group(),s.mail);
    }else k_idx_select_pair<true,true><<<T,QSA_SEL_THREADS,0,s.st>>>(score,cells,counts,pools,s.pos,sh.stride,{});
    ck(cudaGetLastError(),"QSA launch");
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
void check_mail(std::array<std::unique_ptr<State>,2>& s){
    unsigned gen[2];for(int c=0;c<2;++c){ck(cudaSetDevice(c),"check device");unsigned e=0;
        ck(cudaMemcpy(&e,s[c]->mail.errors,4,cudaMemcpyDeviceToHost),"mail error");
        if(e)throw std::runtime_error("QSA peer error card="+std::to_string(c)+" bits="+std::to_string(e));
        ck(cudaMemcpy(&gen[c],s[c]->mail.generation,4,cudaMemcpyDeviceToHost),"generation");
    }
    if(gen[0]!=gen[1])throw std::runtime_error("QSA generation skew");
}
void run(){
    std::string why;if(!peer_enable_and_probe({0,1},why))throw std::runtime_error(why);
    std::printf("PEER %s\n",why.c_str());std::fflush(stdout);
    std::array<std::unique_ptr<State>,2> s;for(int c=0;c<2;++c)s[c]=std::make_unique<State>(c);
    std::vector<Shape> shapes;
    for(int T:{1,4,5,15,16,31,32,63,64,255,256,511,512})shapes.push_back({1,T,4096});
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
                        compare_scores<<<dim3((pools+255)/256,sh.T),256,0,a.st>>>(a.ref_score,a.score,sh.T,a.pos,sh.group(),c,sh.split(),a.check+2);
                    }
                    sync(s);check_mail(s);
                    for(int c=0;c<2;++c){ck(cudaSetDevice(c),"result device");unsigned e[3];ck(cudaMemcpy(e,s[c]->check,12,cudaMemcpyDeviceToHost),"results");
                        if(e[0]||e[1]||e[2]){
                            std::fprintf(stderr,"QSA_SHARD_FAIL card=%d pattern=%d kind=%d T=%d pos=%d replay=%d cells=%u counts=%u scores=%u\n",c,pattern,sh.kind,sh.T,position,replay,e[0],e[1],e[2]);
                            throw std::runtime_error("QSA query partition mismatch");
                        }++checks;
                    }
                }
            }
            std::printf("QSA_SHARD_CHECK pattern=%d kind=%d T=%d checks=%zu\n",pattern,sh.kind,sh.T,checks);std::fflush(stdout);
        }
        for(auto& a:s)a->mem.guards();
    }
    std::printf("QSA_QUERY_SPLIT passed %zu exact score/list/count/guard/direct/graph/wrap checks\n",checks);std::fflush(stdout);
    const std::vector<Shape> timed={{1,256,4096},{1,512,4096}};
    for(Shape sh:timed)for(int position:{32767,74401,204799})for(int cold=0;cold<2;++cold){
        const int iterations=cold?1:8;
        Graphs reference(s,sh,false,iterations),candidate(s,sh,true,iterations);
        for(int c=0;c<2;++c){ck(cudaSetDevice(c),"timing device");ck(cudaMemcpy(s[c]->pos,&position,4,cudaMemcpyHostToDevice),"timing position");}
        std::array<std::vector<double>,2> gpu,wall;
        for(int rep=-2;rep<9;++rep)for(int pass=0;pass<2;++pass){
            const int mode=(std::max(rep,0)+pass)%2;auto& g=mode?candidate:reference;
            if(cold){for(int c=0;c<2;++c){ck(cudaSetDevice(c),"flush device");flush_cache<<<1024,256,0,s[c]->st>>>(s[c]->flush,32*1024*1024);}sync(s);}
            const auto t0=std::chrono::steady_clock::now();
            for(int c=0;c<2;++c){ck(cudaSetDevice(c),"timed device");ck(cudaEventRecord(s[c]->begin,s[c]->st),"start");
                ck(cudaGraphLaunch(g.exec[c],s[c]->st),"timed graph");ck(cudaEventRecord(s[c]->end,s[c]->st),"end");}
            sync(s);const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/iterations;
            double slow=0;for(int c=0;c<2;++c){ck(cudaSetDevice(c),"elapsed device");float ms=0;ck(cudaEventElapsedTime(&ms,s[c]->begin,s[c]->end),"elapsed");slow=std::max(slow,ms*1000.0/iterations);}
            if(rep>=0){gpu[mode].push_back(slow);wall[mode].push_back(us);}
        }
        check_mail(s);for(auto& v:gpu)std::sort(v.begin(),v.end());for(auto& v:wall)std::sort(v.begin(),v.end());
        std::printf("QSA_SHARD_TIME kind=%d T=%d pos=%d cold=%d reference_gpu_us=%.3f shard_gpu_us=%.3f reference_wall_us=%.3f shard_wall_us=%.3f\n",sh.kind,sh.T,position,cold,gpu[0][4],gpu[1][4],wall[0][4],wall[1][4]);std::fflush(stdout);
    }

    // Error paths must return without a cross-card producer or a valid payload.
    int error_checks=0;
    for(int c=0;c<2;++c){
        auto& a=*s[c];ck(cudaSetDevice(c),"negative test device");
        const unsigned generation=901;
        ck(cudaMemcpy(a.mail.generation,&generation,4,cudaMemcpyHostToDevice),"negative generation");
        auto reset=[&]{
            ck(cudaMemset(a.mail.errors,0,4),"negative error clear");
            ck(cudaMemset(a.mail.ready,0,2*max_rows*4),"negative ready clear");
            ck(cudaMemset(a.counts,0,counts_size*4),"negative counts clear");
        };
        auto expect=[&](unsigned expected){
            ck(cudaGetLastError(),"negative launch");
            ck(cudaStreamSynchronize(a.st),"negative completion");
            unsigned got=0;ck(cudaMemcpy(&got,a.mail.errors,4,cudaMemcpyDeviceToHost),"negative result");
            if(got!=expected)throw std::runtime_error("QSA exchange error status mismatch");
            ++error_checks;
        };
        reset();
        // Model one missing row from the other card; use logical card zero
        // on either physical device so row four is always the receive row.
        k_qsx_pull<<<5,256,0,a.st>>>(a.cells,a.counts,5,max_rows,0,4,a.mail);
        expect(1u);
        reset();
        int bad=-2;ck(cudaMemcpy(a.counts,&bad,4,cudaMemcpyHostToDevice),"negative count");
        k_qsx_push<<<1,256,0,a.st>>>(a.cells,a.counts,1,max_rows,0,4,a.mail,s[1-c]->mail);
        expect(2u);
        reset();
        int count=4,unaligned=3;
        ck(cudaMemcpy(a.counts,&count,4,cudaMemcpyHostToDevice),"negative size");
        ck(cudaMemcpy(a.cells,&unaligned,4,cudaMemcpyHostToDevice),"negative block");
        k_qsx_push<<<1,256,0,a.st>>>(a.cells,a.counts,1,max_rows,0,4,a.mail,s[1-c]->mail);
        expect(8u);
        reset();
        const size_t slot=(generation&1u)*max_rows+4;
        unsigned oversized=QSA_MAXSEL+1;
        ck(cudaMemcpy(a.mail.data+slot*QSX_WORDS,&oversized,4,cudaMemcpyHostToDevice),"negative payload");
        ck(cudaMemcpy(a.mail.ready+slot,&generation,4,cudaMemcpyHostToDevice),"negative publication");
        k_qsx_pull<<<5,256,0,a.st>>>(a.cells,a.counts,5,max_rows,0,4,a.mail);
        expect(4u);
    }
    std::printf("QSA_QUERY_SPLIT passed %d bounded timeout/malformed payload checks\n",error_checks);
    for(auto& a:s)a->mem.guards();
}
}
}
int main(){try{fnx::run();}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

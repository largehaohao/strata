// Exact GU arithmetic, complete cache bytes and guards with cold routes and graph replay.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>
namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void ck(cudaError_t e,const char* s) { if(e!=cudaSuccess) throw std::runtime_error(std::string(s)+cudaGetErrorString(e)); }

// Same coalesced copy as k_cache_copy, with a byte tail for the synthetic odd row count.
__global__ void gu_copy_reference(const char* const* host,const char* const* ep,const int* list,int base,size_t bytes) {
    const size_t per=bytes/16, n=list[0];
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n*per;i+=(size_t)gridDim.x*blockDim.x) {
        const int g=list[1+i/per];
        reinterpret_cast<uint4*>(const_cast<char*>(ep[g-base]))[i%per]=reinterpret_cast<const uint4*>(host[g])[i%per];
    }
    for(int j=blockIdx.x;j<(int)n;j+=gridDim.x) {
        const int g=list[1+j];
        for(size_t i=per*16+threadIdx.x;i<bytes;i+=blockDim.x) const_cast<char*>(ep[g-base])[i]=host[g][i];
    }
}
template <ggml_type TA,int T>
void launch(bool fused,const MoeW& w,const Job* jobs,const int* nj,const block_q8_1* x,float* out,CacheGateUpCopy cache,CacheDownCopy down,int maxjobs,cudaStream_t st) {
    const dim3 grid(((w.nrows+7)/8)*maxjobs+down.blocks);
    if(fused) moe_launch<TA>(false,w,jobs,nj,maxjobs,x,out,st,T,&down,&cache);
    else {
        gu_copy_reference<<<160,256,0,st>>>(cache.host,w.ep,cache.list,cache.layer*NEXP,(size_t)w.nrows*w.row_a);
        k_moe_gu_iq<TA,T,16,true><<<grid,128,0,st>>>(w,jobs,nj,x,out,down);
    }
}

template <ggml_type TA> int run(int dev,int rows,bool coherent) {
    constexpr int experts=48,cols=N,base=3*NEXP,guard=256;
    constexpr size_t row_bytes=cols/ggml_cuda_type_traits<TA>::qk*ggml_cuda_type_traits<TA>::bs;
    const size_t raw_bytes=rows*row_bytes,host_stride=(raw_bytes+15)/16*16,slot_stride=host_stride+guard;
    const size_t pool_bytes=experts*slot_stride+guard*2,output_bytes=(size_t)experts*MAXT*rows*4+guard;
    std::vector<void*> mem;auto alloc=[&](size_t n){void*p;ck(cudaMalloc(&p,n),"alloc");mem.push_back(p);return p;};
    char* host=nullptr;ck(hipHostMalloc((void**)&host,experts*host_stride,coherent?hipHostMallocCoherent:hipHostMallocNonCoherent),"host");
    auto* pool=(char*)alloc(pool_bytes);auto* ep=(const char**)alloc(experts*sizeof(char*));
    auto* ht=(const char**)alloc((base+experts)*sizeof(char*));auto* list=(int*)alloc((experts+1)*sizeof(int));
    auto* jobs=(Job*)alloc(experts*sizeof(Job));auto* count=(int*)alloc(sizeof(int));
    auto* x=(block_q8_1*)alloc((size_t)MAXT*cols/32*sizeof(block_q8_1));auto* out=(float*)alloc(output_bytes);
    std::vector<const char*> hep(experts),hht(base+experts);
    for(int e=0;e<experts;++e){hep[e]=pool+guard+((e*5)%experts)*slot_stride;hht[base+e]=host+e*host_stride;}
    ck(cudaMemcpy(ep,hep.data(),hep.size()*sizeof(char*),cudaMemcpyHostToDevice),"expert pointers");
    ck(cudaMemcpy(ht,hht.data(),hht.size()*sizeof(char*),cudaMemcpyHostToDevice),"host pointers");
    MoeW w{nullptr,raw_bytes,row_bytes,rows,cols,ep};CacheGateUpCopy cache{ht,list,3};
    const size_t dn_bytes=TA==GGML_TYPE_IQ3_XXS&&rows==640&&!coherent?460800:4112;
    const size_t dn_stride=dn_bytes+256,dn_pool_bytes=experts*dn_stride+guard*2;
    char* dn_host=nullptr;ck(hipHostMalloc((void**)&dn_host,experts*dn_bytes,coherent?hipHostMallocCoherent:hipHostMallocNonCoherent),"down host");
    auto* dn_pool=(char*)alloc(dn_pool_bytes);auto* dn_ht=(const char**)alloc((base+experts)*sizeof(char*));
    auto* dn_slots=(int*)alloc((base+experts)*sizeof(int));
    std::vector<const char*> hdn(base+experts);std::vector<int> hs(base+experts,-1);
    for(int e=0;e<experts;++e){hdn[base+e]=dn_host+e*dn_bytes;hs[base+e]=17+(e*5)%experts;}
    ck(cudaMemcpy(dn_ht,hdn.data(),hdn.size()*sizeof(char*),cudaMemcpyHostToDevice),"down host pointers");
    ck(cudaMemcpy(dn_slots,hs.data(),hs.size()*sizeof(int),cudaMemcpyHostToDevice),"down slots");
    CacheDownCopy down{dn_ht,dn_slots,list,dn_pool+guard,dn_stride,128,dn_bytes,17,16};
    std::vector<unsigned char> dn_expected(dn_pool_bytes),dn_got(dn_pool_bytes);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");ck(cudaDeviceSynchronize(),"upload wait");
    std::vector<unsigned char> initial(pool_bytes),expected(pool_bytes),got_pool(pool_bytes),ref(output_bytes),got(output_bytes);
    std::vector<block_q8_1> hx(MAXT*cols/32);std::vector<Job> hj(experts);std::vector<int> hl(experts+1);
    int T=1,maxjobs=experts;
    auto call=[&](bool fused){switch(T){
#define CASE(t) case t:launch<TA,t>(fused,w,jobs,count,x,out,cache,down,maxjobs,st);break
        CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
        CASE(9);CASE(10);CASE(11);CASE(12);CASE(13);CASE(14);CASE(15);CASE(16);
#undef CASE
    }};
    auto prepare=[&](int nj,int misses,int seed) {
        for(int e=0;e<experts;++e){
            auto* at=(unsigned char*)host+e*host_stride;
            for(size_t i=0;i<raw_bytes;++i)at[i]=(i*13+(i>>7)*17+e*31+seed*11)&255;
            for(size_t i=0;i<raw_bytes;i+=ggml_cuda_type_traits<TA>::bs){uint16_t d=0x2400+seed*17;std::memcpy(at+i,&d,2);}
            hj[e].w=(e*7+seed*11)%experts;hj[e].n=1+(e+seed)%T;
            for(int t=0;t<hj[e].n;++t){hj[e].y[t]=(t+seed)%T;hj[e].out[t]=e*MAXT+t;}
        }
        std::fill(initial.begin(),initial.end(),0xCC);
        for(int e=0;e<experts;++e)std::memcpy(initial.data()+guard+((e*5)%experts)*slot_stride,host+e*host_stride,raw_bytes);
        expected=initial;hl[0]=std::min(nj,misses);
        std::fill(dn_expected.begin(),dn_expected.end(),0xCD);
        for(size_t i=0;i<experts*dn_bytes;++i)dn_host[i]=(i*19+i/31+seed*7)&255;
        for(int j=0;j<hl[0];++j){const int e=hj[j].w;std::memcpy(dn_expected.data()+guard+((e*5)%experts)*dn_stride+128,dn_host+e*dn_bytes,dn_bytes);}
        for(int j=0;j<hl[0];++j){const int e=hj[j].w;hl[1+j]=base+e;std::memset(initial.data()+guard+((e*5)%experts)*slot_stride,0xA5,raw_bytes);}
        for(size_t i=0;i<hx.size();++i){int sum=0;for(int k=0;k<32;++k)sum+=hx[i].qs[k]=seed==0?0:(int)((i*7+k*11+seed*13)%255)-127;hx[i].ds=make_half2(__float2half(.02f),__float2half(sum*.02f));}
        ck(cudaMemcpyAsync(jobs,hj.data(),hj.size()*sizeof(Job),cudaMemcpyHostToDevice,st),"jobs");
        ck(cudaMemcpyAsync(count,&nj,sizeof(nj),cudaMemcpyHostToDevice,st),"count");
        ck(cudaMemcpyAsync(list,hl.data(),hl.size()*sizeof(int),cudaMemcpyHostToDevice,st),"misses");
        ck(cudaMemcpyAsync(x,hx.data(),hx.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice,st),"input");
        ck(cudaStreamSynchronize(st),"prepare wait");
    };
    auto reset=[&]{ck(cudaMemcpyAsync(pool,initial.data(),pool_bytes,cudaMemcpyHostToDevice,st),"poison cold cache");ck(cudaMemsetAsync(out,0xB3,output_bytes,st),"output guards");ck(cudaMemsetAsync(dn_pool,0xCD,dn_pool_bytes,st),"down guards");};
    int checks=0;
    for(T=1;T<=MAXT;++T){
        cudaGraph_t graph;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");call(true);ck(cudaStreamEndCapture(st,&graph),"capture end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"instantiate");
        for(int nj:{0,1,7,experts})for(int misses:{0,1,7,experts}){
            prepare(nj,misses,(nj+misses+T)%8);reset();call(false);ck(cudaStreamSynchronize(st),"reference wait");
            ck(cudaMemcpy(ref.data(),out,output_bytes,cudaMemcpyDeviceToHost),"reference");
            ck(cudaMemcpy(got_pool.data(),pool,pool_bytes,cudaMemcpyDeviceToHost),"reference cache");
            ck(cudaMemcpy(dn_got.data(),dn_pool,dn_pool_bytes,cudaMemcpyDeviceToHost),"reference down");
            if(got_pool!=expected||dn_got!=dn_expected)throw std::runtime_error("reference cache mismatch");
            for(bool replay:{false,true}){
                reset();if(replay)ck(cudaGraphLaunch(ex,st),"replay");else call(true);ck(cudaStreamSynchronize(st),"wait");
                ck(cudaMemcpy(got.data(),out,output_bytes,cudaMemcpyDeviceToHost),"output");ck(cudaMemcpy(got_pool.data(),pool,pool_bytes,cudaMemcpyDeviceToHost),"cache");
                ck(cudaMemcpy(dn_got.data(),dn_pool,dn_pool_bytes,cudaMemcpyDeviceToHost),"down copy result");
                if(got!=ref||got_pool!=expected||dn_got!=dn_expected)throw std::runtime_error("fused GU cache mismatch type="+std::to_string(TA)+" T="+std::to_string(T));++checks;
            }
        }cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(st);cudaFreeHost(host);cudaFreeHost(dn_host);for(void*p:mem)cudaFree(p);return checks;
}
} // namespace fnx
int main(){try{setenv("FNX_IQ_REUSE","1",1);setenv("FNX_IQ2_XXS_REUSE","1",1);setenv("FNX_IQ_GROUP","16",1);int n=0;for(int dev=0;dev<2;++dev){fnx::ck(cudaSetDevice(dev),"device");for(bool coherent:{false,true})for(int rows:{640,641}){n+=fnx::run<GGML_TYPE_IQ2_XXS>(dev,rows,coherent);n+=fnx::run<GGML_TYPE_IQ2_S>(dev,rows,coherent);n+=fnx::run<GGML_TYPE_IQ3_XXS>(dev,rows,coherent);n+=fnx::run<GGML_TYPE_IQ3_S>(dev,rows,coherent);}}std::cout<<"PASS "<<n<<" exact fused GU/cache + down-copy checks\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

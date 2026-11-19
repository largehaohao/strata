#define FNX_KPART 99
#include "../src/kernels.cu"
#include <array>
#include <numeric>
#include <random>

namespace fnx {
std::vector<Q8Twin>& q8_twins(){static std::vector<Q8Twin> v;return v;}
void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
namespace {
struct Memory {
    std::vector<void*> p;
    template<class T>T* get(size_t n){void* v;ck(cudaMalloc(&v,n*sizeof(T)),"allocate");p.push_back(v);return (T*)v;}
    ~Memory(){for(void* v:p)cudaFree(v);}
};
__global__ void scale_exact(unsigned* errors){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=0x7c00u*256u)return;
    const unsigned q=i&255, scale=i>>8;
    const __half d=__ushort_as_half(scale);
    const uint2 h=attn_q8_four(q*0x01010101u,__half2half2(d),__half2half2(d));
    const unsigned ref=__half_as_ushort(__float2half_rn((float)(int8_t)q*__half2float(d)));
    if(h.x!=(ref|(ref<<16))||h.y!=h.x){atomicAdd(errors,1);atomicMin(errors+1,i);}
}
__global__ void fill_kv(__half* k,__half* v,KvPages g,int count){
    const int pos=blockIdx.x,dim=threadIdx.x;
    if(pos>=count)return;
    for(int part=0;part<2;++part){
        unsigned x=(pos*997+dim*37+part*517)*1664525u+1013904223u;
        const size_t at=kvq_row(g,pos)+part*HD+dim;
        ((int8_t*)k)[at]=(int8_t)(x>>24);((int8_t*)v)[at]=(int8_t)((x*17)>>24);
        if(!(dim&31)){
            const int i=(pos+dim/32+part)%19;
            k[kvq_scale(g,pos)+part*8+dim/32]=__float2half_rn(i==0?0.f:0.0001220703125f*(i+1));
            v[kvq_scale(g,pos)+part*8+dim/32]=__float2half_rn(i==1?0.f:0.00006103515625f*(i+1));
        }
    }
}
__global__ void fill_query(float* q,size_t n,int seed){
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)q[i]=seed==2?0.f:((int)((i*131+seed*719)%511)-255)*0.00390625f;
}
__global__ void fill_cells(int* cells,int* counts,const int* pos,int T,int seed){
    const int t=blockIdx.x,part=blockIdx.y,tid=threadIdx.x;
    const int ctx=kv_owned(*pos+t,part);
    const int n=seed==2?0:min(ctx,seed==1?QSA_MAXSEL/3:QSA_MAXSEL);
    if(!tid)counts[part*T+t]=seed==1&&*pos<=2048?-1:n;
    for(int j=tid;j<n;j+=blockDim.x)
        cells[((size_t)part*T+t)*QSA_MAXSEL+j]=(int)((long long)j*ctx/n);
}
__global__ void compare(const float* ref,const float* out,size_t n,unsigned* errors){
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&__float_as_uint(ref[i])!=__float_as_uint(out[i]))atomicAdd(errors,1);
    if(i<32&&__float_as_uint(out[n+i])!=0x5a5a5a5a)atomicAdd(errors+1,1);
}
void run(int dev){
    ck(cudaSetDevice(dev),"device");Memory mem;cudaStream_t st;ck(cudaStreamCreate(&st),"stream");
    auto* errors=mem.get<unsigned>(2);
    const unsigned init[2]={0,0xffffffff};ck(cudaMemcpy(errors,init,8,cudaMemcpyHostToDevice),"scale errors");
    scale_exact<<<(0x7c00u*256u+255)/256,256,0,st>>>(errors);ck(cudaStreamSynchronize(st),"scale exact");
    unsigned err[2];ck(cudaMemcpy(err,errors,8,cudaMemcpyDeviceToHost),"scale errors");
    if(err[0])throw std::runtime_error("packed scale mismatch count="+std::to_string(err[0])+" first="+std::to_string(err[1]));
    std::printf("device=%d all %u nonnegative finite half scales x 256 signed bytes exact\n",dev,0x7c00);std::fflush(stdout);
    constexpr int positions=262144,maxT=4096,npages=(positions+KVP_POS-1)/KVP_POS;
    auto* pt=mem.get<int>(npages);std::vector<int> host_pages(npages);std::iota(host_pages.begin(),host_pages.end(),0);
    std::mt19937 random(719);std::shuffle(host_pages.begin(),host_pages.end(),random);
    ck(cudaMemcpy(pt,host_pages.data(),npages*4,cudaMemcpyHostToDevice),"page table");
    KvPages g{pt,kvp_page(dev,true),kvp_rows(dev),1};
    auto* k=mem.get<__half>((size_t)npages*g.page);auto* v=mem.get<__half>((size_t)npages*g.page);
    fill_kv<<<npages*g.rows,HD,0,st>>>(k,v,g,npages*g.rows);
    auto* q=mem.get<float>((size_t)maxT*ATT_OWN*2*HD);
    auto* cells=mem.get<int>((size_t)2*maxT*QSA_MAXSEL);auto* counts=mem.get<int>(2*maxT);auto* pos=mem.get<int>(1);
    float* out[4];for(auto& p:out)p=mem.get<float>((size_t)2*maxT*ATT_OWN*AP_W+32);
    auto launch=[&](int mode,int T,int owner){
        switch(mode){
        case 0:k_pf_attn_head_direct<<<dim3(T,2),256,0,st>>>(q,k,v,g,pos,cells,counts,owner,T,0,out[0]);break;
        case 1:k_pf_attn_head_packed<1><<<dim3(T,2),256,0,st>>>(q,k,v,g,pos,cells,counts,owner,T,0,out[1]);break;
        case 2:k_pf_attn_head_packed<2><<<dim3(T,2),256,0,st>>>(q,k,v,g,pos,cells,counts,owner,T,0,out[2]);break;
        case 3:k_pf_attn_head_packed<3><<<dim3(T,2),256,0,st>>>(q,k,v,g,pos,cells,counts,owner,T,0,out[3]);break;
        }
    };
    int checks=0;
    for(int T:{1,8,15,32,256,4096})for(int start:{0,193,2048,74401,204799})for(int seed:{0,1,2}){
        if(T==4096&&(seed!=0||(start!=0&&start!=204799)))continue;
        ck(cudaMemcpy(pos,&start,4,cudaMemcpyHostToDevice),"position");
        const size_t n=(size_t)2*T*ATT_OWN*AP_W;
        fill_query<<<((size_t)T*ATT_OWN*2*HD+255)/256,256,0,st>>>(q,(size_t)T*ATT_OWN*2*HD,seed);
        fill_cells<<<dim3(T,2),256,0,st>>>(cells,counts,pos,T,seed);
        for(int owner:{0,1}){
            ck(cudaMemsetAsync(out[0],0x5a,(n+32)*4,st),"reference guard");launch(0,T,owner);
            for(int mode=1;mode<4;++mode){
                cudaGraph_t graph;cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");launch(mode,T,owner);
                ck(cudaStreamEndCapture(st,&graph),"capture end");ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
                for(bool replay:{false,true}){
                    ck(cudaMemsetAsync(out[mode],0x5a,(n+32)*4,st),"guard");ck(cudaMemsetAsync(errors,0,8,st),"errors");
                    if(replay)ck(cudaGraphLaunch(exec,st),"graph");else launch(mode,T,owner);
                    compare<<<(n+255)/256,256,0,st>>>(out[0],out[mode],n,errors);ck(cudaStreamSynchronize(st),"exact");
                    ck(cudaMemcpy(err,errors,8,cudaMemcpyDeviceToHost),"errors");
                    if(err[0]||err[1])throw std::runtime_error("attention mismatch mode="+std::to_string(mode)+" T="+std::to_string(T)+" start="+std::to_string(start)+" data="+std::to_string(err[0])+" guard="+std::to_string(err[1]));
                    ++checks;
                }
                ck(cudaGraphExecDestroy(exec),"destroy");ck(cudaGraphDestroy(graph),"destroy");
            }
        }
    }
    std::printf("device=%d PASS %d exact partial output/softmax/graph/guard comparisons page_halves=%zu\n",dev,checks,g.page);std::fflush(stdout);
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin),"event");ck(cudaEventCreate(&end),"event");
    for(int T:{1,32,4096})for(int start:{0,74401,204799}){
        ck(cudaMemcpy(pos,&start,4,cudaMemcpyHostToDevice),"position");
        fill_query<<<((size_t)T*ATT_OWN*2*HD+255)/256,256,0,st>>>(q,(size_t)T*ATT_OWN*2*HD,0);
        fill_cells<<<dim3(T,2),256,0,st>>>(cells,counts,pos,T,0);
        std::array<std::array<float,5>,4> times;
        for(int rep=-1;rep<5;++rep)for(int pass=0;pass<4;++pass){const int mode=(pass+std::max(rep,0))%4,iterations=T==4096?4:15;
            ck(cudaEventRecord(begin,st),"begin");for(int it=0;it<iterations;++it)launch(mode,T,0);
            ck(cudaEventRecord(end,st),"end");ck(cudaEventSynchronize(end),"wait");float ms;ck(cudaEventElapsedTime(&ms,begin,end),"time");
            if(rep>=0)times[mode][rep]=ms*1000.f/iterations;
        }
        for(auto& a:times)std::sort(a.begin(),a.end());
        std::printf("device=%d T=%d pos=%d reference_us=%.3f packed_us=%.3f page_us=%.3f combined_us=%.3f\n",dev,T,start,times[0][2],times[1][2],times[2][2],times[3][2]);std::fflush(stdout);
    }
    ck(cudaEventDestroy(begin),"destroy");ck(cudaEventDestroy(end),"destroy");ck(cudaStreamDestroy(st),"destroy");
}
}}
int main(){try{for(int d=0;d<2;++d)fnx::run(d);}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

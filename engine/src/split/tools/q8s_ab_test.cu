// Compare the existing fused R=2 kernel at the two-card 8192-row geometry
// against the production fork / Q8S + FP32 / join sequence.
#define FNX_KPART 7
#include "../src/kernels.cu"
#include <iostream>
#include <random>
namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> x; return x; }
void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e)); }
struct Streams {
    cudaStream_t main,side;
    cudaEvent_t fork,join;
};
template<int T,bool SPLIT>
void launch(bool fused,const Q8Segs& g,int rows,const block_q8_1* y,const int8_t* tq,const float* td,
            const float* W,const float* x,float* ab,int rf,const Streams& s) {
    if(fused) {
        k_q8s_ab<T,2,SPLIT><<<rf+(rows+15)/16,256,0,s.main>>>(g,rows,N,y,tq,td,W,x,ab,rf);
    } else {
        ck(cudaEventRecord(s.fork,s.main),"fork");
        ck(cudaStreamWaitEvent(s.side,s.fork,0),"fork wait");
        k_gemv_f32<T,4><<<rf,128,0,s.side>>>(W,x,ab,rf);
        ck(cudaEventRecord(s.join,s.side),"join");
        k_q8s<T,2,SPLIT,false><<<(rows+15)/16,256,0,s.main>>>(g,rows,N,y,tq,td);
        ck(cudaStreamWaitEvent(s.main,s.join,0),"join wait");
    }
}
void dispatch(int T,bool split,bool fused,const Q8Segs& g,int rows,const block_q8_1* y,const int8_t* tq,const float* td,
              const float* W,const float* x,float* ab,int rf,const Streams& s) {
    if(fused) {
        q8_twins().clear();
        if(split)q8_twins().push_back({(const char*)y,(size_t)MAXT*(N/32)*sizeof(block_q8_1),const_cast<int8_t*>(tq),const_cast<float*>(td)});
        const Q8S weights{const_cast<int8_t*>(g.q[0]),const_cast<uint16_t*>(g.d[0])};
        q8s_gemv_f32(weights,rows,N,y,g.out[0],W,x,ab,rf,T,s.main);
        return;
    }
#define CASE(t) case t: if(split)launch<t,true>(fused,g,rows,y,tq,td,W,x,ab,rf,s);else launch<t,false>(fused,g,rows,y,tq,td,W,x,ab,rf,s);break
    switch(T) { CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);CASE(9);CASE(10);CASE(11);CASE(12);CASE(13);CASE(14);CASE(15);CASE(16); }
#undef CASE
}
int run(int dev,int rows,int rf,bool bench) {
    constexpr int guard=64,copies=8,nb=N/32;
    const size_t qb=(size_t)rows*N,db=(size_t)rows*nb*2,wb=(size_t)rf*N*4;
    const size_t ob=((size_t)MAXT*rows+2*guard)*4,ab=((size_t)MAXT*rf+2*guard)*4;
    std::vector<void*> mem;
    auto alloc=[&](size_t n){void* p;ck(cudaMalloc(&p,n),"alloc");mem.push_back(p);return p;};
    auto* wq=(int8_t*)alloc(copies*qb);auto* wd=(uint16_t*)alloc(copies*db);
    auto* wf=(float*)alloc(copies*wb);auto* x=(float*)alloc((size_t)MAXT*N*4);
    auto* y=(block_q8_1*)alloc(MAXT*nb*sizeof(block_q8_1));
    auto* tq=(int8_t*)alloc((size_t)MAXT*N);auto* td=(float*)alloc(MAXT*nb*4);
    auto* ref=(float*)alloc(ob);auto* out=(float*)alloc(ob);
    auto* ar=(float*)alloc(ab);auto* ao=(float*)alloc(ab);
    std::mt19937 rng(260000+rows+rf);
    std::vector<int8_t> hq(qb);std::vector<uint16_t> hd(db/2);std::vector<float> hf(wb/4);
    for(auto& v:hq)v=int(rng()%255)-127;
    for(auto& v:hd)v=0x2000+rng()%0x1000;
    for(auto& v:hf)v=(int(rng()%4097)-2048)*0.001953125f;
    for(int c=0;c<copies;++c) {
        ck(cudaMemcpy(wq+c*qb,hq.data(),qb,cudaMemcpyHostToDevice),"weight q");
        ck(cudaMemcpy(wd+c*db/2,hd.data(),db,cudaMemcpyHostToDevice),"weight d");
        ck(cudaMemcpy(wf+c*wb/4,hf.data(),wb,cudaMemcpyHostToDevice),"weight f");
    }
    Streams s;ck(cudaStreamCreateWithFlags(&s.main,cudaStreamNonBlocking),"main stream");
    ck(cudaStreamCreateWithFlags(&s.side,cudaStreamNonBlocking),"side stream");
    ck(cudaEventCreateWithFlags(&s.fork,cudaEventDisableTiming),"fork event");
    ck(cudaEventCreateWithFlags(&s.join,cudaEventDisableTiming),"join event");
    std::vector<block_q8_1> hy(MAXT*nb);std::vector<int8_t> hq8(MAXT*N);
    std::vector<float> hx(MAXT*N),hsc(MAXT*nb);
    auto call=[&](int T,bool split,bool fused,int copy,bool reference){
        Q8Segs g{};
        for(int k=0;k<3;++k) {
            g.q[k]=wq+copy*qb;g.d[k]=wd+copy*db/2;
            g.out[k]=(reference?ref:out)+guard;g.end[k]=rows;g.stride[k]=rows;
        }
        dispatch(T,split,fused,g,rows,y,tq,td,wf+copy*wb/4,x,(reference?ar:ao)+guard,rf,s);
    };
    int checks=0;
    for(bool split:{false,true})for(int T=1;T<=MAXT;++T)for(int rep=0;rep<3;++rep) {
        for(size_t b=0;b<hy.size();++b) {
            hsc[b]=rep==2?0.f:(b%2?0.0625f:0.03125f);
            hy[b].ds=make_half2(__float2half(hsc[b]),__float2half(0.f));
            for(int i=0;i<32;++i)hy[b].qs[i]=hq8[b*32+i]=rep==2?0:int(rng()%255)-127;
        }
        for(auto& v:hx)v=rep==2?0.f:(int(rng()%2049)-1024)*0.00390625f;
        ck(cudaMemcpyAsync(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice,s.main),"packed input");
        ck(cudaMemcpyAsync(tq,hq8.data(),hq8.size(),cudaMemcpyHostToDevice,s.main),"split input");
        ck(cudaMemcpyAsync(td,hsc.data(),hsc.size()*4,cudaMemcpyHostToDevice,s.main),"split scales");
        ck(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,s.main),"float input");
        ck(cudaMemsetAsync(ref,0xa5,ob,s.main),"ref guard");ck(cudaMemsetAsync(ar,0xa5,ab,s.main),"ab ref guard");
        call(T,split,false,0,true);ck(cudaStreamSynchronize(s.main),"reference sync");
        std::vector<uint32_t> expected(ob/4),actual(ob/4),expected_ab(ab/4),actual_ab(ab/4);
        ck(cudaMemcpy(expected.data(),ref,ob,cudaMemcpyDeviceToHost),"read ref");
        ck(cudaMemcpy(expected_ab.data(),ar,ab,cudaMemcpyDeviceToHost),"read ab ref");
        cudaGraph_t graph;cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(s.main,cudaStreamCaptureModeRelaxed),"capture");call(T,split,true,1,false);
        ck(cudaStreamEndCapture(s.main,&graph),"end capture");ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
        for(bool replay:{false,true}) {
            ck(cudaMemsetAsync(out,0xa5,ob,s.main),"out guard");ck(cudaMemsetAsync(ao,0xa5,ab,s.main),"ab guard");
            if(replay)ck(cudaGraphLaunch(exec,s.main),"replay");else call(T,split,true,2,false);
            ck(cudaStreamSynchronize(s.main),"candidate sync");
            ck(cudaMemcpy(actual.data(),out,ob,cudaMemcpyDeviceToHost),"read out");
            ck(cudaMemcpy(actual_ab.data(),ao,ab,cudaMemcpyDeviceToHost),"read ab");
            if(actual!=expected||actual_ab!=expected_ab)throw std::runtime_error("fused Q8S/AB output or guard differs: T="+std::to_string(T));
            ++checks;
        }
        ck(cudaGraphExecDestroy(exec),"destroy exec");ck(cudaGraphDestroy(graph),"destroy graph");
        if(bench&&rows==8192&&rf==48&&split&&rep==0&&(T==1||T==3||T==5||T==7||T==16)) {
            cudaGraph_t graphs[2][copies];cudaGraphExec_t execs[2][copies];
            for(int m=0;m<2;++m)for(int c=0;c<copies;++c) {
                ck(cudaStreamBeginCapture(s.main,cudaStreamCaptureModeRelaxed),"bench capture");call(T,true,m,c,false);
                ck(cudaStreamEndCapture(s.main,&graphs[m][c]),"bench capture end");
                ck(cudaGraphInstantiate(&execs[m][c],graphs[m][c],nullptr,nullptr,0),"bench instantiate");
            }
            cudaEvent_t begin,end;ck(cudaEventCreate(&begin),"begin");ck(cudaEventCreate(&end),"end");
            for(int m:{0,1,1,0}) {
                for(int i=0;i<64;++i)ck(cudaGraphLaunch(execs[m][i%copies],s.main),"warm graph");
                ck(cudaEventRecord(begin,s.main),"begin time");
                for(int i=0;i<160;++i)ck(cudaGraphLaunch(execs[m][i%copies],s.main),"timed graph");
                ck(cudaEventRecord(end,s.main),"end time");ck(cudaEventSynchronize(end),"bench sync");
                float ms;ck(cudaEventElapsedTime(&ms,begin,end),"elapsed");
                std::cout<<"card "<<dev<<" T "<<T<<" fused "<<m<<" us "<<ms*6.25<<std::endl;
            }
            for(int m=0;m<2;++m)for(int c=0;c<copies;++c){cudaGraphExecDestroy(execs[m][c]);cudaGraphDestroy(graphs[m][c]);}
            cudaEventDestroy(begin);cudaEventDestroy(end);
        }
    }
    cudaEventDestroy(s.fork);cudaEventDestroy(s.join);cudaStreamDestroy(s.side);cudaStreamDestroy(s.main);
    q8_twins().clear();for(void* p:mem)cudaFree(p);return checks;
}
}
int main(int argc,char** argv) {try {
    if(argc>2||(argc==2&&std::string(argv[1])!="--bench"))throw std::runtime_error("usage: split_q8s_ab_test [--bench]");
    setenv("FNX_Q8S_ROW_GROUP","2",1);
    int checks=0;
    for(int dev=0;dev<2;++dev){fnx::ck(cudaSetDevice(dev),"device");checks+=fnx::run(dev,4097,48,false);checks+=fnx::run(dev,8192,48,argc==2);checks+=fnx::run(dev,8197,137,false);}
    std::cout<<"PASS "<<checks<<" exact fused Q8S/AB direct/graph/guard checks\n";
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

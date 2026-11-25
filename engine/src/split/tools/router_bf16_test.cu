#define FNX_KPART 99
#include "../src/kernels.cu"
#include <cmath>
#include <iostream>
#include <random>
namespace fnx {
std::vector<Q8Twin>& q8_twins(){static std::vector<Q8Twin>x;return x;}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+cudaGetErrorString(e));}
template<int T> void launch(bool packed,const void*w,const float*x,float*l,int*i,float*p,Job*g,Job*d,int*n,unsigned*c,cudaStream_t st,unsigned* error,
                            int* wl1=nullptr,int* wlm=nullptr,int* n_wl=nullptr){
    const dim3 grid((NEXP+1+RR_ROWS-1)/RR_ROWS);
    if(packed)k_router_route<T,true><<<grid,RR_THREADS,0,st>>>((const uint16_t*)w,x,l,i,p,g,d,n,c,ExpertCache{},0,nullptr,ShexpUp{},error,wl1,wlm,n_wl);
    else k_router_route<T><<<grid,RR_THREADS,0,st>>>((const float*)w,x,l,i,p,g,d,n,c,ExpertCache{},0,nullptr,ShexpUp{},error,wl1,wlm,n_wl);
}
int run(int dev,bool bench){
    constexpr int pad=64,cap=MAXT*TOPK,copies=64,L=NEXP+1;
    const size_t wn=(size_t)L*N,lb=(MAXT*L+2*pad)*4,pb=(cap+2*pad)*4,jb=(cap+2)*sizeof(Job);
    const size_t w1b=(cap+2*pad)*sizeof(int),wmb=(cap/2+2*pad)*sizeof(int),nwb=(2+2*pad)*sizeof(int);
    std::vector<uint16_t>hw(wn);std::vector<float>fw(wn),hx(MAXT*N);
    std::mt19937 rng(20261104);
    for(size_t i=0;i<wn;++i){hw[i]=(rng()%2?0x8000:0)|(0x3b00+rng()%0x300);uint32_t u=(uint32_t)hw[i]<<16;std::memcpy(&fw[i],&u,4);}
    std::vector<void*>mem;auto alloc=[&](size_t n){void*p;ck(cudaMalloc(&p,n),"alloc");mem.push_back(p);return p;};
    auto*wf=(float*)alloc(copies*wn*4);auto*wb=(uint16_t*)alloc(copies*wn*2);
    for(int i=0;i<copies;++i){ck(cudaMemcpy(wf+i*wn,fw.data(),wn*4,cudaMemcpyHostToDevice),"float weights");ck(cudaMemcpy(wb+i*wn,hw.data(),wn*2,cudaMemcpyHostToDevice),"BF16 weights");}
    auto*x=(float*)alloc(hx.size()*4);auto*logits=(float*)alloc(lb);auto*indices=(int*)alloc(pb);auto*prob=(float*)alloc(pb);
    auto*gu=(Job*)alloc(jb);auto*dn=(Job*)alloc(jb);auto*nj=(int*)alloc(4);auto*counter=(unsigned*)alloc(4);auto*error=(unsigned*)alloc(4);
    auto*wl1=(int*)alloc(w1b);auto*wlm=(int*)alloc(wmb);auto*n_wl=(int*)alloc(nwb);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");ck(cudaMemset(counter,0,4),"counter");
    cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");int checks=0,copy=0;
    auto reset=[&](){for(auto v:{std::pair<void*,size_t>{logits,lb},{indices,pb},{prob,pb},{gu,jb},{dn,jb},{wl1,w1b},{wlm,wmb},{n_wl,nwb}})ck(cudaMemsetAsync(v.first,0xa5,v.second,st),"guard");};
    for(int T=1;T<=MAXT;++T){
        auto call=[&](bool packed,bool with_wl){int k=copy++%copies;const void*w=packed?(const void*)(wb+k*wn):(const void*)(wf+k*wn);switch(T){
#define DO(t) case t:launch<t>(packed,w,x,logits+pad,indices+pad,prob+pad,gu+1,dn+1,nj,counter,st,error,with_wl?wl1+pad:nullptr,with_wl?wlm+pad:nullptr,with_wl?n_wl+pad:nullptr);break
            DO(1);DO(2);DO(3);DO(4);DO(5);DO(6);DO(7);DO(8);DO(9);DO(10);DO(11);DO(12);DO(13);DO(14);DO(15);DO(16);
#undef DO
        }};
        for(int rep=0;rep<4;++rep){
            for(auto&v:hx)v=rep==3?NAN:rep==2?0.f:(int(rng()%2049)-1024)/2048.f;
            ck(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st),"activation");
            std::vector<unsigned char>rl(lb),ri(pb),rp(pb),gl(lb),gi(pb),gp(pb);
            std::vector<unsigned char>gw1(w1b),gwm(wmb),gnw(nwb);
            std::vector<Job>rg(cap+2),rd(cap+2),gg(cap+2),gd(cap+2);int rn,gn;
            ck(cudaMemsetAsync(error,0,4,st),"reset router error");
            reset();call(false,false);ck(cudaStreamSynchronize(st),"reference wait");
            ck(cudaMemcpy(rl.data(),logits,lb,cudaMemcpyDeviceToHost),"ref logits");ck(cudaMemcpy(ri.data(),indices,pb,cudaMemcpyDeviceToHost),"ref ids");ck(cudaMemcpy(rp.data(),prob,pb,cudaMemcpyDeviceToHost),"ref probs");
            ck(cudaMemcpy(rg.data(),gu,jb,cudaMemcpyDeviceToHost),"ref jobs");ck(cudaMemcpy(rd.data(),dn,jb,cudaMemcpyDeviceToHost),"ref jobs");ck(cudaMemcpy(&rn,nj,4,cudaMemcpyDeviceToHost),"ref count");
            cudaGraph_t graph;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"capture");call(true,true);ck(cudaStreamEndCapture(st,&graph),"capture end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"instantiate");
            for(bool replay:{false,true}){
                reset();if(replay)ck(cudaGraphLaunch(ex,st),"replay");else call(true,true);ck(cudaStreamSynchronize(st),"wait");
                ck(cudaMemcpy(gl.data(),logits,lb,cudaMemcpyDeviceToHost),"logits");ck(cudaMemcpy(gi.data(),indices,pb,cudaMemcpyDeviceToHost),"ids");ck(cudaMemcpy(gp.data(),prob,pb,cudaMemcpyDeviceToHost),"probs");
                ck(cudaMemcpy(gg.data(),gu,jb,cudaMemcpyDeviceToHost),"jobs");ck(cudaMemcpy(gd.data(),dn,jb,cudaMemcpyDeviceToHost),"jobs");ck(cudaMemcpy(&gn,nj,4,cudaMemcpyDeviceToHost),"count");
                ck(cudaMemcpy(gw1.data(),wl1,w1b,cudaMemcpyDeviceToHost),"wl1");ck(cudaMemcpy(gwm.data(),wlm,wmb,cudaMemcpyDeviceToHost),"wlm");ck(cudaMemcpy(gnw.data(),n_wl,nwb,cudaMemcpyDeviceToHost),"n_wl");
                unsigned flags;
                ck(cudaMemcpy(&flags,error,4,cudaMemcpyDeviceToHost),"router error");
                if (flags != (rep == 3 ? 32u : 0u)) throw std::runtime_error("invalid router error bit");
                if (rep == 3) {
                    for (int j = 1; j <= gn; ++j)
                        if (gg[j].w < 0 || gg[j].w >= NEXP || gg[j].n > T) throw std::runtime_error("invalid fallback job");
                    std::vector<float> pw(T * TOPK);
                    std::memcpy(pw.data(), gp.data() + pad * 4, pw.size() * sizeof(float));
                    for (float wv : pw)
                        if (!std::isfinite(wv)) throw std::runtime_error("non-finite router weight on NaN fallback");
                }
                if(rl!=gl||ri!=gi||rp!=gp||rn!=gn)throw std::runtime_error("router logits/prob/ids/count mismatch");
                for(auto pair:{std::make_pair(&rg,&gg),std::make_pair(&rd,&gd)}){
                    auto&ref=*pair.first;auto&got=*pair.second;
                    for(int j=0;j<cap+2;++j){
                        if(j==0||j>rn){if(std::memcmp(&ref[j],&got[j],sizeof(Job)))throw std::runtime_error("job guard");continue;}
                        if(ref[j].w!=got[j].w||ref[j].n!=got[j].n)throw std::runtime_error("job header");
                        std::vector<std::pair<int,int>>r,v;
                        for(int k=0;k<ref[j].n;++k){r.emplace_back(ref[j].y[k],ref[j].out[k]);v.emplace_back(got[j].y[k],got[j].out[k]);}
                        std::sort(r.begin(),r.end());std::sort(v.begin(),v.end());if(r!=v)throw std::runtime_error("job routes");
                        for(int k=ref[j].n;k<MAXT;++k)if(ref[j].y[k]!=got[j].y[k]||ref[j].out[k]!=got[j].out[k])throw std::runtime_error("job tail guard");
                    }
                }
                std::vector<int> ew1, ewm;
                for (int j = 0; j < rn; ++j) {
                    if (rg[j + 1].n == 1) ew1.push_back(j);
                    else ewm.push_back(j);
                }
                int got_n[2];
                std::memcpy(got_n, gnw.data() + pad * sizeof(int), sizeof(got_n));
                if (got_n[0] != (int) ew1.size() || got_n[1] != (int) ewm.size() || got_n[0] + got_n[1] != gn)
                    throw std::runtime_error("split worklist count mismatch");
                auto chk_wl = [&](const std::vector<unsigned char>& raw, const std::vector<int>& exp, const char* name) {
                    const size_t lo = pad * sizeof(int), hi = lo + exp.size() * sizeof(int);
                    for (size_t i = 0; i < raw.size(); ++i)
                        if ((i < lo || i >= hi) && raw[i] != 0xa5)
                            throw std::runtime_error(std::string(name) + " guard overwrite");
                    if (!exp.empty()) {
                        std::vector<int> got_v(exp.size());
                        std::memcpy(got_v.data(), raw.data() + lo, exp.size() * sizeof(int));
                        if (got_v != exp) throw std::runtime_error(std::string(name) + " index mismatch");
                    }
                };
                chk_wl(gw1, ew1, "wl1");
                chk_wl(gwm, ewm, "wlm");
                for (size_t i = 0; i < gnw.size(); ++i)
                    if ((i < pad * sizeof(int) || i >= (pad + 2) * sizeof(int)) && gnw[i] != 0xa5)
                        throw std::runtime_error("n_wl guard overwrite");
                unsigned count;ck(cudaMemcpy(&count,counter,4,cudaMemcpyDeviceToHost),"counter reset");if(count)throw std::runtime_error("counter not reset");++checks;
            }
            cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
            for(int probe_rows:{137,513}) {
                auto probe=[&](bool packed){switch(T){
#define PROBE(t) case t: if(probe_rows<=160){if(packed)k_gemv_f32<t,4,true><<<probe_rows,128,0,st>>>(wb,x,logits+pad,probe_rows);else k_gemv_f32<t,4><<<probe_rows,128,0,st>>>(wf,x,logits+pad,probe_rows);}else{if(packed)k_gemv_f32<t,1,true><<<(probe_rows+7)/8,256,0,st>>>(wb,x,logits+pad,probe_rows);else k_gemv_f32<t,1><<<(probe_rows+7)/8,256,0,st>>>(wf,x,logits+pad,probe_rows);}break
                    PROBE(1);PROBE(2);PROBE(3);PROBE(4);PROBE(5);PROBE(6);PROBE(7);PROBE(8);PROBE(9);PROBE(10);PROBE(11);PROBE(12);PROBE(13);PROBE(14);PROBE(15);PROBE(16);
#undef PROBE
                }};
                ck(cudaMemsetAsync(logits,0xa5,lb,st),"probe guard");probe(false);ck(cudaStreamSynchronize(st),"probe ref wait");ck(cudaMemcpy(rl.data(),logits,lb,cudaMemcpyDeviceToHost),"probe ref");
                ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"probe capture");probe(true);ck(cudaStreamEndCapture(st,&graph),"probe end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"probe graph");
                for(bool replay:{false,true}){
                    ck(cudaMemsetAsync(logits,0xa5,lb,st),"probe guard");if(replay)ck(cudaGraphLaunch(ex,st),"probe replay");else probe(true);
                    ck(cudaStreamSynchronize(st),"probe wait");ck(cudaMemcpy(gl.data(),logits,lb,cudaMemcpyDeviceToHost),"probe output");
                    unsigned flags;
                ck(cudaMemcpy(&flags,error,4,cudaMemcpyDeviceToHost),"router error");
                if (flags != (rep == 3 ? 32u : 0u)) throw std::runtime_error("invalid router error bit");
                if (rep == 3) {
                    for (int j = 1; j <= gn; ++j)
                        if (gg[j].w < 0 || gg[j].w >= NEXP || gg[j].n > T) throw std::runtime_error("invalid fallback job");
                    std::vector<float> pw(T * TOPK);
                    std::memcpy(pw.data(), gp.data() + pad * 4, pw.size() * sizeof(float));
                    for (float wv : pw)
                        if (!std::isfinite(wv)) throw std::runtime_error("non-finite router weight on NaN fallback");
                }
                if(rl!=gl)throw std::runtime_error("probe GEMV or guard mismatch");++checks;
                }
                cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
            }
            if(bench&&rep==0&&(T==1||T==3||T==7||T==16))for(bool packed:{false,true,true,false}){
                ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed),"time capture");for(int i=0;i<copies;++i)call(packed,false);ck(cudaStreamEndCapture(st,&graph),"time end");ck(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0),"time graph");
                for(int i=0;i<8;++i)ck(cudaGraphLaunch(ex,st),"warm");ck(cudaEventRecord(a,st),"start");for(int i=0;i<8;++i)ck(cudaGraphLaunch(ex,st),"time launch");ck(cudaEventRecord(b,st),"end");ck(cudaEventSynchronize(b),"time wait");float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");
                std::cout<<"card "<<dev<<" T "<<T<<" BF16 "<<packed<<" us "<<ms*1000/(8*copies)<<std::endl;cudaGraphExecDestroy(ex);cudaGraphDestroy(graph);
            }
        }
    }
    cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(st);for(void*p:mem)cudaFree(p);return checks;
}
}
int main(int argc,char**argv){try{if(argc>2||(argc==2&&std::string(argv[1])!="--bench"))throw std::runtime_error("usage: split_router_bf16_test [--bench]");const bool bench=argc==2;int n=0;for(int d=0;d<2;++d){fnx::ck(cudaSetDevice(d),"device");n+=fnx::run(d,bench);}std::cout<<"PASS "<<n<<" exact BF16 router/logit/top/job/probe/guard/direct/graph checks\n";}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}

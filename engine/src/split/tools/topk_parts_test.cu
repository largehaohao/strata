#include "../src/kernels.hpp"
#include "../src/env.hpp"
#include "../src/owned.hpp"
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnx {
#include "../src/topk.cuh"
#include "../src/topk_parts.cuh"
}

namespace fnx {
namespace {std::mutex test_owned_mutex;std::vector<std::function<void()>> test_owned;}
void own(std::function<void()> release) {
    std::lock_guard<std::mutex> lock(test_owned_mutex);test_owned.push_back(std::move(release));
}
std::vector<std::function<void()>> take_owned() {
    std::lock_guard<std::mutex> lock(test_owned_mutex);auto result=std::move(test_owned);test_owned.clear();return result;
}
}
using namespace fnx;
void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
struct Memory {
    std::vector<void*> allocations;
    template<class T> T* make(size_t n) {
        T* p; check(cudaMalloc((void**)&p, n * sizeof(T)), "allocate"); allocations.push_back(p); return p;
    }
    ~Memory() { for (void* p : allocations) cudaFree(p); }
};
void merge(int parts, int T, float* v, int* i, float* out_v, int* out_i, cudaStream_t st) {
    switch (parts) {
        case 2: topk_merge_launch<2>(T,v,i,out_v,out_i,st,topk_exact_fast_mode()); break;
        case 4: topk_merge_launch<4>(T,v,i,out_v,out_i,st,topk_exact_fast_mode()); break;
        case 8: topk_merge_launch<8>(T,v,i,out_v,out_i,st,topk_exact_fast_mode()); break;
        case 16: topk_merge_launch<16>(T,v,i,out_v,out_i,st,topk_exact_fast_mode()); break;
        case 32: topk_merge_launch<32>(T,v,i,out_v,out_i,st,topk_exact_fast_mode()); break;
        default: throw std::runtime_error("invalid partition count");
    }
}
int exact(int device) {
    int checks = 0;
    constexpr int pad = 64, maxT = 16, stride = 17;
    check(cudaSetDevice(device), "set device");
    for (int n : {1025, 4097, 53149, 124160}) {
        Memory m;
        const size_t cells = (size_t) maxT * n + 2 * pad, outputs = maxT * TOPK_N + 2 * pad;
        const size_t temps = maxT * 32 * TOPK_N + 2 * pad;
        auto* a = m.make<float>(cells); auto* b = m.make<float>(cells);
        auto* av = m.make<float>(outputs); auto* bv = m.make<float>(outputs);
        auto* ai = m.make<int>(outputs); auto* bi = m.make<int>(outputs);
        auto* tmpv = m.make<float>(temps); auto* tmpi = m.make<int>(temps);
        auto* ea = m.make<unsigned>(1); auto* eb = m.make<unsigned>(1);
        auto* pn = m.make<int>(maxT); auto* pt = m.make<int>(maxT * stride); auto* pc = m.make<int>(maxT * stride);
        auto* par = m.make<float>(maxT * 3);
        std::vector<float> input(cells), pa(maxT * 3), ha(cells), hb(cells), va(outputs), vb(outputs);
        std::vector<int> nt(maxT), toks(maxT * stride), cnts(maxT * stride), ia(outputs), ib(outputs);
        cudaStream_t st; check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        constexpr int v0 = 19013;
        for (int T : {1, 3, 8, 16}) for (int pattern = 0; pattern < 7; ++pattern) for (int penalized : {0, 1}) {
            std::memset(input.data(), 0xA5, cells * 4);
            uint32_t state = 12345;
            for (int row = 0; row < T; ++row) {
                for (int j = 0; j < n; ++j) {
                    state = state * 1664525u + 1013904223u;
                    float x = (int(state >> 8) % 32768 - 16384) * (1.f / 2048);
                    if (pattern == 1) x = 3.f;
                    if (pattern == 2) x = j & 1 ? -0.f : 0.f;
                    if (pattern == 3) x = float((j * 17 + row) % 101) / 16.f;
                    if (pattern == 4) x = float(j) / 4096.f;
                    if (pattern == 6) {
                        const uint32_t edge[] = {1u,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x7f7fffffu,0xff7fffffu};
                        const uint32_t bits=edge[j%8];std::memcpy(&x,&bits,4);
                    }
                    if (pattern == 5 && j == n / 3) { uint32_t bits = 0x7fc00007u; std::memcpy(&x, &bits, 4); }
                    input[pad + (size_t) row * n + j] = x;
                }
                nt[row] = penalized ? stride : 0;
                pa[3 * row] = 1.125f; pa[3 * row + 1] = .0625f; pa[3 * row + 2] = .125f;
                for (int j = 0; j < stride; ++j) {
                    toks[row * stride + j] = v0 + j * (n - 1) / (stride - 1);
                    cnts[row * stride + j] = j % 5 + 1;
                }
            }
            check(cudaMemcpyAsync(pn, nt.data(), maxT * 4, cudaMemcpyHostToDevice, st), "counts");
            check(cudaMemcpyAsync(pt, toks.data(), maxT * stride * 4, cudaMemcpyHostToDevice, st), "tokens");
            check(cudaMemcpyAsync(pc, cnts.data(), maxT * stride * 4, cudaMemcpyHostToDevice, st), "penalties");
            check(cudaMemcpyAsync(par, pa.data(), maxT * 3 * 4, cudaMemcpyHostToDevice, st), "parameters");
            for (int parts : {2, 4, 8, 16, 32}) {
                const int width = (n + parts - 1) / parts;
                if (n - (parts - 1) * width < TOPK_N) continue;
                check(cudaMemcpyAsync(a, input.data(), cells * 4, cudaMemcpyHostToDevice, st), "reference input");
                check(cudaMemcpyAsync(b, input.data(), cells * 4, cudaMemcpyHostToDevice, st), "candidate input");
                for (auto* p : {av, bv}) check(cudaMemsetAsync(p, 0xA5, outputs * 4, st), "value guards");
                for (auto* p : {ai, bi}) check(cudaMemsetAsync(p, 0xA5, outputs * 4, st), "id guards");
                check(cudaMemsetAsync(tmpv, 0xA5, temps * 4, st), "temporary guards");
                check(cudaMemsetAsync(tmpi, 0xA5, temps * 4, st), "temporary id guards");
                check(cudaMemsetAsync(ea, 0, 4, st), "reference error"); check(cudaMemsetAsync(eb, 0, 4, st), "candidate error");
                k_topk_rows<<<T, TOPK_THREADS, 0, st>>>(a + pad, n, v0, pt, pc, pn, stride, par, av + pad, ai + pad, ea);
                auto launch = [&] {
                    topk_partition_launch(b+pad,n,T,v0,parts,pt,pc,pn,stride,par,tmpv+pad,tmpi+pad,eb,st,topk_exact_fast_mode());
                    merge(parts, T, tmpv + pad, tmpi + pad, bv + pad, bi + pad, st);
                };
                if (pattern == 3) {
                    check(cudaStreamSynchronize(st), "pre-capture");
                    cudaGraph_t graph; cudaGraphExec_t exec;
                    check(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture"); launch();
                    check(cudaStreamEndCapture(st, &graph), "end capture");
                    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
                    check(cudaGraphLaunch(exec, st), "graph replay"); check(cudaStreamSynchronize(st), "graph wait");
                    check(cudaGraphExecDestroy(exec), "destroy execution"); check(cudaGraphDestroy(graph), "destroy graph");
                } else launch();
                check(cudaGetLastError(), "launch"); check(cudaStreamSynchronize(st), "wait");
                unsigned erra, errb;
                check(cudaMemcpy(&erra, ea, 4, cudaMemcpyDeviceToHost), "reference error read");
                check(cudaMemcpy(&errb, eb, 4, cudaMemcpyDeviceToHost), "candidate error read");
                if (erra != errb || bool(errb & 16) != (pattern == 5)) throw std::runtime_error("error semantics mismatch");
                check(cudaMemcpy(va.data(), av, outputs * 4, cudaMemcpyDeviceToHost), "reference values");
                check(cudaMemcpy(vb.data(), bv, outputs * 4, cudaMemcpyDeviceToHost), "candidate values");
                check(cudaMemcpy(ia.data(), ai, outputs * 4, cudaMemcpyDeviceToHost), "reference ids");
                check(cudaMemcpy(ib.data(), bi, outputs * 4, cudaMemcpyDeviceToHost), "candidate ids");
                if (std::memcmp(va.data(), vb.data(), outputs * 4) || std::memcmp(ia.data(), ib.data(), outputs * 4)) {
                    std::fprintf(stderr, "mismatch device%d n%d T%d pattern%d penalty%d parts%d\n", device, n, T, pattern, penalized, parts);
                    throw std::runtime_error("score-bit/id/guard mismatch");
                }
                check(cudaMemcpy(ha.data(), a, cells * 4, cudaMemcpyDeviceToHost), "reference logits");
                check(cudaMemcpy(hb.data(), b, cells * 4, cudaMemcpyDeviceToHost), "candidate logits");
                if (std::memcmp(ha.data(), hb.data(), cells * 4)) throw std::runtime_error("penalized logits mismatch");
                std::vector<unsigned char> temp(temps * 4);
                for (void* p : {(void*)tmpv, (void*)tmpi}) {
                    check(cudaMemcpy(temp.data(), p, temp.size(), cudaMemcpyDeviceToHost), "scratch guards");
                    for (size_t j = 0; j < temp.size(); ++j)
                        if ((j < pad * 4 || j >= (size_t)(pad + T * parts * TOPK_N) * 4) && temp[j] != 0xA5)
                            throw std::runtime_error("scratch guard overwritten");
                }
                ++checks;
            }
        }
        check(cudaStreamDestroy(st), "destroy stream");
    }
    std::printf("{\"kind\":\"exact\",\"device\":%d,\"checks\":%d}\n", device, checks); std::fflush(stdout);
    return checks;
}
int dispatcher_lifetime(int device) {
    check(cudaSetDevice(device),"dispatch device");
    Memory memory;
    constexpr int T=2, maximum=53149;
    auto* x=memory.make<float>(T*maximum);
    auto* ref_v=memory.make<float>(T*TOPK_N);auto* out_v=memory.make<float>(T*TOPK_N);
    auto* ref_i=memory.make<int>(T*TOPK_N);auto* out_i=memory.make<int>(T*TOPK_N);
    auto* pn=memory.make<int>(T);auto* par=memory.make<float>(3*T);auto* error=memory.make<unsigned>(1);
    std::vector<float> input(T*maximum);
    for(size_t i=0;i<input.size();++i)input[i]=(int((i*1337+31)%65521)-32760)*0x1p-10f;
    check(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice),"dispatch input");
    check(cudaMemset(pn,0,T*4),"dispatch no penalties");check(cudaMemset(par,0,T*3*4),"dispatch parameters");
    check(cudaMemset(error,0,4),"dispatch error");
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"dispatch stream");
    int checks=0;
    for(int n:{maximum,1025}) {
        k_topk_rows<<<T,TOPK_THREADS,0,stream>>>(x,n,7,pn,pn,pn,0,par,ref_v,ref_i,error);
        check(cudaStreamSynchronize(stream),"reference ready");
        auto launch=[&] {
            const bool fast=topk_parts(x,n,T,7,pn,pn,pn,0,par,out_v,out_i,error,stream);
            if(!fast)k_topk_rows<<<T,TOPK_THREADS,0,stream>>>(x,n,7,pn,pn,pn,0,par,out_v,out_i,error);
            check(cudaGetLastError(),"dispatch launch");return fast;
        };
        auto compare=[&] {
            check(cudaStreamSynchronize(stream),"dispatch wait");
            std::vector<float> a(T*TOPK_N),b(T*TOPK_N);std::vector<int> ia(T*TOPK_N),ib(T*TOPK_N);
            check(cudaMemcpy(a.data(),ref_v,a.size()*4,cudaMemcpyDeviceToHost),"reference values");
            check(cudaMemcpy(b.data(),out_v,b.size()*4,cudaMemcpyDeviceToHost),"dispatch values");
            check(cudaMemcpy(ia.data(),ref_i,ia.size()*4,cudaMemcpyDeviceToHost),"reference ids");
            check(cudaMemcpy(ib.data(),out_i,ib.size()*4,cudaMemcpyDeviceToHost),"dispatch ids");
            if(std::memcmp(a.data(),b.data(),a.size()*4)||ia!=ib)throw std::runtime_error("dispatcher score/id mismatch");
            ++checks;
        };
        const int parts=topk_parts_mode();
        const bool geometry=parts>0&&n-(parts-1)*((n+parts-1)/parts)>=TOPK_N;
        // A fresh registry falls back during capture. Existing scratch can be
        // reused by later captures with another valid vocabulary size.
        cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"first capture");
        const bool first_fast=launch();
        check(cudaStreamEndCapture(stream,&graph),"first capture end");
        if(first_fast!=(n!=maximum&&geometry))throw std::runtime_error("capture used stale or newly allocated scratch");
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"first instantiate");
        check(cudaGraphLaunch(exec,stream),"first replay");compare();
        check(cudaGraphExecDestroy(exec),"first exec destroy");check(cudaGraphDestroy(graph),"first graph destroy");
        const bool eager_fast=launch();compare();
        const bool expected_fast=geometry;
        if(eager_fast!=expected_fast)throw std::runtime_error("dispatcher fallback policy mismatch");
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"warm capture");
        const bool warm_fast=launch();
        check(cudaStreamEndCapture(stream,&graph),"warm capture end");
        if(warm_fast!=expected_fast)throw std::runtime_error("cached scratch not reused in capture");
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"warm instantiate");
        for(int replay=0;replay<3;++replay){check(cudaGraphLaunch(exec,stream),"warm replay");compare();}
        check(cudaGraphExecDestroy(exec),"warm exec destroy");check(cudaGraphDestroy(graph),"warm graph destroy");
    }
    check(cudaStreamSynchronize(stream),"release wait");
    auto releases=fnx::take_owned();
    if(releases.size()!=(topk_parts_mode()?1u:0u))throw std::runtime_error("scratch release not registered");
    for(auto it=releases.rbegin();it!=releases.rend();++it)(*it)();
    if(!fnx::take_owned().empty())throw std::runtime_error("scratch owner not cleared");
    check(cudaStreamDestroy(stream),"dispatch stream destroy");return checks;
}
int main() {
    try { for (int device : {0, 1}) { exact(device);dispatcher_lifetime(device);dispatcher_lifetime(device); } return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}

// Byte-for-byte oracle for decode staging, including changed routing on replay.
#include "kernels.hpp"
#include <iostream>
#include <vector>
namespace fnx {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}
#include "moe_stage.cuh"
using namespace fnx;

int main() {
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "devices");
    if (devices < 2) throw std::runtime_error("this validation requires two physical devices");
    for (int dev = 0; dev < 2; ++dev) {
        ck(cudaSetDevice(dev), "device");
        for (size_t stride : {size_t(256), size_t(1056), size_t(32800)}) {
          for (int mode = 0; mode < 3; ++mode) {
            constexpr int count = MAXT * TOPK;
            const size_t bytes = NEXP * stride, scratch_bytes = count * stride + 256;
            char *host = nullptr, *gpu = nullptr, *scratch = nullptr;
            const char **ep = nullptr, **staged_ep = nullptr;
            Job* jobs = nullptr;
            int* njobs = nullptr;
            unsigned char* hot = nullptr;
            ck(cudaHostAlloc((void**)&host, bytes, cudaHostAllocMapped), "host");
            void* mapped = nullptr;
            ck(cudaHostGetDevicePointer(&mapped, host, 0), "mapped");
            ck(cudaMalloc((void**)&gpu, bytes), "gpu");
            ck(cudaMalloc((void**)&scratch, scratch_bytes), "scratch");
            ck(cudaMalloc((void**)&ep, NEXP * sizeof(char*)), "ep");
            ck(cudaMalloc((void**)&staged_ep, NEXP * sizeof(char*)), "staged ep");
            ck(cudaMalloc((void**)&jobs, count * sizeof(Job)), "jobs");
            ck(cudaMalloc((void**)&njobs, sizeof(int)), "njobs");
            ck(cudaMalloc((void**)&hot, NEXP), "hot mask");
            cudaStream_t stream;
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
            std::vector<const char*> pointers(NEXP);
            for (int e = 0; e < NEXP; ++e) pointers[e] = (e % 2 ? (char*) mapped : gpu) + e * stride;
            ck(cudaMemcpy(ep, pointers.data(), NEXP * sizeof(char*), cudaMemcpyHostToDevice), "pointers");
            std::vector<unsigned char> hot_mask(NEXP);
            for (int e = 0; e < NEXP; ++e) hot_mask[e] = e % 2 == 0;
            ck(cudaMemcpy(hot, hot_mask.data(), NEXP, cudaMemcpyHostToDevice), "hot mask copy");
            DW w{}; w.p = gpu; w.ep = mode == 1 ? nullptr : ep;
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
            stage_moe(w, stride, jobs, njobs, count, scratch, staged_ep, stream, mode == 2 ? hot : nullptr);
            ck(cudaStreamEndCapture(stream, &graph), "end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            for (int rep = 0; rep < 24; ++rep) {
                for (size_t i = 0; i < bytes; ++i) host[i] = (char)((i * 17 + (i / stride) * 23 + rep * 31) % 251);
                ck(cudaMemcpy(gpu, host, bytes, cudaMemcpyHostToDevice), "weights");
                const int n = rep % 4 == 0 ? 0 : (rep % 4 == 1 ? 1 : (rep % 4 == 2 ? 37 : count));
                std::vector<Job> js(count);
                for (int j = 0; j < count; ++j) js[j].w = (j * 7 + rep * 19) % NEXP;
                ck(cudaMemcpy(jobs, js.data(), count * sizeof(Job), cudaMemcpyHostToDevice), "routing");
                ck(cudaMemcpy(njobs, &n, sizeof(int), cudaMemcpyHostToDevice), "job count");
                ck(cudaMemset(scratch, 0xA5, scratch_bytes), "guard");
                if (rep % 2) ck(cudaGraphLaunch(exec, stream), "replay");
                else stage_moe(w, stride, jobs, njobs, count, scratch, staged_ep, stream, mode == 2 ? hot : nullptr);
                ck(cudaStreamSynchronize(stream), "wait");
                std::vector<unsigned char> got(scratch_bytes);
                ck(cudaMemcpy(got.data(), scratch, scratch_bytes, cudaMemcpyDeviceToHost), "read scratch");
                std::vector<const char*> at(NEXP);
                ck(cudaMemcpy(at.data(), staged_ep, NEXP * sizeof(char*), cudaMemcpyDeviceToHost), "read pointers");
                for (int j = 0; j < n; ++j) {
                    const int e = js[j].w;
                    const bool skip = mode == 2 && hot_mask[e];
                    if (at[e] != (skip ? pointers[e] : scratch + j * stride)) throw std::runtime_error("stale pointer");
                    for (size_t k = 0; k < stride; ++k)
                        if (got[j * stride + k] != (skip ? 0xA5 : (unsigned char)host[e * stride + k])) throw std::runtime_error("weight mismatch");
                }
                for (size_t i = n * stride; i < got.size(); ++i)
                    if (got[i] != 0xA5) throw std::runtime_error("scratch overrun or inactive job copied");
            }
            ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
            ck(cudaStreamDestroy(stream), "destroy stream");
            for (void* ptr : {(void*)gpu, (void*)scratch, (void*)ep, (void*)staged_ep, (void*)jobs, (void*)njobs, (void*)hot}) ck(cudaFree(ptr), "free");
            ck(cudaFreeHost(host), "free host");
          }
        }
    }
    std::cout << "MoE staging: two physical GPUs, 432 direct/graph checks passed\n";
}

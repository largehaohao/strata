#include "static_prefill_copy.cuh"
#include "static_prefill_stage.hpp"
#include "strata/prefill/hip_batch_copy.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
static void check(bool ok) { if (!ok) throw std::runtime_error("selective prefill copy mismatch"); }

int main() {
    int devices = 0, checks = 0, batches = 0, fallbacks = 0;
    ck(cudaGetDeviceCount(&devices), "devices");
    for (int dev = 0; dev < std::min(devices, 2); ++dev) {
        ck(cudaSetDevice(dev), "device");
        cudaStream_t stream;
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        cudaStream_t copy_stream;
        cudaEvent_t producer, copied;
        ck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "copy stream");
        ck(cudaEventCreateWithFlags(&producer, cudaEventDisableTiming), "producer event");
        ck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "copy event");
        for (size_t gs : {size_t(16), size_t(4096), size_t(627200)})
            for (size_t ds : {size_t(32), size_t(460800)}) {
                constexpr int E = 9;
                const size_t bytes = E * (gs + ds), guard = 32;
                char *source = nullptr, *mapped = nullptr, *stage = nullptr;
                ck(cudaHostAlloc(&source, bytes, cudaHostAllocMapped), "source");
                void* alias = nullptr;
                ck(cudaHostGetDevicePointer(&alias, source, 0), "mapped");
                mapped = static_cast<char*>(alias);
                ck(cudaMalloc(&stage, bytes + 2 * guard), "stage");
                for (size_t i = 0; i < bytes; ++i) source[i] = char((i * 17 + i / 19) % 251);
                std::vector<const char*> srcg(E), srcd(E), dstg(E), dstd(E);
                for (int e = 0; e < E; ++e) {
                    srcg[e] = mapped + e * gs;
                    srcd[e] = mapped + E * gs + e * ds;
                    dstg[e] = e % 3 == 0 ? srcg[e] : stage + guard + e * gs;
                    dstd[e] = e % 3 == 0 ? srcd[e] : stage + guard + E * gs + e * ds;
                }
                const char **sg, **sd, **dg, **dd;
                int *counts;
                unsigned char* ready;
                unsigned long long* moved;
                for (auto p : {&sg, &sd, &dg, &dd}) ck(cudaMalloc(p, E * sizeof(char*)), "table");
                ck(cudaMalloc(&counts, E * sizeof(int)), "counts");
                ck(cudaMalloc(&ready, E), "ready");
                ck(cudaMalloc(&moved, sizeof(*moved)), "moved");
                for (auto pair : {std::make_pair(sg, srcg.data()), std::make_pair(sd, srcd.data()),
                                  std::make_pair(dg, dstg.data()), std::make_pair(dd, dstd.data())})
                    ck(cudaMemcpy(pair.first, pair.second, E * sizeof(char*), cudaMemcpyHostToDevice), "table upload");
                for (int mode = 0; mode < 3; ++mode)
                for (int cycle = 0; cycle < 2; ++cycle) {
                    ck(cudaMemsetAsync(stage, 0x5a, bytes + 2 * guard, stream), "stage clear");
                    ck(cudaMemsetAsync(ready, 0, E, stream), "ready clear");
                    ck(cudaMemsetAsync(moved, 0, sizeof(*moved), stream), "bytes clear");
                    std::vector<char> expected(bytes + 2 * guard, char(0x5a)), actual(expected.size());
                    std::vector<unsigned char> seen(E, 0), got_ready(E);
                    std::vector<unsigned char> dma_ready(E, 0);
                    unsigned long long expected_bytes = 0;
                    unsigned long long dma_bytes = 0;
                    for (int turn = 0; turn < 6; ++turn) {
                        std::vector<int> selected(E);
                        for (int e = 0; e < E; ++e) selected[e] = turn && ((e + turn + cycle) % 4) ? e + 1 : 0;
                        ck(cudaMemcpyAsync(counts, selected.data(), E * sizeof(int), cudaMemcpyHostToDevice, stream), "routes");
                        if (mode == 1) {
                            auto plan = fnx::routed_prefill_transfers(srcg.data(), srcd.data(), dstg.data(), dstd.data(),
                                selected.data(), dma_ready, gs, ds);
                            ck(cudaEventRecord(producer, stream), "producer");
                            ck(cudaStreamWaitEvent(copy_stream, producer, 0), "stage reuse");
                            std::vector<void*> dsts, srcs;
                            std::vector<size_t> sizes;
                            for (const auto& cp : plan.copies) {
                                dsts.push_back(cp.dst); srcs.push_back(const_cast<char*>(cp.src)); sizes.push_back(cp.n);
                            }
                            bool used_batch = false;
                            ck(strata::prefill::hip_batch_copy::enqueue(dsts.data(), srcs.data(), sizes.data(), sizes.size(),
                                copy_stream, nullptr, &used_batch), "routed DMA");
                            if (!sizes.empty()) { ++batches; if (!used_batch) ++fallbacks; }
                            ck(cudaEventRecord(copied, copy_stream), "copied");
                            ck(cudaStreamWaitEvent(stream, copied, 0), "consumer");
                            for (int e : plan.experts) dma_ready[e] = 1;
                            dma_bytes += plan.bytes;
                        } else if (mode == 2) {
                            ck(cudaEventRecord(producer, stream), "GPU producer");
                            ck(cudaStreamWaitEvent(copy_stream, producer, 0), "GPU stage reuse");
                            fnx::static_prefill_copy(sg, sd, dg, dd, counts, ready, gs, ds, E, moved, copy_stream, 1);
                            ck(cudaEventRecord(copied, copy_stream), "GPU copied");
                            ck(cudaStreamWaitEvent(stream, copied, 0), "GPU consumer");
                        } else fnx::static_prefill_copy(sg, sd, dg, dd, counts, ready, gs, ds, E, moved, stream);
                        ck(cudaGetLastError(), "copy launch");
                        ck(cudaStreamSynchronize(stream), "copy done");
                        for (int e = 0; e < E; ++e) {
                            if (!selected[e] || seen[e] || e % 3 == 0) continue;
                            std::memcpy(expected.data() + guard + e * gs, source + e * gs, gs);
                            std::memcpy(expected.data() + guard + E * gs + e * ds, source + E * gs + e * ds, ds);
                            seen[e] = 1;
                            expected_bytes += gs + ds;
                        }
                        unsigned long long got_bytes = 0;
                        ck(cudaMemcpy(actual.data(), stage, actual.size(), cudaMemcpyDeviceToHost), "bytes back");
                        if (mode == 1) got_ready = dma_ready, got_bytes = dma_bytes;
                        else {
                            ck(cudaMemcpy(got_ready.data(), ready, E, cudaMemcpyDeviceToHost), "ready back");
                            ck(cudaMemcpy(&got_bytes, moved, sizeof(got_bytes), cudaMemcpyDeviceToHost), "counter back");
                        }
                        check(actual == expected && got_ready == seen && got_bytes == expected_bytes);
                        ++checks;
                    }
                }
                for (auto p : {sg, sd, dg, dd}) ck(cudaFree(p), "free table");
                ck(cudaFree(counts), "free counts"); ck(cudaFree(ready), "free ready"); ck(cudaFree(moved), "free counter");
                ck(cudaFree(stage), "free stage"); ck(cudaFreeHost(source), "free source");
            }
        ck(cudaStreamDestroy(stream), "destroy stream");
        ck(cudaStreamDestroy(copy_stream), "destroy copy stream");
        ck(cudaEventDestroy(producer), "destroy producer");
        ck(cudaEventDestroy(copied), "destroy copied");
    }
    check(devices >= 2);
    std::cout << "selective prefill copy: " << checks << " GPU byte/route/reuse/guard cases passed\n";
    std::cout << "routed DMA: " << batches << " nonempty batches, " << fallbacks << " individual fallbacks\n";
}

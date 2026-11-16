// Compare fused and precomputed MoE activations for every decode window.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <GgmlType TA>
void run_type(int dev) {
    constexpr int experts = 48, rows = 2560, cols = 320, pairs = MAXT * TOPK;
    constexpr size_t row_bytes = cols / qtraits<TA>::qk * qtraits<TA>::bs;
    std::vector<unsigned char> weights(experts * rows * row_bytes);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (i * 13 + i / 71) % 256;
    for (size_t i = 0; i < weights.size(); i += qtraits<TA>::bs) {
        const uint16_t scale = 0x2400;
        std::memcpy(weights.data() + i, &scale, sizeof scale);
    }
    std::vector<float> gu(pairs * cols * 2);
    for (size_t i = 0; i < gu.size(); ++i) gu[i] = ((int) (i * 23 % 107) - 53) * .03f;
    char* dw = nullptr;
    float *dg = nullptr, *out = nullptr;
    Job* dj = nullptr;
    int* dn = nullptr;
    const size_t output_bytes = pairs * rows * sizeof(float) + 256;
    ck(cudaMalloc((void**) &dw, weights.size()), "weights");
    ck(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice), "weights copy");
    ck(cudaMalloc((void**) &dg, gu.size() * sizeof(float)), "activations");
    ck(cudaMemcpy(dg, gu.data(), gu.size() * sizeof(float), cudaMemcpyHostToDevice), "activations copy");
    ck(cudaMalloc((void**) &out, output_bytes), "output");
    ck(cudaMalloc((void**) &dj, experts * sizeof(Job)), "jobs");
    ck(cudaMalloc((void**) &dn, sizeof(int)), "count");
    const MoeW w{dw, rows * row_bytes, row_bytes, rows, cols, nullptr};
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ck(cudaDeviceSynchronize(), "initial uploads");
    std::vector<unsigned char> ref(output_bytes), got(output_bytes);
    for (int T = 1; T <= MAXT; ++T) {
        auto launch = [&](bool pre) {
            constexpr int rpb = DN_THREADS / (cols / 32);
            const dim3 grid((rows + rpb - 1) / rpb, experts);
            const size_t shared = T * DN_THREADS * sizeof(float);
            if (pre) {
                k_moe_h<<<(T * TOPK * (cols / 32) + 7) / 8, 256, 0, stream>>>(dg, cols, T * TOPK);
                k_moe_down<TA, true><<<grid, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
            } else {
                k_moe_down<TA, false><<<grid, DN_THREADS, shared, stream>>>(w, dj, dn, dg, out);
            }
        };
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
        launch(true);
        ck(cudaStreamEndCapture(stream, &graph), "capture end");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int rep = 0; rep < 8; ++rep) {
            const int counts[] = {0, 1, 7, std::min(48, T * TOPK)};
            const int count = counts[rep % 4], total = std::min(T * TOPK, count * T);
            std::vector<Job> jobs(experts);
            int cursor = 0;
            for (int j = 0; j < count; ++j) {
                jobs[j].w = (j * 7 + rep * 11) % experts;
                jobs[j].n = total / count + (j < total % count);
                for (int t = 0; t < jobs[j].n; ++t)
                    jobs[j].y[t] = jobs[j].out[t] = (cursor++ * 37 + rep * 13) % (T * TOPK);
            }
            ck(cudaMemcpyAsync(dj, jobs.data(), jobs.size() * sizeof(Job), cudaMemcpyHostToDevice, stream), "jobs");
            ck(cudaMemcpyAsync(dn, &count, sizeof count, cudaMemcpyHostToDevice, stream), "count");
            ck(cudaMemsetAsync(out, 0xA5, output_bytes, stream), "reference guard");
            launch(false);
            ck(cudaStreamSynchronize(stream), "reference wait");
            ck(cudaMemcpy(ref.data(), out, output_bytes, cudaMemcpyDeviceToHost), "reference read");
            ck(cudaMemsetAsync(out, 0xA5, output_bytes, stream), "candidate guard");
            if (rep >= 4) ck(cudaGraphLaunch(exec, stream), "replay");
            else launch(true);
            ck(cudaStreamSynchronize(stream), "candidate wait");
            ck(cudaMemcpy(got.data(), out, output_bytes, cudaMemcpyDeviceToHost), "candidate read");
            if (ref != got)
                throw std::runtime_error("MoE down mismatch: card " + std::to_string(dev) + " type " + std::to_string(TA) +
                                         " T " + std::to_string(T) + " rep " + std::to_string(rep));
        }
        ck(cudaGraphExecDestroy(exec), "destroy exec");
        ck(cudaGraphDestroy(graph), "destroy graph");
    }
    ck(cudaStreamDestroy(stream), "free stream");
    for (void* p : {(void*) dw, (void*) dg, (void*) out, (void*) dj, (void*) dn}) ck(cudaFree(p), "free");
}
} // namespace fnx

int main() {
    int count = 0;
    fnx::ck(cudaGetDeviceCount(&count), "devices");
    if (count < 2) throw std::runtime_error("two physical devices required");
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        fnx::run_type<fnx::T_IQ4_NL>(dev);
        fnx::run_type<fnx::T_Q4_0>(dev);
    }
    std::cout << "MoE down prequant: 512 exact direct/graph comparisons passed\n";
}

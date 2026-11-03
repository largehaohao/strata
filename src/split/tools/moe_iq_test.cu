// Compare the decoded IQ weight reuse to the original kernel, including graph replay.
// Synthetic packed weights have valid finite scales and exercise every codeword.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <ggml_type TA>
void launch_test(int T, int cap, const MoeW& w, const Job* jobs, const int* njobs,
                 const block_q8_1* x, float* out, cudaStream_t stream) {
    const dim3 grid((w.nrows + 3) / 4, 48);
#define GU_CASE(t) case t: \
    if (cap) k_moe_gu_iq<TA, t><<<grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out); \
    else k_moe_gu<TA, 32, t><<<grid, MMVQ_THREADS, 0, stream>>>(w, jobs, njobs, x, out); \
    break
    switch (T) {
        GU_CASE(1); GU_CASE(2); GU_CASE(3); GU_CASE(4);
        GU_CASE(5); GU_CASE(6); GU_CASE(7); GU_CASE(8);
        GU_CASE(9); GU_CASE(10); GU_CASE(11); GU_CASE(12);
        GU_CASE(13); GU_CASE(14); GU_CASE(15); GU_CASE(16);
        default: throw std::runtime_error("invalid test window");
    }
#undef GU_CASE
    ck(cudaGetLastError(), "gate/up launch");
}

template <ggml_type TA>
void run_type(int dev) {
    constexpr int experts = 48, cols = 2560, rows = 641;
    constexpr size_t row_bytes = cols / ggml_cuda_type_traits<TA>::qk * ggml_cuda_type_traits<TA>::bs;
    const size_t weight_bytes = experts * rows * row_bytes;
    std::vector<unsigned char> weights(weight_bytes);
    for (size_t i = 0; i < weight_bytes; ++i) weights[i] = (i * 13 + (i >> 7) * 17) & 255;
    for (size_t i = 0; i < weight_bytes; i += ggml_cuda_type_traits<TA>::bs) {
        const uint16_t scale = 0x2400; // finite half: all four formats put d first
        std::memcpy(weights.data() + i, &scale, sizeof scale);
    }
    std::vector<block_q8_1> x(MAXT * cols / 32);
    for (size_t i = 0; i < x.size(); ++i) {
        int sum = 0;
        for (int k = 0; k < 32; ++k) sum += x[i].qs[k] = (int)((i * 7 + k * 11) % 255) - 127;
        x[i].ds = make_half2(__float2half(0.02f), __float2half(sum * 0.02f));
    }
    char* dw = nullptr;
    block_q8_1* dx = nullptr;
    Job* dj = nullptr;
    int* dn = nullptr;
    float *baseline = nullptr, *candidate = nullptr;
    const size_t output_bytes = (size_t) experts * MAXT * rows * sizeof(float) + 256;
    ck(cudaMalloc((void**)&dw, weight_bytes), "weights");
    ck(cudaMalloc((void**)&dx, x.size() * sizeof(block_q8_1)), "activations");
    ck(cudaMalloc((void**)&dj, experts * sizeof(Job)), "jobs");
    ck(cudaMalloc((void**)&dn, sizeof(int)), "count");
    ck(cudaMalloc((void**)&baseline, output_bytes), "baseline");
    ck(cudaMalloc((void**)&candidate, output_bytes), "candidate");
    ck(cudaMemcpy(dw, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "weights copy");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "activation copy");
    const MoeW w{dw, rows * row_bytes, row_bytes, rows, cols, nullptr};
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    std::vector<unsigned char> ref(output_bytes), got(output_bytes);
    for (int T = 1; T <= MAXT; ++T) {
        for (int cap : {1}) {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
            launch_test<TA>(T, cap, w, dj, dn, dx, candidate, stream);
            ck(cudaStreamEndCapture(stream, &graph), "end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            constexpr int counts[] = {0, 1, 7, experts};
            for (int rep = 0; rep < 8; ++rep) {
                const int count = counts[rep % 4];
                std::vector<Job> jobs(experts);
                for (int j = 0; j < experts; ++j) {
                    jobs[j].w = (j * 7 + rep * 11) % experts;
                    jobs[j].n = 1 + (j + rep) % T;
                    for (int t = 0; t < jobs[j].n; ++t) {
                        jobs[j].y[t] = (t + rep) % T;
                        jobs[j].out[t] = j * MAXT + t;
                    }
                }
                ck(cudaMemcpy(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice), "job copy");
                ck(cudaMemcpy(dn, &count, sizeof(int), cudaMemcpyHostToDevice), "count copy");
                ck(cudaMemset(baseline, 0xA5, output_bytes), "baseline guard");
                ck(cudaMemset(candidate, 0xA5, output_bytes), "candidate guard");
                launch_test<TA>(T, 0, w, dj, dn, dx, baseline, stream);
                if ((rep / 4) % 2) ck(cudaGraphLaunch(exec, stream), "graph replay");
                else launch_test<TA>(T, cap, w, dj, dn, dx, candidate, stream);
                ck(cudaStreamSynchronize(stream), "wait");
                ck(cudaMemcpy(ref.data(), baseline, output_bytes, cudaMemcpyDeviceToHost), "reference read");
                ck(cudaMemcpy(got.data(), candidate, output_bytes, cudaMemcpyDeviceToHost), "candidate read");
                if (ref != got) throw std::runtime_error("gate/up bit mismatch or guard overwrite: type " +
                    std::to_string(TA) + " T " + std::to_string(T) + " cap " + std::to_string(cap));
            }
            ck(cudaGraphExecDestroy(exec), "destroy exec");
            ck(cudaGraphDestroy(graph), "destroy graph");
        }
    }
    // Same live jobs for each timing arm; events bracket 40 ordered launches.
    cudaEvent_t start, end;
    ck(cudaEventCreate(&start), "start"); ck(cudaEventCreate(&end), "end");
    for (int count : {10, 24, 48}) {
        ck(cudaMemcpy(dn, &count, sizeof count, cudaMemcpyHostToDevice), "timing count");
        std::vector<Job> jobs(experts);
        for (int j = 0; j < count; ++j) {
            jobs[j].w = j;
            jobs[j].n = 60 / count + (j < 60 % count); // 6 tokens x top-10 picks
            for (int t = 0; t < jobs[j].n; ++t) jobs[j].y[t] = (j + t) % 6, jobs[j].out[t] = j * MAXT + t;
        }
        ck(cudaMemcpy(dj, jobs.data(), experts * sizeof(Job), cudaMemcpyHostToDevice), "timing jobs");
        for (int cap : {0, 1}) {
            for (int i = 0; i < 4; ++i) launch_test<TA>(6, cap, w, dj, dn, dx, candidate, stream);
            ck(cudaEventRecord(start, stream), "record start");
            for (int i = 0; i < 40; ++i) launch_test<TA>(6, cap, w, dj, dn, dx, candidate, stream);
            ck(cudaEventRecord(end, stream), "record end");
            ck(cudaEventSynchronize(end), "event wait");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, start, end), "elapsed");
            std::cout << "card " << dev << " type " << TA << " jobs " << count << " reuse " << cap << " us " << ms * 1000 / 40 << '\n';
        }
    }
    ck(cudaEventDestroy(start), "free event"); ck(cudaEventDestroy(end), "free event");
    ck(cudaStreamDestroy(stream), "free stream");
    for (void* p : {(void*)dw, (void*)dx, (void*)dj, (void*)dn, (void*)baseline, (void*)candidate}) ck(cudaFree(p), "free");
}
} // namespace fnx

int main() {
    int count = 0;
    fnx::ck(cudaGetDeviceCount(&count), "devices");
    if (count < 2) throw std::runtime_error("two physical devices required");
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        fnx::run_type<GGML_TYPE_IQ2_S>(dev);
        fnx::run_type<GGML_TYPE_IQ3_XXS>(dev);
        fnx::run_type<GGML_TYPE_IQ3_S>(dev);
        fnx::run_type<GGML_TYPE_IQ4_XS>(dev);
    }
    std::cout << "MoE gate/up: 1024 exact direct/graph checks passed\n";
}

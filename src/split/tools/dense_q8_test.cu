// Compare optional row grouping against the original four-row large-matrix path.
#define FNX_KPART 6
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> twins; return twins; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <bool SEG> int run_dense(int rows, int cols) {
    constexpr int guard = 64;
    const int nb = cols / 32;
    std::vector<void*> allocations;
    auto alloc = [&](size_t bytes) {
        void* p = nullptr; ck(cudaMalloc(&p, bytes), "allocate"); allocations.push_back(p); return p;
    };
    std::vector<int8_t> weights((size_t) rows * cols), x(MAXT * cols);
    std::vector<uint16_t> scales((size_t) rows * nb);
    std::vector<float> xd(MAXT * nb);
    std::vector<block_q8_1> y(MAXT * nb);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (int) ((i * 19 + i / 31) % 255) - 127;
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = 0x2000 + (i % 8) * 0x80;
    auto* wq = (int8_t*) alloc(weights.size());
    auto* wd = (uint16_t*) alloc(scales.size() * 2);
    auto* tq = (int8_t*) alloc(x.size());
    auto* td = (float*) alloc(xd.size() * sizeof(float));
    auto* yq = (block_q8_1*) alloc(y.size() * sizeof(block_q8_1));
    ck(cudaMemcpy(wq, weights.data(), weights.size(), cudaMemcpyHostToDevice), "weights");
    ck(cudaMemcpy(wd, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice), "scales");
    Q8Segs original{}, candidate{};
    std::vector<float*> refs, outputs;
    std::vector<size_t> sizes;
    int start = 0;
    for (int s = 0; s < (SEG ? 3 : 1); ++s) {
        const int n = SEG ? (s == 0 ? rows / 2 + 3 : s == 1 ? rows / 4 + 1 : rows - start) : rows;
        const int stride = n + 17;
        const size_t bytes = (MAXT * stride + 2 * guard) * sizeof(float);
        auto* ref = (float*) alloc(bytes);
        auto* out = (float*) alloc(bytes);
        refs.push_back(ref); outputs.push_back(out); sizes.push_back(bytes);
        original.q[s] = candidate.q[s] = wq + (size_t) start * cols;
        original.d[s] = candidate.d[s] = wd + (size_t) start * nb;
        original.out[s] = ref + guard; candidate.out[s] = out + guard;
        original.stride[s] = candidate.stride[s] = stride;
        start += n; original.end[s] = candidate.end[s] = start;
    }
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ck(cudaDeviceSynchronize(), "uploads");
    ck(cudaMemsetAsync(yq, 0, y.size() * sizeof(block_q8_1), stream), "initial packed input");
    ck(cudaMemsetAsync(tq, 0, x.size(), stream), "initial split input");
    ck(cudaMemsetAsync(td, 0, xd.size() * sizeof(float), stream), "initial split scales");
    int checks = 0;
    for (bool split : {false, true}) {
        q8_twins().clear();
        if (split) q8_twins().push_back({(const char*) yq, y.size() * sizeof(block_q8_1), tq, td});
        for (int T = 1; T <= MAXT; ++T) {
            auto launch = [&] { q8s_run<SEG>(candidate, rows, cols, yq, T, stream); };
            // Initialize the environment dispatch before capture; the same production launcher is captured below.
            launch(); ck(cudaStreamSynchronize(stream), "initialize dispatch");
            cudaGraph_t graph; cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
            launch();
            ck(cudaStreamEndCapture(stream, &graph), "capture end");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            for (int rep = 0; rep < 3; ++rep) {
                for (size_t b = 0; b < y.size(); ++b) {
                    const uint32_t bits = b % 2 ? 0x2c00 : 0x2800; // exact 1/16 or 1/32, upper half unused
                    std::memcpy(&y[b].ds, &bits, sizeof(bits));
                    xd[b] = b % 2 ? 0.0625f : 0.03125f;
                    for (int i = 0; i < 32; ++i)
                        y[b].qs[i] = x[b * 32 + i] = rep == 2 ? 0 : (int) ((b * 23 + i * 11 + rep * 37) % 255) - 127;
                }
                ck(cudaMemcpyAsync(yq, y.data(), y.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice, stream), "packed activations");
                ck(cudaMemcpyAsync(tq, x.data(), x.size(), cudaMemcpyHostToDevice, stream), "split activations");
                ck(cudaMemcpyAsync(td, xd.data(), xd.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "activation scales");
                for (size_t s = 0; s < refs.size(); ++s) ck(cudaMemsetAsync(refs[s], 0xA5, sizes[s], stream), "reference guards");
                q8s_launch<4, SEG>(original, rows, cols, yq, T, stream);
                for (bool replay : {false, true}) {
                    for (size_t s = 0; s < outputs.size(); ++s) ck(cudaMemsetAsync(outputs[s], 0xA5, sizes[s], stream), "output guards");
                    if (replay) ck(cudaGraphLaunch(exec, stream), "replay"); else launch();
                    ck(cudaStreamSynchronize(stream), "compare wait");
                    for (size_t s = 0; s < refs.size(); ++s) {
                        std::vector<unsigned char> expected(sizes[s]), got(sizes[s]);
                        ck(cudaMemcpy(expected.data(), refs[s], sizes[s], cudaMemcpyDeviceToHost), "reference");
                        ck(cudaMemcpy(got.data(), outputs[s], sizes[s], cudaMemcpyDeviceToHost), "actual");
                        if (expected != got) throw std::runtime_error("dense Q8 row grouping or guard mismatch");
                        const int n = original.end[s] - (s ? original.end[s - 1] : 0);
                        for (int t = 0; t < T; ++t) for (int r = 0; r < n; ++r) {
                            float v; std::memcpy(&v, got.data() + (guard + t * original.stride[s] + r) * sizeof(float), sizeof(v));
                            if (!std::isfinite(v) || (rep == 2 && v != 0)) throw std::runtime_error("dense Q8 invalid/zero output");
                        }
                    }
                    ++checks;
                }
            }
            ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
        }
    }
    q8_twins().clear(); ck(cudaStreamDestroy(stream), "destroy stream");
    for (void* p : allocations) ck(cudaFree(p), "free");
    return checks;
}
}  // namespace fnx

int main() {
    try {
        int devices = 0; fnx::ck(cudaGetDeviceCount(&devices), "device count");
        if (devices < 2) throw std::runtime_error("requires two visible devices");
        int total = 0;
        for (int dev = 0; dev < 2; ++dev) {
            fnx::ck(cudaSetDevice(dev), "device");
            int checks = 0;
            for (auto shape : {std::make_pair(8197, 2560), std::make_pair(8192, 320)}) {
                checks += fnx::run_dense<false>(shape.first, shape.second);
                checks += fnx::run_dense<true>(shape.first, shape.second);
            }
            total += checks; std::cout << "card " << dev << ": " << checks << " exact direct/graph checks\n";
        }
        std::cout << "PASS " << total << " dense Q8 checks\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

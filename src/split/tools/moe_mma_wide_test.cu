// Compare independent narrow/wide instantiations, including split-K jobs,
// partial weight tiles, pointer tables, graph replay and untouched output bytes.
#include "../src/moe_mma.cu"
#include <cstring>
#include <iostream>

namespace fnx {
void moe_gate_up_mma_wide(const DW&, size_t, const BigJob*, const int*, int, const uint16_t*, float*, void*);
void moe_down_mma_wide(const DW&, size_t, const BigJob*, const int*, int, const uint16_t*, float*, void*);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
struct Format { int type, block, bytes; const char* name; };

static int run(const Format& fmt, int cols, int rows, bool micro) {
    const int experts = micro ? 256 : 3, capacity = micro ? 512 : 3;
    const size_t rb = (size_t) cols / fmt.block * fmt.bytes, stride = rb * rows;
    const size_t xsize = (size_t) capacity * PF_JOB * cols;
    const size_t obytes = (size_t) capacity * PF_JOB * rows * sizeof(float) + 256;
    std::vector<void*> owned;
    auto alloc = [&](size_t n) { void* p; ck(cudaMalloc(&p, n), "allocate"); owned.push_back(p); return p; };
    char* weights = (char*) alloc(experts * stride);
    auto* input = (uint16_t*) alloc(xsize * 2);
    auto* jobs = (BigJob*) alloc(capacity * sizeof(BigJob));
    auto* count = (int*) alloc(sizeof(int));
    auto* out = (float*) alloc(obytes);
    auto* ep = (const char**) alloc(experts * sizeof(char*));
    std::vector<const char*> pointers(experts);
    for (int e = 0; e < experts; ++e) pointers[e] = weights + (size_t) ((e * 5) % experts) * stride;
    ck(cudaMemcpy(ep, pointers.data(), experts * sizeof(char*), cudaMemcpyHostToDevice), "pointers");
    DW w{}; w.p = weights; w.ep = ep; w.type = fmt.type; w.n_in = cols; w.n_out = rows; w.row_bytes = rb;
    std::vector<unsigned char> raw(experts * stride), ref(obytes), got(obytes);
    std::vector<__half> x(xsize);
    std::vector<BigJob> hj(capacity);
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    auto launch = [&](bool wide) {
        if (wide) moe_gate_up_mma_wide(w, stride, jobs, count, capacity, input, out, st);
        else moe_gate_up_mma(w, stride, jobs, count, capacity, input, out, st);
        ck(cudaGetLastError(), "launch");
    };
    int checked = 0;
    const std::vector<int> seeds = micro ? std::vector<int>{1} : std::vector<int>{0, 1};
    const std::vector<int> job_counts = micro ? std::vector<int>{capacity} : std::vector<int>{0, 1, 3};
    const std::vector<int> lengths = micro ? std::vector<int>{PF_JOB} : std::vector<int>{0, 1, 15, 16, 17, 31, 32, 33, PF_JOB - 1, PF_JOB};
    for (int seed : seeds) {
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = (i * 13 + (i >> 7) * 17 + seed * 23) & 255;
        for (size_t i = 0; i < raw.size(); i += fmt.bytes) {
            const uint16_t edge[] = {0, 1, 3, 0x200, 0x3ff, 0x400, 0x1400, 0x2400, 0x3c00, 0x5400, 0x5800};
            const uint16_t magnitude = micro ? (uint16_t) (0x2400 + ((i / fmt.bytes) % 16) * 19)
                                            : edge[(i / fmt.bytes) % (sizeof(edge) / sizeof(edge[0]))];
            const uint16_t d = magnitude | ((i / fmt.bytes) % 7 == 0 ? 0x8000 : 0);
            std::memcpy(raw.data() + i, &d, 2);
        }
        for (size_t i = 0; i < xsize; ++i)
            x[i] = __float2half(seed ? .013f * ((int) ((i * 7 + (i >> 6) * 11) % 511) - 255) : 0.f);
        ck(cudaMemcpyAsync(weights, raw.data(), raw.size(), cudaMemcpyHostToDevice, st), "weights");
        ck(cudaMemcpyAsync(input, x.data(), xsize * 2, cudaMemcpyHostToDevice, st), "input");
        for (int nj : job_counts) for (int n : lengths) {
            for (int j = 0; j < capacity; ++j) {
                hj[j].w = (j + seed) % experts;
                hj[j].n = j % 2 ? std::min(PF_JOB, n + 1) : n;
                for (int k = 0; k < hj[j].n; ++k) {
                    hj[j].y[k] = (j * 13 + k * 7) % (capacity * PF_JOB);
                    hj[j].out[k] = j * PF_JOB + k;
                }
            }
            ck(cudaMemcpyAsync(jobs, hj.data(), capacity * sizeof(BigJob), cudaMemcpyHostToDevice, st), "jobs");
            ck(cudaMemcpyAsync(count, &nj, sizeof(nj), cudaMemcpyHostToDevice, st), "count");
            ck(cudaStreamSynchronize(st), "prepared");
            for (bool graph : {false, true}) {
                for (bool wide : {false, true}) {
                    ck(cudaMemsetAsync(out, 0xB3, obytes, st), "guards");
                    if (graph) {
                        cudaGraph_t g; cudaGraphExec_t ex;
                        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "capture"); launch(wide);
                        ck(cudaStreamEndCapture(st, &g), "end capture");
                        ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                        for (int k = 0; k < 3; ++k) ck(cudaGraphLaunch(ex, st), "replay");
                        ck(cudaStreamSynchronize(st), "replay done");
                        ck(cudaGraphExecDestroy(ex), "free graph exec"); ck(cudaGraphDestroy(g), "free graph");
                    } else launch(wide);
                    ck(cudaStreamSynchronize(st), "result ready");
                    ck(cudaMemcpy((wide ? got : ref).data(), out, obytes, cudaMemcpyDeviceToHost), "result");
                }
                if (got != ref) throw std::runtime_error(std::string("wide mismatch ") + fmt.name + " K=" +
                    std::to_string(cols) + " M=" + std::to_string(rows) + " n=" + std::to_string(n));
                ++checked;
            }
        }
    }
    if (micro) {
        cudaEvent_t a, b; ck(cudaEventCreate(&a), "start event"); ck(cudaEventCreate(&b), "end event");
        float ms[2] = {};
        for (int round = 0; round < 3; ++round) for (int i = 0; i < 2; ++i) {
            const bool wide = (i + round) % 2;
            for (int k = 0; k < 3; ++k) launch(wide);
            ck(cudaEventRecord(a, st), "start");
            for (int k = 0; k < 8; ++k) launch(wide);
            ck(cudaEventRecord(b, st), "end"); ck(cudaEventSynchronize(b), "timed");
            float dt; ck(cudaEventElapsedTime(&dt, a, b), "elapsed"); ms[wide] += dt / 24;
        }
        std::cout << "micro " << fmt.name << " K=" << cols << " M=" << rows << " jobs=" << capacity
                  << " narrow_ms=" << ms[0] << " wide_ms=" << ms[1] << '\n';
        ck(cudaEventDestroy(a), "free start"); ck(cudaEventDestroy(b), "free end");
    }
    ck(cudaStreamDestroy(st), "free stream");
    for (void* p : owned) ck(cudaFree(p), "free");
    return checked;
}
} // namespace fnx

int main() { try {
    std::cout.setf(std::ios::unitbuf);
    int checked = 0;
    const fnx::Format formats[] = {{fnx::T_IQ3_S, 256, 110, "IQ3_S"}, {fnx::T_IQ4_NL, 32, 18, "IQ4_NL"},
                                 {fnx::T_Q2_0, 64, 18, "Q2_0"}, {fnx::T_Q4_0, 32, 18, "Q4_0"},
                                 {fnx::T_IQ3_XXS, 256, 98, "IQ3_XXS"}, {fnx::T_IQ2_S, 256, 82, "IQ2_S"},
                                 {fnx::T_IQ2_XXS, 256, 66, "IQ2_XXS"}, {fnx::T_Q5_0, 32, 22, "Q5_0"}};
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        for (const auto& f : formats) {
            const int k = f.block == 256 ? 256 : 320;
            checked += fnx::run(f, k, 257, false);
            std::cout << "card " << dev << ' ' << f.name << " partial/split-K/graph exact PASS\n";
        }
        checked += fnx::run(formats[0], 2560, 640, true);
        checked += fnx::run(formats[1], 320, 2560, true);
        checked += fnx::run(formats[2], 320, 2560, true);
        checked += fnx::run(formats[4], 2560, 640, true);
        checked += fnx::run(formats[5], 2560, 640, true);
        checked += fnx::run(formats[6], 2560, 640, true);
    }
    std::cout << "PASS " << checked << " exact wide MoE comparisons\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }

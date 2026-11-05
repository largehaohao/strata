// Compare both GPU selectors to a CPU top-K, including unused storage and graph replays.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <array>
#include <numeric>
#include <random>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> twins; return twins; }
void ck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
namespace {
constexpr int pad = 13, guard = 0x5a5a5a5a, pools = QSA_PAIR_MAX_POOLS;
struct Buffers {
    std::vector<void*> allocated;
    template<class T> T* get(size_t n) {
        void* p;
        ck(cudaMalloc(&p, n * sizeof(T)), "allocate");
        allocated.push_back(p);
        return static_cast<T*>(p);
    }
    ~Buffers() { for (void* p : allocated) cudaFree(p); }
};

// Descending nonnegative score, then ascending global block on ties. The
// attention kernel consumes each partition's winners in chronological order.
void oracle(const float* scores, int pos, int row, int stride, std::vector<int>& cells, std::vector<int>& counts) {
    const int visible = (pos + 1) / QSA_POOL;
    if (visible <= QSA_TOP) {
        counts[pad + row] = counts[pad + stride + row] = -1;
        return;
    }
    std::vector<int> ids(visible);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + QSA_TOP, ids.end(), [&](int a, int b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    ids.resize(QSA_TOP);
    std::sort(ids.begin(), ids.end());
    int n[2] = {};
    auto emit = [&](int j, int m) {
        const int owner = j % 2;
        cells[pad + ((size_t) owner * stride + row) * QSA_MAXSEL + n[owner]++] = j / 2 * QSA_POOL + m;
    };
    for (int j : ids) for (int m = 0; m < QSA_POOL; ++m) emit(j, m);
    for (int m = 0; m < (pos + 1) % QSA_POOL; ++m) emit(visible, m);
    for (int owner = 0; owner < 2; ++owner) counts[pad + owner * stride + row] = n[owner];
}

void run(int dev) {
    ck(cudaSetDevice(dev), "set device");
    Buffers mem;
    constexpr int max_t = 16, capacity = 2 * (max_t + 3);
    const size_t cell_size = 2 * pad + (size_t) capacity * QSA_MAXSEL, count_size = 2 * pad + capacity;
    auto* scores = mem.get<float>((size_t) max_t * pools);
    auto* cells = mem.get<int>(cell_size);
    auto* counts = mem.get<int>(count_size);
    auto* pos = mem.get<int>(max_t);
    auto** lane_pos = mem.get<const int*>(max_t);
    std::vector<const int*> pointers(max_t);
    for (int t = 0; t < max_t; ++t) pointers[t] = pos + t;
    ck(cudaMemcpy(lane_pos, pointers.data(), max_t * sizeof(int*), cudaMemcpyHostToDevice), "lane pointers");
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin), "event");
    ck(cudaEventCreate(&end), "event");
    std::vector<float> keys((size_t) max_t * pools);
    int checks = 0;
    for (int pattern = 0; pattern < 5; ++pattern) {
        std::mt19937 rng(217 + pattern);
        for (size_t i = 0; i < keys.size(); ++i) {
            const unsigned u = rng();
            switch (pattern) {
                case 0: keys[i] = 0.f; break;
                case 1: keys[i] = (float) (i % pools); break;
                case 2: keys[i] = (float) (u % 11); break;
                case 3: keys[i] = 1.f + (u % 1024) * 0x1p-23f; break;
                default: keys[i] = (float) u * 0x1p-28f; break;
            }
        }
        ck(cudaMemcpy(scores, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice), "scores");
        for (int T = 1; T <= max_t; ++T) for (bool lanes : {false, true}) {
            const int stride = T + 3;
            LaneRows lr{};
            lr.pos = lanes ? lane_pos : nullptr;
            auto launch = [&](bool paired) {
                if (paired) {
                    k_idx_select_pair<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells + pad, counts + pad, pools,
                        lanes ? nullptr : pos, stride, lr);
                } else {
                    for (int c = 0; c < 2; ++c) k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores,
                        cells + pad + (size_t) c * stride * QSA_MAXSEL, counts + pad + c * stride, pools,
                        lanes ? nullptr : pos, c, lr);
                }
            };
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "capture");
            launch(true);
            ck(cudaStreamEndCapture(s, &graph), "end capture");
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            for (int start : {0, 2046, 2051, 74401, 200751, pools * QSA_POOL - max_t}) {
                std::vector<int> positions(T);
                std::vector<int> expected_cells(cell_size, guard), expected_counts(count_size, guard);
                for (int t = 0; t < T; ++t) {
                    positions[t] = lanes ? std::max(0, start - t * 751) : start + t;
                    oracle(keys.data() + (size_t) t * pools, positions[t], t, stride, expected_cells, expected_counts);
                }
                ck(cudaMemcpy(pos, positions.data(), T * sizeof(int), cudaMemcpyHostToDevice), "positions");
                for (int mode = 0; mode < 3; ++mode) {
                    ck(cudaMemsetAsync(cells, 0x5a, cell_size * sizeof(int), s), "guard cells");
                    ck(cudaMemsetAsync(counts, 0x5a, count_size * sizeof(int), s), "guard counts");
                    if (mode == 2) ck(cudaGraphLaunch(exec, s), "replay");
                    else launch(mode == 1);
                    ck(cudaStreamSynchronize(s), "select");
                    std::vector<int> got_cells(cell_size), got_counts(count_size);
                    ck(cudaMemcpy(got_cells.data(), cells, cell_size * sizeof(int), cudaMemcpyDeviceToHost), "cells");
                    ck(cudaMemcpy(got_counts.data(), counts, count_size * sizeof(int), cudaMemcpyDeviceToHost), "counts");
                    if (got_counts != expected_counts || got_cells != expected_cells) {
                        std::fprintf(stderr, "mismatch device=%d T=%d pattern=%d start=%d lanes=%d mode=%d\n",
                                     dev, T, pattern, start, lanes, mode);
                        throw std::runtime_error("top-K mismatch");
                    }
                    ++checks;
                }
            }
            ck(cudaGraphExecDestroy(exec), "destroy graph exec");
            ck(cudaGraphDestroy(graph), "destroy graph");
        }
    }
    std::printf("device=%d exact checks=%d\n", dev, checks);
    for (int p : {74401, 200751}) for (int T : {1, 4, 6, 12, 16}) {
        ck(cudaMemcpy(pos, &p, sizeof(p), cudaMemcpyHostToDevice), "timed position");
        auto timed = [&](bool pair) {
            ck(cudaEventRecord(begin, s), "begin");
            constexpr int repetitions = 40;
            for (int i = 0; i < repetitions; ++i) {
                if (pair) k_idx_select_pair<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells, counts, pools, pos, T, {});
                else for (int c = 0; c < 2; ++c)
                    k_idx_select<<<T, QSA_SEL_THREADS, 0, s>>>(scores, cells + (size_t) c * T * QSA_MAXSEL,
                        counts + c * T, pools, pos, c, {});
            }
            ck(cudaEventRecord(end, s), "end");
            ck(cudaEventSynchronize(end), "wait");
            float ms;
            ck(cudaEventElapsedTime(&ms, begin, end), "time");
            return ms * 1000.f / repetitions;
        };
        for (bool pair : {false, true}) timed(pair);
        std::array<std::array<float, 5>, 2> sample;
        for (int rep = 0; rep < 5; ++rep)
            for (int pass = 0; pass < 2; ++pass) { int mode = (rep + pass) % 2; sample[mode][rep] = timed(mode); }
        for (auto& values : sample) std::sort(values.begin(), values.end());
        std::printf("device=%d pos=%d T=%d separate_us=%.2f paired_us=%.2f\n", dev, p, T, sample[0][2], sample[1][2]);
    }
    ck(cudaEventDestroy(begin), "destroy event");
    ck(cudaEventDestroy(end), "destroy event");
    ck(cudaStreamDestroy(s), "destroy stream");
}
}  // namespace
}  // namespace fnx

int main() {
    try {
        for (int device = 0; device < 2; ++device) fnx::run(device);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

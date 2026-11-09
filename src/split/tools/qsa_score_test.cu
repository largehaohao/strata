// Compare long-prompt tiles with the unchanged reference WMMA kernel.
#define FNX_KPART 99
#include "../src/kernels.cu"
#include <array>
#include <numeric>
#include <random>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> twins; return twins; }
void ck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}
namespace {
struct Memory {
    std::vector<void*> pointers;
    template<class T> T* get(size_t n) {
        void* p;
        ck(cudaMalloc(&p, n * sizeof(T)), "allocate");
        pointers.push_back(p);
        return static_cast<T*>(p);
    }
    ~Memory() { for (void* p : pointers) cudaFree(p); }
};

__global__ void init_keys(__half* pooled, KvPages pages) {
    const int j = blockIdx.x, dim = threadIdx.x;
    unsigned value = (j * QSA_DIM + dim) * 1664525u + 1013904223u;
    value ^= value >> 11;
    pooled[kvp_block(pages, j) + dim] = __float2half_rn((int(value % 65537) - 32768) * 0x1p-12f);
}

__global__ void compare_scores(const float* expected, const float* actual, int T, int pools, const int* pos,
                               unsigned* errors) {
    const int j = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y, visible = (*pos + t + 1) / QSA_POOL;
    const size_t index = (size_t)t * pools + j;
    // Future columns within a tile may differ; selection reads only visible keys.
    if (j < visible && visible > QSA_TOP && __float_as_uint(expected[index]) != __float_as_uint(actual[index]))
        atomicAdd(errors, 1);
    if (t == 0 && j < 32 && __float_as_uint(actual[(size_t)T * pools + j]) != 0x5a5a5a5a)
        atomicAdd(errors + 1, 1);
}

void run(int device) {
    ck(cudaSetDevice(device), "set device");
    Memory memory;
    constexpr int pools = 65536, max_rows = 256, num_pages = (pools + KVP_BLOCKS - 1) / KVP_BLOCKS;
    // Preserve the large gaps between a layer's pooled rows inside real KV pages.
    const size_t page_size = kvp_page(device, true);
    auto* pooled = memory.get<__half>(num_pages * page_size);
    auto* page_table = memory.get<int>(num_pages);
    auto* queries = memory.get<float>(max_rows * QSA_HEADS * QSA_DIM);
    auto* expected = memory.get<float>((size_t)max_rows * pools + 32);
    auto* actual = memory.get<float>((size_t)max_rows * pools + 32);
    auto* position = memory.get<int>(1);
    auto* errors = memory.get<unsigned>(2);
    std::vector<int> host_pages(num_pages);
    std::iota(host_pages.begin(), host_pages.end(), 0);
    std::mt19937 random(51739);
    std::shuffle(host_pages.begin(), host_pages.end(), random);
    std::vector<float> host_queries(max_rows * QSA_HEADS * QSA_DIM);
    ck(cudaMemcpy(page_table, host_pages.data(), num_pages * sizeof(int), cudaMemcpyHostToDevice), "page table");
    KvPages pages{page_table, page_size, kvp_rows(device), 1};
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "stream");
    init_keys<<<pools, QSA_DIM, 0, stream>>>(pooled, pages);
    auto launch = [&](bool wide, int T) {
        if (wide)
            k_idx_score_tc_wide<<<dim3((pools + 255) / 256, (T + 3) / 4), 256, 0, stream>>>(
                queries, pooled, pages, actual, pools, position, T);
        else
            k_idx_score_tc<<<dim3((pools + 63) / 64, (T + 15) / 16), 256, 0, stream>>>(
                queries, pooled, pages, expected, pools, position, T);
    };
    int checks = 0;
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (float& value : host_queries)
            value = pattern == 0 ? 0.f : (int(random() % 65537) - 32768) * (pattern == 1 ? 0x1p-28f : 0x1p-13f);
        ck(cudaMemcpy(queries, host_queries.data(), host_queries.size() * sizeof(float), cudaMemcpyHostToDevice), "queries");
        for (int start : {0, 2040, 2051, 32761, 65531, 131065, 204793, 261888}) {
            for (int T : {1, 7, 16, 19, 64, 255, 256}) {
                ck(cudaMemcpy(position, &start, sizeof(start), cudaMemcpyHostToDevice), "position");
                const size_t bytes = ((size_t)T * pools + 32) * sizeof(float);
                ck(cudaMemsetAsync(expected, 0x5a, bytes, stream), "reference guard");
                ck(cudaMemsetAsync(actual, 0x5a, bytes, stream), "candidate guard");
                ck(cudaMemsetAsync(errors, 0, 2 * sizeof(unsigned), stream), "error counters");
                launch(false, T);
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
                launch(true, T);
                ck(cudaStreamEndCapture(stream, &graph), "end capture");
                ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
                ck(cudaGraphLaunch(executable, stream), "replay");
                compare_scores<<<dim3((pools + 255) / 256, T), 256, 0, stream>>>(
                    expected, actual, T, pools, position, errors);
                ck(cudaStreamSynchronize(stream), "compare");
                unsigned host_errors[2];
                ck(cudaMemcpy(host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost), "read counters");
                if (host_errors[0] || host_errors[1]) {
                    std::fprintf(stderr, "device=%d pattern=%d pos=%d T=%d unequal=%u guards=%u\n",
                                 device, pattern, start, T, host_errors[0], host_errors[1]);
                    throw std::runtime_error("QSA score mismatch");
                }
                ck(cudaGraphExecDestroy(executable), "destroy executable");
                ck(cudaGraphDestroy(graph), "destroy graph");
                ++checks;
            }
        }
    }
    std::printf("device=%d exact_visible_score_graph_guard_checks=%d page_halves=%zu\n", device, checks, page_size);
    std::fflush(stdout);
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin), "begin event");
    ck(cudaEventCreate(&end), "end event");
    for (int start : {32767, 65535, 131071, 204799}) for (int T : {32, 256}) {
        ck(cudaMemcpy(position, &start, sizeof(start), cudaMemcpyHostToDevice), "timed position");
        std::array<std::array<float, 5>, 2> times;
        for (int repeat = -1; repeat < 5; ++repeat) for (int pass = 0; pass < 2; ++pass) {
            const int mode = (pass + std::max(repeat, 0)) % 2;
            ck(cudaEventRecord(begin, stream), "begin");
            for (int i = 0; i < 20; ++i) launch(mode != 0, T);
            ck(cudaEventRecord(end, stream), "end");
            ck(cudaEventSynchronize(end), "wait");
            float ms;
            ck(cudaEventElapsedTime(&ms, begin, end), "elapsed");
            if (repeat >= 0) times[mode][repeat] = ms * 50.f;
        }
        for (auto& values : times) std::sort(values.begin(), values.end());
        std::printf("device=%d pos=%d T=%d reference_us=%.3f wide_us=%.3f\n",
                    device, start, T, times[0][2], times[1][2]);
        std::fflush(stdout);
    }
    ck(cudaEventDestroy(begin), "destroy begin event");
    ck(cudaEventDestroy(end), "destroy end event");
    ck(cudaStreamDestroy(stream), "destroy stream");
}
}  // namespace
}  // namespace fnx

int main() {
    try { for (int device = 0; device < 2; ++device) fnx::run(device); }
    catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}

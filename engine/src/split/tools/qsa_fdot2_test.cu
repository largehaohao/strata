// Parity, edge-case guards, graph replay, multi-lane LaneRows, and A/B/C benchmark for opt-in half2 fdot2 decode QSA scoring.
#define FNX_KPART 99
#include "../src/kernels.cu"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> twins; return twins; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

namespace {

constexpr int pools = 65536;
constexpr int max_rows = 16;
constexpr int max_stride = 16;
constexpr int pad = 13;
constexpr size_t cells_size = 2 * pad + (size_t) 2 * max_stride * QSA_MAXSEL;
constexpr size_t counts_size = 2 * pad + 2 * max_stride;
constexpr unsigned sentinel_bits = 0x5a5a5a5au;

struct Memory {
    int dev;
    struct Block {
        unsigned char* p;
        size_t bytes;
    };
    std::vector<Block> blocks;
    explicit Memory(int device) : dev(device) {}
    template <class V>
    V* make(size_t n) {
        unsigned char* p = nullptr;
        const size_t bytes = n * sizeof(V);
        ck(cudaMalloc((void**) &p, bytes + 256), "allocation");
        ck(cudaMemset(p, 0xa5, bytes + 256), "allocation guards");
        blocks.push_back({p, bytes});
        return reinterpret_cast<V*>(p + 128);
    }
    void guards() const {
        ck(cudaSetDevice(dev), "guard device");
        unsigned char h[256];
        for (auto b : blocks) {
            ck(cudaMemcpy(h, b.p, 128, cudaMemcpyDeviceToHost), "front guard");
            ck(cudaMemcpy(h + 128, b.p + 128 + b.bytes, 128, cudaMemcpyDeviceToHost), "back guard");
            for (auto v : h)
                if (v != 0xa5) throw std::runtime_error("QSA fdot2 buffer guard overwritten");
        }
    }
    ~Memory() {
        (void) cudaSetDevice(dev);
        for (auto b : blocks) (void) cudaFree(b.p);
    }
};

__global__ void initialize_keys(__half* keys, KvPages pages) {
    const int j = blockIdx.x, d = threadIdx.x;
    unsigned v = (unsigned) (j * QSA_DIM + d) * 1664525u + 1013904223u;
    v ^= v >> 11;
    keys[kvp_block(pages, j) + d] = __float2half_rn((int(v % 65537) - 32768) * 0x1p-12f);
}

__global__ void flush_cache(unsigned* data, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) blockDim.x * gridDim.x)
        data[i] = data[i] * 1664525u + 1013904223u;
}

__global__ void compare_bits(const float* a, const float* b, int T, unsigned* diff) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= pools || t >= T) return;
    const size_t i = (size_t) t * pools + j;
    if (__float_as_uint(a[i]) != __float_as_uint(b[i])) atomicAdd(diff, 1u);
}

__global__ void compare_ints(const int* a, const int* b, size_t n, unsigned* errors) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) blockDim.x * gridDim.x)
        if (a[i] != b[i]) atomicAdd(errors, 1u);
}

// stats[0]=bad_bits (NaN/Inf/neg/guard/zero_mismatch/tol), stats[1]=max_abs_bits (uint float), stats[2]=max_rel_bits (uint float)
__global__ void check_accuracy(const float* ref, const float* got, int T, const int* pos0p, LaneRows lr, bool exact_zero,
                               float tol_rel, float tol_abs, unsigned* stats) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= pools || t >= T) return;
    const int p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL;
    const size_t i = (size_t) t * pools + j;
    const unsigned gb = __float_as_uint(got[i]);
    if (nv <= QSA_TOP || j >= nv) {
        if (gb != sentinel_bits) atomicAdd(&stats[0], 1u);
        return;
    }
    const float rv = ref[i], gv = got[i];
    if (isnan(gv) || isinf(gv) || (gb & 0x80000000u) != 0u) {
        atomicAdd(&stats[0], 1u);
        return;
    }
    if (exact_zero) {
        if (gb != 0u || __float_as_uint(rv) != 0u) atomicAdd(&stats[0], 1u);
        return;
    }
    const float ad = fabsf(gv - rv);
    if (tol_abs >= 0.f && !(ad <= tol_rel * fabsf(rv) + tol_abs)) atomicAdd(&stats[0], 1u);
    const float rd = ad / fmaxf(fabsf(rv), 1.0f);
    atomicMax(&stats[1], __float_as_uint(ad));
    atomicMax(&stats[2], __float_as_uint(rd));
}

struct State {
    int dev;
    Memory mem;
    cudaStream_t st;
    cudaEvent_t begin, end;
    __half* keys;
    float *query, *ref_score, *m1_score, *m2_score, *aux_score;
    int *ref_cells, *m1_cells, *m2_cells, *aux_cells;
    int *ref_counts, *m1_counts, *m2_counts, *aux_counts;
    int *pos, *lane_pos_vals;
    const int** lane_pos_ptrs;
    const int** lane_pt_ptrs;
    int *table0, *table1;
    KvPages pages;
    unsigned *diff, *stats, *flush;

    explicit State(int device) : dev(device), mem(device) {
        ck(cudaSetDevice(dev), "state device");
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        ck(cudaEventCreate(&begin), "begin event");
        ck(cudaEventCreate(&end), "end event");
        constexpr int np = (pools + KVP_BLOCKS - 1) / KVP_BLOCKS;
        const size_t ps = kvp_page(dev, true);
        table0 = mem.make<int>(np);
        table1 = mem.make<int>(np);
        std::vector<int> hp(np);
        std::iota(hp.begin(), hp.end(), 0);
        {
            std::mt19937 rng(51739 + device * 17);
            std::shuffle(hp.begin(), hp.end(), rng);
            ck(cudaMemcpy(table0, hp.data(), np * sizeof(int), cudaMemcpyHostToDevice), "page table 0");
        }
        {
            std::mt19937 rng(98213 + device * 29);
            std::shuffle(hp.begin(), hp.end(), rng);
            ck(cudaMemcpy(table1, hp.data(), np * sizeof(int), cudaMemcpyHostToDevice), "page table 1");
        }
        pages = {table0, ps, kvp_rows(dev), 1};
        keys = mem.make<__half>(np * ps);
        query = mem.make<float>((size_t) max_rows * 512);
        ref_score = mem.make<float>((size_t) max_rows * pools);
        m1_score = mem.make<float>((size_t) max_rows * pools);
        m2_score = mem.make<float>((size_t) max_rows * pools);
        aux_score = mem.make<float>((size_t) max_rows * pools);
        ref_cells = mem.make<int>(cells_size);
        m1_cells = mem.make<int>(cells_size);
        m2_cells = mem.make<int>(cells_size);
        aux_cells = mem.make<int>(cells_size);
        ref_counts = mem.make<int>(counts_size);
        m1_counts = mem.make<int>(counts_size);
        m2_counts = mem.make<int>(counts_size);
        aux_counts = mem.make<int>(counts_size);
        pos = mem.make<int>(1);
        lane_pos_vals = mem.make<int>(max_rows);
        lane_pos_ptrs = mem.make<const int*>(max_rows);
        lane_pt_ptrs = mem.make<const int*>(max_rows);
        diff = mem.make<unsigned>(4);
        stats = mem.make<unsigned>(6);
        flush = mem.make<unsigned>(32 * 1024 * 1024);   // 128 MiB cache flush buffer
        ck(cudaMemsetAsync(flush, 1, 32ull * 1024 * 1024 * sizeof(unsigned), st), "init flush");
        initialize_keys<<<pools, QSA_DIM, 0, st>>>(keys, pages);
        ck(cudaStreamSynchronize(st), "key initialization");
    }
    ~State() {
        (void) cudaSetDevice(dev);
        (void) cudaEventDestroy(begin);
        (void) cudaEventDestroy(end);
        (void) cudaStreamDestroy(st);
    }
};

enum class ScoreKind { FP32, FP32_PERS, FDOT2_M1, FDOT2_M1_PERS, FDOT2_M2, FDOT2_M2_PERS };

void launch_score(State& s, ScoreKind kind, float* score, int T, int budget, const int* pos0p, LaneRows lr) {
    const int full_blocks = (pools + 255) / 256;
    const int pers_blocks = std::min(full_blocks, std::max(1, (budget + T - 1) / T));
    switch (kind) {
        case ScoreKind::FP32:
            k_idx_score<<<dim3(full_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
        case ScoreKind::FP32_PERS:
            k_idx_score_persistent<<<dim3(pers_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
        case ScoreKind::FDOT2_M1:
            k_idx_score_fdot2<false, false><<<dim3(full_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
        case ScoreKind::FDOT2_M1_PERS:
            k_idx_score_fdot2<true, false><<<dim3(pers_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
        case ScoreKind::FDOT2_M2:
            k_idx_score_fdot2<false, true><<<dim3(full_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
        case ScoreKind::FDOT2_M2_PERS:
            k_idx_score_fdot2<true, true><<<dim3(pers_blocks, T), 256, 0, s.st>>>(s.query, s.keys, s.pages, score, pools, pos0p, lr);
            break;
    }
    ck(cudaGetLastError(), "qsa fdot2 score launch");
}

void launch_score_and_select(State& s, ScoreKind kind, float* score, int* cells, int* counts, int T, int budget,
                             const int* pos0p, LaneRows lr) {
    launch_score(s, kind, score, T, budget, pos0p, lr);
    k_idx_select_pair<true, true><<<T, QSA_SEL_THREADS, 0, s.st>>>(score, cells + pad, counts + pad, pools, pos0p, max_stride, lr);
    ck(cudaGetLastError(), "qsa fdot2 select launch");
}

struct Graphs {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    Graphs(State& s, ScoreKind k1, ScoreKind k2, int T, int budget) {
        ck(cudaSetDevice(s.dev), "graph capture device");
        ck(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal), "begin capture");
        launch_score_and_select(s, k1, s.m1_score, s.m1_cells, s.m1_counts, T, budget, s.pos, {});
        launch_score_and_select(s, k2, s.m2_score, s.m2_cells, s.m2_counts, T, budget, s.pos, {});
        ck(cudaStreamEndCapture(s.st, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    }
    ~Graphs() {
        if (exec) (void) cudaGraphExecDestroy(exec);
        if (graph) (void) cudaGraphDestroy(graph);
    }
};

float bits_to_f32(unsigned u) {
    float f = 0.f;
    std::memcpy(&f, &u, 4);
    return f;
}

double compute_top512_overlap(const std::vector<float>& ref, const std::vector<float>& got, int T, int pos0) {
    std::vector<int> idx;
    double overlap_sum = 0.0;
    int active_rows = 0;
    for (int t = 0; t < T; ++t) {
        const int nv = (pos0 + t + 1) / QSA_POOL;
        if (nv <= QSA_TOP) continue;
        idx.resize((size_t) nv);
        std::iota(idx.begin(), idx.end(), 0);
        const float* rp = ref.data() + (size_t) t * pools;
        const float* gp = got.data() + (size_t) t * pools;
        auto cmp_ref = [rp](int a, int b) {
            if (rp[a] != rp[b]) return rp[a] > rp[b];
            return a < b;
        };
        auto cmp_got = [gp](int a, int b) {
            if (gp[a] != gp[b]) return gp[a] > gp[b];
            return a < b;
        };
        std::vector<int> top_r = idx, top_g = idx;
        std::nth_element(top_r.begin(), top_r.begin() + QSA_TOP, top_r.end(), cmp_ref);
        std::nth_element(top_g.begin(), top_g.begin() + QSA_TOP, top_g.end(), cmp_got);
        std::unordered_set<int> set_r(top_r.begin(), top_r.begin() + QSA_TOP);
        int hits = 0;
        for (int k = 0; k < QSA_TOP; ++k)
            if (set_r.count(top_g[(size_t) k])) ++hits;
        overlap_sum += (double) hits / (double) QSA_TOP;
        ++active_rows;
    }
    return active_rows ? (overlap_sum / active_rows) * 100.0 : 100.0;
}

void run_parity() {
    std::array<std::unique_ptr<State>, 2> states;
    for (int c = 0; c < 2; ++c) states[c] = std::make_unique<State>(c);

    // Prepare multi-lane LaneRows metadata on each card.
    for (int c = 0; c < 2; ++c) {
        State& s = *states[c];
        ck(cudaSetDevice(c), "lane init device");
        std::vector<int> hpos(max_rows);
        std::vector<const int*> hpos_ptrs(max_rows), hpt_ptrs(max_rows);
        for (int t = 0; t < max_rows; ++t) {
            // Mix <= QSA_TOP and > QSA_TOP positions per lane.
            hpos[t] = (t % 3 == 0) ? 1024 : (32761 + t * 5003);
            hpos_ptrs[t] = s.lane_pos_vals + t;
            hpt_ptrs[t] = (t & 1) ? s.table1 : s.table0;
        }
        ck(cudaMemcpy(s.lane_pos_vals, hpos.data(), max_rows * sizeof(int), cudaMemcpyHostToDevice), "lane pos vals");
        ck(cudaMemcpy(s.lane_pos_ptrs, hpos_ptrs.data(), max_rows * sizeof(int*), cudaMemcpyHostToDevice), "lane pos ptrs");
        ck(cudaMemcpy(s.lane_pt_ptrs, hpt_ptrs.data(), max_rows * sizeof(int*), cudaMemcpyHostToDevice), "lane pt ptrs");
    }

    size_t checks = 0;
    double worst_m1_rel = 0.0, worst_m2_rel = 0.0, min_m1_ov = 100.0, min_m2_ov = 100.0;

    // patterns: 0=zero, 1=subnormal/tiny, 2=normal FP32, 3=exact FP16
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::mt19937 rng(7329 + pattern);
        std::vector<float> q((size_t) max_rows * 512);
        for (float& v : q) {
            if (pattern == 0) v = 0.f;
            else if (pattern == 1) v = (int(rng() % 65537) - 32768) * 0x1p-28f;
            else if (pattern == 2) v = (int(rng() % 65537) - 32768) * 0x1p-13f;
            else v = (int(rng() % 1025) - 512) * 0x1p-8f;
        }
        for (int c = 0; c < 2; ++c) {
            ck(cudaSetDevice(c), "copy query device");
            ck(cudaMemcpy(states[c]->query, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "queries");
        }

        // Sweep non-persistent vs persistent equivalence across budgets & T + graph replay.
        for (int budget : {64, 128, 256, 512}) {
            for (int T : {1, 2, 4, 8, 16}) {
                // Only run full position sweep on budget=256 and subset on other budgets to keep runtime tight while covering all combinations.
                const std::vector<int> positions = (budget == 256)
                    ? std::vector<int>{0, 2046, 2051, 32761, 74401, 204793, 262144 - T}
                    : std::vector<int>{2046, 2051, 74401};
                for (int c = 0; c < 2; ++c) {
                    State& s = *states[c];
                    Graphs pers_graph(s, ScoreKind::FDOT2_M1_PERS, ScoreKind::FDOT2_M2_PERS, T, budget);
                    for (int position : positions) {
                        ck(cudaSetDevice(c), "pos device");
                        ck(cudaMemcpy(s.pos, &position, sizeof(int), cudaMemcpyHostToDevice), "position");

                        // 1) Non-persistent reference + fdot2 m1/m2
                        ck(cudaMemsetAsync(s.ref_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "ref score memset");
                        ck(cudaMemsetAsync(s.aux_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "aux score memset");
                        ck(cudaMemsetAsync(s.m1_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "m1 score memset");
                        ck(cudaMemsetAsync(s.m2_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "m2 score memset");
                        ck(cudaMemsetAsync(s.ref_cells, 0x5a, cells_size * sizeof(int), s.st), "ref cells memset");
                        ck(cudaMemsetAsync(s.aux_cells, 0x5a, cells_size * sizeof(int), s.st), "aux cells memset");
                        ck(cudaMemsetAsync(s.ref_counts, 0x5a, counts_size * sizeof(int), s.st), "ref counts memset");
                        ck(cudaMemsetAsync(s.aux_counts, 0x5a, counts_size * sizeof(int), s.st), "aux counts memset");

                        launch_score(*states[c], ScoreKind::FP32, s.ref_score, T, budget, s.pos, {});
                        launch_score_and_select(s, ScoreKind::FDOT2_M1, s.aux_score, s.ref_cells, s.ref_counts, T, budget, s.pos, {});
                        launch_score_and_select(s, ScoreKind::FDOT2_M2, s.m2_score, s.aux_cells, s.aux_counts, T, budget, s.pos, {});
                        ck(cudaStreamSynchronize(s.st), "sync ref+nonpers");

                        // Check accuracy & exact FP16 equality between m1 and m2
                        if (budget == 256) {
                            const float tol_r1 = pattern == 1 ? 0.f : 1e-2f;
                            const float tol_a1 = pattern == 1 ? 1e-6f : 2e-2f;
                            const float tol_r2 = pattern == 1 ? 0.f : 1e-5f;
                            const float tol_a2 = pattern == 1 ? 1e-6f : 5e-5f;
                            ck(cudaMemsetAsync(s.stats, 0, 6 * sizeof(unsigned), s.st), "zero stats");
                            check_accuracy<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(
                                s.ref_score, s.aux_score, T, s.pos, {}, pattern == 0, tol_r1, tol_a1, s.stats);
                            check_accuracy<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(
                                s.ref_score, s.m2_score, T, s.pos, {}, pattern == 0, tol_r2, tol_a2, s.stats + 3);
                            if (pattern == 3) {
                                ck(cudaMemsetAsync(s.diff, 0, 4 * sizeof(unsigned), s.st), "zero diff");
                                compare_bits<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(s.aux_score, s.m2_score, T, s.diff);
                            }
                            ck(cudaStreamSynchronize(s.st), "sync acc");
                            unsigned st[6]{};
                            ck(cudaMemcpy(st, s.stats, 6 * sizeof(unsigned), cudaMemcpyDeviceToHost), "copy stats");
                            const float m1_abs = bits_to_f32(st[1]), m1_rel = bits_to_f32(st[2]);
                            const float m2_abs = bits_to_f32(st[4]), m2_rel = bits_to_f32(st[5]);
                            if (st[0] || st[3]) {
                                std::fprintf(stderr,
                                             "FDOT2_FAIL card=%d pat=%d pos=%d T=%d m1_bad=%u (abs=%.3e rel=%.3e) m2_bad=%u (abs=%.3e rel=%.3e)\n",
                                             c, pattern, position, T, st[0], m1_abs, m1_rel, st[3], m2_abs, m2_rel);
                                throw std::runtime_error("fdot2 score/sentinel/tolerance check failed");
                            }
                            if (pattern == 2) {
                                worst_m1_rel = std::max(worst_m1_rel, (double) m1_rel);
                                worst_m2_rel = std::max(worst_m2_rel, (double) m2_rel);
                            } else if (pattern == 3) {
                                unsigned d0 = 0;
                                ck(cudaMemcpy(&d0, s.diff, 4, cudaMemcpyDeviceToHost), "copy m1==m2 diff");
                                if (d0 != 0) {
                                    std::fprintf(stderr, "FDOT2_EXACT_FP16_FAIL diff=%u\n", d0);
                                    throw std::runtime_error("fdot2 exact FP16 mismatch");
                                }
                            }
                        }

                        // Save non-persistent M2 score for comparison with persistent M1/M2 (direct & graph)
                        // Compare aux_score (nonpers M1) vs graph m1_score, and m2_score (we'll copy m2_score -> m2_score_ref? No, run nonpers M2 before persistent!)
                        // Wait: overwrite m2_score in graph launch, so save nonpers M2 into ref_score first? Better: only check nonpers M2 bit equality via m2_cells/counts + overwrite check cleanly:
                        ck(cudaMemcpyAsync(s.ref_score, s.m2_score, (size_t) T * pools * sizeof(float), cudaMemcpyDeviceToDevice, s.st), "save nonpers m2");

                        // 2) Persistent direct + 2 graph replays
                        for (int replay = 0; replay < 3; ++replay) {
                            ck(cudaMemsetAsync(s.m1_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "pers m1 memset");
                            ck(cudaMemsetAsync(s.m2_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "pers m2 memset");
                            ck(cudaMemsetAsync(s.m1_cells, 0x5a, cells_size * sizeof(int), s.st), "pers m1 cells");
                            ck(cudaMemsetAsync(s.m2_cells, 0x5a, cells_size * sizeof(int), s.st), "pers m2 cells");
                            ck(cudaMemsetAsync(s.m1_counts, 0x5a, counts_size * sizeof(int), s.st), "pers m1 counts");
                            ck(cudaMemsetAsync(s.m2_counts, 0x5a, counts_size * sizeof(int), s.st), "pers m2 counts");
                            ck(cudaMemsetAsync(s.diff, 0, 4 * sizeof(unsigned), s.st), "zero pers diff");

                            if (replay == 0) {
                                launch_score_and_select(s, ScoreKind::FDOT2_M1_PERS, s.m1_score, s.m1_cells, s.m1_counts, T, budget, s.pos, {});
                                launch_score_and_select(s, ScoreKind::FDOT2_M2_PERS, s.m2_score, s.m2_cells, s.m2_counts, T, budget, s.pos, {});
                            } else {
                                ck(cudaGraphLaunch(pers_graph.exec, s.st), "replay fdot2 pers");
                            }
                            compare_bits<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(s.aux_score, s.m1_score, T, s.diff);
                            compare_bits<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(s.ref_score, s.m2_score, T, s.diff + 1);
                            compare_ints<<<1024, 256, 0, s.st>>>(s.ref_cells, s.m1_cells, cells_size, s.diff + 2);
                            compare_ints<<<1024, 256, 0, s.st>>>(s.aux_cells, s.m2_cells, cells_size, s.diff + 2);
                            compare_ints<<<64, 256, 0, s.st>>>(s.ref_counts, s.m1_counts, counts_size, s.diff + 3);
                            compare_ints<<<64, 256, 0, s.st>>>(s.aux_counts, s.m2_counts, counts_size, s.diff + 3);
                            ck(cudaStreamSynchronize(s.st), "sync pers check");
                            unsigned df[4]{};
                            ck(cudaMemcpy(df, s.diff, 16, cudaMemcpyDeviceToHost), "copy pers diff");
                            if (df[0] || df[1] || df[2] || df[3]) {
                                std::fprintf(stderr, "FDOT2_PERS_FAIL card=%d pat=%d budget=%d T=%d pos=%d replay=%d d=%u,%u,%u,%u\n",
                                             c, pattern, budget, T, position, replay, df[0], df[1], df[2], df[3]);
                                throw std::runtime_error("fdot2 persistent mismatch");
                            }
                            ++checks;
                        }
                    }
                }
            }
        }

        // Top-512 block overlap check on normal pattern (pattern == 2), T = 16
        if (pattern == 2) {
            constexpr int T = 16;
            State& s = *states[0];
            ck(cudaSetDevice(0), "overlap device");
            std::vector<float> href((size_t) T * pools), hm1((size_t) T * pools), hm2((size_t) T * pools);
            for (int position : {32761, 74401, 204793}) {
                ck(cudaMemcpy(s.pos, &position, 4, cudaMemcpyHostToDevice), "overlap pos");
                launch_score(s, ScoreKind::FP32, s.ref_score, T, 0, s.pos, {});
                launch_score(s, ScoreKind::FDOT2_M1, s.m1_score, T, 0, s.pos, {});
                launch_score(s, ScoreKind::FDOT2_M2, s.m2_score, T, 0, s.pos, {});
                ck(cudaStreamSynchronize(s.st), "sync overlap");
                ck(cudaMemcpy(href.data(), s.ref_score, href.size() * 4, cudaMemcpyDeviceToHost), "copy ref ov");
                ck(cudaMemcpy(hm1.data(), s.m1_score, hm1.size() * 4, cudaMemcpyDeviceToHost), "copy m1 ov");
                ck(cudaMemcpy(hm2.data(), s.m2_score, hm2.size() * 4, cudaMemcpyDeviceToHost), "copy m2 ov");
                const double ov1 = compute_top512_overlap(href, hm1, T, position);
                const double ov2 = compute_top512_overlap(href, hm2, T, position);
                min_m1_ov = std::min(min_m1_ov, ov1);
                min_m2_ov = std::min(min_m2_ov, ov2);
                std::printf("QSA_FDOT2_OVERLAP pos=%d T=%d mode1_top512=%.3f%% mode2_top512=%.3f%%\n", position, T, ov1, ov2);
                if (ov2 < 99.5) throw std::runtime_error("fdot2 Mode 2 top-512 overlap below 99.5%");
            }
        }

        // Multi-lane LaneRows parity on both cards
        for (int c = 0; c < 2; ++c) {
            State& s = *states[c];
            ck(cudaSetDevice(c), "lane parity device");
            const int T = 8;
            LaneRows lr{s.lane_pos_ptrs, s.lane_pt_ptrs};
            ck(cudaMemsetAsync(s.ref_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "lr ref memset");
            ck(cudaMemsetAsync(s.m1_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "lr m1 memset");
            ck(cudaMemsetAsync(s.m2_score, 0x5a, (size_t) T * pools * sizeof(float), s.st), "lr m2 memset");
            ck(cudaMemsetAsync(s.stats, 0, 6 * sizeof(unsigned), s.st), "lr stats zero");
            launch_score(s, ScoreKind::FP32, s.ref_score, T, 0, nullptr, lr);
            launch_score(s, ScoreKind::FDOT2_M1, s.m1_score, T, 0, nullptr, lr);
            launch_score(s, ScoreKind::FDOT2_M2, s.m2_score, T, 0, nullptr, lr);
            const float lr_r1 = pattern == 1 ? 0.f : 1e-2f, lr_a1 = pattern == 1 ? 1e-6f : 2e-2f;
            const float lr_r2 = pattern == 1 ? 0.f : 1e-5f, lr_a2 = pattern == 1 ? 1e-6f : 5e-5f;
            check_accuracy<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(
                s.ref_score, s.m1_score, T, nullptr, lr, pattern == 0, lr_r1, lr_a1, s.stats);
            check_accuracy<<<dim3((pools + 255) / 256, T), 256, 0, s.st>>>(
                s.ref_score, s.m2_score, T, nullptr, lr, pattern == 0, lr_r2, lr_a2, s.stats + 3);
            ck(cudaStreamSynchronize(s.st), "sync lr");
            unsigned st[6]{};
            ck(cudaMemcpy(st, s.stats, 6 * sizeof(unsigned), cudaMemcpyDeviceToHost), "copy lr stats");
            if (st[0] || st[3]) {
                std::fprintf(stderr, "FDOT2_LR_FAIL card=%d pat=%d m1_bad=%u (rel=%.3e) m2_bad=%u (rel=%.3e)\n",
                             c, pattern, st[0], bits_to_f32(st[2]), st[3], bits_to_f32(st[5]));
                throw std::runtime_error("fdot2 LaneRows parity failure");
            }
            ++checks;
        }
    }

    for (auto& s : states) s->mem.guards();
    std::printf("QSA_FDOT2 passed %zu checks (worst_m1_rel=%.3e worst_m2_rel=%.3e min_m1_ov=%.3f%% min_m2_ov=%.3f%%)\n",
                checks, worst_m1_rel, worst_m2_rel, min_m1_ov, min_m2_ov);
    std::fflush(stdout);
}

void run_bench() {
    State s(0);
    ck(cudaSetDevice(0), "bench device");
    std::mt19937 rng(4242);
    std::vector<float> q((size_t) max_rows * 512);
    for (float& v : q) v = (int(rng() % 65537) - 32768) * 0x1p-13f;
    ck(cudaMemcpy(s.query, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "bench query");

    const std::array<ScoreKind, 6> modes = {
        ScoreKind::FP32,      ScoreKind::FP32_PERS,
        ScoreKind::FDOT2_M1,  ScoreKind::FDOT2_M1_PERS,
        ScoreKind::FDOT2_M2,  ScoreKind::FDOT2_M2_PERS,
    };

    for (int cold : {0, 1}) {
        std::printf("\n=== QSA FDOT2 BENCH (%s) on device 0 ===\n", cold ? "COLD 128MiB flush" : "HOT L2/MALL");
        for (int pos : {32767, 65535, 131071, 204799}) {
            ck(cudaMemcpy(s.pos, &pos, sizeof(int), cudaMemcpyHostToDevice), "bench pos");
            for (int T : {1, 4, 6, 8, 16}) {
                std::array<std::array<float, 5>, 6> times{};
                constexpr int budget = 256;
                const int iters = cold ? 1 : 20;
                for (int repeat = -1; repeat < 5; ++repeat) {
                    for (int p = 0; p < 6; ++p) {
                        const int m = (p + std::max(repeat, 0)) % 6;
                        if (cold) flush_cache<<<1024, 256, 0, s.st>>>(s.flush, 32 * 1024 * 1024);
                        ck(cudaEventRecord(s.begin, s.st), "bench begin");
                        for (int it = 0; it < iters; ++it)
                            launch_score(s, modes[(size_t) m], s.aux_score, T, budget, s.pos, {});
                        ck(cudaEventRecord(s.end, s.st), "bench end");
                        ck(cudaEventSynchronize(s.end), "bench wait");
                        float ms = 0.f;
                        ck(cudaEventElapsedTime(&ms, s.begin, s.end), "bench elapsed");
                        if (repeat >= 0) times[(size_t) m][(size_t) repeat] = (ms * 1000.f) / (float) iters;
                    }
                }
                for (auto& v : times) std::sort(v.begin(), v.end());
                const float b0 = times[0][2], b1 = times[1][2];
                const float m1 = times[2][2], m1p = times[3][2];
                const float m2 = times[4][2], m2p = times[5][2];
                std::printf(
                    "pos=%6d T=%2d | fp32=%.2fus (pers=%.2fus) | m1=%.2fus (%+.1f%%) m1p=%.2fus (%+.1f%%) | m2=%.2fus (%+.1f%%) m2p=%.2fus (%+.1f%%)\n",
                    pos, T, b0, b1,
                    m1, (m1 - b0) / b0 * 100.0f, m1p, (m1p - b1) / b1 * 100.0f,
                    m2, (m2 - b0) / b0 * 100.0f, m2p, (m2p - b1) / b1 * 100.0f);
                std::fflush(stdout);
            }
        }
    }
}

}  // namespace
}  // namespace fnx

int main(int argc, char** argv) {
    try {
        const std::string arg = argc > 1 ? argv[1] : "";
        if (arg != "--bench-only") fnx::run_parity();
        if (arg == "--bench" || arg == "--bench-only") fnx::run_bench();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "QSA_FDOT2_ERROR: %s\n", e.what());
        return 1;
    }
}

#define FNX_KPART 99
#include "../src/kernels.cu"
#include "../src/hip/peer.h"
#include "wmma_gemm.h"
#include <array>
#include <chrono>
#include <memory>
#include <thread>
namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; }
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}
#include "hc_row_exchange.cuh"
constexpr int MaxRows = 4096, ProbeK = 10240, ProbeN = 320;
static_assert(ProbeK == D && ProbeN == LR && HC == 4, "model shape");
struct Memory {
    int dev;
    struct Block { unsigned char* raw; size_t bytes; };
    std::vector<Block> blocks;
    explicit Memory(int d) : dev(d) {}
    template<class V> V* get(size_t n, bool fine = false) {
        unsigned char* p = nullptr; size_t bytes = n * sizeof(V);
        if (fine) ck(hipExtMallocWithFlags((void**)&p, bytes + 256, hipDeviceMallocFinegrained), "fine allocation");
        else ck(cudaMalloc((void**)&p, bytes + 256), "allocation");
        ck(cudaMemset(p, 0xa5, bytes + 256), "guards"); blocks.push_back({p, bytes});
        return reinterpret_cast<V*>(p + 128);
    }
    void guards() {
        ck(cudaSetDevice(dev), "guard device"); unsigned char x[256];
        for (auto b : blocks) {
            ck(cudaMemcpy(x, b.raw, 128, cudaMemcpyDeviceToHost), "front guard");
            ck(cudaMemcpy(x + 128, b.raw + 128 + b.bytes, 128, cudaMemcpyDeviceToHost), "back guard");
            for (auto v : x) if (v != 0xa5) throw std::runtime_error("guard overwrite");
        }
    }
    ~Memory() { cudaSetDevice(dev); for (auto b : blocks) cudaFree(b.raw); }
};
__global__ void hcs_init(__half* x, size_t n, unsigned seed, int pattern) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
        unsigned u = (unsigned)e * 1664525u + 1013904223u + seed; u ^= u >> 13;
        float v = pattern == 2 ? 0.f : (int(u % 65537) - 32768) * 0x1p-15f;
        if (pattern == 1) v = e % 31 == 0 ? v * 4.f : v * 0.015625f;
        x[e] = __float2half_rn(v);
    }
}
__global__ void hcs_flush(unsigned* x, size_t n) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x)
        x[e] = x[e] * 1664525u + 1013904223u;
}
struct State {
    int dev; Memory mem; cudaStream_t st; __half *x, *w, *lo; float* down;
    unsigned* flush; HcsMail mail{}, peer{};
    State(int d) : dev(d), mem(d) {
        ck(cudaSetDevice(d), "state device"); ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        x = mem.get<__half>((size_t)MaxRows * ProbeK); w = mem.get<__half>((size_t)ProbeN * ProbeK);
        lo = mem.get<__half>((size_t)MaxRows * ProbeN); down = mem.get<float>((size_t)MaxRows * ProbeN);
        flush = mem.get<unsigned>(32 * 1024 * 1024);
        mail.capacity_words = MaxRows / 2 * ProbeN / 2;
        mail.capacity_blocks = (mail.capacity_words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK;
        mail.data = mem.get<unsigned>(2 * mail.capacity_words, true);
        mail.ready = mem.get<unsigned>(2 * mail.capacity_blocks, true);
        mail.generation = mem.get<unsigned>(1); mail.errors = mem.get<unsigned>(1);
        int khz = 0; ck(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, d), "wall clock rate");
        mail.timeout_ticks = (unsigned long long)khz * 1000;
    }
    void inputs(int pattern) {
        ck(cudaSetDevice(dev), "input device");
        hcs_init<<<1024, 256, 0, st>>>(x, (size_t)MaxRows * ProbeK, 0, pattern);
        hcs_init<<<1024, 256, 0, st>>>(w, (size_t)ProbeN * ProbeK, 137, pattern);
    }
    void reset(unsigned generation) {
        ck(cudaSetDevice(dev), "reset device");
        ck(cudaMemsetAsync(lo, 0xa5, (size_t)MaxRows * ProbeN * 2, st), "lo reset");
        ck(cudaMemsetAsync(down, 0xa5, (size_t)MaxRows * ProbeN * 4, st), "down reset");
        ck(cudaMemsetAsync(mail.data, 0, 2 * mail.capacity_words * 4, st), "mail reset");
        ck(cudaMemsetAsync(mail.ready, 0, 2 * mail.capacity_blocks * 4, st), "ready reset");
        ck(cudaMemsetAsync(mail.errors, 0, 4, st), "error reset");
        ck(cudaMemcpyAsync(mail.generation, &generation, 4, cudaMemcpyHostToDevice, st), "generation reset");
        ck(cudaStreamSynchronize(st), "reset wait");
    }
    std::vector<uint16_t> snapshot(bool check_cleared_flags = true) {
        ck(cudaSetDevice(dev), "read device"); unsigned error = 0;
        ck(cudaMemcpy(&error, mail.errors, 4, cudaMemcpyDeviceToHost), "error read");
        if (error) throw std::runtime_error("mailbox error " + std::to_string(error));
        std::vector<uint16_t> a((size_t)MaxRows * ProbeN);
        ck(cudaMemcpy(a.data(), lo, a.size() * 2, cudaMemcpyDeviceToHost), "lo read");
        if (check_cleared_flags) {
            std::vector<unsigned> flags(2 * mail.capacity_blocks);
            ck(cudaMemcpy(flags.data(), mail.ready, flags.size() * 4, cudaMemcpyDeviceToHost), "consumed flags");
            if (std::any_of(flags.begin(), flags.end(), [](unsigned v) { return v != 0; }))
                throw std::runtime_error("consumed mailbox retained a ready tag");
        }
        mem.guards(); return a;
    }
    std::vector<unsigned> partial() {
        std::vector<unsigned> a((size_t)MaxRows * ProbeN);
        ck(cudaSetDevice(dev), "partial device");
        ck(cudaMemcpy(a.data(), down, a.size() * 4, cudaMemcpyDeviceToHost), "down read"); return a;
    }
    ~State() { cudaSetDevice(dev); cudaStreamDestroy(st); }
};
using States = std::array<std::unique_ptr<State>, 2>;
void sync(States& s) {
    for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "sync device"); ck(cudaStreamSynchronize(s[c]->st), "sync"); }
}
void launch(State& s, int rows, int mode) {
    // A 64-row tile boundary makes each card's output arithmetic identical.
    const int boundary = ((rows + 127) / 128) * 64;
    const int begin = mode == 0 || s.dev == 0 ? 0 : boundary;
    const int count = mode == 0 ? rows : s.dev == 0 ? boundary : rows - boundary;
    auto* input = reinterpret_cast<uint16_t*>(s.x + (size_t)begin * ProbeK);
    auto* output = s.down + (size_t)begin * ProbeN;
    auto* lo = s.lo + (size_t)begin * ProbeN;
    if (mode) hcs_tick<<<1, 1, 0, s.st>>>(s.mail.generation);
    if (!strata_pf_gemm_narrow_f16_ld(input, ProbeK, reinterpret_cast<uint16_t*>(s.w), ProbeK,
                                    output, count, ProbeN, ProbeK, ProbeN, s.st))
        throw std::runtime_error("narrow GEMM dispatch rejected");
    if (mode != 2) hcs_lo<<<(count * ProbeN + 255) / 256, 256, 0, s.st>>>(output, count * ProbeN, lo);
    if (mode) {
        const int words = count * ProbeN / 2, blocks = (words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK;
        if (mode == 2) hcs_push<true><<<blocks, 256, 0, s.st>>>(output, lo, words, s.mail, s.peer);
        else hcs_push<false><<<blocks, 256, 0, s.st>>>(output, lo, words, s.mail, s.peer);
        const int other_begin = s.dev == 0 ? boundary : 0, other_words = (rows - count) * ProbeN / 2;
        hcs_pull<<<(other_words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK, 256, 0, s.st>>>(
            s.lo + (size_t)other_begin * ProbeN, other_words, s.mail);
    }
    ck(cudaGetLastError(), "HC launch");
}
struct Graphs {
    cudaGraph_t graph[2]{}; cudaGraphExec_t exec[2]{};
    Graphs(States& s, int rows, int mode, int loops) {
        for (int c = 0; c < 2; ++c) {
            ck(cudaSetDevice(c), "capture device"); ck(cudaStreamBeginCapture(s[c]->st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < loops; ++i) launch(*s[c], rows, mode);
            ck(cudaStreamEndCapture(s[c]->st, &graph[c]), "end capture");
            ck(cudaGraphInstantiate(&exec[c], graph[c], nullptr, nullptr, 0), "instantiate");
        }
    }
    void run(States& s) { for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "graph device"); ck(cudaGraphLaunch(exec[c], s[c]->st), "graph launch"); } }
    ~Graphs() { for (int c = 0; c < 2; ++c) { cudaSetDevice(c); cudaGraphExecDestroy(exec[c]); cudaGraphDestroy(graph[c]); } }
};
void negative_checks() {
    int checks = 0;
    for (int dev = 0; dev < 2; ++dev) {
        State state(dev);
        // The production limit remains one second. Use 20 ms for negative
        // fixtures so all missing/stale-mailbox cases finish promptly.
        state.mail.timeout_ticks /= 50;
        for (int kind = 0; kind < 3; ++kind) {
            state.reset(2);
            if (kind == 1) {
                std::vector<unsigned> stale(2 * state.mail.capacity_blocks, 1u);
                ck(cudaMemcpy(state.mail.ready, stale.data(), stale.size() * 4, cudaMemcpyHostToDevice), "stale flags");
            }
            if (kind == 2) {
                const unsigned error = 1;
                ck(cudaMemcpy(state.mail.errors, &error, 4, cudaMemcpyHostToDevice), "existing error");
            }
            const auto start = std::chrono::steady_clock::now();
            hcs_pull<<<state.mail.capacity_blocks, 256, 0, state.st>>>(state.lo, state.mail.capacity_words, state.mail);
            ck(cudaGetLastError(), "negative pull launch");
            ck(cudaStreamSynchronize(state.st), "negative pull wait");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            unsigned error = 0;
            ck(cudaMemcpy(&error, state.mail.errors, 4, cudaMemcpyDeviceToHost), "negative error");
            if (error != 1 || ms > 3000) throw std::runtime_error("unbounded or unreported mailbox failure");
            ck(cudaMemset(state.mail.errors, 0, 4), "clear checked error");
            const auto got = state.snapshot(false);
            if (std::any_of(got.begin(), got.end(), [](uint16_t v) { return v != 0xa5a5u; }))
                throw std::runtime_error("failed pull modified output");
            std::printf("HC_ROW_SPLIT_ERROR card=%d kind=%d bounded_ms=%.3f guards=pass output=unchanged\n", dev, kind, ms);
            ++checks;
        }
    }
    std::printf("PASS %d missing/stale/existing-error checks\n", checks);
}
void variable_rows_wrap_checks(States& s) {
    for (auto& x : s) { x->inputs(0); x->reset(0); }
    for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "old large device"); launch(*s[c], 4096, 2); }
    sync(s);
    for (auto& x : s) x->snapshot();
    // Simulate many shorter calls without resetting mailboxes. The first
    // generation used every slot; later short calls leave the tail unused.
    for (auto& x : s) x->inputs(1);
    for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "new reference device"); launch(*s[c], 4096, 0); }
    sync(s);
    const auto expected = s[0]->snapshot();
    if (expected != s[1]->snapshot()) throw std::runtime_error("new wrap reference differs by card");
    for (int c = 0; c < 2; ++c) {
        ck(cudaSetDevice(c), "near wrap device");
        const unsigned generation = 0xfffffffcu;
        ck(cudaMemcpy(s[c]->mail.generation, &generation, 4, cudaMemcpyHostToDevice), "near wrap generation");
    }
    int checks = 0;
    for (int rows : {3072, 3072, 4096}) {
        ck(cudaSetDevice(0), "early card"); launch(*s[0], rows, 2);
        // The receiver must not accept a tail's stale generation-one flag
        // before this delayed peer writes the new generation-one payload.
        if (rows == 4096) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ck(cudaSetDevice(1), "delayed card"); launch(*s[1], rows, 2);
        sync(s);
        for (auto& x : s) {
            if (x->snapshot() != expected) throw std::runtime_error("variable-row wrap returned a stale payload");
            ++checks;
        }
    }
    std::printf("PASS %d variable-row wrap/delayed-peer/cleared-flag checks\n", checks);
}
int run() {
    std::string why; if (!peer_enable_and_probe({0, 1}, why)) throw std::runtime_error(why);
    std::printf("peer probe: %s\n", why.c_str());
    States s; for (int c = 0; c < 2; ++c) s[c] = std::make_unique<State>(c);
    for (int c = 0; c < 2; ++c) s[c]->peer = s[1 - c]->mail;
    int checks = 0;
    for (int rows : {2048, 2112, 2560, 3072, 4032, 4095, 4096}) for (int pattern = 0; pattern < 3; ++pattern) {
        const unsigned seed = pattern == 1 ? 0xfffffffcu : 1u;
        for (auto& x : s) { x->inputs(pattern); x->reset(seed); }
        for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "ref device"); launch(*s[c], rows, 0); } sync(s);
        auto ref = s[0]->snapshot(); auto down = s[0]->partial();
        if (ref != s[1]->snapshot() || down != s[1]->partial()) throw std::runtime_error("reference card mismatch");
        for (int mode : {1, 2}) {
            for (auto& x : s) x->reset(seed);
            Graphs graph(s, rows, mode, 1);
            for (int replay = 0; replay < 4; ++replay) {
                if (!replay) { for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "direct device"); launch(*s[c], rows, mode); } }
                else graph.run(s);
                sync(s);
                for (int c = 0; c < 2; ++c) {
                    if (s[c]->snapshot() != ref) throw std::runtime_error("lo16 mismatch C=" + std::to_string(rows) + " mode=" + std::to_string(mode));
                    auto got = s[c]->partial(); const int boundary = ((rows + 127) / 128) * 64;
                    const size_t begin = (size_t)(c ? boundary : 0) * ProbeN, end = (size_t)(c ? rows : boundary) * ProbeN;
                    for (size_t i = 0; i < got.size(); ++i)
                        if (got[i] != (i >= begin && i < end ? down[i] : 0xa5a5a5a5u)) throw std::runtime_error("FP32 shard/guard mismatch");
                    ++checks;
                }
            }
        }
    }
    std::printf("PASS %d exact HC-down FP32/lo16/card/guard/direct/graph/wrap checks\n", checks); std::fflush(stdout);
    variable_rows_wrap_checks(s);
    negative_checks();
    for (int rows : {2048, 3072, 4096}) for (bool cold : {false, true}) {
        for (auto& x : s) { x->inputs(0); x->reset(17); }
        std::array<std::unique_ptr<Graphs>, 3> graphs;
        const int loops = cold ? 1 : 16;
        for (int m = 0; m < 3; ++m) graphs[m] = std::make_unique<Graphs>(s, rows, m, loops);
        std::array<std::vector<double>, 3> elapsed;
        for (int rep = -2; rep < 13; ++rep) for (int p = 0; p < 3; ++p) {
            const int mode = (p + std::max(rep, 0)) % 3;
            if (cold) {
                for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "flush device"); hcs_flush<<<1024, 256, 0, s[c]->st>>>(s[c]->flush, 32 * 1024 * 1024); } sync(s);
            } else { graphs[mode]->run(s); sync(s); }
            const auto begin = std::chrono::steady_clock::now(); graphs[mode]->run(s); sync(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / loops;
            if (rep >= 0) elapsed[mode].push_back(us);
        }
        for (int mode = 0; mode < 3; ++mode) {
            std::sort(elapsed[mode].begin(), elapsed[mode].end());
            std::printf("HC_ROW_SPLIT_TIME C=%d cold=%d mode=%d wall_us=%.3f\n", rows, (int)cold, mode, elapsed[mode][6]);
        }
        for (auto& x : s) x->snapshot(); std::fflush(stdout);
    }
    return 0;
}
}
int main() {
#if defined(_WIN32)
    _putenv_s("STRATA_PF_NARROW_BK64", "1");
    _putenv_s("STRATA_PF_HC_DOWN_INCREMENT", "1");
#else
    setenv("STRATA_PF_NARROW_BK64", "1", 1);
    setenv("STRATA_PF_HC_DOWN_INCREMENT", "1", 1);
#endif
    try { return fnx::run(); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}

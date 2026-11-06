// Bit-exact regression of the HC compact layout against the existing 96-row layout.
#define FNX_KPART 3
#include "../src/kernels.cu"
#include <iostream>
#include <functional>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
struct Alloc {
    struct Block { unsigned char* raw; size_t bytes; };
    std::vector<Block> blocks;
    template<class V> V* make(size_t n) {
        unsigned char* raw = nullptr;
        ck(cudaMalloc((void**)&raw, n * sizeof(V) + 256), "alloc");
        ck(cudaMemset(raw, 0xa5, n * sizeof(V) + 256), "init guards");
        blocks.push_back({raw, n * sizeof(V)});
        return reinterpret_cast<V*>(raw + 128);
    }
    void guards() {
        unsigned char h[256];
        for (const auto& b : blocks) {
            ck(cudaMemcpy(h, b.raw, 128, cudaMemcpyDeviceToHost), "front guard");
            ck(cudaMemcpy(h + 128, b.raw + 128 + b.bytes, 128, cudaMemcpyDeviceToHost), "back guard");
            for (auto v : h) if (v != 0xa5) throw std::runtime_error("HC guard overwritten");
        }
    }
    ~Alloc() { for (const auto& b : blocks) cudaFree(b.raw); }
};
template<class V> void upload(V* d, const std::vector<V>& h, cudaStream_t st) {
    ck(cudaMemcpyAsync(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice, st), "upload");
}
__global__ void timeout_probe(unsigned* bar, unsigned* err, unsigned long long ticks) {
    grid_sync(bar, err, 2, ticks); // one resident block: the second arrival is deliberately absent
}

template<int T, bool ONE> size_t run(int dev, unsigned long long ticks) {
    Alloc mem; HcArgs a{}; a.timeout_ticks = ticks;
    a.R = mem.make<float>(T * D);
    auto wn = mem.make<float>(D); a.w_norm = wn;
    auto bo = mem.make<float>(T * N), inj = mem.make<float>(T * HC);
    auto wi = mem.make<uint16_t>(HC * D);
    a.w_down = {mem.make<int8_t>(LR * D), mem.make<uint16_t>(LR * D / 32)};
    a.w_up = {mem.make<int8_t>(D * LR), mem.make<uint16_t>(D * LR / 32)};
    a.eps = 1e-5f; a.mixed = mem.make<float>(T * N); a.inject = mem.make<float>(T * HC);
    a.part = mem.make<float>(H_NS * T * H_ROWS); a.ssp = mem.make<float>(T * H_NS);
    a.loq = mem.make<int8_t>(T * LR); a.lod = mem.make<float>(T * LR / 32);
    a.bar = mem.make<unsigned>(1); a.err = mem.make<unsigned>(1);
    auto q8 = mem.make<block_q8_1>(T * N / 32);
    auto tq = mem.make<int8_t>(T * N); auto td = mem.make<float>(T * N / 32);
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    std::vector<int8_t> w(LR * D);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (int)((i * 19 + i / 31) % 255) - 127;
    upload(a.w_down.q, w, st); upload(a.w_up.q, w, st);
    std::vector<uint16_t> sc(LR * D / 32);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = i % 7 == 0 ? 0 : i % 2 ? 0xa400 : 0x2000;
    upload(a.w_down.d, sc, st); upload(a.w_up.d, sc, st);
    std::vector<uint16_t> hw(HC * D);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = i % 7 == 0 ? 0 : i % 2 ? 0xbb80 : 0x3c00;
    upload(wi, hw, st);
    auto scratch = mem.make<float>(hc_scratch_floats(MAXT));
    const int compact_grid = hc_prepare(scratch);
    if (compact_grid != 80) throw std::runtime_error("compact grid 80 unavailable");
    hc_preset_barrier(scratch, 1);
    unsigned preset[2] = {};
    ck(cudaMemcpy(preset, scratch + hc_scratch_floats(MAXT) - 2, sizeof preset, cudaMemcpyDeviceToHost), "preset");
    if (preset[0] != hc_bar_mod(48) - 48 || preset[1] != hc_bar_mod(80) - 80)
        throw std::runtime_error("incorrect wide/compact barrier preset");
    q8_twins().push_back({(const char*)q8, T * N / 32 * sizeof(block_q8_1), tq, td});
    auto launch = [&](int mode) {
        if (mode == 0) k_hc<T, ONE, true><<<48, 768, 0, st>>>(a);
        else if (mode == 1) {
            if constexpr (T <= 12) k_hc<T, ONE, true, 64><<<48, 512, 0, st>>>(a);
            else k_hc<T, ONE, true><<<48, 768, 0, st>>>(a);
        } else hc_read(a.R, a.bo, a.inj_prev, a.w_norm, a.w_down, a.w_up, a.w_inj,
                       a.eps, a.mixed, a.inject, scratch, a.err, T, st, a.q8);
    };
    auto snapshot = [&] {
        std::vector<unsigned char> v;
        for (auto x : {std::pair<const void*, size_t>{a.R, T * D * 4}, {a.mixed, T * N * 4},
             {a.inject, T * HC * 4}, {q8, T * N / 32 * sizeof(block_q8_1)}, {tq, T * N}, {td, T * N / 32 * 4}}) {
            size_t n = v.size(); v.resize(n + x.second);
            ck(cudaMemcpy(v.data() + n, x.first, x.second, cudaMemcpyDeviceToHost), "snapshot");
        }
        unsigned err = 0; ck(cudaMemcpy(&err, a.err, 4, cudaMemcpyDeviceToHost), "error flags");
        if (err) throw std::runtime_error("HC barrier timed out");
        mem.guards(); return v;
    };
    size_t checks = 0;
    for (int mask = 0; mask < 8; ++mask) {
        a.bo = mask & 1 ? bo : nullptr; a.inj_prev = inj;
        a.w_inj = mask & 2 ? wi : nullptr;
        a.q8 = mask & 4 ? q8 : nullptr; a.tq = tq; a.td = td;
        cudaGraph_t graph[3]; cudaGraphExec_t exec[3];
        ck(cudaStreamSynchronize(st), "before capture");
        for (int mode = 0; mode < 3; ++mode) {
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
            launch(mode); ck(cudaStreamEndCapture(st, &graph[mode]), "end capture");
            ck(cudaGraphInstantiate(&exec[mode], graph[mode], nullptr, nullptr, 0), "instantiate");
        }
        for (int pattern = 0; pattern < 3; ++pattern) {
            std::vector<float> r(T * D), b(T * N), h(T * HC), norm(D);
            for (size_t i = 0; i < r.size(); ++i) r[i] = pattern == 2 ? 0 : ((int)((i * 17 + pattern * 23) % 103) - 51) * .03f;
            for (size_t i = 0; i < b.size(); ++i) b[i] = pattern == 2 ? 0 : ((int)((i * 7 + pattern * 31) % 67) - 33) * .02f;
            for (size_t i = 0; i < h.size(); ++i) h[i] = ((int)((i * 11 + pattern * 7) % 29) - 14) * .07f;
            for (size_t i = 0; i < norm.size(); ++i) norm[i] = pattern == 0 ? 1.f : ((int)(i * 5 % 19) - 9) * .1f;
            upload(bo, b, st); upload(inj, h, st); upload(wn, norm, st);
            std::vector<unsigned char> reference;
            for (int replay = 0; replay < 2; ++replay) for (int mode = 0; mode < 3; ++mode) {
                upload(a.R, r, st);
                for (auto x : {std::pair<void*, size_t>{a.mixed, T * N * 4}, {a.inject, T * HC * 4},
                     {q8, T * N / 32 * sizeof(block_q8_1)}, {tq, T * N}, {td, T * N / 32 * 4}})
                    ck(cudaMemsetAsync(x.first, 0xa5, x.second, st), "reset output");
                unsigned grid = mode == 2 && T <= 12 ? 80 : 48;
                unsigned bar = pattern == 0 ? 0 : hc_bar_mod(grid) - grid;
                ck(cudaMemcpyAsync(a.bar, &bar, 4, cudaMemcpyHostToDevice, st), "reset barrier");
                const unsigned bars[2] = {pattern ? hc_bar_mod(48) - 48 : 0, pattern ? hc_bar_mod(80) - 80 : 0};
                ck(cudaMemcpyAsync(scratch + hc_scratch_floats(MAXT) - 2, bars, sizeof bars, cudaMemcpyHostToDevice, st), "reset dispatch barriers");
                ck(cudaMemsetAsync(a.err, 0, 4, st), "reset error");
                if (replay) ck(cudaGraphLaunch(exec[mode], st), "graph replay"); else launch(mode);
                ck(cudaStreamSynchronize(st), "HC wait");
                auto got = snapshot();
                if (mode == 0 && !replay) reference = got;
                else {
                    if (reference != got) throw std::runtime_error("HC mismatch T=" + std::to_string(T) + " one=" + std::to_string(ONE) + " mask=" + std::to_string(mask) + " pattern=" + std::to_string(pattern) + " mode=" + std::to_string(mode) + " replay=" + std::to_string(replay));
                    ++checks;
                }
            }
        }
        for (int mode = 0; mode < 3; ++mode) { cudaGraphExecDestroy(exec[mode]); cudaGraphDestroy(graph[mode]); }
    }
    if constexpr (T == 16 && !ONE) {
        // Switch grid sizes on the same scratch without resetting counters between
        // reads, including wrap and one-/two-barrier kernels in one captured graph.
        std::vector<float> r(T * D), b(T * N), norm(D, 1.f);
        for (size_t i = 0; i < r.size(); ++i) r[i] = ((int)(i * 17 % 103) - 51) * .03f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = ((int)(i * 7 % 67) - 33) * .02f;
        upload(bo, b, st); upload(wn, norm, st);
        auto sequence = [&](bool dispatch) {
            for (int rows : {1, 16, 7, 13, 2, 12, 4, 16, 1}) {
                if (dispatch) hc_read(a.R, a.bo, a.inj_prev, a.w_norm, a.w_down, a.w_up, a.w_inj,
                                      a.eps, a.mixed, a.inject, scratch, a.err, rows, st, a.q8);
                else switch (rows) {
#define HC_REF(R) case R: k_hc<R, (R <= 3), true><<<48, 768, 0, st>>>(a); break
                    HC_REF(1); HC_REF(2); HC_REF(4); HC_REF(7); HC_REF(12); HC_REF(13); HC_REF(16);
#undef HC_REF
                }
            }
        };
        ck(cudaStreamSynchronize(st), "before sequence capture");
        cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
        for (int mode = 0; mode < 2; ++mode) {
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "sequence capture");
            sequence(mode); ck(cudaStreamEndCapture(st, &graph[mode]), "sequence end");
            ck(cudaGraphInstantiate(&exec[mode], graph[mode], nullptr, nullptr, 0), "sequence instantiate");
        }
        std::vector<unsigned char> reference;
        for (int replay = 0; replay < 2; ++replay) for (int mode = 0; mode < 2; ++mode) {
            upload(a.R, r, st);
            unsigned bar = hc_bar_mod(48) - 48;
            unsigned bars[2] = {bar, hc_bar_mod(80) - 80};
            ck(cudaMemcpyAsync(a.bar, &bar, 4, cudaMemcpyHostToDevice, st), "sequence counter");
            ck(cudaMemcpyAsync(scratch + hc_scratch_floats(MAXT) - 2, bars, sizeof bars, cudaMemcpyHostToDevice, st), "sequence counters");
            ck(cudaMemsetAsync(a.err, 0, 4, st), "sequence error");
            if (replay) ck(cudaGraphLaunch(exec[mode], st), "sequence replay"); else sequence(mode);
            ck(cudaStreamSynchronize(st), "sequence wait");
            auto got = snapshot();
            if (!mode && !replay) reference = got;
            else { if (got != reference) throw std::runtime_error("mixed-window HC dispatch mismatch"); ++checks; }
        }
        for (int mode = 0; mode < 2; ++mode) { cudaGraphExecDestroy(exec[mode]); cudaGraphDestroy(graph[mode]); }
    }
    q8_twins().clear();
    cudaStreamDestroy(st);
    std::cout << "card " << dev << " T " << T << " one " << ONE << " dispatch grid " << (T <= 12 ? 80 : 48) << ": " << checks << " exact checks\n";
    return checks;
}
template<int T> size_t rows(int dev, unsigned long long ticks) {
    size_t n = run<T, false>(dev, ticks);
    if constexpr (T <= 8) n += run<T, true>(dev, ticks);
    return n;
}
}
int main() {
    try {
        setenv("FNX_HC_ROW_GROUP", "64", 1);
        std::cout.setf(std::ios::unitbuf); size_t checks = 0;
        for (int dev = 0; dev < 2; ++dev) {
            fnx::ck(cudaSetDevice(dev), "device");
            int rate = 0; fnx::ck(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, dev), "wall-clock rate");
            auto ticks = (unsigned long long) rate * 1000;
            { fnx::Alloc mem; auto bar = mem.make<unsigned>(1), err = mem.make<unsigned>(1);
              fnx::ck(cudaMemset(bar, 0, 4), "bar"); fnx::ck(cudaMemset(err, 0, 4), "err");
              fnx::timeout_probe<<<1, 32>>>(bar, err, ticks / 100); // 10 ms bounded deliberate timeout
              fnx::ck(cudaDeviceSynchronize(), "timeout probe"); unsigned h = 0;
              fnx::ck(cudaMemcpy(&h, err, 4, cudaMemcpyDeviceToHost), "timeout result");
              if (h != 4) throw std::runtime_error("missing timeout flag"); }
#define ROW(T) checks += fnx::rows<T>(dev, ticks)
            ROW(1); ROW(2); ROW(3); ROW(4); ROW(5); ROW(6); ROW(7); ROW(8);
            ROW(9); ROW(10); ROW(11); ROW(12); ROW(13); ROW(14); ROW(15); ROW(16);
#undef ROW
        }
        std::cout << "HC compact: " << checks << " bit-exact checks, guards, graph replay, counter wrap and watchdog passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

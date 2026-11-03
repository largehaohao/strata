// Exact first-state/update-log rollback against the original full snapshots.
#define FNX_KPART 4
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Buffers {
    std::vector<void*> mem;
    template <class T> T* alloc(size_t n) {
        void* p; ck(cudaMalloc(&p, n * sizeof(T)), "allocate"); mem.push_back(p); return (T*) p;
    }
    ~Buffers() { for (void* p : mem) cudaFree(p); }
};
template <class T> void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
}
std::vector<unsigned char> read_bytes(const void* p, size_t n) {
    std::vector<unsigned char> out(n); ck(cudaMemcpy(out.data(), p, n, cudaMemcpyDeviceToHost), "download"); return out;
}
void same(const void* a, const void* b, size_t bytes, const char* what) {
    if (read_bytes(a, bytes) != read_bytes(b, bytes)) throw std::runtime_error(what);
}

void run_window(int T, int nv, long& checks, long& restores) {
    constexpr int pad = 64;
    const int nk = nv / 3, ch = (2 * nk + nv) * HS, stride = ch + nv * HS;
    const size_t ns = (size_t) nv * HS * HS, nh = (size_t) 2 * 3 * ch;
    const size_t qbytes = (size_t) MAXT * nv * HS / 32 * sizeof(block_q8_1);
    const size_t nf = (size_t) MAXT * nv * HS;
    const size_t scratch = nf + (size_t) nv * MAXT * GDN_Q + MAXT * nv;
    const size_t log_floats = (size_t) (MAXT - 2) * nv * GDN_REPLAY_STRIDE;
    Buffers mem;
    auto* qkv = mem.alloc<float>((size_t) MAXT * stride);
    auto* ab = mem.alloc<float>(MAXT * 2 * nv);
    auto* conv = mem.alloc<float>(ch * 4);
    auto* dt = mem.alloc<float>(nv); auto* alpha = mem.alloc<float>(nv); auto* norm = mem.alloc<float>(HS);
    auto* vmap = mem.alloc<int>(nv); auto* ctl = mem.alloc<int>(2);
    float *state[2], *hist[2], *work[2], *hsnap[2], *y32[2], *twin_d[2];
    int8_t* twin_q[2]; unsigned char* yq[2];
    for (int k = 0; k < 2; ++k) {
        state[k] = mem.alloc<float>(ns + pad); hist[k] = mem.alloc<float>(nh + pad);
        work[k] = mem.alloc<float>(scratch); hsnap[k] = mem.alloc<float>((size_t) MAXT * 3 * ch + pad);
        y32[k] = mem.alloc<float>(nf + pad); twin_d[k] = mem.alloc<float>(nf / 32 + pad);
        twin_q[k] = mem.alloc<int8_t>(nf + pad * 4); yq[k] = mem.alloc<unsigned char>(qbytes + pad * 4);
    }
    auto* old_snap = mem.alloc<float>((MAXT - 1) * ns + pad);
    auto* base_raw = mem.alloc<float>(ns + 2 * pad); auto* base = base_raw + pad;
    auto* log_raw = mem.alloc<float>(log_floats + 2 * pad); auto* log = log_raw + pad;
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    q8_twins().clear();
    for (int k = 0; k < 2; ++k) q8_twins().push_back({(const char*) yq[k], qbytes, twin_q[k], twin_d[k]});
    auto core = [&](bool logging) {
        const int k = logging;
        gdn_core(qkv, ab, hist[k], conv, dt, alpha, norm, state[k], yq[k], ctl, work[k],
                 logging ? base : old_snap, hsnap[k], nk, nv, vmap, T, 1e-6f, stream, y32[k], logging ? log : nullptr);
    };
    auto* rotated_base = mem.alloc<float>(ns);
    auto* rotated_log = mem.alloc<float>(log_floats);
    auto* rotated_state = mem.alloc<float>(ns + pad);
    auto* replay_jobs = mem.alloc<GdnReplayJob>(2);
    const std::vector<GdnReplayJob> jobs{{state[1], base, log}, {rotated_state, rotated_base, rotated_log}};
    upload(replay_jobs, jobs, stream); ck(cudaStreamSynchronize(stream), "jobs ready");
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture"); core(true);
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    std::vector<cudaGraph_t> restore_graphs(T);
    std::vector<cudaGraphExec_t> restore_execs(T);
    for (int keep = 1; keep < T; ++keep) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "restore capture");
        gdn_replay(replay_jobs, 2, nv, keep, stream);
        ck(cudaStreamEndCapture(stream, &restore_graphs[keep]), "restore end capture");
        ck(cudaGraphInstantiate(&restore_execs[keep], restore_graphs[keep], nullptr, nullptr, 0), "restore instantiate");
    }
    std::vector<float> x(MAXT * stride), gates(MAXT * 2 * nv), cw(ch * 4), d(nv), a(nv), n(HS), s(ns), h(nh);
    std::vector<int> maps(nv);
    for (int i = 0; i < nv; ++i) maps[i] = i % nk | ((i < nk) << 8);
    for (int rep = 0; rep < 3; ++rep) for (bool replay : {false, true}) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = rep == 2 ? 0 : std::sin((i + 1) * 0.019 + rep) * 1.25f;
        for (size_t i = 0; i < gates.size(); ++i) gates[i] = std::cos(i * 0.11 + rep) * 2.f;
        for (size_t i = 0; i < cw.size(); ++i) cw[i] = std::sin(i * 0.13 + rep) * 0.3f;
        for (size_t i = 0; i < d.size(); ++i) d[i] = (i % 5) * 0.13f - 0.2f, a[i] = -0.4f - (i % 7) * 0.1f;
        for (size_t i = 0; i < n.size(); ++i) n[i] = 0.8f + std::sin(i * 0.1) * 0.1f;
        for (size_t i = 0; i < s.size(); ++i) s[i] = rep == 2 ? 0 : std::sin(i * 0.17 + rep) * 0.1f;
        for (size_t i = 0; i < h.size(); ++i) h[i] = rep == 2 ? 0 : std::cos(i * 0.21 + rep) * 0.4f;
        upload(qkv, x, stream); upload(ab, gates, stream); upload(conv, cw, stream); upload(dt, d, stream);
        upload(alpha, a, stream); upload(norm, n, stream); upload(vmap, maps, stream);
        std::vector<int> control{37, rep}; upload(ctl, control, stream);
        for (int k = 0; k < 2; ++k) {
            ck(cudaMemsetAsync(state[k], 0xA5, (ns + pad) * 4, stream), "state guards"); upload(state[k], s, stream);
            ck(cudaMemsetAsync(hist[k], 0xA5, (nh + pad) * 4, stream), "hist guards"); upload(hist[k], h, stream);
            ck(cudaMemsetAsync(work[k], 0, scratch * 4, stream), "work");
            for (auto item : {std::make_pair((void*) hsnap[k], ((size_t) MAXT * 3 * ch + pad) * 4),
                              std::make_pair((void*) y32[k], (nf + pad) * 4), std::make_pair((void*) twin_q[k], nf + pad * 4),
                              std::make_pair((void*) twin_d[k], (nf / 32 + pad) * 4), std::make_pair((void*) yq[k], qbytes + pad * 4)})
                ck(cudaMemsetAsync(item.first, 0xA5, item.second, stream), "output guards");
        }
        ck(cudaMemsetAsync(old_snap, 0xA5, ((MAXT - 1) * ns + pad) * 4, stream), "old snapshot");
        ck(cudaMemsetAsync(base_raw, 0xA5, (ns + 2 * pad) * 4, stream), "base guards");
        ck(cudaMemsetAsync(log_raw, 0xA5, (log_floats + 2 * pad) * 4, stream), "log guards");
        core(false); if (replay) ck(cudaGraphLaunch(exec, stream), "replay"); else core(true);
        ck(cudaStreamSynchronize(stream), "core wait");
        same(state[0], state[1], (ns + pad) * 4, "final state mismatch");
        same(hist[0], hist[1], (nh + pad) * 4, "history mismatch");
        same(hsnap[0], hsnap[1], ((size_t) MAXT * 3 * ch + pad) * 4, "history snapshot mismatch");
        same(yq[0], yq[1], qbytes + pad * 4, "Q8 output mismatch");
        same(y32[0], y32[1], (nf + pad) * 4, "float output mismatch");
        same(twin_q[0], twin_q[1], nf + pad * 4, "split Q8 mismatch");
        same(twin_d[0], twin_d[1], (nf / 32 + pad) * 4, "split scales mismatch");
        for (auto span : {std::make_pair(base_raw, (size_t) pad), std::make_pair(base + ns, (size_t) pad),
                          std::make_pair(log_raw, (size_t) pad), std::make_pair(log + log_floats, (size_t) pad)})
            if (read_bytes(span.first, span.second * 4) != std::vector<unsigned char>(span.second * 4, 0xA5))
                throw std::runtime_error("base/log guard overwrite");
        if (T > 1) same(old_snap, base, ns * 4, "first state mismatch");
        ++checks;
        // Rotate heads in a second job. The original full snapshots are an
        // independent byte oracle for both job destinations and source offsets.
        const auto first_bytes = read_bytes(base, ns * 4), log_bytes = read_bytes(log, log_floats * 4);
        std::vector<unsigned char> rb(ns * 4), rl(log_floats * 4);
        const size_t head_bytes = (size_t) HS * HS * 4, record_bytes = GDN_REPLAY_STRIDE * 4;
        for (int head = 0; head < nv; ++head) {
            const int from = (head + 1) % nv;
            std::memcpy(rb.data() + head * head_bytes, first_bytes.data() + from * head_bytes, head_bytes);
            for (int t = 0; t < MAXT - 2; ++t)
                std::memcpy(rl.data() + ((size_t) t * nv + head) * record_bytes,
                            log_bytes.data() + ((size_t) t * nv + from) * record_bytes, record_bytes);
        }
        ck(cudaMemcpyAsync(rotated_base, rb.data(), rb.size(), cudaMemcpyHostToDevice, stream), "rotated first");
        ck(cudaMemcpyAsync(rotated_log, rl.data(), rl.size(), cudaMemcpyHostToDevice, stream), "rotated updates");
        for (int keep = 1; keep < T; ++keep) {
            const auto expected = read_bytes(old_snap + (size_t) (keep - 1) * ns, ns * 4);
            std::vector<unsigned char> rotated_expected(ns * 4);
            for (int head = 0; head < nv; ++head)
                std::memcpy(rotated_expected.data() + head * head_bytes,
                            expected.data() + ((head + 1) % nv) * head_bytes, head_bytes);
            ck(cudaMemsetAsync(rotated_state, 0xA5, (ns + pad) * 4, stream), "rotated output guard");
            if (replay) ck(cudaGraphLaunch(restore_execs[keep], stream), "restore replay");
            else gdn_replay(replay_jobs, 2, nv, keep, stream);
            ck(cudaStreamSynchronize(stream), "restore wait");
            if (read_bytes(state[1], ns * 4) != expected || read_bytes(rotated_state, ns * 4) != rotated_expected)
                throw std::runtime_error("restored state mismatch: T=" + std::to_string(T) + " keep=" + std::to_string(keep));
            for (const float* end : {state[1] + ns, rotated_state + ns})
                if (read_bytes(end, pad * 4) != std::vector<unsigned char>(pad * 4, 0xA5))
                    throw std::runtime_error("restored state guard overwrite");
            restores += 2;
        }
    }
    for (int keep = 1; keep < T; ++keep) {
        ck(cudaGraphExecDestroy(restore_execs[keep]), "destroy restore exec");
        ck(cudaGraphDestroy(restore_graphs[keep]), "destroy restore graph");
    }
    q8_twins().clear();
    ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
    ck(cudaStreamDestroy(stream), "destroy stream");
}
}  // namespace fnx

int main() {
    try {
        long checks = 0, restores = 0;
        int devices = 0; fnx::ck(cudaGetDeviceCount(&devices), "devices");
        if (devices < 2) throw std::runtime_error("requires two gfx1100 devices");
        for (int dev = 0; dev < 2; ++dev) {
            hipDeviceProp_t prop{}; fnx::ck(hipGetDeviceProperties(&prop, dev), "device properties");
            if (std::string(prop.gcnArchName).rfind("gfx1100", 0) != 0) throw std::runtime_error("requires gfx1100");
            fnx::ck(cudaSetDevice(dev), "device");
            for (int nv : {3, 24}) for (int t = 1; t <= fnx::MAXT; ++t) fnx::run_window(t, nv, checks, restores);
            std::cout << "card " << dev << ": cumulative " << checks << " exact cores, " << restores << " exact restored states" << std::endl;
        }
        std::cout << "PASS GDN replay: " << checks << " core and " << restores << " restoration comparisons" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}

#define FNX_KPART 99
#include "../src/split/src/kernels.cu"
#include <iostream>
#include <array>

namespace fnx {
#include "gdn_log.cuh"
std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> empty; return empty; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__global__ void restore_log(const float* base, const float* log, float* S, int nv, int keep, int mode) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nv * HS * HS) return;
    const int lv = i / (HS * HS), row = (i / HS) % HS, col = i % HS;
    float s = base[i];
    for (int t = 1; t < keep; ++t) {
        const float* p = log + ((size_t) (t - 1) * nv + lv) * GDN_LOG_STRIDE;
        const float g = p[0], k = p[1 + row], delta = p[1 + HS + col];
        if (mode == 0) s = g * s + k * delta;
        else if (mode == 1) s = fmaf(k, delta, __fmul_rn(g, s));
        else if (mode == 2) s = fmaf(g, s, __fmul_rn(k, delta));
        else if (mode == 3) s = __fadd_rn(__fmul_rn(g, s), __fmul_rn(k, delta));
        else s = row % 4 == 0 ? fmaf(g, s, __fmul_rn(k, delta)) : fmaf(k, delta, __fmul_rn(g, s));
    }
    S[i] = s;
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

template <int T> void run_window(int nv, std::array<long, 5>& failures, long& checks, long& restores) {
    constexpr int pad = 64;
    const int nk = nv / 3, ch = (2 * nk + nv) * HS, stride = ch + nv * HS;
    const size_t ns = (size_t) nv * HS * HS, nh = (size_t) 2 * 3 * ch;
    const size_t qbytes = (size_t) MAXT * nv * HS / 32 * sizeof(block_q8_1);
    const size_t nf = (size_t) MAXT * nv * HS;
    const size_t scratch = nf + (size_t) nv * MAXT * GDN_Q + MAXT * nv;
    const size_t log_floats = (size_t) (MAXT - 2) * nv * GDN_LOG_STRIDE;
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
    auto core = [&](bool logging) {
        const int k = logging;
        if (logging)
            k_gdn_core_log<T><<<nv * GDN_Q, GDN_THREADS, 0, stream>>>(qkv, ab, hist[k], conv, dt, alpha, norm,
                state[k], (block_q8_1*) yq[k], ctl, work[k], base, hsnap[k], nk, nv, vmap, 1e-6f,
                twin_q[k], twin_d[k], y32[k], GdnLanes{}, log);
        else
            k_gdn_core<T><<<nv * GDN_Q, GDN_THREADS, 0, stream>>>(qkv, ab, hist[k], conv, dt, alpha, norm,
                state[k], (block_q8_1*) yq[k], ctl, work[k], old_snap, hsnap[k], nk, nv, vmap, 1e-6f,
                twin_q[k], twin_d[k], y32[k], GdnLanes{});
        ck(cudaGetLastError(), "core launch");
    };
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture"); core(true);
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
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
        for (int keep = 1; keep < T; ++keep) {
            const auto expected = read_bytes(old_snap + (size_t) (keep - 1) * ns, ns * 4);
            for (int mode = 0; mode < 5; ++mode) {
                restore_log<<<(ns + 255) / 256, 256, 0, stream>>>(base, log, state[1], nv, keep, mode);
                ck(cudaStreamSynchronize(stream), "restore wait");
                const auto got = read_bytes(state[1], ns * 4);
                if (got != expected) {
                    ++failures[mode];
                    if (T == 3 && nv == 3 && rep == 0 && !replay && keep == 2) {
                        size_t count = 0, first = ns;
                        for (size_t i = 0; i < ns; ++i) if (std::memcmp(got.data()+i*4, expected.data()+i*4, 4)) { ++count; first = std::min(first, i); }
                        float want, actual; std::memcpy(&want, expected.data()+first*4, 4); std::memcpy(&actual,got.data()+first*4,4);
                        auto hb=read_bytes(base,ns*4),hl=read_bytes(log,log_floats*4);
                        const size_t head=first/(HS*HS), row=(first/HS)%HS,col=first%HS;
                        float start,g,k,d;std::memcpy(&start,hb.data()+first*4,4);
                        std::memcpy(&g,hl.data()+head*GDN_LOG_STRIDE*4,4);
                        std::memcpy(&k,hl.data()+(head*GDN_LOG_STRIDE+1+row)*4,4);
                        std::memcpy(&d,hl.data()+(head*GDN_LOG_STRIDE+1+HS+col)*4,4);
                        std::cout << "mode "<<mode<<" bad "<<count<<" first "<<first<<" want "<<std::hexfloat<<want<<" got "<<actual<<" base "<<start<<" g "<<g<<" k "<<k<<" delta "<<d<<std::defaultfloat<<std::endl;
                    }
                }
            }
            ++restores;
        }
    }
    ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
    ck(cudaStreamDestroy(stream), "destroy stream");
}
}  // namespace fnx

int main() {
    try {
        std::array<long, 5> failures{}; long checks = 0, restores = 0;
        for (int dev = 0; dev < 2; ++dev) {
            fnx::ck(cudaSetDevice(dev), "device");
            for (int nv : {3, 24}) for (int t = 1; t <= 16; ++t) {
#define CASE(T) case T: fnx::run_window<T>(nv, failures, checks, restores); break
                switch (t) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
                             CASE(9); CASE(10); CASE(11); CASE(12); CASE(13); CASE(14); CASE(15); CASE(16); }
#undef CASE
            }
            std::cout << "card " << dev << " core checks " << checks << " restore cases " << restores
                      << " mismatches expr/fma_k/fma_g/separate/mixed " << failures[0] << '/' << failures[1] << '/' << failures[2] << '/' << failures[3] << '/' << failures[4] << std::endl;
        }
        if (failures[0] && failures[1] && failures[2] && failures[3] && failures[4]) throw std::runtime_error("no exact replay mode");
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}

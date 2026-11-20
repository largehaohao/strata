// Opt-in two-card HIP HC-down partition. Included inside namespace fnx after Pf.
#if FNX_HIP && FNX_NCARD == 2 && !FNX_27B
namespace {
#include "hc_row_exchange.cuh"
struct HcsState { int dev = -1; HcsMail mail{}; };
HcsState hcs_states[2];
bool hcs_enabled() {
    static const auto mode = env_int("FNX_PF_HC_ROW_SPLIT", 0);
    if (mode != 0 && mode != 1) throw std::runtime_error("FNX_PF_HC_ROW_SPLIT: expected 0 or 1");
    return mode != 0;
}
}
void hc_row_split_prepare(int card, bool peer, int lanes, unsigned* errors) {
    if (!hcs_enabled()) return;
    if (!peer || lanes != 1 || card < 0 || card > 1 || !attn_head_split())
        throw std::runtime_error("HC row split requires two probed peers, one lane and head split");
    int dev = -1; ckc(cudaGetDevice(&dev), "HC split device");
    hipDeviceProp_t prop{}; ckc(hipGetDeviceProperties(&prop, dev), "HC split properties");
    if (std::string(prop.gcnArchName).find("gfx1100") != 0)
        throw std::runtime_error("HC row split requires gfx1100");
    if (!strata_pf_padded_gemm_available() || std::getenv("FNX_PF_HC_CHECK"))
        throw std::runtime_error("HC row split requires native gfx11 GEMM and FNX_PF_HC_CHECK unset");
    if (hcs_states[card].mail.data) throw std::runtime_error("HC split already allocated");
    HcsMail m{};
    m.capacity_words = 2048 * LR / 2;
    m.capacity_blocks = (m.capacity_words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK;
    m.timeout_ticks = ring_wait_ticks(dev);
    m.errors = errors;
    auto fine = [](size_t bytes) -> unsigned* {
        void* data = nullptr;
        ckc(hipExtMallocWithFlags(&data, bytes, hipDeviceMallocFinegrained), "HC split allocation");
        own_dev(data); ckc(cudaMemset(data, 0, bytes), "HC split clear"); return static_cast<unsigned*>(data);
    };
    m.data = fine((size_t)2 * m.capacity_words * sizeof(unsigned));
    m.ready = fine((size_t)2 * m.capacity_blocks * sizeof(unsigned));
    ckc(cudaMalloc(reinterpret_cast<void**>(&m.generation), sizeof(unsigned)), "HC split generation");
    own_dev(m.generation); ckc(cudaMemset(m.generation, 0, sizeof(unsigned)), "HC split generation clear");
    own([card] { hcs_states[card] = {}; });
    hcs_states[card] = {dev, m};
    std::fprintf(stderr, "fnx: HC row split card %d: %zu KiB, main-model chunks 3072..4096\n", card,
        ((size_t)2 * (m.capacity_words + m.capacity_blocks) * sizeof(unsigned) + sizeof(unsigned)) >> 10);
}
namespace {
bool hcs_down(Engine::Pf& p, const Q8S& w, int rows, cudaStream_t stream) {
    if (!hcs_enabled() || rows < 3072 || rows > 4096 || !p.native_gemm || p.f16acc || p.hcs) return false;
    int card = -1;
    for (int c = 0; c < 2; ++c) if (hcs_states[c].dev == p.dev) card = c;
    if (card < 0 || !hcs_states[card].mail.data || !hcs_states[1-card].mail.data)
        throw std::runtime_error("HC split mailboxes not initialized");
    const auto own = hcs_states[card].mail, peer = hcs_states[1-card].mail;
    const int boundary = ((rows + 127) / 128) * 64;
    const int first = card ? boundary : 0, count = card ? rows - boundary : boundary;
    auto* input = reinterpret_cast<const uint16_t*>(p.xh + (size_t)first * D);
    auto* down = p.lo32 + (size_t)first * LR;
    auto* lo = p.lo16 + (size_t)first * LR;
    k_q8s_f16<<<grid_for((size_t)LR * D), 256, 0, stream>>>(w.q, w.d, (size_t)LR * D, p.wh);
    hcs_tick<<<1, 1, 0, stream>>>(own.generation);
    if (!strata_pf_gemm_narrow_f16_ld(input, D, reinterpret_cast<const uint16_t*>(p.wh), D,
                                    down, count, LR, D, LR, stream))
        throw std::runtime_error("HC split GEMM rejected");
    const int words = count * LR / 2, blocks = (words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK;
    hcs_push<true><<<blocks, 256, 0, stream>>>(down, lo, words, own, peer);
    const int other_first = card ? 0 : boundary, other_words = (rows - count) * LR / 2;
    hcs_pull<<<(other_words + HCS_WORDS_PER_BLOCK - 1) / HCS_WORDS_PER_BLOCK, 256, 0, stream>>>(
        p.lo16 + (size_t)other_first * LR, other_words, own);
    ckc(cudaGetLastError(), "HC split launch");
    return true;
}
}
#else
void hc_row_split_prepare(int, bool, int, unsigned*) {
    if (env_int("FNX_PF_HC_ROW_SPLIT", 0)) throw std::runtime_error("HC row split requires two-card HIP Flash-Next");
}
namespace { bool hcs_down(Engine::Pf&, const Q8S&, int, cudaStream_t) { return false; } }
#endif

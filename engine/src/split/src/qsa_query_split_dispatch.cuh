#if FNX_HIP && FNX_NCARD == 2 && !FNX_27B
// Included in the kernels_p5 launcher section, inside namespace fnx.
namespace {
QsxMail qsx_mail[2]{};
bool qsx_enabled() {
    static const bool enabled = [] {
        const auto value = env_int("FNX_PF_QSA_QUERY_SPLIT", 0);
        if (value != 0 && value != 1)
            throw std::runtime_error("FNX_PF_QSA_QUERY_SPLIT: expected 0 or 1");
        return value != 0;
    }();
    return enabled;
}
void qsx_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}
}

void qsa_query_split_prepare(int card, bool peer_on, int lanes, unsigned* errors) {
    if (!qsx_enabled()) return;
    if (!peer_on || lanes != 1 || !attn_head_split() || card < 0 || card > 1)
        throw std::runtime_error("QSA query split needs the probed peer link, head split and one lane");
    for (const char* key : {"FNX_PF_QSA_WIDE", "FNX_QSA_PAIRED", "FNX_QSA_SAMPLE_PIVOT",
                            "FNX_PF_QSA_SAMPLE", "FNX_PF_QSA_SAMPLE_LONG", "FNX_QSA_COALESCED"})
        if (!env_flag(key)) throw std::runtime_error(std::string("QSA query split requires ") + key);
    int dev = -1;
    qsx_check(cudaGetDevice(&dev), "QSA split device");
    hipDeviceProp_t prop{};
    qsx_check(hipGetDeviceProperties(&prop, dev), "QSA split properties");
    if (std::string(prop.gcnArchName).find("gfx1100") != 0)
        throw std::runtime_error("QSA query split requires gfx1100");
    if (qsx_mail[card].data) throw std::runtime_error("QSA split mailbox already active");
    QsxMail mail{};
    mail.capacity = 512;
    mail.timeout_ticks = ring_wait_ticks(dev);
    mail.errors = errors;
    auto fine = [](size_t bytes) -> void* {
        void* data = nullptr;
        qsx_check(hipExtMallocWithFlags(&data, bytes, hipDeviceMallocFinegrained), "QSA split allocation");
        own_dev(data);
        qsx_check(cudaMemset(data, 0, bytes), "QSA split clear");
        return data;
    };
    mail.data = static_cast<int*>(fine((size_t)2 * mail.capacity * QSX_WORDS * sizeof(int)));
    mail.ready = static_cast<unsigned*>(fine(2 * mail.capacity * sizeof(unsigned)));
    qsx_check(cudaMalloc(reinterpret_cast<void**>(&mail.generation), sizeof(unsigned)), "QSA split generation");
    own_dev(mail.generation);
    qsx_check(cudaMemset(mail.generation, 0, sizeof(unsigned)), "QSA split generation clear");
    own([card] { qsx_mail[card] = {}; });
    qsx_mail[card] = mail;
    std::fprintf(stderr, "fnx: QSA query split card %d: %zu KiB, rows 256..512, prefix >=24576, chunk >512\n",
                 card, ((size_t)2 * mail.capacity * (QSX_WORDS * sizeof(int) + sizeof(unsigned)) + sizeof(unsigned)) >> 10);
}

bool qsa_query_split(const float* q, const uint16_t* pooled, const KvPages& pages,
                     float* scores, int* cells, int* counts, int max_pools,
                     const int* pos, int rows, int card, void* stream,
                     int stride, int host_position) {
    if (!qsx_enabled() || stride <= 512 || rows < 256 || rows > 512 ||
        host_position < 24576 || max_pools > QSA_PAIR_MAX_POOLS) return false;
    if (card < 0 || card > 1 || !qsx_mail[card].data || !qsx_mail[1-card].data)
        throw std::runtime_error("QSA query split mailbox not initialized");
    const auto s = (cudaStream_t)stream;
    const auto own = qsx_mail[card], peer = qsx_mail[1-card];
    k_idx_score_tc_wide<true><<<dim3((max_pools + 255) / 256, (rows + 3) / 4), 256, 0, s>>>(
        q, reinterpret_cast<const __half*>(pooled), pages, scores, max_pools, pos, rows, card, own.generation);
    k_idx_select_pair<true, true, true><<<rows, QSA_SEL_THREADS, 0, s>>>(
        scores, cells, counts, max_pools, pos, stride, LaneRows{}, card, 4);
    k_qsx_push<<<rows, 256, 0, s>>>(cells, counts, rows, stride, card, 4, own, peer);
    k_qsx_pull<<<rows, 256, 0, s>>>(cells, counts, rows, stride, card, 4, own);
    ck_launch("QSA query split");
    return true;
}

#else
void qsa_query_split_prepare(int, bool, int, unsigned*) {
    if (env_int("FNX_PF_QSA_QUERY_SPLIT", 0) != 0)
        throw std::runtime_error("FNX_PF_QSA_QUERY_SPLIT requires two-card HIP Flash-Next");
}
bool qsa_query_split(const float*, const uint16_t*, const KvPages&, float*, int*, int*, int,
                     const int*, int, int, void*, int, int) { return false; }
#endif

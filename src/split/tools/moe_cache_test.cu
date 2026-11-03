// Exercise the production CLOCK planner and copies across layers and graph replays.
#define FNX_KPART 0
#include "../src/kernels.cu"
#include <iostream>

namespace fnx {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
__global__ void read_cached(ExpertCache ec, int layer, const Job* jobs, const int* count, unsigned char* out,
                            int gu_bytes, int dn_bytes) {
    const int j = blockIdx.x;
    if (j >= *count) return;
    const int g = layer * NEXP + jobs[j].w, stride = gu_bytes + dn_bytes;
    for (int i = threadIdx.x; i < stride; i += blockDim.x)
        out[j * stride + i] = i < gu_bytes ? ec.ep_gu[g][i] : ec.ep_dn[g][i - gu_bytes];
}

void run_cache(int dev, int slots, int pattern, bool coherent) {
    constexpr int layers = 4, experts = layers * NEXP, capacity = MAXT * TOPK;
    constexpr size_t gu_max = 448, dn_max = 224, stride = gu_max + dn_max, guard = 256;
    ExpertCache ec;
    ec.n_slots = ec.n[0] = slots;
    ec.n_pools = 1;
    ec.dn_off[0] = gu_max;
    ec.slot_bytes[0] = stride;
    std::vector<void*> device;
    auto alloc = [&](size_t n) {
        void* p = nullptr; ck(cudaMalloc(&p, n), "allocate"); device.push_back(p); return p;
    };
    unsigned char* host = nullptr;
#if FNX_HIP
    ck(hipHostMalloc((void**) &host, experts * stride, coherent ? hipHostMallocCoherent : hipHostMallocNonCoherent), "host weights");
#else
    (void) coherent;
    ck(cudaMallocHost((void**) &host, experts * stride), "host weights");
#endif
    for (int g = 0; g < experts; ++g)
        for (size_t i = 0; i < stride; ++i) host[g * stride + i] = (g * 17 + i * 31 + g / 7) & 255;
    char* hot = (char*) alloc(experts * stride);
    ck(cudaMemcpy(hot, host, experts * stride, cudaMemcpyHostToDevice), "hot weights");
    std::vector<const char*> gu(experts), dn(experts);
    std::vector<int> places(experts, -1), owners(slots, -1);
    for (int g = 0; g < experts; ++g) {
        const bool fixed = pattern == 2 || (pattern == 1 && g % 7 == 0);
        const char* src = fixed ? hot : (const char*) host;
        gu[g] = src + g * stride;
        dn[g] = src + g * stride + gu_max;
        if (fixed) places[g] = -2;
    }
    ec.ep_gu = (const char**) alloc(experts * sizeof(char*));
    ec.ep_dn = (const char**) alloc(experts * sizeof(char*));
    ec.host_gu = (const char**) alloc(experts * sizeof(char*));
    ec.host_dn = (const char**) alloc(experts * sizeof(char*));
    for (const auto& pair : {std::make_pair((void*) ec.ep_gu, gu.data()), std::make_pair((void*) ec.host_gu, gu.data()),
                             std::make_pair((void*) ec.ep_dn, dn.data()), std::make_pair((void*) ec.host_dn, dn.data())})
        ck(cudaMemcpy(pair.first, pair.second, experts * sizeof(char*), cudaMemcpyHostToDevice), "tables");
    ec.slot_of = (int*) alloc(experts * sizeof(int));
    ec.owner = (int*) alloc(slots * sizeof(int));
    ck(cudaMemcpy(ec.slot_of, places.data(), experts * sizeof(int), cudaMemcpyHostToDevice), "places");
    ck(cudaMemcpy(ec.owner, owners.data(), slots * sizeof(int), cudaMemcpyHostToDevice), "owners");
    ec.ref = (unsigned char*) alloc(slots);
    ec.hand = (int*) alloc(EC_POOLS * sizeof(int));
    ec.lent = (int*) alloc(EC_POOLS * sizeof(int));
    char* pool = (char*) alloc(slots * stride + 2 * guard);
    ec.pool[0] = pool + guard;
    int* list = (int*) alloc((capacity + 1) * sizeof(int));
    Job* jobs = (Job*) alloc(capacity * sizeof(Job));
    int* count = (int*) alloc(sizeof(int));
    unsigned char* out = (unsigned char*) alloc(capacity * stride + guard);
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ck(cudaDeviceSynchronize(), "initial uploads");
    for (const auto& p : {std::make_pair((void*) ec.ref, (size_t) slots), std::make_pair((void*) ec.hand, EC_POOLS * sizeof(int)),
                          std::make_pair((void*) ec.lent, EC_POOLS * sizeof(int))})
        ck(cudaMemsetAsync(p.first, 0, p.second, stream), "cache reset");
    ck(cudaMemsetAsync(pool, 0xA5, slots * stride + 2 * guard, stream), "pool guards");
    ck(cudaStreamSynchronize(stream), "reset wait");
    auto launch = [&](int layer) {
        const int gb = 256 + layer * 64, db = 128 + layer * 32;
        moe_cache(layer, gb, db, ec, list, stream, jobs, count, true);
        read_cached<<<capacity, 256, 0, stream>>>(ec, layer, jobs, count, out, gb, db);
    };
    cudaGraph_t graphs[layers]; cudaGraphExec_t execs[layers];
    for (int l = 0; l < layers; ++l) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture"); launch(l);
        ck(cudaStreamEndCapture(stream, &graphs[l]), "capture end");
        ck(cudaGraphInstantiate(&execs[l], graphs[l], nullptr, nullptr, 0), "instantiate");
    }
    std::vector<unsigned char> got(capacity * stride + guard), expected(got.size()), guards(guard * 2);
    std::vector<int> now(experts), owner(slots);
    std::vector<const char*> ep_gu(experts), ep_dn(experts), backing(experts);
    for (int rep = 0; rep < 64; ++rep) {
        const int layer = rep % layers, counts[] = {0, 1, 7, capacity};
        const int n = counts[(rep / layers) % 4], gb = 256 + layer * 64, db = 128 + layer * 32;
        std::vector<Job> js(capacity);
        std::fill(expected.begin(), expected.end(), 0xA5);
        for (int j = 0; j < n; ++j) {
            js[j].w = (j * 3 + rep * 31) % NEXP;
            const int g = layer * NEXP + js[j].w;
            std::memcpy(expected.data() + j * (gb + db), host + g * stride, gb);
            std::memcpy(expected.data() + j * (gb + db) + gb, host + g * stride + gu_max, db);
        }
        ck(cudaMemcpyAsync(jobs, js.data(), js.size() * sizeof(Job), cudaMemcpyHostToDevice, stream), "jobs");
        ck(cudaMemcpyAsync(count, &n, sizeof n, cudaMemcpyHostToDevice, stream), "count");
        ck(cudaMemsetAsync(out, 0xA5, got.size(), stream), "output guard");
        if (rep < 32) launch(layer); else ck(cudaGraphLaunch(execs[layer], stream), "replay");
        ck(cudaStreamSynchronize(stream), "wait");
        ck(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost), "read output");
        if (got != expected) throw std::runtime_error("cache output or output guard mismatch");
        ck(cudaMemcpy(guards.data(), pool, guard, cudaMemcpyDeviceToHost), "first pool guard");
        ck(cudaMemcpy(guards.data() + guard, ec.pool[0] + slots * stride, guard, cudaMemcpyDeviceToHost), "last pool guard");
        if (!std::all_of(guards.begin(), guards.end(), [](unsigned char b) { return b == 0xA5; }))
            throw std::runtime_error("cache pool guard overwritten");
        ck(cudaMemcpy(now.data(), ec.slot_of, experts * sizeof(int), cudaMemcpyDeviceToHost), "read places");
        ck(cudaMemcpy(owner.data(), ec.owner, slots * sizeof(int), cudaMemcpyDeviceToHost), "read owners");
        ck(cudaMemcpy(ep_gu.data(), ec.ep_gu, experts * sizeof(char*), cudaMemcpyDeviceToHost), "read gate/up table");
        ck(cudaMemcpy(ep_dn.data(), ec.ep_dn, experts * sizeof(char*), cudaMemcpyDeviceToHost), "read down table");
        for (int g = 0; g < experts; ++g) {
            const int s = now[g];
            if ((places[g] == -2 && s != -2) || s >= slots || s < -2) throw std::runtime_error("invalid cache place");
            const char* eg = s < 0 ? gu[g] : ec.pool[0] + s * stride;
            const char* ed = s < 0 ? dn[g] : eg + gu_max;
            if (ep_gu[g] != eg || ep_dn[g] != ed || (s >= 0 && owner[s] != g))
                throw std::runtime_error("cache pointer or owner mismatch");
        }
    }
    ck(cudaMemcpy(backing.data(), ec.host_gu, experts * sizeof(char*), cudaMemcpyDeviceToHost), "backing gate/up");
    if (backing != gu) throw std::runtime_error("immutable backing gate/up table changed");
    ck(cudaMemcpy(backing.data(), ec.host_dn, experts * sizeof(char*), cudaMemcpyDeviceToHost), "backing down");
    if (backing != dn) throw std::runtime_error("immutable backing down table changed");
    for (int l = 0; l < layers; ++l) { cudaGraphExecDestroy(execs[l]); cudaGraphDestroy(graphs[l]); }
    cudaStreamDestroy(stream);
    for (void* p : device) cudaFree(p);
    cudaFreeHost(host);
    std::cout << "card " << dev << " slots " << slots << " pattern " << pattern << " coherent " << coherent << ": 64 checks passed\n";
}
} // namespace fnx

int main() {
    int cards = 0; fnx::ck(cudaGetDeviceCount(&cards), "devices");
    if (cards < 2) throw std::runtime_error("two physical devices required");
    for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        for (int slots : {160, 256, 1024}) for (int pattern : {0, 1, 2}) for (bool coherent : {false, true})
            fnx::run_cache(dev, slots, pattern, coherent);
    }
    std::cout << "Static decode cache: 2304 exact direct/graph checks passed\n";
}

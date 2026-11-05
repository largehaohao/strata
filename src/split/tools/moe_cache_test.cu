// Exercise the production CLOCK planner and copies across layers and graph replays.
#define FNX_KPART 0
#include "../src/kernels.cu"
#include <iostream>
#include "../src/static_kv_cache.hpp"

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

void run_cache(int dev, int slots, int pattern, bool coherent, int groups = 1, bool shared_kv = false, int pool_hint = -1, int max_age = 1) {
    constexpr int layers = 4, experts = layers * NEXP, capacity = MAXT * TOPK;
    constexpr size_t gu_max = 448, dn_max = 224, stride = gu_max + dn_max, guard = 256;
    std::vector<StaticCacheShape> shapes;
    for (int l = 0; l < layers; ++l) shapes.push_back({(size_t) 256 + l * 64, (size_t) 128 + l * 32});
    const auto plan = static_kv_cache_plan(static_cache_layout(shapes, slots, groups, capacity),
                                            shared_kv ? 65536 : 0, shared_kv ? 4096 : 0, pool_hint);
    const auto& layout = plan.cache;
    slots = layout.slots;
    ExpertCache ec;
    ec.n_slots = slots; ec.n_pools = (int) layout.pools.size(); ec.arena_pool = plan.pool;
    for (int p = 0; p < ec.n_pools; ++p) {
        ec.s0[p] = layout.pools[p].start; ec.n[p] = layout.pools[p].slots;
        ec.dn_off[p] = layout.pools[p].gu; ec.slot_bytes[p] = layout.pools[p].stride();
    }
    for (int l = 0; l < layers; ++l) ec.pool_of[l] = layout.pool_of[l];
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
    ec.max_age = max_age;
    if (max_age > 1) { ec.age = (unsigned char*) alloc(slots); ck(cudaMemset(ec.age, 0, slots), "age reset"); }
    ec.hand = (int*) alloc(EC_POOLS * sizeof(int));
    ec.lent = (int*) alloc(EC_POOLS * sizeof(int));
    std::vector<char*> pools;
    for (int p = 0; p < ec.n_pools; ++p) {
        const size_t prefix = p == plan.pool ? plan.prefix_bytes : 0;
        char* pool = (char*) alloc(prefix + ec.n[p] * ec.slot_bytes[p] + 2 * guard);
        pools.push_back(pool); ec.pool[p] = pool + guard + prefix;
    }
    int* list = (int*) alloc((capacity + 1) * sizeof(int));
    Job* jobs = (Job*) alloc(capacity * sizeof(Job));
    int* count = (int*) alloc(sizeof(int));
    unsigned char* out = (unsigned char*) alloc(capacity * stride + guard);
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ck(cudaDeviceSynchronize(), "initial uploads");
    for (const auto& p : {std::make_pair((void*) ec.ref, (size_t) slots), std::make_pair((void*) ec.hand, EC_POOLS * sizeof(int)),
                          std::make_pair((void*) ec.lent, EC_POOLS * sizeof(int))})
        ck(cudaMemsetAsync(p.first, 0, p.second, stream), "cache reset");
    ck(cudaMemcpyAsync(ec.hand, ec.s0, EC_POOLS * sizeof(int), cudaMemcpyHostToDevice, stream), "initial hands");
    for (int p = 0; p < ec.n_pools; ++p) {
        const size_t prefix = p == plan.pool ? plan.prefix_bytes : 0;
        ck(cudaMemsetAsync(pools[p], 0xA5, prefix + ec.n[p] * ec.slot_bytes[p] + 2 * guard, stream), "pool guards");
    }
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
    std::vector<int> oracle_slot = places, oracle_owner = owners;
    std::vector<unsigned char> oracle_ref(slots, 0), oracle_age(slots, 0), got_ref(slots), got_age(slots);
    std::vector<int> oracle_hand(ec.s0, ec.s0 + EC_POOLS), got_hand(EC_POOLS);
    int lent = 0;
    for (int rep = 0; rep < 64; ++rep) {
        if (shared_kv && ec.n_pools == 2 && env_flag("FNX_CACHE_R_TEST")) {
            ExpertCache residual = ec;
            const int pool = 1 - ec.arena_pool;
            residual.arena_pool = pool;
            const int take = 1 + (rep * 37) % (ec.n[pool] - 1), first = ec.s0[pool];
            for (int i=0; i<take; ++i) {
                const int slot=first+i, old=oracle_owner[slot];
                if (old >= 0) oracle_slot[old]=-1;
                oracle_owner[slot]=-1;
                oracle_ref[slot]=0;
                oracle_age[slot]=0;
            }
            oracle_hand[pool]=std::max(oracle_hand[pool], first+take);
            arena_slots(residual,0,take,stream);
            std::vector<int> during(EC_POOLS);
            ck(cudaMemcpyAsync(during.data(),ec.lent,EC_POOLS*sizeof(int),cudaMemcpyDeviceToHost,stream),"residual ownership");
            ck(cudaStreamSynchronize(stream),"residual borrow");
            if(during[ec.arena_pool]!=lent || during[pool]!=take) throw std::runtime_error("residual loan changed KV ownership");
            ck(cudaMemsetAsync(ec.pool[pool],0x6b,(size_t)take*ec.slot_bytes[pool],stream),"overwrite residual scratch");
            // Return also models cancellation after a partly completed read.
            arena_slots(residual,take,0,stream);
            ck(cudaMemcpyAsync(during.data(),ec.lent,EC_POOLS*sizeof(int),cudaMemcpyDeviceToHost,stream),"residual return ownership");
            ck(cudaStreamSynchronize(stream),"residual return");
            if(during[ec.arena_pool]!=lent || during[pool]!=0) throw std::runtime_error("residual return changed KV ownership");
        }
        if (shared_kv) {
            const int sizes[] = {0, plan.extra_slots, plan.extra_slots / 2, 1, plan.extra_slots, 0};
            const int next = sizes[rep % 6];
            const int p = ec.arena_pool, s0 = ec.s0[p];
            for (int i = std::min(lent, next); i < std::max(lent, next); ++i) {
                const int slot = s0 + i, old = oracle_owner[slot];
                if (old >= 0) oracle_slot[old] = -1;
                oracle_owner[slot] = -1; oracle_ref[slot] = next > lent ? 3 : 0; oracle_age[slot] = 0;
            }
            if (oracle_hand[p] < s0 + next) oracle_hand[p] = s0 + next;
            arena_slots(ec, lent, next, stream);
            lent = next;
            // K/V may overwrite every byte of a lent slot. Neither direct launches
            // nor captured graphs may read/write these slots until they are returned.
            ck(cudaMemsetAsync(ec.pool[ec.arena_pool], 0xC7, (size_t) lent * ec.slot_bytes[ec.arena_pool], stream), "K/V writes");
        }
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
        // Sequential CLOCK oracle: same hit pinning and job-order insertion as the GPU warp.
        const int cp = ec.pool_of[layer], start = ec.s0[cp] + (cp == ec.arena_pool ? lent : 0);
        const int size = ec.n[cp] - (cp == ec.arena_pool ? lent : 0);
        int hand = oracle_hand[cp] - start;
        if (hand < 0 || hand >= size) hand = 0;
        for (int j = 0; j < n; ++j) {
            const int slot = oracle_slot[layer * NEXP + js[j].w];
            if (slot >= 0) { oracle_ref[slot] = 2; oracle_age[slot] = std::min(max_age, (int) oracle_age[slot] + 1); }
        }
        for (int j = 0; j < n; ++j) {
            const int expert = layer * NEXP + js[j].w;
            if (oracle_slot[expert] != -1) continue;
            int slot;
            for (;;) {
                slot = start + hand; hand = (hand + 1) % size;
                if (oracle_ref[slot] == 0) break;
                if (oracle_ref[slot] == 1) {
                    if (oracle_age[slot] > 1) --oracle_age[slot];
                    else { oracle_ref[slot] = 0; oracle_age[slot] = 0; }
                }
            }
            if (oracle_owner[slot] >= 0) oracle_slot[oracle_owner[slot]] = -1;
            oracle_owner[slot] = expert; oracle_slot[expert] = slot; oracle_ref[slot] = 2; oracle_age[slot] = 1;
        }
        for (int j = 0; j < n; ++j) {
            const int slot = oracle_slot[layer * NEXP + js[j].w];
            if (slot >= 0) oracle_ref[slot] = 1;
        }
        oracle_hand[cp] = start + hand;
        ck(cudaMemcpyAsync(jobs, js.data(), js.size() * sizeof(Job), cudaMemcpyHostToDevice, stream), "jobs");
        ck(cudaMemcpyAsync(count, &n, sizeof n, cudaMemcpyHostToDevice, stream), "count");
        ck(cudaMemsetAsync(out, 0xA5, got.size(), stream), "output guard");
        if (rep < 32) launch(layer); else ck(cudaGraphLaunch(execs[layer], stream), "replay");
        ck(cudaStreamSynchronize(stream), "wait");
        ck(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost), "read output");
        if (got != expected) throw std::runtime_error("cache output or output guard mismatch");
        for (int p = 0; p < ec.n_pools; ++p) {
            ck(cudaMemcpy(guards.data(), pools[p], guard, cudaMemcpyDeviceToHost), "first pool guard");
            ck(cudaMemcpy(guards.data() + guard, ec.pool[p] + ec.n[p] * ec.slot_bytes[p], guard, cudaMemcpyDeviceToHost), "last pool guard");
            if (!std::all_of(guards.begin(), guards.end(), [](unsigned char b) { return b == 0xA5; }))
                throw std::runtime_error("cache pool guard overwritten");
        }
        if (shared_kv) {
            std::vector<unsigned char> kv(plan.prefix_bytes + guard), borrowed((size_t) lent * ec.slot_bytes[ec.arena_pool]);
            ck(cudaMemcpy(kv.data(), pools[ec.arena_pool], kv.size(), cudaMemcpyDeviceToHost), "K/V prefix");
            if (!std::all_of(kv.begin(), kv.end(), [](unsigned char b) { return b == 0xA5; }))
                throw std::runtime_error("K/V prefix overwritten");
            if (!borrowed.empty()) ck(cudaMemcpy(borrowed.data(), ec.pool[ec.arena_pool], borrowed.size(), cudaMemcpyDeviceToHost), "lent K/V slots");
            if (!std::all_of(borrowed.begin(), borrowed.end(), [](unsigned char b) { return b == 0xC7; }))
                throw std::runtime_error("cache wrote a lent K/V slot");
        }
        ck(cudaMemcpy(now.data(), ec.slot_of, experts * sizeof(int), cudaMemcpyDeviceToHost), "read places");
        ck(cudaMemcpy(owner.data(), ec.owner, slots * sizeof(int), cudaMemcpyDeviceToHost), "read owners");
        ck(cudaMemcpy(got_ref.data(), ec.ref, slots, cudaMemcpyDeviceToHost), "read refs");
        ck(cudaMemcpy(got_hand.data(), ec.hand, EC_POOLS * sizeof(int), cudaMemcpyDeviceToHost), "read hands");
        if (now != oracle_slot || owner != oracle_owner || got_ref != oracle_ref || got_hand != oracle_hand)
            throw std::runtime_error("CLOCK policy differs from sequential oracle");
        if (ec.age) {
            ck(cudaMemcpy(got_age.data(), ec.age, slots, cudaMemcpyDeviceToHost), "read ages");
            if (got_age != oracle_age) throw std::runtime_error("CLOCK age differs from sequential oracle");
        }
        ck(cudaMemcpy(ep_gu.data(), ec.ep_gu, experts * sizeof(char*), cudaMemcpyDeviceToHost), "read gate/up table");
        ck(cudaMemcpy(ep_dn.data(), ec.ep_dn, experts * sizeof(char*), cudaMemcpyDeviceToHost), "read down table");
        for (int g = 0; g < experts; ++g) {
            const int s = now[g];
            if ((places[g] == -2 && s != -2) || s >= slots || s < -2) throw std::runtime_error("invalid cache place");
            const int p = ec.pool_of[g / NEXP];
            if (s >= 0 && (s < ec.s0[p] + (p == ec.arena_pool ? lent : 0) || s >= ec.s0[p] + ec.n[p])) throw std::runtime_error("expert placed in wrong pool");
            const char* eg = s < 0 ? gu[g] : ec.pool[p] + (s - ec.s0[p]) * ec.slot_bytes[p];
            const char* ed = s < 0 ? dn[g] : eg + ec.dn_off[p];
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
    std::cout << "card " << dev << " slots " << slots << " pools " << ec.n_pools << " pattern " << pattern << " coherent " << coherent << " max age " << max_age << " shared K/V " << shared_kv << ": 64 checks passed\n";
}
} // namespace fnx

int main() {
    int cards = 0; fnx::ck(cudaGetDeviceCount(&cards), "devices");
    if (cards < 2) throw std::runtime_error("two physical devices required");
    for (int max_age : {1, 2, 4}) for (int dev = 0; dev < 2; ++dev) {
        fnx::ck(cudaSetDevice(dev), "device");
        for (int slots : {160, 256, 1024}) for (int pattern : {0, 1, 2}) for (bool coherent : {false, true})
            fnx::run_cache(dev, slots, pattern, coherent, 1, false, -1, max_age);
        for (int slots : {352, 1024}) for (int pattern : {0, 1, 2}) for (bool coherent : {false, true})
            fnx::run_cache(dev, slots, pattern, coherent, 2, false, -1, max_age);
        for (int groups : {1, 2}) for (int pattern : {0, 1, 2}) for (bool coherent : {false, true})
            fnx::run_cache(dev, 352, pattern, coherent, groups, true, -1, max_age);
        for (int pattern : {0, 1, 2}) for (bool coherent : {false, true})
            fnx::run_cache(dev, 352, pattern, coherent, 2, true, 1, max_age);
    }
    std::cout << "Static decode cache: 18432 exact direct/graph/policy checks passed (6912 with K/V lending)\n";
}

// CPU-only ownership/lifetime checks using a tiny mixed-quantization GGUF.
#include "../src/gguf.cpp"
#include <fstream>
#include <iostream>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <class T> void append(std::vector<uint8_t>& out, T value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof value);
}
void str(std::vector<uint8_t>& out, const std::string& value) {
    append<uint64_t>(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}
void align(std::vector<uint8_t>& out) { out.resize((out.size() + 31) / 32 * 32); }
struct Fixture {
    std::string path;
    explicit Fixture(bool mixed) {
        struct Item { std::string name; uint32_t type; std::vector<uint64_t> shape; std::vector<uint8_t> bytes; };
        std::vector<Item> items;
        auto bf16 = [&](const char* name) {
            std::vector<uint8_t> v;
            for (int i = 0; i < 64; ++i) append<uint16_t>(v, uint16_t(0x3f00 + i));
            items.push_back({name, fnx::T_BF16, {32, 2}, v});
        };
        if (mixed) {
            std::vector<uint8_t> q2;
            for (int b = 0; b < 4; ++b) {
                append<uint16_t>(q2, 0x3c00);
                q2.insert(q2.end(), 16, 0xe4);
            }
            items.push_back({"blk.0.ffn_down_exps.weight", fnx::T_Q2_0, {64, 2, 2}, q2});
            bf16("blk.0.attn_q.weight");
            bf16("blk.0.ffn_gate_inp.weight");
            bf16("token_embd.weight");
            bf16("output.weight");
            items.push_back({"per_layer_token_embd.weight", fnx::T_IQ4_NL, {32, 2}, std::vector<uint8_t>(36, 0)});
        }
        std::vector<uint8_t> plain;
        for (int i = 0; i < 64; ++i) append<float>(plain, float(i) / 8);
        items.push_back({"plain.weight", fnx::T_F32, {32, 2}, plain});
        std::vector<uint8_t> file{'G', 'G', 'U', 'F'}, data;
        append<uint32_t>(file, 3);
        append<uint64_t>(file, items.size());
        append<uint64_t>(file, 0);
        for (const auto& item : items) {
            str(file, item.name);
            append<uint32_t>(file, item.shape.size());
            for (auto dim : item.shape) append<uint64_t>(file, dim);
            append<uint32_t>(file, item.type);
            append<uint64_t>(file, data.size());
            data.insert(data.end(), item.bytes.begin(), item.bytes.end());
            align(data);
        }
        align(file);
        file.insert(file.end(), data.begin(), data.end());
        char tmp[] = "/tmp/strata-gguf-release-XXXXXX";
        const int fd = mkstemp(tmp);
        require(fd >= 0, "temporary fixture");
        close(fd);
        path = tmp;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(file.data()), file.size());
        require(bool(out), "write fixture");
    }
    ~Fixture() { if (!path.empty()) std::remove(path.c_str()); }
};
struct Snapshot { fnx::Tensor tensor; std::vector<uint8_t> bytes; };
void check_packed_down() {
    Fixture f(true);
    fnx::Gguf original(f.path), packed(f.path, true);
    const auto& raw = packed.t("blk.0.ffn_down_exps.weight");
    const auto& q4 = original.t("blk.0.ffn_down_exps.weight");
    require(packed.converted() == 4 && original.converted() == 5, "only down repacking is skipped");
    require(raw.type == fnx::T_Q2_0 && raw.nbytes == 72 && raw.row_bytes() == 18 && raw.shard >= 0 &&
            q4.type == fnx::T_Q4_0 && q4.nbytes == 144 && q4.row_bytes() == 36 && q4.shard == -1, "packed metadata");
    for (int b = 0; b < 4; ++b) {
        const auto* source = raw.data + b * 18;
        for (int h = 0; h < 2; ++h) {
            const auto* expanded = q4.data + (b * 2 + h) * 18;
            require(std::memcmp(source, expanded, 2) == 0, "packed scale");
            for (int j = 0; j < 32; ++j) {
                const int i = 32 * h + j;
                const int code = (source[2 + i / 4] >> (2 * (i % 4))) & 3;
                const int nibble = (expanded[2 + j % 16] >> (4 * (j / 16))) & 15;
                require(code + 7 == nibble, "packed value");
            }
        }
    }
    for (const auto& kv : original.tensors()) {
        if (kv.first == raw.name) continue;
        const auto& a = kv.second; const auto& b = packed.t(kv.first);
        require(a.type == b.type && a.nbytes == b.nbytes && std::memcmp(a.data, b.data, a.nbytes) == 0, "unrelated conversion changed");
    }
    const auto* before = raw.data;
    const std::vector<uint8_t> bytes(raw.data, raw.data + raw.nbytes);
    packed.release_owned_except({});
    require(raw.data == before && std::memcmp(raw.data, bytes.data(), bytes.size()) == 0, "packed mapping released");
}
void check_mixed() {
    Fixture f(true);
    fnx::Gguf g(f.path);
    require(g.converted() == 5, "fixture must exercise five conversions");
    std::map<std::string, Snapshot> before;
    for (const auto& kv : g.tensors())
        before.emplace(kv.first, Snapshot{kv.second, {kv.second.data, kv.second.data + kv.second.nbytes}});
    bool refused = false;
    try { g.release_owned_except({"missing.weight"}); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "missing keep name must fail before release");
    for (const auto& kv : before) {
        const auto& t = g.t(kv.first);
        require(t.data == kv.second.tensor.data, "failed request changed pointer");
        require(std::memcmp(t.data, kv.second.bytes.data(), t.nbytes) == 0, "failed request changed bytes");
    }
    const std::set<std::string> keep{"token_embd.weight", "per_layer_token_embd.weight", "output.weight"};
    require(g.release_owned_except(keep) == 468, "release byte count");
    for (const auto& kv : before) {
        const auto& t = g.t(kv.first);
        const auto& old = kv.second.tensor;
        require(t.name == old.name && t.type == old.type && t.n_dims == old.n_dims && t.nbytes == old.nbytes &&
                t.shard == old.shard && t.offset == old.offset && std::memcmp(t.ne, old.ne, sizeof t.ne) == 0, "metadata changed");
        const bool retained = keep.count(kv.first) || old.shard >= 0;
        if (retained) {
            require(t.data == old.data, "retained pointer changed");
            require(std::memcmp(t.data, kv.second.bytes.data(), t.nbytes) == 0, "retained bytes changed");
        } else require(t.data == nullptr, "released tensor left a dangling pointer");
    }
    require(g.release_owned_except(keep) == 0, "repeat release must be idempotent");
    refused = false;
    try { g.release_owned_except({"blk.0.attn_q.weight"}); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "cannot retain an already released tensor");
    require(g.release_owned_except({}) == 136, "release remaining owned embeddings/head");
    require(g.t("plain.weight").data && g.t("per_layer_token_embd.weight").data, "mapped tensors must remain available");
    require(g.release_owned_except({}) == 0, "fully released state must be idempotent");
}
void check_header_bounds() {
    auto expect_throw = [](const std::vector<uint8_t>& bytes, const char* substr, const char* label) {
        char tmp[] = "/tmp/strata-gguf-bounds-XXXXXX";
        const int fd = mkstemp(tmp);
        require(fd >= 0, "temporary bounds fixture");
        close(fd);
        struct Guard {
            std::string p;
            ~Guard() { std::remove(p.c_str()); }
        } guard{tmp};
        {
            std::ofstream out(tmp, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            require(bool(out), "write bounds fixture");
        }
        bool threw = false;
        try {
            fnx::Gguf g(tmp);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find(substr) != std::string::npos;
        }
        require(threw, label);
    };
    {
        std::vector<uint8_t> bad{'G', 'G', 'U', 'F'};
        append<uint32_t>(bad, 3);
        append<uint64_t>(bad, 0);
        append<uint64_t>(bad, 1);
        append<uint64_t>(bad, UINT64_MAX - 8);
        expect_throw(bad, "truncated string", "Cur::str must reject huge length without pointer wrap");
    }
    {
        std::vector<uint8_t> bad{'G', 'G', 'U', 'F'};
        append<uint32_t>(bad, 3);
        append<uint64_t>(bad, 0);
        append<uint64_t>(bad, 1);
        str(bad, "bad.array");
        append<uint32_t>(bad, 9);
        append<uint32_t>(bad, 6);
        append<uint64_t>(bad, UINT64_MAX / 4);
        expect_throw(bad, "truncated array", "load_shard must reject huge scalar array without pointer wrap");
    }
}
}
int main() {
    check_mixed();
    check_packed_down();
    check_header_bounds();
    Fixture f(false);
    fnx::Gguf g(f.path);
    const auto* p = g.t("plain.weight").data;
    require(g.converted() == 0 && g.release_owned_except({}) == 0 && g.t("plain.weight").data == p, "unconverted GGUF changed");
    std::cout << "GGUF ownership: mixed/plain, packed down, metadata, retention, rejection and repeated release passed\n";
}

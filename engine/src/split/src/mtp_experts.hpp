// Draft-layer (MTP) routed expert resolver and loader for strata-split:
// supports legacy S2 Q2_0 (experts.bin unpacked to Q4_0), planar block files
// (experts-q4_0.bin / experts-iq4_nl.bin), and native GGUF expert-major blobs
// described by experts.txt (format native, including IQ4_NL / Q4_0 / Q8_0).
#pragma once

#include "env.hpp"
#include "gguf.hpp"

#ifdef _WIN32
#include <sys/stat.h>
#else
#include <sys/stat.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnx {

// Inline block metadata for standalone CPU tests that do not link gguf.cpp.
inline bool mtp_expert_type_info(uint32_t t, TypeInfo& out) {
    switch (t) {
        case T_Q4_0: out = {32, 18}; return true;
        case T_Q5_0: out = {32, 22}; return true;
        case T_Q5_1: out = {32, 24}; return true;
        case T_Q8_0: out = {32, 34}; return true;
        case T_Q4_K: out = {256, 144}; return true;
        case T_Q5_K: out = {256, 176}; return true;
        case T_Q6_K: out = {256, 210}; return true;
        case T_IQ2_XXS: out = {256, 66}; return true;
        case T_IQ3_XXS: out = {256, 98}; return true;
        case T_IQ4_NL: out = {32, 18}; return true;
        case T_IQ3_S: out = {256, 110}; return true;
        case T_IQ2_S: out = {256, 82}; return true;
        case T_IQ4_XS: out = {256, 136}; return true;
        case T_Q2_0: out = {64, 18}; return true;
        default: return false;
    }
}

inline const char* mtp_expert_type_name(uint32_t t) {
    switch (t) {
        case T_Q4_0: return "Q4_0";
        case T_Q5_0: return "Q5_0";
        case T_Q5_1: return "Q5_1";
        case T_Q8_0: return "Q8_0";
        case T_Q4_K: return "Q4_K";
        case T_Q5_K: return "Q5_K";
        case T_Q6_K: return "Q6_K";
        case T_IQ2_XXS: return "IQ2_XXS";
        case T_IQ3_XXS: return "IQ3_XXS";
        case T_IQ4_NL: return "IQ4_NL";
        case T_IQ3_S: return "IQ3_S";
        case T_IQ2_S: return "IQ2_S";
        case T_IQ4_XS: return "IQ4_XS";
        case T_Q2_0: return "Q2_0";
        default: return "UNKNOWN";
    }
}

inline bool mtp_expert_gu_supported(uint32_t t) {
    switch (t) {
        case T_Q4_0:
        case T_Q5_0:
        case T_Q5_1:
        case T_Q8_0:
        case T_Q4_K:
        case T_Q5_K:
        case T_IQ2_XXS:
        case T_IQ3_XXS:
        case T_IQ4_NL:
        case T_IQ3_S:
        case T_IQ2_S:
        case T_IQ4_XS: return true;
        default: return false;
    }
}

inline bool mtp_expert_down_supported(uint32_t t) {
    switch (t) {
        case T_Q4_0:
        case T_Q5_0:
        case T_Q5_1:
        case T_Q8_0:
        case T_IQ4_NL: return true;
        default: return false;
    }
}

// Legacy Strata S2 Q2_0 per-expert packed blob size in bytes.
constexpr size_t mtp_s2_blob_bytes(int n_embd, int n_ff) {
    const size_t row_gu = (size_t) n_embd / 4;
    const size_t row_d = (size_t) n_ff / 4;
    const size_t sc_gu = (size_t) n_embd / 64;
    const size_t sc_d = (size_t) n_ff / 64;
    const size_t o_d = 2ull * (size_t) n_ff * row_gu;
    const size_t o_gs = o_d + (size_t) n_embd * row_d;
    const size_t o_ds = o_gs + 2ull * (size_t) n_ff * sc_gu * 2;
    return o_ds + (size_t) n_embd * sc_d * 2;
}

static_assert(mtp_s2_blob_bytes(2560, 640) == 1382400, "Strata Q2_0 expert blob");

struct MtpExpertMeta {
    bool present = false;
    std::string format;
    int64_t gu_type = -1;
    int64_t d_type = -1;
    int64_t n_expert = 0;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int64_t blob_bytes = 0;
};

enum class MtpExpertLayout { S2_Q2_0, NATIVE_EXPERT_MAJOR, PLANAR_BLOCKS };

struct MtpExpertPlan {
    MtpExpertLayout layout = MtpExpertLayout::S2_Q2_0;
    std::string path;
    uint32_t gu_type = T_Q4_0;
    uint32_t d_type = T_Q4_0;
    size_t gu_row = 0;
    size_t d_row = 0;
    size_t blob_bytes = 0;
    std::string desc;
};

namespace detail {

inline bool mtp_is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

inline std::string mtp_trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && mtp_is_space(s[a])) ++a;
    size_t b = s.size();
    while (b > a && mtp_is_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline int64_t mtp_parse_i64(const std::string& source, const std::string& key, const std::string& val, int64_t min_val) {
    long long x = 0;
    if (!parse_whole(val.c_str(), x) || x < min_val)
        throw std::runtime_error("mtp: " + source + ": invalid " + key + " '" + val + "'");
    return (int64_t) x;
}

inline bool mtp_file_exists(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

inline uint64_t mtp_file_size(std::FILE* f, const std::string& path) {
#ifdef _WIN32
    struct _stat64 stb {};
    const int ok = _fstat64(_fileno(f), &stb);
#else
    struct stat stb {};
    const int ok = fstat(fileno(f), &stb);
#endif
    if (ok != 0 || stb.st_size < 0)
        throw std::runtime_error("mtp: cannot stat " + path);
    return (uint64_t) stb.st_size;
}

}  // namespace detail

// Parse key-value lines from an experts.txt stream (supports native and s2 metadata).
inline MtpExpertMeta parse_mtp_experts_txt(std::istream& in, const std::string& source) {
    MtpExpertMeta meta;
    meta.present = true;
    std::string line;
    while (std::getline(in, line)) {
        const std::string trimmed = detail::mtp_trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        size_t sp = 0;
        while (sp < trimmed.size() && !detail::mtp_is_space(trimmed[sp])) ++sp;
        if (sp == trimmed.size())
            throw std::runtime_error("mtp: " + source + ": malformed line '" + trimmed + "'");
        const std::string key = trimmed.substr(0, sp);
        const std::string val = detail::mtp_trim(trimmed.substr(sp));
        if (val.empty())
            throw std::runtime_error("mtp: " + source + ": missing value for " + key);
        for (char c : val) {
            if (detail::mtp_is_space(c))
                throw std::runtime_error("mtp: " + source + ": unexpected extra token on " + key + " line");
        }
        if (key == "format") meta.format = val;
        else if (key == "gu_type") meta.gu_type = detail::mtp_parse_i64(source, key, val, 0);
        else if (key == "d_type") meta.d_type = detail::mtp_parse_i64(source, key, val, 0);
        else if (key == "n_expert") meta.n_expert = detail::mtp_parse_i64(source, key, val, 1);
        else if (key == "n_embd") meta.n_embd = detail::mtp_parse_i64(source, key, val, 1);
        else if (key == "n_ff") meta.n_ff = detail::mtp_parse_i64(source, key, val, 1);
        else if (key == "blob_bytes") meta.blob_bytes = detail::mtp_parse_i64(source, key, val, 1);
    }
    if (meta.format != "native" && meta.format != "s2")
        throw std::runtime_error("mtp: " + source + ": unsupported format '" + meta.format + "'");
    if (meta.format == "native" && (meta.gu_type < 0 || meta.d_type < 0))
        throw std::runtime_error("mtp: " + source + ": native format requires gu_type and d_type");
    return meta;
}

inline MtpExpertMeta read_mtp_experts_txt(const std::string& path) {
    if (!detail::mtp_file_exists(path)) return MtpExpertMeta{};
    std::ifstream in(path);
    if (!in) throw std::runtime_error("mtp: cannot open " + path);
    return parse_mtp_experts_txt(in, path);
}

// Resolve on-disk MTP expert layout and validate quantization types and dimensions.
inline MtpExpertPlan resolve_mtp_experts_plan(const std::string& dir, const char* env_val, int n_expert, int n_embd, int n_ff) {
    if (n_expert <= 0 || n_embd <= 0 || n_ff <= 0)
        throw std::runtime_error("mtp: invalid expert geometry");
    const std::string txt_path = dir + "/experts.txt";
    const MtpExpertMeta meta = read_mtp_experts_txt(txt_path);

    auto make_native_plan = [&]() -> MtpExpertPlan {
        if (!meta.present || meta.format != "native")
            throw std::runtime_error("mtp: " + txt_path + " with format native is required");
        if (meta.n_expert > 0 && meta.n_expert != n_expert)
            throw std::runtime_error("mtp: experts.txt n_expert mismatch (" + std::to_string(meta.n_expert) + " vs " + std::to_string(n_expert) + ")");
        if (meta.n_embd > 0 && meta.n_embd != n_embd)
            throw std::runtime_error("mtp: experts.txt n_embd mismatch (" + std::to_string(meta.n_embd) + " vs " + std::to_string(n_embd) + ")");
        if (meta.n_ff > 0 && meta.n_ff != n_ff)
            throw std::runtime_error("mtp: experts.txt n_ff mismatch (" + std::to_string(meta.n_ff) + " vs " + std::to_string(n_ff) + ")");
        if (meta.gu_type < 0 || meta.gu_type > (int64_t) UINT32_MAX || meta.d_type < 0 || meta.d_type > (int64_t) UINT32_MAX)
            throw std::runtime_error("mtp: experts.txt expert type out of range");
        const uint32_t gu_type = (uint32_t) meta.gu_type;
        const uint32_t d_type = (uint32_t) meta.d_type;
        if (!mtp_expert_gu_supported(gu_type))
            throw std::runtime_error("mtp: unsupported native gate/up expert type " + std::string(mtp_expert_type_name(gu_type)) + " (" + std::to_string(gu_type) + ")");
        if (!mtp_expert_down_supported(d_type))
            throw std::runtime_error("mtp: unsupported native down expert type " + std::string(mtp_expert_type_name(d_type)) + " (" + std::to_string(d_type) + ")");
        TypeInfo gu_ti{}, d_ti{};
        if (!mtp_expert_type_info(gu_type, gu_ti) || !mtp_expert_type_info(d_type, d_ti))
            throw std::runtime_error("mtp: unknown native expert type");
        if (n_embd % gu_ti.block != 0 || n_ff % d_ti.block != 0)
            throw std::runtime_error("mtp: native expert dimensions do not align with quant block size");
        const size_t gu_row = (size_t) (n_embd / gu_ti.block) * (size_t) gu_ti.bytes;
        const size_t d_row = (size_t) (n_ff / d_ti.block) * (size_t) d_ti.bytes;
        const uint64_t blob64 = 2ull * (uint64_t) n_ff * gu_row + (uint64_t) n_embd * d_row;
        if (meta.blob_bytes > 0 && (uint64_t) meta.blob_bytes != blob64)
            throw std::runtime_error("mtp: experts.txt blob_bytes mismatch (" + std::to_string(meta.blob_bytes) + " vs " + std::to_string(blob64) + ")");
        MtpExpertPlan p;
        p.layout = MtpExpertLayout::NATIVE_EXPERT_MAJOR;
        p.path = dir + "/experts.bin";
        p.gu_type = gu_type;
        p.d_type = d_type;
        p.gu_row = gu_row;
        p.d_row = d_row;
        p.blob_bytes = (size_t) blob64;
        p.desc = std::string("native ") + mtp_expert_type_name(gu_type) + "/" + mtp_expert_type_name(d_type);
        return p;
    };

    auto make_s2_plan = [&]() -> MtpExpertPlan {
        if (n_embd % 64 != 0 || n_ff % 64 != 0)
            throw std::runtime_error("mtp: S2 Q2_0 layout requires dimensions divisible by 64");
        MtpExpertPlan p;
        p.layout = MtpExpertLayout::S2_Q2_0;
        p.path = dir + "/experts.bin";
        p.gu_type = T_Q4_0;
        p.d_type = T_Q4_0;
        p.gu_row = (size_t) (n_embd / 32) * 18;
        p.d_row = (size_t) (n_ff / 32) * 18;
        p.blob_bytes = mtp_s2_blob_bytes(n_embd, n_ff);
        p.desc = "q2_0 (s2 -> Q4_0)";
        return p;
    };

    auto make_planar_plan = [&](const std::string& path, uint32_t type) -> MtpExpertPlan {
        TypeInfo ti{};
        if (!mtp_expert_gu_supported(type) || !mtp_expert_down_supported(type) || !mtp_expert_type_info(type, ti) || ti.block != 32)
            throw std::runtime_error("mtp: planar expert layout requires 32-value blocks");
        if (n_embd % ti.block != 0 || n_ff % ti.block != 0)
            throw std::runtime_error("mtp: planar expert dimensions do not align with 32-value blocks");
        MtpExpertPlan p;
        p.layout = MtpExpertLayout::PLANAR_BLOCKS;
        p.path = path;
        p.gu_type = type;
        p.d_type = type;
        p.gu_row = (size_t) (n_embd / ti.block) * (size_t) ti.bytes;
        p.d_row = (size_t) (n_ff / ti.block) * (size_t) ti.bytes;
        p.blob_bytes = 2ull * (size_t) n_ff * p.gu_row + (size_t) n_embd * p.d_row;
        p.desc = std::string("planar ") + mtp_expert_type_name(type);
        return p;
    };

    const std::string ex = env_val ? std::string(env_val) : std::string("auto");
    if (ex == "auto") {
        if (meta.present && meta.format == "native") return make_native_plan();
        return make_s2_plan();
    }
    if (ex == "native") return make_native_plan();
    if (ex == "q2_0") {
        if (meta.present && meta.format == "native")
            throw std::runtime_error("mtp: FNX_MTP_EXPERTS=q2_0 conflicts with " + txt_path + " (format native)");
        return make_s2_plan();
    }
    if (ex == "q4_0") {
        const std::string planar = dir + "/experts-q4_0.bin";
        if (detail::mtp_file_exists(planar)) return make_planar_plan(planar, T_Q4_0);
        if (meta.present && meta.format == "native" && meta.gu_type == T_Q4_0 && meta.d_type == T_Q4_0)
            return make_native_plan();
        throw std::runtime_error("mtp: cannot open " + planar + " (or native Q4_0 experts.bin via experts.txt)");
    }
    if (ex == "iq4_nl") {
        const std::string planar = dir + "/experts-iq4_nl.bin";
        if (detail::mtp_file_exists(planar)) return make_planar_plan(planar, T_IQ4_NL);
        if (meta.present && meta.format == "native" && meta.gu_type == T_IQ4_NL && meta.d_type == T_IQ4_NL)
            return make_native_plan();
        throw std::runtime_error("mtp: cannot open " + planar + " (or native IQ4_NL experts.bin via experts.txt)");
    }
    throw std::runtime_error("mtp: FNX_MTP_EXPERTS=" + ex + ": expected auto, q2_0, q4_0, iq4_nl, or native");
}

// Load MTP routed expert tensors according to `plan` with strict file-size and EOF validation.
inline void load_mtp_experts(const MtpExpertPlan& plan, int n_expert, int n_embd, int n_ff,
                             std::vector<uint8_t>& gate_b, std::vector<uint8_t>& up_b, std::vector<uint8_t>& down_b) {
    std::FILE* f = std::fopen(plan.path.c_str(), "rb");
    if (!f) {
        const char* hint = plan.layout == MtpExpertLayout::PLANAR_BLOCKS ? " (tools/mtp_experts.py makes it)" : "";
        throw std::runtime_error("mtp: cannot open " + plan.path + hint);
    }
    struct FileCloser {
        std::FILE* fp;
        ~FileCloser() {
            if (fp) std::fclose(fp);
        }
    } closer{f};

    const uint64_t file_bytes = detail::mtp_file_size(f, plan.path);
    const uint64_t expected_bytes = (uint64_t) n_expert * (uint64_t) plan.blob_bytes;
    if (file_bytes != expected_bytes) {
        const char* hint = plan.layout == MtpExpertLayout::S2_Q2_0 ? " (missing experts.txt?)" : "";
        throw std::runtime_error("mtp: " + plan.path + " is " + std::to_string(file_bytes) +
                                 " bytes, expected " + std::to_string(expected_bytes) + hint);
    }

    const size_t gate_exp_bytes = (size_t) n_ff * plan.gu_row;
    const size_t down_exp_bytes = (size_t) n_embd * plan.d_row;
    gate_b.resize((size_t) n_expert * gate_exp_bytes);
    up_b.resize((size_t) n_expert * gate_exp_bytes);
    down_b.resize((size_t) n_expert * down_exp_bytes);

    if (plan.layout == MtpExpertLayout::S2_Q2_0) {
        const size_t row_gu = (size_t) n_embd / 4, row_d = (size_t) n_ff / 4;
        const size_t sc_gu = (size_t) n_embd / 64, sc_d = (size_t) n_ff / 64;
        const size_t o_d = 2ull * (size_t) n_ff * row_gu;
        const size_t o_gs = o_d + (size_t) n_embd * row_d;
        const size_t o_ds = o_gs + 2ull * (size_t) n_ff * sc_gu * 2;
        std::vector<uint8_t> blob(plan.blob_bytes);
        auto unpack_row = [](const uint8_t* codes, const uint8_t* scales, int n, uint8_t* out) {
            for (int b = 0; b < n / 64; ++b) {
                uint16_t d = 0;
                std::memcpy(&d, scales + (size_t) b * 2, 2);
                const uint8_t* c = codes + (size_t) b * 16;
                auto code = [&](int j) { return (c[j / 4] >> (2 * (j % 4))) & 3; };
                for (int h = 0; h < 2; ++h) {
                    uint8_t* o = out + (size_t) (2 * b + h) * 18;
                    std::memcpy(o, &d, 2);
                    for (int i = 0; i < 16; ++i)
                        o[2 + i] = (uint8_t) ((code(32 * h + i) + 7) | ((code(32 * h + 16 + i) + 7) << 4));
                }
            }
        };
        for (int e = 0; e < n_expert; ++e) {
            if (std::fread(blob.data(), 1, plan.blob_bytes, f) != plan.blob_bytes)
                throw std::runtime_error("mtp: " + plan.path + " read truncated");
            for (int r = 0; r < n_ff; ++r) {
                unpack_row(&blob[(2 * (size_t) r) * row_gu], &blob[o_gs + (2 * (size_t) r) * sc_gu * 2], n_embd,
                           &gate_b[((size_t) e * (size_t) n_ff + (size_t) r) * plan.gu_row]);
                unpack_row(&blob[(2 * (size_t) r + 1) * row_gu], &blob[o_gs + (2 * (size_t) r + 1) * sc_gu * 2], n_embd,
                           &up_b[((size_t) e * (size_t) n_ff + (size_t) r) * plan.gu_row]);
            }
            for (int r = 0; r < n_embd; ++r) {
                unpack_row(&blob[o_d + (size_t) r * row_d], &blob[o_ds + (size_t) r * sc_d * 2], n_ff,
                           &down_b[((size_t) e * (size_t) n_embd + (size_t) r) * plan.d_row]);
            }
        }
    } else if (plan.layout == MtpExpertLayout::PLANAR_BLOCKS) {
        if (std::fread(gate_b.data(), 1, gate_b.size(), f) != gate_b.size() ||
            std::fread(up_b.data(), 1, up_b.size(), f) != up_b.size() ||
            std::fread(down_b.data(), 1, down_b.size(), f) != down_b.size())
            throw std::runtime_error("mtp: " + plan.path + " read truncated");
    } else {
        std::vector<uint8_t> blob(plan.blob_bytes);
        for (int e = 0; e < n_expert; ++e) {
            if (std::fread(blob.data(), 1, plan.blob_bytes, f) != plan.blob_bytes)
                throw std::runtime_error("mtp: " + plan.path + " read truncated");
            std::memcpy(&gate_b[(size_t) e * gate_exp_bytes], blob.data(), gate_exp_bytes);
            std::memcpy(&up_b[(size_t) e * gate_exp_bytes], blob.data() + gate_exp_bytes, gate_exp_bytes);
            std::memcpy(&down_b[(size_t) e * down_exp_bytes], blob.data() + 2 * gate_exp_bytes, down_exp_bytes);
        }
    }

    if (std::fgetc(f) != EOF)
        throw std::runtime_error("mtp: " + plan.path + " has unexpected trailing bytes");
}

}  // namespace fnx

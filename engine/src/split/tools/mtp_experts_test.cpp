#include "mtp_experts.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

void need(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

template <class F>
void rejects(F f, const std::string& label) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("invalid input was accepted: " + label);
}

struct TempDir {
    std::string path;
    std::vector<std::string> files;

    TempDir() {
#ifdef _WIN32
        char buf[L_tmpnam]{};
        need(std::tmpnam(buf) != nullptr, "tmpnam");
        path = buf;
        need(_mkdir(path.c_str()) == 0, "mkdir failed: " + path);
#else
        char tmpl[] = "/tmp/fnx_mtp_experts_test_XXXXXX";
        char* d = mkdtemp(tmpl);
        need(d != nullptr, "mkdtemp failed");
        path = d;
#endif
    }

    void track(const std::string& name) {
        files.push_back(path + "/" + name);
    }

    void write_text(const std::string& name, const std::string& content) {
        track(name);
        std::ofstream out(path + "/" + name, std::ios::binary | std::ios::trunc);
        need(out.good(), "write_text open failed: " + name);
        out.write(content.data(), (std::streamsize) content.size());
        need(out.good(), "write_text write failed: " + name);
    }

    void write_bytes(const std::string& name, const std::vector<uint8_t>& data) {
        track(name);
        std::ofstream out(path + "/" + name, std::ios::binary | std::ios::trunc);
        need(out.good(), "write_bytes open failed: " + name);
        out.write((const char*) data.data(), (std::streamsize) data.size());
        need(out.good(), "write_bytes write failed: " + name);
    }

    void remove_file(const std::string& name) {
        std::remove((path + "/" + name).c_str());
    }

    ~TempDir() {
        for (const auto& f : files) std::remove(f.c_str());
#ifdef _WIN32
        _rmdir(path.c_str());
#else
        rmdir(path.c_str());
#endif
    }
};

void test_parser() {
    {
        std::istringstream ss(
            "# native GGUF mtp experts\r\n"
            "\r\n"
            "format native\r\n"
            "gu_type 20\r\n"
            "d_type 20\r\n"
            "n_expert 512\r\n"
            "n_embd 2560\r\n"
            "n_ff 640\r\n"
            "blob_bytes 2764800\r\n"
            "future_note ignored_value\r\n");
        const auto m = fnx::parse_mtp_experts_txt(ss, "iq4_nl.txt");
        need(m.present && m.format == "native", "format native");
        need(m.gu_type == fnx::T_IQ4_NL && m.d_type == fnx::T_IQ4_NL, "iq4_nl types");
        need(m.n_expert == 512 && m.n_embd == 2560 && m.n_ff == 640, "geometry");
        need(m.blob_bytes == 2764800, "blob_bytes");
    }
    {
        std::istringstream ss("format native\ngu_type 20\nd_type 2\n");
        const auto m = fnx::parse_mtp_experts_txt(ss, "mixed.txt");
        need(m.gu_type == fnx::T_IQ4_NL && m.d_type == fnx::T_Q4_0, "mixed types");
    }
    {
        std::istringstream ss("format s2\n");
        const auto m = fnx::parse_mtp_experts_txt(ss, "s2.txt");
        need(m.present && m.format == "s2", "s2 format");
    }

    rejects([] {
        std::istringstream ss("format bogus\ngu_type 20\nd_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "bad_fmt");
    }, "bad format");
    rejects([] {
        std::istringstream ss("gu_type 20\nd_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "no_fmt");
    }, "missing format");
    rejects([] {
        std::istringstream ss("format native\ngu_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "no_dtype");
    }, "missing d_type");
    rejects([] {
        std::istringstream ss("format native\ngu_type 20x\nd_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "bad_int");
    }, "non-integer type");
    rejects([] {
        std::istringstream ss("format native\ngu_type 20 extra\nd_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "trailing");
    }, "trailing token");
    rejects([] {
        std::istringstream ss("format native\ngu_type\nd_type 20\n");
        fnx::parse_mtp_experts_txt(ss, "missing_val");
    }, "missing value");
}

void test_resolution_matrix() {
    TempDir d;

    // 1) Absent experts.txt -> auto falls back to S2 Q2_0; native throws.
    {
        const auto p_def = fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640);
        need(p_def.layout == fnx::MtpExpertLayout::S2_Q2_0 && p_def.gu_type == fnx::T_Q4_0 && p_def.d_type == fnx::T_Q4_0,
             "default fallback to s2");
        need(p_def.blob_bytes == 1382400, "s2 blob_bytes");
        const auto p_auto = fnx::resolve_mtp_experts_plan(d.path, "auto", 512, 2560, 640);
        need(p_auto.layout == fnx::MtpExpertLayout::S2_Q2_0, "explicit auto without txt");
        const auto p_q2 = fnx::resolve_mtp_experts_plan(d.path, "q2_0", 512, 2560, 640);
        need(p_q2.layout == fnx::MtpExpertLayout::S2_Q2_0, "explicit q2_0 without txt");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, "native", 512, 2560, 640); }, "native without experts.txt");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, "bogus", 512, 2560, 640); }, "bogus env");
    }

    // 2) Native experts.txt (IQ4_NL/IQ4_NL) -> auto resolves native, q2_0/q4_0 reject.
    d.write_text("experts.txt",
                 "format native\n"
                 "gu_type 20\n"
                 "d_type 20\n"
                 "n_expert 512\n"
                 "n_embd 2560\n"
                 "n_ff 640\n"
                 "blob_bytes 2764800\n");
    {
        const auto p_def = fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640);
        need(p_def.layout == fnx::MtpExpertLayout::NATIVE_EXPERT_MAJOR, "auto native iq4_nl");
        need(p_def.gu_type == fnx::T_IQ4_NL && p_def.d_type == fnx::T_IQ4_NL && p_def.blob_bytes == 2764800,
             "native iq4_nl plan fields");
        const auto p_nat = fnx::resolve_mtp_experts_plan(d.path, "native", 512, 2560, 640);
        need(p_nat.layout == fnx::MtpExpertLayout::NATIVE_EXPERT_MAJOR, "explicit native");
        const auto p_iq = fnx::resolve_mtp_experts_plan(d.path, "iq4_nl", 512, 2560, 640);
        need(p_iq.layout == fnx::MtpExpertLayout::NATIVE_EXPERT_MAJOR && p_iq.gu_type == fnx::T_IQ4_NL,
             "iq4_nl fallback to native");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, "q2_0", 512, 2560, 640); }, "q2_0 conflict with native");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, "q4_0", 512, 2560, 640); }, "q4_0 mismatch with native iq4_nl");
    }

    // 3) Planar files win when present for explicit q4_0 / iq4_nl.
    d.write_bytes("experts-q4_0.bin", {0});
    d.write_bytes("experts-iq4_nl.bin", {0});
    {
        const auto pq4 = fnx::resolve_mtp_experts_plan(d.path, "q4_0", 512, 2560, 640);
        need(pq4.layout == fnx::MtpExpertLayout::PLANAR_BLOCKS && pq4.gu_type == fnx::T_Q4_0 && pq4.d_type == fnx::T_Q4_0,
             "planar q4_0");
        const auto piq = fnx::resolve_mtp_experts_plan(d.path, "iq4_nl", 512, 2560, 640);
        need(piq.layout == fnx::MtpExpertLayout::PLANAR_BLOCKS && piq.gu_type == fnx::T_IQ4_NL && piq.d_type == fnx::T_IQ4_NL,
             "planar iq4_nl");
    }
    d.remove_file("experts-q4_0.bin");
    d.remove_file("experts-iq4_nl.bin");

    // 4) Native Q4_0 fallback when planar experts-q4_0.bin is absent.
    d.write_text("experts.txt", "format native\ngu_type 2\nd_type 2\n");
    {
        const auto pq4 = fnx::resolve_mtp_experts_plan(d.path, "q4_0", 512, 2560, 640);
        need(pq4.layout == fnx::MtpExpertLayout::NATIVE_EXPERT_MAJOR && pq4.gu_type == fnx::T_Q4_0, "q4_0 fallback to native");
    }

    // 5) Metadata geometry and unsupported type rejections.
    d.write_text("experts.txt", "format native\ngu_type 20\nd_type 20\nn_expert 256\nn_embd 2560\nn_ff 640\n");
    rejects([&] { fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640); }, "n_expert mismatch");
    d.write_text("experts.txt", "format native\ngu_type 20\nd_type 20\nblob_bytes 123\n");
    rejects([&] { fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640); }, "blob_bytes mismatch");
    for (int bad_gu : {42, 14, 40, 999}) {
        d.write_text("experts.txt", "format native\ngu_type " + std::to_string(bad_gu) + "\nd_type 2\n");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640); }, "bad gu_type");
    }
    for (int bad_d : {12, 21, 42, 40, 999}) {
        d.write_text("experts.txt", "format native\ngu_type 20\nd_type " + std::to_string(bad_d) + "\n");
        rejects([&] { fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640); }, "bad d_type");
    }

    // 6) Valid mixed & larger type resolution check.
    d.write_text("experts.txt", "format native\ngu_type 8\nd_type 8\n");
    need(fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640).gu_type == fnx::T_Q8_0, "native Q8_0");
    d.write_text("experts.txt", "format native\ngu_type 12\nd_type 2\n");
    const auto p_q4k = fnx::resolve_mtp_experts_plan(d.path, nullptr, 512, 2560, 640);
    need(p_q4k.gu_type == fnx::T_Q4_K && p_q4k.d_type == fnx::T_Q4_0, "native Q4_K/Q4_0");
}

void test_load_native_and_planar() {
    TempDir d;
    constexpr int NE = 2, H = 64, F = 64;
    constexpr size_t GU_ROW = (H / 32) * 18;
    constexpr size_t D_ROW = (F / 32) * 18;
    constexpr size_t G_EXP = F * GU_ROW;
    constexpr size_t D_EXP = H * D_ROW;
    constexpr size_t BLOB = 2 * G_EXP + D_EXP;

    std::vector<uint8_t> exp_gate((size_t) NE * G_EXP);
    std::vector<uint8_t> exp_up((size_t) NE * G_EXP);
    std::vector<uint8_t> exp_down((size_t) NE * D_EXP);
    for (int e = 0; e < NE; ++e) {
        for (size_t i = 0; i < G_EXP; ++i) {
            exp_gate[(size_t) e * G_EXP + i] = (uint8_t) ((e * 73 + i * 3 + 11) & 0xff);
            exp_up[(size_t) e * G_EXP + i] = (uint8_t) ((e * 97 + i * 5 + 29) & 0xff);
        }
        for (size_t i = 0; i < D_EXP; ++i)
            exp_down[(size_t) e * D_EXP + i] = (uint8_t) ((e * 41 + i * 7 + 53) & 0xff);
    }

    // Native expert-major IQ4_NL/IQ4_NL and mixed IQ4_NL/Q4_0.
    std::vector<uint8_t> native_bin((size_t) NE * BLOB);
    for (int e = 0; e < NE; ++e) {
        uint8_t* dst = native_bin.data() + (size_t) e * BLOB;
        std::memcpy(dst, &exp_gate[(size_t) e * G_EXP], G_EXP);
        std::memcpy(dst + G_EXP, &exp_up[(size_t) e * G_EXP], G_EXP);
        std::memcpy(dst + 2 * G_EXP, &exp_down[(size_t) e * D_EXP], D_EXP);
    }
    d.write_bytes("experts.bin", native_bin);

    for (auto pair : std::vector<std::pair<int, int>>{{fnx::T_IQ4_NL, fnx::T_IQ4_NL}, {fnx::T_IQ4_NL, fnx::T_Q4_0}}) {
        d.write_text("experts.txt",
                     "format native\n"
                     "gu_type " + std::to_string(pair.first) + "\n"
                     "d_type " + std::to_string(pair.second) + "\n"
                     "n_expert " + std::to_string(NE) + "\n"
                     "n_embd " + std::to_string(H) + "\n"
                     "n_ff " + std::to_string(F) + "\n"
                     "blob_bytes " + std::to_string(BLOB) + "\n");
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, nullptr, NE, H, F);
        std::vector<uint8_t> g, u, dn;
        fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
        need(g == exp_gate && u == exp_up && dn == exp_down, "native layout byte-exact check");
    }

    // Missing experts.txt causes native experts.bin to fail S2 exact-size check cleanly.
    d.remove_file("experts.txt");
    rejects([&] {
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, nullptr, NE, H, F);
        std::vector<uint8_t> g, u, dn;
        fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
    }, "oversized experts.bin without experts.txt");

    // Truncated & oversized native file checks.
    d.write_text("experts.txt", "format native\ngu_type 20\nd_type 20\n");
    {
        auto trunc = native_bin;
        trunc.pop_back();
        d.write_bytes("experts.bin", trunc);
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, "native", NE, H, F);
        rejects([&] {
            std::vector<uint8_t> g, u, dn;
            fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
        }, "truncated native experts.bin");
    }
    {
        auto extra = native_bin;
        extra.push_back(0xAA);
        d.write_bytes("experts.bin", extra);
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, "native", NE, H, F);
        rejects([&] {
            std::vector<uint8_t> g, u, dn;
            fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
        }, "trailing byte native experts.bin");
    }

    // Planar layout (experts-iq4_nl.bin & experts-q4_0.bin).
    std::vector<uint8_t> planar;
    planar.insert(planar.end(), exp_gate.begin(), exp_gate.end());
    planar.insert(planar.end(), exp_up.begin(), exp_up.end());
    planar.insert(planar.end(), exp_down.begin(), exp_down.end());
    d.write_bytes("experts-iq4_nl.bin", planar);
    {
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, "iq4_nl", NE, H, F);
        std::vector<uint8_t> g, u, dn;
        fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
        need(g == exp_gate && u == exp_up && dn == exp_down, "planar iq4_nl byte-exact");
    }
    planar.push_back(1);
    d.write_bytes("experts-iq4_nl.bin", planar);
    rejects([&] {
        const auto plan = fnx::resolve_mtp_experts_plan(d.path, "iq4_nl", NE, H, F);
        std::vector<uint8_t> g, u, dn;
        fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);
    }, "trailing byte planar");
}

void test_load_s2_q2_0() {
    TempDir d;
    constexpr int NE = 2, H = 64, F = 64;
    constexpr size_t ROW_GU = H / 4, ROW_D = F / 4, SC_GU = H / 64, SC_D = F / 64;
    constexpr size_t O_D = 2ull * F * ROW_GU;
    constexpr size_t O_GS = O_D + (size_t) H * ROW_D;
    constexpr size_t O_DS = O_GS + 2ull * F * SC_GU * 2;
    constexpr size_t BLOB = fnx::mtp_s2_blob_bytes(H, F);

    std::vector<uint8_t> s2((size_t) NE * BLOB, 0);
    auto set_code = [](uint8_t* codes, int j, uint8_t c) {
        codes[j / 4] |= (uint8_t) ((c & 3u) << (2 * (j % 4)));
    };
    auto set_row_pattern = [&](uint8_t* codes, uint8_t* scales, uint16_t scale, uint8_t base) {
        std::memcpy(scales, &scale, 2);
        for (int j = 0; j < 64; ++j) set_code(codes, j, (uint8_t) ((base + j) & 3));
    };

    for (int e = 0; e < NE; ++e) {
        uint8_t* b = s2.data() + (size_t) e * BLOB;
        set_row_pattern(&b[0], &b[O_GS], (uint16_t) (0x3C00 + e), (uint8_t) (e + 1));       // gate row 0
        set_row_pattern(&b[ROW_GU], &b[O_GS + SC_GU * 2], (uint16_t) (0x4000 + e), (uint8_t) (e + 2));   // up row 0
        set_row_pattern(&b[O_D], &b[O_DS], (uint16_t) (0x4200 + e), (uint8_t) (e + 3));      // down row 0
    }
    d.write_bytes("experts.bin", s2);

    const auto plan = fnx::resolve_mtp_experts_plan(d.path, nullptr, NE, H, F);
    need(plan.layout == fnx::MtpExpertLayout::S2_Q2_0, "s2 resolve");
    std::vector<uint8_t> g, u, dn;
    fnx::load_mtp_experts(plan, NE, H, F, g, u, dn);

    for (int e = 0; e < NE; ++e) {
        auto check_q4 = [](const uint8_t* q4_row, uint16_t expected_scale, uint8_t base) {
            for (int h = 0; h < 2; ++h) {
                const uint8_t* blk = q4_row + (size_t) h * 18;
                uint16_t d = 0;
                std::memcpy(&d, blk, 2);
                need(d == expected_scale, "s2 unpacked scale mismatch");
                for (int i = 0; i < 16; ++i) {
                    const uint8_t c_lo = (uint8_t) ((base + 32 * h + i) & 3);
                    const uint8_t c_hi = (uint8_t) ((base + 32 * h + 16 + i) & 3);
                    const uint8_t want = (uint8_t) ((c_lo + 7) | ((c_hi + 7) << 4));
                    need(blk[2 + i] == want, "s2 unpacked nibble mismatch");
                }
            }
        };
        const size_t g_stride = (size_t) e * F * ((H / 32) * 18);
        const size_t d_stride = (size_t) e * H * ((F / 32) * 18);
        check_q4(&g[g_stride], (uint16_t) (0x3C00 + e), (uint8_t) (e + 1));
        check_q4(&u[g_stride], (uint16_t) (0x4000 + e), (uint8_t) (e + 2));
        check_q4(&dn[d_stride], (uint16_t) (0x4200 + e), (uint8_t) (e + 3));
    }

    s2.pop_back();
    d.write_bytes("experts.bin", s2);
    rejects([&] {
        std::vector<uint8_t> g2, u2, dn2;
        fnx::load_mtp_experts(plan, NE, H, F, g2, u2, dn2);
    }, "truncated s2 experts.bin");
}

}  // namespace

int main() {
    try {
        test_parser();
        test_resolution_matrix();
        test_load_native_and_planar();
        test_load_s2_q2_0();
        std::cout << "PASS mtp_experts parser, resolution fallback matrix, and S2/planar/native loading\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

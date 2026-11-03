#include "attn_head_split.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string expected = argv[1];
    try {
        const bool enabled = fnx::attn_head_split();
        if (expected == "error") return 1;
        if (enabled != (expected == "on")) return 1;
        if (fnx::attn_projected_q_heads() != (enabled ? 12 : fnx::NQH)) return 1;
        if (fnx::attn_projected_kv_heads() != (enabled ? 1 : fnx::NKVH)) return 1;
        if (fnx::attn_position_partitions() != (enabled ? 2 : 1)) return 1;
        for (int lanes : {1, 2}) for (bool batched : {false, true}) {
            bool rejected = false;
            try { fnx::attn_head_check_runtime(lanes, batched); }
            catch (const std::runtime_error&) { rejected = true; }
            if (rejected != (enabled && (lanes != 1 || batched))) return 1;
        }
    } catch (const std::runtime_error& e) {
        if (expected != "error" || std::string(e.what()).find("FNX_ATTN_HEAD_SPLIT") == std::string::npos) return 1;
    }
}

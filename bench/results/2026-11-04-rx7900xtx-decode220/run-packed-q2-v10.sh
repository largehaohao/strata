#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
build=build-tensor-parralize-baseline
out=bench/results/2026-11-04-rx7900xtx-decode220
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib
timeout -k 10s 300s "$build/src/split/split_moe_q2_down_test" > "$out/packed-q2-v10-checks.txt" 2>&1
tail -1 "$out/packed-q2-v10-checks.txt"
timeout -k 10s 300s "$build/src/split/split_moe_q2_mma_test" > "$out/packed-q2-v10-mma-checks.txt" 2>&1
tail -1 "$out/packed-q2-v10-mma-checks.txt"
"$build/src/split/split_gguf_release_test" > "$out/packed-q2-v10-reader-checks.txt" 2>&1
git diff --binary -- src/split > "$out/candidate-v10-packed-q2.patch"
export TP_EXE=$build/strata-split-packed-q2-v10 FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.625
bash "$out/run-heads-experiment.sh" "$out/packed-q2-v10" > "$build/packed-q2-v10-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/gu-cache-v7" "$out/packed-q2-v10" --out "$out/packed-q2-v10-comparison.json"
export FNX_PACKED_Q2_DOWN=0 TP_EXPERT_RAM_GIB=13.5
bash "$out/run-heads-experiment.sh" "$out/packed-q2-v10-control" > "$build/packed-q2-v10-control-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/packed-q2-v10-control" "$out/packed-q2-v10" --out "$out/packed-q2-v10-same-binary.json"

#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
build=build-tensor-parralize-baseline
out=bench/results/2026-11-04-rx7900xtx-decode220
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib
timeout 600s stdbuf -oL "$build/src/split/split_moe_cache_gu_test" > "$build/gu-cache-v7-gu-checks.txt" 2>&1
tail -n 2 "$build/gu-cache-v7-gu-checks.txt"
git diff --binary -- src/split docs/TENSOR_PARALLEL_RX7900XTX.md tools/tp_decode_bench.py > "$out/candidate-v7-gu-cache.patch"
cp "$build/gu-cache-v7-gu-checks.txt" "$build/gu-cache-v7-cache-checks.txt" "$out/"
export TP_EXE=$build/strata-split-gu-cache-v7 FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
bash "$out/run-heads-experiment.sh" "$out/gu-cache-v7" > "$build/gu-cache-v7-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/cache-shared-v6-control" "$out/gu-cache-v7" --out "$out/gu-cache-v7-comparison.json"
export FNX_STATIC_CACHE_FUSED_GU=0
bash "$out/run-heads-experiment.sh" "$out/gu-cache-v7-control" > "$build/gu-cache-v7-control-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/gu-cache-v7-control" "$out/gu-cache-v7" --out "$out/gu-cache-v7-same-binary.json"

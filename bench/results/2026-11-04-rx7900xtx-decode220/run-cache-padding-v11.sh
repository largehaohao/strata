#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
build=build-tensor-parralize-baseline
out=bench/results/2026-11-04-rx7900xtx-decode220
"$build/src/split/split_static_cache_layout_test" > "$out/cache-padding-v11-layout-checks.txt" 2>&1
"$build/src/split/split_static_kv_cache_test" > "$out/cache-padding-v11-kv-checks.txt" 2>&1
git diff --binary -- src/split > "$out/candidate-v11-cache-padding.patch"
export TP_EXE=$build/strata-split-cache-padding-v11 FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.625 FNX_STATIC_CACHE_MIN_PADDING=1
bash "$out/run-heads-experiment.sh" "$out/cache-padding-v11" > "$build/cache-padding-v11-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/packed-q2-v10" "$out/cache-padding-v11" --out "$out/cache-padding-v11-comparison.json"
export FNX_STATIC_CACHE_MIN_PADDING=0
bash "$out/run-heads-experiment.sh" "$out/cache-padding-v11-control" > "$build/cache-padding-v11-control-console.txt" 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/cache-padding-v11-control" "$out/cache-padding-v11" --out "$out/cache-padding-v11-same-binary.json"

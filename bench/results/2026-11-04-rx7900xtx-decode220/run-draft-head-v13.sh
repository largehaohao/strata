#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
build=build-tensor-parralize-baseline
out=bench/results/2026-11-04-rx7900xtx-decode220
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib
timeout -k 10 300 "$build/draft_head_prod_scratch" > "$out/draft-head-v13-checks.txt" 2>&1
git diff --binary -- src/split > "$out/candidate-v13-draft-head.patch"
export TP_EXE=$build/strata-split-draft-head-v13 FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.625 FNX_STATIC_CACHE_MIN_PADDING=1
for mode in 2 3 0; do
  export FNX_MTP_HEAD_MODE=$mode
  bash "$out/run-heads-experiment.sh" "$out/draft-head-v13-mode$mode" > "$build/draft-head-v13-mode$mode-console.txt" 2>&1
done
for mode in 2 3; do
  .venv/bin/python "$out/compare_draft_panels.py" "$out/draft-head-v13-mode0" "$out/draft-head-v13-mode$mode" --out "$out/draft-head-v13-mode$mode-comparison.json"
done

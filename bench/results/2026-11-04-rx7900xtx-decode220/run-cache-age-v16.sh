#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
build=build-tensor-parralize-baseline
out=bench/results/2026-11-04-rx7900xtx-decode220
export TP_EXE=$build/strata-split-cache-age-v16 FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.625 FNX_STATIC_CACHE_MIN_PADDING=1 FNX_MTP_HEAD_MODE=3
for age in 2 1; do
 export FNX_STATIC_CACHE_MAX_AGE=$age
 bash "$out/run-heads-experiment.sh" "$out/cache-age-v16-age$age" > "$build/cache-age-v16-age$age-console.txt" 2>&1
done
.venv/bin/python "$out/compare_panels.py" "$out/cache-age-v16-age1" "$out/cache-age-v16-age2" --out "$out/cache-age-v16-comparison.json"

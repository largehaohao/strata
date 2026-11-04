#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
out=bench/results/2026-11-04-rx7900xtx-decode220
export TP_EXE=build-tensor-parralize-baseline/strata-split-cache-padding-v11
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.5 FNX_STATIC_CACHE_MIN_PADDING=1
bash "$out/run-heads-experiment.sh" "$out/cache-padding-v11-ram11p5" > build-tensor-parralize-baseline/cache-padding-v11-ram11p5-console.txt 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/cache-padding-v11" "$out/cache-padding-v11-ram11p5" --out "$out/cache-padding-v11-ram11p5-comparison.json"

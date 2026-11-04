#!/usr/bin/env bash
set -euo pipefail
export TP_EXE=${TP_EXE:-build-tensor-parralize-baseline/strata-split-packed-q2-v10}
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1 FNX_PACKED_Q2_DOWN=1
export TP_EXPERT_RAM_GIB=11.625
exec bash "$(dirname -- "$0")/run-heads-experiment.sh" "$@"

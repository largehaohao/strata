#!/usr/bin/env bash
set -euo pipefail
export TP_EXE=${TP_EXE:-build-tensor-parralize-baseline/strata-split-draft-head-v13}
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1 FNX_PACKED_Q2_DOWN=1
export FNX_STATIC_CACHE_MIN_PADDING=1 FNX_MTP_HEAD_MODE=3 TP_EXPERT_RAM_GIB=11.625
exec bash "$(dirname -- "$0")/run-heads-experiment.sh" "$@"

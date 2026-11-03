#!/usr/bin/env bash
# Validated November 4 candidate; a new output directory is required.
set -euo pipefail
tp_recipe_dir=$(cd -- "$(dirname -- "$0")" && pwd)
export TP_EXE=${TP_EXE:-build-tensor-parralize-baseline/strata-split-gu-cache-v7}
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
exec bash "$tp_recipe_dir/run-heads-experiment.sh" "$@"

#!/usr/bin/env bash
# Reproduce the validated single-request panel on 192.168.1.102.
set -euo pipefail
if (( $# < 1 )); then
    printf 'Usage: bash %s NEW_OUTPUT_DIRECTORY [tp_decode_bench.py options]\n' "$0" >&2
    exit 2
fi
tp_repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$tp_repo"
tp_out=$1
shift
export HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1
export FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB=13.875 FNX_STATIC_CACHE_SLOTS=1024
export FNX_STATIC_CACHE_GROUPS=2 FNX_STATIC_CACHE_FUSED_DN=16
export FNX_MOE_STAGE=0 FNX_PF_ROWS=256 FNX_IQ_REUSE=1 FNX_IQ_GROUP=16 FNX_MOE_PRE_ROWS=1
export FNX_STATIC_HOST_CACHE=1 FNX_EMBED_WARM_AFTER_LOAD=1 FNX_RELEASE_COPIED_WEIGHTS=1
export FNX_KV_CONTEXT_ONLY=1 FNX_HC_ROW_GROUP=64 FNX_Q8S_ROW_GROUP=2
export FNX_GDN_REPLAY=1 FNX_BATCH_DRAFTS=0
export FNX_MTP_EXPERTS=q4_0 FNX_MTP_HEAD_Q4=1 FNX_MTP_LAST_HEAD=1 FNX_DECODE_TIMING=1
export FNX_EXPERT_PROFILE="$tp_repo/bench/results/2026-11-03-rx7900xtx-tensor-parallel/swift-decode-routes.bin"
export LD_LIBRARY_PATH="$tp_repo/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$tp_repo/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib"
exec "${TP_PYTHON:-.venv/bin/python}" tools/tp_decode_bench.py \
    --exe "${TP_EXE:-build-tensor-parralize-baseline/strata-split-mtp-last-head}" \
    --model "${TP_MODEL:-/home/john/Strata-data/models/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}" \
    --mtp "${TP_MTP:-build-tensor-parralize-baseline/mtp-rt-q4}" \
    --tokenizer "${TP_TOKENIZER:-/home/john/Strata-data/packs/swift-iq3_s/tokenizer}" \
    --context 32768 --tokens 2048 --repeats 2 --spec 6 --lookup 15 --out "$tp_out" "$@"

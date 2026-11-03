#!/bin/bash
set -eu
cd /home/john/Strata
export HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1
export FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB="${FNX_EXPERT_RAM_GIB:-16}" FNX_STATIC_CACHE_SLOTS="${FNX_STATIC_CACHE_SLOTS:-1024}"
export FNX_MOE_STAGE=0 FNX_PF_ROWS=256 FNX_IQ_REUSE=1 FNX_IQ_GROUP=16 FNX_MOE_PRE_ROWS=1
export FNX_STATIC_HOST_CACHE=1 FNX_EMBED_WARM_AFTER_LOAD=1 FNX_DECODE_TIMING=1
export FNX_RELEASE_COPIED_WEIGHTS="$3"
export FNX_EXPERT_PROFILE="$PWD/bench/results/2026-11-03-rx7900xtx-tensor-parallel/swift-decode-routes.bin"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib"
bench_exe="$1"
bench_out="$2"
shift 3
exec .venv/bin/python build-tensor-parralize-baseline/tp_sample_check.py \
 --exe "$bench_exe" \
 --model /home/john/Strata-data/models/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
 --mtp /home/john/Strata-data/mtp/rt \
 --tokenizer /home/john/Strata-data/packs/swift-iq3_s/tokenizer \
 --context 32768 --tokens 2048 --repeats 2 --out "$bench_out" "$@"

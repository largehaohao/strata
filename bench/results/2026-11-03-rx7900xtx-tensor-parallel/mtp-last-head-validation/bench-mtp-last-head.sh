#!/bin/bash
set -eu
cd /home/john/Strata
export FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=32 FNX_STATIC_CACHE_GROUPS=2
export FNX_KV_CONTEXT_ONLY=1 FNX_Q8S_ROW_GROUP=2 FNX_MTP_EXPERTS=q4_0
export FNX_GDN_REPLAY=1 FNX_BATCH_DRAFTS=0 FNX_MTP_HEAD_Q4=1 FNX_EXPERT_RAM_GIB=13.875
root=bench/results/2026-11-03-rx7900xtx-tensor-parallel
runner=build-tensor-parralize-baseline/run-tp-bench.sh
exe=build-tensor-parralize-baseline/strata-split-mtp-last-head
mtp=build-tensor-parralize-baseline/mtp-rt-q4
FNX_MTP_LAST_HEAD=1 bash "$runner" "$exe" "$root/bench-mtp-last-head" 1 --mtp "$mtp" > build-tensor-parralize-baseline/bench-mtp-last-head-console.txt 2>&1
FNX_MTP_LAST_HEAD=0 bash "$runner" "$exe" "$root/bench-mtp-last-head-control" 1 --mtp "$mtp" > build-tensor-parralize-baseline/bench-mtp-last-head-control-console.txt 2>&1

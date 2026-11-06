#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
out=bench/results/2026-11-04-rx7900xtx-decode220
export TP_EXE=build-tensor-parralize-baseline/strata-split-q8s-ab-v26
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.25 TP_PREFILL_ROWS=256 FNX_STATIC_CACHE_MIN_PADDING=1
export FNX_STATIC_CACHE_MAX_AGE=2 FNX_ROUTER_BF16=1 FNX_MTP_HEAD_MODE=3
export TP_CPU_AFFINITY=0-3 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
unset HSA_ENABLE_INTERRUPT HSA_ENABLE_MWAITX FNX_GPU_TIMING
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib
build-tensor-parralize-baseline/src/split/split_q8s_ab_test > "$out/q8s-ab-v26-checks.txt" 2>&1
for mode in 1 0; do
 FNX_Q8S_FUSED_AB=$mode bash "$out/run-heads-experiment.sh" "$out/q8s-ab-v26-mode$mode" > "build-tensor-parralize-baseline/q8s-ab-v26-mode$mode-console.txt" 2>&1
done
.venv/bin/python "$out/compare_panels.py" "$out/q8s-ab-v26-mode0" "$out/q8s-ab-v26-mode1" --out "$out/q8s-ab-v26-comparison.json"
.venv/bin/python "$out/compare_panels.py" "$out/cpu-affinity-v21-physical4" "$out/q8s-ab-v26-mode0" --out "$out/q8s-ab-v26-control-compatibility.json"
.venv/bin/python - <<'PY'
from pathlib import Path
import json
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
assert 'PASS 1152' in (out/'q8s-ab-v26-checks.txt').read_text()
for mode in (1,0):
 p=out/f'q8s-ab-v26-mode{mode}'
 r=json.loads((p/'results.json').read_text())
 assert r['status']=='complete' and r['engine_returncode']==0
 assert all(x['decode_faults_after_first_token']['major']==0 for x in r['runs'])
 assert r['env']['FNX_Q8S_FUSED_AB']==str(mode) and r['env']['FNX_PF_ROWS']=='256'
 for key in ('cpu_affinity_after_load','cpu_affinity_after_decode'):
  assert r[key]['main']==[0,1,2,3]
  assert r[key]['threads'] and all(mask==[0,1,2,3] for mask in r[key]['threads'].values())
 log=(p/'engine.log').read_text()
 assert ('fused two-card Q8S and alpha/beta projections' in log)==(mode==1)
 print(mode,r['weighted_decode_tps'],'exact outputs/draft counts, no decode major faults, all CPU masks verified')
PY

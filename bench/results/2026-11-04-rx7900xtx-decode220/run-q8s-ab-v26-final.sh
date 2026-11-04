#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
out=bench/results/2026-11-04-rx7900xtx-decode220
export TP_EXE=build-tensor-parralize-baseline/strata-split-q8s-ab-v26
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.25 FNX_STATIC_CACHE_MIN_PADDING=1
export FNX_MTP_HEAD_MODE=3 FNX_STATIC_CACHE_MAX_AGE=2 FNX_ROUTER_BF16=1
export TP_CPU_AFFINITY=0-3 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 TP_PREFILL_ROWS=256 FNX_Q8S_FUSED_AB=1
unset HSA_ENABLE_INTERRUPT HSA_ENABLE_MWAITX FNX_GPU_TIMING
bash "$out/run-heads-experiment.sh" "$out/q8s-ab-v26-mode1-repeat" > build-tensor-parralize-baseline/q8s-ab-v26-mode1-repeat-console.txt 2>&1
.venv/bin/python "$out/compare_panels.py" "$out/q8s-ab-v26-mode1" "$out/q8s-ab-v26-mode1-repeat" --out "$out/q8s-ab-v26-repeat-comparison.json"
.venv/bin/python "$out/compare_panels.py" "$out/draft-head-v13-mode3" "$out/q8s-ab-v26-mode1" --out "$out/q8s-ab-v26-v13-comparison.json"
.venv/bin/python - <<'PY'
from pathlib import Path
import json
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
a=json.loads((out/'q8s-ab-v26-mode1/results.json').read_text())
b=json.loads((out/'q8s-ab-v26-mode1-repeat/results.json').read_text())
assert a['engine_pid']!=b['engine_pid']
assert a['command']==b['command'] and a['env']==b['env']
for p in (a,b):
    assert p['status']=='complete' and p['engine_returncode']==0
    assert p['env']['FNX_Q8S_FUSED_AB']=='1' and p['env']['FNX_PF_ROWS']=='256'
    assert p['weighted_decode_tps']>=220, p['weighted_decode_tps']
    assert all(r['decode_faults_after_first_token']['major']==0 for r in p['runs'])
    for key in ('cpu_affinity_after_load','cpu_affinity_after_decode'):
        assert p[key]['main']==[0,1,2,3]
        assert p[key]['threads'] and all(mask==[0,1,2,3] for mask in p[key]['threads'].values())
print('PASS two independent complete panels >=220, same workload, all thread masks verified')
PY
taskset -c 0-3 bash "$out/validate-q8s-ab-v26.sh"

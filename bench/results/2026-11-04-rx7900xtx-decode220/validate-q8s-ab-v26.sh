#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
old=bench/results/2026-11-03-rx7900xtx-tensor-parallel
out=bench/results/2026-11-04-rx7900xtx-decode220
runner=$out/run-heads-experiment.sh
export TP_CPU_AFFINITY=0-3 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 TP_PREFILL_ROWS=256 FNX_Q8S_FUSED_AB=1
unset HSA_ENABLE_INTERRUPT HSA_ENABLE_MWAITX FNX_GPU_TIMING
.venv/bin/python - <<'CHECK_AFFINITY'
import os
assert sorted(os.sched_getaffinity(0))==[0,1,2,3], 'Run the entire QA script under taskset -c 0-3'
CHECK_AFFINITY
export TP_EXE=build-tensor-parralize-baseline/strata-split-q8s-ab-v26
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.25 FNX_STATIC_CACHE_MIN_PADDING=1 FNX_MTP_HEAD_MODE=3 FNX_STATIC_CACHE_MAX_AGE=2 FNX_ROUTER_BF16=1

bash "$runner" "$out/quality-q8s-ab-v26" --cases "$old/quality-prompts.json" --repeats 1 > build-tensor-parralize-baseline/quality-q8s-ab-v26-console.txt 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
old=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
a=json.loads((out/'quality-q8s-ab-v26/results.json').read_text())
b=json.loads((old/'quality-split/results.json').read_text())
assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs'])
for x,y in zip(a['runs'],b['runs']):
    for k in ['case','rep','ids','input_ids','finish']:
        assert x[k]==y[k], (x['case'],k)
print('Quality input/output IDs exactly match reviewed reference.')
PY
.venv/bin/python tools/tp_code_quality.py "$out/quality-q8s-ab-v26"
bash "$runner" "$out/near-context-q8s-ab-v26" --cases "$old/near-context-prompts.json" --prefix-file "$old/near-context-prefix.txt" --tokens 64 --repeats 1 --timeout 1800 > build-tensor-parralize-baseline/near-context-q8s-ab-v26-console.txt 2>&1
bash "$runner" "$out/heldout-q8s-ab-v26" --cases "$old/mtp-heldout-prompts.json" --tokens 1024 --repeats 1 > build-tensor-parralize-baseline/heldout-q8s-ab-v26-console.txt 2>&1
FNX_ARENA_CHURN=1 bash "$runner" "$out/churn-q8s-ab-v26" --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/churn-q8s-ab-v26-console.txt 2>&1

FNX_PREFETCH_PROBE=1 bash "$runner" "$out/probe-q8s-ab-v26" --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/probe-q8s-ab-v26-console.txt 2>&1

# The sampling harness uses the same model and fixed seed/penalties as the
# previously checked baseline. Export settings normally owned by the runner.
export FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=16 FNX_STATIC_CACHE_GROUPS=2
export FNX_KV_CONTEXT_ONLY=1 FNX_Q8S_ROW_GROUP=2 FNX_MTP_EXPERTS=q4_0
export FNX_ATTN_HEAD_SPLIT=1 FNX_MTP_LAST_HEAD=1 FNX_GDN_REPLAY=1
export FNX_BATCH_DRAFTS=0 FNX_EXPERT_RAM_GIB=11.25 FNX_MTP_HEAD_Q4=1
bash "$out/run-sampling-experiment.sh" "$TP_EXE" "$out/sampling-q8s-ab-v26" 1 --mtp build-tensor-parralize-baseline/mtp-rt-q4 --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/sampling-q8s-ab-v26-console.txt 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path
old=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
report=[]
for new,ref in [('near-context','near-context-mtp-last-head'),('heldout','heldout-mtp-last-head'),('sampling','sampling-mtp-last-head')]:
    tag=f'{new}-q8s-ab-v26'
    a=json.loads((out/tag/'results.json').read_text())
    b=json.loads((old/ref/'results.json').read_text())
    assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs']), new
    for x,y in zip(a['runs'],b['runs']):
        for k in ['case','rep','ids','input_ids','finish']:
            assert x[k]==y[k], (new,x['case'],k)
        assert x['decode_faults_after_first_token']['major']==0
    if new=='sampling':
        assert a['sampling']==b['sampling']
        assert all(x[k]==y[k] for x,y in zip(a['runs'],b['runs']) for k in ('accepted','offered'))
    report.append(dict(candidate=tag,reference=ref,all_ids_exact=True,draft_acceptance_exact=all(x[k]==y[k] for x,y in zip(a['runs'],b['runs']) for k in ('accepted','offered'))))
a=json.loads((out/'churn-q8s-ab-v26/results.json').read_text())
b=json.loads((out/'heldout-q8s-ab-v26/results.json').read_text())
assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs'])
for x,y in zip(a['runs'],b['runs']):
    assert x['input_ids']==y['input_ids'] and x['ids']==y['ids'][:len(x['ids'])], x['case']
    assert x['decode_faults_after_first_token']['major']==0
p=json.loads((out/'probe-q8s-ab-v26/results.json').read_text())
assert p['status']=='complete' and p['engine_returncode']==0 and len(p['runs'])==len(b['runs'])
for x,y in zip(p['runs'],b['runs']):
    assert x['input_ids']==y['input_ids'] and x['ids']==y['ids'][:len(x['ids'])], ('probe',x['case'])
for prefix in ['quality','near-context','heldout','churn','probe','sampling']:
    manifest=json.loads((out/f'{prefix}-q8s-ab-v26/results.json').read_text())
    assert manifest['status']=='complete' and manifest['engine_returncode']==0
    assert all(r['decode_faults_after_first_token']['major']==0 for r in manifest['runs']),prefix
    assert manifest['env']['FNX_Q8S_FUSED_AB']=='1' and manifest['env']['FNX_PF_ROWS']=='256'
    assert manifest['env']['TP_CPU_AFFINITY']=='0-3' and manifest['env']['HSA_OVERRIDE_CPU_AFFINITY_DEBUG']=='0'
    for key in ('cpu_affinity_after_load','cpu_affinity_after_decode'):
        assert manifest[key]['main']==[0,1,2,3], (prefix,key)
        assert manifest[key]['threads'] and all(mask==[0,1,2,3] for mask in manifest[key]['threads'].values()), (prefix,key)
q=json.loads((out/'quality-q8s-ab-v26/quality.json').read_text())
assert q['passed'] and sum(x['checks_passed'] for x in q['cases'])==1326
report=dict(comparisons=report,churn_output_prefixes_exact=True,probe_output_prefixes_exact=True,reviewed_code_checks=1326,all_cpu_affinity_masks_verified=True)
(out/'q8s-ab-v26-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY

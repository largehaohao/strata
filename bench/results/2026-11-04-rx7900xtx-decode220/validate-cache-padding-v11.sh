#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
old=bench/results/2026-11-03-rx7900xtx-tensor-parallel
out=bench/results/2026-11-04-rx7900xtx-decode220
runner=$out/run-heads-experiment.sh
export TP_EXE=build-tensor-parralize-baseline/strata-split-cache-padding-v11
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1
export FNX_PACKED_Q2_DOWN=1 TP_EXPERT_RAM_GIB=11.625 FNX_STATIC_CACHE_MIN_PADDING=1

bash "$runner" "$out/quality-cache-padding-v11" --cases "$old/quality-prompts.json" --repeats 1 > build-tensor-parralize-baseline/quality-cache-padding-v11-console.txt 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
old=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
a=json.loads((out/'quality-cache-padding-v11/results.json').read_text())
b=json.loads((old/'quality-split/results.json').read_text())
assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs'])
for x,y in zip(a['runs'],b['runs']):
    for k in ['case','rep','ids','input_ids','finish']:
        assert x[k]==y[k], (x['case'],k)
print('Quality input/output IDs exactly match reviewed reference.')
PY
.venv/bin/python tools/tp_code_quality.py "$out/quality-cache-padding-v11"
bash "$runner" "$out/near-context-cache-padding-v11" --cases "$old/near-context-prompts.json" --prefix-file "$old/near-context-prefix.txt" --tokens 64 --repeats 1 --timeout 1800 > build-tensor-parralize-baseline/near-context-cache-padding-v11-console.txt 2>&1
bash "$runner" "$out/heldout-cache-padding-v11" --cases "$old/mtp-heldout-prompts.json" --tokens 1024 --repeats 1 > build-tensor-parralize-baseline/heldout-cache-padding-v11-console.txt 2>&1
FNX_ARENA_CHURN=1 bash "$runner" "$out/churn-cache-padding-v11" --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/churn-cache-padding-v11-console.txt 2>&1

FNX_PREFETCH_PROBE=1 bash "$runner" "$out/probe-cache-padding-v11" --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/probe-cache-padding-v11-console.txt 2>&1

# The sampling harness uses the same model and fixed seed/penalties as the
# previously checked baseline. Export settings normally owned by the runner.
export FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=16 FNX_STATIC_CACHE_GROUPS=2
export FNX_KV_CONTEXT_ONLY=1 FNX_Q8S_ROW_GROUP=2 FNX_MTP_EXPERTS=q4_0
export FNX_ATTN_HEAD_SPLIT=1 FNX_MTP_LAST_HEAD=1 FNX_GDN_REPLAY=1
export FNX_BATCH_DRAFTS=0 FNX_EXPERT_RAM_GIB=11.625 FNX_MTP_HEAD_Q4=1
bash "$out/run-sampling-experiment.sh" "$TP_EXE" "$out/sampling-cache-padding-v11" 1 --mtp build-tensor-parralize-baseline/mtp-rt-q4 --cases "$old/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/sampling-cache-padding-v11-console.txt 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path
old=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
out=Path('bench/results/2026-11-04-rx7900xtx-decode220')
report=[]
for new,ref in [('near-context','near-context-mtp-last-head'),('heldout','heldout-mtp-last-head'),('sampling','sampling-mtp-last-head')]:
    a=json.loads((out/f'{new}-cache-padding-v11/results.json').read_text())
    b=json.loads((old/ref/'results.json').read_text())
    assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs']), new
    for x,y in zip(a['runs'],b['runs']):
        for k in ['case','rep','ids','input_ids','finish','accepted','offered']:
            assert x[k]==y[k], (new,x['case'],k)
        assert x['decode_faults_after_first_token']['major']==0
    if new=='sampling': assert a['sampling']==b['sampling']
    report.append(dict(candidate=f'{new}-cache-padding-v11',reference=ref,all_ids_and_draft_acceptance_exact=True))
a=json.loads((out/'churn-cache-padding-v11/results.json').read_text())
b=json.loads((out/'heldout-cache-padding-v11/results.json').read_text())
assert a['status']=='complete' and a['engine_returncode']==0 and len(a['runs'])==len(b['runs'])
for x,y in zip(a['runs'],b['runs']):
    assert x['input_ids']==y['input_ids'] and x['ids']==y['ids'][:len(x['ids'])], x['case']
    assert x['decode_faults_after_first_token']['major']==0
p=json.loads((out/'probe-cache-padding-v11/results.json').read_text())
assert p['status']=='complete' and p['engine_returncode']==0 and len(p['runs'])==len(b['runs'])
for x,y in zip(p['runs'],b['runs']):
    assert x['input_ids']==y['input_ids'] and x['ids']==y['ids'][:len(x['ids'])], ('probe',x['case'])
q=json.loads((out/'quality-cache-padding-v11/quality.json').read_text())
assert q['passed'] and sum(x['checks_passed'] for x in q['cases'])==1326
report=dict(comparisons=report,churn_output_prefixes_exact=True,probe_output_prefixes_exact=True,reviewed_code_checks=1326)
(out/'cache-padding-v11-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY

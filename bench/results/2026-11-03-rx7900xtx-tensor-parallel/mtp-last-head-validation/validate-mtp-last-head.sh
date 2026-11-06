#!/bin/bash
set -eu
cd /home/john/Strata
export FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=16 FNX_STATIC_CACHE_GROUPS=2
export FNX_KV_CONTEXT_ONLY=1 FNX_Q8S_ROW_GROUP=2 FNX_MTP_EXPERTS=q4_0
export FNX_MTP_LAST_HEAD=1
export FNX_GDN_REPLAY=1 FNX_BATCH_DRAFTS=0 FNX_EXPERT_RAM_GIB=13.875 FNX_MTP_HEAD_Q4=1
root=bench/results/2026-11-03-rx7900xtx-tensor-parallel
runner=build-tensor-parralize-baseline/run-tp-bench.sh
exe=build-tensor-parralize-baseline/strata-split-mtp-last-head
mtp=build-tensor-parralize-baseline/mtp-rt-q4
bash "$runner" "$exe" "$root/bench-mtp-last-head-fused16-repeat" 1 --mtp "$mtp" > build-tensor-parralize-baseline/bench-mtp-last-head-fused16-repeat-console.txt 2>&1
bash "$runner" "$exe" "$root/quality-mtp-last-head" 1 --mtp "$mtp" --cases "$root/quality-prompts.json" --repeats 1 > build-tensor-parralize-baseline/quality-mtp-last-head-console.txt 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
a=json.loads((p/'quality-mtp-last-head/results.json').read_text())
b=json.loads((p/'quality-split/results.json').read_text())
assert a['status']=='complete' and len(a['runs'])==len(b['runs'])
for x,y in zip(a['runs'],b['runs']):
    for k in ['case','rep','ids','input_ids','finish']:
        assert x[k]==y[k], (x['case'], k)
print('Quality input/output IDs exactly match the reviewed reference.')
PY
.venv/bin/python tools/tp_code_quality.py "$root/quality-mtp-last-head"
bash "$runner" "$exe" "$root/near-context-mtp-last-head" 1 --mtp "$mtp" --cases "$root/near-context-prompts.json" --prefix-file "$root/near-context-prefix.txt" --tokens 64 --repeats 1 --timeout 1800 > build-tensor-parralize-baseline/near-context-mtp-last-head-console.txt 2>&1
bash "$runner" "$exe" "$root/heldout-mtp-last-head" 1 --mtp "$mtp" --cases "$root/mtp-heldout-prompts.json" --tokens 1024 --repeats 1 > build-tensor-parralize-baseline/heldout-mtp-last-head-console.txt 2>&1
sample_runner=build-tensor-parralize-baseline/run-tp-sample.sh
FNX_MTP_LAST_HEAD=1 bash "$sample_runner" "$exe" "$root/sampling-mtp-last-head" 1 --mtp "$mtp" --cases "$root/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/sampling-mtp-last-head-console.txt 2>&1
FNX_MTP_LAST_HEAD=0 bash "$sample_runner" "$exe" "$root/sampling-mtp-last-head-control" 1 --mtp "$mtp" --cases "$root/mtp-heldout-prompts.json" --tokens 128 --repeats 1 > build-tensor-parralize-baseline/sampling-mtp-last-head-control-console.txt 2>&1

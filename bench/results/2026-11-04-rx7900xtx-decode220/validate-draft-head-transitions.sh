#!/usr/bin/env bash
set -euo pipefail
cd /home/john/Strata
out=bench/results/2026-11-04-rx7900xtx-decode220
export TP_EXE=build-tensor-parralize-baseline/strata-split-draft-head-v13
export FNX_STATIC_CACHE_ARENA=1 FNX_STATIC_CACHE_FUSED_GU=1 FNX_PACKED_Q2_DOWN=1 FNX_STATIC_CACHE_MIN_PADDING=1
export FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=16 FNX_STATIC_CACHE_GROUPS=2
export FNX_KV_CONTEXT_ONLY=1 FNX_Q8S_ROW_GROUP=2 FNX_MTP_EXPERTS=q4_0
export FNX_ATTN_HEAD_SPLIT=1 FNX_MTP_LAST_HEAD=1 FNX_GDN_REPLAY=1
export FNX_BATCH_DRAFTS=0 FNX_EXPERT_RAM_GIB=11.625 FNX_MTP_HEAD_Q4=1
for mode in 3 0; do
 export FNX_MTP_HEAD_MODE=$mode
 bash "$out/run-draft-mode-experiment.sh" "$TP_EXE" "$out/draft-head-v13-transitions-mode$mode" 1 --mtp build-tensor-parralize-baseline/mtp-rt-q4 --cases "$out/draft-mode-prompts.json" --tokens 64 --repeats 1 > "build-tensor-parralize-baseline/draft-head-v13-transitions-mode$mode-console.txt" 2>&1
done
.venv/bin/python "$out/compare_draft_panels.py" "$out/draft-head-v13-transitions-mode0" "$out/draft-head-v13-transitions-mode3" --out "$out/draft-head-v13-transitions-comparison.json"
.venv/bin/python - <<'CHECK'
import json
from pathlib import Path
p=Path('bench/results/2026-11-04-rx7900xtx-decode220')
a=json.loads((p/'draft-head-v13-transitions-mode0/results.json').read_text())
b=json.loads((p/'draft-head-v13-transitions-mode3/results.json').read_text())
for x,y in zip(a['runs'],b['runs']):
 assert x['sampling']==y['sampling']
 if x['sampling']['temperature'] or x['sampling']['penalty_repeat']!=1:
  assert x['accepted']==y['accepted'] and x['offered']==y['offered'],x['case']
print('PASS 7 alternating greedy, sampled, penalized requests; exact final IDs and full-score fallback draft counts')
CHECK

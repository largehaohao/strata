import json,re
from pathlib import Path
import numpy as np
root=Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
path=root/'draft-cost-calibration'
pattern=re.compile(r'draft-cost rows (\d+) count (\d+) verify_ms ([\d.]+) commit_ms ([\d.]+)')
blocks=[];pending=[];drafts=[]
for line in (path/'engine.log').read_text().splitlines():
    m=pattern.search(line)
    if m:
        row,n,verify,commit=m.groups();pending.append(dict(rows=int(row),count=int(n),verify_ms=float(verify),commit_ms=float(commit)))
    if 'sfnx: slot 0: prompt' in line:
        blocks.append(pending);pending=[]
    m=re.search(r'draft ([\d.]+) ms \(GPU [\d.]+, (\d+) calls\)',line)
    if m:drafts.append((float(m[1]),int(m[2])))
manifest=json.loads((path/'results.json').read_text())
assert manifest['status']=='complete' and len(blocks)==len(manifest['runs'])
points=[];request_fits=[]
for run,block in zip(manifest['runs'],blocks):
    if run['case']=='warmup':continue
    chosen=[q for q in block if 2<=q['rows']<=9 and q['count']>=10]
    points.extend(chosen)
    if len(chosen)>=2:
        X=np.array([[1,q['rows']-1] for q in chosen],dtype=float)
        y=np.array([(q['verify_ms']+q['commit_ms'])/q['count'] for q in chosen])
        w=np.sqrt([q['count'] for q in chosen]);fit=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
        request_fits.append(dict(case=run['case'],verify_base_ms=fit[0],per_draft_row_ms=fit[1],points=chosen))
X=np.array([[1,q['rows']-1]for q in points],dtype=float)
y=np.array([(q['verify_ms']+q['commit_ms'])/q['count']for q in points]);w=np.sqrt([q['count']for q in points])
fit=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
draft_ms=sum(q[0]for q in drafts[1:])/sum(q[1]for q in drafts[1:])
report=dict(scope='diagnostic calibration prompts; weighted affine fit to rows 2..9 with at least 10 calls per request; not acceptance throughput',
            verify_base_ms=fit[0],per_draft_row_ms=fit[1],mean_draft_call_ms=draft_ms,
            round_with_first_draft_ms=fit[0]+draft_ms,requests=request_fits)
(root/'draft-cost-fit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

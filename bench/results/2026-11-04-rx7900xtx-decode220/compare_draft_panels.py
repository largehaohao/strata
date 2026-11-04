#!/usr/bin/env python3
"""Compare a complete decode panel and exact generated token streams; draft proposals may differ."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--out', type=Path)
args = parser.parse_args()
ref = json.loads((args.reference / 'results.json').read_text())
new = json.loads((args.candidate / 'results.json').read_text())
for label, panel in [('reference', ref), ('candidate', new)]:
    assert panel['status'] == 'complete' and panel['engine_returncode'] == 0, label
assert len(ref['runs']) == len(new['runs'])
fields = ['case', 'rep', 'input_ids', 'ids', 'tokens', 'finish']
rows = []
for a, b in zip(ref['runs'], new['runs']):
    for field in fields:
        assert a[field] == b[field], (b['case'], b['rep'], field)
    assert a['decode_faults_after_first_token']['major'] == b['decode_faults_after_first_token']['major'] == 0
    if b['case'] == 'warmup':
        continue
    rows.append(dict(case=b['case'], rep=b['rep'], tokens=b['tokens'],
                     reference_tps=a['decode_tps'], candidate_tps=b['decode_tps'],
                     faults=b['decode_faults_after_first_token'],
                     reference_drafts=dict(accepted=a['accepted'], offered=a['offered']),
                     candidate_drafts=dict(accepted=b['accepted'], offered=b['offered'])))
measured = [r for r in new['runs'] if r['case'] != 'warmup']
rate = sum(r['tokens'] for r in measured) * 1000 / sum(r['decode_ms'] for r in measured)
assert abs(rate - new['weighted_decode_tps']) < 1e-8
report = dict(reference=str(args.reference), candidate=str(args.candidate),
              binary_sha256=new['binary_sha256'], all_input_and_output_streams_exact=True,
              draft_streams_exact=all(a[k]==b[k] for a,b in zip(ref["runs"],new["runs"]) for k in ("accepted","offered")),
              weighted_decode_tps=rate, reference_tps=ref['weighted_decode_tps'],
              relative_gain=rate / ref['weighted_decode_tps'] - 1,
              all_decode_major_faults_zero=all(r['faults']['major'] == 0 for r in rows), runs=rows)
result = json.dumps(report, indent=2) + '\n'
if args.out:
    args.out.write_text(result)
print(result, end='')

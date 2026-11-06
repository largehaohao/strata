"""Verify final single-request target, references, and sampled penalty behavior."""
import json
from pathlib import Path

p = Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
pairs = [
    ('bench-mtp-last-head', 'bench-mtp-head-q4-ram13p875', True),
    ('bench-mtp-last-head-control', 'bench-mtp-last-head', True),
    ('bench-mtp-last-head-fused16', 'bench-mtp-last-head', True),
    ('bench-mtp-last-head-repeat', 'bench-mtp-last-head', True),
    ('bench-mtp-last-head-fused16-repeat', 'bench-mtp-last-head-fused16', True),
    ('quality-mtp-last-head', 'quality-split', False),
    ('near-context-mtp-last-head', 'near-context-mtp-head-q4', True),
    ('heldout-mtp-last-head', 'heldout-mtp-head-q4', True),
    ('sampling-mtp-last-head', 'sampling-mtp-last-head-control', True),
]
report = []
for candidate, reference, check_drafts in pairs:
    x = json.loads((p / candidate / 'results.json').read_text())
    y = json.loads((p / reference / 'results.json').read_text())
    assert x['status'] == y['status'] == 'complete', candidate
    assert x['engine_returncode'] == y['engine_returncode'] == 0, candidate
    assert len(x['runs']) == len(y['runs']), candidate
    keys = ['case', 'rep', 'ids', 'input_ids', 'finish']
    if check_drafts:
        keys += ['accepted', 'offered']
    for a, b in zip(x['runs'], y['runs']):
        for key in keys:
            assert a[key] == b[key], (candidate, a['case'], key)
        assert a['decode_faults_after_first_token']['major'] == 0, (candidate, a['case'])
    report.append(dict(candidate=candidate, reference=reference,
                       all_input_output_ids_equal=True, all_finish_reasons_equal=True,
                       draft_acceptance_checked=check_drafts, decode_major_faults=0,
                       weighted_decode_tps=x['weighted_decode_tps'], runs=len(x['runs'])))
q = json.loads((p / 'quality-mtp-last-head/quality.json').read_text())
assert q['passed'] and sum(x['checks_passed'] for x in q['cases']) == 1326
sample = json.loads((p / 'sampling-mtp-last-head/results.json').read_text())
assert sample['sampling'] == dict(temperature=0.7, top_k=20, top_p=0.9,
    penalty_repeat=1.05, penalty_freq=0.05, penalty_present=0.03, penalty_last_n=128, seed=20261104)
report = dict(comparisons=report, reviewed_code_checks=1326,
    gpu_q4_product_checks=384, gpu_bitexact_last_row_checks=384,
    sampling=sample['sampling'],
    scope='Swift IQ3_S; two gfx1100 cards; one request; context 32768; warmup excluded from rate, included in equality checks',
    kernel_log='mtp-last-head-gpu-checks.txt')
(p / 'mtp-last-head-checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

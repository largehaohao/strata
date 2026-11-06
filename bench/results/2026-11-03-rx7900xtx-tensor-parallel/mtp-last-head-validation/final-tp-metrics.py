"""Assemble acceptance evidence without treating short quality tasks as rate tests."""
import hashlib
import json
from pathlib import Path

p = Path('bench/results/2026-11-03-rx7900xtx-tensor-parallel')
checks = json.loads((p / 'mtp-last-head-checks.json').read_text())
names = ['bench-mtp-last-head-fused16', 'bench-mtp-last-head-fused16-repeat']
expected_sha = 'b051993f8a7793706ee848959226d99876263aa0c9d98630e981d88b2db5da3e'
panels = []
all_runs = []
for name in names:
    d = json.loads((p / name / 'results.json').read_text())
    assert d['status'] == 'complete' and d['engine_returncode'] == 0
    assert d['binary_sha256'] == expected_sha
    assert d['env']['FNX_MTP_LAST_HEAD'] == '1'
    assert d['env']['FNX_STATIC_CACHE_FUSED_DN'] == '16'
    assert d['env']['FNX_EXPERT_RAM_GIB'] == '13.875'
    runs = [r for r in d['runs'] if r['case'] != 'warmup']
    assert len(runs) == 6
    assert {(r['case'], r['rep']) for r in runs} == {(c, rep) for c in ('code', 'prose', 'edit') for rep in (0, 1)}
    assert all(r['tokens'] == 2048 and r['finish'] == 'length' for r in runs)
    assert all(r['decode_faults_after_first_token']['major'] == 0 for r in runs)
    assert d['weighted_decode_tps'] >= 200
    all_runs.extend(runs)
    panels.append(dict(directory=name, weighted_decode_tps=d['weighted_decode_tps'],
                       total_tokens=sum(r['tokens'] for r in runs),
                       total_decode_ms=sum(r['decode_ms'] for r in runs),
                       runs=[{k:r[k] for k in ('case', 'rep', 'tokens', 'decode_ms', 'decode_tps')} for r in runs]))
rate = sum(r['tokens'] for r in all_runs) * 1000 / sum(r['decode_ms'] for r in all_runs)
report = dict(status='passed', target_tps=200, weighted_decode_tps=rate, panels=panels,
    definition='Total generated tokens / total engine decode time. Warmup excluded. Independent engine process per panel. No concurrent compile or profiler.',
    model='Installed Swift-1.5 IQ3_S; target weights unchanged',
    workload='One request at a time; greedy; thinking off; short code, prose and edit prompts; 2048 output tokens each; two passes per panel',
    context=32768, cards='2 x RX 7900 XTX gfx1100, 24 GiB each',
    binary_sha256=expected_sha,
    source_patch_sha256=hashlib.sha256((p / 'mtp-last-head-source.patch').read_bytes()).hexdigest(),
    source_base='a82d4e240dfd5ab4136ec96cba6ffff4dd1ba358',
    profile_sha256=hashlib.sha256((p / 'swift-decode-routes.bin').read_bytes()).hexdigest(),
    checks='mtp-last-head-checks.json',
    limitations=['This is a weighted panel result, not a 200 tokens/s minimum for every prompt.',
        'The configured 204800 context and concurrent serving are not validated by this recipe.',
        'Split and native engine arithmetic differ. Exact-output checks compare split variants.',
        'CUDA is not built or tested. New execution paths remain opt-in.'])
(p / 'acceptance-200.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

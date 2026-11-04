#!/usr/bin/env python3
"""Audit the fixed single-request 220-token/s panel and its quality evidence."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
sha = '83edcdb64b0b620c186b759dac207e458019a5eb7b358e5944b169e77a91cf45'
selected = 'q8s-ab-v26-mode1'
repeat = selected + '-repeat'
control = 'q8s-ab-v26-mode0'

def read(name):
    return json.loads((root / name).read_text())

def panel(name):
    p = read(name + '/results.json')
    assert p['status'] == 'complete' and p['engine_returncode'] == 0, name
    assert all(r['decode_faults_after_first_token']['major'] == 0 for r in p['runs']), name
    return p

def affinity(p):
    assert p['env']['TP_CPU_AFFINITY']=='0-3'
    assert p['env']['HSA_OVERRIDE_CPU_AFFINITY_DEBUG']=='0'
    for key in ('cpu_affinity_after_load','cpu_affinity_after_decode'):
        assert p[key]['main']==[0,1,2,3]
        assert p[key]['threads'] and all(mask==[0,1,2,3] for mask in p[key]['threads'].values())

ref = panel('draft-head-v13-mode3')
base = panel(control)
assert base['binary_sha256'] == sha and base['env']['FNX_Q8S_FUSED_AB'] == '0'
affinity(base)
runs = []
for name in (selected, repeat):
    p = panel(name)
    assert p['binary_sha256'] == sha
    affinity(p)
    assert p['cases'] == ref['cases'] and p['prefix_sha256'] == ref['prefix_sha256']
    assert len(p['runs']) == len(ref['runs']) == 7
    for a, b in zip(p['runs'], ref['runs']):
        for key in ('case', 'rep', 'input_ids', 'ids', 'tokens', 'finish', 'accepted', 'offered'):
            assert a[key] == b[key], (name, a['case'], key)
    measured = [r for r in p['runs'] if r['case'] != 'warmup']
    assert [(r['case'], r['rep']) for r in measured] == [(c, n) for n in range(2) for c in ('code', 'prose', 'edit')]
    assert all(r['tokens'] == 2048 and r['finish'] == 'length' for r in measured)
    rate = sum(r['tokens'] for r in measured) * 1000 / sum(r['decode_ms'] for r in measured)
    assert abs(rate - p['weighted_decode_tps']) < 1e-8 and rate >= 220, (name, rate)
    for flag, value in {'FNX_EXPERT_RAM_GIB': '11.25', 'FNX_ROUTER_BF16': '1', 'FNX_STATIC_CACHE_MAX_AGE': '2', 'FNX_MTP_HEAD_MODE': '3', 'FNX_PACKED_Q2_DOWN': '1', 'HIP_VISIBLE_DEVICES': '0,1', 'FNX_Q8S_FUSED_AB': '1', 'FNX_Q8S_ROW_GROUP': '2', 'FNX_PF_ROWS': '256'}.items():
        assert p['env'][flag] == value, (name, flag)
    for flag, value in {'--max-context': '32768', '--spec': '6', '--lookup': '15', '--conversation-cache-mib': '0'}.items():
        assert p['command'][p['command'].index(flag) + 1] == value
    if name == repeat:
        first = panel(selected)
        assert p['env'] == first['env'] and p['command'] == first['command']
        assert p['source'] == first['source']
        assert p['engine_pid'] != first['engine_pid'], 'Repeat must start a fresh engine'
    free = re.search(r'VRAM free (\d+) / (\d+) MiB', (root / name / 'engine.log').read_text())
    assert free is not None, name
    runs.append(dict(directory=name, weighted_decode_tps=rate, output_tokens=12288,
                     decode_ms=sum(r['decode_ms'] for r in measured),
                     loaded_free_vram_mib=[int(x) for x in free.groups()],
                     per_case=[dict(case=r['case'], rep=r['rep'], tokens=r['tokens'], decode_tps=r['decode_tps']) for r in measured]))

v = read('q8s-ab-v26-validation.json')
assert v['reviewed_code_checks'] == 1326 and v['all_cpu_affinity_masks_verified']
assert v['churn_output_prefixes_exact'] and v['probe_output_prefixes_exact']
assert all(c['all_ids_exact'] for c in v['comparisons'])
quality = []
old = root.parent / '2026-11-03-rx7900xtx-tensor-parallel'
references = {'quality': 'quality-split', 'near-context': 'near-context-mtp-last-head',
              'heldout': 'heldout-mtp-last-head', 'sampling': 'sampling-mtp-last-head'}
for prefix in ('quality', 'near-context', 'heldout', 'churn', 'probe', 'sampling'):
    name = prefix + '-q8s-ab-v26'
    p = panel(name)
    assert p['binary_sha256'] == sha
    affinity(p)
    assert p['env']['FNX_EXPERT_RAM_GIB'] == '11.25'
    assert p['env']['FNX_ROUTER_BF16'] == '1' and p['env']['FNX_STATIC_CACHE_MAX_AGE'] == '2'
    assert p['env']['FNX_Q8S_FUSED_AB'] == '1' and p['env']['FNX_PF_ROWS'] == '256'
    if prefix in references:
        reference = json.loads((old / references[prefix] / 'results.json').read_text())
        assert p['cases'] == reference['cases'] and p['prefix_sha256'] == reference['prefix_sha256']
        assert len(p['runs']) == len(reference['runs'])
        for a, b in zip(p['runs'], reference['runs']):
            for key in ('case', 'rep', 'input_ids', 'ids', 'tokens', 'finish'):
                assert a[key] == b[key], (prefix, a['case'], key)
        if prefix == 'sampling':
            assert p['sampling'] == reference['sampling']
            assert all(a[k] == b[k] for a, b in zip(p['runs'], reference['runs']) for k in ('accepted', 'offered'))
    else:
        reference = panel('heldout-q8s-ab-v26')
        assert len(p['runs']) == len(reference['runs'])
        for a, b in zip(p['runs'], reference['runs']):
            assert a['case'] == b['case'] and a['rep'] == b['rep']
            assert a['input_ids'] == b['input_ids'] and a['ids'] == b['ids'][:len(a['ids'])]
    quality.append(dict(directory=name, all_decode_major_faults_zero=True, engine_returncode=0))
q = read('quality-q8s-ab-v26/quality.json')
assert q['passed'] and sum(c['checks_passed'] for c in q['cases']) == 1326
assert 'PASS 576' in (root / 'router-bf16-v17-gpu-checks.txt').read_text()
assert 'PASS 65536 exact BF16 bit patterns, 65535 nonrepresentable values' in (root / 'router-bf16-v17-host-checks.txt').read_text()
assert '18432 exact direct/graph/policy checks passed (6912 with K/V lending)' in (root / 'cache-age-v16-checks.txt').read_text()
assert 'PASS 1152 exact fused Q8S/AB direct/graph/guard checks' in (root / 'q8s-ab-v26-checks.txt').read_text()
for name in ('q8s-ab-v26-comparison.json', 'q8s-ab-v26-repeat-comparison.json', 'q8s-ab-v26-v13-comparison.json', 'q8s-ab-v26-control-compatibility.json'):
    c = read(name)
    assert c['all_token_and_draft_streams_exact'] and c['all_decode_major_faults_zero']

report = dict(status='passed', target_tokens_per_second=220, binary_sha256=sha,
              model='Swift-1.5 IQ3_S', request_concurrency=1, context_allocation=32768,
              metric='Total generated tokens / engine DONE decode time; warmup and prefill excluded.',
              performance=runs, control_directory=control, control_tps=base['weighted_decode_tps'],
              combined_weighted_decode_tps=sum(r['output_tokens'] for r in runs) * 1000 / sum(r['decode_ms'] for r in runs),
              prior_checkpoint='draft-head-v13-mode3', input_output_and_draft_counts_exact=True,
              all_decode_major_faults_zero=True, quality=quality, quality_checks=1326,
              bf16_gpu_checks=576, bf16_host_exact_patterns=65536,
              bf16_host_nonrepresentable_rejections=65535, clock_policy_checks=18432,
              q8s_ab_fusion_checks=1152,
              loaded_free_vram_mib_per_card=min(x for r in runs for x in r['loaded_free_vram_mib']),
              process_cpu_mask=[0,1,2,3], all_engine_thread_masks_verified=True,
              q8s_ab_unfused_control_tps=base['weighted_decode_tps'],
              router_storage='Lossless BF16, exact original FP32 bits before arithmetic.',
              router_saved_mib_per_card=120.234375,
              limitations=['Recipe validated at context allocation 32768 on the recorded dual RX7900XTX host.',
                           '220 tokens/s is the six-request aggregate; individual prose requests are below 220.',
                           'The small margin above 220 is specific to the recorded host and run conditions.',
                           'Draft-head screening is approximate candidate selection; main-model verification remains enabled.',
                           'The 11.25 GiB RAM recipe leaves only about 34 MiB free VRAM per card at load.'])
(root / 'acceptance-220.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

#!/usr/bin/env python3
"""Alternating greedy/sampled/penalized draft graph integration check.

Native engine: --config CONFIG; split: --exe EXE --model GGUF --mtp DIR
--tokenizer DIR. Outputs are new directories with exact prompts, token IDs,
timings, binary/source fingerprints and logs. No server or config is changed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.strata_tokenizer import Tokenizer

CASES = {
    'code': 'Write a complete Python module implementing a thread-safe LRU cache with TTL expiration, an injectable clock, capacity validation, get, put, delete and clear methods. Include a unittest test suite covering expiration, eviction order, updates, missing keys, invalid capacity, and concurrent access. Explain the invariants briefly. Return executable code.',
    'prose': 'Explain how database transactions implement atomicity, consistency, isolation and durability. Compare write-ahead logging, snapshot isolation and serializable execution using concrete examples of concurrent bank transfers. Discuss crash recovery and common application mistakes. Write a detailed technical tutorial.',
    'edit': 'Refactor the following Python module to support TTL and thread safety while preserving all existing method signatures. Return the complete updated module and a comprehensive unittest suite.\n```python\nfrom collections import OrderedDict\nclass Cache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.data = OrderedDict()\n    def get(self, key):\n        if key not in self.data:\n            return None\n        self.data.move_to_end(key)\n        return self.data[key]\n    def put(self, key, value):\n        self.data[key] = value\n        self.data.move_to_end(key)\n        while len(self.data) > self.capacity:\n            self.data.popitem(last=False)\n```',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def process_faults(pid):
    """Linux counters for the launched process, excluding child processes."""
    try:
        # comm is parenthesized and can contain spaces or parentheses.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'minor': int(fields[7]), 'major': int(fields[9])}
    except (OSError, ValueError, IndexError):
        return None


def fault_delta(before, after):
    return {k: after[k] - before[k] for k in before} if before is not None and after is not None else None


def tokenizer(path):
    path = Path(path)
    meta = json.loads((path / 'tokenizer.json').read_text())
    vocab = json.loads((path / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return Tokenizer(tokens, (path / 'merges.txt').read_text().split('\n'),
                     json.loads((path / 'token_type.json').read_text()),
                     meta.get('pre') or 'qwen35', meta.get('special_ids'))


def source_info():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    try:
        new_files = git('ls-files', '--others', '--exclude-standard', 'src/split',
                        'tools/tp_decode_bench.py').decode().splitlines()
        return {'head': git('rev-parse', 'HEAD').decode().strip(),
                'diff_sha256': sha(git('diff', 'HEAD', '--binary')),
                'new_source_sha256': {p: sha((ROOT / p).read_bytes()) for p in new_files}}
    except subprocess.CalledProcessError:
        return {'head': None}


def command(a):
    env = dict(os.environ)
    if a.config:
        from serve.server import child_env, engine_args
        cfg = json.loads(a.config.read_text())
        exe = Path(a.exe or cfg['exe']).resolve()
        args = engine_args(cfg, str(exe), announce=False)
        env = child_env(cfg)
        td = a.tokenizer or cfg['tokenizer']
        cwd = cfg.get('cwd') or ROOT
    else:
        if not all((a.exe, a.model, a.mtp, a.tokenizer)):
            raise ValueError('split needs --exe, --model, --mtp and --tokenizer')
        exe = Path(a.exe).resolve()
        args = ['--native', str(a.model.resolve()), '--mtp', str(a.mtp.resolve()),
                '--draft-vocab', str((a.mtp / 'draft_vocab.bin').resolve()),
                '--spec', str(a.spec), '--lookup', str(a.lookup),
                '--conversation-cache-mib', '0']
        td, cwd = a.tokenizer, ROOT
    if a.context is not None:
        if '--max-context' in args:
            args[args.index('--max-context') + 1] = str(a.context)
        else:
            args += ['--max-context', str(a.context)]
    elif not a.config:
        args += ['--max-context', '32768']
    args += a.engine_arg
    if '--serve' not in args:
        args.insert(0, '--serve')
    return [str(exe), *args], env, cwd, tokenizer(td)


def measure(a):
    cmd, env, cwd, tok = command(a)
    cases = json.loads(a.cases.read_text()) if a.cases else CASES
    if not isinstance(cases, dict) or not cases or 'warmup' in cases or any(not isinstance(v, str) for v in cases.values()):
        raise ValueError('--cases must contain a nonempty JSON object of name: prompt; warmup is reserved')
    prefix = a.prefix_file.read_text() + '\n\n' if a.prefix_file else ''
    a.out.mkdir(parents=True, exist_ok=False)
    safe_env = {k: v for k, v in env.items()
                if (k.startswith(('FNX_', 'STRATA_')) or k in
                    ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HSA_FORCE_FINE_GRAIN_PCIE',
                     'LD_LIBRARY_PATH', 'HIP_FORCE_DEV_KERNARG'))
                and not any(word in k.upper() for word in ('API_KEY', 'PASSWORD', 'SECRET', 'ACCESS_TOKEN'))}
    manifest = {'status': 'running', 'command': cmd, 'env': safe_env, 'source': source_info(),
                'binary_sha256': sha(Path(cmd[0]).read_bytes()), 'cases': cases, 'sampling': 'Per-request options selected by case prefix and recorded in each run', 'test_harness_sha256': sha(Path(__file__).read_bytes()),
                'prefix_sha256': sha(prefix.encode()), 'runs': [],
                'fault_counter_scope': 'Linux counters of the launched process; wrapper child processes are excluded. '
                                       'Decode fault deltas start when the client receives the first token.',
                'metric': 'decode_tps = DONE generated / DONE decode_ms * 1000; '
                          'client_decode_tps excludes first token; warmup excluded from weighted rate'}

    def save():
        temp = a.out / 'results.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(a.out / 'results.json')

    save()
    q = queue.Queue()
    with (a.out / 'engine.log').open('w') as log, (a.out / 'protocol.log').open('w') as protocol:
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        except OSError as error:
            manifest.update(status='failed', error=str(error))
            save()
            raise
        manifest['engine_pid'] = proc.pid
        save()

        def reader():
            for line in proc.stdout:
                protocol.write(line)
                protocol.flush()
                q.put(line.strip())
            q.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        def line():
            value = q.get(timeout=a.timeout)
            if value is None:
                raise RuntimeError(f'engine exited: {proc.poll()}; see engine.log')
            if value.startswith('ERR'):
                raise RuntimeError(value)
            return value

        try:
            while not line().startswith('READY'):
                pass
            for rep in range(a.repeats):
                requests = ([('warmup', 'Say hello in one sentence.')] if rep == 0 else []) + list(cases.items())
                for case_index, (name, prompt) in enumerate(requests):
                    body = prompt if name == 'warmup' else prefix + prompt
                    rendered = '<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                    ids = tok.encode(rendered, parse_special=True)
                    cap = 16 if name == 'warmup' else a.tokens
                    first, output = None, []
                    faults_before, faults_first = process_faults(proc.pid), None
                    start = time.perf_counter()
                    sampled = name.startswith('sampled_')
                    penalties = name.startswith(('sampled_pen', 'greedy_pen'))
                    options = dict(temperature=0.7 if sampled else 0, top_k=20, top_p=0.9,
                                   penalty_repeat=1.05 if penalties else 1, penalty_freq=0.05 if penalties else 0,
                                   penalty_present=0.03 if penalties else 0, penalty_last_n=128, seed=20261104)
                    words = ' '.join(f'{k}={v}' for k,v in options.items())
                    proc.stdin.write(f'GEN {cap} {words} ' + ','.join(map(str, ids)) + '\n')
                    proc.stdin.flush()
                    while True:
                        msg = line()
                        if msg.startswith('T '):
                            if first is None:
                                first = time.perf_counter()
                                faults_first = process_faults(proc.pid)
                            output.append(int(msg.split()[1]))
                        elif msg.startswith('DONE '):
                            end = time.perf_counter()
                            fields = msg.split()
                            n, pn = map(int, fields[1:3])
                            pm, dm = map(float, fields[3:5])
                            if n != len(output) or n > cap or not all(math.isfinite(v) and v >= 0 for v in (pm, dm)):
                                raise RuntimeError('invalid DONE token count or timing')
                            if fields[5] not in ('stop', 'length'):
                                raise RuntimeError('unexpected finish: ' + fields[5])
                            result = {'case': name, 'rep': rep, 'sampling': options, 'prompt_tokens': pn, 'tokens': n,
                                      'prompt_ms': pm, 'decode_ms': dm, 'decode_tps': n * 1000 / dm if dm else 0,
                                      'wall_s': end - start, 'ttft_s': first - start if first else None,
                                      'client_decode_tps': max(0, n - 1) / (end - first) if first and end > first else 0,
                                      'finish': fields[5], 'accepted': int(fields[6]), 'offered': int(fields[7]),
                                      'ids': output, 'input_ids': ids, 'input_sha256': sha(rendered.encode())}
                            faults_end = process_faults(proc.pid)
                            result['request_faults'] = fault_delta(faults_before, faults_end)
                            result['decode_faults_after_first_token'] = fault_delta(faults_first, faults_end)
                            (a.out / f'output-{rep}-{case_index}.txt').write_text(tok.decode(output))
                            manifest['runs'].append(result)
                            save()
                            print(json.dumps({k: v for k, v in result.items() if k not in ('ids', 'input_ids')}), flush=True)
                            break
            measured = [r for r in manifest['runs'] if r['case'] != 'warmup']
            duration = sum(r['decode_ms'] for r in measured)
            manifest['weighted_decode_tps'] = sum(r['tokens'] for r in measured) * 1000 / duration if duration else 0
            manifest['status'] = 'complete'
        except BaseException as error:
            manifest.update(status='failed', error=str(error))
            raise
        finally:
            save()
            shutdown = 'quit'
            if proc.poll() is None:
                try:
                    proc.stdin.write('QUIT\n')
                    proc.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                shutdown = 'terminate'
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    shutdown = 'kill'
                    proc.kill()
                    proc.wait()
            thread.join(timeout=2)
            manifest.update(engine_returncode=proc.returncode, engine_shutdown=shutdown)
            save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='native server config; read only')
    p.add_argument('--exe')
    for flag in ('model', 'mtp', 'tokenizer', 'cases', 'prefix-file'):
        p.add_argument('--' + flag, type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context', type=int, help='explicit override; otherwise native config or split 32768')
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--spec', type=int, default=6, help='split MTP draft limit')
    p.add_argument('--lookup', type=int, default=15, help='split lookup draft limit')
    p.add_argument('--timeout', type=float, default=600, help='seconds of engine silence before failing')
    p.add_argument('--engine-arg', action='append', default=[], help='append one engine argument, e.g. --engine-arg=--stats')
    a = p.parse_args()
    if a.tokens < 1 or a.repeats < 1 or a.timeout <= 0 or (a.context is not None and a.context < 1):
        p.error('tokens, repeats, timeout and context must be positive')
    measure(a)


if __name__ == '__main__':
    main()

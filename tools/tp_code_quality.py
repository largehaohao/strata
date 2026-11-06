#!/usr/bin/env python3
"""Check the three fixed code-quality tasks produced by tp_decode_bench.

Only use on reviewed outputs for these tasks. Children have time/memory limits,
a temporary working directory, a small import allowlist and no inherited secrets.
These checks are not a security sandbox for arbitrary generated Python.
"""
import ast
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile


def validate(code):
    tree = ast.parse(code)
    imports = {'typing', 'collections', 'heapq', 'bisect', 'math', 'struct'}
    forbidden = {'open', 'exec', 'eval', 'compile', 'getattr', 'setattr', 'delattr',
                 'globals', 'locals', 'vars', '__import__', 'input', 'breakpoint'}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError('unexpected top-level executable statement')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(x.name not in imports for x in node.names):
            raise ValueError('unexpected import')
        if isinstance(node, ast.ImportFrom) and (node.module not in imports or node.level):
            raise ValueError('unexpected import')
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError('unexpected builtin')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise ValueError('unexpected reflection')
    return tree


def child(path, case):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    code = Path(path).read_text()
    scope = {}
    exec(compile(validate(code), str(path), 'exec'), scope)
    total = 0

    def check(ok, message):
        nonlocal total
        if not ok:
            raise AssertionError(message)
        total += 1

    def rejects(fn, *args):
        try:
            fn(*args)
        except ValueError:
            check(True, '')
            return
        raise AssertionError('expected ValueError: ' + repr(args))

    rng = random.Random(20261103)
    if case == 'merge_intervals':
        fn = scope['merge_intervals']
        cases = [[], [(1, 2), (3, 4)], [(1, 3), (3, 7)], [(0, 0)], [(5, 9), (-2, 5), (2, 3)]]
        for _ in range(100):
            pairs = []
            for _ in range(rng.randrange(20)):
                start = rng.randrange(-50, 50)
                pairs.append((start, start + rng.randrange(20)))
            cases.append(pairs)
        for pairs in cases:
            # Independent oracle: connected components of the overlap graph.
            unseen = set(range(len(pairs)))
            expected = []
            while unseen:
                group = {unseen.pop()}
                while True:
                    more = {j for j in unseen if any(max(pairs[i][0], pairs[j][0]) <= min(pairs[i][1], pairs[j][1]) for i in group)}
                    if not more:
                        break
                    group |= more
                    unseen -= more
                expected.append((min(pairs[i][0] for i in group), max(pairs[i][1] for i in group)))
            before = list(pairs)
            check(fn(pairs) == sorted(expected), 'interval union mismatch')
            check(pairs == before, 'input mutated')
            check(fn(iter(pairs)) == sorted(expected), 'iterator mismatch')
        rejects(fn, [(2, 1)])
    elif case == 'topological_sort':
        fn = scope['topological_sort']
        check(fn([], []) == [], 'empty graph')
        for _ in range(60):
            nodes = list(range(rng.randrange(1, 7)))
            rng.shuffle(nodes)
            edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if rng.random() < 0.4]
            expected = next(list(p) for p in itertools.permutations(sorted(nodes)) if all(p.index(a) < p.index(b) for a, b in edges))
            nn, ee = nodes + nodes[:1], edges + edges[:2]
            before = (nn[:], ee[:])
            check(fn(nn, ee) == expected, 'not the lexicographically smallest order')
            check((nn, ee) == before, 'graph inputs mutated')
        for nn, ee in [([1], [(1, 1)]), ([1, 2], [(1, 2), (2, 1)]), ([1], [(1, 2)]), ([], [(1, 2)])]:
            rejects(fn, nn, ee)
    elif case == 'varint':
        encode, decode = scope['encode_varint'], scope['decode_varint']
        values = {0, 1, 127, 128, 255, 2**64 - 1}
        for bit in range(64):
            values.update((max(0, 2**bit - 1), 2**bit, min(2**64 - 1, 2**bit + 1)))
        values.update(rng.getrandbits(64) for _ in range(100))
        for value in sorted(values):
            length = max(1, (value.bit_length() + 6) // 7)
            expected = bytes(((value >> (7 * i)) & 127) | (128 if i + 1 < length else 0) for i in range(length))
            encoded = encode(value)
            check(isinstance(encoded, bytes) and encoded == expected, 'encoding mismatch')
            check(decode(expected) == (value, len(expected)), 'decoding mismatch')
            check(decode(b'abc' + expected + b'xyz', 3) == (value, 3 + len(expected)), 'offset/trailing bytes mismatch')
        for value in (-1, 2**64, 1.0, '1', None):
            rejects(encode, value)
        for blob in (b'', b'\x80', b'\x80\x00', b'\x81\x00', b'\x80' * 10, b'\xff' * 9 + b'\x02', b'\x81' * 10 + b'\x00'):
            rejects(decode, blob)
        for offset in (-1, 1, 2):
            rejects(decode, b'\x00', offset)
    else:
        raise ValueError('unknown quality task')
    print(json.dumps({'case': case, 'checks_passed': total}))


def main(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'results.json').read_text())
    report = {'binary_sha256': manifest['binary_sha256'], 'cases': []}
    indices = {}
    for run in manifest['runs']:
        rep = run['rep']
        index = indices.get(rep, 0)
        indices[rep] = index + 1
        if run['case'] == 'warmup':
            continue
        item = {'case': run['case'], 'finish': run['finish'], 'input_sha256': run['input_sha256']}
        try:
            if run['finish'] != 'stop':
                raise ValueError('reply did not finish')
            text = (directory / f'output-{run["rep"]}-{index}.txt').read_text()
            blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
            if len(blocks) != 1:
                raise ValueError('expected one complete code block')
            source = blocks[0]
            validate(source)
            item['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='strata-quality-') as temp:
                path = Path(temp) / 'candidate.py'
                path.write_text(source)
                result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--child', str(path), run['case']],
                                        cwd=temp, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                item.update(json.loads(result.stdout), passed=True)
        except Exception as error:
            item.update(passed=False, error=str(error))
        report['cases'].append(item)
    expected = {'merge_intervals', 'topological_sort', 'varint'}
    report['passed'] = bool(report['cases']) and all(x['passed'] for x in report['cases']) and {
        x['case'] for x in report['cases']} == expected
    (directory / 'quality.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('directory', help='tp_decode_bench output directory with the three reviewed quality tasks')
        raise SystemExit(main(parser.parse_args().directory))

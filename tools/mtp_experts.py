"""Build a split-engine Q4_0 MTP runtime from the installed BF16 expert tensors.

Example: python tools/mtp_experts.py --src /path/to/mtp --out /new/path/rt-q4
The source directory contains mtp-manifest.json, tensors/, and the existing rt/.
The new directory reuses rt's dense weights and draft vocabulary with symlinks.
Select it with --mtp /new/path/rt-q4 and FNX_MTP_EXPERTS=q4_0. The target GGUF
is unchanged. This is an optional draft-quality experiment, not a default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo.resolve() / 'tools'))
    from mtp_pack import q4_0, gguf

    src = args.src.resolve()
    manifest = {x['name']: x for x in json.loads((src / 'mtp-manifest.json').read_text())}
    names = ('mtp.layers.0.mlp.experts.gate_up_proj', 'mtp.layers.0.mlp.experts.down_proj')
    shapes = ((512, 1280, 2560), (512, 2560, 640))
    arrays, inputs = [], []
    if args.out.exists():
        raise ValueError('output directory must be new')
    for name, shape in zip(names, shapes):
        item = manifest[name]
        path = (src / item['file']).resolve()
        if not path.is_relative_to(src) or item['dtype'] != 'BF16' or tuple(item['shape']) != shape:
            raise ValueError(f'invalid source tensor metadata: {name}')
        if path.stat().st_size != np.prod(shape) * 2 or path.stat().st_size != item['bytes']:
            raise ValueError(f'invalid source tensor size: {name}')
        sha = digest(path)
        if sha != item['sha256']:
            raise ValueError(f'source hash mismatch: {name}')
        arrays.append(np.memmap(path, dtype='<u2', mode='r', shape=shape))
        inputs.append({'name': name, 'path': str(path), 'shape': shape, 'sha256': sha})
        print('verified', name, sha, flush=True)

    rng = np.random.default_rng(20261104)
    samples = rng.normal(size=(1024, 32)).astype(np.float32)
    samples[0] = 0
    samples[1] = np.tile(np.array([-1, 1], dtype=np.float32), 16)
    samples[2] = -samples[1]
    reference = gguf.quants.quantize(samples, gguf.GGMLQuantizationType.Q4_0).reshape(-1)
    if not np.array_equal(q4_0(samples), reference):
        raise ValueError('Q4_0 quantizer differs from the pinned gguf reference')

    args.out.mkdir(parents=True)
    for name in ('dense.bin', 'dense.txt', 'draft_vocab.bin'):
        path = (src / 'rt' / name).resolve(strict=True)
        (args.out / name).symlink_to(path)
    target = args.out / 'experts-q4_0.bin'
    temporary = args.out / 'experts-q4_0.bin.partial'
    start = time.monotonic()
    count, h = 0, hashlib.sha256()
    checks = 1024
    with temporary.open('xb') as f:
        # The split loader reads complete gate, up, then down planes, each ordered by expert.
        for label, raw, part in (('gate', arrays[0], slice(0, 640)),
                                 ('up', arrays[0], slice(640, 1280)),
                                 ('down', arrays[1], slice(None))):
            for expert in range(512):
                x = (np.asarray(raw[expert, part]).astype(np.uint32) << 16).view(np.float32)
                if not np.isfinite(x).all():
                    raise ValueError(f'non-finite BF16 tensor: {label} {expert}')
                packed = q4_0(x)
                if expert % 16 == 0:
                    sample = x[:4]
                    ref = gguf.quants.quantize(sample, gguf.GGMLQuantizationType.Q4_0).reshape(-1)
                    if not np.array_equal(packed[:ref.size], ref):
                        raise ValueError(f'quantizer mismatch: {label} {expert}')
                    checks += sample.size // 32
                blob = packed.tobytes()
                f.write(blob); h.update(blob); count += len(blob)
                if expert % 64 == 0:
                    print(label, expert, count, 'bytes', flush=True)
    expected = 3 * 512 * 640 * 2560 // 32 * 18
    if count != expected or temporary.stat().st_size != expected:
        raise ValueError('packed size mismatch')
    temporary.rename(target)
    report = {'format': 'Q4_0', 'layout': 'gate, up, down; expert-major row-major blocks',
              'source_manifest_sha256': digest(src / 'mtp-manifest.json'), 'inputs': inputs,
              'output_bytes': count, 'output_sha256': h.hexdigest(),
              'quantizer': 'tools/mtp_pack.py:q4_0', 'quantizer_source_sha256': digest(args.repo / 'tools/mtp_pack.py'),
              'reference_blocks_checked': checks, 'elapsed_s': time.monotonic() - start,
              'runtime_files': {name: {'sha256': digest(src / 'rt' / name)}
                                for name in ('dense.bin', 'dense.txt', 'draft_vocab.bin')}}
    (args.out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

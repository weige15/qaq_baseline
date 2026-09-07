"""Run/audit the frozen CPU analysis. Never writes a closed result or reuses output.

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venv/bin/python \
    scripts/analyze_batch_interference.py --out results/batch-interference-v1/run-1
Use --audit results/batch-interference-v1/run-1 to rehash and recompute read-only.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from qaq.batch_analysis import (analyze, canonical, compact_summary, load_inputs,
                                sha256, verify_hashes)

RAW_ROOT = Path('results/batch-interference-v1')
SEAL = RAW_ROOT / 'protocol-freeze.json'
SOURCES = ('BATCH_INTERFERENCE_PROTOCOL.md', 'configs/batch_interference_protocol.json',
           'src/qaq/batch_analysis.py', 'scripts/analyze_batch_interference.py',
           'tests/test_batch_analysis.py')
OUTPUTS = ('analysis.json', 'summary.json')


def frozen_protocol(root):
    seal = json.loads((root / SEAL).read_text())
    if set(seal['files']) != set(SOURCES[:2]):
        raise ValueError('invalid freeze file inventory')
    verify_hashes(root, seal['files'])
    cfg = json.loads((root / SOURCES[1]).read_text())
    if cfg['protocol_id'] != seal['protocol_id']:
        raise ValueError('freeze protocol ID mismatch')
    return cfg


def output_path(root, name):
    out = (root / name).resolve()
    raw_root = (root / RAW_ROOT).resolve()
    if not out.is_relative_to(raw_root) or out == raw_root:
        raise ValueError('output must be a new subdirectory under results/batch-interference-v1')
    return out


def write_new(path, content):
    with path.open('xb') as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU-only analysis requires CUDA_VISIBLE_DEVICES='' explicitly")
    root = Path.cwd().resolve()
    cfg = frozen_protocol(root)
    out = output_path(root, args.out or args.audit)
    sources = {name: sha256(root / name) for name in SOURCES}
    if args.out and out.exists():
        raise FileExistsError(f'append-only output already exists: {out}')
    if args.audit:
        manifest = json.loads((out / 'manifest.json').read_text())
        if (manifest['source_sha256'] != sources or manifest['input_sha256'] != cfg['input_sha256']
                or manifest['freeze_sha256'] != sha256(root / SEAL)
                or set(manifest['output_sha256']) != set(OUTPUTS)):
            raise ValueError('run manifest does not match frozen inputs/current analysis source')
        expected_files = {'manifest.json', *OUTPUTS, *('source/' + name for name in SOURCES)}
        actual_files = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
        if actual_files != expected_files:
            raise ValueError('raw output inventory mismatch')
        verify_hashes(out / 'source', sources)
        verify_hashes(out, manifest['output_sha256'])
    counts, datasets = load_inputs(root, cfg)
    result = analyze(counts, datasets, cfg)
    data = {'analysis.json': canonical(result), 'summary.json': canonical(compact_summary(result))}
    # Verify inputs and sources have not changed during computation.
    verify_hashes(root, cfg['input_sha256'])
    verify_hashes(root, sources)
    if args.audit:
        for name, content in data.items():
            if (out / name).read_bytes() != content:
                raise ValueError(f'raw output does not exactly recompute: {name}')
        print(json.dumps({'passed': True, 'audit_directory': str(out.relative_to(root)),
                          'exact_recomputation': list(OUTPUTS), 'scenarios': len(result['scenarios']),
                          'input_hashes_verified': len(cfg['input_sha256']),
                          'source_snapshots_verified': len(SOURCES), 'decision': result['decision']}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        snapshot = out / 'source' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        write_new(snapshot, (root / name).read_bytes())
    for name, content in data.items():
        write_new(out / name, content)
    manifest = {'protocol_id': cfg['protocol_id'], 'source_sha256': sources,
                'input_sha256': cfg['input_sha256'], 'freeze_sha256': sha256(root / SEAL),
                'output_sha256': {name: sha256(out / name) for name in OUTPUTS},
                'command': {'argv': sys.argv, 'executable': sys.executable, 'cwd': str(root),
                            'python': sys.version, 'cuda_visible_devices': ''},
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], text=True).splitlines()}
    write_new(out / 'manifest.json', canonical(manifest))
    print(json.dumps({'out': str(out.relative_to(root)), 'output_sha256': manifest['output_sha256'],
                      'scenarios': len(result['scenarios']), 'decision': result['decision']}, indent=2))


if __name__ == '__main__':
    main()

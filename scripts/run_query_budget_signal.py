"""New fixed6 runner. Launch ONLY serially through the unchanged GPU preflight."""
import argparse
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qaq.evaluation import evaluate, summarize
from qaq.model import install_replacements, prepare_replacements, set_profile
from qaq.quantization import apply_fixed, reconstruct
from qaq.query_budget_signal import (CONFIG, ROOT, accounting, new_run, read_json,
    read_rows, require, sha256, validate_rows, verify_freeze, verify_hashes, write_new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = verify_freeze()
    if not args.smoke:
        projection = read_json(ROOT/'query_budget_signal_projection.json')
        require(projection['passed'] and projection['projected_job_seconds'] <= 1800 and
                projection['projected_total_seconds'] <= 21600, 'full run cost gate')
        require(projection['smoke_result_sha256'] == sha256(ROOT/'query_budget_signal_smoke/query_budget_signal_results.json'),
                'projection smoke identity')
    out = new_run(args.out)
    start = time.monotonic()
    # In-process deadline complements the external 1800s timeout, never enlarges it.
    signal.alarm(1800)
    try:
        require(bool(os.environ.get('CUDA_VISIBLE_DEVICES')) and
                ',' not in os.environ['CUDA_VISIBLE_DEVICES'], 'one preflight-selected GPU required')
        frozen = Path(cfg['core_frozen'])
        hashes = read_json(frozen/'freeze_hashes.json')
        verify_hashes(frozen, hashes)
        core = read_json(frozen/'protocol.json')
        env = read_json(frozen/'environment.json')
        versions = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'safetensors', 'tokenizers')}
        require(all(v == env['packages'][p] for p, v in versions.items()), 'frozen runtime mismatch')
        info = read_json(frozen/'model_manifest.json')
        require(info['revision'] == core['model']['revision'], 'model revision mismatch')
        verify_hashes(info['local_path'], info['file_sha256'])
        random.seed(core['evaluation']['seed'])
        np.random.seed(core['evaluation']['seed'])
        torch.manual_seed(core['evaluation']['seed'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        require(torch.cuda.device_count() == 1, 'one visible GPU required')
        command = {'argv': sys.argv, 'executable': sys.executable, 'cwd': os.getcwd(),
            'pid': os.getpid(), 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'started_unix': time.time(),
            'frozen_hashes': hashes, 'model_files_verified': info['file_sha256'], 'versions': versions,
            'phase_a_freeze_sha256': sha256(ROOT/'query_budget_signal_phase_a_freeze.json'),
            'runner_sha256': sha256(__file__), 'config_sha256': sha256(CONFIG), 'smoke_only': args.smoke,
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'deterministic': True, 'seed': 1729, 'threads': 4, 'allow_tf32': False,
            'hardware': {'gpu': torch.cuda.get_device_name(0), 'properties': str(torch.cuda.get_device_properties(0)),
                         'cuda_runtime': torch.version.cuda}}
        write_new(out/'query_budget_signal_command.json', command)
        examples = read_rows(frozen/'examples.jsonl')
        if args.smoke:
            selected = set(map(tuple, cfg['smoke_ids']))
            examples = [e for e in examples if (e['task'], e['index']) in selected]
            require([[e['task'], e['index']] for e in examples] == cfg['smoke_ids'], 'smoke identity')
        model = AutoModelForCausalLM.from_pretrained(info['local_path'], dtype=torch.float16,
            attn_implementation=core['model']['attention'], local_files_only=True, trust_remote_code=False).eval().to('cuda:0')
        model.requires_grad_(False)
        require(sum(p.numel() for p in model.parameters()) == cfg['quantized_parameters'] + cfg['excluded_fp16_parameters'],
                'full model parameter count')
        replacements = prepare_replacements(model, 128) if args.smoke else None
        modules = apply_fixed(model, 6, core['quantization']['group_size'])
        budget = accounting(modules, cfg)
        write_new(out/'query_budget_signal_modules.json', modules)
        tensor_checks = []
        if args.smoke:
            for parent, name, packed, block in replacements:
                equal = torch.equal(getattr(parent, name).weight, reconstruct(packed.q, packed.scale, 6))
                tensor_checks.append({'block': block, 'projection': name, 'exact': equal, 'parameters': packed.q.numel()})
            require(len(tensor_checks) == 252 and all(c['exact'] for c in tensor_checks), 'independent/nested tensor mismatch')
        torch.cuda.synchronize()
        setup_seconds = time.monotonic()-start
        task_seconds = {}
        rows = []
        for task in ('wikitext2', 'hellaswag', 'arc_challenge'):
            exs = [e for e in examples if e['task'] == task]
            t0 = time.monotonic()
            path = out/f'query_budget_signal_{task}_samples.jsonl'
            evaluate(model, exs, path)
            torch.cuda.synchronize()
            task_seconds[task] = time.monotonic()-t0
            rows.extend(read_rows(path))
        with (out/'query_budget_signal_samples.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        validate_rows(rows, examples, 'new fixed6 raw rows')
        if args.smoke:
            install_replacements(replacements)
            set_profile(model, [6]*72)
            path = out/'query_budget_signal_nested_samples.jsonl'
            evaluate(model, examples, path)
            torch.cuda.synchronize()
            require(path.read_bytes() == (out/'query_budget_signal_samples.jsonl').read_bytes(),
                    'smoke independent/nested raw mismatch')
            write_new(out/'query_budget_signal_integration_gate.json', {'passed': True,
                'tensor_checks': tensor_checks, 'smoke_ids': cfg['smoke_ids'], 'raw_samples_byte_identical': True,
                'fixed6_sample_sha256': sha256(out/'query_budget_signal_samples.jsonl'),
                'nested_sample_sha256': sha256(path), 'accounting': budget})
        write_new(out/'query_budget_signal_results.json', {'mode': 'fixed6', 'smoke_only': args.smoke,
            'metrics': summarize(rows), 'sample_count': len(rows), 'accounting': budget,
            'setup_seconds': setup_seconds, 'task_eval_seconds': task_seconds,
            'eval_seconds': sum(task_seconds.values()), 'total_seconds': time.monotonic()-start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        print('FIXED6 JOB COMPLETE', out, flush=True)
    except Exception:
        with (out/'query_budget_signal_failure.txt').open('x') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()

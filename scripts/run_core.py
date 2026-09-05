"""One baseline GPU job; launch only through scripts/gpu_preflight.sh."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qaq.evaluation import evaluate, sha256, write_json
from qaq.quantization import apply_fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="results/core-v1/frozen")
    parser.add_argument("--mode", required=True, choices=["fp16", "fixed4", "fixed8"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("one preflight-selected CUDA_VISIBLE_DEVICES required")
        frozen = Path(args.frozen)
        hashes = json.loads((frozen / "freeze_hashes.json").read_text())
        for name, digest in hashes.items():
            if sha256(frozen / name) != digest:
                raise RuntimeError(f"frozen input changed: {name}")
        config = json.loads((frozen / "protocol.json").read_text())
        env = json.loads((frozen / "environment.json").read_text())
        for package in ("torch", "transformers", "numpy", "safetensors", "tokenizers"):
            if importlib.metadata.version(package) != env["packages"][package]:
                raise RuntimeError(f"frozen runtime changed: {package}")
        write_json(out / "command.json", {"argv": sys.argv, "executable": sys.executable,
            "cwd": os.getcwd(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "frozen_hashes": hashes, "smoke_only": args.smoke, "started_unix": started})
        # Keep runnable sources, not just a git hash that omits working-tree changes.
        for directory in ("src", "scripts", "configs", "tests"):
            shutil.copytree(directory, out / "source" / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (out / "git-diff.patch").write_text(subprocess.run(["git", "diff"], text=True,
                                                         capture_output=True, check=True).stdout)
        (out / "git-head.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                                                       capture_output=True, check=True).stdout)
        random.seed(config["evaluation"]["seed"])
        np.random.seed(config["evaluation"]["seed"])
        torch.manual_seed(config["evaluation"]["seed"])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one visible GPU")
        write_json(out / "hardware.json", {"gpu": torch.cuda.get_device_name(0),
            "properties": str(torch.cuda.get_device_properties(0)),
            "cuda_runtime": torch.version.cuda})
        model_info = json.loads((frozen / "model_manifest.json").read_text())
        model = AutoModelForCausalLM.from_pretrained(model_info["local_path"],
            dtype=torch.float16, attn_implementation=config["model"]["attention"],
            local_files_only=True, trust_remote_code=False).eval().to("cuda:0")
        model.requires_grad_(False)
        if args.mode != "fp16":
            manifest = apply_fixed(model, int(args.mode.removeprefix("fixed")),
                                   config["quantization"]["group_size"])
            write_json(out / "quantized_modules.json", manifest)
        examples = [json.loads(s) for s in (frozen / "examples.jsonl").read_text().splitlines()]
        if args.smoke:
            counts = Counter()
            selected = []
            for ex in examples:
                if counts[ex["task"]] < config["gates"]["smoke_examples_per_task"]:
                    selected.append(ex)
                    counts[ex["task"]] += 1
            examples = selected
        torch.cuda.synchronize()
        evaluation_start = time.time()
        metrics = evaluate(model, examples, out / "samples.jsonl")
        torch.cuda.synchronize()
        write_json(out / "results.json", {"mode": args.mode, "smoke_only": args.smoke,
            "metrics": metrics, "sample_count": len(examples),
            "eval_seconds": time.time()-evaluation_start, "total_seconds": time.time()-started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

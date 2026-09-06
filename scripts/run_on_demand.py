"""Run one isolated packed-weight storage measurement through GPU preflight."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from qaq.evaluation import evaluate, sha256, write_json
from qaq.model import load_quantized
from qaq.on_demand import (PackedLinear, convert_nested_model, manifest_sha256,
                           study_source_manifest)

ROOT = Path("results/on-demand-v1")
PROTOCOL_PATH = Path("configs/on_demand_protocol.json")
CORE = Path("results/core-v1")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def canonical_profiles(routes: list[dict]) -> list[dict]:
    return [{"task": row["task"], "index": row["index"], "profile": row["profile"]}
            for row in routes]


def canonical_sha(rows: list[dict]) -> str:
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def gpu_model_inventory(model) -> dict:
    parameters = [(name, tensor.numel() * tensor.element_size())
                  for name, tensor in model.named_parameters() if tensor.device.type == "cuda"]
    buffers = [(name, tensor.numel() * tensor.element_size())
               for name, tensor in model.named_buffers() if tensor.device.type == "cuda"]
    return {
        "parameter_bytes": sum(size for _, size in parameters),
        "buffer_bytes": sum(size for _, size in buffers),
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
        "buffer_names": [name for name, _ in buffers],
    }


class SavedRouteSession:
    def __init__(self, examples: list[dict], profiles: list[list[int]], storage):
        self.examples = examples
        self.profiles = profiles
        self.storage = storage
        self.index = 0

    def __call__(self, context_tokens: list[int]) -> list[int]:
        if self.index >= len(self.examples):
            raise RuntimeError("more route requests than frozen examples")
        example = self.examples[self.index]
        expected = example["tokens"][:example["prefix_length"]]
        if context_tokens != expected:
            raise RuntimeError("scorer route context differs from frozen WikiText prefix")
        profile = self.profiles[self.index]
        self.storage.set_profile(profile)
        self.index += 1
        return profile


def verify_inputs(protocol: dict) -> tuple[list[dict], list[dict], dict]:
    inherited = protocol["inherits"]
    paths = {
        "checkpoint": Path(inherited["checkpoint"]),
        "examples": Path(inherited["examples"]),
        "routes_repeat1": Path(inherited["routes_repeat1"]),
        "routes_repeat2": Path(inherited["routes_repeat2"]),
        "core_samples": Path(inherited["core_samples"]),
        "integration_gate": Path(inherited["integration_gate"]),
        "comparison_gate": Path(inherited["comparison_gate"]),
        "router_checkpoint": Path(inherited["router_checkpoint"]),
        "router_selection": Path(inherited["router_selection"]),
    }
    for key, path in paths.items():
        if sha256(path) != inherited[f"{key}_sha256"]:
            raise RuntimeError(f"protected input changed: {path}")
    for key, path in (("core_quantization_source", Path("src/qaq/quantization.py")),
                      ("core_model_source", Path("src/qaq/model.py")),
                      ("core_router_source", Path("src/qaq/router.py"))):
        if sha256(path) != inherited[f"{key}_sha256"]:
            raise RuntimeError(f"protected core source changed: {path}")
    if not json.loads(paths["integration_gate"].read_text())["passed"]:
        raise RuntimeError("completed integration gate is not passing")
    if not json.loads(paths["comparison_gate"].read_text())["evidence_checks_passed"]:
        raise RuntimeError("completed comparison evidence checks are not passing")
    selection = json.loads(paths["router_selection"].read_text())
    if not selection["passed"] or selection["router_sha256"] != inherited["router_checkpoint_sha256"]:
        raise RuntimeError("completed A1 router selection changed")
    examples = jsonl(paths["examples"])
    routes1 = json.loads(paths["routes_repeat1"].read_text())
    routes2 = json.loads(paths["routes_repeat2"].read_text())
    canonical1, canonical2 = canonical_profiles(routes1), canonical_profiles(routes2)
    if canonical1 != canonical2:
        raise RuntimeError("saved adaptive task/index/profiles differ between core repeats")
    if canonical_sha(canonical1) != inherited["canonical_task_index_profile_sha256"]:
        raise RuntimeError("canonical saved adaptive profiles changed")
    if [(r["task"], r["index"]) for r in canonical1] != [
            (e["task"], e["index"]) for e in examples]:
        raise RuntimeError("saved profiles do not align with frozen examples")
    return examples, canonical1, {key: sha256(path) for key, path in paths.items()}


def write_trace(path: Path, events: list[dict]) -> None:
    with open(path, "x") as stream:
        for event in events:
            stream.write(json.dumps(event, allow_nan=False) + "\n")


def main() -> None:
    process_started = time.perf_counter()
    wall_started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["resident", "ondemand_sync"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-gate", default="results/on-demand-v1/cpu-gate/results.json")
    parser.add_argument("--smoke-gate", default="results/on-demand-v1/smoke-gate.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        if (not os.environ.get("CUDA_VISIBLE_DEVICES")
                or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1
                or torch.cuda.device_count() != 1):
            raise RuntimeError("one preflight-selected CUDA_VISIBLE_DEVICES entry required")
        protocol = json.loads(PROTOCOL_PATH.read_text())
        if protocol["protocol_id"] != "qaq-on-demand-v1":
            raise RuntimeError("unexpected protocol")
        source_manifest = study_source_manifest()
        source_manifest_hash = manifest_sha256(source_manifest)
        cpu_gate = json.loads(Path(args.cpu_gate).read_text())
        cpu_manifest = json.loads((Path(args.cpu_gate).parent / "source_manifest.json").read_text())
        if (not cpu_gate.get("passed") or cpu_gate["source_manifest_sha256"] != source_manifest_hash
                or cpu_manifest != source_manifest):
            raise RuntimeError("passing CPU/tiny-Qwen gate for the identical study source is required")
        if not args.smoke:
            gate = json.loads(Path(args.smoke_gate).read_text())
            if (not gate.get("passed") or not gate.get("repeat_authorized")
                    or gate.get("source_manifest_sha256") != source_manifest_hash):
                raise RuntimeError("passing smoke gate for the identical source is required")
        examples, routes, input_hashes = verify_inputs(protocol)
        selected = [(example, route) for example, route in zip(examples, routes)
                    if example["task"] == "wikitext2"]
        if len(selected) != protocol["evaluation"]["repeat_examples"]:
            raise RuntimeError("unexpected frozen WikiText example count")
        if args.smoke:
            selected = selected[:protocol["evaluation"]["smoke_examples"]]
        selected_examples = [example for example, _ in selected]
        selected_routes = [route["profile"] for _, route in selected]
        if any(len(profile) != 72 or any(bits not in (4, 6, 8) for bits in profile)
               for profile in selected_routes):
            raise RuntimeError("invalid saved precision profile")

        environment = json.loads((CORE / "frozen/environment.json").read_text())
        package_names = ("torch", "transformers", "numpy", "safetensors",
                         "tokenizers", "scipy", "pyarrow")
        versions = {name: importlib.metadata.version(name) for name in package_names}
        for name, version in versions.items():
            if version != environment["packages"][name]:
                raise RuntimeError(f"frozen runtime changed: {name}")

        command = {
            "argv": sys.argv,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "args": vars(args),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "input_hashes": input_hashes,
            "canonical_profiles_sha256": canonical_sha(routes),
            "selected_profiles_sha256": canonical_sha([
                {"task": e["task"], "index": e["index"], "profile": p}
                for e, p in zip(selected_examples, selected_routes)]),
            "versions": versions,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True).splitlines(),
            "started_unix": wall_started,
            "run_uuid": str(uuid.uuid4()),
            "source_manifest_sha256": source_manifest_hash,
            "cpu_gate": args.cpu_gate,
            "smoke_gate": args.smoke_gate,
        }
        write_json(out / "command.json", command)
        write_json(out / "source_manifest.json", source_manifest)
        write_json(out / "profiles.json", [
            {"task": e["task"], "index": e["index"], "profile": p}
            for e, p in zip(selected_examples, selected_routes)])
        for directory in ("src", "scripts", "configs", "tests"):
            shutil.copytree(directory, out / "source" / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2("ON_DEMAND_PROTOCOL.md", out / "source" / "ON_DEMAND_PROTOCOL.md")
        if study_source_manifest(out / "source") != source_manifest:
            raise RuntimeError("runnable source snapshot differs from executed source")
        (out / "git-diff.patch").write_text(subprocess.check_output(
            ["git", "diff"], text=True))

        torch.set_num_threads(4)
        torch.manual_seed(protocol["evaluation"]["deterministic_seed"])
        np.random.seed(protocol["evaluation"]["deterministic_seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        hardware = {
            "gpu": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "cuda_runtime": torch.version.cuda,
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        }
        write_json(out / "hardware.json", hardware)

        phases: dict[str, float] = {
            "integrity_and_snapshot_seconds": time.perf_counter() - process_started,
        }
        phase = time.perf_counter()
        model, metadata = load_quantized(protocol["inherits"]["checkpoint"], "cpu")
        phases["checkpoint_cpu_load_seconds"] = time.perf_counter() - phase
        if metadata["group_size"] != 128:
            raise RuntimeError("core group size changed")

        phase = time.perf_counter()
        storage = convert_nested_model(model, args.mode)
        gc.collect()
        phases["packing_and_original_buffer_removal_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        model.to(device)
        torch.cuda.synchronize(device)
        phases["model_remainder_h2d_seconds"] = time.perf_counter() - phase

        storage_setup = storage.prepare(device)
        phases["storage_prepare_seconds"] = storage_setup["seconds"]
        inventory = storage.inventory()
        if args.mode == "resident":
            if (inventory["source_gpu_bytes"] != inventory["source_storage_bytes"]
                    or inventory["slot_capacity_bytes"] != 0):
                raise RuntimeError("resident source placement is invalid")
        elif (inventory["source_gpu_bytes"] != 0
              or inventory["source_devices"] != ["cpu"]
              or inventory["slot_capacity_bytes"] <= 0):
            raise RuntimeError("on-demand source/slot placement is invalid")
        if sum(isinstance(module, PackedLinear) for module in model.modules()) != 252:
            raise RuntimeError("expected 252 packed projections")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        startup_seconds = time.perf_counter() - process_started
        startup_peak_allocated = torch.cuda.max_memory_allocated(device)
        startup_peak_reserved = torch.cuda.max_memory_reserved(device)

        storage.reset_stats()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        measurement_start_allocated = torch.cuda.memory_allocated(device)
        measurement_start_reserved = torch.cuda.memory_reserved(device)
        route = SavedRouteSession(selected_examples, selected_routes, storage)
        evaluation_started = time.perf_counter()
        metrics = evaluate(model, selected_examples, out / "samples.jsonl", route)
        torch.cuda.synchronize(device)
        synchronized_wall_seconds = time.perf_counter() - evaluation_started
        if route.index != len(selected_examples):
            raise RuntimeError("not every saved route was consumed")
        transfer = storage.stats()
        if (transfer["requests"] != 72 * len(selected_examples)
                or transfer["view_closes"] != transfer["requests"]
                or transfer["active_slots_at_end"] != 0
                or transfer["max_active_slots"] != 1):
            raise RuntimeError("block request/release/slot accounting mismatch")
        if args.mode == "ondemand_sync":
            if transfer["loads"] != transfer["requests"] or transfer["releases"] != transfer["requests"]:
                raise RuntimeError("on-demand load/release count mismatch")
        elif transfer["loads"] != 0 or transfer["releases"] != 0 or transfer["copied_bytes"] != 0:
            raise RuntimeError("resident mode performed an execution transfer")

        measurement = {
            "start_allocated_bytes": measurement_start_allocated,
            "start_reserved_bytes": measurement_start_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "end_allocated_bytes": torch.cuda.memory_allocated(device),
            "end_reserved_bytes": torch.cuda.memory_reserved(device),
        }
        final_inventory = storage.inventory()
        if final_inventory["active_slot_bytes"] != 0:
            raise RuntimeError("active block payload survived evaluation")
        write_trace(out / "transfer_events.jsonl", storage.events)
        core_rows = [row for row in jsonl(Path(protocol["inherits"]["core_samples"]))
                     if row["task"] == "wikitext2"][:len(selected_examples)]
        actual_rows = jsonl(out / "samples.jsonl")
        core_samples_exact = actual_rows == core_rows
        if not core_samples_exact:
            raise RuntimeError("packed runtime output differs from completed core samples")
        result = {
            "protocol_id": protocol["protocol_id"],
            "mode": args.mode,
            "smoke_only": args.smoke,
            "sample_count": len(selected_examples),
            "metrics": metrics,
            "core_samples_exact": core_samples_exact,
            "startup": {
                "seconds": startup_seconds,
                "phases": phases,
                "storage_h2d_copied_bytes": storage_setup["h2d_copied_bytes"],
                "peak_allocated_bytes": startup_peak_allocated,
                "peak_reserved_bytes": startup_peak_reserved,
            },
            "measurement": {
                "synchronized_wall_seconds": synchronized_wall_seconds,
                **measurement,
            },
            "transfer": transfer,
            "storage": final_inventory,
            "model_gpu_inventory": gpu_model_inventory(model),
            "samples_sha256": sha256(out / "samples.jsonl"),
            "profiles_sha256": sha256(out / "profiles.json"),
            "transfer_events_sha256": sha256(out / "transfer_events.jsonl"),
            "source_manifest_sha256": source_manifest_hash,
            "process_wall_seconds": time.perf_counter() - process_started,
        }
        write_json(out / "results.json", result)
        print(json.dumps(result, indent=2), flush=True)
        storage.close()
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

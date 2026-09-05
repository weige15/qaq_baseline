"""Guarded, serial GPU stages: local labels, real-model gate, final comparisons."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM

from qaq.evaluation import evaluate, sha256, write_json
from qaq.model import NestedLinear, block_parameter_counts, blocks, load_quantized, set_profile
from qaq.quantization import reconstruct
from qaq.router import (check_budget, collect_context, distribution, load_router, quota_profile,
                        random_profile, router_costs)


ROOT = Path("results/core-v1")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def smoke_examples(examples, count=2):
    counts, selected = Counter(), []
    for ex in examples:
        task = ex.get("task", ex.get("split"))
        if counts[task] < count:
            selected.append(ex)
            counts[task] += 1
    return selected


def setup(args, out):
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise RuntimeError("one preflight-selected GPU required")
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hashes = json.loads((ROOT/"frozen/freeze_hashes.json").read_text())
    router_hashes = json.loads(Path("configs/router_data_lock.json").read_text())
    for directory, protected in ((ROOT/"frozen", hashes), (ROOT/"router-frozen", router_hashes)):
        for name, digest in protected.items():
            assert sha256(directory/name) == digest, name
    environment = json.loads((ROOT/"frozen/environment.json").read_text())
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "numpy", "safetensors", "tokenizers", "scipy", "pyarrow")}
    for name, version in versions.items():
        assert version == environment["packages"][name], f"runtime drift: {name}"
    assert json.loads((ROOT/"baseline-gate.json").read_text())["passed"]
    gate = json.loads((ROOT/"integration/gate.json").read_text())
    assert gate["passed"]
    checkpoint = ROOT/"integration/quantized_model.pt"
    assert sha256(checkpoint) == gate["checkpoint_sha256"]
    write_json(out/"command.json", {"argv": sys.argv, "executable": sys.executable,
        "cwd": os.getcwd(), "args": vars(args), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": torch.cuda.get_device_name(0), "properties": str(torch.cuda.get_device_properties(0)),
        "versions": versions, "cuda_runtime": torch.version.cuda,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_hashes": hashes, "router_hashes": router_hashes, "checkpoint_sha256": gate["checkpoint_sha256"]})
    for directory in ("src", "scripts", "configs", "tests"):
        shutil.copytree(directory, out/"source"/directory, ignore=shutil.ignore_patterns("__pycache__"))
    (out/"git-diff.patch").write_text(subprocess.check_output(["git", "diff"], text=True))
    model, _ = load_quantized(checkpoint, "cuda:0")
    return model, json.loads((ROOT/"router-frozen/protocol.json").read_text())


def collect(model, cfg, args, out):
    examples = jsonl(ROOT/"router-frozen/examples.jsonl")
    if args.smoke:
        examples = smoke_examples(examples)
    info = json.loads((ROOT/"frozen/model_manifest.json").read_text())
    teacher = AutoModelForCausalLM.from_pretrained(info["local_path"], dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).eval().to("cuda:0")
    teacher.requires_grad_(False)
    features, errors, timings = [], [], []
    for i, ex in enumerate(examples):
        started = time.time()
        x, y = collect_context(model, ex["context_tokens"], teacher)
        torch.cuda.synchronize()
        features.append(x)
        errors.append(y)
        timings.append(time.time()-started)
        print(f"collected {i+1}/{len(examples)} {ex['split']} {ex['article_index']}: {timings[-1]:.3f}s", flush=True)
    cache = {"features": torch.stack(features), "errors": torch.stack(errors),
             "ids": [[s["split"], s["article_index"]] for s in examples],
             "context_hashes": [s["context_sha256"] for s in examples]}
    torch.save(cache, out/"cache.pt")
    if not args.smoke:
        smoke = torch.load(ROOT/"router-collect-smoke/cache.pt", weights_only=True)
        for i, key in enumerate(smoke["ids"]):
            j = cache["ids"].index(key)
            assert torch.equal(smoke["features"][i], cache["features"][j]), f"feature repeat mismatch {key}"
            assert torch.equal(smoke["errors"][i], cache["errors"][j]), f"target repeat mismatch {key}"
    del teacher
    gc.collect()
    torch.cuda.empty_cache()  # Only this process's unused allocator memory.
    return {"contexts": len(examples), "per_context_seconds": timings,
            "cache_sha256": sha256(out/"cache.pt"), "smoke_repeat_exact": not args.smoke,
            "target_mean_by_precision": cache["errors"].mean((0, 1)).tolist()}


class RoutingSession:
    def __init__(self, model, mode, cfg, exact_weights=False):
        self.model, self.mode, self.cfg = model, mode, cfg
        self.counts = block_parameter_counts(model)
        training = ROOT/"router-training"
        selection = json.loads((training/"selection.json").read_text())
        assert selection["passed"]
        path = training/selection["selected"]/"router.pt"
        assert sha256(path) == selection["router_sha256"]
        assert sha256(training/"static_policy.json") == selection["static_policy_sha256"]
        self.router, self.saved = load_router(path)
        self.static = json.loads((training/"static_policy.json").read_text())["profile"]
        self.active = False
        self.records, self.handles = [], []
        self.exact_weights = exact_weights
        self.verified_weight_profiles = set()
        for index, block in enumerate(blocks(model)):
            for name, linear in block.named_children():
                if isinstance(linear, NestedLinear):
                    self.handles.append(linear.register_forward_hook(self.trace(index, name)))

    def trace(self, index, name):
        def hook(linear, args, output):
            if not self.active:
                return
            record = self.records[-1]
            assert linear.bits == record["profile"][index] and linear.active_weight is not None
            record["projection_calls"][index] += 1
            key = (index, name, linear.bits)
            if self.exact_weights and key not in self.verified_weight_profiles:
                assert torch.equal(linear.active_weight, reconstruct(linear.q, linear.scale, linear.bits, linear.dtype))
                self.verified_weight_profiles.add(key)
        return hook

    def __call__(self, tokens):
        self.active = False
        start = time.time()
        raw, _ = collect_context(self.model, tokens)
        costs = router_costs(self.router, raw.unsqueeze(0), self.saved["attempt"], self.saved["normalization"])[0]
        if self.mode == "adaptive":
            profile = quota_profile(costs.numpy())
        elif self.mode == "static":
            profile = self.static.copy()
        elif self.mode == "random":
            profile = random_profile(tokens, self.cfg["random_seed"])
        else:
            raise ValueError(self.mode)
        budget = check_budget(profile, self.counts)
        set_profile(self.model, profile)
        record = {"context_tokens": tokens, "profile": profile, "costs": costs.tolist(),
                  "raw_features": raw.tolist(), "budget": budget,
                  "projection_calls": [0]*len(profile), "probe_seconds": time.time()-start}
        self.records.append(record)
        self.active = True
        return profile

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def verify(model, cfg, out):
    dev = [e for e in jsonl(ROOT/"router-frozen/examples.jsonl") if e["split"] == "validation"]
    cache = torch.load(ROOT/"router-collect/cache.pt", weights_only=True)
    session = RoutingSession(model, "adaptive", cfg, exact_weights=True)
    try:
        profiles, raw_logits, features_exact = [], {}, []
        for i, ex in enumerate(dev):
            profile = session(ex["context_tokens"])
            profiles.append(profile)
            j = cache["ids"].index([ex["split"], ex["article_index"]])
            features_exact.append(torch.equal(torch.tensor(session.records[-1]["raw_features"]), cache["features"][j]))
            assert features_exact[-1], f"real feature mismatch {i}"
            if i < 3:
                inputs = torch.tensor([ex["context_tokens"]], device=model.device)
                with torch.inference_mode():
                    routed = model(inputs, use_cache=False, logits_to_keep=1).logits.cpu()
                session.active = False
                other = {}
                for bits in (4, 8):
                    set_profile(model, [bits]*72)
                    with torch.inference_mode():
                        other[str(bits)] = model(inputs, use_cache=False, logits_to_keep=1).logits.cpu()
                    assert torch.isfinite(other[str(bits)]).all() and not torch.equal(routed, other[str(bits)])
                assert torch.isfinite(routed).all()
                raw_logits[str(i)] = {"adaptive": routed, **other}
        selected = json.loads((ROOT/"router-training/selection.json").read_text())["selected"]
        fitted = json.loads((ROOT/"router-training"/selected/"results.json").read_text())
        assert profiles == fitted["dev_profiles"], "CPU fit vs real-model profiles differ"
        # Real scorer leak/repeat test: same dev context, changed WT2 suffix and MC answers/gold.
        prefix = dev[0]["context_tokens"]
        tokens = dev[0]["tokens"]
        wt = {"task": "wikitext2", "index": 0, "tokens": tokens, "prefix_length": 128}
        changed = {**wt, "tokens": prefix+list(reversed(tokens[128:]))}
        mc = {"task": "hellaswag", "index": 0, "choices": ["a", "bb"], "gold": 0,
              "encoded_choices": [(prefix+[10, 11], 128), (prefix+[12, 13], 128)]}
        mc_changed = {**mc, "choices": ["ccc", "dddd"], "gold": 1,
                      "encoded_choices": [(prefix+[14], 128), (prefix+[15], 128)]}
        offset = len(session.records)
        for i, ex in enumerate((wt, changed, mc, mc_changed, wt)):
            evaluate(model, [ex], out/f"causality-{i}.jsonl", session)
        checks = session.records[offset:]
        assert all(r["profile"] == profiles[0] and r["context_tokens"] == prefix for r in checks)
        assert jsonl(out/"causality-0.jsonl") == jsonl(out/"causality-4.jsonl")
        actual_counts = [r["projection_calls"] for r in checks]
        assert all(all(c > 0 for c in counts) for counts in actual_counts)
        write_json(out/"dev_routes.json", session.records)
        torch.save(raw_logits, out/"raw_logits.pt")
        weight_checks = len(session.verified_weight_profiles)
    finally:
        session.close()
    # Already scored endpoints are implementation checks only; no model selection.
    endpoint_checks = {}
    examples = smoke_examples(jsonl(ROOT/"frozen/examples.jsonl"))
    for mode, bits in (("fixed4", 4), ("fixed8", 8)):
        set_profile(model, [bits]*72)
        metrics = evaluate(model, examples, out/f"{mode}-samples.jsonl")
        actual = jsonl(out/f"{mode}-samples.jsonl")
        reference = jsonl(ROOT/f"smoke-{mode}"/"samples.jsonl")
        assert actual == reference, f"integrated endpoint mismatch {mode}"
        endpoint_checks[mode] = {"raw_samples_exact": True, "metrics": metrics}
    return {"passed": True, "feature_recompute_exact": all(features_exact), "dev_profiles_exact_to_fit": True,
            "dev_distribution": distribution(profiles), "genuine_projection_weight_checks": weight_checks,
            "causality_and_repeat": True, "different_finite_logits_vs_fixed4_and8": True,
            "endpoint_smoke": endpoint_checks}


def evaluate_routing(model, cfg, args, out):
    assert json.loads((ROOT/"router-verify/results.json").read_text())["passed"]
    examples = jsonl(ROOT/"frozen/examples.jsonl")
    if args.smoke:
        examples = smoke_examples(examples)
    session = RoutingSession(model, args.mode, cfg)
    try:
        metrics = evaluate(model, examples, out/"samples.jsonl", session)
    finally:
        session.close()
    for record, ex in zip(session.records, examples):
        calls = 1 if ex["task"] == "wikitext2" else len(ex["choices"])
        assert record["projection_calls"] == [calls*(4 if i%2 == 0 else 3) for i in range(72)]
        record.update(task=ex["task"], index=ex["index"])
        scoring_tokens = (len(ex["tokens"])-1 if ex["task"] == "wikitext2" else
                          sum(len(t)-1 for t, _ in ex["encoded_choices"]))
        record["scoring_forward_input_tokens"] = scoring_tokens
        record["probe_forward_input_tokens"] = len(record["context_tokens"])
        record["token_work_mean_bits"] = (6*scoring_tokens+8*len(record["context_tokens"]))/(scoring_tokens+len(record["context_tokens"]))
    write_json(out/"routes.json", session.records)
    write_json(out/"routing_distribution.json", distribution([r["profile"] for r in session.records]))
    counts = block_parameter_counts(model)
    return {"metrics": metrics, "sample_count": len(examples),
        "scoring_budget": check_budget(session.records[0]["profile"], counts),
        "block_parameter_counts": counts,
        "excluded_fp16_parameters": sum(p.numel() for p in model.parameters()),
        "scale_bytes": sum(m.scale.numel()*m.scale.element_size() for m in model.modules() if isinstance(m, NestedLinear)),
        "probe_tokens": sum(r["probe_forward_input_tokens"] for r in session.records),
        "scoring_forward_tokens": sum(r["scoring_forward_input_tokens"] for r in session.records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["collect", "verify", "evaluate"])
    parser.add_argument("--mode", choices=["adaptive", "static", "random"], default="adaptive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    try:
        model, cfg = setup(args, out)
        stage_start = time.time()
        if args.stage == "collect":
            result = collect(model, cfg, args, out)
        elif args.stage == "verify":
            assert not args.smoke
            result = verify(model, cfg, out)
        else:
            result = evaluate_routing(model, cfg, args, out)
        torch.cuda.synchronize()
        result.update(stage=args.stage, mode=args.mode, smoke_only=args.smoke,
            stage_seconds=time.time()-stage_start, total_seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(out/"results.json", result)
        if result["peak_allocated_bytes"] > cfg["limits"]["gpu_peak_bytes"]:
            raise RuntimeError("own-process peak exceeded20GB; pause")
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

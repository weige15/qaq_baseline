"""Real Qwen3 weight/compute/checkpoint gate. Must run through GPU preflight."""
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qaq.evaluation import sha256, write_json
from qaq.model import (block_parameter_counts, install_replacements, load_quantized,
                       prepare_replacements, save_quantized, set_profile)
from qaq.quantization import apply_fixed, reconstruct


@torch.inference_mode()
def probe(model, inputs):
    return [model(x, use_cache=False).logits.cpu() for x in inputs]


def main():
    root = Path("results/core-v1")
    out = root / "integration"
    out.mkdir(exist_ok=False)
    try:
        start = time.time()
        assert json.loads((root / "baseline-gate.json").read_text())["passed"]
        hashes = json.loads((root / "frozen/freeze_hashes.json").read_text())
        for name, digest in hashes.items():
            assert sha256(root / "frozen" / name) == digest
        assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
        torch.set_num_threads(4)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(out / "command.json", {"argv": sys.argv, "executable": sys.executable,
            "cwd": os.getcwd(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu": torch.cuda.get_device_name(0), "frozen_hashes": hashes})
        for directory in ("src", "scripts", "configs", "tests"):
            shutil.copytree(directory, out / "source" / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        info = json.loads((root / "frozen/model_manifest.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(info["local_path"], local_files_only=True)
        texts = ["Explain why the sky appears blue during the day.",
                 "A recipe combines flour, water, salt and yeast before baking.",
                 "Find the derivative of x squared and discuss its sign."]
        inputs = [torch.tensor([tokenizer.encode(s, add_special_tokens=False)], device="cuda:0") for s in texts]
        write_json(out / "probes.json", {"texts": texts, "tokens": [x.cpu().tolist() for x in inputs]})
        model = AutoModelForCausalLM.from_pretrained(info["local_path"], dtype=torch.float16,
            attn_implementation="sdpa", local_files_only=True).eval().to("cuda:0")
        model.requires_grad_(False)
        replacements = prepare_replacements(model)
        # Same fixed8 path as the completed baseline; q/scales were made BEFORE
        # mutating FP16 source weights, so there is no requantization of dequantized values.
        apply_fixed(model, 8)
        equalities = []
        for parent, name, packed, block in replacements:
            exact = torch.equal(getattr(parent, name).weight, reconstruct(packed.q, packed.scale, 8))
            equalities.append({"block": block, "projection": name, "exact": exact,
                               "parameters": packed.q.numel()})
        assert len(equalities) == 252 and all(s["exact"] for s in equalities)
        reference = probe(model, inputs)
        install_replacements(replacements)
        full8 = probe(model, inputs)
        assert all(torch.equal(a,b) for a,b in zip(full8, reference))
        block_counts = block_parameter_counts(model)
        assert len(block_counts) == 72 and len(set(block_counts[::2])) == len(set(block_counts[1::2])) == 1
        profiles = {"fixed4": [4]*72, "fixed6": [6]*72,
                    "attn0_only4": [4]+[8]*71, "ffn0_only4": [8,4]+[8]*70,
                    "alternating": [4,6,8]*24}
        outputs = {"reference8": reference, "integrated8": full8}
        deltas = {}
        for name, profile in profiles.items():
            set_profile(model, profile)
            outputs[name] = probe(model, inputs)
            delta = [(x-y).abs().max().item() for x,y in zip(outputs[name],full8)]
            assert all(torch.isfinite(x).all() for x in outputs[name]) and all(v > 0 for v in delta)
            deltas[name] = delta
        checkpoint = out / "quantized_model.pt"
        save_quantized(model, checkpoint, {"group_size": 128, "source": info,
            "format": "qaq-core-v1-signed-midpoint", "frozen_hashes": hashes})
        # State hashes establish exact reload of EVERY persisted tensor, not logits only.
        import hashlib
        state_hashes = {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for k,v in model.state_dict().items()}
        write_json(out / "state_hashes.json", state_hashes)
        del model, replacements, packed, parent
        gc.collect()
        torch.cuda.empty_cache()  # releases only this process's unused allocations
        loaded, _ = load_quantized(checkpoint, "cuda:0")
        for k,v in loaded.state_dict().items():
            assert hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() == state_hashes[k], k
        assert len(loaded.state_dict()) == len(state_hashes)
        reload_exact = {}
        for name, profile in {"integrated8": [8]*72, **profiles}.items():
            set_profile(loaded, profile)
            actual = probe(loaded, inputs)
            reload_exact[name] = all(torch.equal(a,b) for a,b in zip(actual,outputs[name]))
            assert reload_exact[name], f"reload mismatch: {name}"
        torch.save(outputs, out / "raw_probe_logits.pt")
        write_json(out / "gate.json", {"stage": "integration_only", "passed": True,
            "full_width_tensor_checks": equalities, "full8_logits_exact_to_fixed8": True,
            "lower_and_independent_block_max_logit_deltas": deltas, "profiles": profiles,
            "reload_exact": reload_exact, "state_tensor_count": len(state_hashes),
            "block_parameter_counts": block_counts, "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size, "seconds": time.time()-start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "transient_cache": "one FP16 reconstruction per projection; never serialized",
            "does_not_verify": ["trained_router", "unseen_query_variation", "matched-budget quality"]})
        print("INTEGRATION GATE PASSED", flush=True)
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

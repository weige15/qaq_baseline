"""Bounded CPU-only router fitting. Select first noncollapsed declared attempt."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from qaq.evaluation import sha256, write_json
from qaq.router import (BlockRouter, distribution, normalize_features, quota_profile,
                        router_costs, select_features)


def main():
    root = Path("results/core-v1")
    out = root / "router-training"
    out.mkdir(exist_ok=False)
    started = time.time()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("router fitting must explicitly hide CUDA")
        frozen = root / "router-frozen"
        hashes = json.loads(Path("configs/router_data_lock.json").read_text())
        for name, digest in hashes.items():
            assert sha256(frozen/name) == digest, name
        cfg = json.loads((frozen/"protocol.json").read_text())
        collection = root / "router-collect"
        status = json.loads((collection/"results.json").read_text())
        assert status["stage"] == "collect" and not status["smoke_only"]
        assert sha256(collection/"cache.pt") == status["cache_sha256"]
        data = torch.load(collection/"cache.pt", weights_only=True)
        examples = [json.loads(s) for s in (frozen/"examples.jsonl").read_text().splitlines()]
        assert data["ids"] == [[s["split"], s["article_index"]] for s in examples]
        train = torch.tensor([s["split"] == "train" for s in examples])
        assert train.sum() == 192 and (~train).sum() == 32
        features, errors = data["features"], data["errors"]
        assert features.shape == (224, 72, 132) and errors.shape == (224, 72, 3)
        assert torch.isfinite(features).all() and torch.isfinite(errors).all() and (errors >= 0).all()
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        for directory in ("src", "scripts", "configs", "tests"):
            shutil.copytree(directory, out/"source"/directory, ignore=shutil.ignore_patterns("__pycache__"))
        write_json(out/"command.json", {"argv": sys.argv, "executable": sys.executable,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "frozen_hashes": hashes, "cache_sha256": status["cache_sha256"], "torch": torch.__version__})
        static_costs = errors[train].mean(0)
        static_profile = quota_profile(static_costs.numpy())
        write_json(out/"static_policy.json", {"profile": static_profile,
            "train_mean_nmse": static_costs.tolist(), "source": "train-only mean raw local FP16 distillation NMSE"})
        attempts = []
        selected = None
        train_cfg = cfg["training"]
        for number, spec in enumerate(cfg["attempts"], 1):
            directory = out/spec["name"]
            directory.mkdir()
            seed = train_cfg["seed_base"]+number
            torch.manual_seed(seed)
            x = select_features(features, spec["groups"])
            std = x[train].std(0, unbiased=False)
            normalization = {"x_mean": x[train].mean(0), "x_std": std.clamp_min(train_cfg["std_floor"]),
                             "x_mask": std >= train_cfg["std_floor"]}
            x = normalize_features(x, normalization)
            if spec["objective"] == "log_nmse":
                y = (errors+1e-12).log()
                normalization.update(y_mean=y[train].mean(0),
                    y_std=y[train].std(0, unbiased=False).clamp_min(train_cfg["std_floor"]))
                y = (y-normalization["y_mean"])/normalization["y_std"]
                loss_fn = F.mse_loss
            else:
                y = torch.tensor([[(b-4)//2 for b in quota_profile(e.numpy())] for e in errors])
                def loss_fn(prediction, target):
                    return F.cross_entropy(prediction.reshape(-1, 3), target.reshape(-1))
            router = BlockRouter(x.shape[-1], spec["hidden"])
            optimizer = torch.optim.AdamW(router.parameters(), lr=train_cfg["lr"],
                betas=(0.9, 0.999), eps=1e-8, weight_decay=train_cfg["weight_decay"])
            with torch.no_grad():
                initial = {"train": loss_fn(router(x[train]), y[train]).item(),
                           "dev": loss_fn(router(x[~train]), y[~train]).item()}
            history = []
            for epoch in range(train_cfg["epochs"]):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(router(x[train]), y[train])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite training loss in {spec['name']}")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(router.parameters(), train_cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                history.append({"epoch": epoch+1, "train_loss_before_step": loss.item(), "gradient_norm": grad.item()})
            router.eval()
            with torch.no_grad():
                final = {"train": loss_fn(router(x[train]), y[train]).item(),
                         "dev": loss_fn(router(x[~train]), y[~train]).item()}
                dev_costs = router_costs(router, features[~train], spec, normalization)
                dev_profiles = [quota_profile(c.numpy()) for c in dev_costs]
                # This ablation detects accidental query-index/tie randomization.
                constant = features[train].mean(0).expand(int((~train).sum()), -1, -1)
                constant_profiles = [quota_profile(c.numpy()) for c in router_costs(router, constant, spec, normalization)]
            dist = distribution(dev_profiles)
            gate = cfg["noncollapse"]
            passed = (final["train"] < initial["train"] and
                dist["unique_profiles"] >= gate["min_unique_profiles"] and
                dist["largest_profile_fraction"] <= gate["max_profile_fraction"] and
                dist["varying_attention_blocks"] >= gate["min_varying_blocks_per_type"] and
                dist["varying_ffn_blocks"] >= gate["min_varying_blocks_per_type"])
            assert distribution(constant_profiles)["unique_profiles"] == 1
            record = {"attempt": spec, "seed": seed, "initial_loss": initial, "final_loss": final,
                      "passed": passed, "dev_distribution": dist, "dev_profiles": dev_profiles,
                      "constant_feature_unique_profiles": 1, "parameter_count": sum(p.numel() for p in router.parameters()),
                      "masked_feature_coordinates": int((~normalization["x_mask"]).sum())}
            write_json(directory/"history.json", history)
            write_json(directory/"results.json", record)
            torch.save({"attempt": spec, "state_dict": router.state_dict(), "normalization": normalization,
                        "data_hashes": hashes, "cache_sha256": status["cache_sha256"], "seed": seed}, directory/"router.pt")
            torch.save({"costs": dev_costs, "profiles": dev_profiles}, directory/"dev_predictions.pt")
            attempts.append(record)
            print(json.dumps({"attempt": spec["name"], "passed": passed, "initial": initial,
                              "final": final, "unique_dev_profiles": dist["unique_profiles"]}), flush=True)
            if passed:
                selected = spec["name"]
                break
        write_json(out/"selection.json", {"passed": selected is not None, "selected": selected,
            "attempts_executed": [a["attempt"]["name"] for a in attempts],
            "not_run": [a["name"] for a in cfg["attempts"] if a["name"] not in [r["attempt"]["name"] for r in attempts]],
            "seconds": time.time()-started,
            "router_sha256": sha256(out/selected/"router.pt") if selected else None,
            "static_policy_sha256": sha256(out/"static_policy.json"),
            "does_not_verify": ["real_model_genuine_use", "final_generalization", "equal_bit_quality_benefit"]})
        if selected is None:
            raise RuntimeError("all declared router attempts collapsed; stop without final scoring")
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

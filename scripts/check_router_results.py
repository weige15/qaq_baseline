"""CPU evidence audit and locked paired comparison. No model/metric selection.

This covers router/data/comparison evidence, not the paper-reading, exclusion,
license, final-report content, or whole-goal completion audit.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from check_baselines import check as check_baselines
from prepare_router import articles, spans, text_hash
from qaq.evaluation import sha256, summarize, write_json
from qaq.router import (check_budget, distribution, load_router, normalize_features,
                        quota_profile, random_profile, router_costs, select_features)


ROOT = Path("results/core-v1")


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def validate_samples(rows, examples, metrics, cfg):
    assert [(r["task"], r["index"]) for r in rows] == [(e["task"], e["index"]) for e in examples]
    json.dumps(rows, allow_nan=False)
    for row, ex in zip(rows, examples):
        if ex["task"] == "wikitext2":
            assert len(row["token_nll"]) == row["scored_tokens"] == len(ex["tokens"])-ex["prefix_length"]
            assert row["nll_sum"] == sum(row["token_nll"])
        else:
            assert row["gold"] == ex["gold"]
            assert [len(t) for t in row["token_nll"]] == [len(t)-n for t, n in ex["encoded_choices"]]
            assert row["loglikelihoods"] == [-sum(t) for t in row["token_nll"]]
            assert row["normalized_loglikelihoods"] == [ll/len(c) for ll, c in zip(row["loglikelihoods"], ex["choices"])]
            for prediction, score in (("prediction", "loglikelihoods"), ("prediction_norm", "normalized_loglikelihoods")):
                assert row[prediction] == max(range(len(ex["choices"])), key=row[score].__getitem__)
    assert summarize(rows) == metrics
    assert set(metrics) == set(cfg["data"])
    for task, values in metrics.items():
        key = "wikitext_metrics" if task == "wikitext2" else "multiple_choice_metrics"
        assert set(values) == set(cfg["evaluation"][key])


def data_audit(examples):
    frozen = ROOT/"router-frozen"
    hashes = read_json(Path("configs/router_data_lock.json"))
    assert hashes == read_json(frozen/"freeze_hashes.json")
    for name, digest in hashes.items():
        assert sha256(frozen/name) == digest, name
    manifest = read_json(frozen/"data_manifest.json")
    documents = {}
    for split, spec in manifest["source"].items():
        assert sha256(spec["path"]) == spec["sha256"]
        documents[split] = articles(pq.read_table(spec["path"]).to_pylist())
    info = read_json(ROOT/"frozen/model_manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(info["local_path"], local_files_only=True)
    final_spans = set()
    for ex in examples:
        for tokens in ([ex["tokens"]] if ex["task"] == "wikitext2" else [t for t, _ in ex["encoded_choices"]]):
            final_spans.update(spans(tokens))
    selected = read_rows(frozen/"examples.jsonl")
    split_spans, ids, contexts = {}, set(), set()
    for ex in selected:
        key = (ex["split"], ex["article_index"])
        assert key not in ids
        ids.add(key)
        assert ex["context_sha256"] not in contexts
        contexts.add(ex["context_sha256"])
        original = documents[ex["split"]][ex["article_index"]]
        for name in ("title", "text", "start_row", "text_hash", "title_hash"):
            assert ex[name] == original[name]
        forbidden = documents["test"] + (documents["validation"] if ex["split"] == "train" else [])
        assert ex["title_hash"] not in {d["title_hash"] for d in forbidden}
        assert ex["text_hash"] not in {d["text_hash"] for d in forbidden}
        tokens = tokenizer.encode(ex["text"], add_special_tokens=False)
        assert ex["tokens"] == tokens[ex["offset"]:ex["offset"]+512] and len(ex["tokens"]) == 512
        assert ex["context_tokens"] == ex["tokens"][:128]
        import hashlib
        assert ex["context_sha256"] == hashlib.sha256(json.dumps(ex["context_tokens"]).encode()).hexdigest()
        assert ex["text_hash"] == text_hash(ex["text"])
        ng = spans(ex["tokens"])
        assert not ng & final_spans
        split_spans.setdefault(ex["split"], set()).update(ng)
    assert not split_spans["train"] & split_spans["validation"]
    assert sum(s[0] == "train" for s in ids) == 192 and sum(s[0] == "validation" for s in ids) == 32
    return selected, {"passed": True, "train_articles": 192, "dev_articles": 32,
                      "unique_contexts": len(contexts), "cross_split_32_token_overlaps": 0, "hashes": hashes}


def paired_comparison(left, right, indices, exp=False):
    left, right = np.array(left), np.array(right)
    if exp:
        delta = float(np.exp(left.mean())-np.exp(right.mean()))
        samples = np.exp(left[indices].mean(1))-np.exp(right[indices].mean(1))
    else:
        difference = left-right
        delta = float(difference.mean())
        samples = difference[indices].mean(1)
    return {"delta": delta, "ci95": np.quantile(samples, [.025, .975]).tolist()}


def main():
    torch.set_num_threads(4)
    baseline = check_baselines(ROOT)
    examples = read_rows(ROOT/"frozen/examples.jsonl")
    selected_data, data_evidence = data_audit(examples)
    cfg = read_json(ROOT/"router-frozen/protocol.json")
    training = ROOT/"router-training"
    choice = read_json(training/"selection.json")
    assert choice["passed"] and not (training/"failure.txt").exists()
    router_path = training/choice["selected"]/"router.pt"
    assert sha256(router_path) == choice["router_sha256"]
    assert sha256(training/"static_policy.json") == choice["static_policy_sha256"]
    router, saved = load_router(router_path)
    assert saved["data_hashes"] == data_evidence["hashes"]
    collection = ROOT/"router-collect"
    assert not (collection/"failure.txt").exists()
    assert sha256(collection/"cache.pt") == saved["cache_sha256"] == read_json(collection/"results.json")["cache_sha256"]
    cache = torch.load(collection/"cache.pt", weights_only=True)
    assert cache["ids"] == [[r["split"], r["article_index"]] for r in selected_data]
    assert cache["context_hashes"] == [r["context_sha256"] for r in selected_data]
    train = torch.tensor([r["split"] == "train" for r in selected_data])
    x = select_features(cache["features"], saved["attempt"]["groups"])
    norm = saved["normalization"]
    assert torch.equal(norm["x_mean"], x[train].mean(0))
    assert torch.equal(norm["x_std"], x[train].std(0, unbiased=False).clamp_min(1e-5))
    assert torch.equal(norm["x_mask"], x[train].std(0, unbiased=False) >= 1e-5)
    log_targets = (cache["errors"]+1e-12).log()
    if saved["attempt"]["objective"] == "log_nmse":
        assert torch.equal(norm["y_mean"], log_targets[train].mean(0))
        assert torch.equal(norm["y_std"], log_targets[train].std(0, unbiased=False).clamp_min(1e-5))
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(router(normalize_features(x[train], norm)),
                                               (log_targets[train]-norm["y_mean"])/norm["y_std"]).item()
        assert loss == read_json(training/choice["selected"]/"results.json")["final_loss"]["train"]
    static = read_json(training/"static_policy.json")
    assert static["profile"] == quota_profile(cache["errors"][train].mean(0).numpy())
    assert static["train_mean_nmse"] == cache["errors"][train].mean(0).tolist()
    dev_profiles = [quota_profile(c.numpy()) for c in router_costs(router, cache["features"][~train], saved["attempt"], norm)]
    fit = read_json(training/choice["selected"]/"results.json")
    assert dev_profiles == fit["dev_profiles"]
    dist = distribution(dev_profiles)
    assert dist == fit["dev_distribution"]
    assert dist["unique_profiles"] >= 4 and dist["largest_profile_fraction"] <= .9
    assert dist["varying_attention_blocks"] >= 2 and dist["varying_ffn_blocks"] >= 2
    assert fit["final_loss"]["train"] < fit["initial_loss"]["train"]
    assert len(read_json(training/choice["selected"]/"history.json")) == 300
    assert choice["attempts_executed"] == [s["name"] for s in cfg["attempts"][:len(choice["attempts_executed"])]]
    assert choice["attempts_executed"][-1] == choice["selected"]
    verification = read_json(ROOT/"router-verify/results.json")
    assert verification["passed"] and not (ROOT/"router-verify/failure.txt").exists()
    assert all(verification[k] for k in ("feature_recompute_exact", "dev_profiles_exact_to_fit", "causality_and_repeat", "different_finite_logits_vs_fixed4_and8"))
    logits = torch.load(ROOT/"router-verify/raw_logits.pt", weights_only=True)
    assert len(logits) == 3
    for values in logits.values():
        assert all(torch.isfinite(v).all() for v in values.values())
        assert all(not torch.equal(values["adaptive"], values[k]) for k in ("4", "8"))
    all_rows = {m: read_rows(ROOT/f"baseline-{m}-r1/samples.jsonl") for m in ("fixed4", "fixed8", "fp16")}
    metrics = dict(baseline["metrics"])
    repeat, distributions, routes_first, job_hashes = {}, {}, {}, {}
    weighted_budget = None
    expected_base_hashes = read_json(ROOT/"frozen/freeze_hashes.json")
    for mode in cfg["comparison_modes"]:
        for repetition in (1, 2):
            directory = ROOT/f"comparison-{mode}-r{repetition}"
            result, rows = read_json(directory/"results.json"), read_rows(directory/"samples.jsonl")
            routes = read_json(directory/"routes.json")
            assert not (directory/"failure.txt").exists()
            assert result["mode"] == mode and result["stage"] == "evaluate" and not result["smoke_only"]
            assert len(rows) == len(routes) == result["sample_count"] == 576
            command = read_json(directory/"command.json")
            assert command["base_hashes"] == expected_base_hashes and command["router_hashes"] == data_evidence["hashes"]
            assert command["checkpoint_sha256"] == read_json(ROOT/"integration/gate.json")["checkpoint_sha256"]
            validate_samples(rows, examples, result["metrics"], read_json(ROOT/"frozen/protocol.json"))
            for row, route, ex in zip(rows, routes, examples):
                assert (route["task"], route["index"]) == (ex["task"], ex["index"])
                expected_context = ex["tokens"][:128] if ex["task"] == "wikitext2" else ex["encoded_choices"][0][0][:ex["encoded_choices"][0][1]]
                assert route["context_tokens"] == expected_context
                costs = router_costs(router, torch.tensor(route["raw_features"]).unsqueeze(0), saved["attempt"], norm)[0]
                assert costs.tolist() == route["costs"]
                expected = quota_profile(costs.numpy()) if mode == "adaptive" else (
                    static["profile"] if mode == "static" else random_profile(expected_context, cfg["random_seed"]))
                assert expected == route["profile"] == row["profile"]
                budget = check_budget(expected, result["block_parameter_counts"])
                assert budget == route["budget"] == result["scoring_budget"]
                if weighted_budget is None:
                    weighted_budget = budget
                assert budget == weighted_budget
                calls = 1 if ex["task"] == "wikitext2" else len(ex["choices"])
                assert route["projection_calls"] == [calls*(4 if i%2 == 0 else 3) for i in range(72)]
                scoring = len(ex["tokens"])-1 if ex["task"] == "wikitext2" else sum(len(t)-1 for t, _ in ex["encoded_choices"])
                assert route["scoring_forward_input_tokens"] == scoring
                assert route["probe_forward_input_tokens"] == len(expected_context)
                assert route["token_work_mean_bits"] == (6*scoring+8*len(expected_context))/(scoring+len(expected_context))
            assert sum(r["probe_forward_input_tokens"] for r in routes) == result["probe_tokens"]
            assert sum(r["scoring_forward_input_tokens"] for r in routes) == result["scoring_forward_tokens"]
            actual_dist = distribution([r["profile"] for r in rows])
            assert actual_dist == read_json(directory/"routing_distribution.json")
            job_hashes[directory.name] = {f: sha256(directory/f) for f in ("results.json", "samples.jsonl", "routes.json", "command.json")}
            if repetition == 1:
                all_rows[mode], routes_first[mode] = rows, routes
                metrics[mode] = result["metrics"]
                distributions[mode] = {task: distribution([r["profile"] for r in rows if r["task"] == task]) for task in metrics[mode]}
            else:
                assert rows == all_rows[mode], f"exact repeat raw sample mismatch {mode}"
                for a, b in zip(routes_first[mode], routes):
                    assert {k:v for k,v in a.items() if k != "probe_seconds"} == {k:v for k,v in b.items() if k != "probe_seconds"}
                repeat[mode] = {"samples_exact": True, "profiles_features_costs_exact": True}
    for mode in ("static", "random"):
        for a, b in zip(routes_first["adaptive"], routes_first[mode]):
            assert a["context_tokens"] == b["context_tokens"] and a["raw_features"] == b["raw_features"]
            assert a["token_work_mean_bits"] == b["token_work_mean_bits"]
    assert all(d["unique_profiles"] >= 2 for d in distributions["adaptive"].values())
    assert all(d["unique_profiles"] == 1 for d in distributions["static"].values())
    comparisons = {}
    for task in metrics["adaptive"]:
        rows_by_mode = {m:[r for r in rows if r["task"] == task] for m,rows in all_rows.items()}
        rng = np.random.default_rng(cfg["bootstrap"]["seed"])
        n = len(rows_by_mode["adaptive"])
        indices = rng.integers(0, n, (cfg["bootstrap"]["draws"], n))
        for comparator in ("static", "random", "fixed4", "fixed8", "fp16"):
            left, right = rows_by_mode["adaptive"], rows_by_mode[comparator]
            fields = ("mean_nll", "token_perplexity") if task == "wikitext2" else ("acc", "acc_norm")
            for metric in fields:
                if task == "wikitext2":
                    a, b = [[r["nll_sum"]/r["scored_tokens"] for r in rows] for rows in (left, right)]
                else:
                    pred = "prediction_norm" if metric == "acc_norm" else "prediction"
                    a, b = [[float(r[pred] == r["gold"]) for r in rows] for rows in (left, right)]
                comparisons.setdefault(comparator, {}).setdefault(task, {})[metric] = paired_comparison(a, b, indices, metric == "token_perplexity")
    primary = comparisons["static"]
    improved = primary["wikitext2"]["mean_nll"]["ci95"][1] < 0
    guardrail = all(primary[t]["acc_norm"]["delta"] >= -.02 for t in ("hellaswag", "arc_challenge"))
    write_json(ROOT/"comparison-gate.json", {"evidence_checks_passed": True,
        "data": data_evidence, "selected_router": choice, "dev_distribution": dist,
        "repeatability": repeat, "weighted_scoring_budget": weighted_budget,
        "matched_probe_and_token_work_exact": True, "metrics": metrics,
        "routing_by_task": distributions, "comparisons": comparisons,
        "primary_benefit_passed": improved and guardrail,
        "decision": "positive_under_frozen_criterion" if improved and guardrail else "negative_or_inconclusive_stop_no_more_search",
        "job_hashes": job_hashes, "does_not_verify": ["paper_reading", "license_scope", "gpu_guard_logs", "final_report_content", "whole_goal_completion"]})
    print(json.dumps({"evidence_checks_passed": True, "primary_benefit_passed": improved and guardrail,
                      "adaptive_minus_static": primary, "repeatability": repeat}, indent=2))


if __name__ == "__main__":
    main()

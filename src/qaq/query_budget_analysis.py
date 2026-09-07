"""Frozen, CPU-only endpoint analysis; no model, router fitting, or serving code."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TASKS = ("wikitext2", "hellaswag", "arc_challenge")
BUDGETS = (4, 5, 6, 7, 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_hashes(root, expected):
    for name, digest in expected.items():
        path = Path(root) / name
        require(path.is_file(), f"missing input: {path}")
        actual = sha256(path)
        require(actual == digest, f"hash-invalid {path}: expected {digest}, actual {actual}")


def query_ids(rows, label):
    ids = []
    for row in rows:
        key = (row["task"], row["index"])
        require(key[0] in TASKS and type(key[1]) is int and key[1] >= 0,
                f"invalid task/index: {label}: {key}")
        require(key not in ids, f"repeated ID: {label}: {key}")
        ids.append(key)
    require(bool(ids), f"empty input: {label}")
    return ids


def aligned(rows, expected, label):
    require(query_ids(rows, label) == expected, f"missing/extra/out-of-order queries: {label}")


def exact_repeat(left, right, label, timing=()):
    if isinstance(left, list):
        require(len(left) == len(right), f"mismatched repeats: {label}: count")
        for i, (a, b) in enumerate(zip(left, right)):
            exact_repeat(a, b, f"{label}[{i}]", timing)
    else:
        require({k: v for k, v in left.items() if k not in timing} ==
                {k: v for k, v in right.items() if k not in timing},
                f"mismatched repeats: {label}")


def validate_sample(row, ex, label):
    # Validation only: no aggregation across queries or model/scorer invocation.
    json.dumps(row, allow_nan=False)
    if ex["task"] == "wikitext2":
        require(row["scored_tokens"] == len(row["token_nll"]) ==
                len(ex["tokens"]) - ex["prefix_length"] > 0, f"token count: {label}")
        require(row["nll_sum"] == sum(row["token_nll"]), f"NLL sum: {label}")
        require(all(x >= 0 for x in row["token_nll"]), f"negative NLL: {label}")
    else:
        losses = row["token_nll"]
        require(row["gold"] == ex["gold"], f"gold mismatch: {label}")
        require(len(losses) == len(ex["choices"]), f"choice count: {label}")
        require([len(x) for x in losses] == [len(t)-n for t, n in ex["encoded_choices"]],
                f"choice token counts: {label}")
        require(all(x >= 0 for choice in losses for x in choice), f"negative NLL: {label}")
        ll = [-sum(x) for x in losses]
        norm = [x/len(c) for x, c in zip(ll, ex["choices"])]
        require(row["loglikelihoods"] == ll and row["normalized_loglikelihoods"] == norm,
                f"choice scores: {label}")
        for pred, scores in (("prediction", ll), ("prediction_norm", norm)):
            require(row[pred] == max(range(len(scores)), key=scores.__getitem__),
                    f"prediction: {label}: {pred}")


def router_score(costs, counts):
    require(len(costs) == len(counts) > 0 and
            all(type(c) is int and c > 0 for c in counts), "router block counts")
    require(all(len(c) == 3 and all(math.isfinite(x) and x >= 0 for x in c)
                for c in costs), "router costs must be finite nonnegative [blocks,3]")
    return math.fsum(p * (c[0] - c[2]) for p, c in zip(counts, costs)) / sum(counts)


def transition(correct4, correct8):
    if not correct4 and correct8:
        return "fixed4_wrong_fixed8_correct"
    if correct4 and not correct8:
        return "fixed4_correct_fixed8_wrong"
    return "both_unchanged"


def make_pair(four, eight, route, counts):
    require((four["task"], four["index"]) == (eight["task"], eight["index"]) ==
            (route["task"], route["index"]), "pair identity mismatch")
    pair = {"task": four["task"], "index": four["index"],
            "router_score": router_score(route["costs"], counts),
            "prompt_tokens": len(route["context_tokens"])}
    require(pair["prompt_tokens"] > 0, "empty prompt")
    if four["task"] == "wikitext2":
        require(four["scored_tokens"] == eight["scored_tokens"] > 0, "paired token counts")
        pair.update(nll_sum4=four["nll_sum"], nll_sum8=eight["nll_sum"],
                    scored_tokens4=four["scored_tokens"], scored_tokens8=eight["scored_tokens"])
        pair["benefit"] = four["nll_sum"]/four["scored_tokens"] - eight["nll_sum"]/eight["scored_tokens"]
    else:
        require(four["gold"] == eight["gold"], "paired gold mismatch")
        pair["gold"] = four["gold"]
        for metric, pred in (("acc_norm", "prediction_norm"), ("acc", "prediction")):
            a, b = int(four[pred] == four["gold"]), int(eight[pred] == eight["gold"])
            pair.update({metric+"4": a, metric+"8": b,
                         metric+"_transition": transition(a, b),
                         pred+"4": four[pred], pred+"8": eight[pred]})
        pair["benefit"] = pair["acc_norm8"] - pair["acc_norm4"]
    json.dumps(pair, allow_nan=False)
    return pair


def load_pairs(root, cfg):
    root = Path(root)
    verify_hashes(root, cfg["input_sha256"])
    verify_hashes(root, {cfg["historical_inventory"]: cfg["historical_inventory_sha256"]})
    historical = read_json(root/cfg["historical_inventory"])["protected_raw"]
    for p, digest in cfg["input_sha256"].items():
        require(historical[p]["sha256"] == digest, f"historical hash mismatch: {p}")
    base = root/"results/core-v1"
    frozen = read_json(base/"frozen/freeze_hashes.json")
    verify_hashes(base/"frozen", frozen)
    examples = read_rows(base/"frozen/examples.jsonl")
    ids = query_ids(examples, "frozen/examples.jsonl")
    require(Counter(t for t, _ in ids) == {t: s["count"] for t, s in cfg["tasks"].items()},
            "frozen task counts")
    manifest = read_json(base/"frozen/data_manifest.json")
    for task in TASKS:
        require([i for t, i in ids if t == task] == manifest[task]["indices"],
                f"data manifest indices: {task}")
    endpoints = {}
    for mode in ("fixed4", "fixed8"):
        runs, results = [], []
        for repeat in (1, 2):
            folder = base/f"baseline-{mode}-r{repeat}"
            rows = read_rows(folder/"samples.jsonl")
            aligned(rows, ids, str(folder/"samples.jsonl"))
            result = read_json(folder/"results.json")
            require(result["mode"] == mode and not result["smoke_only"] and
                    result["sample_count"] == len(ids), f"baseline run metadata: {folder}")
            command = read_json(folder/"command.json")
            require(command["frozen_hashes"] == frozen, f"frozen command hashes: {folder}")
            require(not (folder/"failure.txt").exists(), f"failed closed run: {folder}/failure.txt")
            for row, ex in zip(rows, examples):
                validate_sample(row, ex, f"{folder}/samples.jsonl:{ex['task']}:{ex['index']}")
                require(row["profile"] is None, f"unexpected endpoint profile: {folder}")
            runs.append(rows)
            results.append(result)
        exact_repeat(*runs, str(base/f"baseline-{mode}-r2/samples.jsonl"))
        exact_repeat(*results, str(base/f"baseline-{mode}-r2/results.json"), cfg["baseline_timing_fields"])
        endpoints[mode] = runs[0]
    gate = read_json(base/"integration/gate.json")
    counts = gate["block_parameter_counts"]
    require(gate["passed"] and len(counts) == 72, "integration/gate.json")
    projections = gate["full_width_tensor_checks"]
    require(len(projections) == len({(p["block"], p["projection"]) for p in projections}) == 252,
            "integration projection inventory")
    require(all(p["exact"] for p in projections), "integration projection equality")
    require([sum(p["parameters"] for p in projections if p["block"] == b)
             for b in range(72)] == counts, "integration projection counts")
    for mode in ("fixed4", "fixed8"):
        for repeat in (1, 2):
            p = base/f"baseline-{mode}-r{repeat}/quantized_modules.json"
            modules = read_json(p)
            require(len(modules) == len({m["name"] for m in modules}) == 252, f"module inventory: {p}")
            actual = [0]*72
            for m in modules:
                layer, kind, _ = m["name"].split(".")
                require(kind in ("attn", "ffn") and m["parameters"] == math.prod(m["shape"])
                        and m["bits"] == int(mode[-1]), f"module shape/bits: {p}")
                actual[2*int(layer)+(kind == "ffn")] += m["parameters"]
            require(actual == counts, f"module block counts: {p}")
    routes = []
    for repeat in (1, 2):
        folder = base/f"comparison-adaptive-r{repeat}"
        records = read_json(folder/"routes.json")
        aligned(records, ids, str(folder/"routes.json"))
        result, command = read_json(folder/"results.json"), read_json(folder/"command.json")
        require(result["block_parameter_counts"] == counts and result["sample_count"] == len(ids),
                f"adaptive count metadata: {folder}")
        require(command["base_hashes"] == frozen and command["checkpoint_sha256"] == gate["checkpoint_sha256"],
                f"adaptive provenance: {folder}")
        for r, ex in zip(records, examples):
            prefix = (ex["tokens"][:ex["prefix_length"]] if ex["task"] == "wikitext2" else
                      ex["encoded_choices"][0][0][:ex["encoded_choices"][0][1]])
            require(r["context_tokens"] == prefix and r["probe_forward_input_tokens"] == len(prefix),
                    f"prompt mismatch: {folder}/routes.json:{ex['task']}:{ex['index']}")
            require(r["budget"]["quantized_parameters"] == sum(counts), f"route parameter count: {folder}")
        routes.append(records)
    exact_repeat(*routes, str(base/"comparison-adaptive-r2/routes.json"), cfg["adaptive_route_timing_fields"])
    return [make_pair(a, b, r, counts) for a, b, r in zip(endpoints["fixed4"], endpoints["fixed8"], routes[0])]


def quota(n, budget):
    require(type(n) is int and n > 0 and budget in BUDGETS, "invalid query count/budget")
    k = Fraction(n*(budget-4), 4)
    require(k.denominator == 1, "nonintegral exact fixed8 quota")
    return int(k)


def ranks(pairs, salts):
    ids = query_ids(pairs, "analysis pairs")
    keys = {"outcome_informed": [-p["benefit"] for p in pairs],
            "router_score": [-p["router_score"] for p in pairs],
            "prompt_length": [-p["prompt_tokens"] for p in pairs]}
    for salt in salts:
        keys[f"id_hash_{salt}"] = [hashlib.sha256(f"{salt}:{t}:{i}".encode()).hexdigest() for t, i in ids]
    return {name: sorted(range(len(pairs)), key=lambda i: (values[i], ids[i]))
            for name, values in keys.items()}


def aggregate(pairs, fixed8):
    """Also accepts a uniform probability vector for the exact-quota expectation."""
    require(len(pairs) == len(fixed8) > 0, "aggregate count mismatch")
    require(len({p["task"] for p in pairs}) == 1, "cannot pool tasks")
    require(all(0 <= x <= 1 for x in fixed8), "invalid selection probabilities")
    def total(field):
        return math.fsum((1-f)*p[field+"4"] + f*p[field+"8"] for p, f in zip(pairs, fixed8))
    if pairs[0]["task"] == "wikitext2":
        require(all(p["scored_tokens4"] == p["scored_tokens8"] > 0 for p in pairs), "paired token counts")
        count = sum(p["scored_tokens4"] for p in pairs)
        nll = total("nll_sum")
        return {"nll_sum": nll, "scored_tokens": count, "mean_nll": nll/count}
    return {"count": len(pairs), **{m: total(m)/len(pairs) for m in ("acc_norm", "acc")}}


def improvement(actual, expected):
    if "mean_nll" in actual:
        return {"mean_nll": expected["mean_nll"] - actual["mean_nll"]}
    return {m: actual[m] - expected[m] for m in ("acc_norm", "acc")}


def budget_result(pairs, order, budget):
    n = len(pairs)
    require(sorted(order) == list(range(n)), "dropped/duplicated query in ranking")
    k = quota(n, budget)
    selected = set(order[:k])
    mask = [int(i in selected) for i in range(n)]
    bits = sum(8 if x else 4 for x in mask)
    require(sum(mask) == k and Fraction(bits, n) == budget, "exact average-bit invariant")
    ids = [[p["task"], p["index"]] for p in pairs]
    return {"fixed8_count": k, "fixed4_count": n-k, "total_query_bits": bits,
            "average_bits": {"numerator": bits, "denominator": n, "value": bits/n},
            "fixed8_ids": [ids[i] for i in range(n) if mask[i]],
            "fixed4_ids": [ids[i] for i in range(n) if not mask[i]],
            "metrics": aggregate(pairs, mask)}


def bootstrap(pairs, orders, cfg):
    """Same paired draws for all rules, re-ranking each multiset at exact six bits."""
    n = len(pairs)
    k = quota(n, 6)
    draws = np.random.default_rng(cfg["seed"]).integers(0, n, (cfg["draws"], n))
    metrics = ("mean_nll",) if pairs[0]["task"] == "wikitext2" else ("acc_norm", "acc")
    if metrics == ("mean_nll",):
        benefits = {"mean_nll": np.array([p["nll_sum4"]-p["nll_sum8"] for p in pairs])}
        denominators = np.array([p["scored_tokens4"] for p in pairs])[draws].sum(axis=1)
    else:
        benefits = {m: np.array([p[m+"8"]-p[m+"4"] for p in pairs]) for m in metrics}
        denominators = n
    answer = {}
    for name, order in orders.items():
        require(sorted(order) == list(range(n)), "bootstrap rank inventory")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        positions = np.argsort(rank[draws], axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(draws, positions, axis=1)
        answer[name] = {}
        for metric, benefit in benefits.items():
            delta = (benefit[selected].sum(axis=1) - .5*benefit[draws].sum(axis=1))/denominators
            answer[name][metric] = np.quantile(delta, [.025, .975], method="linear").tolist()
    return answer


def screening(oracle_delta, oracle_ci, router_delta, rules):
    require(all(math.isfinite(x) for x in (oracle_delta, *oracle_ci, router_delta)) and
            len(oracle_ci) == 2 and oracle_ci[0] <= oracle_ci[1], "ambiguous/nonfinite screening inputs")
    opportunity = (oracle_delta >= rules["material_improvement"] and
                   oracle_ci[0] > rules["interval_lower_strictly_above"])
    recovery = router_delta/oracle_delta if oracle_delta > 0 else None
    recovered = opportunity and router_delta >= 0 and recovery >= rules["minimum_router_recovery"]
    return {"material_opportunity": opportunity, "router_recovery_fraction": recovery,
            "router_predictive_rule_passed": recovered}


def decision(screens, rules):
    require(set(screens) == set(TASKS), "decision needs exactly three separate tasks")
    opportunities = [t for t in TASKS if screens[t]["material_opportunity"]]
    recovered = [t for t in opportunities if screens[t]["router_predictive_rule_passed"]]
    if len(opportunities) >= rules["minimum_opportunity_tasks"]:
        action = ("continue_separate_router_study" if len(recovered) >= rules["minimum_recovered_tasks"]
                  else "revise_signal_design")
    else:
        action = "revise_task_specific" if len(opportunities) == 1 else "stop_broad_direction"
    return {"action": action, "opportunity_tasks": opportunities, "recovered_tasks": recovered,
            "broad_direction_stopped": len(opportunities) < rules["minimum_opportunity_tasks"]}


def analyze(pairs, cfg):
    query_ids(pairs, "all analysis inputs")
    require(Counter(p["task"] for p in pairs) == {t: s["count"] for t, s in cfg["tasks"].items()},
            "analysis task counts")
    tasks, screens = {}, {}
    for task in TASKS:
        rows = sorted((p for p in pairs if p["task"] == task), key=lambda p: p["index"])
        orders = ranks(rows, cfg["deterministic_salts"])
        intervals = bootstrap(rows, orders, cfg["bootstrap"])
        budgets = {}
        for budget in cfg["budgets"]:
            k = quota(len(rows), budget)
            expected = aggregate(rows, [k/len(rows)]*len(rows))
            selections = {}
            for name, order in orders.items():
                r = budget_result(rows, order, budget)
                r["improvement_vs_expectation"] = improvement(r["metrics"], expected)
                if budget == cfg["primary_budget"]:
                    r["paired_ci95"] = intervals[name]
                selections[name] = r
            budgets[str(budget)] = {"query_independent": {"fixed8_count_per_assignment": k,
                "fixed8_probability": k/len(rows), "average_bits": budget, "metrics": expected},
                "selections": selections}
        primary = budgets[str(cfg["primary_budget"])]["selections"]
        metric = cfg["tasks"][task]["primary_metric"]
        screens[task] = screening(primary["outcome_informed"]["improvement_vs_expectation"][metric],
            intervals["outcome_informed"][metric], primary["router_score"]["improvement_vs_expectation"][metric],
            cfg["screening"])
        if task == "wikitext2":
            transitions = {"beneficial": sum(p["benefit"] > 0 for p in rows),
                           "unchanged": sum(p["benefit"] == 0 for p in rows),
                           "harmful": sum(p["benefit"] < 0 for p in rows)}
        else:
            transitions = {m: {"categories": {category: sum(p[m+"_transition"] == category for p in rows)
                for category in ("fixed4_wrong_fixed8_correct", "both_unchanged", "fixed4_correct_fixed8_wrong")},
                "both_correct": sum(p[m+"4"] == p[m+"8"] == 1 for p in rows),
                "both_wrong": sum(p[m+"4"] == p[m+"8"] == 0 for p in rows)} for m in ("acc_norm", "acc")}
        tasks[task] = {"queries": rows, "ranked_ids": {name: [rows[i]["index"] for i in order]
            for name, order in orders.items()}, "transitions": transitions, "budgets": budgets,
            "screening": screens[task]}
    return {"protocol_id": cfg["protocol_id"], "tasks": tasks,
            "decision": decision(screens, cfg["screening"])}

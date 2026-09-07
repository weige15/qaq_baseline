"""Hand-computable checks for the frozen query-budget feasibility study."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qaq.query_budget_analysis import (aggregate, aligned, analyze, bootstrap, budget_result,
    decision, exact_repeat, improvement, make_pair, query_ids, quota, ranks, router_score,
    screening, transition, validate_sample, verify_hashes)


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/"configs/query_budget_feasibility_protocol.json").read_text())


def wt_pairs():
    # Equal four-bit mean NLL=4; benefits per token [4,2,0,-2]. Unequal lengths.
    result = []
    for i, (tokens, mean8) in enumerate(zip((1, 2, 3, 4), (0, 2, 4, 6))):
        a = {"task": "wikitext2", "index": i, "scored_tokens": tokens, "nll_sum": 4*tokens}
        b = {**a, "nll_sum": mean8*tokens}
        r = {"task": "wikitext2", "index": i, "costs": [[i+1, 0, 0]], "context_tokens": [1, 2]}
        result.append(make_pair(a, b, r, [1]))
    return result


def mc_pairs(task="hellaswag"):
    # Primary benefits +1, 0, 0, -1; ordinary accuracy reverses the endpoint benefit.
    result = []
    for i, (a, b) in enumerate(((0, 1), (1, 1), (0, 0), (1, 0))):
        four = {"task": task, "index": i, "gold": 0, "prediction_norm": 1-a, "prediction": a}
        eight = {**four, "prediction_norm": 1-b, "prediction": b}
        route = {"task": task, "index": i, "costs": [[i+1, 0, 0]], "context_tokens": [1, 2]}
        result.append(make_pair(four, eight, route, [1]))
    return result


class QueryBudgetTests(unittest.TestCase):
    def test_nll_direction_negative_benefit_and_counts(self):
        pairs = wt_pairs()
        self.assertEqual([p["benefit"] for p in pairs], [4, 2, 0, -2])
        self.assertEqual(pairs[3]["nll_sum4"], 16)
        self.assertEqual(pairs[3]["nll_sum8"], 24)
        self.assertEqual(pairs[3]["scored_tokens4"], pairs[3]["scored_tokens8"])
        self.assertEqual(improvement({"mean_nll": 3}, {"mean_nll": 4}), {"mean_nll": 1})

    def test_mc_transition_categories_and_secondary_metric(self):
        for task in ("hellaswag", "arc_challenge"):
            pairs = mc_pairs(task)
            self.assertEqual([p["acc_norm_transition"] for p in pairs],
                ["fixed4_wrong_fixed8_correct", "both_unchanged", "both_unchanged", "fixed4_correct_fixed8_wrong"])
            self.assertEqual(transition(True, True), transition(False, False))
            self.assertEqual([p["benefit"] for p in pairs], [1, 0, 0, -1])
            order = ranks(pairs, [1729])["outcome_informed"]
            result = budget_result(pairs, order, 6)["metrics"]
            self.assertEqual(result, {"count": 4, "acc_norm": .75, "acc": .25})
            expected = aggregate(pairs, [.5]*4)
            self.assertEqual(improvement(result, expected), {"acc_norm": .25, "acc": -.25})

    def test_all_five_budgets_exact_and_no_dropped_or_duplicated_query(self):
        for pairs in (wt_pairs(), mc_pairs()):
            order = [2, 3, 1, 0]
            for b, k in zip((4, 5, 6, 7, 8), range(5)):
                r = budget_result(pairs, order, b)
                self.assertEqual((r["fixed8_count"], r["fixed4_count"]), (k, 4-k))
                self.assertEqual(r["total_query_bits"], 4*b)
                self.assertEqual(r["average_bits"], {"numerator": 4*b, "denominator": 4, "value": b})
                all_ids = r["fixed4_ids"] + r["fixed8_ids"]
                self.assertEqual(sorted(all_ids), [[pairs[0]["task"], i] for i in range(4)])
                self.assertEqual(len({tuple(x) for x in all_ids}), 4)
        self.assertEqual(quota(64, 6), 32)
        self.assertEqual(quota(256, 5), 64)
        for n, b in ((3, 5), (3, 6), (0, 4), (4, 9)):
            with self.assertRaises(ValueError):
                quota(n, b)
        with self.assertRaisesRegex(ValueError, "dropped/duplicated"):
            budget_result(wt_pairs(), [0, 1, 1, 3], 6)

    def test_deterministic_ties_use_task_then_integer_index_not_input_order(self):
        pairs = list(reversed(wt_pairs()))
        for p in pairs:
            p["router_score"] = p["benefit"] = 1
        for field in ("router_score", "outcome_informed", "prompt_length"):
            order = ranks(pairs, [1729])[field]
            self.assertEqual([pairs[i]["index"] for i in order], [0, 1, 2, 3])
        same_index = [mc_pairs("hellaswag")[1], mc_pairs("arc_challenge")[1]]
        self.assertEqual(ranks(same_index, [1729])["prompt_length"], [1, 0])

    def test_query_independent_expectation_matches_all_exact_quota_subsets(self):
        for pairs in (wt_pairs(), mc_pairs()):
            for b in (4, 5, 6, 7, 8):
                k = quota(4, b)
                expected = aggregate(pairs, [k/4]*4)
                outcomes = [aggregate(pairs, [int(i in subset) for i in range(4)])
                            for subset in itertools.combinations(range(4), k)]
                for key, value in expected.items():
                    self.assertAlmostEqual(sum(o[key] for o in outcomes)/len(outcomes), value)

    def test_unequal_token_counts_are_ratio_of_sums_not_mean_of_means(self):
        pairs = wt_pairs()
        order = ranks(pairs, [1729])["outcome_informed"]
        self.assertEqual(order, [0, 1, 2, 3])
        expected = aggregate(pairs, [.5]*4)
        self.assertEqual(expected, {"nll_sum": 40, "scored_tokens": 10, "mean_nll": 4})
        actual = budget_result(pairs, order, 6)["metrics"]
        self.assertEqual(actual, {"nll_sum": 32, "scored_tokens": 10, "mean_nll": 3.2})
        self.assertNotEqual(actual["mean_nll"], np.mean([0, 2, 4, 4]))
        self.assertAlmostEqual(improvement(actual, expected)["mean_nll"], .8)
        broken = copy.deepcopy(pairs)
        broken[0]["scored_tokens8"] = 2
        with self.assertRaisesRegex(ValueError, "paired token counts"):
            aggregate(broken, [.5]*4)

    def test_outcome_limit_does_not_skip_negative_benefit_to_meet_budget(self):
        pairs = wt_pairs()
        for p in pairs:
            p["benefit"] = -p["index"]-1
        r = budget_result(pairs, ranks(pairs, [1729])["outcome_informed"], 7)
        self.assertEqual(r["fixed8_ids"], [["wikitext2", 0], ["wikitext2", 1], ["wikitext2", 2]])
        self.assertEqual(budget_result(pairs, [0, 1, 2, 3], 8)["fixed8_count"], 4)

    def test_frozen_router_parameter_weighted_ranking_not_block_mean(self):
        a, b = [[10, 999, 0], [0, 999, 0]], [[0, 0, 0], [2, 0, 0]]
        self.assertEqual(router_score(a, [1, 9]), 1)
        self.assertEqual(router_score(b, [1, 9]), 1.8)
        self.assertGreater(np.mean([10, 0]), np.mean([0, 2]))  # opposite unweighted ordering
        pairs = wt_pairs()[:2]
        pairs[0]["router_score"], pairs[1]["router_score"] = router_score(a, [1, 9]), router_score(b, [1, 9])
        self.assertEqual(ranks(pairs, [1729])["router_score"], [1, 0])
        self.assertEqual(router_score([[1, 50, 3]], [4]), -2)  # never clip negative reductions
        for costs, counts in (([[1, 2]], [1]), ([[float("nan"), 0, 0]], [1]), ([[1, 0, 0]], [0])):
            with self.assertRaises(ValueError):
                router_score(costs, counts)

    def test_hash_selections_are_predeclared_and_reproducible(self):
        pairs = wt_pairs()
        for salt in CFG["deterministic_salts"]:
            expected = sorted(range(4), key=lambda i: (hashlib.sha256(f"{salt}:wikitext2:{i}".encode()).hexdigest(), i))
            self.assertEqual(ranks(pairs, CFG["deterministic_salts"])[f"id_hash_{salt}"], expected)
        self.assertEqual(CFG["deterministic_salts"], [1729, 2718, 31415])

    def test_repeated_ids_missing_queries_and_wrong_order_fail(self):
        pairs = wt_pairs()
        ids = query_ids(pairs, "fixture")
        for broken in (pairs[:-1], pairs+[pairs[0]], list(reversed(pairs))):
            with self.assertRaisesRegex(ValueError, "repeated ID|missing/extra/out-of-order"):
                aligned(broken, ids, "fixture.jsonl")
        self.assertEqual(len(query_ids(mc_pairs()+mc_pairs("arc_challenge"), "tasks")), 8)
        with self.assertRaisesRegex(ValueError, "cannot pool"):
            aggregate(mc_pairs()+mc_pairs("arc_challenge"), [.5]*8)

    def test_mismatched_repeats_and_only_declared_timing_ignored(self):
        a = [{"task": "wikitext2", "index": 1, "nll_sum": 3, "probe_seconds": 1}]
        b = copy.deepcopy(a)
        b[0]["probe_seconds"] = 2
        exact_repeat(a, b, "routes.json", ("probe_seconds",))
        for field in ("nll_sum", "index"):
            changed = copy.deepcopy(b)
            changed[0][field] += 1
            with self.assertRaisesRegex(ValueError, "mismatched repeats: routes.json"):
                exact_repeat(a, changed, "routes.json", ("probe_seconds",))
        with self.assertRaises(ValueError):
            exact_repeat(a, [], "samples.jsonl")
        with self.assertRaises(ValueError):
            exact_repeat({"peak_allocated_bytes": 1}, {"peak_allocated_bytes": 2}, "results.json",
                         CFG["baseline_timing_fields"])

    def test_missing_and_hash_invalid_paths_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"raw.json"
            p.write_bytes(b"old")
            digest = hashlib.sha256(b"old").hexdigest()
            verify_hashes(tmp, {"raw.json": digest})
            p.write_bytes(b"new")
            with self.assertRaisesRegex(ValueError, f"hash-invalid {p}"):
                verify_hashes(tmp, {"raw.json": digest})
            with self.assertRaisesRegex(ValueError, "missing input: .*missing.json"):
                verify_hashes(tmp, {"missing.json": digest})

    def test_token_loss_validation_and_pair_identity(self):
        ex = {"task": "wikitext2", "index": 1, "tokens": [1, 2, 3], "prefix_length": 1}
        row = {"task": "wikitext2", "index": 1, "token_nll": [1, 2], "scored_tokens": 2, "nll_sum": 3}
        validate_sample(row, ex, "fixture")
        with self.assertRaisesRegex(ValueError, "NLL sum"):
            validate_sample({**row, "nll_sum": 4}, ex, "fixture")
        with self.assertRaisesRegex(ValueError, "pair identity"):
            make_pair(row, {**row, "index": 2}, {"task": "wikitext2", "index": 1}, [1])

    def test_paired_resampling_hand_draws_reselect_exact_half_and_preserve_tokens(self):
        pairs = wt_pairs()
        # Draw 1 oracle upgrade NLL gain 8, total endpoint gain 0, denominator 10 => .8.
        # Draw 2 oracle upgrades both occurrences of ID0: gain8; total gain12,
        # denominator7 => 2/7. A fluctuating-membership bootstrap would differ.
        draws = np.array([[0, 1, 2, 3], [0, 0, 1, 2]])
        with patch("qaq.query_budget_analysis.np.random.default_rng") as rng:
            rng.return_value.integers.return_value = draws
            result = bootstrap(pairs, {"oracle": [0, 1, 2, 3]}, {"seed": 4242, "draws": 2})
            rng.assert_called_once_with(4242)
        np.testing.assert_allclose(result["oracle"]["mean_nll"],
                                   [2/7 + .025*(.8-2/7), 2/7 + .975*(.8-2/7)], rtol=0, atol=1e-15)
        zeros = wt_pairs()
        for p in zeros:
            p["nll_sum8"] = p["nll_sum4"]
        self.assertEqual(bootstrap(zeros, {"tie": [0, 1, 2, 3]}, {"seed": 1, "draws": 20}),
                         {"tie": {"mean_nll": [0, 0]}})

    def test_screening_exact_boundaries_negative_signal_and_ci_touching_zero(self):
        rules = CFG["screening"]
        self.assertTrue(screening(.01, [.001, .02], .0025, rules)["router_predictive_rule_passed"])
        self.assertFalse(screening(np.nextafter(.01, 0), [.001, .02], .003, rules)["material_opportunity"])
        self.assertFalse(screening(.01, [0, .02], .003, rules)["material_opportunity"])
        self.assertFalse(screening(.01, [-.001, .02], .003, rules)["material_opportunity"])
        self.assertFalse(screening(.01, [.001, .02], np.nextafter(.0025, 0), rules)["router_predictive_rule_passed"])
        negative = screening(.02, [.001, .03], -.01, rules)
        self.assertEqual(negative["router_recovery_fraction"], -.5)
        self.assertFalse(negative["router_predictive_rule_passed"])
        self.assertIsNone(screening(0, [0, 0], 0, rules)["router_recovery_fraction"])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            screening(float("nan"), [0, 1], .1, rules)

    def test_all_decisions_and_descriptive_budgets_cannot_rescue_primary(self):
        rules = CFG["screening"]
        none = screening(0, [0, 0], 0, rules)
        yes = screening(.02, [.001, .03], .01, rules)
        failed_signal = screening(.02, [.001, .03], -.01, rules)
        screens = {t: none for t in CFG["tasks"]}
        self.assertEqual(decision(screens, rules)["action"], "stop_broad_direction")
        screens["wikitext2"] = yes
        self.assertEqual(decision(screens, rules)["action"], "revise_task_specific")
        self.assertTrue(decision(screens, rules)["broad_direction_stopped"])
        screens["hellaswag"] = failed_signal
        self.assertEqual(decision(screens, rules)["action"], "revise_signal_design")
        screens["hellaswag"] = yes
        self.assertEqual(decision(screens, rules)["action"], "continue_separate_router_study")
        # Full synthetic analysis has all descriptive budgets, but screen uses ONLY six.
        cfg = copy.deepcopy(CFG)
        for spec in cfg["tasks"].values():
            spec["count"] = 4
        cfg["bootstrap"]["draws"] = 20
        result = analyze(wt_pairs()+mc_pairs()+mc_pairs("arc_challenge"), cfg)
        for task, data in result["tasks"].items():
            self.assertEqual(set(data["budgets"]), {"4", "5", "6", "7", "8"})
            primary = data["budgets"]["6"]["selections"]
            m = cfg["tasks"][task]["primary_metric"]
            self.assertEqual(data["screening"], screening(
                primary["outcome_informed"]["improvement_vs_expectation"][m],
                primary["outcome_informed"]["paired_ci95"][m],
                primary["router_score"]["improvement_vs_expectation"][m], rules))
            for b in ("4", "5", "7", "8"):
                self.assertNotIn("paired_ci95", data["budgets"][b]["selections"]["router_score"])


if __name__ == "__main__":
    unittest.main()

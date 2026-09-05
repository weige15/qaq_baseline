import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qaq.evaluation import evaluate
from qaq.model import blocks, install_replacements, prepare_replacements, set_profile
from qaq.router import (BlockRouter, check_budget, collect_context, distribution,
                        normalize_features, quota_profile, random_profile, router_costs,
                        select_features)


class RouterTests(unittest.TestCase):
    def test_solver_is_exact_and_budget_is_weighted(self):
        costs = np.random.default_rng(3).uniform(size=(6, 3))
        profile = quota_profile(costs)
        for kind in (0, 1):
            actual = sum(costs[2*i+kind, (profile[2*i+kind]-4)//2] for i in range(3))
            oracle = min(sum(costs[2*i+kind, p[i]] for i in range(3))
                         for p in itertools.permutations(range(3)))
            self.assertEqual(actual, oracle)
        self.assertEqual(check_budget(profile, [10, 100]*3)["mean_bits"], 6)
        self.assertEqual(quota_profile(np.zeros((72, 3))), quota_profile(np.zeros((72, 3))))
        with self.assertRaises(ValueError):
            check_budget([4]*6, [10, 100]*3)
        with self.assertRaises(ValueError):
            check_budget(profile, [10, 100, 11, 100, 10, 100])
        with self.assertRaises(ValueError):
            quota_profile([[float("nan")]*3]*6)
        p = random_profile([1, 2, 3])
        self.assertEqual(p, random_profile([1, 2, 3]))
        self.assertNotEqual(p, random_profile([1, 2, 4]))
        check_budget(p, [10, 100]*36)

    def test_constant_feature_mask_and_feature_reduction(self):
        raw = torch.randn(8, 72, 132)
        raw[..., -1] = np.log(128)
        x = select_features(raw, 32)
        std = x.std(0, unbiased=False)
        norm = {"x_mean": x.mean(0), "x_std": std.clamp_min(1e-5), "x_mask": std >= 1e-5,
                "y_mean": torch.zeros(72, 3), "y_std": torch.ones(72, 3)}
        x_other_length = x.clone()
        x_other_length[..., -1] = np.log(2048)
        self.assertTrue(torch.equal(normalize_features(x, norm), normalize_features(x_other_length, norm)))
        model = BlockRouter(68, 32)
        a = router_costs(model, raw, {"groups": 32, "objective": "log_nmse"}, norm)
        raw[..., -1] = np.log(28)
        b = router_costs(model, raw, {"groups": 32, "objective": "log_nmse"}, norm)
        self.assertTrue(torch.equal(a, b))
        loss = model(normalize_features(x, norm)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.w1.grad).all())

    def test_local_labels_preserve_probe_and_real_causal_routing(self):
        torch.manual_seed(1)
        config = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=64)
        config._attn_implementation = "sdpa"
        teacher = Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
        student = copy.deepcopy(teacher)
        install_replacements(prepare_replacements(student))
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        with torch.no_grad():
            before = student(torch.tensor([tokens]), use_cache=False).logits
        features, errors = collect_context(student, tokens, teacher)
        features_only, _ = collect_context(student, tokens)
        self.assertTrue(torch.equal(features, features_only))
        self.assertEqual(tuple(errors.shape), (6, 3))
        self.assertGreater(errors[:, 0].sum(), errors[:, 2].sum())
        with torch.no_grad():
            after = student(torch.tensor([tokens]), use_cache=False).logits
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(all(not b._forward_hooks for b in blocks(student)))
        prefixes, profiles = [], []
        def route(prefix):
            f, _ = collect_context(student, prefix)
            # Deterministic miniature learned network, not answer-dependent.
            costs = f[:, :3].numpy()
            p = quota_profile(costs)
            set_profile(student, p)
            prefixes.append(prefix)
            profiles.append(p)
            return p
        ex = {"task": "wikitext2", "index": 1, "tokens": tokens, "prefix_length": 4}
        changed = {**ex, "tokens": tokens[:4]+[9, 10, 11, 12]}
        mc = {"task": "hellaswag", "index": 2, "gold": 0, "choices": ["a", "bb"],
              "encoded_choices": [(tokens[:4]+[5, 6], 4), (tokens[:4]+[7, 8], 4)]}
        mc_changed = {**mc, "gold": 1, "choices": ["ccc", "dddd"],
                      "encoded_choices": [(tokens[:4]+[9], 4), (tokens[:4]+[10], 4)]}
        with tempfile.TemporaryDirectory() as d:
            for i, example in enumerate((ex, changed, mc, mc_changed)):
                evaluate(student, [example], Path(d)/f"{i}.jsonl", route)
        self.assertTrue(all(p == tokens[:4] for p in prefixes))
        self.assertEqual(distribution(profiles)["unique_profiles"], 1)
        self.assertEqual(distribution(profiles)["precision_counts"], {"4": 8, "6": 8, "8": 8})


if __name__ == "__main__":
    unittest.main()

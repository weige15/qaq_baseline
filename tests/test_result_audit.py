"""Verifier coverage: valid artifacts accepted, metric/prediction omissions rejected."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_router_results import paired_comparison, validate_samples
from qaq.evaluation import summarize


class ResultAuditTests(unittest.TestCase):
    def test_raw_metric_and_prediction_tampering_are_rejected(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/core_protocol.json").read_text())
        examples = [{"task": "wikitext2", "index": 1, "tokens": [1,2,3], "prefix_length": 1}]
        rows = [{"task": "wikitext2", "index": 1, "token_nll": [1.,2.], "nll_sum": 3., "scored_tokens": 2}]
        for task in ("hellaswag", "arc_challenge"):
            examples.append({"task": task, "index": 0, "gold": 0, "choices": ["a", "bb"],
                             "encoded_choices": [([1,2],1), ([1,3],1)]})
            rows.append({"task": task, "index": 0, "gold": 0, "token_nll": [[1.],[3.]],
                         "loglikelihoods": [-1.,-3.], "normalized_loglikelihoods": [-1.,-1.5],
                         "prediction": 0, "prediction_norm": 0})
        metrics = summarize(rows)
        validate_samples(rows, examples, metrics, config)
        changed = copy.deepcopy(rows)
        changed[1]["prediction_norm"] = 1
        with self.assertRaises(AssertionError):
            validate_samples(changed, examples, summarize(changed), config)
        missing = copy.deepcopy(metrics)
        del missing["hellaswag"]["acc"]
        with self.assertRaises(AssertionError):
            validate_samples(rows, examples, missing, config)
        changed = copy.deepcopy(rows)
        changed[0]["token_nll"][0] = 100.
        with self.assertRaises(AssertionError):
            validate_samples(changed, examples, metrics, config)

    def test_paired_bootstrap_uses_paired_indices_and_ppl_of_mean(self):
        indices = np.random.default_rng(1).integers(0, 4, (100,4))
        left, right = np.arange(4.)+1, np.arange(4.)
        self.assertEqual(paired_comparison(left, right, indices), {"delta":1., "ci95":[1.,1.]})
        self.assertEqual(paired_comparison(left, left, indices), {"delta":0., "ci95":[0.,0.]})
        ppl = paired_comparison(left, right, indices, exp=True)
        self.assertEqual(ppl["delta"], np.exp(left.mean())-np.exp(right.mean()))


if __name__ == "__main__":
    unittest.main()

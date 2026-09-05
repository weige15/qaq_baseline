import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from qaq.evaluation import clean_hellaswag, continuation_nll, evaluate, make_mc, summarize
from qaq.quantization import quantize, reconstruct


class QuantizationTests(unittest.TestCase):
    def test_all_signed_codes_and_nested_bins(self):
        q = torch.arange(-128, 128).to(torch.int8).reshape(1, 1, 256)
        scale = torch.ones(1, 1, 1)
        for b in (4, 6, 8):
            actual = reconstruct(q, scale, b, torch.float32).flatten()
            expected = torch.tensor([(v // 2**(8-b))*2**(8-b) + (2**(8-b)-1)/2
                                     for v in range(-128, 128)])
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(actual.unique()), 2**b)
        self.assertTrue(torch.equal(reconstruct(q, scale, 8), q.reshape(1, -1).half()))

    def test_independent_reference_and_group_axis(self):
        torch.manual_seed(1)
        w = torch.randn(4, 256).half()
        w[1, :128] = 0
        q, scales = quantize(w)
        expected = torch.empty_like(w)
        for row in range(4):
            for start in (0, 128):
                group = w[row, start:start+128].float()
                scale = float(group.abs().max()) / 127
                if scale == 0:
                    expected[row, start:start+128] = 0
                else:
                    expected[row, start:start+128] = (torch.round(group/scale).clamp(-127,127)*scale).half()
        self.assertTrue(torch.equal(reconstruct(q, scales, 8), expected))
        for b in (4, 6, 8):
            self.assertEqual(reconstruct(q, scales, b)[1, :128].count_nonzero(), 0)
        x = torch.randn(2, 256)
        outputs = [x @ reconstruct(q, scales, b).float().T for b in (4, 6, 8)]
        self.assertFalse(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[1], outputs[2]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"weights.pt"
            torch.save({"q": q, "scale": scales}, path)
            restored = torch.load(path, weights_only=True)
            for b in (4, 6, 8):
                self.assertTrue(torch.equal(reconstruct(q, scales, b),
                    reconstruct(restored["q"], restored["scale"], b)))

    def test_rounding_and_validation(self):
        w = torch.tensor([[127., -127., 0.5, 1.5, -0.5, -1.5, 0., 2.5]])
        q, _ = quantize(w, 8)
        self.assertEqual(q.flatten().tolist(), [127, -127, 0, 2, 0, -2, 0, 2])
        for w in (torch.empty(1, 0), torch.ones(2, 129), torch.full((2,128), float("nan"))):
            with self.assertRaises(ValueError):
                quantize(w)
        q, s = quantize(torch.zeros(2, 128))
        with self.assertRaises(ValueError):
            reconstruct(q, s, 5)


class UniformModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, use_cache):
        # Nonzero logits for observed token allow testing causal shift (not uniform).
        logits = torch.zeros(*input_ids.shape, 5)
        logits.scatter_(-1, input_ids[..., None], 2)
        return type("Output", (), {"logits": logits})()


class EvaluationTests(unittest.TestCase):
    def test_shift_and_prefix_mask(self):
        losses = continuation_nll(UniformModel(), [0, 1, 1, 2], 2)
        logz = math.log(math.exp(2)+4)
        self.assertAlmostEqual(losses[0], logz-2, places=6)
        self.assertAlmostEqual(losses[1], logz, places=6)
        self.assertEqual(len(losses), 2)

    def test_prompt_rules_and_choice_metrics(self):
        self.assertEqual(clean_hellaswag(" Foo [title] [artifact]  Bar "), "Foo.  Bar")
        row = make_mc("arc_challenge", {"question": "X?", "choices": {
            "text": ["a", "bb"], "label": ["1", "2"]}, "answerKey": "2"}, 1)
        self.assertEqual(row["context"], "Question: X?\nAnswer:")
        self.assertEqual(row["gold"], 1)
        result = summarize([{"task": "arc_challenge", "gold": 1,
                             "prediction": 0, "prediction_norm": 1}])
        self.assertEqual(set(result["arc_challenge"]),
            {"acc", "acc_norm", "acc_stderr", "acc_norm_stderr", "count"})
        self.assertEqual(result["arc_challenge"]["acc_norm"], 1)

    def test_route_sees_only_context_once_for_all_options(self):
        calls = []
        def router(context):
            calls.append(context)
            return [4, 6, 8]
        examples = [{"task": "wikitext2", "index": 0, "tokens": [0,1,2,3], "prefix_length": 2},
                    {"task": "arc_challenge", "index": 1, "gold": 0,
                     "choices": ["a", "bb"], "encoded_choices": [([0,1,2],2), ([0,1,3,4],2)]}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"samples.jsonl"
            evaluate(UniformModel(), examples, path, router)
            self.assertEqual(calls, [[0,1], [0,1]])
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(rows[0]["profile"], rows[1]["profile"])
        # Changing a scored suffix/answer is not passed to the routing function.
        examples[0]["tokens"][2:] = [4,4]
        examples[1]["encoded_choices"][0] = ([0,1,4],2)
        calls.clear()
        with tempfile.TemporaryDirectory() as d:
            evaluate(UniformModel(), examples, Path(d)/"samples.jsonl", router)
        self.assertEqual(calls, [[0,1], [0,1]])


if __name__ == "__main__":
    unittest.main()

"""Focused Phase-A tests. Phase-B implementation is forbidden until the A gate."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qaq.model import (block_parameter_counts, install_replacements, prepare_replacements, set_profile)
from qaq.quantization import apply_fixed, quantize, reconstruct
from qaq.query_budget_analysis import (aggregate, budget_result, make_pair, ranks, validate_sample)
from qaq.query_budget_signal import (ROOT, accounting, bridge_intervals, check_repeat, material,
    new_run, phase_a_decision, values, write_new)

CFG = json.loads(Path('configs/query_budget_signal_protocol.json').read_text())


def fixture():
    pairs, six = [], []
    for i, (n, m8) in enumerate(zip((1, 2, 3, 4), (0, 2, 4, 6))):
        a = {'task': 'wikitext2', 'index': i, 'scored_tokens': n, 'nll_sum': 4*n}
        b = {**a, 'nll_sum': m8*n}
        pairs.append(make_pair(a, b, {'task': 'wikitext2', 'index': i,
            'costs': [[i+1, 0, 0]], 'context_tokens': [1, 2]}, [1]))
        six.append({**a, 'nll_sum': 3*n})
    return pairs, six


class QueryBudgetSignalTests(unittest.TestCase):
    def test_six_bit_independent_arithmetic_every_signed_code_and_zero_scale(self):
        q = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(2, 1, 128)
        scale = torch.tensor([[[.25]], [[0.0]]])
        expected = torch.tensor([(math.floor(int(c)/4)*4+1.5)*float(scale[i//128,0,0])
                                 for i, c in enumerate(q.flatten())]).reshape(2,128)
        self.assertTrue(torch.equal(reconstruct(q, scale, 6, torch.float32), expected))
        # All discarded two-bit patterns in a nonzero group share the midpoint.
        full = reconstruct(q, torch.ones_like(scale), 6, torch.float32).flatten()
        for start in range(0,256,4):
            self.assertEqual(len(set(full[start:start+4].tolist())), 1)
        zeros, scales = quantize(torch.zeros(2,128))
        self.assertTrue(torch.equal(reconstruct(zeros, scales, 6), torch.zeros(2,128).half()))

    def test_apply_fixed6_nested_exact_exclusions_and_parameter_accounting(self):
        torch.manual_seed(1729)
        model = Qwen3ForCausalLM(Qwen3Config(hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            vocab_size=64, tie_word_embeddings=True)).half().eval().requires_grad_(False)
        embedding = model.model.embed_tokens.weight.clone()
        norms = {n: p.clone() for n,p in model.named_parameters() if 'norm' in n}
        replacements = prepare_replacements(model)
        modules = apply_fixed(model, 6)
        for parent, name, packed, _ in replacements:
            expected = ((torch.floor(packed.q.float()/4)*4+1.5)*packed.scale).reshape(
                getattr(parent,name).weight.shape).half()
            self.assertTrue(torch.equal(getattr(parent,name).weight, expected))
        x = torch.tensor([[1,2,3,4]])
        with torch.no_grad(): dense = model(x, use_cache=False).logits
        install_replacements(replacements)
        set_profile(model, [6]*4)
        with torch.no_grad(): nested = model(x, use_cache=False).logits
        self.assertTrue(torch.equal(dense, nested))
        self.assertTrue(torch.equal(embedding, model.model.embed_tokens.weight))
        for n,p in model.named_parameters():
            if n in norms: self.assertTrue(torch.equal(norms[n],p))
        self.assertEqual(len(modules),14)
        count = sum(m['parameters'] for m in modules)
        self.assertEqual(count, 2*(128*128*2+64*128*2+3*256*128))
        self.assertEqual(sum(block_parameter_counts(model)), count)
        reference = [{**m, 'bits': 8} for m in modules]
        cfg = {**CFG, 'quantized_parameters': count, 'quantized_projections': 14}
        with patch('qaq.query_budget_signal.read_json', return_value=reference):
            self.assertEqual(accounting(modules, cfg)['selected_weight_bits'], 6*count)
            bad = copy.deepcopy(modules);bad[0]['bits'] = 4
            with self.assertRaises(ValueError): accounting(bad, cfg)
            with self.assertRaises(ValueError): accounting(modules[:-1], cfg)

    def test_decision_all_boundaries_and_no_rounding(self):
        self.assertTrue(material(.01, [np.nextafter(0,1), .02]))
        self.assertFalse(material(np.nextafter(.01,0), [.001,.02]))
        self.assertFalse(material(.01, [0,.02]))
        self.assertFalse(material(-.01, [-.02,-.001]))
        for gain,ci in ((float('nan'),[0,1]), (.01,[1,0]), (.01,[0,float('inf')])):
            with self.assertRaises(ValueError): material(gain,ci)
        tasks = list(CFG['tasks'])
        for n, action in enumerate(('stop_endpoint_budget_direction','revise_task_specific',
                                    'continue_phase_b','continue_phase_b')):
            r = phase_a_decision({t: i<n for i,t in enumerate(tasks)})
            self.assertEqual(r['action'], action)
            self.assertEqual(r['broad_direction_stopped'], n<2)
        with self.assertRaises(ValueError): phase_a_decision({'wikitext2': True})

    def test_bootstrap_vs_fixed6_exact_occurrences_and_ratio_of_sums(self):
        pairs, six = fixture()
        draws = np.array([[0,1,2,3],[0,0,1,2]])
        orders = ranks(pairs, CFG['deterministic_salts'])
        sec = {'other': [{**r,'nll_sum': r['nll_sum']+1} for r in six]}
        with patch('qaq.query_budget_signal.np.random.default_rng') as rng:
            rng.return_value.integers.return_value = draws
            out = bridge_intervals(pairs,six,sec,orders,{'seed':4242,'draws':2})
            rng.assert_called_once_with(4242)
        for name in ('query_independent', *orders, 'other'):
            gains, expected_gains = [], []
            for draw in draws:
                ps = [pairs[i] for i in draw]
                den = sum(p['scored_tokens4'] for p in ps)
                baseline = sum(six[i]['nll_sum'] for i in draw)/den
                expectation = aggregate(ps,[.5]*4)['mean_nll']
                if name == 'query_independent': actual = expectation
                elif name == 'other': actual = sum(sec[name][i]['nll_sum'] for i in draw)/den
                else:
                    selected = sorted(range(4),key=lambda j: (orders[name].index(draw[j]),j))[:2]
                    actual = aggregate(ps,[int(j in selected) for j in range(4)])['mean_nll']
                    self.assertEqual(sum(8 if j in selected else 4 for j in range(4)),24)
                gains.append(baseline-actual);expected_gains.append(expectation-actual)
            np.testing.assert_allclose(out[name]['vs_fixed6']['mean_nll'],np.quantile(gains,[.025,.975]),atol=1e-15)
            np.testing.assert_allclose(out[name]['vs_expectation']['mean_nll'],np.quantile(expected_gains,[.025,.975]),atol=1e-15)
        self.assertLess(out['outcome_informed']['vs_fixed6']['mean_nll'][0],0)
        self.assertEqual([p['benefit'] for p in pairs], [4,2,0,-2])

    def test_mc_signs_and_same_primary_ranks_for_ordinary_accuracy(self):
        pairs, six = [], []
        for i,(a,b) in enumerate(((0,1),(1,1),(0,0),(1,0))):
            four = {'task':'hellaswag','index':i,'gold':0,'prediction_norm':1-a,'prediction':a}
            eight = {**four,'prediction_norm':1-b,'prediction':b}
            pairs.append(make_pair(four,eight,{'task':'hellaswag','index':i,'costs':[[1,0,0]],'context_tokens':[1]},[1]))
            six.append(four)
        orders = ranks(pairs,CFG['deterministic_salts'])
        draw = np.array([[0,1,2,3]])
        with patch('qaq.query_budget_signal.np.random.default_rng') as rng:
            rng.return_value.integers.return_value = draw
            out = bridge_intervals(pairs,six,{},orders,{'seed':4242,'draws':1})
        self.assertEqual(out['outcome_informed']['vs_fixed6'],{'acc_norm':[.25,.25],'acc':[-.25,-.25]})
        np.testing.assert_equal(values(six,'acc_norm'),[0,1,0,1])

    def test_exact_half_ties_hashes_and_expectation(self):
        pairs, _ = fixture()
        for p in pairs: p['benefit'] = -1
        orders = ranks(list(reversed(pairs)),CFG['deterministic_salts'])
        self.assertEqual(orders['outcome_informed'], [3,2,1,0])
        r = budget_result(list(reversed(pairs)), orders['outcome_informed'], 6)
        self.assertEqual(sorted(r['fixed8_ids']), [['wikitext2',0],['wikitext2',1]])
        self.assertEqual(r['average_bits']['value'],6)
        expected = aggregate(pairs,[.5]*4)['mean_nll']
        actuals = [aggregate(pairs,[int(i in s) for i in range(4)])['mean_nll']
                   for s in itertools.combinations(range(4),2)]
        self.assertEqual(expected,sum(actuals)/len(actuals))
        self.assertEqual(CFG['deterministic_salts'],[1729,2718,31415])
        with self.assertRaises(ValueError):budget_result(pairs,[0,1,1,3],6)
        with self.assertRaises(ValueError):budget_result(pairs[:3],[0,1,2],6)

    def test_repeat_tolerances_ids_predictions_and_token_counts(self):
        a = [{'task':'wikitext2','index':0,'nll_sum':0,'scored_tokens':1},
             {'task':'hellaswag','index':0,'gold':0,'prediction':0,'prediction_norm':0,'loglikelihoods':[0,0]}]
        self.assertTrue(check_repeat(a,a,CFG)['raw_rows_identical'])
        b = copy.deepcopy(a);b[0]['nll_sum']=1e-5;b[1]['loglikelihoods'][0]=1e-3
        self.assertFalse(check_repeat(a,b,CFG)['raw_rows_identical'])
        for field,value in [('nll_sum',np.nextafter(1e-5,1)),('scored_tokens',2),('index',1)]:
            c=copy.deepcopy(b);c[0][field]=value
            with self.assertRaises(ValueError):check_repeat(a,c,CFG)
        c=copy.deepcopy(b);c[1]['prediction_norm']=1
        with self.assertRaises(ValueError):check_repeat(a,c,CFG)
        c=copy.deepcopy(b);c[1]['loglikelihoods'][0]=np.nextafter(1e-3,1)
        with self.assertRaises(ValueError):check_repeat(a,c,CFG)

    def test_append_only_and_output_path_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'query_budget_signal_record.json'
            write_new(path,{'passed':True})
            with self.assertRaises(FileExistsError):write_new(path,{'passed':False})
            self.assertEqual(json.loads(path.read_text()),{'passed':True})
            with self.assertRaises(ValueError):new_run(Path(tmp)/'query_budget_signal_outside')
        with self.assertRaises(ValueError):new_run(ROOT)
        with self.assertRaises(ValueError):new_run(ROOT/'..'/'query_budget_signal_outside')
        with self.assertRaises(ValueError):new_run(ROOT/'not_study_specific')

    def test_protocol_frozen_constants_and_smoke_ids(self):
        self.assertEqual(CFG['bootstrap']['draws'],10000)
        self.assertEqual(CFG['bootstrap']['seed'],4242)
        self.assertEqual(CFG['cost'],{'per_job_seconds':1800,'cumulative_seconds':21600,'projection_safety_factor':2})
        self.assertEqual(CFG['smoke_ids'],[['wikitext2',5],['wikitext2',6],['hellaswag',15],['hellaswag',73],['arc_challenge',1],['arc_challenge',9]])
        self.assertEqual(CFG['phase_b']['requires'],'continue_phase_b')


if __name__ == '__main__':
    unittest.main()

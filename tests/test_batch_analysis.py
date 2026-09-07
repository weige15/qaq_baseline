import copy
import json
from pathlib import Path
import tempfile
import unittest

from qaq.batch_analysis import (analyze_grouping, batch_metrics, canonical, checked_routes,
                                concentration, decision, describe, diversity, group_requests,
                                ordered_requests, parameter_counts, sha256, verify_hashes)
from scripts.analyze_batch_interference import frozen_protocol, output_path, write_new


def requests(profiles):
    return [{'task': 'toy', 'index': i, 'profile': p} for i, p in enumerate(profiles)]


class BatchArithmeticTests(unittest.TestCase):
    def test_hand_written_unequal_parameter_counts_and_denominators(self):
        result = batch_metrics([[4, 8], [8, 4]], [1, 3])
        a, b = result['queries']
        self.assertEqual(result['maximum_profile'], [8, 8])
        self.assertEqual(result['shared_selected_bits'], 32)
        self.assertEqual([a['requested_selected_bits'], b['requested_selected_bits']], [28, 20])
        self.assertEqual([a['requested_bits_per_parameter'], b['requested_bits_per_parameter']], [7, 5])
        self.assertEqual([a['extra_selected_bits'], b['extra_selected_bits']], [4, 12])
        self.assertEqual([a['extra_bits_per_parameter'], b['extra_bits_per_parameter']], [1, 3])
        self.assertAlmostEqual(a['extra_percent'], 100 / 7)
        self.assertEqual(b['extra_percent'], 60)
        self.assertEqual(result['per_block_extra_selected_bits'], [4, 12])
        self.assertEqual(result['total_extra_selected_bits'], 16)
        self.assertEqual(result['promoted_to_8_blocks'], [0, 1])
        self.assertEqual(a['promoted_to_8_blocks'], [0])
        self.assertEqual(b['promoted_to_8_parameters'], 3)
        self.assertEqual(result['diversity']['unique_profiles'], 2)
        self.assertEqual(result['diversity']['varying_blocks'], [0, 1])
        self.assertEqual(describe([a['extra_percent'], b['extra_percent']])['median'],
                         (100 / 7 + 60) / 2)

    def test_six_to_eight_and_unanimous_eight_are_distinct(self):
        result = batch_metrics([[4, 6, 8], [8, 8, 8]], [1, 2, 3])
        self.assertEqual(result['per_block_extra_selected_bits'], [4, 4, 0])
        self.assertEqual(result['queries'][0]['requested_selected_bits'], 40)
        self.assertEqual(result['queries'][0]['extra_percent'], 20)
        self.assertEqual(result['promoted_to_8_blocks'], [0, 1])
        self.assertEqual(result['queries'][0]['promoted_to_8_parameters'], 3)
        self.assertEqual(result['queries'][1]['promoted_to_8_count'], 0)
        six = batch_metrics([[4, 6], [6, 4]], [1, 3])
        self.assertEqual(six['maximum_profile'], [6, 6])
        self.assertEqual(six['total_extra_selected_bits'], 8)
        self.assertEqual(six['promoted_to_8_count'], 0)

    def test_size_one_identity_for_every_order_and_grouping(self):
        rows = requests([[4, 8], [6, 4], [8, 6]])
        for order in ('saved', 'reverse', 'sha256'):
            ordered = ordered_requests(rows, order)
            for grouping in ('arrival', 'compatibility'):
                result = analyze_grouping(ordered, [1, 3], 1, grouping, 16)
                for i, (batch, query) in enumerate(zip(result['batches'], result['queries'])):
                    self.assertEqual(batch['maximum_profile'], ordered[i]['profile'])
                    self.assertEqual(query['arrival_position'], query['output_position'])
                    self.assertEqual(query['requested_selected_bits'], query['shared_selected_bits'])
                    for field in ('extra_selected_bits', 'extra_percent', 'promoted_to_8_count',
                                  'readiness_wait_positions', 'extra_readiness_wait_positions'):
                        self.assertEqual(query[field], 0)

    def test_identical_static_and_fixed_profiles_have_zero_inflation(self):
        for profile in ([4, 4], [6, 6], [8, 8], [4, 8]):
            for size in (1, 2, 4, 8):
                for grouping in ('arrival', 'compatibility'):
                    result = analyze_grouping(requests([profile] * 19), [1, 3], size, grouping, 16)
                    self.assertEqual(result['summary']['total_extra_selected_bits'], 0)
                    self.assertEqual(result['summary']['extra_percent']['max'], 0)
                    self.assertEqual(result['summary']['promoted_to_8_count']['max'], 0)

    def test_diversity_has_known_frequencies_and_block_counts(self):
        result = diversity([[4, 6], [4, 8], [4, 6]])
        self.assertEqual(result['unique_profiles'], 2)
        self.assertEqual(result['largest_profile_fraction'], 2 / 3)
        self.assertEqual(result['per_block_counts'], [{'4': 3, '6': 0, '8': 0}, {'4': 0, '6': 2, '8': 1}])
        self.assertEqual(result['varying_attention_blocks'], 0)
        self.assertEqual(result['varying_ffn_blocks'], 1)
        self.assertEqual(describe([4, 1, 9, 2]), {'min': 1, 'median': 3.0, 'mean': 4, 'max': 9})

    def test_invalid_profiles_counts_and_windows_fail(self):
        for profiles, counts in (([], [1]), ([[4]], []), ([[4, 8]], [1]),
                                  ([[5]], [1]), ([[4.0]], [1]), ([[True]], [1]),
                                  ([[4]], [0]), ([[4]], [-1]), ([[4]], [1.5])):
            with self.subTest(profiles=profiles, counts=counts), self.assertRaises(ValueError):
                batch_metrics(profiles, counts)
        for size, window in ((0, 16), (True, 16), (2, 3), (8, 4)):
            with self.assertRaises(ValueError):
                group_requests([[4]], [1], size, 'arrival', window)
        with self.assertRaises(ValueError):
            group_requests([[4]], [1], 1, 'other', 16)
        with self.assertRaises(ValueError):
            analyze_grouping([{'task': 'toy', 'index': 0, 'profile': [4]}] * 2, [1], 2, 'arrival', 16)


class GroupingTests(unittest.TestCase):
    def test_known_greedy_pairing_recovery_and_position_wait(self):
        profiles = [[4, 8], [8, 4], [4, 8], [8, 4]]
        self.assertEqual(group_requests(profiles, [1, 3], 2, 'compatibility', 4), [[0, 2], [1, 3]])
        a = analyze_grouping(requests(profiles), [1, 3], 2, 'arrival', 4)
        b = analyze_grouping(requests(profiles), [1, 3], 2, 'compatibility', 4)
        self.assertAlmostEqual(a['summary']['extra_percent']['median'], (100 / 7 + 60) / 2)
        self.assertEqual(b['summary']['extra_percent']['median'], 0)
        self.assertEqual([q['displacement_positions'] for q in b['queries']], [0, -1, 1, 0])
        self.assertEqual([q['readiness_wait_positions'] for q in b['queries']], [3, 1, 2, 0])
        self.assertEqual([q['extra_readiness_wait_positions'] for q in b['queries']], [2, 0, 2, 0])
        self.assertEqual(b['summary']['unique_shared_payload_bits_sum'], 48)
        self.assertEqual(a['summary']['unique_shared_payload_bits_sum'], 64)

    def test_weighted_greedy_criterion_not_unweighted_hamming(self):
        # Seed [4,4]: [8,4] costs 4, whereas [4,6] costs 20 with unequal block sizes.
        profiles = [[4, 4], [4, 6], [8, 4], [8, 8]]
        self.assertEqual(group_requests(profiles, [1, 10], 2, 'compatibility', 4)[0], [0, 2])
        self.assertEqual(group_requests([[4, 8]] * 4, [1, 3], 2, 'compatibility', 4), [[0, 1], [2, 3]])

    def test_partial_windows_coverage_determinism_and_conservation(self):
        rows = requests([[4, 8], [8, 4], [6, 6]] * 7 + [[4, 8], [8, 4]])
        for size in (1, 2, 4, 8):
            for grouping in ('arrival', 'compatibility'):
                a = analyze_grouping(rows, [1, 3], size, grouping, 16)
                b = analyze_grouping(rows, [1, 3], size, grouping, 16)
                self.assertEqual(canonical(a), canonical(b))
                self.assertEqual(sorted(q['index'] for q in a['queries']), list(range(23)))
                self.assertEqual(len(a['batches'][-1]['request_ids']), 23 % size or size)
                for q in a['queries']:
                    self.assertEqual(q['arrival_position'] // 16, q['output_position'] // 16)
                    self.assertLessEqual(abs(q['displacement_positions']), 15)
                    self.assertTrue(0 <= q['readiness_wait_positions'] <= 15)
                self.assertEqual(sum(q['extra_selected_bits'] for q in a['queries']),
                                 sum(a['summary']['per_block_extra_selected_bits']))

    def test_orders_are_profile_independent_and_keep_duplicate_profiles(self):
        rows = requests([[4, 8]] * 20)
        changed = requests([[8, 4]] * 20)
        for order in ('saved', 'reverse', 'sha256'):
            first = [r['index'] for r in ordered_requests(rows, order)]
            self.assertEqual(sorted(first), list(range(20)))
            self.assertEqual(first, [r['index'] for r in ordered_requests(changed, order)])
        self.assertEqual(ordered_requests(rows, 'saved'), rows)
        self.assertEqual(ordered_requests(rows, 'reverse'), rows[::-1])
        self.assertNotEqual(ordered_requests(rows, 'sha256'), rows)
        with self.assertRaises(ValueError):
            ordered_requests(rows, 'other')


class IntegrityTests(unittest.TestCase):
    def test_exact_route_repeats_only_ignore_probe_seconds(self):
        counts = [1, 3] * 3
        profiles = [[4, 4, 6, 6, 8, 8], [8, 8, 6, 6, 4, 4]]
        first = [{**r, 'budget': {'quantized_parameters': 12, 'selected_weight_bits': 72, 'mean_bits': 6.0},
                  'probe_seconds': 1, 'raw_features': [1.0]} for r in requests(profiles)]
        second = copy.deepcopy(first)
        second[0]['probe_seconds'] = 999
        ids, quota = [('toy', 0), ('toy', 1)], {4: 1, 6: 1, 8: 1}
        self.assertEqual(checked_routes(first, second, ids, counts, quota), requests(profiles))
        for change in ('profile', 'float_profile', 'feature', 'missing', 'duplicate', 'budget', 'nan'):
            a, b = copy.deepcopy(first), copy.deepcopy(second)
            if change == 'profile':
                b[0]['profile'] = profiles[1]
            elif change == 'float_profile':
                b[0]['profile'][0] = 4.0
            elif change == 'feature':
                b[0]['raw_features'][0] += 1e-12
            elif change == 'missing':
                b.pop()
            elif change == 'duplicate':
                a[1]['index'] = b[1]['index'] = 0
            elif change == 'budget':
                a[0]['budget']['mean_bits'] = b[0]['budget']['mean_bits'] = 7
            else:
                b[0]['probe_seconds'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_routes(a, b, ids, counts, quota)
        bad = copy.deepcopy(first)
        bad[0]['profile'] = [6] * 6
        with self.assertRaises(ValueError):
            checked_routes(bad, bad, ids, counts, quota)

    def test_actual_projection_counts_crosscheck_saved_shapes(self):
        checks, modules = [], []
        for j, names in enumerate((('q_proj', 'k_proj', 'v_proj', 'o_proj'), ('gate_proj', 'up_proj', 'down_proj'))):
            for name in names:
                checks.append({'block': j, 'projection': name, 'parameters': 6, 'exact': True})
                modules.append({'name': f"0.{'attn' if j == 0 else 'ffn'}.{name}",
                                'shape': [2, 3], 'parameters': 6, 'bits': 8})
        gate = {'passed': True, 'full_width_tensor_checks': checks, 'block_parameter_counts': [24, 18]}
        self.assertEqual(parameter_counts(gate, modules, 2), [24, 18])
        modules[0]['shape'] = [3, 3]
        with self.assertRaises(ValueError):
            parameter_counts(gate, modules, 2)

    def test_hash_missing_tamper_freeze_and_append_only_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'input'
            write_new(path, b'original')
            hashes = {'input': sha256(path)}
            verify_hashes(root, hashes)
            with self.assertRaises(FileExistsError):
                write_new(path, b'overwritten')
            self.assertEqual(path.read_bytes(), b'original')
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_hashes(root, {'absent': '0' * 64})
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_hashes(root, {'input': '0' * 64})
            for forbidden in ('results/core-v1/new', 'results/on-demand-v1/new', '../new', 'results/batch-interference-v1'):
                with self.assertRaises(ValueError):
                    output_path(root, Path(forbidden))
            self.assertTrue(output_path(root, Path('results/batch-interference-v1/new')).is_relative_to(root))
            raw = root / 'results/batch-interference-v1'
            raw.mkdir(parents=True)
            (root / 'configs').mkdir()
            (root / 'BATCH_INTERFERENCE_PROTOCOL.md').write_text('frozen')
            (root / 'configs/batch_interference_protocol.json').write_text('{"protocol_id":"toy"}')
            files = {n: sha256(root / n) for n in ('BATCH_INTERFERENCE_PROTOCOL.md', 'configs/batch_interference_protocol.json')}
            (raw / 'protocol-freeze.json').write_text(json.dumps({'files': files, 'protocol_id': 'toy'}))
            self.assertEqual(frozen_protocol(root)['protocol_id'], 'toy')
            (root / 'BATCH_INTERFERENCE_PROTOCOL.md').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                frozen_protocol(root)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def fixture():
        cfg = json.loads(Path('configs/batch_interference_protocol.json').read_text())
        div = {'unique_profiles': 10, 'largest_profile_fraction': .1,
               'varying_attention_blocks': 5, 'varying_ffn_blocks': 5}
        scenarios = []
        for size in (4, 8):
            for order in cfg['orderings']:
                for grouping in ('arrival', 'compatibility'):
                    scenarios.append({'mode': 'adaptive', 'population': 'pooled', 'batch_size': size,
                                      'ordering': order, 'grouping': grouping,
                                      'summary': {'extra_percent': {'median': 10 if grouping == 'arrival' else 5},
                                                  'window_respected': True, 'all_requests_once': True,
                                                  'absolute_displacement_positions': {'max': 15},
                                                  'readiness_wait_positions': {'max': 15},
                                                  'per_block_extra_selected_bits': [1] * 72}})
        return cfg, div, scenarios

    def test_thresholds_inclusive_and_all_orders_same_size_required(self):
        cfg, div, scenarios = self.fixture()
        self.assertEqual(decision(div, scenarios, cfg)['eligible_batch_sizes'], [4, 8])
        for r in scenarios:
            if r['grouping'] == 'arrival' and ((r['batch_size'] == 4 and r['ordering'] == 'saved')
                                              or (r['batch_size'] == 8 and r['ordering'] == 'reverse')):
                r['summary']['extra_percent']['median'] = 9.999999
        self.assertEqual(decision(div, scenarios, cfg)['recommendation'], 'stop_scheduler_direction')
        scenarios.append(copy.deepcopy(scenarios[0]))
        with self.assertRaises(ValueError):
            decision(div, scenarios, cfg)

    def test_secondary_populations_and_controls_cannot_rescue_primary(self):
        cfg, div, scenarios = self.fixture()
        extras = copy.deepcopy(scenarios)
        for r in scenarios:
            r['summary']['extra_percent']['median'] = 0
        for mode, population in (('random', 'pooled'), ('adaptive', 'wikitext2')):
            for r in copy.deepcopy(extras):
                r.update(mode=mode, population=population)
                scenarios.append(r)
        self.assertEqual(decision(div, scenarios, cfg)['eligible_batch_sizes'], [])
        cfg['orderings'] = cfg['orderings'][:2]
        with self.assertRaises(ValueError):
            decision(div, scenarios, cfg)

    def test_recovery_window_and_small_variation_failures(self):
        for cause in ('recovery', 'window', 'variation', 'dominance', 'few_blocks'):
            cfg, div, scenarios = self.fixture()
            if cause == 'variation':
                div['unique_profiles'] = 3
            elif cause == 'dominance':
                div['largest_profile_fraction'] = .91
            elif cause == 'few_blocks':
                div['varying_attention_blocks'] = 1
            else:
                for r in scenarios:
                    if r['grouping'] == 'compatibility':
                        if cause == 'recovery':
                            r['summary']['extra_percent']['median'] = 5.000001
                        else:
                            r['summary']['absolute_displacement_positions']['max'] = 16
            with self.subTest(cause=cause):
                self.assertEqual(decision(div, scenarios, cfg)['recommendation'], 'stop_scheduler_direction')

    def test_concentration_narrows_only_after_main_gate(self):
        self.assertFalse(concentration([0] * 72, 8, .5)['concentrated'])
        self.assertEqual(concentration([1] * 16, 8, .5)['top_share'], .5)
        cfg, div, scenarios = self.fixture()
        for r in scenarios:
            r['summary']['per_block_extra_selected_bits'] = [100] + [0] * 71
        result = decision(div, scenarios, cfg)
        self.assertEqual(result['recommendation'], 'recommend_separate_block_specific_gpu_experiment_only')
        div['unique_profiles'] = 1
        self.assertEqual(decision(div, scenarios, cfg)['recommendation'], 'stop_scheduler_direction')
        self.assertFalse(result['gpu_run_authorized_in_this_goal'])


if __name__ == '__main__':
    unittest.main()

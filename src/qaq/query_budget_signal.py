"""Phase-A bridge only. Closed scorer/rank rules are reused without mutation."""
from collections import Counter
import math
from pathlib import Path
import json

import numpy as np

from qaq.query_budget_analysis import (TASKS, aggregate, aligned, budget_result,
    exact_repeat, improvement, load_pairs, query_ids, ranks, read_json, read_rows,
    require, sha256, validate_sample, verify_hashes)

ROOT = Path('results/query-budget-signal-v1')
CONFIG = Path('configs/query_budget_signal_protocol.json')


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def new_run(path):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()) and path.resolve() != ROOT.resolve(),
            'output must be a new study run beneath results/query-budget-signal-v1')
    require('query_budget_signal' in path.name, 'study-specific run name required')
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_freeze():
    cfg = read_json(CONFIG)
    verify_hashes('.', read_json(ROOT/'query_budget_signal_phase_a_freeze.json')['files'])
    verify_hashes('.', {str(ROOT/'query_budget_signal_initial.json'): cfg['initial_sha256'],
                        'configs/query_budget_feasibility_protocol.json': cfg['prior_config_sha256']})
    verify_hashes('.', cfg['input_sha256'])
    return cfg


def accounting(modules, cfg):
    reference = read_json('results/core-v1/baseline-fixed8-r1/quantized_modules.json')
    require(len(modules) == cfg['quantized_projections'], 'projection count')
    require([{**m, 'bits': 8} for m in modules] == reference, 'projection names/shapes/parameters')
    require(all(m['bits'] == 6 and m['parameters'] == math.prod(m['shape']) for m in modules),
            'all projections must actually be six bits')
    count = sum(m['parameters'] for m in modules)
    require(count == cfg['quantized_parameters'], 'parameter count')
    return {'quantized_parameters': count, 'selected_weight_bits': 6*count,
            'mean_bits': 6, 'excluded_fp16_parameters': cfg['excluded_fp16_parameters']}


def check_repeat(a, b, cfg):
    aligned(b, query_ids(a, 'repeat1'), 'repeat2')
    wt_a = [r for r in a if r['task'] == 'wikitext2']
    wt_b = [r for r in b if r['task'] == 'wikitext2']
    for x, y in zip(wt_a, wt_b):
        require(x['scored_tokens'] == y['scored_tokens'], 'repeat token counts')
    mean = lambda rows: math.fsum(r['nll_sum'] for r in rows)/sum(r['scored_tokens'] for r in rows)
    nll_delta = abs(mean(wt_a)-mean(wt_b)) if wt_a else 0
    mc = [(x, y) for x, y in zip(a, b) if x['task'] != 'wikitext2']
    require(all(x['gold'] == y['gold'] and len(x['loglikelihoods']) == len(y['loglikelihoods'])
                for x, y in mc), 'repeat gold/choices')
    changed = sum(any(x[p] != y[p] for p in ('prediction', 'prediction_norm')) for x, y in mc)
    delta = max((abs(i-j) for x, y in mc for i, j in zip(x['loglikelihoods'], y['loglikelihoods'])), default=0)
    require(changed == 0 and delta <= cfg['repeat']['max_choice_logprob_delta'] and
            nll_delta <= cfg['repeat']['mean_nll_delta'], 'repeat tolerance failed')
    return {'changed_predictions': changed, 'max_choice_logprob_delta': delta,
            'mean_nll_delta': nll_delta, 'raw_rows_identical': a == b}


def material(gain, ci):
    require(len(ci) == 2 and all(math.isfinite(x) for x in (gain, *ci)) and ci[0] <= ci[1],
            'nonfinite/invalid decision input')
    return gain >= .01 and ci[0] > 0


def phase_a_decision(passes):
    require(set(passes) == set(TASKS) and all(type(v) is bool for v in passes.values()),
            'exact three task booleans required')
    tasks = [t for t in TASKS if passes[t]]
    action = ('continue_phase_b' if len(tasks) >= 2 else
              'revise_task_specific' if tasks else 'stop_endpoint_budget_direction')
    return {'action': action, 'opportunity_tasks': tasks, 'broad_direction_stopped': len(tasks) < 2}


def values(rows, metric):
    if metric == 'mean_nll':
        return np.array([r['nll_sum'] for r in rows], dtype=np.float64)
    field = 'prediction_norm' if metric == 'acc_norm' else 'prediction'
    return np.array([r[field] == r['gold'] for r in rows], dtype=np.float64)


def bridge_intervals(pairs, six, secondary, orders, cfg):
    """Paired endpoint/six draws, reranked with exact half EIGHT occurrences."""
    n = len(pairs)
    require(n > 0 and n % 2 == 0 and len(six) == n, 'exact-half paired count')
    draws = np.random.default_rng(cfg['seed']).integers(0, n, (cfg['draws'], n))
    metrics = ('mean_nll',) if pairs[0]['task'] == 'wikitext2' else ('acc_norm', 'acc')
    denom = (np.array([p['scored_tokens4'] for p in pairs])[draws].sum(1)
             if metrics == ('mean_nll',) else n)
    masks = {}
    for name, order in orders.items():
        require(sorted(order) == list(range(n)), 'invalid rank inventory')
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        positions = np.argsort(rank[draws], axis=1, kind='stable')[:, :n//2]
        masks[name] = np.take_along_axis(draws, positions, axis=1)
    result = {name: {'vs_fixed6': {}, 'vs_expectation': {}}
              for name in ('query_independent', *orders, *secondary)}
    for metric in metrics:
        field = 'nll_sum' if metric == 'mean_nll' else metric
        four = np.array([p[field+'4'] for p in pairs], dtype=np.float64)
        eight = np.array([p[field+'8'] for p in pairs], dtype=np.float64)
        base = values(six, metric)[draws].sum(1)/denom
        expectation = (.5*(four+eight))[draws].sum(1)/denom
        actual = {'query_independent': expectation}
        for name, selected in masks.items():
            actual[name] = (four[draws].sum(1)+(eight-four)[selected].sum(1))/denom
        actual.update({name: values(rows, metric)[draws].sum(1)/denom for name, rows in secondary.items()})
        sign = -1 if metric == 'mean_nll' else 1
        for name, score in actual.items():
            for ref, baseline in (('vs_fixed6', base), ('vs_expectation', expectation)):
                result[name][ref][metric] = np.quantile(sign*(score-baseline), [.025, .975], method='linear').tolist()
    return result


def validate_rows(rows, examples, label):
    aligned(rows, query_ids(examples, 'frozen'), label)
    for row, ex in zip(rows, examples):
        validate_sample(row, ex, label)


def load_fixed6(folder, examples, cfg):
    from qaq.evaluation import summarize
    result = read_json(folder/'query_budget_signal_results.json')
    command = read_json(folder/'query_budget_signal_command.json')
    rows = read_rows(folder/'query_budget_signal_samples.jsonl')
    require(result['mode'] == 'fixed6' and not result['smoke_only'] and result['sample_count'] == 576,
            'full fixed6 metadata')
    require(command['frozen_hashes'] == read_json('results/core-v1/frozen/freeze_hashes.json') and
            command['phase_a_freeze_sha256'] == sha256(ROOT/'query_budget_signal_phase_a_freeze.json'),
            'fixed6 provenance')
    require(not (folder/'query_budget_signal_failure.txt').exists(), 'failed fixed6 run')
    validate_rows(rows, examples, str(folder))
    require(all(r['profile'] is None for r in rows), 'fixed6 independent path')
    require(summarize(rows) == result['metrics'], 'fixed6 raw summary')
    require(result['accounting'] == accounting(read_json(folder/'query_budget_signal_modules.json'), cfg),
            'fixed6 accounting')
    return rows


def analyze_phase_a(cfg):
    from qaq.evaluation import summarize
    prior = read_json('configs/query_budget_feasibility_protocol.json')
    pairs = load_pairs('.', prior)
    examples = read_rows('results/core-v1/frozen/examples.jsonl')
    six_runs = [load_fixed6(ROOT/f'query_budget_signal_fixed6_r{r}', examples, cfg) for r in (1, 2)]
    repeats = {'fixed6': check_repeat(*six_runs, cfg)}
    secondary = {}
    for mode in ('adaptive', 'static', 'random'):
        runs = []
        for repeat in (1, 2):
            folder = Path(f'results/core-v1/comparison-{mode}-r{repeat}')
            rows = read_rows(folder/'samples.jsonl')
            metadata = read_json(folder/'results.json')
            command = read_json(folder/'command.json')
            routes = read_json(folder/'routes.json')
            validate_rows(rows, examples, str(folder))
            aligned(routes, query_ids(examples, 'frozen'), 'secondary routes')
            require(metadata['sample_count'] == 576 and not metadata['smoke_only'] and metadata['mode'] == mode,
                    'secondary run metadata')
            require(command['base_hashes'] == read_json('results/core-v1/frozen/freeze_hashes.json'),
                    'secondary frozen input provenance')
            require(metadata['metrics'] == summarize(rows), 'secondary raw summary')
            counts = metadata['block_parameter_counts']
            require(counts == read_json('results/core-v1/integration/gate.json')['block_parameter_counts'],
                    'secondary block inventory')
            require(metadata['scoring_budget'] == {'quantized_parameters': sum(counts),
                    'selected_weight_bits': 6*sum(counts), 'mean_bits': 6.0}, 'secondary scoring budget')
            for row, route in zip(rows, routes):
                profile = row['profile']
                require(len(profile) == 72 and all(Counter(profile[s::2]) == {4:12, 6:12, 8:12} for s in (0,1)),
                        'secondary profile quota')
                require(sum(p*b for p, b in zip(counts, profile)) == 6*sum(counts), 'secondary parameter bits')
                require(route['profile'] == profile, 'secondary route/sample profile')
            runs.append(rows)
        repeats[mode] = check_repeat(*runs, cfg)
        secondary[mode] = runs[0]
    tasks, passes = {}, {}
    for task in TASKS:
        ps = [p for p in pairs if p['task'] == task]
        six = [r for r in six_runs[0] if r['task'] == task]
        sec = {m: [r for r in rows if r['task'] == task] for m, rows in secondary.items()}
        orders = ranks(ps, cfg['deterministic_salts'])
        intervals = bridge_intervals(ps, six, sec, orders, cfg['bootstrap'])
        baseline = summarize(six)[task]
        expected = aggregate(ps, [.5]*len(ps))
        methods = {'query_independent': {'metrics': expected, 'fixed8_count_per_assignment': len(ps)//2,
                   'fixed4_count_per_assignment': len(ps)//2, 'average_bits': 6}}
        methods.update({m: budget_result(ps, order, 6) for m, order in orders.items()})
        methods.update({m: {'metrics': summarize(rows)[task], 'average_bits': 6} for m, rows in sec.items()})
        for name, entry in methods.items():
            entry['improvement_vs_fixed6'] = improvement(entry['metrics'], baseline)
            entry['improvement_vs_expectation'] = improvement(entry['metrics'], expected)
            entry['ci95'] = intervals[name]
        metric = cfg['tasks'][task]['primary_metric']
        oracle = methods['outcome_informed']
        gain = oracle['improvement_vs_fixed6'][metric]
        ci = oracle['ci95']['vs_fixed6'][metric]
        passes[task] = material(gain, ci)
        tasks[task] = {'fixed6': baseline, 'methods': methods,
            'primary_gain': gain, 'primary_ci95': ci, 'material_opportunity': passes[task],
            'queries': [{**p, 'fixed6': r, 'secondary': {m: rows[i] for m, rows in sec.items()}}
                        for i, (p, r) in enumerate(zip(ps, six))],
            'endpoint_negative_benefit_count': sum(p['benefit'] < 0 for p in ps),
            'ranked_ids': {m: [ps[i]['index'] for i in order] for m, order in orders.items()}}
    return {'protocol_id': cfg['protocol_id'], 'phase': 'A', 'repeatability': repeats,
            'tasks': tasks, 'decision': phase_a_decision(passes)}

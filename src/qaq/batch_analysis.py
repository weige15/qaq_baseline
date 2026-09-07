"""CPU-only, offline shared-maximum precision accounting. No model or scheduler.

Selected bits are logical projection-weight quantities, not throughput or latency.
All grouping choices and decision thresholds are frozen in the separate protocol.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


PRECISIONS = (4, 6, 8)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def verify_hashes(root, expected):
    """Fail closed before parsing/aggregating any required raw input."""
    for name, digest in expected.items():
        path = Path(root) / name
        if not path.is_file():
            raise ValueError(f'PAUSE: missing required input: {name}')
        if sha256(path) != digest:
            raise ValueError(f'PAUSE: recorded SHA256 mismatch: {name}')


def validate_profiles(profiles, counts):
    if not counts or any(type(c) is not int or c <= 0 for c in counts):
        raise ValueError('positive integer block parameter counts required')
    if not profiles or any(len(p) != len(counts) or any(
            type(b) is not int or b not in PRECISIONS for b in p) for p in profiles):
        raise ValueError('nonempty equal-width integer 4/6/8 profiles required')


def selected_bits(profile, counts):
    return sum(b * n for b, n in zip(profile, counts))


def checked_routes(first, second, expected_ids, counts, quota):
    """Only the recorded probe timer is excluded from exact repeat equality."""
    canonical(first)
    canonical(second)
    if len(first) != len(second) or len(first) != len(expected_ids):
        raise ValueError('route repeat/request count mismatch')
    ids = [(r['task'], r['index']) for r in first]
    if len(set(ids)) != len(ids) or ids != expected_ids:
        raise ValueError('route IDs duplicated, missing, or reordered')
    profiles = [r['profile'] for r in first]
    validate_profiles(profiles, counts)
    validate_profiles([r['profile'] for r in second], counts)
    for a, b in zip(first, second):
        if a.keys() != b.keys() or canonical({k: v for k, v in a.items() if k != 'probe_seconds'}) != canonical({
                k: v for k, v in b.items() if k != 'probe_seconds'}):
            raise ValueError('route repeat not exactly equal (except probe_seconds)')
        for kind in (0, 1):
            if Counter(a['profile'][kind::2]) != quota:
                raise ValueError('route precision quota mismatch')
        bits = selected_bits(a['profile'], counts)
        if a['budget'] != {'quantized_parameters': sum(counts),
                           'selected_weight_bits': bits, 'mean_bits': bits / sum(counts)}:
            raise ValueError('saved weighted route budget mismatch')
    return [{k: r[k] for k in ('task', 'index', 'profile')} for r in first]


def parameter_counts(gate, modules, nblocks):
    """Cross-check measured q.numel counts with independent saved weight shapes."""
    checks = gate['full_width_tensor_checks']
    counts = [0] * nblocks
    projection_counts = Counter()
    measured = {}
    for row in checks:
        j, name, n = row['block'], row['projection'], row['parameters']
        if (type(j) is not int or not 0 <= j < nblocks or type(n) is not int or n <= 0
                or row['exact'] is not True or (j, name) in measured):
            raise ValueError('invalid/duplicate integration projection count')
        measured[j, name] = n
        counts[j] += n
        projection_counts[j] += 1
    shapes = {}
    for row in modules:
        layer, kind, projection = row['name'].split('.')
        if kind not in ('attn', 'ffn'):
            raise ValueError('unknown projection kind')
        j = 2 * int(layer) + (kind == 'ffn')
        if (len(row['shape']) != 2 or any(type(n) is not int or n <= 0 for n in row['shape'])
                or math.prod(row['shape']) != row['parameters'] or row['bits'] != 8
                or (j, projection) in shapes):
            raise ValueError('invalid/duplicate saved module shape')
        shapes[j, projection] = row['parameters']
    if (not gate['passed'] or counts != gate['block_parameter_counts'] or measured != shapes
            or any(projection_counts[j] != (4 if j % 2 == 0 else 3) for j in range(nblocks))):
        raise ValueError('actual block counts/projections disagree')
    return counts


def load_inputs(root, cfg):
    verify_hashes(root, cfg['input_sha256'])

    def read(name):
        return json.loads((Path(root) / name).read_text())

    counts = parameter_counts(read('results/core-v1/integration/gate.json'),
                              read('results/core-v1/baseline-fixed8-r1/quantized_modules.json'),
                              cfg['blocks'])
    examples = [json.loads(line) for line in (Path(root) / 'results/core-v1/frozen/examples.jsonl')
                .read_text().splitlines()]
    ids = [(r['task'], r['index']) for r in examples]
    if Counter(t for t, _ in ids) != {k: v for k, v in cfg['populations'].items() if k != 'pooled'}:
        raise ValueError('frozen population mismatch')
    quota = {int(k): v for k, v in cfg['quota_per_type'].items()}
    datasets = {}
    for mode in cfg['modes']:
        repeats = []
        for repeat in (1, 2):
            base = f'results/core-v1/comparison-{mode}-r{repeat}'
            result = read(base + '/results.json')
            if (result['block_parameter_counts'] != counts or result['sample_count'] != len(ids)
                    or result['mode'] != mode or result['stage'] != 'evaluate' or result['smoke_only']):
                raise ValueError('saved result count/mode mismatch')
            repeats.append(read(base + '/routes.json'))
        datasets[mode] = checked_routes(*repeats, ids, counts, quota)
    return counts, datasets


def diversity(profiles):
    frequencies = Counter(tuple(p) for p in profiles)
    per_block = [{str(b): column.count(b) for b in PRECISIONS} for column in zip(*profiles)]
    varying = [j for j, column in enumerate(per_block) if sum(v > 0 for v in column.values()) > 1]
    return {'queries': len(profiles), 'unique_profiles': len(frequencies),
            'largest_profile_fraction': max(frequencies.values()) / len(profiles),
            'profile_frequencies': [{'profile': list(p), 'count': n} for p, n in sorted(frequencies.items())],
            'per_block_counts': per_block, 'varying_blocks': varying,
            'varying_attention_blocks': sum(j % 2 == 0 for j in varying),
            'varying_ffn_blocks': sum(j % 2 == 1 for j in varying)}


def describe(values):
    return {'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values)}


def batch_metrics(profiles, counts):
    validate_profiles(profiles, counts)
    merged = list(map(max, zip(*profiles)))
    shared = selected_bits(merged, counts)
    total = sum(counts)
    queries = []
    for p in profiles:
        requested = selected_bits(p, counts)
        extra = shared - requested
        promoted = [j for j, (b, m) in enumerate(zip(p, merged)) if b < 8 and m == 8]
        queries.append({'requested_selected_bits': requested, 'requested_bits_per_parameter': requested / total,
                        'shared_selected_bits': shared, 'shared_bits_per_parameter': shared / total,
                        'extra_selected_bits': extra, 'extra_bits_per_parameter': extra / total,
                        'extra_percent': 100 * extra / requested,
                        'promoted_to_8_blocks': promoted, 'promoted_to_8_count': len(promoted),
                        'promoted_to_8_parameters': sum(counts[j] for j in promoted)})
    contributions = [n * sum(m - p[j] for p in profiles) for j, (n, m) in enumerate(zip(counts, merged))]
    promoted = sorted({j for q in queries for j in q['promoted_to_8_blocks']})
    return {'maximum_profile': merged, 'shared_selected_bits': shared,
            'shared_bits_per_parameter': shared / total, 'diversity': diversity(profiles),
            'promoted_to_8_blocks': promoted, 'promoted_to_8_count': len(promoted),
            'total_extra_selected_bits': sum(contributions), 'per_block_extra_selected_bits': contributions,
            'queries': queries}


def ordered_requests(rows, ordering, prefix='qaq-batch-v1'):
    if ordering == 'saved':
        return list(rows)
    if ordering == 'reverse':
        return list(reversed(rows))
    if ordering == 'sha256':
        return [r for _, r in sorted(enumerate(rows), key=lambda item: (
            hashlib.sha256(f"{prefix}|{item[1]['task']}|{item[1]['index']}".encode()).hexdigest(), item[0]))]
    raise ValueError('unknown ordering')


def group_requests(profiles, counts, batch_size, grouping, window):
    validate_profiles(profiles, counts)
    if (type(batch_size) is not int or batch_size < 1 or type(window) is not int
            or window < batch_size or window % batch_size):
        raise ValueError('positive batch size must divide fixed window')
    if grouping not in ('arrival', 'compatibility'):
        raise ValueError('unknown grouping')
    n = len(profiles)
    if grouping == 'arrival' or batch_size == 1:
        return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
    batches = []
    requested = [selected_bits(p, counts) for p in profiles]
    # ponytail: quadratic search within the frozen 16-request window; no global optimizer.
    for start in range(0, n, window):
        remaining = list(range(start, min(start + window, n)))
        while remaining:
            batch = [remaining.pop(0)]
            merged = profiles[batch[0]].copy()
            request_sum = requested[batch[0]]
            while remaining and len(batch) < batch_size:
                def cost(i):
                    shared = selected_bits([max(a, b) for a, b in zip(merged, profiles[i])], counts)
                    return ((len(batch) + 1) * shared - request_sum - requested[i], i)
                chosen = min(remaining, key=cost)
                batch.append(chosen)
                remaining.remove(chosen)
                request_sum += requested[chosen]
                merged = [max(a, b) for a, b in zip(merged, profiles[chosen])]
            batches.append(batch)
    return batches


def analyze_grouping(rows, counts, batch_size, grouping, window):
    ids = [(r['task'], r['index']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate request IDs')
    profiles = [r['profile'] for r in rows]
    groups = group_requests(profiles, counts, batch_size, grouping, window)
    positions = [i for group in groups for i in group]
    if sorted(positions) != list(range(len(rows))):
        raise ValueError('grouping dropped or duplicated requests')
    batches, queries = [], []
    block_extra = [0] * len(counts)
    for batch_id, members in enumerate(groups):
        batch = batch_metrics([profiles[i] for i in members], counts)
        per_query = batch.pop('queries')
        batch.update(batch_id=batch_id, arrival_positions=members,
                     request_ids=[list(ids[i]) for i in members])
        frontier = (min((members[0] // window + 1) * window, len(rows)) - 1
                    if grouping == 'compatibility' and batch_size > 1 else max(members))
        for i, query in zip(members, per_query):
            output = len(queries)
            displacement = output - i
            wait = frontier - i
            baseline_wait = min((i // batch_size + 1) * batch_size, len(rows)) - 1 - i
            if (i // window != output // window or abs(displacement) >= window
                    or not 0 <= wait < window):
                raise ValueError('reorder/readiness window exceeded')
            query.update(task=ids[i][0], index=ids[i][1], batch_id=batch_id,
                         arrival_position=i, output_position=output,
                         displacement_positions=displacement, absolute_displacement_positions=abs(displacement),
                         readiness_frontier_position=frontier, readiness_wait_positions=wait,
                         extra_readiness_wait_positions=wait - baseline_wait)
            queries.append(query)
        block_extra = [a + b for a, b in zip(block_extra, batch['per_block_extra_selected_bits'])]
        batches.append(batch)
    query_fields = ('requested_selected_bits', 'requested_bits_per_parameter', 'shared_selected_bits',
                    'shared_bits_per_parameter', 'extra_selected_bits', 'extra_bits_per_parameter',
                    'extra_percent', 'promoted_to_8_count', 'promoted_to_8_parameters',
                    'displacement_positions', 'absolute_displacement_positions',
                    'readiness_wait_positions', 'extra_readiness_wait_positions')
    summary = {key: describe([q[key] for q in queries]) for key in query_fields}
    summary.update(queries=len(queries), batches=len(batches), all_requests_once=True, window_respected=True,
                   total_extra_selected_bits=sum(block_extra), per_block_extra_selected_bits=block_extra,
                   unique_shared_payload_bits_sum=sum(b['shared_selected_bits'] for b in batches),
                   batch_statistics={key: describe([b[key] for b in batches]) for key in
                                     ('shared_selected_bits', 'shared_bits_per_parameter', 'promoted_to_8_count')})
    summary['batch_statistics'].update(
        unique_profiles=describe([b['diversity']['unique_profiles'] for b in batches]),
        varying_blocks=describe([len(b['diversity']['varying_blocks']) for b in batches]))
    if sum(q['extra_selected_bits'] for q in queries) != sum(block_extra):
        raise ValueError('per-block excess conservation failed')
    return {'summary': summary, 'batches': batches, 'queries': queries}


def concentration(block_extra, top_n, minimum):
    order = sorted(range(len(block_extra)), key=lambda j: (-block_extra[j], j))
    total = sum(block_extra)
    top = order[:top_n]
    share = sum(block_extra[j] for j in top) / total if total else 0
    return {'total_extra_selected_bits': total, 'top_blocks': top, 'top_share': share,
            'concentrated': bool(total) and share >= minimum,
            'ranked_blocks': [{'block': j, 'extra_selected_bits': block_extra[j],
                               'share': block_extra[j] / total if total else 0} for j in order]}


def decision(profile_diversity, scenarios, cfg):
    gate = cfg['decision']
    small = (profile_diversity['unique_profiles'] < gate['min_unique_profiles']
             or profile_diversity['largest_profile_fraction'] > gate['max_largest_profile_fraction']
             or min(profile_diversity['varying_attention_blocks'], profile_diversity['varying_ffn_blocks'])
             < gate['min_varying_blocks_per_type'])
    primary = {}
    for r in scenarios:
        if r['mode'] == cfg['primary_mode'] and r['population'] == cfg['primary_population']:
            key = (r['batch_size'], r['ordering'], r['grouping'])
            if key in primary:
                raise ValueError('duplicate primary scenario')
            primary[key] = r['summary']
    if len(cfg['orderings']) < gate['minimum_orderings']:
        raise ValueError('insufficient fixed orderings')
    sizes, eligible = {}, []
    for size in gate['batch_sizes']:
        comparisons = []
        block_extra = [0] * cfg['blocks']
        for order in cfg['orderings']:
            arrival = primary[size, order, 'arrival']
            grouped = primary[size, order, 'compatibility']
            before, after = arrival['extra_percent']['median'], grouped['extra_percent']['median']
            recovery = before - after
            bounded = (grouped['window_respected'] and grouped['all_requests_once']
                       and grouped['absolute_displacement_positions']['max'] <= cfg['max_displacement_positions']
                       and grouped['readiness_wait_positions']['max'] <= cfg['max_readiness_wait_positions'])
            comparisons.append({'ordering': order, 'arrival_median_extra_percent': before,
                                'compatibility_median_extra_percent': after, 'recovery_percentage_points': recovery,
                                'excess_passed': before >= gate['min_arrival_median_extra_percent'],
                                'recovery_passed': recovery >= gate['min_recovery_percentage_points'],
                                'window_passed': bounded,
                                'concentration': concentration(arrival['per_block_extra_selected_bits'],
                                    gate['concentration_top_blocks'], gate['concentration_min_fraction'])})
            block_extra = [a + b for a, b in zip(block_extra, arrival['per_block_extra_selected_bits'])]
        passed = not small and all(all(r[k] for k in ('excess_passed', 'recovery_passed', 'window_passed'))
                                   for r in comparisons)
        sizes[str(size)] = {'passed': passed, 'orderings': comparisons,
                           'concentration': concentration(block_extra, gate['concentration_top_blocks'],
                                                          gate['concentration_min_fraction'])}
        if passed:
            eligible.append(size)
    if not eligible:
        recommendation = 'stop_scheduler_direction'
    elif any(sizes[str(s)]['concentration']['concentrated'] for s in eligible):
        recommendation = 'recommend_separate_block_specific_gpu_experiment_only'
    else:
        recommendation = 'recommend_separate_gpu_batch_experiment_only'
    return {'small_variation': small, 'eligible_batch_sizes': eligible, 'by_batch_size': sizes,
            'recommendation': recommendation, 'gpu_run_authorized_in_this_goal': False}


def analyze(counts, datasets, cfg):
    populations, scenarios = {}, []
    for mode in cfg['modes']:
        populations[mode] = {}
        for population, n in cfg['populations'].items():
            rows = [r for r in datasets[mode] if population == 'pooled' or r['task'] == population]
            if len(rows) != n:
                raise ValueError('population count mismatch')
            validate_profiles([r['profile'] for r in rows], counts)
            populations[mode][population] = diversity([r['profile'] for r in rows])
            for ordering in cfg['orderings']:
                ordered = ordered_requests(rows, ordering, cfg['ordering_hash_prefix'])
                for size in cfg['batch_sizes']:
                    for grouping in cfg['groupings']:
                        result = analyze_grouping(ordered, counts, size, grouping, cfg['window_requests'])
                        result.update(mode=mode, population=population, ordering=ordering,
                                      batch_size=size, grouping=grouping)
                        scenarios.append(result)
    return {'protocol_id': cfg['protocol_id'], 'block_parameter_counts': counts,
            'quantized_parameters': sum(counts), 'requests': datasets,
            'profile_diversity': populations, 'scenarios': scenarios,
            'decision': decision(populations[cfg['primary_mode']][cfg['primary_population']], scenarios, cfg)}


def compact_summary(analysis):
    return {**{k: v for k, v in analysis.items() if k not in ('requests', 'scenarios')},
            'scenarios': [{k: v for k, v in r.items() if k not in ('batches', 'queries')}
                          for r in analysis['scenarios']]}

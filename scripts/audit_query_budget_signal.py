"""Independent Phase-A arithmetic audit. Does NOT import any qaq/main analysis code.

Uses raw JSON and separate scalar aggregation / occurrence sorting. No model work.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path('results/query-budget-signal-v1')
CORE = Path('results/core-v1')
TASKS = ('wikitext2', 'hellaswag', 'arc_challenge')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def near(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 2e-12, (a, b)


def ids(xs):
    result = [(x['task'], x['index']) for x in xs]
    assert len(set(result)) == len(result)
    return result


def check_samples(xs, examples):
    assert ids(xs) == ids(examples)
    json.dumps(xs, allow_nan=False)
    for x, e in zip(xs, examples):
        if e['task'] == 'wikitext2':
            assert x['scored_tokens'] == len(x['token_nll']) == len(e['tokens'])-e['prefix_length']
            assert all(v >= 0 for v in x['token_nll'])
            assert x['nll_sum'] == sum(x['token_nll'])
        else:
            assert x['gold'] == e['gold']
            assert len(x['token_nll']) == len(e['choices'])
            assert [len(ls) for ls in x['token_nll']] == [len(t)-n for t,n in e['encoded_choices']]
            assert all(v >= 0 for ls in x['token_nll'] for v in ls)
            ll = [-sum(ls) for ls in x['token_nll']]
            norm = [v/len(c) for v,c in zip(ll, e['choices'])]
            assert x['loglikelihoods'] == ll and x['normalized_loglikelihoods'] == norm
            assert x['prediction'] == max(range(len(ll)), key=ll.__getitem__)
            assert x['prediction_norm'] == max(range(len(norm)), key=norm.__getitem__)


def metrics(xs):
    if xs[0]['task'] == 'wikitext2':
        total = math.fsum(x['nll_sum'] for x in xs)
        n = sum(x['scored_tokens'] for x in xs)
        return {'nll_sum': total, 'scored_tokens': n, 'mean_nll': total/n}
    return {m: sum(x[p] == x['gold'] for x in xs)/len(xs)
            for m,p in (('acc_norm','prediction_norm'),('acc','prediction'))}


def audit():
    cfg = read('configs/query_budget_signal_protocol.json')
    initial = read(ROOT/'query_budget_signal_initial.json')
    assert digest(ROOT/'query_budget_signal_initial.json') == cfg['initial_sha256']
    for name, sha in cfg['input_sha256'].items():
        assert digest(name) == sha == initial['protected_raw'][name]['sha256'], name
    seal = read(ROOT/'query_budget_signal_phase_a_freeze.json')
    for name, sha in seal['files'].items():
        assert digest(name) == sha, name
    analyses = [read(ROOT/f'query_budget_signal_analysis_r{r}/query_budget_signal_analysis.json') for r in (1,2)]
    for file in ('query_budget_signal_analysis.json', 'query_budget_signal_manifest.json'):
        assert (ROOT/'query_budget_signal_analysis_r1'/file).read_bytes() == (ROOT/'query_budget_signal_analysis_r2'/file).read_bytes()
    for r in (1,2):
        directory = ROOT/f'query_budget_signal_analysis_r{r}'
        manifest = read(directory/'query_budget_signal_manifest.json')
        assert digest(directory/'query_budget_signal_analysis.json') == manifest['analysis_sha256']
        for field in ('source_sha256','input_sha256'):
            for name, sha in manifest[field].items(): assert digest(name) == sha
    a = analyses[0]
    examples = rows(CORE/'frozen/examples.jsonl')
    assert Counter(e['task'] for e in examples) == {'wikitext2':64,'hellaswag':256,'arc_challenge':256}
    policies = {}
    raw_repeat_checks = {}
    for name in ('fixed4','fixed6','fixed8','adaptive','static','random'):
        runs = []
        for r in (1,2):
            directory = (ROOT/f'query_budget_signal_fixed6_r{r}' if name == 'fixed6' else
                         CORE/f'baseline-{name}-r{r}' if name.startswith('fixed') else CORE/f'comparison-{name}-r{r}')
            prefix = 'query_budget_signal_' if name == 'fixed6' else ''
            xs = rows(directory/f'{prefix}samples.jsonl')
            check_samples(xs,examples)
            metadata = read(directory/f'{prefix}results.json')
            assert metadata['sample_count'] == 576 and not metadata['smoke_only'] and metadata['mode'] == name
            for task in TASKS:
                for m,v in metrics([x for x in xs if x['task'] == task]).items():near(metadata['metrics'][task][m],v)
            if name.startswith('fixed'):
                assert all(x['profile'] is None for x in xs)
                mods = read(directory/f'{prefix}{"modules" if name == "fixed6" else "quantized_modules"}.json')
                ref = read(CORE/'baseline-fixed8-r1/quantized_modules.json')
                assert len(mods) == len({x['name'] for x in mods}) == 252
                assert [{**m,'bits':8} for m in mods] == ref
                assert all(m['bits'] == int(name[-1]) and m['parameters'] == math.prod(m['shape']) for m in mods)
                assert sum(m['parameters'] for m in mods) == 3633315840
            else:
                counts = metadata['block_parameter_counts']
                assert counts == read(CORE/'integration/gate.json')['block_parameter_counts']
                routes = read(directory/'routes.json')
                assert ids(routes) == ids(examples)
                for ex,x,route in zip(examples,xs,routes):
                    profile = x['profile']
                    assert profile == route['profile'] and len(profile) == 72
                    assert Counter(profile[::2]) == Counter(profile[1::2]) == {4:12,6:12,8:12}
                    assert sum(p*b for p,b in zip(counts,profile)) == 6*3633315840
                    assert route['budget']['mean_bits'] == 6 and route['budget']['quantized_parameters'] == 3633315840
                    prefix_tokens = (ex['tokens'][:128] if ex['task']=='wikitext2' else
                                     ex['encoded_choices'][0][0][:ex['encoded_choices'][0][1]])
                    assert route['context_tokens'] == prefix_tokens
                    assert route['probe_forward_input_tokens'] == len(prefix_tokens)
                command = read(directory/'command.json')
                assert command['checkpoint_sha256'] == read(CORE/'integration/gate.json')['checkpoint_sha256']
            if name == 'fixed6':
                assert b''.join((directory/f'query_budget_signal_{t}_samples.jsonl').read_bytes() for t in TASKS) == (
                    directory/'query_budget_signal_samples.jsonl').read_bytes()
            runs.append(xs)
        assert runs[0] == runs[1], name  # observed stronger property than tolerated repeat gate
        policies[name] = runs[0]
        raw_repeat_checks[name] = True
    assert all(v['raw_rows_identical'] and v['max_choice_logprob_delta']==v['mean_nll_delta']==v['changed_predictions']==0
               for v in a['repeatability'].values())
    smoke = ROOT/'query_budget_signal_smoke'
    sg = read(smoke/'query_budget_signal_integration_gate.json')
    selected = [e for e in examples if [e['task'],e['index']] in cfg['smoke_ids']]
    check_samples(rows(smoke/'query_budget_signal_samples.jsonl'), selected)
    assert ids(selected) == list(map(tuple,cfg['smoke_ids']))
    assert (smoke/'query_budget_signal_samples.jsonl').read_bytes() == (smoke/'query_budget_signal_nested_samples.jsonl').read_bytes()
    assert sg['passed'] and sg['raw_samples_byte_identical']
    assert len(sg['tensor_checks']) == len({(c['block'],c['projection']) for c in sg['tensor_checks']}) == 252
    assert all(c['exact'] for c in sg['tensor_checks'])
    assert sum(c['parameters'] for c in sg['tensor_checks']) == 3633315840
    assert sg['accounting']['selected_weight_bits'] == 6*3633315840
    counts = read(CORE/'integration/gate.json')['block_parameter_counts']
    routes = read(CORE/'comparison-adaptive-r1/routes.json')
    task_evidence, passes = {}, {}
    ci_checks = 0
    for task in TASKS:
        exs = [e for e in examples if e['task']==task]
        inputs = {p:[x for x in xs if x['task']==task] for p,xs in policies.items()}
        rr = [r for r in routes if r['task']==task]
        n = len(exs)
        wt = task=='wikitext2'
        ms = ('mean_nll',) if wt else ('acc_norm','acc')
        per = {}
        for policy,xs in inputs.items():
            per[policy] = {}
            for m in ms:
                per[policy][m] = [x['nll_sum'] if wt else int(x['prediction_norm' if m=='acc_norm' else 'prediction']==x['gold']) for x in xs]
        tokens = [x['scored_tokens'] for x in inputs['fixed4']] if wt else [1]*n
        assert all(not wt or x['scored_tokens']==t for xs in inputs.values() for x,t in zip(xs,tokens))
        benefit = [(per['fixed4']['mean_nll'][i]-per['fixed8']['mean_nll'][i])/tokens[i] if wt else
                   per['fixed8']['acc_norm'][i]-per['fixed4']['acc_norm'][i] for i in range(n)]
        router = [math.fsum(p*(c[0]-c[2]) for p,c in zip(counts,r['costs']))/sum(counts) for r in rr]
        keys = {'outcome_informed':[-b for b in benefit], 'router_score':[-x for x in router],
                'prompt_length':[-len(r['context_tokens']) for r in rr]}
        for salt in (1729,2718,31415):
            keys[f'id_hash_{salt}'] = [hashlib.sha256(f'{salt}:{task}:{e["index"]}'.encode()).hexdigest() for e in exs]
        orders = {m:sorted(range(n),key=lambda i:(key[i],task,exs[i]['index'])) for m,key in keys.items()}
        actual = a['tasks'][task]
        assert actual['endpoint_negative_benefit_count'] == sum(b<0 for b in benefit)
        assert [q['index'] for q in actual['queries']] == [e['index'] for e in exs]
        for i,q in enumerate(actual['queries']):
            assert q['fixed6'] == inputs['fixed6'][i]
            assert q['secondary'] == {m:inputs[m][i] for m in ('adaptive','static','random')}
            near(q['benefit'],benefit[i]);near(q['router_score'],router[i])
            assert q['prompt_tokens'] == len(rr[i]['context_tokens'])
            for b in (4,8):
                if wt:
                    assert q[f'nll_sum{b}'] == inputs[f'fixed{b}'][i]['nll_sum']
                    assert q[f'scored_tokens{b}'] == tokens[i]
                else:
                    for m in ms: assert q[f'{m}{b}'] == per[f'fixed{b}'][m][i]
        den = sum(tokens)
        scores = {'query_independent':{m:math.fsum(.5*(x+y) for x,y in zip(per['fixed4'][m],per['fixed8'][m]))/den for m in ms}}
        for m,order in orders.items():
            assert actual['ranked_ids'][m] == [exs[i]['index'] for i in order]
            selected = set(order[:n//2])
            rec = actual['methods'][m]
            expected8 = [[task,exs[i]['index']] for i in range(n) if i in selected]
            expected4 = [[task,exs[i]['index']] for i in range(n) if i not in selected]
            assert rec['fixed8_ids']==expected8 and rec['fixed4_ids']==expected4
            assert rec['fixed8_count']==rec['fixed4_count']==n//2
            assert rec['total_query_bits']==6*n and rec['average_bits']=={'numerator':6*n,'denominator':n,'value':6}
            scores[m] = {metric:math.fsum(per['fixed8' if i in selected else 'fixed4'][metric][i] for i in range(n))/den for metric in ms}
        for m in ('adaptive','static','random'):
            scores[m] = {metric:math.fsum(per[m][metric])/den for metric in ms}
        base = {m:math.fsum(per['fixed6'][m])/den for m in ms}
        for m,v in base.items():near(actual['fixed6'][m],v)
        for name,score in scores.items():
            for m,v in score.items():
                sign = -1 if wt else 1
                near(actual['methods'][name]['metrics'][m],v)
                near(actual['methods'][name]['improvement_vs_fixed6'][m],sign*(v-base[m]))
                near(actual['methods'][name]['improvement_vs_expectation'][m],sign*(v-scores['query_independent'][m]))
        # Independent scalar occurrence sorting in each paired draw, not main vectorized ranks.
        draws = np.random.default_rng(4242).integers(0,n,size=(10000,n))
        sampled = {name:{ref:{m:[] for m in ms} for ref in ('vs_fixed6','vs_expectation')} for name in scores}
        for draw in draws:
            draw = draw.tolist()
            denom = sum(tokens[i] for i in draw)
            selections = {name:set(sorted(range(n),key=lambda j:(key[draw[j]],task,exs[draw[j]]['index'],j))[:n//2])
                          for name,key in keys.items()}
            assert all(len(s)==n//2 and 8*len(s)+4*(n-len(s))==6*n for s in selections.values())
            for m in ms:
                four = per['fixed4'][m];eight = per['fixed8'][m]
                fixed6 = math.fsum(per['fixed6'][m][i] for i in draw)/denom
                expected = math.fsum(.5*(four[i]+eight[i]) for i in draw)/denom
                score = {'query_independent':expected}
                score.update({name:math.fsum(eight[i] if j in s else four[i] for j,i in enumerate(draw))/denom for name,s in selections.items()})
                score.update({name:math.fsum(per[name][m][i] for i in draw)/denom for name in ('adaptive','static','random')})
                for name,v in score.items():
                    sign = -1 if wt else 1
                    sampled[name]['vs_fixed6'][m].append(sign*(v-fixed6))
                    sampled[name]['vs_expectation'][m].append(sign*(v-expected))
        for name,refs in sampled.items():
            for ref,vals in refs.items():
                for m,data in vals.items():
                    # Explicit linear interpolation, not np.quantile from main analysis.
                    ordered = sorted(data)
                    ci=[]
                    for q in (.025,.975):
                        pos=9999*q;lo=math.floor(pos);frac=pos-lo
                        ci.append(ordered[lo]+frac*(ordered[lo+1]-ordered[lo]))
                    for x,y in zip(actual['methods'][name]['ci95'][ref][m],ci):near(x,y)
                    ci_checks+=1
        primary = 'mean_nll' if wt else 'acc_norm'
        gain = (base[primary]-scores['outcome_informed'][primary] if wt else scores['outcome_informed'][primary]-base[primary])
        ci = actual['methods']['outcome_informed']['ci95']['vs_fixed6'][primary]
        passes[task] = gain>=.01 and ci[0]>0
        near(actual['primary_gain'],gain)
        assert actual['primary_ci95']==ci and actual['material_opportunity']==passes[task]
        task_evidence[task]={'primary_gain':gain,'primary_ci95':ci,'material_opportunity':passes[task],
                            'query_count':n,'bootstrap_draws':10000,'negative_endpoint_benefit_count':sum(b<0 for b in benefit)}
    count=sum(passes.values())
    action='continue_phase_b' if count>=2 else 'revise_task_specific' if count==1 else 'stop_endpoint_budget_direction'
    assert a['decision']=={'action':action,'opportunity_tasks':[t for t in TASKS if passes[t]],'broad_direction_stopped':count<2}
    projection=read(ROOT/'query_budget_signal_projection.json')
    smoke_result=read(smoke/'query_budget_signal_results.json')
    projected=2*(smoke_result['setup_seconds']+sum(smoke_result['task_eval_seconds'][t]*cfg['tasks'][t]['count']/2 for t in TASKS))
    near(projection['projected_job_seconds'],projected)
    assert projected<=1800 and projection['projected_total_seconds']<=21600 and projection['pending_full_jobs']==2
    jobs=[]
    for name in ('query_budget_signal_smoke','query_budget_signal_fixed6_r1','query_budget_signal_fixed6_r2'):
        log=(ROOT/f'{name}.log').read_text()
        match=re.search(r'exit=(\d+) wall_seconds=(\d+)',log)
        assert match and int(match[1])==0 and int(match[2])<1800
        assert 'bash scripts/gpu_preflight.sh --run timeout --signal=TERM --kill-after=10s 1800s' in log
        directory=ROOT/name
        cmd=read(directory/'query_budget_signal_command.json')
        assert cmd['phase_a_freeze_sha256']==digest(ROOT/'query_budget_signal_phase_a_freeze.json')
        jobs.append({'name':name,'wall_seconds':int(match[2]),'pid':cmd['pid'],'started_unix':cmd['started_unix']})
    assert len({j['pid'] for j in jobs})==3 and sum(j['wall_seconds'] for j in jobs)<=21600
    assert all(b['started_unix']>a['started_unix']+a['wall_seconds'] for a,b in zip(jobs,jobs[1:]))
    return {'passed':True,'decision':action,'task_evidence':task_evidence,'intervals_independently_checked':ci_checks,
            'raw_repeat_checks':raw_repeat_checks,'analyses_byte_identical':True,'manifests_byte_identical':True,
            'smoke_nested_scores_byte_identical':True,'tensor_accounting_checks':252,'jobs':jobs,
            'total_gpu_job_wall_seconds':sum(j['wall_seconds'] for j in jobs),'projected_job_seconds':projected,
            'phase_b_not_authorized':count<2,'source_sha256':digest(__file__),
            'phase_a_freeze_sha256':digest(ROOT/'query_budget_signal_phase_a_freeze.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out)
    assert out.resolve().is_relative_to(ROOT.resolve()) and 'query_budget_signal' in out.name
    result=audit()
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))

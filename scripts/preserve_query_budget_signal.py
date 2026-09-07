"""Read-only completion checks; writes only one new, exclusive audit receipt."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('results/query-budget-signal-v1')
ALLOWED = {
    'QUERY_BUDGET_SIGNAL_PROTOCOL.md', 'QUERY_BUDGET_SIGNAL_REPORT.md', 'QUERY_BUDGET_SIGNAL_AUDIT.md',
    'configs/query_budget_signal_protocol.json', 'src/qaq/query_budget_signal.py',
    'scripts/run_query_budget_signal.py', 'scripts/analyze_query_budget_signal.py',
    'scripts/audit_query_budget_signal.py', 'scripts/preserve_query_budget_signal.py',
    'tests/test_query_budget_signal.py',
}
PROTECTED = ('results/core-v1', 'results/on-demand-v1', 'results/batch-interference-v1',
             'results/query-budget-feasibility-v1')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def info(p):
    stat = Path(p).stat()
    return {'sha256': sha(p), 'size': stat.st_size, 'mode': stat.st_mode, 'mtime_ns': stat.st_mtime_ns}


def numerical_report_check():
    a = read(ROOT/'query_budget_signal_analysis_r1/query_budget_signal_analysis.json')
    report = Path('QUERY_BUDGET_SIGNAL_REPORT.md').read_text()
    f = lambda x: f'{x:.8f}'
    ci = lambda xs: '['+', '.join(f(x) for x in xs)+']'
    checked = 0
    for t, data in a['tasks'].items():
        m = 'mean_nll' if t == 'wikitext2' else 'acc_norm'
        primary = f"| {t} | {len(data['queries'])} | {f(data['fixed6'][m])} | {f(data['methods']['outcome_informed']['metrics'][m])} | {f(data['primary_gain'])} | {ci(data['primary_ci95'])} | No |"
        assert primary in report, ('primary report row', t)
        section = report.split(f'### {t}: {m}\n')[1].split('\n### ')[0]
        for name, row in data['methods'].items():
            expected = f"| {name} | {f(row['metrics'][m])} | {f(row['improvement_vs_fixed6'][m])} | {ci(row['ci95']['vs_fixed6'][m])} | {f(row['improvement_vs_expectation'][m])} | {ci(row['ci95']['vs_expectation'][m])} |"
            assert expected in section, ('report comparison row', t, name)
            checked += 1
    for name in a['tasks']['hellaswag']['methods']:
        data = [a['tasks'][t]['methods'][name] for t in ('hellaswag','arc_challenge')]
        expected = '| '+name+' | '+' | '.join(f"{f(r['metrics']['acc'])} | {f(r['improvement_vs_fixed6']['acc'])} | {ci(r['ci95']['vs_fixed6']['acc'])}" for r in data)+' |'
        assert expected in report, ('secondary report row', name)
        checked += 1
    assert a['decision']['action'] == 'stop_endpoint_budget_direction'
    assert '**Completed Phase A: `stop_endpoint_budget_direction` (0/3 tasks pass).**' in report
    return {'primary_rows': 3, 'comparison_rows': checked, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert out.resolve().is_relative_to(ROOT.resolve()) and 'query_budget_signal' in out.name
    initial = read(ROOT/'query_budget_signal_initial.json')
    cfg = read('configs/query_budget_signal_protocol.json')
    assert sha(ROOT/'query_budget_signal_initial.json') == cfg['initial_sha256']
    assert not initial['status_porcelain'] and initial['historical_checks_passed']
    assert git('rev-parse','HEAD') == initial['head'] == cfg['parent_head']
    assert git('branch','--show-current') == 'research/query-budget-signal-v1'
    assert git('rev-parse','qaq-baseline-closed-v1') == initial['tag_object']
    assert git('rev-parse','qaq-baseline-closed-v1^{commit}') == initial['tag_commit']
    current_refs = dict(line.split()[::-1] for line in git('show-ref').splitlines())
    for line in initial['refs'].splitlines():
        digest, ref = line.split()
        assert current_refs.pop(ref) == digest, ref
    assert current_refs == {'refs/heads/research/query-budget-signal-v1': initial['head']}
    remote = git('ls-remote','origin','refs/heads/research/query-budget-feasibility-v1','refs/heads/research/query-budget-signal-v1')
    assert remote == initial['pushed_remote']
    assert not git('diff','--name-status') and not git('diff','--cached','--name-status')
    assert set(git('ls-files').splitlines()) == set(initial['protected_tracked'])
    untracked = set(git('ls-files','--others','--exclude-standard').splitlines())
    assert untracked == ALLOWED, untracked ^ ALLOWED
    for name, expected in initial['protected_tracked'].items():
        assert Path(name).is_file() and info(name) == expected, name
    raw = {str(p) for d in PROTECTED for p in Path(d).rglob('*') if p.is_file()}
    assert raw == set(initial['protected_raw']), raw ^ set(initial['protected_raw'])
    for name, expected in initial['protected_raw'].items():
        assert Path(name).is_file() and info(name) == expected, name
    for p in ROOT.rglob('*'):
        assert not p.is_symlink(), p
        assert 'query_budget_signal' in p.name, p
    freeze = read(ROOT/'query_budget_signal_phase_a_freeze.json')
    for name, digest in freeze['files'].items():
        assert sha(name) == digest, name
    seal_time = datetime.datetime.fromisoformat(freeze['utc']).timestamp()
    for name in ('query_budget_signal_smoke','query_budget_signal_fixed6_r1','query_budget_signal_fixed6_r2'):
        command = read(ROOT/name/'query_budget_signal_command.json')
        assert command['started_unix'] > seal_time
        assert command['runner_sha256'] == sha('scripts/run_query_budget_signal.py')
        assert command['phase_a_freeze_sha256'] == sha(ROOT/'query_budget_signal_phase_a_freeze.json')
    for stem, count in (('cpu_prefreeze',49),('cpu_final',49),('focused_final',9)):
        log=(ROOT/f'query_budget_signal_{stem}.log').read_text()
        assert f'Ran {count} tests' in log and '\nOK\n\nexit=0\n' in log
    for r in (1,2):
        assert read(ROOT/f'query_budget_signal_independent_r{r}.json')['passed']
        for kind in ('analysis','independent'):
            assert (ROOT/f'query_budget_signal_{kind}_r{r}.log').read_text().endswith('\nexit=0\n')
    assert (ROOT/'query_budget_signal_independent_r1.json').read_bytes() == (ROOT/'query_budget_signal_independent_r2.json').read_bytes()
    assert read(ROOT/'query_budget_signal_review_receipt.json')['ok']
    whitespace = []
    commands = [['git','diff','--check']]+[['git','diff','--no-index','--check','/dev/null',p] for p in sorted(ALLOWED)]
    for argv in commands:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert not result.stdout and result.returncode in ((0,1) if '--no-index' in argv else (0,)), (argv,result.stdout)
        whitespace.append({'argv':argv,'exit':result.returncode,'output':result.stdout,'passed':True})
    result = {'passed': True, 'decision': 'stop_endpoint_budget_direction',
        'protected_tracked_hash_size_mode_mtime_checks':len(initial['protected_tracked']),
        'protected_raw_hash_size_mode_mtime_checks':len(initial['protected_raw']),
        'all_protected_inventories_unchanged':True, 'initial_refs_preserved':True,
        'head':git('rev-parse','HEAD'), 'branch':git('branch','--show-current'),
        'tag_object':initial['tag_object'], 'tag_commit':initial['tag_commit'],
        'pushed_parent_rechecked':remote, 'no_staged_files':True, 'tracked_diff_empty':True,
        'status_porcelain':git('status','--porcelain=v1','--untracked-files=all'),
        'allowed_additions':sorted(ALLOWED), 'whitespace_checks':whitespace,
        'numerical_report_check':numerical_report_check(),
        'frozen_protocol_and_code_unchanged':True, 'new_files_sha256':{p:sha(p) for p in sorted(ALLOWED)},
        'raw_sha256':{str(p):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p != out and
                      not p.name.endswith('preservation_final.log')},
        'phase_b_authorized':False, 'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Artifact and preservation gate; explicit requirement coverage is in QUERY_BUDGET_SIGNAL_AUDIT.md'}
    with out.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_sha256','new_files_sha256','whitespace_checks')},indent=2))


if __name__ == '__main__':
    main()

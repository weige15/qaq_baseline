"""Deterministic, exclusive-create Phase-A analysis CLI (CPU only)."""
import argparse
import sys
from pathlib import Path

from qaq.query_budget_signal import (ROOT, analyze_phase_a, new_run, sha256,
    verify_freeze, write_new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = verify_freeze()
    out = new_run(args.out)
    result = analyze_phase_a(cfg)
    write_new(out/'query_budget_signal_analysis.json', result)
    write_new(out/'query_budget_signal_manifest.json', {
        'protocol_id': cfg['protocol_id'],
        'source_sha256': {p: sha256(p) for p in ('src/qaq/query_budget_signal.py',
            'scripts/analyze_query_budget_signal.py', 'src/qaq/query_budget_analysis.py')},
        'input_sha256': {**cfg['input_sha256'], **{str(p): sha256(p)
            for r in (1, 2) for p in sorted((ROOT/f'query_budget_signal_fixed6_r{r}').glob('*')) if p.is_file()}},
        'phase_a_freeze_sha256': sha256(ROOT/'query_budget_signal_phase_a_freeze.json'),
        'analysis_sha256': sha256(out/'query_budget_signal_analysis.json')})
    write_new(out/'query_budget_signal_command.json', {'argv': sys.argv, 'executable': sys.executable})
    print(result['decision'])
    for task, data in result['tasks'].items():
        print(task, data['primary_gain'], data['primary_ci95'], data['material_opportunity'])


if __name__ == '__main__':
    main()

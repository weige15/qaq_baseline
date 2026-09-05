# Repair Plan

## Repair Policy

Only minimal repairs traceable to `doc/review.md`; no opportunistic refactor or
new experiment. This completion pass modifies no implementation source.

## Priority Order

1. Blockers: none.
2. Major correctness issues: none.
3. Error handling/security/test gaps requiring repair: none.
4. Documentation and local evidence preservation: completed; see
   `doc/submission-check.md`.

## Fixes To Apply

No code repair is appropriate. The frozen experimental result remains unchanged.
A checksum-verified local archive addresses the pending preservation handoff;
this does not provide off-host disaster recovery.

## Fixes Explicitly Not Included

Further training, seeds, final-score search, stronger downstream static policies,
new dependencies, model/data scope expansion, loading/kernel work and a new-host
installer are outside this closed experiment.

## Verification Commands

Executed with the recorded venv, CUDA hidden and offline HF cache:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_baselines.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python results/core-v1/meta/final-artifact-audit.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src HF_HUB_OFFLINE=1 python results/completion-20260905T193817Z/independent_checks.py
python scripts/audit_table.py
git diff --check
```

All passed. The 20-test suite also passed on a clean source extraction of
`a703e66` in the same venv. No full clean-host install or new GPU run was needed.
Outputs, review findings, archive verification and restore smoke are retained
under `results/completion-20260905T193817Z/`.

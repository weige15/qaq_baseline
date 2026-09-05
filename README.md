# QAQ core reproduction — Qwen3-4B

A verified, bounded implementation of query-dependent4/6/8-bit attention and FFN
precision using one signed/scaled model checkpoint. **Not** an exact QAQ Table1
reproduction, low-bit kernel, or on-demand loader.

**Result:** adaptive routing narrowly beats our equal-bit, local-error static
policy, but **random routing and even fixed4 have better WikiText perplexity**.
Do not interpret the implemented mechanism as a general quality/efficiency win.
The preregistered experiment is closed; no further final-score-driven search.

Start with [REPLICATION_REPORT.md](REPLICATION_REPORT.md) and the requirement-by-
requirement [completion audit](COMPLETION_AUDIT.md).

## Quickstart: inspect and verify existing evidence

From this repository on the recorded host, using its existing runtime:

```bash
source ~/.venv/bin/activate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
python scripts/audit_table.py
```

Expected:20 passing tests; a passing raw-artifact audit and narrow primary
comparison. The table script checks **paper arithmetic**, not model results.

Runtime: Python3.12.3, torch2.5.1+cu121, Transformers5.16.1 and the full package
record in `results/core-v1/frozen/`. `pyproject.toml` is the historical toy
manifest, **not** a complete inference installer. A clean installation has not
been verified. Raw results, PDF and checkpoints are git-ignored and must be
preserved separately; code alone is not the evidence bundle. A checksum-verified
local archive now exists; see [preservation and restore details](doc/submission-check.md).
It is on the same filesystem, not an off-host backup.

## Code and documentation

| Path | Purpose |
|---|---|
| `src/qaq/{quantization,model}.py` | Signed nested weights, serialization, independent block profiles |
| `src/qaq/router.py` | Local teacher targets, causal features, trained MLP, exact quotas |
| `src/qaq/evaluation.py` | Frozen causal scorer, all metrics/raw losses |
| `scripts/{prepare_router,train_router,router_job}.py` | Frozen data, bounded CPU fitting, guarded GPU stages |
| `scripts/check_router_results.py` | Provenance, raw metrics, profiles/budgets, repeats, paired CIs |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), [ROUTER_PROTOCOL.md](ROUTER_PROTOCOL.md) | Preregistered settings and stopping rules |
| [BASELINE_RESULTS.md](BASELINE_RESULTS.md), [INTEGRATION_RESULTS.md](INTEGRATION_RESULTS.md), [ROUTER_RESULTS.md](ROUTER_RESULTS.md) | Stage evidence |
| [DECISIONS.md](DECISIONS.md), [PAPER_AUDIT.md](PAPER_AUDIT.md) | Choices versus paper statements and unknowns |
| [doc/onboarding.md](doc/onboarding.md), [doc/runbook.md](doc/runbook.md) | Handoff and exact commands |

## GPU safety and troubleshooting

Every GPU job must use `scripts/gpu_preflight.sh --run`, one idle process-free
GPU with at least20,000MiB free, and a30-minute timeout. Do not reuse completed
output directories, kill other processes, or bypass a refusal. The existing
results need no GPU rerun for inspection. See the runbook for missing artifacts,
runtime drift and exact-reload troubleshooting.

Historical Base-checkpoint/paper-matching/on-demand proposals in
`FIRST_GPU_RUN.md` and `configs/baseline_candidate.yaml` are explicitly superseded.
The active target is `Qwen/Qwen3-4B`, not Base;8B models/PTB/full-table/kernel work
remain out of scope.

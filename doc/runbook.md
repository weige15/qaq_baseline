# Runbook

## Project Summary

Closed, bounded Qwen/Qwen3-4B QAQ core reproduction. Start with
`REPLICATION_REPORT.md`; adaptive narrowly beats the declared local-error static
policy, but random has substantially better WT2 perplexity. No further search
against these final examples is authorized.

## Setup

All commands below run from `/nfs/home/s314511048/qaq_baseline`:

```bash
source ~/.venv/bin/activate
python --version
python -m pip show torch transformers numpy scipy pyarrow
```

Verified environment: Python3.12.3, torch2.5.1+cu121, Transformers5.16.1,
NumPy1.26.4, SciPy1.17.1, PyArrow24.0.0. Full package/CPU/RAM/GPU/driver record:
`results/core-v1/frozen/`. No packages changed in the router continuation.
`pyproject.toml` is the historical toy manifest, not a complete inference lock.
A clean-host installer has not been tested; preserve/restore the existing
runtime and artifact/cache bundle rather than guessing package upgrades.

## Run

**Existing results need only CPU auditing** (next section). These are the exact
already-executed construction commands, documented for provenance, not a request
to overwrite/rerun them:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/prepare_core.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/prepare_router.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src timeout 10m python scripts/train_router.py
```

Preparation freezes immutable source revisions/tokenization; fitting requires
`router-collect/cache.pt` and writes `router-training/`. All output collisions
fail. Preparation outputs already exist. The source/cache paths are recorded in
`frozen/model_manifest.json` and `router-frozen/data_manifest.json`.

GPU commands used this prefix **each time**, never bare `python` or only a
manually selected CUDA device:

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
 bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
 python scripts/router_job.py --stage collect \
 --out results/core-v1/router-collect-smoke --smoke
```

Exact sequence after the earlier baseline/integration gate:

| Stage/arguments after `router_job.py` | Output | Observed runtime |
|---|---|---:|
| `--stage collect ... --smoke` | `router-collect-smoke` |230s|
| `--stage collect ...` | `router-collect` |432s|
| CPU `train_router.py` | `router-training` |2.8s|
| `--stage verify ...` | `router-verify` |235s|
| `--stage evaluate --mode adaptive ... --smoke` | `comparison-adaptive-smoke` |224s|
| `--stage evaluate --mode adaptive ...` | `comparison-adaptive-r1`, `-r2` |565/570s|
| `--stage evaluate --mode static ...` | `comparison-static-r1`, `-r2` |568/571s|
| `--stage evaluate --mode random ...` | `comparison-random-r1`, `-r2` |565/570s|

In this table `...` means `--out results/core-v1/OUTPUT`, not additional options.
Commands and exact source snapshots are in every job's `command.json`/`source/`;
shell/preflight output and exits in `logs/*.log`. Smoke runtime projections were
saved before full runs. Successful model jobs use one process-free24GiB RTX3090;
router stages peaked<=13.05GB allocated. No performance-kernel claim.

## Test

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
git diff --check
```

Expected20 passing CPU tests. Exact full-model integration/genuine routed output
checks were already executed through the guard; inspect `integration/` and
`router-verify/`, do not confuse the tiny unit model with those actual GPU gates.
There is no configured lint/type-check/build/CI job for the full inference stack.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_baselines.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
python scripts/audit_table.py
```

The first reaggregates fresh fixed baselines; the second additionally audits
split hashes/tokenization, train-only normalizers/static costs, stored trained
router decisions, every final raw metric/trace/budget, exact repeats and the
frozen paired bootstrap. They update summary JSONs without changing raw runs.
The third verifies paper-table arithmetic only. No GPU is used.

Metrics: WT2 `nll_sum`, `scored_tokens`, `mean_nll`, `token_perplexity`,
`mean_window_nll_stderr`; MC `acc`, `acc_stderr`, `acc_norm`, `acc_norm_stderr`,
`count`. All retained. Passing the primary criterion does not imply superiority
to random; inspect the negative controls in `comparison-gate.json`.

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| Import error | Wrong/missing venv | `which python`; frozen package record | Activate existing environment; do not silently upgrade |
| Missing data/checkpoint | Git-ignored bundle/cache not restored | Source manifests and `results/core-v1/` | Restore exact files, verify SHA256 |
| Preflight exit3/4 | No safe GPU/availability changed | Corresponding shell log | Pause and later recheck; separate retry log |
| Output directory exists | Overwrite protection | Existing `command.json`/results | Preserve; new directory only for authorized new run |
| Frozen hash/runtime mismatch | Input/software drift | Recorded manifest versus actual file | Stop; restore exact version or declare new study |
| Reload logits differ | Lost actual rotary buffers | `doc/debug-report.md` | Preserve buffers; never relax equality |
| Noncollapsed but worse quality | Surrogate mismatch/generalization | Final raw comparisons | Report negative controls; no hidden retuning |

## Recovery Steps

1. Preserve the failed log/output directory; inspect `failure.txt` and exit code.
2. If a guard refused **before** launch, confirm no output directory was created.
3. Activate venv and run `bash scripts/gpu_preflight.sh` for a fresh report.
4. Retry only the same unfinished authorized job via `--run`, with a separate
   `*-retryN.log`. Never reset/kill GPUs or another user's process.
5. If scoring/repeat/hash gates fail, pause with the exact mismatch, rather than
   altering samples/metrics. The current experiment has no such unresolved gate.

The last random-repeat refusal is in `logs/comparison-random-r2.log`; its clean
retry is `logs/comparison-random-r2-retry1.log`. No model was launched by the
refused command. No cleanup command is required or recommended.

## Useful Commands

```bash
git status --short
find results/core-v1 -name failure.txt -print
sha256sum results/core-v1/integration/quantized_model.pt
```

Expected model hash:
`b63aeeed85e11cea5d2790b1aae068920626e5d93117548358f76ae3b2171b85`.
No `git clean`, hard reset, process kill, artifact deletion or push is part of
this runbook.

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `configs/*protocol.json` | Frozen settings | Historical YAML is superseded |
| `configs/router_data_lock.json` | Committed router split hashes | Do not recalculate to hide changes |
| `results/core-v1/frozen`, `router-frozen` | Inputs/revisions/environment | Local cached paths; git-ignored |
| `results/core-v1/integration/quantized_model.pt` | Single stored weight representation | Int8/scales plus FP16 exclusions |
| `results/core-v1/router-training/A1/router.pt` | Trained MLP/normalizers | No other attempt selected |
| `results/core-v1/{baseline,comparison}-*` | Raw metrics/losses/routes | Preserve both complete repeats |
| `results/core-v1/logs`, `meta` | Commands/refusals/tests/review/audits | Includes failures, not only passes |
| `COMPLETION_AUDIT.md` | Human requirement audit | No stage verifier substitutes |

## Operational Notes

No services, ports, credentials or remote inference endpoint. Initial model
loading is ordinary startup, not on-demand CPU/GPU weight streaming. Adaptive
and matched controls pay identical8-bit probes and6-bit scoring budgets, but
reconstruct FP16 matrices. No packed low-bit resident-memory/speed assertion.

Raw files are intentionally not Git artifacts. The616 core-v1 files plus PDF
now have a checksum-verified local archive and complete Git bundle at
`~/qaq-preservation/core-v1-a703e66-20260905T193817Z/`. Exact contents, checksums,
restore instructions and follow-up source/log snapshot:
[submission-check.md](submission-check.md). A fresh clone alone is insufficient.
The original HF cache and venv are not copied. Same-filesystem preservation is
not off-host disaster recovery; no clean-host installation was performed.

## Last Verified

- Date:2026-09-05 UTC (fresh completion/preservation audit after router continuation).
- Verified again:20-test suite in current and clean-source checkouts, table
  arithmetic, baseline/comparison/full-artifact CPU audits, exact regeneration
  of576 final examples, archive member hashes and three-file restore smoke.
  Logs: `results/completion-20260905T193817Z/`. Original GPU commands above retain
  their own raw logs; no GPU job was repeated for this follow-up.
- Known unverified commands: none presented as verified setup commands for a
  new host; clean installation/portability and restore on another host remain
  untested. No further GPU job is needed for the existing evidence audit.

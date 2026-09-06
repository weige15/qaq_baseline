# Onboarding

## What This Project Does

Provides a locally verified QAQ core mechanism for Qwen/Qwen3-4B: one nested
signed/scaled checkpoint, independent attention/FFN precisions and a small
query-dependent router. A separate completed study replays the frozen profiles
through packed two's-complement resident and synchronous on-demand storage. It
does not reproduce the paper table or provide a low-bit compute kernel. The core
quality study and storage study are both closed.

## Quickstart

On the recorded host, from the repo root:

```bash
source ~/.venv/bin/activate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_on_demand.py --stage repeats
```

Expected:24 tests pass and both evidence checks pass. The audit commands need
the ignored `results/core-v1/` and `results/on-demand-v1/` bundles. They use CPU
only. Do not interpret a code-only checkout as containing these artifacts.

## Important Files

| File/directory | Role |
|---|---|
| `REPLICATION_REPORT.md`, `COMPLETION_AUDIT.md` | Closed core findings and requirement-by-requirement evidence |
| `ON_DEMAND_PROTOCOL.md`, `ON_DEMAND_REPORT.md` | Separate packed-storage protocol and measured findings |
| `EVALUATION_PROTOCOL.md`, `ROUTER_PROTOCOL.md` | Frozen core methodology, caps and stop rules |
| `configs/core_protocol.json`, `configs/router_{protocol,data_lock}.json` | Exact settings and frozen split seal |
| `src/qaq/evaluation.py` | Custom WT2/MC scorer; not lm_eval execution |
| `src/qaq/quantization.py`, `src/qaq/model.py` | Signed int8 scales, high-bit midpoint reconstruction,72 blocks |
| `src/qaq/on_demand.py` | Eight packed planes, shared unpack/reconstruct, one-block slot |
| `src/qaq/router.py` | Features/local teacher replay,72 small MLPs, quota solver |
| `scripts/prepare_router.py`, `scripts/train_router.py`, `scripts/router_job.py` | Prepare, fit and core GPU-stage commands |
| `scripts/run_on_demand.py`, `scripts/check_on_demand*.py` | Guarded storage runner and CPU/pair gates |
| `scripts/check_{baselines,router_results}.py` | CPU core evidence and comparison audits |
| `results/core-v1/` | Frozen data, one model, router, raw scores/routes, command/source snapshots |
| `results/on-demand-v1/` | CPU gate, smoke/full pairs, transfer traces, hardware and source snapshots |
| `doc/runbook.md`, `doc/debug-report.md` | Command details, failures and rotary-buffer repair |

## Architecture Map

Frozen tokenizer/data → context-only fixed8 probe → normalized block features →
CPU MLP cost predictions → separate attention/FFN quota assignment → set real
NestedLinear precision → causal suffix/option likelihoods → all metrics/raw traces.

Training replays local FP16 and4/6/8 blocks at the same fixed8 input; only the
small router receives gradients. Static uses mean **training** local errors;
random uses one frozen seed and context hash. All matched controls pay the probe
and exactly six scoring bits. The single core model stores int8 codes/FP32 scales. The separate storage path
packs each code into eight physical planes, replays the saved profiles, and uses
either all-plane GPU residence or CPU sources plus one synchronous block slot.
Both unpack to uncached FP16 weights for the same ordinary matrix multiplication.

## Development Workflow

Do not optimize against the closed final samples. New experiments require a
separate authorized protocol/data namespace. For a correctness change, inspect
callers, update focused tests, preserve original raw evidence, and document any
required rerun rather than silently applying new code to old scores. Read the
paper/audit/decisions inputs required in `GOAL.md` before such changes.
No new packages, GPUs, model licenses or expanded scope should be inferred from
this handoff. Keep one writer and one guarded GPU job at a time.

## Testing

The quickstart suite covers24 CPU tests, including all signed codes, padded
packing, exact tiny-Qwen modes and exception release. Baseline/integration/actual
trained routing evidence is separate. `scripts/check_router_results.py` retokenizes the
selected router windows, checks no exact32-token overlap, verifies train-only
normalization/static policy, recomputes profiles/metrics/budgets and paired CIs,
and demands exact raw repeats. `scripts/check_on_demand.py --stage repeats`
rehashes paired source/hardware/profile/sample/transfer artifacts and the three
accepted hardware-matched pairs. Neither verifier replaces review of paper
claims, GPU guard logs, exclusions or the reports.

## Troubleshooting

- Missing imports: activate the existing venv. The toy `pyproject.toml` alone
  does not install the inference stack. Full versions are frozen; clean-host
  installation remains unverified.
- Missing raw/cache files: restore the preserved bundle to recorded paths; do
  not invent inputs or silently recalculate seals.
- Preflight exit3/4: no safe device or changed availability. Save the refusal,
  later recheck, and use a separate retry log; never bypass or signal processes.
- Output exists: the script intentionally refuses overwriting. Inspect existing
  artifacts. A genuinely new authorized run needs a new output directory.
- On-demand runner rejects the CPU/smoke gate: current runnable-source hashes no
  longer match the completed run snapshot. Audit existing evidence; any new
  study iteration needs new append-only CPU/smoke gates rather than bypassing it.
- Reload differs: retain actual nonpersistent rotary buffers. Do not relax exact
  equality or strip checkpoint data; see `doc/debug-report.md`.

## Documentation Freshness Checklist

- [x] README quickstart still works in the recorded environment/evidence bundle.
- [x] Run commands match the current code.
- [x] Test commands match the current code.
- [x] Important files list is still accurate.
- [x] Architecture map matches the current implementation.
- [x] Troubleshooting section includes recent known failures.

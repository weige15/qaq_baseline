# Submission Check

## Required Files

Scope: local handoff of the bounded functional goal in `GOAL.md` and the active
user request; no external submission specification, upload or PR requirement.
The exhaustive prompt-to-artifact mapping is `COMPLETION_AUDIT.md`, revalidated
against the actual current source and raw evidence on 2026-09-05 UTC.

**PASS:** all six named prerequisite files; frozen model/data/runtime/metric
setup; six fixed baseline runs; one signed/scaled quantized checkpoint; separate
attention/FFN controls; trained A1 router, disjoint train/dev data and unseen
routing evidence; six matched control runs; commands/configurations/tests/raw
losses/distributions/failures; `REPLICATION_REPORT.md`. No required experiment or
report artifact is missing. Reviews: `doc/review.md`, `doc/repair-plan.md`.

## Forbidden Files

**PASS for this local preservation scope:** no new out-of-scope model, PTB,
loading/kernel experiment, old result substitution or final-score tuning.
The archive deliberately includes raw logs, checkpoint, dataset excerpts and
failure evidence because the goal requires them. It excludes unrelated older
`results/` studies, the venv, HF cache, credential files and runtime agent logs.
No public redistribution/license audit or upload is claimed or performed.

## Clean Environment Test

**WARN:** clean source tested, not a fresh dependency installation or another
host. `git archive a703e66` was extracted to `/tmp/qaq-clean-a703e66-UPv1dO`;
all 20 tests passed with the existing `~/.venv`, CUDA hidden. Current Python and
seven relevant runtime packages independently match the frozen environment.
`pyproject.toml` is not a complete inference installer. Full clean-host
portability was not a requirement and remains unclaimed.

## Command Reproduction

**PASS:** reran documented CPU tests, `check_baselines.py`,
`check_router_results.py`, `meta/final-artifact-audit.py`, `audit_table.py` and
`git diff --check`. Extra CPU cross-check rederived every final example from
pinned data/tokenizer/seed and all 72 block sizes from actual checkpoint tensors.
No GPU experiment was repeated; retained exact job commands/source snapshots,
raw outputs, two complete repeats and all guarded launch logs were checked.

Exact commands, timestamps and exit statuses:
`results/completion-20260905T193817Z/{checks,independent-checks,clean-source-tests}.log`.
The long-tokenizer warning during full WT2 text tokenization is not a model run:
only frozen 512-token windows enter scoring. No samples or caps were changed.

## Output Format Check

**PASS:** all 576 ordered samples per run; all declared metric keys; finite token
losses; correct loss totals, normalization, predictions and counts; exact
repeats; actual trained profiles; per-query weighted budgets; source/file/tensor
hashes. Checkers were read for coverage, not accepted solely for green flags.
Tests reject tampered metrics/predictions. The final report separates paper,
implementation, findings and unresolved questions.

## Score / Metric

**PASS:** raw results reproduce adaptive-minus-static WT2 mean NLL
−0.0079655286, 95% CI [−0.0127502170, −0.0030201834], and the declared MC
guardrails. Adaptive is worse than random (+0.129041 NLL) and fixed4 (+0.070445)
on WT2. Both negative controls remain prominent. Every returned metric is in
`results/core-v1/comparison-gate.json`. No stronger quality/performance claim.

## Packaging

**PASS for local preservation; off-host backup remains unperformed.**

Directory (mode 0700):
`/nfs/home/s314511048/qaq-preservation/core-v1-a703e66-20260905T193817Z/`.

- `evidence.tar`: all 616 files under `results/core-v1/` plus `references/QAQ.pdf`;
  617 files totaling 5,651,273,427 source bytes. Every member name, size and SHA256
  was independently read back from the archive and matched to the source manifest.
- Archive SHA256: `8524fc94d2fc16f7b108904765a0e2f7a860b660b887c3470c61038a7ecffc05`.
- `manifest.json`, `files.txt`, `SHA256SUMS.files`: per-file inventory/digests.
- `repository.bundle`: complete Git history through `a703e66`; `git bundle verify`
  passed. This original commit predates the documentation-only follow-up.
- `audit-followup.tar`: current code/docs plus this continuation's CPU checks,
  reviews and preservation logs, recorded separately from immutable core evidence.
- `SHA256SUMS`: top-level archive/bundle/manifest checksums.

A three-file restore smoke (PDF, trained A1 checkpoint, comparison JSON) matched
all hashes; restored JSON parsed. A full archive extraction/new-host run was
not performed. Exact creation/verification commands and outputs:
`results/completion-20260905T193817Z/{preservation,restore-smoke}.log`.

Restore into a **new, absent** destination, never over existing results:

```bash
P="$HOME/qaq-preservation/core-v1-a703e66-20260905T193817Z"
(cd "$P" && sha256sum -c SHA256SUMS)
DEST=/path/to/new/qaq-restored
git clone "$P/repository.bundle" "$DEST"
tar -xf "$P/evidence.tar" -C "$DEST"
tar -xf "$P/audit-followup.tar" -C "$DEST"
(cd "$DEST" && sha256sum -c "$P/SHA256SUMS.files")
```

These are restore instructions, not a claim that this exact full sequence has
been executed on another host. The original pinned HF model/data cache and venv
are **not** copied; raw evidence survives without them, but replay/data-source
auditing requires restoring those caches or retrieving the recorded revisions.
Both copies reside on the same NFS filesystem: protection against checkout
loss, not disk failure. No remote backup destination was supplied.

## Final Checklist

| Gate | Status | Evidence | Action Needed |
|---|---|---|---|
| Required files | PASS | Full current prompt checklist and actual files | None |
| Forbidden files | PASS | Local archive inventory and executed scope | Review terms separately before any public upload |
| Clean environment | WARN | 20 clean-source tests; same frozen venv | New-host installation only if separately requested |
| Command reproduction | PASS | CPU audit logs, original guarded jobs/repeats | None for this bounded goal |
| Output format | PASS | Raw sample/metric/profile/hash recomputation | None |
| Score / metric | PASS | Recomputed matched comparisons, negative controls retained | Stop; no further tuning |
| Packaging | PASS | 617 member hashes, bundle verification, restore smoke | Off-host backup optional; requires destination |

**READY WITH WARNINGS** for local handoff. The bounded functional objective is
complete; warnings are disclosed nonrequirements, not missing experimental work.
No push, PR or public submission was performed. The runtime exposes no
`update_goal` tool in this session, so accounting status cannot be changed here;
creating a replacement goal would not preserve the existing usage accounting.

# Code Review

## Scope Reviewed

Completion review of `a703e66`, not a new implementation diff. Read the six
required inputs, protocols, `REPLICATION_REPORT.md`, the full completion
checklist, source/scorer/quantization/router/integration code, tests and actual
`results/core-v1/` artifacts. Two independent read-only reviewers examined method
correctness and report/evidence coverage. Current logs and their full findings:
`results/completion-20260905T193817Z/`.

## Correctness

No blocking defect found. Recomputed all metrics, paired comparisons and exact
repeats. Rehashed the single checkpoint and all 651 state tensors; inspected
actual full8/lower-width/independent-block logits and original equality checks.
Independently rederived all 576 final examples and all 72 block parameter counts.
The modest adaptive/static benefit and worse random/fixed4 controls match the
report. This is functional completion, not general quality superiority.

## Edge Cases

Tests cover all signed codes, zero groups, rounding, invalid precision, causal
prefix boundaries and constant-feature masking. Real routed-weight equality
covers 304 observed projection/precision combinations, not every possible
combination. Three dev contexts provide the routed-versus-endpoint logit check;
final execution traces cover all projections. This coverage supports the bounded
mechanism claim, not universal behavior.

## Error Handling

Both GPU refusals and the earlier rotary reload repair remain recorded. Current
CPU verifiers passed without relaxed thresholds. Output-collision refusal
preserves completed experiments. Frozen-input/runtime mismatches must still stop
future replay rather than silently update the protocol.

## Concurrency

Single GPU per original job; guard/log audit covers all 20 successful jobs and
both refusals. No GPU job, process signaling or model fitting in this review.
The guard is a point-in-time availability check, not a cluster reservation.

## Performance

No kernel or physical low-bit memory claim. The selected precision materializes
FP16 weights; matched controls also pay the same fixed8 feature probe.

## Security

Checkpoint reads use `weights_only=True`; this is a local artifact review, not a
security certification. No upload or credentials were needed. Preservation is
in a private local directory, not a public release of weights/data.

## Readability

The final report separates paper statements, choices, findings and limitations.
Historical Base/checkpoint/table plans are explicitly superseded. Follow-up
changes are documentation and preservation records only.

## Test Coverage

All 20 tests pass in the current checkout and an extracted clean `a703e66`
source tree using the same venv (six tests concern the historical toy). Baseline,
comparison and final-artifact audits also pass. Tests alone do not establish
benchmark benefit, training-data separation, guarded execution or report honesty;
those were checked separately against source and raw artifacts.

## Unnecessary Complexity

No refactor, extra training attempt, dependency or experiment is warranted.

## Review Summary

Findings: **0 blocker, 0 major, 0 minor, 0 nit**. Both reviewers: OK with notes.
Nonblocking limits: imperfect decontamination, local-error surrogate/generalization,
bounded probes, same-venv rather than clean-host validation, and same-filesystem
rather than off-host preservation. These are disclosed, not repaired by tuning.

## Do Not Fix Yet

Do not search for a stronger final score, change the frozen quantizer/router,
expand datasets/models, build kernels or claim complete host portability. A new
study or external backup destination requires separate user authorization.

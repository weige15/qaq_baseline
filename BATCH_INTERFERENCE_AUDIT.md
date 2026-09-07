# Batch-interference requirement-to-evidence audit

## Concrete completion contract

Determine whether the closed Qwen3-4B adaptive routes justify a separately
scoped GPU batching experiment, using a frozen CPU-only shared-maximum analysis,
without implementing a scheduler or touching the closed studies. Deliver the
seven permitted files, complete append-only raw measurements, tested arithmetic,
repeat/input integrity, the threshold-based decision, a qualified report and
this requirement-level audit. A negative result is a valid completion outcome.

**Status: complete at the artifact/requirement level.** Analysis, independent
verification and preservation/whitespace checks pass. No GPU experiment is
recommended: both primary thresholds fail at sizes 4 and 8 in every ordering.
No required research measurement is substituted with a test count or a
verifier's self-reported status.

Raw paths below are relative to `results/batch-interference-v1/` unless written
in full. Boundary evidence: `preservation-reviewed.json`, then
`final-preservation.json` sealing the finalized report and this audit.

## 1. Prompt-to-artifact checklist

| ID | Explicit requirement | Concrete inspected evidence / coverage |
|---|---|---|
| S1 | Tag and every file at it immutable; record HEAD/status/hashes before editing | `preflight/initial-state.json`: HEAD/tag `b59adbcffc90b870f014d22e142923d569596986`, empty initial status/diff; SHA256, mode and mtime of all 59 tracked files. Snapshot SHA256 `d2a46063e9809b9064b9312e6be2dfb8d07ca3a0628373a3ae0513bed7a74e0e`. Created before protocol/source additions. |
| S2 | Hash QAQ implementation and closed reports | The same snapshot records all `src/qaq/*.py`, every closed report/results/audit and every other tracked path; initial command output also printed implementation/report hashes. Final checker compares content, modes and mtimes, not just `git status`. |
| S3 | Do not modify/rename/delete/regenerate/overwrite existing tracked files or anything under core/on-demand raw roots | Snapshot includes all 1,144 protected raw files with SHA256, size and mtime. `preservation_audit_v2.py` compares exact inventories plus hashes/mtimes and tracked diff to tag. All checks pass; see final seal below. No cleanup path exists. |
| S4 | Only the seven permitted source-controlled additions | Exact whitelist enforced by final checker against `git ls-files --others --exclude-standard`; see deliverable table below. Additions intentionally remain untracked/uncommitted; no staging required. |
| S5 | New retained raw output only append-only under `results/batch-interference-v1` | Distinct preflight/run-1/run-2/audit/check directories; `write_new` uses `xb`, run directories use `exist_ok=False`; CLI rejects out-of-tree and already-existing paths. Unit test verifies failed overwrite leaves bytes intact. All raw logs/check scripts retained at this root. |
| S6 | No commit/push/merge/switch/history rewrite | Commands used Git only for read-only status/diff/hash information. Final checker requires original HEAD/tag, empty staged diff and zero tracked diff to closed tag. |
| S7 | Compare to tag at completion; pause on any other changed path, no destructive cleanup | `preservation_audit_v2.py --out ...` checks both tracked diff and untracked whitelist, plus protected ignored raw trees. Any mismatch is recorded, exits nonzero and says PAUSE; never restores or deletes. Passed with empty issues/diff. |
| I1 | Read README.md, GOAL.md, REPLICATION_REPORT.md, ROUTER_RESULTS.md, ON_DEMAND_REPORT.md before changes | All five read completely in the initial tool transcript; their hashes appear in initial snapshot. Read-only understanding includes prior negative quality controls and closed search. |
| I2 | Read configs/router_protocol.json and configs/on_demand_protocol.json before changes | Both read completely before writes. Original quotas, dataset/revision/hashes and frozen decisions inherited unchanged. Their hashes are locked in new config and rechecked each analysis run. |
| I3 | Read src/qaq/model.py, src/qaq/router.py, src/qaq/on_demand.py before changes | All three read completely before writes. Block order/count source, causal profiles, selected-bit budgets and physical-vs-logical distinction traced; no edits to any. |
| I4 | Read scripts/router_job.py and scripts/check_router_results.py before changes | Both read completely before writes. Saved route schema/count provenance/exact-repeat behavior inspected; discovered the verifier overwrites a closed gate, so its output writer was intercepted. |
| I5 | Verify existing CPU checks before new analysis | `preflight/closed-checks/checks.json` and `checks.log`: baseline, router, on-demand audit gates JSON-exact to originals; 24 original CPU tests pass; paper-arithmetic command succeeds. |
| I6 | Verify recorded local raw route hashes; pause if ignored raw missing/mismatched, never recreate | `preflight/recorded-hashes.json` verifies 33 prior recorded input/source hashes including all six route files and associated results/samples/commands. Frozen config adds independent count-shape/config hashes for 36 inputs. `load_inputs` hashes before parse/aggregate, and run/audit rehashes afterward. No missing/hash-mismatched original input occurred. Synthetic missing/tamper tests fail closed. |
| I7 | Preserve core/on-demand reports/protocols/raw/router/quantizer/frozen evaluation | Entire tracked and protected raw snapshot checks, plus exact old-gate recomputation; no model/data/evaluation changes. No final-score-driven search. |
| P1 | Create and freeze both protocol files before new aggregates | `protocol-freeze.json`: frozen phase/time and two hashes. Freeze occurred before implementing analysis; both successful run manifests are later. `frozen_protocol` rehashes both on run/audit; final preservation verifies unchanged protocol and chronology. |
| A1 | CPU-only analysis using saved 72-block 4/6/8 profiles | `load_inputs` checks all entries/types, IDs, quotas and weighted budgets for all three policies; no router/model forward call. Manifest CPU visibility is empty. `analysis.json.requests` contains canonical saved profiles, not generated requests. |
| A2 | Actual block parameter counts | `parameter_counts`: 252 measured integration projection counts equal saved fixed8 shape products; totals equal all six result-file count vectors. Independent audit derives counts again from shapes: 36 × 26,214,400 attention and 36 × 74,711,040 FFN, total 3,633,315,840. |
| A3 | Batch sizes 1,2,4,8; deterministic predeclared orderings | Frozen JSON and protocol §3; all saved/reverse/SHA256 orders in full grid. Independent set-product audit checks all 288 scenarios, not a selected subset. SHA input contains only fixed salt/task/index, no outcomes/profiles. |
| A4 | Shared batch takes maximum requested precision independently per block | `batch_metrics` and every `batches[].maximum_profile`; unequal-count hand calculation and independent recomputation of all 38,880 batch maxima. No approximate/averaged profile substitution. |
| A5 | Profile diversity and varying blocks | `profile_diversity` plus each batch's `diversity`: complete profile frequencies, largest fraction, 4/6/8 histograms and varying IDs/by-type counts. Independent checker verifies both population and batch diversity. Report gives pooled/task counts and size trends. |
| A6 | Each query's weighted requested bits; shared maximum-profile bits | Every query has requested/shared selected bits and bits/parameter; batches separately record one shared payload. Independent checker proves all original requests cost 21,799,895,040 bits (6 bits/parameter) and every shared value equals the weighted maximum. |
| A7 | Extra bits imposed on queries; blocks promoted to 8 | Every query records absolute excess, bits/parameter excess, relative percent, promoted IDs/count/parameters; batch and per-block contributions also saved. Independent checker recomputes all 82,944 request records, verifies 4/6→8 versus unanimous 8, and sum conservation. |
| A8 | Quantities across batch size | Report §3 includes full 1/2/4/8 primary table, shared/extra bits and promotions, batch varying-block trends; all min/median/mean/max and individual records remain in outputs. |
| A9 | Arrival versus one bounded compatibility grouping | Frozen oldest-seed/minimum-resulting-total-excess heuristic, one fixed non-overlapping window of 16; no other algorithm/window tried. Independent script reselects every greedy candidate and tie and verifies every raw member list. |
| A10 | Waiting only as positions, not real latency; do not exceed window | Every query has input/output positions, displacement, readiness frontier/wait and extra readiness positions. Independent checker verifies formulas, all IDs once, same original window and displacement ≤15. Report labels waits as positions, never milliseconds/latency. |
| A11 | Separate measurements from assumptions; no throughput/kernel inference | Report §§2–4 distinguish inherited route/count measurements, deterministic derived quantities and hypothetical shared maximum/free-route availability. Explicitly unknown: batched quality/numerics, GPU throughput/latency/memory/kernel behavior, probe and queue costs. |
| T1 | Hand-written known-answer tests | `BatchArithmeticTests`: [4,8]/[8,4] with counts [1,3] yields R=28/20, M=32, E=4/12, block contributions=4/12. Separate [4,6,8]/[8,8,8] case checks 4/6→8 versus unanimous 8 and unequal percentage denominators. |
| T2 | Batch-one identity and zero inflation for identical/static profiles | Focused tests cover all orders/sizes/groupings, fixed4/6/8 and mixed static vectors. Independent real-data audit checks every size-one and static scenario for zero excess/promotions. |
| T3 | No dropped/duplicated requests, deterministic output, partial windows | 23-query toy with repeated profiles but distinct IDs tests tails and all sizes. Independent raw audit covers every ID/position in all 288 real scenarios. Two separate CPU runs have byte-identical measurement files (`cmp` passes). |
| T4 | Exact route repeats before use | `checked_routes` checks strict canonical full-record equality excluding only timers and separately validates both profiles; sub-1e-12 feature changes/type drift/duplicate/missing/budget errors are rejected in tests. Both existing and independent raw checks also verify saved repeats. |
| G1 | Continue to GPU recommendation only if ≥10% median excess and ≥5 pp recovery at size 4 or 8 across ≥3 fixed orders | Frozen same-size/all-three-order condition in protocol §5/JSON; `decision` tests inclusive boundaries, inadequate order counts, failed one-order/same-size combinations, recovery/window failures and control/task non-rescue. Actual 6 primary comparisons fail both thresholds; eligible sizes `[]`. |
| G2 | Revise to narrow blocks if most excess concentrated in a small group | Frozen top8-of72 ≥50% rule. Independent audit recomputes ranks/contributions: size4=39.0206%, size8=40.1602%; not triggered. Tests require the main gate before any narrower GPU recommendation. No post hoc block exclusion/new gate. |
| G3 | Stop scheduler direction on small variation, low excess or unrecoverable grouping | Actual variation screen passes; low excess and low recovery independently fail. Report's first section states stop, preserves the negative result and recommends no GPU experiment. No post-result search. |
| X1 | No GPU jobs, scheduler/serving engine, KV quantization, retraining, quantizer change, new model/task, async loading, custom kernels | Source inspection: new runtime contains only file I/O, validation, offline grouping/math and audit; no model execution or GPU/runtime/serving calls. Existing CPU tests run on CPU; GPU-preflight test uses its existing mocked executable, not real GPU jobs. Protected source hashes unchanged. |
| D1 | Required source, CLI, focused tests, raw outputs, report and final audit | Exact paths and purposes below. Two complete raw outputs plus input/source/output manifests; report explicitly includes measured values, assumptions, limitations, unknowns and stopping decision. |
| V1 | Run existing CPU checks and new tests | Preflight:24 existing. Final safe wrapper:41 complete; `final-on-demand-cpu`:4 targeted and41 complete. `preflight/new-tests-2.log`:17 focused. No required test failed or was skipped. |
| V2 | Raw-output audit | `run-1-audit.log`, `run-2-audit.log`: both hash/source/inventory checks and byte-exact recomputations pass. `independent-audit.log` covers actual mathematical/coverage/greedy/decision requirements without importing analysis code. |
| V3 | git diff --check | `git-diff-check.log` successful. Final checker also runs `git diff --check qaq-baseline-closed-v1` and `git diff --no-index --check /dev/null FILE` for all seven new files, because ordinary diff omits untracked additions. All new files have no whitespace diagnostics. No-index exit1 means a clean addition; deliberate bad fixture returns3. Both ordinary/tag diff checks exit0. |
| V4 | Completion only after actual requirement audit; report blockers rather than invent evidence | This map is based on files/command logs and independent record-level recomputation, not a manifest flag alone. Final preservation, exact raw replay, CPU tests and interpretation audit all pass. No required research artifact or verification remains missing. |

## 2. Required deliverable inventory

| Permitted new path | Inspected implementation / content |
|---|---|
| `BATCH_INTERFERENCE_PROTOCOL.md` | Frozen populations, formulas, ordering/grouping/window, thresholds, concentration definition, tests, safety and limits |
| `configs/batch_interference_protocol.json` | Machine-readable frozen choices and 36 exact input SHA256 values |
| `src/qaq/batch_analysis.py` | Count/input/repeat validation, weighted maxima/diversity/inflation, offline grouping, waiting positions, concentration and frozen decision |
| `scripts/analyze_batch_interference.py` | Explicit CPU environment, freeze check, append-only output, source/input/output provenance and read-only exact replay audit |
| `tests/test_batch_analysis.py` | 17 focused tests; arithmetic, integrity, bounded grouping, deterministic/no-loss behavior, decision branches |
| `BATCH_INTERFERENCE_REPORT.md` | Negative recommendation; tables, assumptions versus derived measurements, scope limits, unknowns, provenance and commands |
| `BATCH_INTERFERENCE_AUDIT.md` | This full requirement map and completion-state assessment |

## 3. Executed commands and real outcomes

Environment for CPU commands: `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, existing `~/.venv/bin/python`;
OMP/MKL threads set to4 and Hugging Face/Transformers offline for closed checks.

| Command (from repository root) | Observed evidence |
|---|---|
| `python results/batch-interference-v1/preflight/run_closed_checks.py results/batch-interference-v1/preflight/closed-checks` | All three old gates exact;24 tests and paper arithmetic pass. Writer interception + protected-write audit hook, no modifications to old scripts/gates. |
| `python -m unittest tests.test_batch_analysis -v` | Initial16 and final17 pass in separately retained `preflight/new-tests-{1,2}.log`; final run preceded first aggregate. |
| `python scripts/analyze_batch_interference.py --out results/batch-interference-v1/run-1` and `--out .../run-2` | Both succeed;288 scenarios each. Exact argv/executable/cwd/Python/environment in manifests; launcher stdout retained. |
| `python scripts/analyze_batch_interference.py --audit results/batch-interference-v1/run-1` and `--audit .../run-2` | Both pass36 input hashes,5 current/snapshot source hashes, strict output inventory/hash and exact recomputation. |
| `cmp results/batch-interference-v1/run-{1,2}/analysis.json` and same for `summary.json` | Both exit0; measurement bytes identical, not merely rounded summary agreement. |
| `python results/batch-interference-v1/independent_audit.py` | `independent-audit.log`:288 scenarios,82,944 queries,38,880 batches; all-record arithmetic and greedy choices pass; independently recovers negative gate. |
| `python results/batch-interference-v1/preflight/run_closed_checks.py results/batch-interference-v1/final-closed-checks` | Old gates still exact;41 tests pass; paper arithmetic succeeds. |
| `python scripts/check_on_demand_cpu.py --out results/batch-interference-v1/final-on-demand-cpu` | Original CPU checker run without modification to a new directory:4 targeted +41 complete tests; inherited hashes/source manifest pass. |
| `git diff --check` | Exit0; tag diff also exit0; all seven new files have empty no-index whitespace diagnostics. |
| `python results/batch-interference-v1/preservation_audit_v2.py --out results/batch-interference-v1/preservation-reviewed.json` and final `--out .../final-preservation.json` | All59 tracked and1,144 protected raw hashes/mtimes, exact whitelist, HEAD/tag/index boundaries, freeze/source snapshots and whitespace checks pass. |

Not run: `scripts/check_integration.py` or `scripts/router_job.py` execution;
these are real-GPU producers, not the CPU audit surface. No build/dependency
installation is required for these Python additions. No PR, commit or push was
created: deliberately outside the allowed scope, not missing release evidence.

## 4. What the verifiers do and do not establish

- The old checkers validate closed baselines/routes/loading evidence. They do
  **not** prove batch interference, nor authorize rewriting their own gate files.
  Their writer was redirected, with exact object comparison before retaining a
  new copy. All closed files are independently rehashed afterward.
- The new replay verifier covers provenance, file inventory, full raw recompute
  and deterministic output. A bug shared with the producer could survive it;
  hand-calculated tests and a separate stdlib all-record implementation cover
  arithmetic, grouping, positions and threshold decisions independently.
- The final preservation checker covers all tracked content/modes/mtimes, all
  protected ignored raw content/sizes/mtimes and path inventories, Git boundaries,
  frozen-protocol/source equality, and whitespace of untracked additions. It is
  not a substitute for reviewing the report's interpretation.
- The report tables were inspected against `independent-audit.log` and full
  `run-1/summary.json`, including secondary controls, concentration, waiting
  positions and shared-bit denominators. No quality/throughput claims are inferred.
- Deliberately unknown, not incomplete requirements: batched GPU output quality,
  real arrival timing, route-probe feasibility/cost, real latency/throughput,
  kernel behavior, and optimal scheduling. The user forbids their measurement
  or implementation in this goal.

## 5. Final preservation and completion verdict

`preservation-reviewed.json` passes all59 tracked and1,144 protected raw
hash/mode-or-size/mtime checks, exact allowed additions, original HEAD/tag, empty
staged/tracked diffs, unchanged freeze/source snapshots and all whitespace checks.
The final rerun in `final-preservation.json` seals the finalized seven files and
the retained raw inventory. No protected path changed and no cleanup was needed.

The first helper misclassified `git diff --no-index --check` exit1 as failure;
its retained `preservation-before-final-audit.json` shows empty diagnostics and
otherwise passing preservation. `whitespace-exit-fixtures/exit-semantics.json`
proves clean additions return1 and real trailing whitespace returns3 with a
diagnostic. Append-only v2 corrects only this wrapper error; original helper,
failed record and fixtures remain. No frozen protocol or measured output changed.

**Completion verdict: all required research artifacts and audits pass. Negative
result: stop scheduler direction; no GPU or scheduler follow-on recommended.**
Batched GPU behavior remains intentionally unknown, not missing authorized work.
No commit/push/PR was made. The native `update_goal` tool is not exposed in this
session, so thread-status accounting cannot be updated through that requested
tool; no goal was recreated or replaced as a workaround.

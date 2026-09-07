# Query-budget feasibility report — closed analysis v1

## Material Passport / conclusion

**Decision: revise to a separately scoped signal-design study. Do not implement
a variable-budget router from this evidence.** All three tasks have material
within-task opportunity under the frozen six-bit screen. The one frozen router
score recovers at least 25% of that opportunity on only ARC-Challenge, not the
required two tasks. No threshold, example, task, score, salt or metric changed.

This is a CPU-only secondary analysis of existing closed Qwen3-4B fixed4/fixed8
results. No Qwen3-4B inference or training was run. The existing CPU test suite
includes tiny synthetic models; it is not a new benchmark evaluation.

**This study does not measure runtime, GPU memory, low-bit kernels, generation,
KV cache, batching, or vLLM behavior.** Logical bits are quantized projection
weight bits averaged over queries, not measured system costs. A six-bit endpoint
mixture is not fixed6 full-model evaluation.

Protocol: [QUERY_BUDGET_FEASIBILITY_PROTOCOL.md](QUERY_BUDGET_FEASIBILITY_PROTOCOL.md),
[config](configs/query_budget_feasibility_protocol.json). Evidence directory
`R = results/query-budget-feasibility-v1/` below. Protocol frozen at
2026-09-07T00:39:55.885745+00:00; first aggregate run started 00:47:27 UTC.
This preregisters a secondary analysis, not the already-known endpoint tables.

## 1. Confirmed measurements: primary six-bit screen

All differences below are **beneficial-direction improvements over the
query-independent expectation at exactly six average bits**. Thus positive is
better: expected NLL minus selected NLL for WT2; selected minus expected
accuracy for MC. Accuracies are fractions, not percentages. Intervals are the
frozen 10,000 paired-query resamples, seed 4242. They reselect exactly half of
each sampled multiset using the unchanged ranking, rather than let its budget
drift. They do not provide multiplicity-controlled or population-wide claims.

| Task / primary metric | Outcome-informed improvement [95% interval] | Frozen router improvement [95% interval] | Recovery | Material opportunity? | Router recovery >=25%? |
|---|---:|---:|---:|---|---|
| WT2 / mean NLL | .037475 [.029050, .045888] | .002987 [−.009384, .015885] | 7.97% | Yes | No |
| HellaSwag / acc_norm | .044922 [.029297, .062500] | −.009766 [−.027344, .007861] | −21.74% | Yes | No |
| ARC-Challenge / acc_norm | .083984 [.062500, .107422] | .021484 [−.005859, .044922] | 25.58% | Yes | Yes |

All three outcome-informed improvements exceed .01 and have lower interval
bounds strictly above zero. Only one router recovery passes. ARC-C's 25.58%
is only slightly over 25%, and its router interval includes zero. The frozen
screen uses point recovery, not router interval exclusion; this is a **screening
pass on ARC-C**, not demonstrated future predictive reliability. WT2's stronger
router result at seven bits cannot rescue its primary six-bit failure.

The outcome-informed intervals use the same outcomes to select and evaluate;
**they are not evidence of future generalization**. Their role is only to assess
opportunity within the saved pairs. No selector was fitted to these final labels.

### Exact budgets

Every rule uses the following fixed8/fixed4 counts, checked from explicit ID
lists in both runs and independently from raw records. No query is dropped or
duplicated. Counts apply separately within each task.

| Average bits | WT2 fixed8 / fixed4 | Each MC task fixed8 / fixed4 |
|---:|---:|---:|
| 4 | 0 / 64 | 0 / 256 |
| 5 | 16 / 48 | 64 / 192 |
| **6** | **32 / 32** | **128 / 128** |
| 7 | 48 / 16 | 192 / 64 |
| 8 | 64 / 0 | 256 / 0 |

For example, `(32*8 + 32*4)/64 = 6` and
`(128*8 + 128*4)/256 = 6`, exactly. The query-independent comparison is the
expectation over uniform subsets with exactly those counts, not a fractional
execution per query. All assignments share the same endpoints and scorer.

### WT2 six-bit quantities (lower mean NLL is better)

Every aggregate retains **24,576 scored tokens**, with 384 per original window.
The ledger preserves BOTH endpoint sums and token counts for every window;
unequal-length hand tests verify ratio-of-sums behavior.

| Selection | NLL sum | Mean NLL | Improvement [95% interval] |
|---|---:|---:|---:|
| Query-independent expectation | 71083.594282 | 2.892399 | reference |
| Outcome-informed, analysis only | 70162.598781 | 2.854923 | .037475 [.029050, .045888] |
| Frozen router score | 71010.197179 | 2.889412 | .002987 [−.009384, .015885] |
| Prompt token count | 71054.536145 | 2.891216 | .001182 [−.014501, .010693] |
| ID hash 1729 | 70936.070571 | 2.886396 | .006003 [−.008193, .016568] |
| ID hash 2718 | 70911.758520 | 2.885407 | .006992 [−.005759, .019212] |
| ID hash 31415 | 71198.739671 | 2.897084 | −.004685 [−.014864, .011255] |

All WT2 prompts have 128 tokens. The length control is therefore just ascending
index tie-breaking, not evidence of a length-based signal on this task.

### HellaSwag six-bit primary and secondary results

All 256 examples remain. Ordinary accuracy uses the **same** acc_norm-based
outcome-informed selection, not a second outcome-informed ranking.

| Selection | acc_norm | acc_norm improvement [95% interval] | Ordinary acc | acc improvement [95% interval] |
|---|---:|---:|---:|---:|
| Query-independent expectation | .673828 | reference | .492188 | reference |
| Outcome-informed, analysis only | .718750 | .044922 [.029297, .062500] | .484375 | −.007813 [−.023438, .011719] |
| Frozen router score | .664063 | −.009766 [−.027344, .007861] | .484375 | −.007813 [−.023438, .007813] |
| Prompt token count | .679688 | .005859 [−.013672, .023438] | .492188 | .000000 [−.013672, .017578] |
| ID hash 1729 | .691406 | .017578 [−.003906, .033203] | .488281 | −.003906 [−.019531, .011719] |
| ID hash 2718 | .679688 | .005859 [−.015625, .023438] | .496094 | .003906 [−.015625, .019531] |
| ID hash 31415 | .675781 | .001953 [−.015625, .019531] | .488281 | −.003906 [−.019531, .013672] |

### ARC-Challenge six-bit primary and secondary results

| Selection | acc_norm | acc_norm improvement [95% interval] | Ordinary acc | acc improvement [95% interval] |
|---|---:|---:|---:|---:|
| Query-independent expectation | .509766 | reference | .455078 | reference |
| Outcome-informed, analysis only | .593750 | .083984 [.062500, .107422] | .484375 | .029297 [.005859, .056641] |
| Frozen router score | .531250 | .021484 [−.005859, .044922] | .453125 | −.001953 [−.027344, .027344] |
| Prompt token count | .488281 | −.021484 [−.044922, .003906] | .445313 | −.009766 [−.033203, .015625] |
| ID hash 1729 | .515625 | .005859 [−.021484, .033203] | .449219 | −.005859 [−.033203, .017578] |
| ID hash 2718 | .496094 | −.013672 [−.037109, .011719] | .453125 | −.001953 [−.025391, .021484] |
| ID hash 31415 | .519531 | .009766 [−.017578, .033203] | .449219 | −.005859 [−.035156, .015625] |

The three ID-hash outcomes are a small predeclared variability check, not seeds
searched for a strong comparator. Their intervals are reported without selecting
one favorable realization. Length does not rescue the primary router failure.

## 2. Harm and negative findings, not hidden by aggregate gains

| Task / metric | Fixed4 wrong / fixed8 correct | Both correctness unchanged | Fixed4 correct / fixed8 wrong |
|---|---:|---:|---:|
| HellaSwag acc_norm (primary) | 17 | 233 (161 both correct, 72 both wrong) | 6 (2.34% of queries) |
| HellaSwag acc (secondary) | 10 | 238 (117 both correct, 121 both wrong) | 8 (3.13%) |
| ARC-C acc_norm (primary) | 34 | 213 (109 both correct, 104 both wrong) | 9 (3.52%) |
| ARC-C acc (secondary) | 34 | 213 (95 both correct, 118 both wrong) | 9 (3.52%) |

WT2: fixed8 reduces per-token NLL on **55/64** windows and increases it on
**9/64 (14.06%)**; zero exactly unchanged. Every MC query is explicitly assigned
one of the three correctness-transition categories in `analysis.json`. Equal
correctness need not mean equal predicted option. Matching ARC transition totals
between acc and acc_norm do not imply the same per-example transitions.

The router has negative primary improvement on HellaSwag. On ARC its primary
acc_norm screen barely passes while ordinary accuracy decreases. Even the
acc_norm outcome-informed limit decreases HellaSwag ordinary accuracy at six
bits. These findings are retained; no metric substitution was made. Higher
precision is not uniformly beneficial on individual queries.

## 3. Descriptive budget curves — cannot rescue primary failure

Here E is query-independent expectation, O is outcome-informed (analysis-only),
R is frozen router score, L is prompt length, and Hs are the three fixed ID salts.
All values are actual endpoint-selection metrics, not interpolated fixed6 model
results. Endpoints at four/eight bits coincide across every selection.

### WikiText-2 mean NLL

| Bits | E | O | R | L | H1729 | H2718 | H31415 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.938453 | 2.938453 | 2.938453 | 2.938453 | 2.938453 | 2.938453 | 2.938453 |
| 5 | 2.915426 | 2.889014 | 2.914260 | 2.915093 | 2.915656 | 2.910385 | 2.917178 |
| 6 | 2.892399 | 2.854923 | 2.889412 | 2.891216 | 2.886396 | 2.885407 | 2.897084 |
| 7 | 2.869372 | 2.836026 | 2.858294 | 2.871605 | 2.868225 | 2.868771 | 2.873641 |
| 8 | 2.846345 | 2.846345 | 2.846345 | 2.846345 | 2.846345 | 2.846345 | 2.846345 |

Every cell retains its NLL sum and 24,576 tokens in the raw output. Fixed4 NLL
sum is 72215.42086501978; fixed8 is 69951.767698063. O at seven bits can outperform
all-fixed8 here by avoiding harmful upgrades; this is outcome-informed hindsight,
not a proposed usable algorithm.

### HellaSwag acc_norm

| Bits | E | O | R | L | H1729 | H2718 | H31415 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .652344 | .652344 | .652344 | .652344 | .652344 | .652344 | .652344 |
| 5 | .663086 | .718750 | .656250 | .667969 | .656250 | .664063 | .656250 |
| 6 | .673828 | .718750 | .664063 | .679688 | .691406 | .679688 | .675781 |
| 7 | .684570 | .718750 | .671875 | .699219 | .695313 | .695313 | .699219 |
| 8 | .695313 | .695313 | .695313 | .695313 | .695313 | .695313 | .695313 |

Ordinary acc at endpoints: .488281 fixed4, .496094 fixed8. Full secondary
ordinary-accuracy curves for all five budgets and six selections are retained
under each `tasks.hellaswag.budgets` record; primary-budget evidence is above.

### ARC-Challenge acc_norm

| Bits | E | O | R | L | H1729 | H2718 | H31415 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .460938 | .460938 | .460938 | .460938 | .460938 | .460938 | .460938 |
| 5 | .485352 | .593750 | .496094 | .472656 | .492188 | .476563 | .484375 |
| 6 | .509766 | .593750 | .531250 | .488281 | .515625 | .496094 | .519531 |
| 7 | .534180 | .593750 | .550781 | .523438 | .550781 | .535156 | .535156 |
| 8 | .558594 | .558594 | .558594 | .558594 | .558594 | .558594 | .558594 |

Ordinary acc at endpoints: .406250 fixed4, .503906 fixed8. All secondary curves
are likewise retained under `tasks.arc_challenge.budgets`.

## 4. Analysis-only limits

- O ranks the observed per-query primary benefit and is **unavailable during
  real inference and unsuitable as a deployable method**. Its intervals cannot
  establish generalization because selection uses the same observed outcomes.
- All actual WT2 scored counts equal 384. The prescribed per-token ranking is
  optimal for this saved task; it need not maximize total weighted NLL benefit
  if window lengths differ. Aggregation still uses sums/counts in that case.
- This analyzes endpoint substitution only. It does not infer any unobserved
  fixed6 full-model quality or other precision's quality by interpolation.
- Six bits means 3,633,315,840 quantized projection parameters per query on
  average. The 389,152,256 excluded FP16 parameters, scales and fixed8 probe
  are not included in that logical budget. No physical cost equivalence follows.
- No combined cross-task quality score was calculated. Within-task gates pass
  on all three tasks, so opportunity is not solely a between-task effect.

## 5. Assumptions

The frozen pairs are the intended evaluation units; raw scorer provenance and
exact-repeat equality are valid. Each endpoint can be substituted independently
in this offline accounting. Bootstrap treats windows/examples as exchangeable
pairs: each draw samples the same paired losses, prompt scores and correctness
together, and exact half-quota selection is reapplied. Dependence between WT2
windows or similar MC questions is not removed by this procedure. Deterministic
repeats are validation, not extra independent samples.

The score is exactly the actual parameter-weighted predicted local block-error
reduction from 4 to 8: `sum_b p_b*(cost4_b-cost8_b) / sum_b p_b`, using 72 blocks
and their 252 verified projections. Attention blocks have 26,214,400 parameters,
FFN blocks 74,711,040. No alternate score, calibration, transformation, subset,
weight change or trained selector was tried.

The score precedes answer scoring but currently needs a **complete fixed8 prompt
probe**. The cheap control only uses prompt token count. A predictive signal
alone would not establish practical benefit, especially with probe cost unpaid.

## 6. Unknowns and future requirements (not completed)

Whether a cheap pre-answer signal can identify useful upgrades out of sample;
why predicted local error poorly transfers to final quality; and how findings
change with new held-out queries remain unknown. The saved subsets are small,
there is no multiple-testing correction, and none of the frozen router's three
primary intervals excludes zero. These are screening results, not a general
quality/efficiency claim or a comparison to the paper's table.

A **separate signal-design study**, if authorized, must use separate training
and held-out evaluation data, not tune against these final pairs. Before any
subsequent variable-total-budget router recommendation can establish practical
benefit, requirements include:

1. **Fixed6 full-model evaluation**, not an assumed interpolated result.
2. **A separately trained inexpensive pre-answer selector**, evaluated without
   reusing these observed outcome-informed labels for final selection.
3. **Router-probe cost** measured and included; predictive ability is insufficient.
4. **Fair global-average-bit comparisons**, including inexpensive query-independent
   alternatives and clearly separated weight/probe/exclusion accounting.

None is implemented or measured here. No new router or serving system is added.
The broad direction is not stopped by the opportunity rule (3/3 pass), but the
current score does **not** justify continuing directly to router implementation.

## 7. Verification, failures, provenance

- Initial HEAD/tag: `b59adbcffc90b870f014d22e142923d569596986`, clean. All 59
  closed tracked files and 1196 protected raw files recorded before editing.
  Initial historical hash validation covers 1200 referenced paths. Baseline
  four/eight repeats have exactly the same ordered 576 task/index pairs and
  identical full sample/non-timing result records. Adaptive records repeat
  exactly except `probe_seconds`.
- 40 CPU tests pass: all 24 existing tests plus 16 new focused tests. The first
  focused attempt had a hand-fixture arithmetic typo: repeated token counts
  1+1+2+3 were mistakenly totaled as 5 instead of 7. Only that expected interval
  and comment were corrected **before real aggregate outcomes**; the log is
  retained. An initial inspection command lacked `python` on PATH; all actual
  checks use the existing `~/.venv/bin/python`. No packages were installed.
- Both fresh processes produce byte-identical `analysis.json` AND `manifest.json`.
  Analysis SHA256: `78eddf15e7e0ebb79957ac3c5ac5b6a1aa49347d71130631abd6d1fbfac57251`.
  Manifest SHA256: `7d7de0590aee3c34adf1060fb9f4635f289e24cdcc0889ff96d9f94b32286ebb`.
- Independent arithmetic v2 imports no `qaq` module or analysis implementation.
  It uses rational point arithmetic, direct-raw rankings, Python occurrence
  sorting and an explicit linear percentile. It checks all 576 per-query
  records, 90 task/budget/selections, 30 primary-budget metric intervals, exact
  counts/IDs and the decision. V1 also passed but used already-checked output
  benefit values for oracle ordering; v2 removes that dependency. Both are
  retained append-only. Tiny summation tolerances are disclosed in the audit
  output; IDs/counts/decisions and run-to-run bytes must be exact.
- Final preservation, command exit statuses, whitespace checks, safety refusals
  and a full prompt-to-artifact map are in
  [QUERY_BUDGET_FEASIBILITY_AUDIT.md](QUERY_BUDGET_FEASIBILITY_AUDIT.md).

### Reproduce without touching a closed result

From the repository, in the existing environment; use a NEW run directory:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python scripts/analyze_query_budget.py --out results/query-budget-feasibility-v1/NEW-RUN
```

Evidence: `R/preflight/{initial-state,input-checks}.json`, `R/protocol-freeze.json`,
`R/run-{1,2}/{analysis,manifest}.json`, separate run logs,
`R/independent_arithmetic_audit_v2.py`, `R/independent-arithmetic-v2.{json,log}`.
Independent audit outputs use exclusive creation: do not overwrite their completed
paths. Raw evidence is git-ignored and must be preserved separately from Git.

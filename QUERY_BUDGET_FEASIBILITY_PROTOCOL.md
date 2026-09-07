# Query-budget feasibility protocol — v1

## Material Passport

CPU-only, deterministic secondary analysis of the closed Qwen3-4B QAQ samples.
Question: is there within-task final-quality benefit to assigning different
queries fixed4 or fixed8, and does one already-recorded pre-answer score identify
it? No model execution, fitting, router implementation, or serving work.
This protocol is frozen before computing new aggregate outcomes, not before the
original endpoint results were known. Historical aggregate tables were read as
required; no new selection performance was examined before this freeze.

Authoritative settings: `configs/query_budget_feasibility_protocol.json`.
Freeze receipt: `results/query-budget-feasibility-v1/protocol-freeze.json`, which
seals both protocol files and the pre-edit preflight records. Never edit a frozen
protocol to improve a result. Any invalid input or unresolved ambiguity means
pause with the exact path/reason; never repair or regenerate closed evidence.

## Inputs, integrity, and boundaries

Initial HEAD and peeled `qaq-baseline-closed-v1` are both
`b59adbcffc90b870f014d22e142923d569596986`; initial status and tag diff are empty.
The preflight records hashes/mode/mtime/size of all 59 tag files and all 1196
files under `results/{core-v1,on-demand-v1,batch-interference-v1}`. Historical
recorded hashes were checked, including the prior preservation inventory and
both batch-analysis manifests. These closed paths remain immutable, including
existing bytecode. Use `PYTHONDONTWRITEBYTECODE=1` for all Python commands.

Analysis inputs are ONLY existing fixed4/fixed8 baseline samples (r1 and r2),
their results/commands/module inventories, frozen examples/manifests/configs,
adaptive r1/r2 routes and result/command metadata, and integration parameter
counts/projection checks. Required hashes are explicit in the config. Adaptive
final quality is not used. Router training/checkpoint files may be hash-checked
for provenance but are not executed or loaded. No new examples, labels or tasks.

Require exact ordered `(task,index)` identity with the frozen examples in every
input, no missing/extra/repeated IDs, exact baseline sample repeats and exact
non-timing baseline result repeats. Adaptive records must be identical except
`probe_seconds`. The only baseline result timing exclusions are `eval_seconds`
and `total_seconds`; no other fields may be ignored. Compare complete route
records before extracting the score. Check routes see only each frozen prompt.
Validate token losses, NLL sums/counts, choice normalization, predictions and
gold identity. Check 72 block sizes against all 252 integration projection counts
and baseline module shapes, and both adaptive run block-count inventories.

Only these seven source/document additions are allowed:
`QUERY_BUDGET_FEASIBILITY_PROTOCOL.md`,
`configs/query_budget_feasibility_protocol.json`,
`src/qaq/query_budget_analysis.py`, `scripts/analyze_query_budget.py`,
`tests/test_query_budget_analysis.py`, `QUERY_BUDGET_FEASIBILITY_REPORT.md`,
`QUERY_BUDGET_FEASIBILITY_AUDIT.md`. Do not edit `src/qaq/__init__.py` or any
existing file. New raw outputs use exclusive creation in new paths beneath
`results/query-budget-feasibility-v1`; no overwrites, even on failed attempts.
No commits/push/merge/branch changes, installations, GPU jobs, retraining,
dynamic batching, KV-cache quantization, or vLLM integration.

## 1. Separate tasks and 2. endpoint outcomes

Analyze WikiText-2 (64 windows), HellaSwag (256 examples), ARC-Challenge (256
examples) separately. Never pool incompatible metrics or select by cross-task
rank. Every query remains in the per-query ledger.

WT2 observed benefit `d_i = NLLsum4_i/tokens_i - NLLsum8_i/tokens_i`.
Positive means fixed8 is better; negative means harm. Preserve both endpoints'
NLL sums and scored-token counts, which must match within each pair. Aggregate
as total NLL sum / total scored tokens, NOT an unweighted average of window
means if lengths differ. Rank the outcome-informed limit by this per-token
benefit. All actual windows score 384 tokens, so this ranking also maximizes
the aggregate improvement; with unequal lengths it is the specified ranking,
not necessarily a global optimum. Tests must exercise unequal lengths.

MC primary benefit is `correct_norm8 - correct_norm4`. Record each query as
`fixed4_wrong_fixed8_correct`, `both_unchanged`, or
`fixed4_correct_fixed8_wrong`; unchanged means correctness unchanged, not
necessarily the predicted option. Retain both-correct and both-wrong counts
as detail. Ordinary accuracy, its transitions, and outcomes of the SAME
selections are secondary; never re-rank or decide using ordinary accuracy.

## 3. Exact budgets and 4. query-independent comparison

Within each task of n queries, budgets B = 4,5,6,7,8 require exactly
`k = n*(B-4)/4` queries at fixed8 and n-k at fixed4. Reject nonintegral quotas;
never round. Verify k, n-k, the selected IDs, `4*(n-k)+8*k`, and its exact
rational average B for every selection. In particular B=6 uses exactly n/2
at each endpoint. These bits mean quantized projection weights, averaged over
QUERIES; exclusions/scales and probe work are not included. This is not fixed6,
not token-work weighting and not a memory or latency budget.

The query-independent comparison is the exact expected endpoint result of a
uniform subset of exactly k queries, without using their information. Each
query has probability f=k/n of fixed8: expected NLL sum is
`sum((1-f)*NLLsum4 + f*NLLsum8)`, retaining total token count; expected accuracy
is the analogous mean of paired correctness. This is an expectation over
exact-quota assignments, NOT fractional execution of any individual query.

Small variability check: exactly three deterministic ID-hash selections, salts
1729, 2718, 31415. Ascending SHA256 of the UTF-8 string
`'{salt}:{task}:{index}'` ranks queries; then ascending task/index breaks hash
ties. First k receive fixed8. The lists are nested across budgets. They are
query-information-free with respect to prompt content and outcomes, but ID
hashing is deterministic rather than an IID experiment. Report all three;
never search or replace salts after viewing outcomes.

## 5. Outcome-informed analysis-only limit

Select the k largest observed primary benefits, with ascending task and integer
index for ties, even if some selected benefits are zero or negative. Fixed
budgets may force harmful upgrades. This selection uses the very answers being
evaluated: UNAVAILABLE during real inference and UNSUITABLE as a deployable
method. It only establishes opportunity in the saved sample; it does not
establish generalization, practical savings, or achievable routing accuracy.

## 6. Exactly one primary pre-answer score and 7. cheap control

For query i and block b in recorded integration order, compute in float64:

`score_i = sum_b(parameters_b * (costs_i[b][0] - costs_i[b][2])) / sum_b(parameters_b)`

`costs` are the existing frozen A1 predicted local NMSE at 4/6/8 bits, in that
order. The denominator is the actual 3,633,315,840 quantized parameters; use
actual per-block sizes, not a block mean. Sum with `math.fsum`. Larger score
gets fixed8 first, with ascending task/index ties. No fitting, calibration,
transformation, alternate formula, feature subset, learned weight, or final-
score-based choice is allowed. Do not use actual adaptive final-quality results.

The one cheap control is prompt token count (`len(context_tokens)`), larger
first, same task/index ties. WT2 length ties are expected; they will reduce this
control to ID order, not justify inventing a replacement. The primary router
score is available before answer scoring but currently requires a COMPLETE
fixed8 prompt probe. Predictive ability alone cannot establish practical benefit.

## 8. Primary budget and 9. paired query resampling

B=6 is the only decision budget. Other budgets are descriptive and cannot rescue
failure. For every method report its beneficial-direction difference above the
query-independent expectation at EXACTLY the same average bits: expectation
minus selection for mean NLL; selection minus expectation for accuracies.

At B=6 report percentile 95% intervals from 10,000 paired query-level draws,
seed 4242, NumPy `default_rng` (PCG64), reset independently for each task.
Generate `rng.integers(0,n,size=(10000,n))`; the same sampled indices are used
for every comparator and both endpoints/secondary metrics within that task.
Plainly: repeatedly sample n of the SAME saved query PAIRS with replacement,
keeping prompt scores, losses, token counts and correctness together.

Within each resampled multiset, reapply each unchanged ranking and select exactly
n/2 fixed8 OCCURRENCES; then compare with its f=1/2 expectation. Repeated draw
occurrences are intentional, not duplicate input IDs. Break duplicate-ID ties
by draw position after task/index; duplicate observations have identical values.
This enforces six bits even inside every draw and measures the fixed rank-and-
quota selection rule, not a fixed ID membership list with a fluctuating budget.
For WT2, divide resampled NLL sums/differences by resampled total token count.
For MC divide by n. Percentiles use NumPy's `method='linear'` (interpolation at
position `(draws-1)*q`). No refitting of scores occurs. The outcome-informed rule
reuses observed outcomes in every draw; its interval is emphatically NOT evidence
of future generalization. Neither repeat run is an independent query sample.

## Frozen screening rules

1. Material within-task opportunity requires outcome-informed B=6 improvement
   >=0.01 mean NLL (WT2) or >=0.01 absolute acc_norm (MC), beneficial direction,
   AND the paired interval lower bound >0. Exact equality at 0.01 passes the
   point threshold; a lower bound equal to zero fails. Do not round for decisions.
2. Continue to RECOMMENDING a separate variable-total-budget router study only
   when >=2 tasks have material opportunity AND on >=2 opportunity tasks the
   frozen router improvement is nonnegative and >=25% of the outcome-informed
   improvement over the same-bit expectation. Recovery is an un-clipped ratio;
   a negative value is retained. Router CI exclusion is NOT an extra decision
   gate. Exactly 25% passes. Prompt length cannot rescue router failure.
3. If >=2 have opportunity but the router rule fails: `revise_signal_design`,
   a separately scoped signal-design question, without implementing it here.
4. Exactly one opportunity task: `revise_task_specific`; STOP the broad direction.
   Zero: `stop_broad_direction`. Thus fewer than two always stops the broad
   direction; the one-task case additionally identifies a task-specific question.
   Between-task aggregate differences never substitute for within-task gates.
5. Record all harmful fixed8 transitions/negative benefits separately, regardless
   of decision; no assumption that higher precision is always better.
6. Invalid evidence, unresolved decision ambiguity or inconsistency: `pause`,
   report exact evidence/blocker and ask for input; never change thresholds,
   formulas, metrics, tasks, examples or data. This is a screening rule, not a
   multiple-testing-controlled discovery or proof of generalization.

## Verification and reporting

Pure deterministic analysis code; standard library plus already-installed NumPy.
Tests use hand-computable inputs for signs, MC transition classes, exact five
budgets, ties, unequal counts, expectation, outcome/score ranks, negative benefit,
IDs/missing queries/repeats/hashes, decision boundaries and complete accounting.

Produce two fresh processes with byte-identical analysis and manifest files
(no run-dependent timestamps/argv/output paths in deterministic payloads).
Save commands separately. An independent arithmetic audit must NOT import the
implementation: reconstruct per-query values/ranks/all budget aggregates and
primary resampling/decisions directly from raw inputs and frozen settings, and
compare them with both output runs. Verify all selected and unselected IDs.

Run ALL existing CPU tests as well as focused tests with GPUs hidden. Any
existing verifier which normally writes a closed gate must not be run in its
writing mode; use read-only functions or redirect its write function solely to
new raw outputs if additional verification is needed. Preserve existing files.
Run `git diff --check`, tag diff, and explicit no-index whitespace checks for
untracked additions. Rehash all protected inventories and confirm only the seven
permitted additions. Pause on extra changes without destructive cleanup.

The report separates confirmed measurements, analysis-only limits, assumptions,
unknowns, negative findings and threshold decision. Explicitly: this study does
NOT measure runtime, GPU memory, low-bit kernels, generation, KV cache, batching,
or vLLM behavior. If continuing is recommended, fixed6 full-model evaluation,
a separately trained inexpensive pre-answer selector, router-probe cost, and
fair global-average-bit comparisons are future requirements, NOT completed work.
The audit maps every user requirement to inspected artifacts and command results;
tests or a manifest alone are not proof of full completion.

# Fixed6 bridge and cheap query-budget signal study v1 — report

## Decision and scope

**Completed Phase A: `stop_endpoint_budget_direction` (0/3 tasks pass).**
Phase B was not authorized and was not run: no new signal features, data splits,
endpoint labels, ridge fits, family choice, signal timing, or final evaluations.
This is a valid negative screening result, not a failed work product.

The outcome-informed endpoint mixtures do **not** establish material opportunity
over actual uniform fixed6 under the frozen rule. HellaSwag and ARC have positive
point estimates, but their intervals fail the strict lower-bound gate. ARC's
lower bound is exactly zero, not a small positive value rounded away. WT2's point
estimate is effectively tied and slightly unfavorable. This stops this bounded
**whole-query fixed4/fixed8 endpoint-routing direction**, not every variable-budget
method. It does not show that no query-dependent precision method can ever work.

The scientific question about a *cheap* pre-answer selector remains **unmeasured**,
not disproven: the necessary Phase-A screen failed. No task-specific direction is
authorized by the zero-pass rule. Positive nominal MC differences are reported,
not used to select replacement tasks, thresholds, features, or samples.

## Confirmed primary evidence

All values below are absolute metric units, not relative percentages. Beneficial
improvement is fixed6 NLL minus mixture NLL for WT2, and mixture acc_norm minus
fixed6 acc_norm for MC. Decisions use unrounded values.

| Task | n | Actual fixed6 | Outcome-informed mixture | Improvement | Paired 95% interval | Material pass |
|---|---:|---:|---:|---:|---|---|
| wikitext2 | 64 | 2.85488096 | 2.85492345 | -0.00004250 | [-0.01765570, 0.01804096] | No |
| hellaswag | 256 | 0.69921875 | 0.71875000 | 0.01953125 | [-0.00390625, 0.04296875] | No |
| arc_challenge | 256 | 0.56250000 | 0.59375000 | 0.03125000 | [0.00000000, 0.06250000] | No |

WT2 uniform fixed6 summed NLL is 70161.55438143417 over 24,576 scored tokens;
token perplexity is 17.372369013050495. All 64 windows score 384 tokens. MC
accuracies use 256 complete queries per task, not candidate counts.

The previous feasibility result (material opportunity vs an exact-half endpoint
*expectation* on 3/3 tasks) is not contradicted: it omitted an actual full-model
fixed6 comparator. Its full-fixed8-probe score recovered only 7.97%, -21.74%, and
25.58% against that expectation on WT2, HellaSwag and ARC respectively. Those old
ratios are not new evidence that the router works against fixed6.

## Complete primary-metric comparisons at six average logical bits

Every endpoint selection upgrades exactly 32/64 WT2 queries or 128/256 MC queries.
The expectation averages over exact-half subsets; it does not execute fractional
queries. All intervals use 10,000 **paired** query resamples, PCG64 seed 4242 reset
per task, unchanged rankings reapplied inside each multiset, exactly half fixed8
occurrences per draw, deterministic ID/occurrence ties, and linear percentiles.
WT2 uses ratio of NLL sums to token sums, not a mean of window means.

Outcome-informed is an answer-informed analysis ceiling, unavailable at inference;
its resampling interval does not establish generalization. Router-score selection
is the old frozen parameter-weighted score from a **full fixed8 prompt probe**,
not a cheap signal. Adaptive/static/random are the prior QAQ block-allocation
policies, not newly optimized baselines; each has parameter-weighted six-bit
scoring and excluded FP16 parameters. Probe costs are not charged by this logical
budget, so this is not a cost-parity or speed comparison.

### wikitext2: mean_nll

Actual fixed6: **2.85488096**. Positive deltas favor the method.

| Method | Metric | Δ vs fixed6 | 95% CI | Δ vs expectation | 95% CI |
|---|---:|---:|---|---:|---|
| query_independent | 2.89239886 | -0.03751790 | [-0.04877427, -0.02530886] | 0.00000000 | [-0.00000000, 0.00000000] |
| outcome_informed | 2.85492345 | -0.00004250 | [-0.01765570, 0.01804096] | 0.03747540 | [0.02905033, 0.04588793] |
| router_score | 2.88941232 | -0.03453136 | [-0.04870773, -0.01887874] | 0.00298654 | [-0.00938408, 0.01588476] |
| prompt_length | 2.89121648 | -0.03633552 | [-0.05392553, -0.02267838] | 0.00118238 | [-0.01450103, 0.01069279] |
| id_hash_1729 | 2.88639610 | -0.03151514 | [-0.05031582, -0.01486205] | 0.00600276 | [-0.00819259, 0.01656798] |
| id_hash_2718 | 2.88540684 | -0.03052588 | [-0.04682262, -0.01310924] | 0.00699202 | [-0.00575897, 0.01921152] |
| id_hash_31415 | 2.89708413 | -0.04220318 | [-0.05761154, -0.01970615] | -0.00468528 | [-0.01486402, 0.01125474] |
| adaptive | 3.00889775 | -0.15401680 | [-0.17048597, -0.13791346] | -0.11649890 | [-0.13110591, -0.10228274] |
| static | 3.01686328 | -0.16198233 | [-0.17934219, -0.14483766] | -0.12446443 | [-0.13953482, -0.10973641] |
| random | 2.87985674 | -0.02497578 | [-0.04749766, -0.00310380] | 0.01254212 | [-0.00774955, 0.03201880] |

### hellaswag: acc_norm

Actual fixed6: **0.69921875**. Positive deltas favor the method.

| Method | Metric | Δ vs fixed6 | 95% CI | Δ vs expectation | 95% CI |
|---|---:|---:|---|---:|---|
| query_independent | 0.67382812 | -0.02539062 | [-0.04882812, -0.00195312] | 0.00000000 | [0.00000000, 0.00000000] |
| outcome_informed | 0.71875000 | 0.01953125 | [-0.00390625, 0.04296875] | 0.04492188 | [0.02929688, 0.06250000] |
| router_score | 0.66406250 | -0.03515625 | [-0.06250000, -0.00781250] | -0.00976562 | [-0.02734375, 0.00786133] |
| prompt_length | 0.67968750 | -0.01953125 | [-0.05078125, 0.01171875] | 0.00585938 | [-0.01367188, 0.02343750] |
| id_hash_1729 | 0.69140625 | -0.00781250 | [-0.03906250, 0.01953125] | 0.01757812 | [-0.00390625, 0.03320312] |
| id_hash_2718 | 0.67968750 | -0.01953125 | [-0.05078125, 0.00781250] | 0.00585938 | [-0.01562500, 0.02343750] |
| id_hash_31415 | 0.67578125 | -0.02343750 | [-0.04687500, 0.00000000] | 0.00195312 | [-0.01562500, 0.01953125] |
| adaptive | 0.66796875 | -0.03125000 | [-0.05859375, -0.00390625] | -0.00585938 | [-0.02929688, 0.01757812] |
| static | 0.66796875 | -0.03125000 | [-0.06250000, 0.00000000] | -0.00585938 | [-0.03125000, 0.01953125] |
| random | 0.66015625 | -0.03906250 | [-0.06640625, -0.01562500] | -0.01367188 | [-0.03515625, 0.00781250] |

### arc_challenge: acc_norm

Actual fixed6: **0.56250000**. Positive deltas favor the method.

| Method | Metric | Δ vs fixed6 | 95% CI | Δ vs expectation | 95% CI |
|---|---:|---:|---|---:|---|
| query_independent | 0.50976562 | -0.05273438 | [-0.08398438, -0.02148438] | 0.00000000 | [0.00000000, 0.00000000] |
| outcome_informed | 0.59375000 | 0.03125000 | [0.00000000, 0.06250000] | 0.08398438 | [0.06250000, 0.10742188] |
| router_score | 0.53125000 | -0.03125000 | [-0.07421875, 0.00781250] | 0.02148438 | [-0.00585938, 0.04492188] |
| prompt_length | 0.48828125 | -0.07421875 | [-0.11328125, -0.03515625] | -0.02148438 | [-0.04492188, 0.00390625] |
| id_hash_1729 | 0.51562500 | -0.04687500 | [-0.08984375, -0.00390625] | 0.00585938 | [-0.02148438, 0.03320312] |
| id_hash_2718 | 0.49609375 | -0.06640625 | [-0.10546875, -0.02734375] | -0.01367188 | [-0.03710938, 0.01171875] |
| id_hash_31415 | 0.51953125 | -0.04296875 | [-0.08593750, -0.00390625] | 0.00976562 | [-0.01757812, 0.03320312] |
| adaptive | 0.53125000 | -0.03125000 | [-0.07031250, 0.00390625] | 0.02148438 | [-0.00976562, 0.05468750] |
| static | 0.50390625 | -0.05859375 | [-0.09765625, -0.02343750] | -0.00585938 | [-0.03710938, 0.02539062] |
| random | 0.53125000 | -0.03125000 | [-0.06640625, 0.00390625] | 0.02148438 | [-0.01171875, 0.05468750] |

### Secondary ordinary accuracy (same normalized-primary selections)

No re-ranking or decision uses ordinary accuracy. Both MC fixed6 ordinary
accuracies are 0.49609375. CIs and deltas vs the endpoint expectation are also
retained for this secondary metric in the deterministic analysis JSON.

| Method | HellaSwag acc | Δ vs fixed6 | 95% CI | ARC acc | Δ vs fixed6 | 95% CI |
|---|---:|---:|---|---:|---:|---|
| query_independent | 0.49218750 | -0.00390625 | [-0.02148438, 0.01367188] | 0.45507812 | -0.04101562 | [-0.06835938, -0.01171875] |
| outcome_informed | 0.48437500 | -0.01171875 | [-0.03515625, 0.01562500] | 0.48437500 | -0.01171875 | [-0.04296875, 0.02343750] |
| router_score | 0.48437500 | -0.01171875 | [-0.03906250, 0.01171875] | 0.45312500 | -0.04296875 | [-0.08203125, -0.00390625] |
| prompt_length | 0.49218750 | -0.00390625 | [-0.02343750, 0.01953125] | 0.44531250 | -0.05078125 | [-0.08984375, -0.01171875] |
| id_hash_1729 | 0.48828125 | -0.00781250 | [-0.03125000, 0.01562500] | 0.44921875 | -0.04687500 | [-0.08984375, -0.00390625] |
| id_hash_2718 | 0.49609375 | 0.00000000 | [-0.02734375, 0.02343750] | 0.45312500 | -0.04296875 | [-0.07812500, -0.00781250] |
| id_hash_31415 | 0.48828125 | -0.00781250 | [-0.02734375, 0.01171875] | 0.44921875 | -0.04687500 | [-0.08984375, -0.01171875] |
| adaptive | 0.48828125 | -0.00781250 | [-0.03515625, 0.01953125] | 0.42968750 | -0.06640625 | [-0.10937500, -0.02343750] |
| static | 0.48828125 | -0.00781250 | [-0.03515625, 0.01953125] | 0.44140625 | -0.05468750 | [-0.09375000, -0.01562500] |
| random | 0.50781250 | 0.01171875 | [-0.01171875, 0.03515625] | 0.43750000 | -0.05859375 | [-0.09765625, -0.02343750] |

## Negative findings

- The query-independent exact-half expectation is worse than actual fixed6 on
  all three primary metrics, with all three paired intervals strictly negative.
- The frozen router-score endpoint selector is worse than fixed6 on all three
  point estimates. WT2 and HellaSwag intervals are strictly negative; ARC's crosses
  zero. This does not justify implementing that router.
- Every non-oracle method in the primary tables has a negative point effect on
  every task. The oracle is slightly worse on WT2 and uncertain on both MC tasks.
- Higher precision sometimes harms individual queries: fixed8 has higher WT2
  per-token NLL on 9/64 windows, and loses normalized correctness relative to
  fixed4 on 6/256 HellaSwag and 9/256 ARC queries. All negatives are retained;
  exact quotas are not relaxed to discard them.
- Ordinary accuracy has additional harms visible above; normalized accuracy is
  the frozen primary MC metric and cannot hide or be replaced by ordinary acc.

## Execution, reconstruction and repeat verification

Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, unchanged
frozen tokenizer and dataset revisions, unchanged 576 prompts/choices and local
causal scorer. Batch size 1, FP16 SDPA, no chat template, seed 1729, four threads,
TF32 disabled, deterministic algorithms. No packages installed or upgraded.
All model/tokenizer files and frozen runtime versions were checked before each
job. `apply_fixed(model, 6)` materializes 252 six-bit projection tensors with
3,633,315,840 parameters and 21,799,895,040 logical weight bits. The 389,152,256
excluded parameters remain FP16. These numbers do not count scales as bits saved.

Before any new fixed6 results, the protocol/config/runner/main analysis/tests were
sealed by `query_budget_signal_phase_a_freeze.json` (SHA256
`eecb4ffbb7b6b7e38b61c10538a357c0985670d097e0245c1e895cc2770ac9b6`). The code was
not changed after this seal. No commits were used for freezing because they were
explicitly prohibited.

Smoke IDs: WT2 5,6; HellaSwag 15,73; ARC 1,9. All 252 independent fixed6 tensors
matched nested reconstruction exactly (nested codes made before dense mutation),
and all six complete scored examples had byte-identical independent/nested
records. CPU tests separately reconstruct every signed byte's six-bit midpoint
using floor arithmetic, including zero groups and excluded tensors.

| Serial job | Physical GPU | PID | Outer wall seconds | Evaluation seconds |
|---|---:|---:|---:|---:|
| Smoke + nested verification | 1 | 152161 | 41 | 1.53513 independent path |
| Full fixed6 process 1 | 1 | 153653 | 169 | 129.28808 |
| Full fixed6 process 2 | 1 | 157416 | 170 | 130.57321 |

Smoke projection (2x safety factor) was **355.352 seconds/full job** and
**751.704 seconds total**, including smoke and two projected full runs. Actual
new GPU-job walltime was **380 seconds (0.10556 GPU-hours)**, well below six hours.
Each job used the unchanged process-free GPU preflight and external 1800-second
timeout, plus an in-process alarm. Jobs were serial; none bypassed a refusal.
These times enforce the research cost bound, **not serving performance claims**.

The two fresh full fixed6 raw sample files are byte-identical: zero changed
ordinary/normalized predictions, zero maximum choice-logprob delta (limit 1e-3),
and zero WT2 mean-NLL delta (limit 1e-5). Historical fixed4/fixed8/adaptive/static/
random repeats also agree exactly. The repeated process is a determinism check,
not extra statistical sample size. Every token NLL and summed/normalized choice
score is preserved in both complete and per-task raw JSONL files.

## Verification artifacts and reproduction

All paths below are beneath `results/query-budget-signal-v1/` unless shown otherwise.

- `query_budget_signal_initial.json`: clean pushed-parent/ref and pre-edit hashes.
- `query_budget_signal_phase_a_freeze.json`: pre-result seal.
- `query_budget_signal_smoke/`: independent/nested samples and 252-tensor gate.
- `query_budget_signal_projection.json`: pre-full cost authorization.
- `query_budget_signal_fixed6_r{1,2}/`: complete raw scores, modules, provenance,
  runtime and task records from distinct fresh processes.
- `query_budget_signal_analysis_r{1,2}/`: byte-identical analysis and manifests,
  all per-query endpoint benefits, six-bit/secondary outcomes, ranks, both selected
  and unselected IDs, exact bit accounting, every metric and paired interval.
- `query_budget_signal_independent_r{1,2}.json`: byte-identical independent
  nonimporting audit outputs. `scripts/audit_query_budget_signal.py` independently
  rebuilds ranks, selections and raw metrics and checks **100 intervals**, using
  scalar occurrence sorting and explicit percentile interpolation, without
  importing either main analysis implementation.
- `query_budget_signal_cpu_final.log`: **49/49 CPU tests** (all 40 existing plus
  9 new); `query_budget_signal_focused_final.log`: 9/9 focused tests. Prefreeze
  test output is separately retained. All tests used hidden GPUs and no bytecode.
- `query_budget_signal_review.md`: independent read-only review found no blockers;
  it does not substitute for the executed arithmetic and preservation checks.
- `query_budget_signal_preservation_final.json`: final hash/inventory/ref,
  allowed-addition, numerical-report and whitespace checks; see the requirement
  mapping in `QUERY_BUDGET_SIGNAL_AUDIT.md` for coverage and gated exclusions.

CPU reproduction (choose **unused** output names; never replace existing runs):

```bash
source ~/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m unittest discover -s tests -v
python scripts/analyze_query_budget_signal.py --out results/query-budget-signal-v1/query_budget_signal_analysis_NEW
python scripts/audit_query_budget_signal.py --out results/query-budget-signal-v1/query_budget_signal_independent_NEW.json
```

Do not rerun GPU jobs merely to reproduce this report. The preserved raw data
suffices for both analyses. A new GPU invocation needs fresh cost accounting and
preflight; the historical projection authorized exactly the executed two full
jobs. All existing tracked files and protected raw results remain unchanged.
No files were staged, committed, pushed or merged; new study files are untracked.

## Assumptions, unknowns and unsupported claims

**Assumptions:** the closed sample is suitable for this specified within-task
screen; query-level resampling approximates its sampling uncertainty; logical
weight-bit averages fairly describe projection precision, not physical costs.
The intervals are not a multiple-testing-controlled discovery. Nearby WT2 windows
may be dependent; no document-level independence or broader power claim is made.

**Unknown:** cheap-signal predictive utility, feasible Phase-B duplicate-free
counts, train/dev/final performance, ridge selection, signal extraction/loading
time, behavior on larger/unseen workloads, other precisions/budget allocations,
and any interaction with QAQ block allocation. Phase-B gates/tests were not
executed or represented as passed. The user-required separate B freeze would be
needed in a differently authorized continuation; this stop authorizes none.

**Unsupported system claims:** no measured low-bit resident footprint or memory
savings, integer/custom kernels, latency, throughput, dynamic batching, scheduler,
vLLM integration, generation, or KV-cache behavior. Computation remained
reconstructed FP16 linear operations. No serving mechanism was built and no
production benefit is asserted.

# QAQ batch-interference protocol v1 — frozen before new aggregates

## Material Passport
- Scope: CPU-only descriptive analysis of closed Qwen/Qwen3-4B routes; no scheduler.
- Inputs: existing final core routes and actual projection counts, not new inference.
- Status: preregistered; the two protocol files are sealed in
  `results/batch-interference-v1/protocol-freeze.json` before analysis.
- Provenance: AI-authored analysis design under the user's explicit decision rule.

## 1. Boundary and prerequisite evidence

`qaq-baseline-closed-v1` and all its 59 tracked files are immutable. Initial HEAD
is `b59adbcffc90b870f014d22e142923d569596986`; initial status and diff to tag are
empty. Before additions, all tracked files and all 1,144 files under
`results/core-v1` and `results/on-demand-v1` were hashed, with mtimes recorded in
`results/batch-interference-v1/preflight/initial-state.json` (SHA256
`d2a46063e9809b9064b9312e6be2dfb8d07ca3a0628373a3ae0513bed7a74e0e`).

The twelve user-named input documents/configs/source files were read before
writes. Recorded inherited hashes, including all six final route files and
results, passed. The original baseline/router/on-demand checks were executed
unchanged except for in-memory interception of their output writer: recomputed
old gate objects must equal the saved objects exactly, and copies go ONLY to the
new study. An audit hook forbids protected writes. The 24 existing CPU tests and
paper arithmetic check passed. See `preflight/closed-checks/` and the preserved
`preflight/run_closed_checks.py` for the exact runnable wrapper. Do not execute
the original CLI audit commands directly: they overwrite closed gate files.

Only these seven source-controlled additions are permitted: this file,
`BATCH_INTERFERENCE_REPORT.md`, `BATCH_INTERFERENCE_AUDIT.md`,
`configs/batch_interference_protocol.json`, `src/qaq/batch_analysis.py`,
`scripts/analyze_batch_interference.py`, `tests/test_batch_analysis.py`.
New retained raw output goes only under `results/batch-interference-v1`, using
exclusive creation and new directories. No commits, pushes, merges, branch
switches, history rewrites, or cleanup of changed protected paths. Missing inputs
or failed recorded hashes mean immediate pause, never recreation. Compare all
protected hashes/mtimes and the working tree against the tag at completion.

## 2. Question, populations, and controls

Does shared-weight batching materially inflate the logical weight precision
selected by the saved adaptive profiles, and can one bounded offline grouping
recover enough to justify a separately authorized GPU experiment?

Primary population: all 576 final requests, in the existing frozen examples
order (64 WikiText-2, 256 HellaSwag, 256 ARC-Challenge). Secondary descriptive
populations: each task separately, filtered before ordering. Requests are keyed
by `(task,index)`, NOT by profile/context; equal contexts remain distinct IDs.
Use adaptive repeat 1 only after repeat 2 passes exact equality. Also analyze
saved static and random policies as controls on identical populations/orders;
no control can trigger a positive adaptive decision. No dev/train data, quality
scores, option-level duplication, exclusion, task expansion, or new profiles.

Require exact ordered `(task,index,profile)` repeats and full route-record
semantic equality excluding ONLY `probe_seconds` (recorded wall time). Route
files intentionally have different byte hashes; each must match its own recorded
hash. No approximate equality, truncation, dropped IDs or duplicated requests.
Compare IDs to frozen examples. Validate all 72 entries as integer 4/6/8, exact
per-type 12/12/12 quotas and saved per-query weighted budgets before aggregation.

Use actual `block_parameter_counts` saved by `qaq.model.block_parameter_counts`
and `scripts/router_job.py`; require agreement across all six result files and
the hash-locked integration gate. Independently sum its 252 recorded projection
parameter counts and cross-check against the saved fixed8 module shapes/counts.
Block order is attention layer 0, FFN layer 0, attention layer 1, FFN layer 1, …,
attention layer 35, FFN layer 35. No model instantiation/loading is necessary.

## 3. Locked orderings, batch sizes, and one grouping heuristic

For each population, use exactly three profile-independent synthetic arrival
orders (not production arrival traces):
1. `saved`: original frozen request order.
2. `reverse`: its exact reverse.
3. `sha256`: sort ascending by SHA256 of the UTF-8 string
   `qaq-batch-v1|<task>|<index>`; break digest ties by original population position.

Analyze batch sizes **1, 2, 4, 8**, retaining a final partial batch if necessary.
Arrival grouping takes consecutive requests. Compatibility grouping uses fixed,
non-overlapping windows of **16 request positions**, reset at each population's
start, never crossing windows. Within a window:
- start each batch with the earliest unassigned arrival;
- repeatedly add the remaining request that minimizes the resulting batch's
  total extra selected bits (formula below), breaking ties by arrival position;
- finalize at the requested batch size, then start from the next earliest
  unassigned arrival. Keep the short last batch. At batch size 1, return arrival
  grouping unchanged with zero wait.

This is an offline partitioning calculation, not a scheduler or serving engine.
One heuristic, one window, no sweeps, global sorting by profile, or post-result
tuning. It assumes every route in the entire window is already available for
free. Operationally producing those routes would require the unchanged fixed8
probe; that cost and feasibility are unknown here.

Record signed displacement `output_position - arrival_position`, and maximum
absolute displacement; it must be at most 15. Record **waiting only in request
positions**: arrival grouping waits to the latest arrival in its batch;
compatibility grouping (except size 1) waits to the last arrival in its fixed
window. For query i, readiness wait is `frontier_position - arrival_position`,
0..15 for compatibility. This is not real latency and excludes service time,
probe time, queueing, timestamps and inter-arrival intervals. Report additional
readiness positions versus the corresponding arrival grouping. Reordered output
positions do not represent wall time.

## 4. Exact accounting and output definitions

Let block j contain n_j quantized parameters, N = sum n_j, and query i request
b_ij. Requested selected bits R_i = sum_j n_j*b_ij and requested bits/parameter
r_i = R_i/N. For batch B, m_Bj = max_{i in B} b_ij, shared selected bits
M_B = sum_j n_j*m_Bj, shared bits/parameter = M_B/N.

For each member i: extra selected bits E_i = M_B - R_i; extra bits/parameter
= E_i/N; **extra percent = 100*E_i/R_i**. Batch total extra =
`len(B)*M_B - sum(R_i)` (this is the greedy criterion). This allocates the shared
precision to every member for interference accounting; it does NOT claim the
weights are transferred/stored once per query. Also report the unique shared
payload M_B once per batch, separately from request-attributed quantities.

A block is "promoted to 8" for query i iff b_ij < 8 and m_Bj = 8. A batch block
is promoted to 8 iff its maximum is 8 and some member requests less than 8;
a block unanimously requesting 8 is not a promotion. Record both counts and
block IDs, and per-query promoted parameter counts. Per-block contributions are
sum_i n_j*(m_B(i),j - b_ij); their sum must equal total extra selected bits.

Outputs retain every request ID/profile/requested bits; each batch's member IDs,
maximum profile/payload; per-query extra bits/percent and promotion/block data;
per-block excess; and ordering/window/displacement/wait evidence. Summaries
include min/median/arithmetic mean/max over requests for bits, excess, promotions,
readiness/displacement; analogous batch-level distributions for shared payload,
within-batch unique profiles, varying blocks and promotion counts. Median for
even counts is the arithmetic mean of the middle two. No bootstrap, p-values,
population inference, quality re-evaluation, or selection among summary metrics.

Profile diversity is number of unique complete profiles, their frequencies,
largest fraction, per-block counts at 4/6/8, and varying block IDs/counts split
by attention/FFN, both population-wide and within batches. Size trends are the
full 1/2/4/8 table, not a selected favorable size.

## 5. Frozen decision (adaptive pooled population only)

"Small variation" means failure of the inherited minimal noncollapse rule:
fewer than 4 unique profiles, largest fraction >0.9, or fewer than 2 varying
blocks in either block type. This is only a permissive variation screen, not
evidence of batching benefit.

For each of batch sizes 4 and 8 separately, in **all three** fixed orderings:
- arrival median query extra percent must be **at least 10%**;
- recovery = arrival median extra percent minus compatibility median extra
  percent must be **at least 5 percentage points**;
- every request must remain within its original 16-position window and maximum
  absolute displacement/readiness wait must be at most 15.

At least one same batch size must meet every condition, with non-small variation,
to recommend a separate GPU batch experiment. Do not mix orders/sizes/tasks to
pass. Secondary task/control results cannot rescue a failed primary gate.
Otherwise **stop the scheduler direction and report the negative result**,
distinguishing insufficient variation, excess, and recovery. Passing is only
permission to recommend a separately scoped measurement, never to implement or
run it in this goal.

Concentration: for each size 4 and 8, sum arrival excess per block over all three
orders; rank descending with block-index ties. "Most excess in a small group"
means the top **8 of 72 blocks contribute at least 50%** of total excess (zero
excess is not concentrated). Report top-eight and individual block shares and
per-order shares. If a passing size is concentrated, narrow the proposed next
question to those blocks rather than general scheduling. If the main gate fails,
concentration can motivate only a descriptive block-specific question, NOT a
GPU/scheduler recommendation. Do not remove blocks or recompute a new gate.

## 6. Tests, reproducibility, audit, and limits

Hand-written unequal-count profiles must establish exact maxima, requested and
extra bits, percentage denominators, 4→8/6→8 versus unanimous-8 promotions,
per-block sum conservation, median/recovery calculations, and known greedy
choices/ties. Require size-one identity, zero inflation for identical/static
profiles, no lost/duplicated requests including duplicate profiles and partial
windows, deterministic output, and strict repeat/hash failures. Test threshold
boundaries, same-size/all-order requirement and concentration/negative branches.

CLI must refuse collisions and paths outside the new raw tree. Freeze hashes
must match before every run/audit. Produce two independently executed analysis
outputs; require byte-identical deterministic measurements. Raw-output audit
rehashes inputs/source snapshots, recomputes all calculations, checks coverage,
and compares raw outputs exactly. Repeat the unchanged CPU checks via the safe
wrapper, focused tests and full suite, and `git diff --check`. Final
`BATCH_INTERFERENCE_AUDIT.md` maps every explicit requirement to real artifacts,
commands and inspected results; a green test suite alone is insufficient.

Measured here means deterministic logical quantities computed from previously
measured single-query routes/counts. Shared per-block maximum is an assumption:
no actual batched forward or output-quality measurement is performed. Excluded
FP16 weights, scales, padding, KV cache, activations, allocation, transfer traffic,
probe computation, changing routes under batched numerics, and kernel behavior
are not in the selected-bit denominator. These logical counts cannot establish
production throughput, latency, kernel speed, memory savings, or end-to-end value.
No GPU jobs, scheduler/serving implementation, KV-cache quantization, router
retraining, quantizer changes, new models/tasks, async loading or custom kernels.

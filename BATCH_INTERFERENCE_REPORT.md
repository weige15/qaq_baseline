# QAQ batch interference — negative next-direction result

## Material Passport
- Study: `qaq-batch-interference-v1`, CPU-only analysis of saved Qwen3-4B routes.
- Verification: two byte-identical runs, exact raw-output replays, independent
  arithmetic audit, and passing existing/new CPU tests.
- Provenance: AI-authored protocol, analysis, tests and report; no new model runs.
- Scope: descriptive logical precision accounting, not measured batched inference.

## Decision

**Stop the scheduler direction for this baseline under the frozen rule. Do not
recommend a GPU batch experiment from these results.** Adaptive profile variation
is not small, but median shared-profile inflation is only **5.075–6.000% at batch
4** and **6.824–8.538% at batch 8**. Neither size reaches 10% in any fixed ordering.
The one window-bounded compatibility heuristic recovers only **0.601–1.371
percentage points** at these sizes, versus the required 5 points.

The top eight blocks account for 39.02%/40.16% of summed arrival excess at sizes
4/8, below the predeclared 50% concentration threshold. Thus neither the general
GPU-experiment gate nor the narrower block-specific branch fires. No alternative
window, ordering, grouping, subset, threshold or router was searched afterward.
This negative research outcome completes the bounded analysis; it is not a claim
that precision-aware batching is never useful in other settings.

## 1. Frozen inputs and actual parameter accounting

The core and synchronous on-demand studies remain read-only. Their quality and
storage conclusions in `REPLICATION_REPORT.md` and `ON_DEMAND_REPORT.md` are not
revised. In particular, adaptive routing's previously documented weaknesses
against random/fixed4 quality controls are not resolved by this analysis.

Before additions, HEAD was `b59adbcffc90b870f014d22e142923d569596986`, matching
`qaq-baseline-closed-v1`, with empty status/diff. The initial snapshot hashes all
59 tracked files and 1,144 protected raw files, including the implementation and
closed reports. Prerequisite recorded raw hashes and all original CPU checks
passed before freezing the new protocol. See the final requirement map in
`BATCH_INTERFERENCE_AUDIT.md` for preservation evidence.

Freeze record: `results/batch-interference-v1/protocol-freeze.json`, created before
implementation and new aggregates. Frozen SHA256:
- `BATCH_INTERFERENCE_PROTOCOL.md`:
  `8bc1095cc93ea9a91fc7c4bbaf2acabea136fa12648c9d03e3882d7e54c44818`
- `configs/batch_interference_protocol.json`:
  `f817bbfbc6ec26c7e205b5e08da1c4d4685247bb8a32cfd00aa0266eea8f2776`

Both saved repeats of adaptive, static and random routes were verified before
use. All ordered `(task,index,profile)` records and all other route fields match
exactly except the explicitly excluded `probe_seconds` timer. The files have
different byte hashes because timers differ; each matches its own recorded hash.
There are 576 unique request IDs: 64 WikiText-2, 256 HellaSwag and 256 ARC-Challenge.
Repeated profiles/contexts are retained, not deduplicated. No new tasks or examples.

All 72 block counts agree across six saved result files and the hash-locked
integration gate. Its 252 measured projection counts independently agree with
the saved fixed8 weight shapes:

| Quantity | Value |
|---|---:|
| Attention block parameters, each of 36 | 26,214,400 |
| FFN block parameters, each of 36 | 74,711,040 |
| Quantized projection parameters N | 3,633,315,840 |
| Requested selected bits R, **every query of every saved policy** | 21,799,895,040 |
| Requested bits/parameter R/N | 6.0 |

Weights are not loaded. Counts come from actual saved `q.numel()` measurements,
not assumed equal-size blocks or an unweighted 72-block average. Embeddings,
head, norms, scales and non-weight state are excluded from these denominators.

## 2. Method and assumptions

**Assumed shared-weight behavior:** choose the maximum requested precision
independently at each block. For a batch, M = sum of block parameter count times
that maximum. Query extra bits are M−R; extra percent is **100(M−R)/R**, not the
percentage increase in numerical bit width averaged equally across blocks.
Each batch's M is also retained once as its unique logical shared payload.
Summing member-attributed excess is an interference statistic, not an assertion
that weights are stored/transferred separately for every member.

The three fixed synthetic arrival orderings are saved order, reverse order, and
SHA256 order of `qaq-batch-v1|task|index`. Each population is filtered before
ordering. Arrival grouping takes consecutive requests. Compatibility grouping
uses non-overlapping 16-request windows: seed each batch with the earliest
unassigned request, then greedily minimize resulting total extra selected bits,
with ties by arrival position. It never crosses the window. Batch size one is an
identity. This is an **offline partitioner**, not a scheduler implementation.

Primary inference is confined to adaptive routing on the pooled 576 requests.
Each size 4 or 8 must meet both thresholds in all three orders at that same size.
Other policies/tasks cannot rescue a failed primary gate. Raw output contains the
complete 3-policy × 4-population × 3-order × 4-size × 2-grouping grid: **288
scenarios**, 82,944 request records and 38,880 batch records.

## 3. Confirmed descriptive measurements

These are computations on saved single-query measurements under the shared-max
assumption, **not observed GPU batch measurements**. Percentages below are rounded
only for display; decisions used the unrounded values.

### Profile diversity

| Policy / population | Queries | Unique profiles | Varying attention / FFN blocks |
|---|---:|---:|---:|
| Adaptive / pooled | 576 | 568 | 27 / 29 |
| Adaptive / WikiText-2 | 64 | 63 | 19 / 21 |
| Adaptive / HellaSwag | 256 | 251 | 25 / 25 |
| Adaptive / ARC-Challenge | 256 | 255 | 26 / 27 |
| Static / pooled | 576 | 1 | 0 / 0 |
| Random / pooled | 576 | 575 | 36 / 36 |

Adaptive's largest pooled profile frequency is 3/576 (0.5208%). Sixteen blocks
never vary across the pooled requests. Every saved query still obeys 12 blocks
at each of 4/6/8 bits in each block type. Many distinct complete profiles do not
imply large weighted disagreements or recoverable batching inflation.

### Primary gate: median per-query extra selected bits

| Batch size | Ordering | Arrival extra % | Compatibility extra % | Recovery, percentage points |
|---:|---|---:|---:|---:|
| 1 | all three | 0.000 | 0.000 | 0.000 |
| 2 | saved | 2.778 | 1.852 | 0.926 |
| 2 | reverse | 2.778 | 1.852 | 0.926 |
| 2 | sha256 | 3.223 | 2.092 | 1.130 |
| 4 | saved | 5.075 | 3.944 | 1.130 |
| 4 | reverse | 5.075 | 3.704 | 1.371 |
| 4 | sha256 | 6.000 | 4.630 | 1.371 |
| 8 | saved | 6.824 | 6.000 | 0.824 |
| 8 | reverse | 6.824 | 6.223 | 0.601 |
| 8 | sha256 | 8.538 | 7.371 | 1.166 |

Both required thresholds fail in all six primary size/order comparisons. Window
compliance passes, and the permissive profile noncollapse screen passes. The
failure is insufficient excess **and** insufficient recovery, not lost requests
or a collapsed router.

### Shared maximum profiles, absolute extra bits and promotions

Ranges below span the three fixed pooled adaptive orders, not confidence
intervals. Each group's median is over its 576 member queries. All batches are
full in these populations; there are 576/288/144/72 batches at sizes 1/2/4/8.

| Size / grouping | Median shared bits/parameter | Median extra selected bits/query, billions | Median blocks promoted to 8/query |
|---|---:|---:|---:|
| 1 / both | 6.0000 | 0 | 0 |
| 2 / arrival | 6.1667–6.1934 | 0.6056–0.7025 | 3–4 |
| 2 / compatibility | 6.1111–6.1255 | 0.4037–0.4561 | 2–3 |
| 4 / arrival | 6.3045–6.3600 | 1.1062–1.3081 | 6–7 |
| 4 / compatibility | 6.2222–6.2778 | 0.8074–1.0093 | 5 |
| 8 / arrival | 6.4095–6.5123 | 1.4877–1.8612 | 8–9 |
| 8 / compatibility | 6.3600–6.4423 | 1.3081–1.6069 | 7–8 |

A promotion means a query requested 4 or 6 but the batch maximum is 8; unanimous
8-bit requests are not promotions. At size 8 the maximum observed per-query
promotion count is 14 for both grouping approaches. Promotion identities,
parameter counts, per-block excess, every query's requested bits, and every
batch's maximum profile and total M are retained in `analysis.json`.

Mean varying-block counts within arrival batches rise from 0 at size 1 to
11.41–13.49 at size 2, 20.56–24.54 at size 4, and 27.93–33.21 at size 8.
Compatibility reduces these ranges to 8.85–9.91, 18.31–20.63, and 26.65–30.15.
Full profile frequencies, per-block 4/6/8 histograms, within-batch diversity and
min/median/mean/max statistics are retained, not reduced to the gate alone.

### Controls and task-specific results (not decision substitutes)

| Population / policy | Size-4 arrival median extra % | Size-8 arrival median extra % |
|---|---:|---:|
| WikiText-2 / adaptive | 4.028–4.269 | 5.297–5.760 |
| HellaSwag / adaptive | 4.389–4.834 | 6.445–6.686 |
| ARC-Challenge / adaptive | 5.898–6.000 | 7.852 |
| Pooled / static | 0 | 0 |
| Pooled / random | 26.166 | 32.065 |

Static has exactly zero inflation/promotions at every size, order and population.
Random's larger disagreements show that the accounting can detect substantial
inflation, but they are not evidence that the adaptive policy needs a scheduler.
Random compatibility medians remain 23.665–23.870% at size 4 and 30.351–30.796% at
size 8. All secondary compatibility and size-1/2 results remain in the raw grid.

### Request-position waiting and displacement — not real latency

Compatibility assumes all 16 routes are known before grouping. For every size
above one its readiness wait has mean/median **7.5 request positions**, maximum
**15**. At size one both methods wait zero.

| Size | Arrival mean / maximum readiness positions | Compatibility extra mean / maximum positions vs arrival |
|---:|---:|---:|
| 2 | 0.5 / 1 | 7 / 14 |
| 4 | 1.5 / 3 | 6 / 12 |
| 8 | 3.5 / 7 | 4 / 8 |

Maximum absolute output displacement observed for pooled adaptive compatibility
is 14 positions at sizes 2/4/8, within the declared bound of 15. Every request
stays in its original window; all 288 scenarios preserve each request exactly
once. No conversion to milliseconds, throughput, or actual queueing latency is
possible from these position counts.

### Block concentration

Sum arrival per-block excess over the three fixed orders; rank by excess, ties
by block index. The frozen small-group test is top 8 of 72 contributing ≥50%.

| Size | Top-eight block IDs (zero-based, alternating attention/FFN) | Combined share | Per-order top-eight share range |
|---:|---|---:|---:|
| 4 | 35, 37, 25, 63, 47, 59, 27, 17 | 39.02% | 38.82–40.58% |
| 8 | 25, 47, 35, 63, 59, 37, 53, 23 | 40.16% | 40.30–41.67% |

These listed blocks are all FFNs. Across all 36 FFNs the share is 78.91%/81.11%
at size 4/8, but half the blocks is not the frozen "small group," and FFNs have
more parameters per block. No small-group concentration revision is warranted
under the declared criterion. The per-order rankings can differ from the
combined ranking; their shares are therefore not required to bracket its share.

## 4. Verification, limitations, and unknowns

### Verified
- All 36 frozen input hashes pass on each run/audit, including prior recorded
  routes, checkpoints, gates, counts and source/config inputs.
- 17 focused tests pass: unequal weights/denominators, 4/6→8 promotion semantics,
  exact known greedy choices/ties, size-one identity, identical/static profiles,
  partial windows, duplicate profiles with distinct IDs, strict repeat/hash
  failures, deterministic output, thresholds and non-rescue/concentration rules.
- Full suite: **41 tests pass** (24 unchanged + 17 new). The separate existing
  on-demand CPU gate also passes its 4 targeted tests and the 41-test full suite.
- Existing baseline/router/on-demand gates recompute JSON-exact to the originals
  through the write-blocked redirection wrapper. Original gates are not rewritten.
- Two fresh CPU analysis processes produce byte-identical `analysis.json` and
  `summary.json`; each passes `--audit` input/source/output hashing and exact
  recomputation. A separate stdlib script, not importing the analysis module,
  checks every one of the 82,944 query and 38,880 batch records, greedy choices,
  coverage, conservation, position waits and the negative decision.
- No model run, scheduler, serving engine, quantizer/router modification or
  performance optimization was performed. Final preservation and whitespace
  evidence is linked from `BATCH_INTERFERENCE_AUDIT.md`.

### Assumptions and limitations
- Routes measured individually might change under batched numerical computation.
  A shared maximum profile might change losses, especially for mixed profiles;
  neither batched outputs nor quality are measured here.
- Compatibility has free advance access to every profile in its window. The
  real router's fixed8 probe, its batching, and the resulting dependency/wait cost
  are not modeled. The hypothetical grouping may be impractical operationally.
- Pooled task proportions, offline ordering and fixed windows are synthetic.
  Reverse and saved orders have identical arrival batch member sets here because
  population lengths divide the batch sizes; they are not independent replicates.
  SHA order changes membership and mixes tasks. No confidence intervals, p-values
  or generalization to production traffic are claimed.
- This one greedy heuristic is not an optimal compatibility bound. The negative
  result stops the frozen direction; it does not prove no other grouping works.
- Bit counts exclude FP16 exclusions, scales, padding, KV cache, activations,
  allocator/cache behavior, transfers, probe work and kernel overhead. Logical
  maximum-profile bits cannot establish GPU memory, kernel speed, real latency,
  throughput or end-to-end gains. Original on-demand batch-one timings must not
  be extrapolated into batch performance.
- No GPU jobs, scheduler/serving code, KV-cache quantization, router retraining,
  quantizer changes, new model/task, asynchronous loading or custom kernel is
  authorized or implemented by this study.

### Failures and deviations
No raw input was missing or hash-mismatched, and no experiment/test failed.
Before the first aggregate, repeat validation was tightened to reject numeric
profile type drift, and one non-rescue test was added (16 initial, then 17 final
focused tests); neither change used batch outcomes. Both pre-analysis logs remain.
The protocol, window, heuristic and decision rule never changed after freezing.
Output-writer interception for the closed checkers is a preservation adaptation,
not an altered gate or regenerated closed artifact. No experimental search follows
this negative result.

The first final-preservation helper falsely flagged clean new files because it
misread `git diff --no-index --check` exit 1 (files differ, no diagnostics) as a
whitespace failure. All protected hashes and Git boundaries already passed.
`preservation-before-final-audit.json` is retained. Clean and deliberate
trailing-space fixtures establish exits 1/3 respectively; append-only
`preservation_audit_v2.py` corrects only that interpretation and passes. No
protocol, measured output, analysis code, test or closed file was changed.

## 5. Artifacts and commands

From the repository root, using the existing environment (no new dependencies):

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
~/.venv/bin/python -m unittest tests.test_batch_analysis -v
~/.venv/bin/python -m unittest discover -s tests -v
# Read-only replays of already completed new analysis:
~/.venv/bin/python scripts/analyze_batch_interference.py --audit results/batch-interference-v1/run-1
~/.venv/bin/python scripts/analyze_batch_interference.py --audit results/batch-interference-v1/run-2
~/.venv/bin/python results/batch-interference-v1/independent_audit.py
# Only if another identical CPU replay is wanted; path MUST be unused:
~/.venv/bin/python scripts/analyze_batch_interference.py --out results/batch-interference-v1/NEW-UNUSED-RUN
# Safe recheck of old studies; path MUST be unused:
~/.venv/bin/python results/batch-interference-v1/preflight/run_closed_checks.py results/batch-interference-v1/NEW-UNUSED-CHECK
# Optional existing CPU gate, also redirected to a new unused path:
~/.venv/bin/python scripts/check_on_demand_cpu.py --out results/batch-interference-v1/NEW-UNUSED-CPU-GATE
git diff --check
```

Do not run the unwrapped baseline/router/on-demand verifier CLIs: they write
closed gate files. Do not reuse any completed output path or recreate missing
inherited artifacts. Scripts refuse output collisions and paths outside this
study's raw tree. Raw artifacts are git-ignored; a code-only clone is not the
evidence bundle. No external backup/export was made in this goal.

| Artifact | Path under `results/batch-interference-v1/` |
|---|---|
| Initial state, hashes, old CPU prerequisites | `preflight/` |
| Frozen protocol seal | `protocol-freeze.json` |
| Complete request/batch/per-block measurements | `run-{1,2}/analysis.json` |
| All summaries, distributions and decision | `run-{1,2}/summary.json` |
| Exact commands, input/output hashes and five source snapshots | `run-{1,2}/manifest.json`, `run-{1,2}/source/` |
| Executed analysis and replay audit logs | `run-{1,2}-launch.log`, `run-{1,2}-audit.log` |
| Independent all-record arithmetic check | `independent_audit.py`, `independent-audit.log` |
| Final closed gates / 41-test suite | `final-closed-checks/` |
| Separate existing targeted/full CPU gate | `final-on-demand-cpu/` |

Both runs' deterministic output SHA256:
- `analysis.json`: `76ab842990fc74f42719c9c9cafec36afa31c233b7d3382ea12e0ff8c5a5d8d9`
- `summary.json`: `3c349ee10adfcee01adbd540521de6a5cee1d9f0b97c71530214d84131b35519`

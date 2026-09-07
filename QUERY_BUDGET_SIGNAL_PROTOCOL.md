# Fixed6 bridge and cheap query-budget signal study v1

## Status and preservation

Phase A is frozen before **any new fixed6 quality result**, including smoke.
This is a bounded follow-up, not a scheduler or serving implementation.
Authoritative machine settings: `configs/query_budget_signal_protocol.json`.
A new exclusive-create receipt under `results/query-budget-signal-v1` seals this
protocol, config, initial inventory, runner, analysis and tests before GPU work.
There are no protocol commits: the user explicitly prohibits commits/pushes.

Initial clean parent is pushed `research/query-budget-feasibility-v1` HEAD
`7160dbfa0622a7557541a717f36bb62963435cd2`, verified directly with `git ls-remote`.
New branch: `research/query-budget-signal-v1`, at exactly that HEAD. Preserve
`qaq-baseline-closed-v1` (object and peeled commit), all 66 initial tracked files,
and all 1222 files under core-v1, on-demand-v1, batch-interference-v1 and
query-budget-feasibility-v1. The pre-edit inventory records SHA256, size, mode,
mtime and refs in `results/query-budget-signal-v1/query_budget_signal_initial.json`.
Historical raw inventories, endpoint input hashes, feasibility freeze, source
manifests and repeated analysis agreement were checked before branch creation.

Only new study-specific filenames containing `query_budget_signal` and the three
uppercase protocol/report/audit files are permitted. All raw output files also
contain `query_budget_signal` and live in new paths under
`results/query-budget-signal-v1`. Exclusive creation prevents run replacement.
No tracked edits, installs/upgrades, commits, pushes, merges, pull, reset, stash,
delete, cleanup or repair of closed evidence. Set `PYTHONDONTWRITEBYTECODE=1`.

## Phase A: actual fixed6 bridge

Use the exact 576 frozen examples (WT2 64, HellaSwag 256, ARC-Challenge 256),
model and tokenizer revisions, quantizer, group size 128, context/choice encoding,
causal loss implementation, FP16 SDPA, batch size 1, seed 1729, four CPU threads,
TF32 disabled and deterministic torch algorithms from the closed core protocol.
Hash-check frozen inputs, all model/tokenizer files and frozen runtime versions.
The new runner imports the unchanged scorer and `apply_fixed(model, 6)`;
`scripts/run_core.py` is never modified. All seven projections of all 36 layers
are six bits (252 tensors, 3,633,315,840 parameters). Embedding/head/norm/bias
exclusions remain FP16. Weight-bit accounting is logical projection precision,
not measured storage, throughput, latency or memory reduction.

Before full execution:
1. CPU-test signed 6-bit midpoint reconstruction, zero groups, all int8 codes,
   parameter accounting, excluded tensors, and independent/nested agreement.
2. Smoke on the first two frozen IDs in each task (frozen in config). Prepare
   nested codes/scales BEFORE applying fixed6 to avoid requantization. Compare
   every one of the 252 independently materialized fixed6 tensors exactly with
   nested reconstruction. Evaluate all six complete examples independently and
   through the integrated all-six profile. Require byte-identical raw scores,
   token losses and predictions. Keep both outputs.
3. Measure separate task evaluation times. Full projection is twice the measured
   independent-path setup plus sum of task smoke seconds * full-count/2 (2x safety
   factor on both); nested verification overhead is conservatively included in
   setup. Pause if a projected job exceeds 1800 seconds or cumulative consumed
   job walltime + pending projection exceeds 21600 seconds (six GPU-hours).
4. Run two fresh full fixed6 processes, **serial**, each through the unchanged
   GPU preflight and `timeout --signal=TERM --kill-after=10s 1800s`. Never bypass
   an occupied-GPU refusal. Log command, hardware, hashes, per-task/full times,
   all per-query token losses/summed and normalized choice scores. Repeat gate:
   identical ordinary and normalized predictions, WT2 mean-NLL delta <=1e-5,
   maximum per-choice summed-logprob delta <=1e-3. Report whether raw records are
   also byte-identical. First full process is the prespecified primary result;
   repeats are not independent observations.

### Comparisons at exactly six average logical bits

Analyze each task separately; no pooled metric. Compare actual uniform fixed6
with (i) exact-half query-independent fixed4/fixed8 **expectation**, (ii)
outcome-informed endpoint selection (analysis-only ceiling), (iii) previously
frozen router score selection. Also report prompt length, the three frozen
ID-hash rankings, and existing adaptive/static/random mean-six policies as
secondary comparisons, including ordinary MC accuracy. Hash-verify secondary
samples, metadata, routes and repeats; recheck their per-query bit accounting.
Do not rerun or retrain old policies. Their full fixed8 probe cost is not charged
by this logical accounting, so quality tables do not establish cost parity.

Reuse only the existing frozen endpoint arithmetic/ranking specification:
WT2 ranks descending NLL4/token - NLL8/token, retaining negatives; MC ranks
normalized correctness8 - correctness4, ties by task then integer index.
Router score is `fsum(parameters_b*(cost4_b-cost8_b))/sum(parameters_b)` with
no feature changes. Prompt length is descending. ID-hash salts are exactly
1729, 2718, 31415 and ascending SHA256(`{salt}:{task}:{index}`). Select exactly
n/2 fixed8 queries including forced zero/negative-benefit selections; all others
fixed4, complete unique IDs, total query bits = 6*n. Uniform subset expectation
uses probability 1/2 for each query, not fractional actual execution.

Primary Phase-A improvement:
- WT2: actual fixed6 mean NLL minus outcome-informed mixture mean NLL.
- MC: outcome-informed mixture acc_norm minus actual fixed6 acc_norm.

All beneficial-direction deltas follow these signs. WT2 always divides summed
NLL by summed scored tokens, including every bootstrap draw. Use 10,000 paired
query resamples, NumPy default_rng/PCG64 seed 4242 reset per task;
`rng.integers(0,n,size=(10000,n))`. Keep all endpoints, six-bit outcomes and
scores paired. Re-rank each draw using unchanged scores, select exactly half
fixed8 **occurrences**; duplicate-ID ties use draw position after task/index.
Same draws for all comparisons, including secondary policies. Percentile 95%
interval with NumPy linear quantiles. Never bootstrap a fixed membership list
with a fluctuating quota. Oracle reuse of outcomes in resampling is not evidence
of out-of-sample generalization. Report expectation-vs-six and all method-vs-six
and method-vs-expectation effects/intervals, with negative effects intact.

A task passes iff unrounded primary point >=0.01 and 95% lower bound >0.
- Two or three pass: `continue_phase_b` (not a final positive study result).
- Exactly one: `revise_task_specific`; stop the broad signal study.
- Zero: `stop_endpoint_budget_direction`.

A negative Phase-A decision completes this bounded scientific study after audit;
it stops only whole-query fixed4/fixed8 endpoint routing, not all variable-budget
methods. No Phase-B features, labels, fitting or final evaluations are required
or authorized after a Phase-A stop/revise decision.

## Phase B: conditional, separately frozen before endpoint collection

Only `continue_phase_b` authorizes a new immutable Phase-B freeze receipt with
exact IDs/token-span checks, feature formulas, split hashes, smoke jobs, cost
projection and implementation hashes. Never use the old closed 576 queries for
fitting, feature choice, normalization, thresholds or final signal evaluation.
Use frozen dataset revisions and tokenizer. Exclude prior final IDs and use
existing exact token-span duplicate checks; pause if the exact counts cannot be
obtained (no smaller/replacement datasets).

| Task | Train | Development | New final |
|---|---:|---:|---:|
| WT2 | 192 prior frozen router-training windows | 32 prior frozen router-development windows | 64 unused test windows |
| HellaSwag | 384 training-split examples | 128 training-split examples | 256 unused validation examples |
| ARC-Challenge | 384 training examples | 128 validation examples | 256 unused test examples |

Exactly three signal families, using only WT2 first 128 tokens or MC common
question/context, never suffix/candidates/scores/predictions/gold/ID/benefit:
1. `surface`: signed unigram/bigram hashing into 256 dimensions, L2 normalized,
   plus explicitly frozen prompt surface statistics.
2. `embedding`: frozen input lookup only, no transformer blocks; 64 contiguous
   channel groups summarized by mean/RMS plus the four existing global stats.
3. `combined`: concatenation of surface and embedding.

Controls only: prompt-token count, exact-quota expectation, three declared
ID-hash rankings. IDs cannot be trainable inputs. No early transformer layer or
full fixed8 probe additions. Exact surface/hash conventions and existing global
stat formulas will be sealed independently before seeing new endpoint labels.

Generate fixed4/fixed8 outcomes for train/dev and actual fixed6 for dev. Labels:
WT2 per-token NLL4-NLL8 with negatives retained; MC normalized gold-margin8 minus
normalized gold-margin4, margin = gold score - max(non-gold scores). Gold only
creates training targets. Fit deterministic per-task ridge per family with
train-only normalization, constant-coordinate masking, lambda {0.1,1,10} selected
by deterministic five-fold **training-only** CV; no other models/feature search.
Rank predicted benefit descending with fixed ties and exact-half quota.

Development: ceiling_gain and signal_gain are beneficial-direction gains over
actual fixed6; recovery = signal_gain/ceiling_gain only when ceiling_gain>0.
Family/task pass requires signal_gain>0 and recovery>=.25. A common family must
pass >=2 tasks. Choose highest median development recovery, exact ties in order
surface, embedding, combined. Nonpositive ceilings have undefined recovery and
cannot count as passes; handling in median is fixed in the separate B freeze
before outcomes. Seal chosen family, task coefficients/hashes and decision
before opening final outcomes. No eligible family => `stop_cheap_signal_v1`,
report task-specific indications separately, no final jobs or post-hoc changes.

Only development pass allows fresh final fixed4/6/8 jobs with fresh repeat
processes and the existing deterministic tolerances. Evaluate the frozen selector
once, never refit on final. Final task pass: signal gain over actual fixed6 >=.01,
paired 95% lower>0 and recovery>=.25 of the final outcome-informed ceiling. Report
all tasks and harms. >=2 passes => `continue_budget_block_integration` (only a
later QAQ block-allocation study, no serving authorization); exactly one =>
`revise_task_specific` (broad stop); zero => `stop_cheap_signal_v1`. This resolves
the overlapping '<2 stop' wording by retaining the explicitly named one-task
revise decision while marking the broad direction stopped.

Smoke and project runtime before all full label generation. Cumulative projected
GPU work, including Phase A, every smoke, repeats and any GPU extraction, must
not exceed six hours. All GPU jobs serial with preflight and 30-minute timeout.
Measure signal extraction with warmup and deterministic repeats separately from
one-time loading. No end-to-end timing, throughput, memory or serving claims.

## Verification and deliverables

Run every existing CPU unittest plus focused new tests with GPU hidden. Phase A
tests cover 6-bit reconstruction/accounting, negative benefits, signs, exact
quotas, ties, WT2 unequal-length arithmetic, repeat tolerances, bootstrap
occurrences, hash/ID checks, append-only path safety and decision boundaries.
If B is authorized, additionally test feature causality, exact split separation,
hashing, train-only normalization, constant masks, ridge/CV arithmetic, negative
labels, family choice/recovery and every B gate. Do not pretend unexecuted B
checks passed after an A stop.

Two fresh deterministic analyses must be byte-identical. Add a separate
arithmetic/selection audit which imports neither new nor old main analysis code:
read raw outcomes, reconstruct metrics/ranks, verify all six-bit memberships,
recalculate paired resampling/decisions and compare both analyses. An independent
read-only review may inspect requirements/implementation. Rehash every protected
file and compare inventories, refs and allowed additions. Run `git diff --check`
and explicit no-index whitespace checks for untracked files. Record actual
command exits and test counts, not a proxy 'green' status.

Deliver `QUERY_BUDGET_SIGNAL_PROTOCOL.md`, machine config,
`QUERY_BUDGET_SIGNAL_REPORT.md` and `QUERY_BUDGET_SIGNAL_AUDIT.md`. The audit maps
every requirement and conditional branch to actual evidence or explicit gated
non-applicability. Report confirmed results, assumptions, negative findings,
unknowns and unsupported system claims separately. Pause only for invalid or
unavailable required evidence/data, preservation mismatch, GPU refusal or
projected compute beyond bounds; record exact blocker without destructive repair.

# Router and equal-bit protocol — v1

Status: freeze this protocol/config/data before label collection or fitting.
Extends (does not replace) `EVALUATION_PROTOCOL.md` and the passed baseline and
integration gates. The current continuation supplies the next work budget.
No exact paper table, additional model, loader, or kernel work is authorized.

## Hypothesis and boundaries

A small block-specific MLP can learn query-conditioned **local FP16-to-quantized
block output error**, and allocate a fixed six-bit average more effectively than
one train-optimized static assignment. This local distillation proxy is our
choice, not the paper's unspecified end-to-end distillation loss. A negative
final comparison ends the search; implementing the mechanism does not establish
its quality benefit.

## Data frozen without model scores

Use the existing WT2 revision `b08601e04326c79dfdd32d625aee71d232d685c3`:
`wikitext-2-raw-v1/train-00000-of-00001.parquet` for fitting and
`wikitext-2-raw-v1/validation-00000-of-00001.parquet` for development.
Split rows into articles at top-level `= Title =` headings. Join rows within
an article using two newlines. Never join across articles. Candidate articles
must contain at least512 tokenizer tokens. Use one complete512-token window
per selected article, randomly chosen from its nonoverlapping complete windows.
Use seed2718, independently reset for each split; sample192 train and32 dev
eligible articles. Preserve sorted article order in the final manifest. Refuse
if fewer eligible articles remain; do not silently reduce or resample by quality.

Decontamination: exclude train/dev articles with normalized title or entire-text
hash matching a WT2 test article; exclude any candidate window sharing any exact
32-token span with ANY frozen final context/continuation (including all MC
options). Exclude train articles whose title/text hash matches any dev article.
Reject dev windows overlapping selected train windows by32 tokens. Save rejected
IDs/reasons, all source hashes, selected article boundaries, window offsets,
raw text, full512 token IDs, context hashes, and a recomputable zero-overlap audit.
This cannot detect all semantic paraphrases or pretraining contamination.
Final data are accessed only for this deterministic exclusion check, not fitting,
normalization, labels, checkpoint selection, or collapse repair.

Use only the first128 tokens for features and local-error targets. The rest are
reserved for dev-only causal/mechanism checks, never labels. Final scorer remains
unchanged: WT2 context128, MC entire context, no answers/suffix/gold to router.

## Feature/label collection and memory

Load the verified single int8/scaled checkpoint and a frozen FP16 teacher of the
same revision on ONE guarded GPU. Perform a fixed8 context-only student pass.
At each normalized attention/FFN block input h, capture features and replay that
one block at4/6 bits and in the FP16 teacher using identical h, causal mask,
position embeddings and no cache. Compare each student block output with the
FP16 output: `mean((y_b-y_FP16)^2) / max(mean(y_FP16^2),1e-12)`, FP32 arithmetic.
The original8 output continues the unchanged fixed8 trajectory. Clear transient
block reconstructions after local replays. No FP16 teacher trajectory or suffix
enters the router. No backbone gradients, quantizer retraining or multiple stored
precision variants. Save all72x3 target errors and features per context.

Raw features132/block: mean over tokens and each of64 contiguous channel groups;
RMS over the same groups; global RMS; global maximum absolute activation;
standard deviation (population) of per-token RMS; natural log(context length).
All computed FP32. Feature normalization (per block/channel) is train-only,
std floor1e-5. Coordinates with train std<1e-5 are masked to zero in BOTH fitting
and inference, including the constant128-token log-length feature. This fixes
the independent review's P1 untrained-weight amplification on variable-length
MC queries; a synthetic out-of-range constant-feature test is required.
Every block owns its own small two-layer MLP with ReLU.

## Three ordered, bounded attempts (no search for final scores)

Common: CPU FP32 training, four threads, seed31415+attempt number, AdamW
lr0.003, betas(0.9,0.999), eps1e-8, weight_decay0.01, full batch192,
300 epochs, constant LR, gradient norm clip1.0; keep final epoch only.
Save every epoch loss, initial/final train and dev losses, all state/normalizers,
profiles and seed. No early stopping or best-dev-quality selection.

1. **A1**: aggregate the raw64 channel groups into32, means by average and RMS
   by root-mean-square; retain4 scalars (68 inputs), hidden32,3 outputs.
   MSE to `log(error+1e-12)`, targets centered/scaled per block/precision using
   train only (std floor1e-5). Invert target normalization then exponentiate
   (clamp predicted log to[-30,10]) for precision-assignment costs.
2. **A2**, only if A1 fails noncollapse: raw132 inputs, hidden64; otherwise A1.
3. **A3**, only if A2 fails: raw132 inputs, hidden64; cross-entropy to train
   oracle quota assignments minimizing actual local NMSE. Assignment costs at
   inference are negative log-softmax (temperature1). Static comparator remains
   optimized from mean raw train NMSE, not a deliberately weakened classifier.

A fit passes iff finite, final training loss below initial, >=4 distinct dev
profiles among32, largest profile frequency<=90%, >=2 attention blocks and>=2
FFN blocks vary across dev queries. Every profile must have12 each4/6/8 in each
block type. This diversity is forced by quotas: report it as such. Only variation
between deterministic query-conditioned profiles counts as learned adaptation.
First passing attempt is irrevocably selected. Otherwise stop with collapse
failure and no final adaptive quality claim. No random tie noise, query ID or
hash feature in adaptive inference. Same features must yield the same profile;
replacing all dev features by their train mean must yield one constant profile.

## Exact allocation, controls, genuine-use gate

Minimize sum of predicted costs under12 each4/6/8 among attention blocks and
independently among FFNs via installed SciPy `linear_sum_assignment`, a36x36
matrix with12 slots per precision. Ordered rows/columns give deterministic
solver tie handling; no cost perturbations. Unit-check optimality by brute force
on a small instance, quotas and parameter-weighted accounting. Fixed4/8 are
endpoints, not equal-cost comparisons. Static minimizes sum of mean raw **train**
NMSE with the same solver. Random permutes each type's multiset using seed1618
plus SHA256 of context tokens, independent of labels, fixed across repeats.

All adaptive/static/random controls run the SAME fixed8 context feature pass
(the latter two discard features). Report scoring-stage quantized weight bits,
FP16 exclusions and scale overhead separately. Thus the extra pass cost is equal
among matched controls, although these reference implementations do NOT claim
speed/memory savings. Also report probe tokens and total forward-input tokens,
so bits weighted by token work (8-bit probe +6-bit scoring) are auditable.

Before final scoring, use dev only for a full-model gate: recompute cached
features, exact deterministic profiles, suffix/answer/gold perturbation through
the real scoring API, and inspect every executed NestedLinear's bits and actual
reconstruction. On3 dev contexts compare last-token logits under learned,
fixed4 and fixed8 profiles; require finite unequal outputs, and raw selected
weights equal their requested reconstructions. Save raw logits/traces/profiles.
A1/A2/A3 selection may not be revised based on these logits or quality.

## Final runs and decision (already declared in core protocol)

After gates, one full run and one exact-repeat run each of adaptive, static and
random on the original576 frozen examples. No alternative random seeds or
adaptive attempts after final scoring. Fixed4/8 endpoints use this project's
fresh repeat-verified core-v1 baselines, not old/external results. An integrated
fixed4/8 smoke on the original6 smoke examples must exactly reproduce those
baseline raw losses before final scoring. This is an implementation check,
not model selection. Save every returned metric and raw loss without filtering.

Paired bootstrap, seed4242,10000 draws, percentile95% intervals, same sampled
indices per task across comparisons. WT2 resamples64 windows; MC resamples256
examples. Compute all adaptive-minus-{static,random,fixed4,fixed8,fp16} deltas
for WT2 mean NLL/token PPL and both MC accuracies. Report limited power and
window dependence; these are subset, not full-benchmark claims.
Primary benefit requires WT2 NLL CI upper bound<0 versus static AND neither
MC acc_norm point delta<-0.02. Otherwise record negative/inconclusive and STOP
without further adaptive search. Repeatability: identical profiles/predictions,
WT2 mean NLL delta<=1e-5 and per-choice summed logprob delta<=1e-3.

## Bounded commands and artifacts

Every GPU process goes through `scripts/gpu_preflight.sh`, one idle process-free
GPU with>=20000MiB free, serial jobs,30min timeout. Collect a4-context smoke
(first2 train+first2 dev); project full224-context collection before running.
Pause if projected job>30min or peak own allocation>20GB. CPU fits have10min
cap each. Final6-example smoke projects576-example runtime before full jobs;
pause if>30min. The predeclared six comparison jobs are NOT a hyperparameter
sweep. Runtime/memory failures are retained, never hidden by selective samples.

Commands (run from repo after `source ~/.venv/bin/activate`; exact invocations
and preflight output additionally saved under `results/core-v1/logs/`):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/prepare_router.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
# Prefix each GPU command with this environment and guard:
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
 bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
 python scripts/router_job.py --stage collect --out results/core-v1/router-collect-smoke --smoke
# Same command without --smoke, output router-collect, after runtime gate.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src timeout 10m python scripts/train_router.py
# router_job.py --stage verify --out results/core-v1/router-verify
# router_job.py --stage evaluate --mode adaptive|static|random --out NEW [--smoke]
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
```

Each job records argv, interpreter, git/source snapshot, input/checkpoint hashes,
physical GPU/hardware, package versions, errors and timings. Artifacts live
under `results/core-v1/router-*` and `comparison-*`. Final `REPLICATION_REPORT.md`
separates paper statements, choices, measurements and unresolved questions.
The completion audit must inspect data provenance, actual trained weights and
profiles, weighted budgets, raw result aggregation/repeats, failures, source
scope and final report; green unit tests alone never establish completion.

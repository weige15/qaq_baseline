# Core reproduction protocol — v1

Status: **frozen before new GPU scores**. Authoritative machine settings:
[`configs/core_protocol.json`](configs/core_protocol.json). This supersedes the
Base-checkpoint, paper-matching, full-split, PTB and loading proposals in
`README.md`, `PAPER_AUDIT.md`, `DECISIONS.md`, `FIRST_GPU_RUN.md`, and
`configs/baseline_candidate.yaml`. Those remain historical source material.
The active user's literal target is **Qwen/Qwen3-4B**, not Qwen3-4B-Base.

## Claims and scope

Functional, not Table-1 reproduction. Table 1 is the page-4 comparison of all
five methods and three models. All five PDF pages were read before code changes;
source SHA256: `f15f1d6cbaaaf7ee74e8300b9a9d2b46d180a3962019cf3a8acc7de252c37c8e`.
The required audit/decisions/first-run/config/audit-script were also read.

No CPU-to-GPU on-demand loading, other models, Penn Treebank, dynamic batching,
asynchronous loading, custom kernels or performance claims. Ordinary initial
model loading is not an on-demand mechanism. Old `results/baseline`, `diagnostic`,
`control`, `smoke`, and `meta` artifacts are NOT new evidence. New evidence lives
under **`results/core-v1/`**, never overwriting old outputs.

The already cached requested checkpoint is public; its exact-revision official
model card declares Apache-2.0. No gated model, license acceptance or alternate
model decision is made. If access later demands new terms, pause.

## Frozen evaluation

- Model and tokenizer: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`. FP16, SDPA, eval/no-grad,
  no chat template/thinking generation, no BOS/EOS insertion, no few-shot.
- Batch size exactly 1, deterministic torch algorithms, TF32 disabled,
  four CPU threads, seed 1729. No silent truncation: fail on a boundary change or
  MC sequence over 2048 tokens.
- WikiText-2 **raw** test: join every row with two newlines, tokenize once,
  nonoverlapping 512-token windows, drop incomplete tail. Sample 64 complete
  windows using the frozen seed; first 128 tokens are context only, last 384
  scored causally. This is custom **prefix-conditioned token perplexity**, NOT
  harness word/byte perplexity or directly comparable to the paper's WT2.
- HellaSwag validation and ARC-Challenge test: 256 examples each, sorted seeded
  random indices. Use the reviewed harness HellaSwag preprocessing and ARC
  `Question: …\nAnswer:` prompt. Scores sum next-token log probabilities of
  each `space + answer`; normalized score divides by Unicode character count of
  answer excluding added space, as in lm_eval. Ties use first option.
- Every returned metric is saved. WT2: NLL sum, scored tokens, mean NLL,
  token perplexity, window-mean NLL SE. MC: acc, acc_norm, both sample SEs, count.
  Primary metrics declared now: WT2 mean NLL, MC acc_norm. No paper-based choice.
- Data revisions, source-file hashes, indices, token IDs and exact prompts are
  in `results/core-v1/frozen/`. `scripts/prepare_core.py` validates all tokenization
  boundaries before model scoring, records hardware, full environment, and hashes
  all model files. An output-directory collision fails rather than overwrites.
- Small benchmark subsets limit statistical power. No full-split inference or
  leaderboard claim is permitted. Samples are selected by seed, not quality.

Installed runtime retained without upgrades: Python 3.12.3, torch 2.5.1+cu121,
Transformers 5.16.1; all versions in frozen environment/pip-freeze. lm_eval
0.4.13 is a reviewed **prompt reference**, not our execution backend; its few
preprocessing rules are the only reused logic. No external model results used.

## Quantizer choice

Group contiguous input columns in groups of128 for every attention q/k/v/o and
FFN gate/up/down Linear. Scale = FP32 absmax/127 from FP16 source weights;
round ties-to-even and clamp -127..127, store signed int8. Preserve zero-group
scale=0; divide safely by1 during quantization only. Norms, embeddings, head and
biases stay FP16.

At b bits, s=8-b: reconstruct signed high bits by arithmetic shift, place the
value at the bin midpoint `(q >> s) * 2**s + (2**s-1)/2`, multiply FP32 scale,
cast FP16. At8 this is exactly chosen q*scale. At4/6 this depends ONLY on retained
high bits; no discarded-bit rounding or special-case correction of zero codes.
Zero codes in nonzero groups shift at lower precisions: this is disclosed,
intentional midpoint quantization. Name the baseline **nested midpoint 4-bit**,
not independently scaled conventional 4-bit RTN. Unlike the historical NumPy
sign-magnitude toy, this is a two's-complement representation.

Baseline jobs materialize a fixed FP16 reconstruction once. Later integration
must store one quantized checkpoint, and reconstruct selected block weights
from it; baseline success alone does not satisfy integration. Neither storage
of8-bit master weights nor FP16 matmuls imply 4-bit resident memory or speedup.

## Gates and bounded compute

1. CPU tests, input freeze, then six-example smoke per fixed mode. Estimate
   runtime before full runs. Per job limit30min; pause before projecting a full
   evaluation >30min or any unplanned GPU sweep. Inspect preflight each time;
   one idle process-free >=20,000MiB GPU only, no concurrent model jobs.
2. FP16/fixed8/fixed4 each receive two fresh-process runs on the complete frozen
   sample set. Predictions identical, mean NLL delta <=1e-5 and each summed
   choice log-likelihood delta <=1e-3. All finite. FP16 MC acc_norm >.30;
   fixed8 MC delta <=.08 and PPL ratio .8..1.25; fixed4 MC acc_norm >.25 and
   PPL ratio .5..3. These are broad internal-coherence gates, NOT paper matching.
   **Pause with exact mismatch on failure; do not train or relax thresholds.**
3. After baseline gate: integrated full8 torch.equal to independent reference
   for every weight, exact serialize/reload, lower4/6 change model logits,
   and independent attention/FFN selection. Revise representation if failing.
4. Before router work, freeze training/development text/document splits, hashes,
   dedup checks and at most three features/objective attempts. Use no final
   examples for fitting, feature normalization, selection or collapse repair.
   A proposed bounded source is WT2 train (fit), validation (dev), test (final).
   No fit has happened yet; details are intentionally a later gated stage.
5. Adaptive proposal: context-only fixed8 probe captures hidden features, a small
   MLP predicts local teacher-student quantization error, then assign exactly
   12 each of4/6/8 within attention and separately FFN. Freeze for scoring all
   options/suffix. This requires an extra pass and is not the paper's unspecified
   exact training method. Test suffix/answer changes cannot change routing.
6. Static: optimize same quota from train-only aggregate errors. Random: seeded
   assignments under same quotas. Check block sizes and actual per-query
   sum(parameters*bits), not only unweighted means. Fixed4/8 are cost endpoints;
   adaptive/random/static must have identical mean bits. Report FP16 exclusions
   and scale overhead separately. Quotas force precision diversity, so **learned
   variation across queries**, not just multiple levels, is the routing gate.
7. Select first noncollapsed predefined attempt using development only, evaluate
   final once (plus predeclared repeat). Primary success: paired-bootstrap95% CI
   for adaptive-minus-static WT2 mean NLL entirely negative, neither MC acc_norm
   regresses by>.02. Report all comparisons and paired uncertainty regardless.
   Otherwise stop with negative/inconclusive evidence, not an extended search.

## Commands and artifacts

All commands run from this repository. Preparation/test jobs are CPU-only:

```bash
source ~/.venv/bin/activate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/prepare_core.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/audit_table.py
```

One GPU command at a time (replace MODE with fp16, fixed8, fixed4 and use a new
output name for each smoke/full/repeat). Never bypass preflight:

```bash
source ~/.venv/bin/activate
bash scripts/gpu_preflight.sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
  bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
  python scripts/run_core.py --mode MODE --out results/core-v1/NAME [--smoke]
```

Capture shell command and preflight stdout/stderr to `results/core-v1/logs/`.
Each successful job records command arguments, source snapshot, frozen hashes,
physical GPU, runtime versions, raw token losses and all metrics. Exceptions
produce `failure.txt`; shell timeout/failure exit status also belongs in the log.
No sample outputs are filtered. Protocol commits precede GPU results.

## Initial review / failures

Independent read-only reviewer found necessary gates for causality, zero groups,
exact weighted budget, separate development selection and superseding historical
proposals; these are incorporated above. Review retained in `results/core-v1/meta/`.
Initial CPU suite had one **test expectation typo**: the official single
`.replace('  ', ' ')` leaves two spaces from four, not one. Expected string fixed
without changing preprocessing; original failing output retained. No model
scores existed at that point.

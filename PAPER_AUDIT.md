# QAQ paper audit

> Paper observations below remain source context. The active user's narrower
> functional goal supersedes the proposed reproduction and acceptance rules:
> see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). No paper-number matching,
> on-demand loader, additional models or PTB is part of the active work.

Source: **QAQ: Query-adaptive Mixed-precision Quantization for Large Language Models**, NeurIPS 2025 MLForSys workshop, five pages.

## Confirmed from the paper

### Method

1. Quantized model weights are decomposed into bit-planes with a maximum width such as 8 bits.
2. A lightweight MLP uses a hidden representation at block `j` to score candidate precisions for that block.
3. A softmax with temperature `alpha` turns scores into precision probabilities.
4. Router training uses a full-precision teacher and a quantized student with an unspecified knowledge-distillation loss.
5. An on-demand mode stores weight data in CPU memory and transfers selected data to GPU memory.
6. Inference is described as block-wise, although the text alternates between transformer layers and the attention/feed-forward sub-blocks shown in Figure 1.

### Reported evaluation

- Models: Qwen3-4B, Qwen3-8B, LLaMA-3.1-8B.
- Accuracy tasks: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande.
- Perplexity data: WikiText-2 and Penn Treebank.
- System measures: WikiText-2 latency and GPU memory.
- Baselines: FP16, static 8-bit, and static 4-bit.

## Missing details that block exact reproduction

| Area | Missing information | Why it matters |
| --- | --- | --- |
| Code | Language, libraries, versions, source, and patches | Changes model kernels, quantizers, memory accounting, and output metrics |
| Models | Exact repository IDs, base versus instruction-tuned variants, and weight revisions | Different checkpoints can change every target number |
| Weight format | Signed representation, scale, zero point, clipping, grouping, packing order | Equation (1) cannot reconstruct negative real weights without these rules |
| Quantization | Static 4/8-bit method, calibration data, group size, and excluded modules | Accuracy and memory depend strongly on these choices |
| Precision choices | The actual low, mid, and high bit widths | Figure 1 names three levels but gives no values |
| Granularity | Whole layer, attention/FFN sub-block, or linear module | Determines router outputs, transfer size, and memory use |
| Router | Input pooling, MLP size, shared versus per-block parameters, and decision timing | Needed to implement equation (2) |
| Training | Distillation formula, precision cost, data, optimizer, schedule, epochs, temperature, and seeds | Distillation alone gives no stated reason to avoid always choosing 8 bits |
| Loading | Cache size, eviction rule, prefetching, pinned memory, stream use, and when weights are offloaded | Determines the memory and latency claim |
| Evaluation | Harness, task versions, metric variants, number of examples, shots, sequence length, and batch size | The paper gives task names but not reproducible commands |
| Hardware | GPU, CPU, interconnect, RAM, software driver, and power state | Latency and memory numbers are hardware-specific |
| Measurement | Warm-up, repeats, synchronization, and definition of peak memory | A single un-synchronized timing is not comparable |

## Equation and notation issues

These are observations about the text, not accusations about the unseen implementation.

1. Equation (1) represents a real-valued matrix as a sum of non-negative binary planes. It omits the quantization scale, zero point, and sign representation required for ordinary transformer weights.
2. Equation (2) defines one score `s_j(x)`, while equation (3) uses a bit-specific score `s_j^b(x)`. A router would normally need one logit per candidate precision, but that output shape is not stated.
3. `W_j^(b)` first denotes the `b`-th binary plane, then equation (4) describes it as the weight reconstructed from the top `b` planes. Those are different objects.
4. The expected weighted sum in equation (4) is suitable for differentiable training, but the hard choice used for inference and its tie-breaking rule are not stated.
5. No term prices lower precision. If the loss is only teacher-student agreement, selecting the highest precision everywhere is a plausible minimum. A precision penalty is therefore a proposed reconstruction choice, not a reported method.

## Executable Table 1 observations

Run `python scripts/audit_table.py` to reproduce these checks.

1. At the two-decimal precision printed in the paper, QAQ with on-demand loading on and off has identical quality in every cell. Those cells also equal static 8-bit except Qwen3-4B WikiText-2, where QAQ is 14.85 and static 8-bit is 14.83.
2. Static 4-bit and static 8-bit report exactly the same GPU memory for each model. QAQ with on-demand loading disabled also has that same value.
3. On-demand memory savings relative to static 8-bit are 5.58% for LLaMA-3.1-8B, 5.08% for Qwen3-4B, and 5.43% for Qwen3-8B.
4. On-demand latency overhead relative to static 8-bit is 41.69%, 41.58%, and 41.86%, averaging 41.71%. This supports the paper's 41.7% summary when that baseline is used.
5. QAQ without on-demand loading is 4.47%, 2.16%, and 4.50% faster than static 4-bit, averaging 3.71%. The stated 4.5% is true for two model rows, not for all three or their simple average.

### HellaSwag check

The LLaMA FP16 HellaSwag value is 78.90, while all three quantized methods report 59.99 or 59.29. The other LLaMA metrics change little between FP16 and 8-bit.

The official HellaSwag task in `lm-evaluation-harness` emits both `acc` and `acc_norm`. Those can differ substantially because one uses total answer log-probability and the other normalizes by answer length. The table may therefore mix metric variants, but the paper does not say and the result has not been rerun. The first baseline experiment must log both values and must not select whichever one happens to match after the fact.

Primary source: [HellaSwag task configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml).

### Penn Treebank check

The paper does not name its evaluation software. The EleutherAI harness has a built-in WikiText task, while a 2024 issue requested adding Penn Treebank perplexity support. That suggests a custom PTB evaluation may have been needed, but it does not establish what the authors used. Tokenization, document joining, stride, and context length must be obtained or reconstructed before comparing PTB values.

Primary sources: [WikiText task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml) and [PTB feature request](https://github.com/EleutherAI/lm-evaluation-harness/issues/1884).

## Proposed first reconstruction

These choices come from this replication plan, not the paper:

- Start with `Qwen/Qwen3-4B-Base` because the benchmark-style results are more consistent with a base model and it is the least expensive target.
- Use Python and PyTorch for correctness work.
- Record all metrics returned by each task before naming the comparison metric.
- Try the model and harness revisions in `configs/baseline_candidate.yaml` because they were visible before the paper's August 2025 submission. Replace them if the authors provide exact revisions.
- Compare several static quantizers before choosing one. Do not let one method explain accuracy while another explains memory.
- Prototype signed bit-planes with sign-magnitude storage and candidate precisions 4, 6, and 8. Compare against affine unsigned and two's-complement alternatives before treating it as the implementation.
- Train on non-evaluation text with a teacher-student loss plus an explicit expected-bit cost. Sweep the cost and disclose it.
- Implement synchronous CPU-to-GPU loading first because the paper attributes its overhead to synchronous transfers. Add asynchronous transfer only as a separately labeled improvement.

## Claims that would count as a functional reproduction

1. Static 8-bit accuracy and perplexity are within declared tolerances of a clearly identified paper metric.
2. The 8-bit plane reconstruction is numerically identical to the chosen static 8-bit weights.
3. The router uses more than one precision across the evaluation queries and its distribution is reported.
4. The adaptive model stays close to static 8-bit quality while its mean selected bits or transferred bytes are lower.
5. On-demand loading lowers measured peak GPU memory under identical input settings and increases latency in a measured, repeatable way.

Matching only the printed accuracy cells is not enough, because a router that always chooses 8 bits would do that trivially.

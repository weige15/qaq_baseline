# QAQ replication starter

This project turns the five-page QAQ paper into a staged, evidence-based replication effort. It does not claim an exact reproduction: the paper does not release code and omits several choices that can materially change the result.

The recommended first target is **Qwen3-4B-Base only**. That is a proposed scope choice because it is the smallest model in the paper. The 8B models should wait until the 4B work shows both a matching baseline and real use of more than one precision.

## What the paper tells us

- The model is a frozen transformer language model.
- Weights are represented as binary slices, called bit-planes. A bit-plane stores one binary digit from every quantized weight.
- A small trainable MLP router selects a precision for each block from query-dependent hidden features.
- Selected weight data can be transferred from CPU memory to GPU memory when needed.
- The reported models are Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B.
- The reported comparisons are FP16, static 8-bit, static 4-bit, QAQ without on-demand loading, and QAQ with on-demand loading.

## What the paper does not tell us

The programming language is not reported. Neither are the signed-weight format, scale and zero-point rules, group size, exact quantizer, candidate bit widths, router dimensions, training data, router loss, optimizer, temperature, random seeds, model revisions, evaluation commands, metric variants, hardware, cache policy, or timing procedure.

Those are not small omissions. Until they are resolved by author information or experiments, an exact result is not identifiable from the paper alone.

## Proposed implementation stack

Use Python, PyTorch, Hugging Face Transformers, and the EleutherAI evaluation harness for the first functional version. This is our choice because the three model families and the benchmark tasks are readily accessible there; it is not a claim about the authors' code. Add custom CUDA or C++ only after the Python version proves the scientific claim, because performance work before that would make failures harder to interpret.

The included toy implementation uses a **sign-magnitude** bit-plane candidate. It stores one sign plane and seven magnitude planes. This is explicitly a candidate: equation (1) in the paper cannot represent negative real weights as written because it provides neither a sign rule nor a scale.

## Run the checks that work without a GPU

From this directory:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/audit_table.py
```

The first command verifies exact 8-bit integer reconstruction and nested lower-precision reconstruction for the candidate bit-plane format. The second checks arithmetic and repeated values in the paper's Table 1.

## Use with pi-goal

Install the public extension in a Pi environment:

```bash
pi install npm:pi-goal
```

Then paste the recommended command from [GOAL.md](GOAL.md). The goal is written as a completion contract: it defines what evidence is needed, what must not be changed, how to choose the next experiment, and when to stop as blocked.

This archive does not vendor pi-goal. The current workspace could inspect its public documentation but could not install it because direct package network access was unavailable.

## Decision stages

| Stage | Work | Continue when | Pause or revise when | Stop when |
| --- | --- | --- | --- | --- |
| 0 | Audit the paper and reported table | Every claim is mapped to reported evidence or a named unknown | A claim is ambiguous | Never silently fill an ambiguity |
| 1 | Reproduce Qwen3-4B FP16 evaluation | Model revision, task data, metric names, shots, batch settings, and environment are recorded; most values are close to the paper | Scores differ beyond the declared tolerance | The tested version matrix is exhausted without a defensible match |
| 2 | Identify static 8-bit and 4-bit behavior | One quantizer family explains the accuracy, memory, and latency pattern better than alternatives | Different methods match different columns | No method gives a coherent match; report non-identifiability |
| 3 | Build bit-planes and router | 8-bit reconstruction is exact, lower precisions are tested, the router selects at least two precisions, and quality is measured against static 8-bit | The router collapses to one choice or training uses evaluation examples | No safe loss or weight format preserves the static baseline |
| 4 | Add synchronous CPU-to-GPU loading | Peak memory falls and transferred bytes agree with selected planes | Memory falls only because unrelated tensors moved, or latency is unstable | No measurable memory change remains after measurement errors are fixed |
| 5 | Attempt the full paper table | Qwen3-4B gates pass and resources are sufficient | An 8B baseline does not match | Required checkpoints, access, or hardware remain unavailable |

## Suggested acceptance rules

These thresholds are proposed starting points because the paper gives two decimal places but no error bars:

- Accuracy: within 0.5 percentage points of the selected paper metric.
- Perplexity: within 2% relative difference, using an identical text preparation and windowing method.
- Memory: report allocated and reserved GPU memory separately; compare only identical batch and sequence settings.
- Latency: at least 10 warm runs and 30 measured runs; report median and spread, not one number.
- Routing: report the precision distribution per block and per query. Matching 8-bit accuracy is not evidence of adaptation if every route selects 8 bits.

Revise the thresholds before running if the evaluation output supplies a larger sampling error. Stop rather than relax thresholds after seeing a disappointing result.

## Hardware gate

The paper reports 14.23 GB peak GPU memory for Qwen3-4B FP16 and 23.12 GB for LLaMA-3.1-8B FP16. A practical first machine therefore needs more than the reported peak, not merely equal capacity. As a proposed allowance, use a 24 GB GPU for the 4B functional work and a 48 GB GPU for 8B development, plus enough CPU memory to hold packed weight data and transfer buffers. Measure the actual requirement before reserving a larger run.

## Files

- [PAPER_AUDIT.md](PAPER_AUDIT.md): known facts, missing details, and table checks.
- [GOAL.md](GOAL.md): recommended narrow goal and a broader alternative.
- [DECISIONS.md](DECISIONS.md): choices, their status, and what would change them.
- [AUTHOR_QUESTIONS.md](AUTHOR_QUESTIONS.md): the smallest set of questions that would remove the largest uncertainty.
- [configs/baseline_candidate.yaml](configs/baseline_candidate.yaml): a proposed first-run configuration with assumptions labeled.
- [FIRST_GPU_RUN.md](FIRST_GPU_RUN.md): the first full-model experiment and its decision gates.
- [SHARED_SERVER.md](SHARED_SERVER.md): safe GPU selection and launch rules for the shared RTX 3090 host.
- [scripts/audit_table.py](scripts/audit_table.py): executable audit of Table 1.
- [src/qaq/bitplanes.py](src/qaq/bitplanes.py): toy signed bit-plane candidate.
- [CLAIM_REPORT_TEMPLATE.md](CLAIM_REPORT_TEMPLATE.md): final report structure for pi-goal.

## Primary sources

- [QAQ paper on OpenReview](https://openreview.net/forum?id=dpHfDasG44)
- [NeurIPS 2025 MLForSys accepted papers](https://mlforsystems.org/neurips2025/accepted_papers.html)
- [pi-goal repository](https://github.com/Michaelliv/pi-goal)
- [Qwen3-4B-Base model card](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [Qwen3-8B-Base model card](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)


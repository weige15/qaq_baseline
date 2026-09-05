# QAQ Core Reproduction Goal

## Purpose

Build the smallest defensible reproduction of QAQ’s central idea:

> A query determines the precision used by individual attention and FFN blocks.

This project is not trying to reproduce every number in the QAQ paper. The paper omits too many implementation and evaluation details for exact reproduction. The goal is to create a documented, testable baseline that can support new research.

## Paste this command into Pi

```text
/goal --tokens 80k Before changing code, read references/QAQ.pdf, PAPER_AUDIT.md, DECISIONS.md, FIRST_GPU_RUN.md, configs/baseline_candidate.yaml, and scripts/audit_table.py. In this goal, “Table 1” means the page-4 comparison of FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on for Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B. Build and verify a functional reproduction of QAQ's core query-dependent block-precision method in this repository using Qwen/Qwen3-4B. Do not pursue an exact reproduction of Table 1. Completion requires all of the following: (1) freeze and record one evaluation setup, including model revision, data revisions, metrics, software versions, commands, and hardware; (2) produce successful FP16, fixed 8-bit, and fixed 4-bit results on WikiText-2, HellaSwag, and ARC-Challenge, recording every returned metric without choosing metrics after seeing which matches the paper; (3) integrate one signed and scaled weight representation in which a single stored quantized model supports 4-bit, 6-bit, and 8-bit reconstruction, full-width reconstruction exactly matches the chosen 8-bit quantized weights, and the selected precision genuinely changes the weights used by the model; (4) allow precision to be selected separately for each attention block and FFN block; (5) train a small query-dependent router using data that is separate from the final evaluation examples; (6) demonstrate on unseen examples that different queries produce different block-precision profiles and that at least two precision levels are genuinely used; (7) compare the adaptive router with fixed 4-bit, fixed 8-bit, random routing, and a query-independent static block policy using the same average number of bits; and (8) save raw results, exact commands, configurations, tests, routing distributions, failures, and a final REPLICATION_REPORT.md that separates paper statements, our implementation choices, measured findings, and unresolved questions. Exclude CPU-to-GPU on-demand loading, Qwen3-8B, LLaMA-3.1-8B, Penn Treebank, the complete paper table, dynamic batching, asynchronous loading, and performance-kernel optimization. Existing code from other repositories may be reused only in small reviewed parts and must be tested and rerun here; never reuse old results as evidence. Use scripts/gpu_preflight.sh before every GPU job, use only one safely available GPU, and never interfere with another user's process. After the FP16 and static-quantization stage, continue only if the runs are repeatable and internally sensible; otherwise pause with the exact mismatch. After weight integration, continue only if full-width reconstruction is exact and lower precisions change model outputs; otherwise revise the representation. After router training, continue only if routing varies across queries and does not collapse to one precision; otherwise revise the features or training objective using a small predeclared set of attempts. If the adaptive method cannot improve on the static policy using the same average bits, stop with that negative result rather than hiding it or extending the search indefinitely. Pause before any unexpectedly long GPU sweep, model-license decision, or major expansion of scope. Mark the goal complete only when every required artifact and REPLICATION_REPORT.md exists.
```

## What completion means

The reproduction is complete when:

* Qwen3-4B runs successfully at fixed 4-bit and 8-bit precision.
* Attention and FFN blocks can independently use 4, 6, or 8 bits.
* The router produces different precision choices for different queries.
* The adaptive method is compared fairly with a static policy using the same average precision.
* All commands, settings, raw results, and failures are recorded.
* The final report states clearly what worked and what did not.

A successful reproduction does not require matching the paper’s printed numbers.

## Decision rules

### Continue

Continue when the current stage passes its recorded checks and the next experiment addresses a remaining question.

### Pause

Pause when:

* No suitable shared GPU is available.
* Model or dataset access requires a decision.
* A run would expand substantially beyond the agreed scope.
* Results cannot be interpreted because required settings were not recorded.

### Revise

Revise the implementation when:

* Eight-bit reconstruction is not exact relative to the chosen quantized weights.
* Different precision settings do not actually change model computation.
* The router always chooses one precision.
* A comparison gives different methods unequal evaluation conditions.

### Stop

Stop and report the evidence when:

* A required dependency or model remains unavailable.
* The baseline cannot be made stable after the documented small set of attempts.
* The adaptive method fails to improve over the equally sized static policy.
* Continuing would only mean trying arbitrary settings until a favorable result appears.

A carefully documented failure is a valid research result and a valid completion outcome.

## Explicitly deferred work

Do not implement these as part of this goal:

* CPU-to-GPU on-demand weight loading
* Dynamic batching
* Custom CUDA performance work
* Qwen3-8B or LLaMA-3.1-8B
* Exact reproduction of every QAQ result
* Penn Treebank evaluation
* Production serving integration

These become separate research decisions after the core reproduction is complete.

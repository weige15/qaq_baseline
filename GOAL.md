# pi-goal commands

## Recommended: Qwen3-4B first

Paste this as one command in Pi after installing `pi-goal`:

```text
/goal --tokens 100k Produce the strongest evidence-backed functional reproduction of the QAQ paper for Qwen3-4B-Base before attempting either 8B model. Completion requires: (1) maintain PAPER_AUDIT.md and DECISIONS.md so every material statement is labeled as reported by the paper, directly measured, inferred, or proposed; (2) reproduce the FP16, static 8-bit, and static 4-bit baselines with model and data revisions, every returned metric name, shot count, batch and sequence settings, software versions, hardware, raw result files, and repeatable commands recorded; (3) investigate the LLaMA-style HellaSwag acc versus acc_norm risk even though the first model is Qwen, and never choose a metric after seeing which one matches; (4) implement and test a signed, scaled 8-bit bit-plane representation whose full reconstruction is exactly equal to the chosen static 8-bit weights, compare plausible sign/scale formats, and label the selected format as a reconstruction unless author evidence confirms it; (5) implement a frozen-model per-block MLP router with explicit, documented teacher-student and precision-cost terms, use no evaluation examples for training, log per-query and per-block precision choices, and require use of at least two precisions; (6) only if those checks pass, implement the paper's synchronous CPU-to-GPU on-demand mode and measure allocated and reserved peak GPU memory, transferred bytes, and warmed repeated latency against identical static settings; and (7) finish with CLAIM_REPORT_TEMPLATE.md completed claim by claim, including confirmed results, approximate matches, failed attempts, and remaining uncertainty. Preserve the original checkpoints and evaluation data, do not claim the authors' programming language or omitted settings, do not tune on the final evaluation sets, and do not expand to Qwen3-8B or LLaMA-3.1-8B until the 4B FP16 baseline, static quantizer, non-collapsed router, and memory measurement gates all pass. After each experiment, update the decision log, compare the result with the declared acceptance rule, and choose the smallest next test that can distinguish the remaining plausible explanations. Pause for a user decision before any material paid compute or model-license action. If the available hardware, model access, missing author details, baseline mismatch, or router collapse leaves no defensible path, stop with commands run, raw evidence, hypotheses ruled out, exact blocker, and the next input that would unlock progress rather than marking the goal complete.
```

Why this is recommended:

- Outcome: a functional Qwen3-4B reproduction, not a vague request to "reproduce the paper."
- Evidence: raw evaluation files, unit tests, routing distributions, transfer counts, memory, latency, and a claim report.
- Constraints: no evaluation-data training, no silent metric selection, no claim about hidden implementation details.
- Boundaries: one model first; larger models require explicit gates.
- Iteration: each run must distinguish remaining explanations.
- Blocked stop: missing access, compute, author detail, or coherent baseline ends with an auditable blocker report.

## Broader alternative: all reported models

Use this only after resources are confirmed or after the recommended goal completes:

```text
/goal --tokens 200k Produce the strongest evidence-backed reproduction of QAQ across Qwen3-4B-Base, Qwen3-8B-Base, and Llama-3.1-8B, verified by repeatable FP16/static-8/static-4/adaptive evaluation artifacts for HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, WikiText-2, and Penn Treebank; unit-tested signed and scaled bit-plane reconstruction; reported router precision distributions; and warmed repeated GPU memory, transfer, and latency measurements with and without synchronous on-demand loading. Treat the paper's omitted model revisions, metric variants, quantizer, precision choices, router loss, training data, cache policy, hardware, and measurement procedure as unresolved until author evidence or controlled experiments distinguish them. Preserve checkpoints and final evaluation data, train no router on evaluation examples, record every attempted configuration and raw result, and label exact matches, approximate functional reconstructions, conflicting evidence, and blocked claims separately. Work model by model, starting with Qwen3-4B, and expand only when the current model's FP16 baseline, coherent static quantizer, exact 8-bit reconstruction, non-collapsed multi-precision routing, and repeatable memory measurement pass their predeclared acceptance rules. Between iterations, update the decision log and run the smallest experiment that can eliminate a plausible implementation choice. Pause before material paid compute or license actions. If exact reproduction is not identifiable or resources are insufficient, stop with the complete evidence map, attempts, raw outputs, exact blocker, and author information or hardware needed; do not report completion merely because adaptive outputs equal static 8-bit outputs.
```


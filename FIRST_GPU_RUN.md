# First GPU run: baseline identification only

> Historical proposal. The active core goal supersedes this paper-matching/Base
> plan with [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md); do not execute the
> broader task list below for the active goal.

Do not train a router in the first GPU session. The first question is whether an identified public checkpoint and evaluation setup can reproduce the paper's unquantized Qwen3-4B scores closely enough to support later comparisons.

## Known before the run

- The official Qwen3-4B-Base card identifies a 4.0B-parameter, 36-layer causal language model and requires Transformers 4.51.0 or newer.
- The paper does not identify its checkpoint revision or evaluation code.
- This project's current workspace has no GPU, PyTorch, Transformers, datasets, or evaluation harness, so the command below has not been executed here.

## Proposed candidate

- Model: `Qwen/Qwen3-4B-Base`
- Revision: `906bfd4`
- Evaluation code: EleutherAI `lm-evaluation-harness` revision `d021bf8` (v0.4.9.1)
- First full task: HellaSwag, zero-shot
- Required output: both `acc` and `acc_norm`, raw samples when supported, full environment, and fixed batch size

For direct Transformers smoke code, pass the precision with the current `dtype` keyword (for example, `dtype=torch.float16`). Do not use the deprecated spelling in new commands.

These are reconstruction choices based on publication timing and score style. They are not reported by the paper.

## Before spending compute

1. Confirm the GPU has comfortably more memory than the model's observed FP16 load.
2. Confirm the model terms and access are acceptable.
3. Install a PyTorch build supported by that GPU, then install Transformers 4.51.0 or newer and the selected harness revision.
4. Record `nvidia-smi`, Python version, package versions, model revision, and dataset revision in the experiment log.
5. Choose a fixed batch size with a small smoke run. A smoke run checks only that the setup works; its limited score must not be compared with Table 1.

## Candidate command shape

After the harness is installed and a fixed batch size has been chosen:

```bash
bash scripts/gpu_preflight.sh --run python -m lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-4B-Base,revision=906bfd4,dtype=float16 \
  --tasks hellaswag \
  --num_fewshot 0 \
  --batch_size FIXED_BATCH_SIZE \
  --log_samples \
  --output_path results/baseline/qwen3-4b-fp16-hellaswag
```

Replace `FIXED_BATCH_SIZE` before execution. Do not silently add a chat template or enable thinking for a base-model multiple-choice run.

## Decision after HellaSwag

- **Continue** to PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and WikiText-2 if the command completes, both HellaSwag metrics are captured, and one metric was predeclared for comparison.
- **Pause** if access, memory, or dataset download fails. Record the exact error and the smallest needed resource or permission.
- **Revise** the model or harness candidate if neither metric is close and an independent reason supports a different revision or model variant.
- **Stop the baseline search** after the predeclared version matrix is exhausted. Report that the paper's baseline is not identifiable rather than continuing to try arbitrary versions.

## Decision after the full FP16 row

- **Continue** to static quantizer identification only if most accuracy values are within 0.5 points and perplexity is within 2% under a fully recorded text preparation method.
- **Revise** metric or data preparation only when the raw outputs expose a specific mismatch; do not select metrics separately per model to improve agreement.
- **Pause** PTB comparison until its tokenizer, document joining, context window, and stride are fixed, because the paper does not provide them.
- **Stop before QAQ training** if no coherent FP16 setup matches. A dynamic method cannot be evaluated fairly against an unidentified baseline.

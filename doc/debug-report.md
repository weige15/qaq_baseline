# Debug Report

## Symptom
Tiny Qwen3 integrated checkpoint reload had identical persisted weights but non-identical logits.

## Reproduction Command
Working directory: `/nfs/home/s314511048/qaq_baseline`
Shell: bash. Runtime: Python3.12.3, torch2.5.1+cu121, Transformers5.16.1.
Environment: `~/.venv`; `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src`.

```bash
source ~/.venv/bin/activate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
```

## Expected Behavior
Exact reload logits at4/6/8 with identical quantized and excluded weights.

## Actual Behavior
State-dict equality passed; forward equality failed. CPU-only; no GPU run occurred.

## Error Log
`results/core-v1/meta/tests-integration-cpu.txt`:
```text
tests/test_model.py:67: AssertionError: False is not true
```
Targeted reproduction: `results/core-v1/meta/reload-diagnosis.txt`:
```text
attention sdpa sdpa
buffer mismatch model.rotary_emb.inv_freq torch.float16 torch.float32 0.00015866756439208984
buffer mismatch model.rotary_emb.original_inv_freq torch.float16 torch.float32 0.00015866756439208984
before 4 0.00048828125
before 6 0.00054931640625
before 8 0.00048828125
after matched rotary 4 True
after matched rotary 6 True
after matched rotary 8 True
```

## Failure Layer Classification
* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: yes (incomplete runtime-buffer representation)
* Code logic problem: yes
* Configuration problem: yes (test constructs `.half()`, loader creates dtype-aware model)
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: serializer missed nonpersistent rotary state; exposed by fixture precision.

## Hypotheses

### Hypothesis 1: quantized weights or tied head differ
Plausible because assign-style load and ties are involved. Against: every state-dict tensor tested exactly equal. Rejected by direct buffer inspection.

### Hypothesis 2: nonpersistent rotary buffers differ
The test builds FP32 then `.half()`; reload constructs FP16 weights but preserves FP32 rotary frequencies. State dict omits both rotary frequency buffers. Confirmed by dtype/value inspection and replacing only `inv_freq`: all3 precision logits then exactly equal. Both use SDPA, so no attention implementation mismatch.

## Most Likely Root Cause
State-dict-only serialization regenerated rather than preserved nonpersistent rotary frequencies. The real frozen pretrained model normally keeps those buffers FP32, but a model checkpoint promising exact reconstruction should preserve its actual rotary state, not depend on initialization history.

## Minimal Fix
Store the two small rotary buffers in the SAME quantized checkpoint and restore them explicitly. Do not store transient reconstructed weights, relax equality, or weaken the fixture. This is within the goal's explicit instruction to revise failing integration.

## Verification
```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
```
Verified after the minimal serializer fix: all15 tests pass in `results/core-v1/meta/tests-integration-fixed.txt`, including exact reload outputs at4/6/8. The separately logged full Qwen gate subsequently passed in `results/core-v1/integration/gate.json`:651 state tensor hashes and all six profiles' complete logits reload exactly. See `INTEGRATION_RESULTS.md`. Neither test nor integration success establishes router quality.

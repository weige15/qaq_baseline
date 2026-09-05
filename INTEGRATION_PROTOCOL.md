# Weight integration gate (after passing baseline stage)

Prerequisite: `results/core-v1/baseline-gate.json` passes and
[BASELINE_RESULTS.md](BASELINE_RESULTS.md) inspected. No router training yet.

Implement `src/qaq/model.py`: one signed int8 backing tensor plus group scales
per projection; excluded model weights retained. Independent72-block profile
order is attention0, FFN0, attention1, FFN1, … . Only one transient FP16
reconstruction per projection is cached; precision changes invalidate that cache.
No multiple stored quantized models and no all-precision float cache. This is
correctness-oriented FP16 computation, not a low-bit memory/performance claim.

Before new GPU evidence, CPU tests require exact4/6/8 checkpoint reload, independent
attention/FFN changes, every-reference8 equality, and no active caches in the
persisted state. A failed CPU reload test revealed nonpersistent rotary buffers:
see `doc/debug-report.md`. The two small buffers are now included in the same
checkpoint instead of regenerated. All15 CPU tests pass in
`results/core-v1/meta/tests-integration-fixed.txt`; initial failure remains saved.

## Exact real-model checks, fixed before running

`python scripts/check_integration.py` will:

1. Refuse missing/failed baseline gate or changed frozen inputs.
2. Load frozen original checkpoint; prepare packed q/scales without mutating it.
3. Apply the completed baseline's fixed8 path independently to the dense model.
   Compare every one of252 projection tensors against packed8 reconstruction.
4. Compare complete reference8 and integrated8 logits for three specified ordinary
   test prompts (listed in the script, not taken from train/dev/final datasets).
5. Test fixed4, fixed6, attention0-only4, FFN0-only4 and repeating4/6/8 profiles.
   Every probe must be finite and differ from full8. Check all36 attention block
   sizes match each other, likewise FFN, supporting later weighted-budget fairness.
6. Save one complete quantized checkpoint. Hash every persisted tensor, discard
   the model, reload without the original pretrained checkpoint, compare every
   tensor and all six profiles' complete logits exactly.
7. Save checkpoint file hash, raw probe logits, all252 equality checks,72 block
   parameter counts, commands, frozen input hashes, source snapshot, timing and
   own-process peak allocation. A failed gate writes `failure.txt` and forbids
   training until repaired. Tests are not proxies for this model gate.

This is a single bounded integration job,30min cap, expected a few minutes for
4.5GB serialization. No sweep, no other model, no asynchronous transfers.

```bash
source ~/.venv/bin/activate
bash scripts/gpu_preflight.sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
  bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
  python scripts/check_integration.py
```

Raw output: `results/core-v1/integration/`; shell/preflight/exit status:
`results/core-v1/logs/integration.log`. A passing gate authorizes predeclaring the
bounded router training attempts, **not** a claim of complete reproduction.

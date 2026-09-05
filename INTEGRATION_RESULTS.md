# Stored-model integration — passed

Preregistration: `71d1102`, [INTEGRATION_PROTOCOL.md](INTEGRATION_PROTOCOL.md).
Fresh command through preflight:
`PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m python scripts/check_integration.py`.

Raw evidence: `results/core-v1/integration/gate.json`, `raw_probe_logits.pt`,
`probes.json`, `state_hashes.json`, `command.json`, `source/`, and
`results/core-v1/logs/integration.log` (exit0; physical GPU6, RTX3090).

- **252/252 actual Qwen projection tensors** exactly match chosen dense fixed8
  reconstruction. Full integrated8 logits equal dense fixed8 on all three probes.
- **72 independently controlled blocks**:36 attention,36 FFN. Attention blocks
  each contain26,214,400 selected weights; FFNs each74,711,040. Thus later equal
  per-type precision quotas can produce exact parameter-weighted bit budgets.
- Full4, full6, attention0-only4, FFN0-only4 and alternating4/6/8 all change
  actual logits on every probe, with finite outputs. Single-attention changes'
  max absolute logit deltas are .22265625/.953125/.1796875; single-FFN changes'
  .3125/.501953125/.265625. Full4/6 changes are larger; all values retained raw.
- **One complete quantized checkpoint** (4,525,416,138 bytes), including excluded
  FP16 weights, q/scales and actual rotary buffers. No reconstructed variant is
  serialized. SHA256:
  `b63aeeed85e11cea5d2790b1aae068920626e5d93117548358f76ae3b2171b85`.
- After discarding the model and reconstructing solely from this checkpoint,
  **651 persisted tensors** match their original hashes. Every complete probe
  logit tensor is exactly equal under all six tested profiles, including4/6/8.
- Job finished in273.38s with own-process peak allocated12,193,000,960 bytes.
  These are operational records, not optimized memory/latency claims.

CPU coverage:15 passing tests. Initial nonpersistent-rotary reload failure and
its causal diagnosis/fix remain in `doc/debug-report.md` and `results/core-v1/meta`.
No equality tolerance was relaxed. Full model succeeded after the CPU repair.

## What this does NOT establish

The representation stores8-bit master weights plus scales and excluded16-bit
weights. It uses one transient FP16 reconstruction per projection for ordinary
FP16 matrix multiplication; this is **not** a4/6-bit GPU storage or kernel-speed
claim. Manual profiles are not learned query variation. No router has been
trained or compared with static/random policies. The active goal is NOT complete.

Next: preregister bounded train/dev data, teacher-local-error targets, small MLP
attempts, causal fixed8 context-only probe, and exact matched-budget controls.
No training or final adaptive search has yet happened.

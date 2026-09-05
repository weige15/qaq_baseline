# Fresh frozen baseline stage — passed

New local runs only, protocol `qaq-core-v1`, preregistration commit `8fc7a05`.
Machine-readable all-metric output and checks:
`results/core-v1/baseline-gate.json`; command:
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_baselines.py`.

| Mode | WT2 token PPL | HellaSwag acc | HellaSwag acc_norm | ARC-C acc | ARC-C acc_norm |
|---|---:|---:|---:|---:|---:|
| FP16 | 17.167127 | 0.500000 | 0.695313 | 0.500000 | 0.562500 |
| Nested8 | 17.224705 | 0.496094 | 0.695313 | 0.503906 | 0.558594 |
| Nested midpoint4 | 18.886606 | 0.488281 | 0.652344 | 0.406250 | 0.460938 |

Each mode has **two fresh processes**, each scoring the same64 WT2 windows
(24,576 continuation tokens),256 HellaSwag and256 ARC-C examples. Raw per-token
losses, choice scores and predictions are **identical** across each pair;
max per-choice logprob delta and mean NLL delta both0. The8/4 PPL ratios vsFP16
are1.003354 and1.100161. All predeclared sanity gates passed. Every returned
metric, including SEs and NLL totals not shown in this compact table, remains in
each `results.json` and the aggregate gate report. No paper values were used to
select metrics or revise thresholds.

Raw directories: `results/core-v1/baseline-{fp16,fixed8,fixed4}-r{1,2}/`.
Exact shell commands/preflight/exit codes: `results/core-v1/logs/baseline-*.log`.
Per run: `command.json`, `hardware.json`, `source/`, `samples.jsonl`,
`results.json`, git HEAD/diff, and quantized-module lists for4/8.
Evaluations took roughly130–135 seconds for the first four runs; timings are
operational observations, **not benchmarked performance claims**.

## Failure preserved

`logs/baseline-fixed4-r1.log`: preflight selected a temporarily idle GPU2, then
its immediate recheck changed availability and refused launch, exit4. The model
command was NOT run. `logs/recheck-after-fixed4-refusal.txt` records a later clean
check; both4-bit processes then completed with separate `*-launch2.log` logs.
No process was killed or reset; the other users' devices were excluded.

## Interpretation and limits

These are bounded functional baseline results, NOT full-split benchmark scores,
not the paper's perplexity definition, and not an exact Table1 reproduction.
The 4-bit quantizer is predictably more damaging, especially on ARC-C, while8bit
stays near FP16. Baseline pass authorizes the **weight integration gate only**.
It does not prove a persistent quantized model, routing, matched-budget benefit,
or whole-goal completion. Those require independent artifacts and checks.

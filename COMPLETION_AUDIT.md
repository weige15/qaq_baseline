# Prompt-to-artifact completion audit

**Goal status: NOT COMPLETE.** Audited after the fresh baseline and full-model
integration stages. A green stage verifier is not a whole-goal verifier.

Concrete success criteria: a functional query-dependent **Qwen/Qwen3-4B**,
frozen evaluation, fresh FP16/4/8 results on WT2/HellaSwag/ARC-C, one signed/scaled
4/6/8 checkpoint, independent attention/FFN routing, disjoint router training,
unseen learned routing variation, fair equal-bit comparisons and an evidence-
separated `REPLICATION_REPORT.md` with all raw commands/config/tests/failures.

| Explicit requirement / gate | Actual evidence inspected | Status / uncovered work |
|---|---|---|
| Read `references/QAQ.pdf` before code | all5 pages read, including page4; SHA256 in protocol and `results/core-v1/meta/paper-extracted.txt` | Done |
| Read `PAPER_AUDIT.md`, `DECISIONS.md`, `FIRST_GPU_RUN.md`, `configs/baseline_candidate.yaml`, `scripts/audit_table.py` before code | inspected before changes; historical criteria explicitly superseded | Done |
| Table1 means page4 five-method/three-model comparison; no exact reproduction | `EVALUATION_PROTOCOL.md`, `results/core-v1/meta/table-audit.txt` | Done; arithmetic audit not experimental evidence |
| (1) Freeze model/revision/tokenizer/data/metrics/software/commands/hardware | `configs/core_protocol.json`; `results/core-v1/frozen/{protocol,model_manifest,data_manifest,environment,freeze_hashes}.json`; tokenized examples, nvidia-smi/cpu/ram/pip-freeze; per-job commands/hardware | Done for baseline and integration; router commands/settings still to freeze |
| (2) New FP16/fixed8/fixed4 on WT2/HellaSwag/ARC-C; every returned metric | six `results/core-v1/baseline-{fp16,fixed8,fixed4}-r{1,2}/` directories,576 examples each; all metrics/raw token losses; `BASELINE_RESULTS.md` | Done on explicitly bounded frozen subsets, not full splits |
| Baseline repeatability and internal sanity before proceeding | `scripts/check_baselines.py`, `results/core-v1/baseline-gate.json`; all six raw sets reaggregated against frozen IDs/counts/token masks | Passed: each pair's raw samples identical; PPL ratios1.003354/1.100161; all declared checks pass |
| (3) Single signed/scaled stored model supporting4/6/8 | `src/qaq/model.py`, `src/qaq/quantization.py`; `results/core-v1/integration/quantized_model.pt` (4,525,416,138 bytes) | Done; no variants persisted; signed int8 +FP32 scales with top-bit midpoints |
| Full8 exactly chosen8; lower4/6 genuinely change model output | integration gate252/252 actual projection equalities, full8 exact logits; raw lower-bit finite logit changes on3 probes; exact checkpoint hash | Passed real-model gate, not toy evidence only |
| (4) Independent attention and FFN selection |72-block mapping; attention0-only4 and FFN0-only4 actual-model outputs change; `tests/test_model.py` tests isolation | Done;36 independent attention and36 FFN blocks |
| Weight integration gate: revise if failing | initial rotary-buffer failure retained; diagnosis/fix;15 CPU tests;651 persisted tensor hashes and six-profile exact model reload | Passed after minimal repair without relaxed tolerance |
| (5) Train small query router on separate data | **No fit/cache/router checkpoint exists** | **Missing**. Freeze train/dev document IDs/hashes, decontamination and bounded attempts first |
| (6) Unseen query profiles vary and >=2 actual levels | Only manual test profiles exist | **Missing**. Manual profiles or quota-enforced diversity are not learned adaptation |
| Router gate: bounded predeclared noncollapse repairs | max3 attempts and dev-only selection rule in frozen evaluation protocol; exact attempts not yet declared | **Missing**; no training may precede attempt declaration |
| (7) Adaptive vs fixed4/fixed8/random/query-independent static at same average bits | baseline anchors exist; integration verified equal block sizes within type | **Missing** adaptive/random/static outputs and actual per-query weighted-budget equality |
| Negative-result stop vs matched static | WT2 paired95% CI and MC guardrail frozen before adaptive scores | Rule recorded; comparison **missing**. No further final-score-driven search allowed |
| Causal routing (context only; same profile across options; no WT2 suffix leak) | scorer context-only API and perturbation unit test; WT2 prefix mask/causal shift test | API verified; **actual trained-router perturbation test missing** |
| (8) Raw results/commands/configs/tests/routing/failures | `results/core-v1/` freeze, six baseline runs,3 smoke runs, integration, per-job source snapshots/logs; failure ledger below | Partial; router training/profiles/distributions/comparisons **missing** |
| Final `REPLICATION_REPORT.md`, paper/choices/findings/unknowns separated | **File does not exist** | **Missing**; stage reports cannot substitute |
| Exclusions: no on-demand, other models, PTB/full table, dynamic batching, async/kernel work | current source, exact logged model commands; historical broader proposals labeled superseded | Preserved so far; final source/job audit still required |
| Small reviewed external code only; tested/rerun locally; no old result evidence | tiny attributed harness prompt rules; local tests; new `core-v1` run namespace; old logs excluded | Done so far; no external implementation or results transplanted |
| `scripts/gpu_preflight.sh` before every GPU job; one safe GPU; no interference | ten successful job logs (3 smoke+6 baseline+1 integration); one refused launch; mock safety tests; each job remaps exactly one device | All current jobs guarded and serial; no process kill/reset |
| Pause before unexpectedly long sweep/license decision/expansion | smoke runtime projection (roughly2–3min/full job),30min caps, public requested Apache-2.0 checkpoint card, bounded integration job273s | No expanded scope or license acceptance; future router runtime must be bounded |
| Mark complete only after all required evidence inspected | this audit explicitly lists missing5/6/7/8 and final report | **Do not mark complete** |

## Coverage and failure checks

- Independent checkpoint audit in `results/core-v1/meta/checkpoint-audit.txt`
  rehashes all12 frozen original model/tokenizer files, all3 dataset source files
  and the4.525GB quantized checkpoint. It reruns the baseline gate and confirms
  that deleting a returned metric or changing a raw prediction is rejected.
- Baseline verifier checks frozen hashes, IDs, counts, prefix target lengths,
  raw-to-aggregate metrics, all required metric keys, finite values, fixed-module
  counts, predictions and repeat tolerances. It explicitly does **not** verify
  router/integration/comparison/final-report requirements.
- Integration gate checks real weights, raw logits, persisted tensor hashes and
  independent profiles. It explicitly does **not** verify training, learned
  query variation or matched-budget benefit.
- Test suite contains6 historical NumPy tests,6 new quantizer/evaluator tests,
  2 GPU safety tests and1 actual tiny-Qwen integration test.15 passing tests do
  not cover the missing router deliverables.
- Failures retained: initial HellaSwag expected-whitespace typo (test fixed,
  preprocessing unchanged); CPU rotary-buffer checkpoint mismatch (diagnosed,
  serializer fixed); one GPU2 availability race (guard refused launch, exit4,
  later clean recheck). No failed model scores hidden or reused as success.

## Next concrete action

Freeze router train/dev data and at most3 exact feature/loss attempts **before
fitting**. Collect bounded teacher local-error labels, train small MLP, test
unseen dev variation/causality/repeatability, then compare once on frozen final
examples against strong trained-static/random controls at exactly equal actual
weighted bits. Account for the fixed8 context feature-pass overhead explicitly;
consider giving all matched controls the same probe so total computation-bit
accounting is equal as well. Stop with a negative result if the locked primary
comparison fails. Finish report and audit only after the remaining evidence exists.

# Prompt-to-artifact completion audit

**Work-product status: VERIFIED COMPLETE**, after inspecting the actual local
artifacts and rerunning CPU evidence checks on2026-09-05 UTC. This is completion
of the bounded functional goal, **not** confirmation of QAQ's paper table or a
general adaptive-quality advantage. No further GPU experiment is required.

## Concrete deliverables

1. One frozen Qwen/Qwen3-4B setup with immutable data/model revisions, all metrics,
   exact commands, software and hardware records.
2. Fresh, repeatable FP16/fixed8/fixed4 on WT2, HellaSwag and ARC-Challenge.
3. One signed/scaled checkpoint with exact8 and real4/6 reconstructions, separately
   controlled attention and FFN blocks.
4. A small actually trained router on separate data; deterministic, noncollapsed
   unseen query-dependent profiles and genuine changed model weights/outputs.
5. Matched-bit adaptive/random/query-independent static comparisons, with fixed
   endpoints and every raw result retained; no final-driven search.
6. All configurations, tests, routing distributions, failures, reproducible
   commands, and an evidence-separated `REPLICATION_REPORT.md`.
7. Preserve explicit scope/GPU safety gates and stop rules throughout.

## Prompt-to-artifact checklist

Paths below are relative to the repo. `R` means `results/core-v1/` in this table.
**Passing a stage verifier was not accepted as proof of the other rows.**

| Explicit requirement, command or gate | Actual evidence inspected | Finding |
|---|---|---|
| Read `references/QAQ.pdf` before code | All5 freshly extracted pages read; SHA256 `f15f1d6c…c37c8e`; archived `R/meta/paper-extracted.txt`; page4 definition checked | Done |
| Read `PAPER_AUDIT.md`, `DECISIONS.md`, `FIRST_GPU_RUN.md`, `configs/baseline_candidate.yaml`, `scripts/audit_table.py` first | All five read before edits in this continuation; historical Base/paper-matching/loading proposals explicitly superseded | Done |
| Table1 means page4 five methods/three models; do not exactly reproduce it | `EVALUATION_PROTOCOL.md`, final report§1; `python scripts/audit_table.py` output `R/meta/table-audit-final.txt` | Done; arithmetic is not model evidence |
| (1) Freeze exact Qwen target/revision/tokenizer | `configs/core_protocol.json`, `R/frozen/model_manifest.json`; final audit rehashed all12 original files | Done: literal `Qwen/Qwen3-4B`, `1cfa9a…df60c`, not Base |
| (1) Data revisions/indices/tokenization/metrics | Frozen protocol, all3 source hashes,576 tokenized examples and source manifests; data audit retokenized all224 router windows | Done; declared bounded subsets, not full-split claims |
| (1) Software versions/commands/hardware | Full frozen environment/pip list/CPU/RAM/driver records; every job `command.json`/source snapshot and guarded shell log; runtime version assertions | Done; same RTX3090 model, one device/job; static repeat2 on7, most new jobs on6 |
| (2) Fresh FP16/fixed8/fixed4 on all three tasks | Six `R/baseline-{fp16,fixed8,fixed4}-r{1,2}/` runs,576 samples each; `BASELINE_RESULTS.md` | Done; only fresh core-v1 results used |
| (2) Every returned metric; no posthoc metric matching | Raw token losses and full result dictionaries; all five WT2/five MC metric keys checked; primary meanNLL/acc_norm fixed before scores | Done; missing metric and altered prediction mutation tests reject artifacts |
| Baseline gate: repeatable and internally sensible before continuation | `scripts/check_baselines.py` rerun directly and through comparison audit; all sample pairs identical, exact metrics, declared sanity thresholds pass | Passed; no relaxed criteria |
| (3) One stored signed/scaled4/6/8 model | `R/integration/quantized_model.pt`, `src/qaq/{quantization,model}.py`; final file/state audit | Done: one4.525GB int8/scaled model, no saved precision variants |
| Full-width reconstruction exact to chosen8 | Original252/252 full-width equality checks and source; raw `reference8`/`integrated8` logits independently compared again; checkpoint651 hashes rechecked | Passed exact equality, not tolerance/proxy only |
| Lower precisions actually change model computation | Original4/6 raw logit deltas positive; all signed-code/zero-group/rounding tests; real trained mixed-profile logits versus4/8 | Passed; lower width changes actual weights, not only metadata |
| (4) Separate attention and FFN precision |72-block model map; original single-attention/single-FFN raw outputs rechecked; isolation unit test; real final projection traces | Done:36 attention+36 FFN |
| Integration gate: revise failing representation, not loosen exactness | Earlier rotary-state failure/diagnosis/fix retained in `doc/debug-report.md` and meta logs;651 tensor hashes/actual logits pass now | Passed after recorded repair |
| (5) Train small query-dependent router on separate data | `R/router-training/A1/router.pt`, train-only normalizers,224 feature/target cache,300 losses, command/source; only192 train articles used for gradients | Done:166,104 parameters, seed31416, real training |
| Freeze train/dev splits and bounded attempts before fitting | Commit `c052ca3`, tracked `configs/router_data_lock.json`, `ROUTER_PROTOCOL.md`, `R/router-frozen/`;32 dev articles, full article/text/token provenance | Done; zero title/text and32-token overlaps with final as declared |
| No final leakage for training/selection/normalization | Recomputed normalizers/target stats/static mean costs from train only; label/cache IDs map to frozen train/dev; scorer receives context only | Done; final access only deterministic exclusion and later locked evaluation |
| Router gate: revise only from predeclared attempts on dev collapse | A1 loss decreases and passes declared variation gate; selection record A2/A3 not run; absent A2/A3 directories; no final-driven changes | Passed first attempt; bounded policy obeyed |
| (6) Different unseen queries yield different profiles | Dev31/32 profiles; final WT2 63/64, HellaSwag251/256, ARC-C255/256; all saved per-block distributions recomputed | Done; actual learned query variation, not manual profiles |
| (6) At least two precision levels genuinely used | All3 levels per query;304 actual executed projection/precision reconstructions checked; final traces count every252 projection per forward | Done; quotas force diversity and this is explicitly disclosed |
| Causality/repeatability under suffix/answer changes | Real GPU scoring API perturbation files and logits in `R/router-verify/`; context-only callback; unit and real repeat tests | Passed; options/suffix/gold cannot alter route |
| Feature-only ablation and constant-feature correctness | Train-mean features yield one dev profile; variable-length constant-coordinate test; reviewer P1 fixed before fit | Passed; masking prevents untrained length-feature amplification |
| (7) Adaptive versus fixed4/fixed8 | Same scorer/sample set; original fresh endpoints plus integrated exact endpoint smoke; all deltas/CIs retained | Done; endpoints correctly labeled4/8, not claimed equal to6 |
| (7) Adaptive/random/static same average bits | Six full `R/comparison-*-r{1,2}/` runs; per-query parameter counts/traces; exact independent weighted checks | Done:21,799,895,040 bits/3,633,315,840 projection weights=6 each |
| Fair query-independent static and random controls | Static recomputed as minimum mean train local NMSE; deterministic random seed/hash reproduced; static one profile; same quota solver | Done; static is surrogate-optimal, not proven downstream-optimal |
| Include probe work/exclusions/scales, not only unweighted bits | Same28,155 probe/147,626 scoring-input tokens and raw probe features across matched controls;389,152,256 FP16 exclusions;113,541,120 scale bytes | Done; logical scoring6, token-work6.320342; no physical low-bit claim |
| Final repeats and all returned results | `scripts/check_router_results.py` recomputes each raw metric, decision/cost/profile and demands exact repeats | Passed all three matched pairs; no filtered samples |
| Negative-result stop, no hidden search | Locked primary static WT2 CI[-.012750,-.003020] and MC guards pass; random/fixed4 negative findings retained prominently; A1 never revised | Study stopped; narrow positive against static is not general success |
| (8) Raw results/commands/configurations/tests/distributions/failures | All stage dirs/logs/source snapshots;20 final CPU tests; full route matrices; both guard refusals and prior test failures retained | Done; raw artifacts remain local and git-ignored |
| Final `REPLICATION_REPORT.md` separates statements/choices/findings/questions | Inspected report§1 paper,§2 choices,§3 measurements including negative controls,§4 failures/unknowns,§5 command/artifact map | Done; stage reports not used as substitutes |
| Exclude on-demand loading,8B models,PTB,full table,dynamic batching,async/kernel optimization | Executed commands/configs and unchanged source snapshots; only Qwen3-4B loaded, ordinary startup only; final report states exclusions | Preserved |
| External code only small reviewed parts; rerun here; no old results | Attributed tiny lm_eval prompt rules only; library APIs otherwise; all numerical evidence scoped to fresh core-v1 jobs | Preserved |
| `scripts/gpu_preflight.sh` before EVERY GPU job; one safe GPU; no interference | Final audit matched20 successful GPU logs to physical-device command records, immediate guard checks,30min timeout and exit0; both refusals exit4 | Passed; serial launches; no kill/reset/signaling other processes |
| Pause before long sweeps/model-license/scope change | Smoke projections before224-context/full576 jobs; own peak<=13.05GB; every full job<10min; predeclared repetitions only; no new license terms/packages/scope | Preserved; no unexpectedly long sweep |
| Named earlier progress evidence files | `BASELINE_RESULTS.md`, `INTEGRATION_RESULTS.md` retained; this completion audit now replaces the earlier pending checklist | Done |
| Only mark complete when every required artifact exists | This full table, final tests, raw-artifact/source/guard audit and report all inspected, not inferred from effort or green summary alone | Work product complete; no unverified required experimental item remains |

## Verification commands and their coverage

From repo root in the recorded `~/.venv`, with GPU hidden for CPU commands:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_baselines.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python results/core-v1/meta/final-artifact-audit.py
python scripts/audit_table.py
git diff --check
```

Actual outputs: `R/meta/tests-final.txt` (20/20), `R/meta/router-comparison-audit.log`,
`R/meta/final-artifact-audit.txt`, `R/meta/table-audit-final.txt`, plus earlier
baseline gate output. The final artifact audit rehashes the original model/data,
all651 checkpoint tensors, checks raw integration logits and checks current
inference sources against **every** recorded job source snapshot. It explicitly
verifies both refusal logs and confirms only A1 was fitted. No new GPU job was
needed or used for final inspection.

Coverage boundaries:

- Unit tests cover specific code invariants, not benchmark benefit or proof of
  separate training data on their own.
- The baseline verifier covers only fixed stage; integration verifier covers
  actual weights/outputs but not learned queries or quality.
- The comparison verifier covers real data/profile/metric/budget/repeat/CI
  artifacts, not paper-reading, scope/license compliance or report honesty.
- The final source/log audit covers artifact integrity and guarded executions,
  not the scientific sufficiency of the local-error surrogate.
- Those uncovered surfaces were inspected explicitly in the table above and
  report. No verifier's success flag was accepted as whole-goal proof.

## Qualified conclusion and residual risks

The narrow static comparison passes, yet random six-bit and fixed4 WT2 results
are better than adaptive. This is a completed **functional** reproduction with
strong negative controls, not reproduction of the paper's fixed8-quality,
performance or full-table claims. No additional tuning is authorized.

Remaining scientific unknowns (surrogate validity, domain shift, semantic or
pretraining contamination, window dependence, exact paper setup) are disclosed,
not missing required artifacts. Clean-host installation, portability, physical
low-bit performance and remote backup were not required/tested and are not
claimed. Preserve the local ignored result/model/PDF bundle separately from Git.
There is no PR/push/CI requirement for this local objective; no PR was created or
push performed. Runtime goal-accounting state is separate from this evidence audit.

## Fresh continuation audit and preservation — 2026-09-05 UTC

Revalidated **every requirement in the checklist above** against current files
at `a703e66`, rather than accepting this document's earlier completion status.
All five PDF pages and the five other prerequisite files were read again before
any changes. No implementation, frozen configuration, router or score changed.

Fresh evidence: `results/completion-20260905T193817Z/`.

- `checks.log`: all20 tests pass; fixed-stage and matched comparison raw metrics,
  profiles/budgets/repeats recomputed; all12 original model files, three final
  dataset sources, checkpoint and651 tensors rehashed; actual saved8/lower/block
  logits checked; all20 guarded GPU jobs and both refusals audited; table
  arithmetic and `git diff --check` pass.
- `independent-checks.log`: all576 final examples rederived exactly from pinned
  raw datasets/tokenizer/seed; current configs equal frozen copies; runtime
  versions match; all72 weighted block sizes match actual stored q tensors.
- `clean-source-tests.log`: all20 tests also pass in a pristine `git archive`
  extraction of `a703e66`, using the existing venv. This is not clean-host testing.
- `method-review.md`, `report-review.md`: independent read-only reviews find no
  mandatory gap. Explicit coverage limit:304 observed routed projection/precision
  equalities, not exhaustive756; final traces still cover every executed projection.
- `preservation.log`, `restore-smoke.log`: all616 core-v1 files plus PDF preserved
  and every archive member hash verified; Git bundle verified; three-file restore
  smoke passes. Archive and restore details: [doc/submission-check.md](doc/submission-check.md).

The scientific conclusion is unchanged: narrow adaptive benefit against the
preregistered static surrogate, negative results against random/fixed4 on WT2.
No further experiment is needed or authorized. **All required work products are
complete.** Same-filesystem preservation and untested new-host installation are
explicit limits, not evidence of remote backup or portability. No `update_goal`
tool is available in this session; no replacement goal or manual accounting
mutation was used.

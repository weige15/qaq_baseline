# Prompt-to-artifact completion audit

**Goal status: NOT COMPLETE.** This is a coverage checklist, not a verifier or
claim that artifacts alone prove success. Last inspected: protocol freeze,
CPU implementation and source/input manifests; GPU/model evidence still pending.

Concrete outcome: a functional, measured query-dependent Qwen3-4B block-precision
model, three baseline rows on three datasets, learned unseen query variation,
fair equal-bit adaptive/random/static comparisons, and an evidence-separated report.

| Requirement / explicit gate | Concrete evidence surface | Current state |
|---|---|---|
| Read `references/QAQ.pdf` before code | all5 pages extracted/read; SHA256 and extraction in `results/core-v1/meta`; protocol | Read, including page4 Table1 |
| Read `PAPER_AUDIT.md`, `DECISIONS.md`, `FIRST_GPU_RUN.md`, `configs/baseline_candidate.yaml`, `scripts/audit_table.py` before code | source inspection; protocol supersession banners | Read; old criteria explicitly superseded |
| Table1 defined as page4 five-method/three-model table, not target reproduction | protocol; `scripts/audit_table.py` raw output | Scope recorded; arithmetic check must be retained |
| (1) Frozen model/revision/tokenizer/data/metrics/software/commands/hardware | `configs/core_protocol.json`, `EVALUATION_PROTOCOL.md`, `results/core-v1/frozen/{protocol,model_manifest,data_manifest,environment,freeze_hashes}.json`, examples.jsonl, nvidia-smi/cpu/ram/pip-freeze | Inputs generated before new model scores; inspect run use/hashes too |
| (2) New FP16, fixed8, fixed4 on WT2/HellaSwag/ARC-C; all metrics | six complete fresh run directories with results.json, samples.jsonl, source and command snapshots | **Missing**; smoke alone cannot count |
| Baseline repeatability + internal-sanity gate | same-input fresh-process pairs and explicit gate report with per-choice/logit/metric tolerances | **Missing**; do not train router yet |
| (3) One signed/scaled stored quantized model supporting4/6/8 | persistent checkpoint + reload tests + actual model use | **Missing integration**; quantizer-only unit tests insufficient |
| Full8 exactly chosen8, lower4/6 change model outputs | every integrated tensor equality audit, actual Qwen logits and roundtrip evidence | **Missing model evidence** |
| (4) Separate attention/FFN selection |72 block mapping, mixed profile tests, changed actual projections/logits | **Missing integration** |
| Weight gate: revise if equality or output-change check fails | passing saved model integration gate | **Missing** |
| (5) Small trained query router, disjoint train/eval | preregistered bounded training attempts, data hashes/dedup, saved features/targets/training logs/checkpoint | **Missing** |
| (6) Unseen cross-query profiles and >=2 actual precisions | dev/final profile distributions, distinct assignments, repeated-query determinism, weight-use evidence | **Missing**; quota-forced diversity alone will not satisfy |
| Router gate: only bounded predeclared collapse repairs | attempt ledger and dev-only selection rationale | **Missing** |
| (7) Compare adaptive, fixed4/8, random, query-independent static with matched average bits | paired raw evaluations, trained static allocation, random seed, per-query sum(params*bits), FP16 exclusions and uncertainty | **Missing**; fixed endpoints not falsely called equal-cost |
| Stop negative vs matched static rather than extend search | preregistered success gate and final decision | Rule frozen; result **missing** |
| Causal routing (no evaluation answer/suffix used as features) | context-only routing API unit test; eventual trained-router perturbation test | API test exists; actual router test **missing** |
| (8) Raw results, commands, config, tests, distributions, failures | immutable `results/core-v1/` run artifacts and readable run records | CPU freeze/tests present; GPU/results/router artifacts **missing** |
| Final `REPLICATION_REPORT.md` distinguishes paper statements / implementation choices / measured findings / unknowns | report linked to actual raw artifacts; fresh completion audit | **Missing** |
| No CPU↔GPU on-demand, other models, PTB, full table, dynamic batching, async, kernel optimization | source and job-command audit | None added in current work; final audit still required |
| Only small reviewed external code; test and rerun locally; no old result evidence | scorer attribution, preprocessing tests, fresh runs and source snapshot | Tiny prompt rules only; old results excluded |
| `scripts/gpu_preflight.sh` before EVERY GPU job; one safe GPU; no process interference | logged preflight and exact shell command per job; refusal tests | Guard strengthened/tested; inspect every eventual launch |
| Pause unexpectedly long sweep / license decision / expansion | projected runtime from smoke,30min job caps, scoped protocol, failure ledger | No new license acceptance; pending GPU runtime checks |
| Mark complete only with every requirement supported | final audit of actual files AND coverage, then completion update | **Do not mark complete** |

Next concrete action: finish CPU checks, commit frozen protocol, inspect safe GPU,
run bounded fresh smoke jobs with logs, estimate full-job runtime, then baseline
pairs only if safe. Do not use old exploratory scores or a green unit suite as
completion evidence.

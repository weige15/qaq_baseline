# Query-budget feasibility completion audit

## Concrete objective / completion criteria

Determine within each of the three closed tasks whether per-query fixed8 versus
fixed4 benefit offers material same-average-bit opportunity, and whether one
frozen pre-answer router score identifies it. Deliver a frozen protocol/config,
pure deterministic analysis/CLI, focused hand tests, two byte-identical fresh
runs, an independent arithmetic audit, an evidence-separated report and this
requirement-by-requirement audit. Preserve the closed tag/files/results exactly;
add only the seven authorized paths and append-only study outputs. Completion
does not depend on obtaining a positive research decision.

**Work-product verdict: verified complete**, supported by the inspected evidence
below and `R/preservation-final.json` with `passed: true` and no issues.
**Frozen scientific decision: `revise_signal_design`.** Three opportunity tasks,
one recovered task; this does not authorize implementing a router. No required
experimental artifact or verification remains pending. The accounting tool
`update_goal` is not exposed in this session, so the agent cannot issue that
status update; no replacement goal or manual accounting mutation was used.

`R` below means `results/query-budget-feasibility-v1/`. Tracked deliverables are
currently seven **untracked additions**, intentionally not committed or staged.
The existing repository and all closed raw outputs remain unchanged.

## Prompt-to-artifact checklist: preparation and numbered analysis

| Explicit requirement | Concrete evidence inspected | Result / coverage limit |
|---|---|---|
| Read `README.md`, `REPLICATION_REPORT.md`, `COMPLETION_AUDIT.md` before changing anything | Complete read-tool outputs in this session before the first new file; hashes in `R/preflight/initial-state.json.protected_tracked` | Done; historical results/limitations informed scope, not score selection |
| Read `ROUTER_PROTOCOL.md`, `ROUTER_RESULTS.md`, `EVALUATION_PROTOCOL.md` | Complete reads before editing; same initial hashes | Done; original NMSE surrogate, likelihood scorer and closed status understood |
| Read `configs/core_protocol.json`, `configs/router_protocol.json` | Complete reads before editing; initial protected hashes | Done; three frozen subsets and primary metrics retained |
| Read `src/qaq/router.py`, `src/qaq/evaluation.py`, `scripts/router_job.py` | Complete reads before editing; `router_costs`, `RoutingSession`, `evaluate` traced | Done; costs columns 4/6/8 and context-only fixed8 probe established |
| Inspect fixed4/fixed8 raw samples, adaptive records, integration counts/manifests/hashes | `R/preflight/input-checks.json`, frozen manifests, endpoint commands/results/module inventories, adaptive commands/results/routes, integration gate | Done; 1200 historical referenced hashes verified; score uses actual 72-block/252-projection inventory |
| Pause on absent or hash-invalid required artifact, do not regenerate a closed result | Preflight passed; `load_pairs` rechecks 30 sealed analysis inputs and historical inventory; focused missing/hash tests; CLI refuses unsafe writes | No required artifact missing/invalid; no model result regenerated |
| Fixed repeats: same ordered task/index pairs and exact non-timing results | Preflight explicit fixed4/fixed8 checks; loader compares each to frozen 576 IDs, entire samples and result dicts excluding ONLY `eval_seconds`, `total_seconds` | Exact, not relaxed numerical tolerance; repeats are not new independent examples |
| Adaptive records repeat / causal linkage | Preflight and both loader executions compare whole routes excluding ONLY `probe_seconds`; match `context_tokens` to frozen prefixes and recorded counts | Done; adaptive final quality is not used for study outcomes |
| Record initial HEAD, status, protected and raw hashes BEFORE editing | `R/preflight/initial-state.json`: clean HEAD/tag `b59adbcffc90b870f014d22e142923d569596986`, 59 tag files, 1196 protected raw files; SHA `fd21426a…63f3` | Done before source/document additions |
| Freeze BOTH proposed protocol files before any new aggregate outcome | `R/protocol-freeze.json`, SHA `5a8b1fba…a115`; 00:39:55.885745 UTC seal versus 00:47:27 first-run start | Done; CLI pins receipt and checks both protocol hashes; no post-result edits |
| (1) WT2, HellaSwag, ARC-C separately; no pooled incompatible score | Protocol §1, config task map; `analyze` task loop and `aggregate` rejects pooling; independent raw task loops | 64 / 256 / 256, no pooled score; report has separate task tables |
| (2) Existing fixed endpoints only; NLL direction | `make_pair`: fixed4 mean NLL minus fixed8 mean NLL; raw WT2 ledger and sign tests | Positive = beneficial; 55 positive, 9 negative, 0 zero |
| (2) Preserve NLL sums AND scored-token counts in aggregation | Endpoint sums and both counts in every WT2 query ledger; all WT2 budget metrics include sum/count/mean; ratio-of-sums test | 24,576 tokens per real aggregate; unequal-token test verifies no mean-of-means shortcut |
| (2) MC primary acc_norm three transition categories; ordinary acc secondary | Per-query `acc_norm_transition` and `acc_transition`, predictions/correctness/gold; task transition tallies; report §§1–3 | All 512 MC examples categorized; same selections for secondary metric, no secondary re-ranking |
| (3) Exact 4/5/6/7/8 budgets, half fixed8 at six, ties deterministic | `quota` rational integrality check, `budget_result` exact rational mean and IDs; all 90 selections independently audited | WT2 fixed8 0/16/32/48/64; each MC 0/64/128/192/256; IDs form disjoint exhaustive partitions |
| (4) Query-independent expected exact-quota assignment and small deterministic variability list | Config salts 1729/2718/31415, protocol SHA256 string recipe, expectation equation; exhaustive hand enumeration of all subsets | All three retained at every budget; no seed search; expectation does not execute fractional queries |
| (5) Outcome-informed top observed benefit limit, unavailable in inference | `ranks.outcome_informed`, all ranked IDs, independent raw-rational ordering; report §§1,4 | Explicitly unavailable and unsuitable as deployable method; interval is not future generalization |
| (6) Exactly one pre-answer score: parameter-weighted predicted cost4 minus cost8 | `router_score` uses `math.fsum`, actual integration counts, cost columns 0 and 2; weighted-versus-unweighted reversal hand test; independent rational recomputation | All 576 scores/ranks checked; denominator 3,633,315,840; no transforms, fitting, alternate weights/subsets/formulas |
| (7) Prompt token count control and fixed8-probe caveat | `make_pair.prompt_tokens`, descending `prompt_length` rank plus task/index ties; report §§1,5 | WT2 all 128 so control becomes index order; complete fixed8 prompt probe explicitly unpaid |
| (8) Six-bit primary only; all methods compared to same-budget expectation | `analyze` primary slice and `screening`; tests show only six-bit metrics/intervals feed decisions; report threshold table | 4/5/7/8 are descriptive only; seven-bit WT2 result cannot rescue failure |
| (9) Paired query resampling with frozen seed/repeats and 95% intervals | Seed 4242, PCG64, 10,000 draws reset per task; shared sampled pairs across methods/metrics; occurrence re-ranking preserves half quota; explicit percentile linear | All 30 method/metric intervals independently recomputed, including secondary acc and hash controls |
| (9) Plain explanation / limits of resampling | Protocol §9 and report §§1,4,5 | Repeatedly sample same saved pairs with replacement; no extra independence from repeat runs; outcome-informed interval not generalization evidence |

## Frozen screening rules and measured decision

| Required frozen rule | Evidence | Outcome |
|---|---|---|
| Material opportunity: >=.01 primary improvement, CI excludes zero beneficially | Exact `screening`, unrounded raw values, independent audit; boundary tests for equality / nextafter / CI zero | WT2 .037475402835773775, CI [.029050332421988602,.04588793194806684]; HS .044921875, CI [.029296875,.0625]; ARC .083984375, CI [.0625,.107421875]; all pass |
| Recommend separate router study only if >=2 opportunity tasks AND >=2 recover >=25%, nonnegative | Same-budget oracle/router deltas and independent un-clipped ratios | WT2 7.9693%, HS −21.7391%, ARC 25.5814%; only ARC passes; no direct continuation recommendation |
| >=2 opportunity tasks but router failure => separate signal-design study, no implementation | `analysis.json.decision` in both runs, independent decision | `revise_signal_design`; no new router/serving code |
| Exactly one opportunity => task-specific revision; between-task differences insufficient | Protocol explicitly reconciles one-task revision with broad-stop rule; one-task decision fixture | Real data has all three within-task gates, not solely task-level differences |
| Fewer than two opportunity tasks => stop broad direction | `decision.broad_direction_stopped`, zero/one-task fixtures | Real data: false (three opportunity tasks); one-task fixture stops broad and revises task-specific |
| Record fixed8 harms; no assumed monotonicity | All per-query benefits/transitions, report negative-results section | WT2 harm 9/64; primary MC losses 6/256 HS and 9/256 ARC; secondary harms also reported |
| Pause on ambiguity instead of adjusting thresholds/data | Nonfinite/invalid screen rejection, fixed protocol hashes and post-run source hash equality | No unresolved rule ambiguity. ARC router CI includes zero but interval exclusion was never its frozen recovery gate; disclosed rather than silently adding/changing a gate |

## Deliverables and implementation boundaries

| Required path or constraint | Actual artifact / verification |
|---|---|
| `QUERY_BUDGET_FEASIBILITY_PROTOCOL.md` | Frozen complete methodological rules, tie/resampling/threshold definitions and scope; SHA `8781baa938ed25259a3cff0ce73f9b62c4854dc4ec60d74616c81623ee1dfbac` |
| `configs/query_budget_feasibility_protocol.json` | Frozen machine settings and all 30 analysis-input hashes; SHA `1ea275b923ea534f0917ca5fb466a2d5ce5288a2b4617b287da37d3bb5bf6bd8` |
| `src/qaq/query_budget_analysis.py` | Pure deterministic validation/pairing/aggregation/ranking/paired-bootstrap/screening; stdlib and existing NumPy only |
| `scripts/analyze_query_budget.py` | CPU-only CLI, pinned seal, pre-aggregation hash/repeat validation, exclusive new output creation; no model imports |
| `tests/test_query_budget_analysis.py` | 16 focused tests listed below, all pass |
| `QUERY_BUDGET_FEASIBILITY_REPORT.md` | Confirmed measurements, analysis-only limits, assumptions, unknowns, negative findings and threshold decision separated; full primary intervals and descriptive curves |
| `QUERY_BUDGET_FEASIBILITY_AUDIT.md` | This prompt-to-artifact audit, not a proxy for missing measurements |
| Only these seven new tracked-eligible paths | `R/preservation-final.json.only_allowed_additions` exactly equals the list; index empty; tag diff empty; additions remain untracked intentionally |
| Do not modify `src/qaq/__init__.py` or ANY existing/tag file | All 59 tag files rehashed with size, mode and mtime equal initial; git diff against `qaq-baseline-closed-v1` empty |
| Immutable `results/core-v1`, `results/on-demand-v1`, `results/batch-interference-v1` | Final inventory exactly same 1196 paths; hashes, sizes, modes and mtimes unchanged, including existing bytecode. No closed gate-writing script run |
| New raw output only append-only in `R` | All production runs/audits/logs below `R`, exclusive creation; failed/earlier checks retained. Temporary unit-test fixtures are synthetic, not benchmark raw outputs |
| No commits, push, merge, switch, packages, GPU jobs, retraining, new task/example search, batching/KV/vLLM | Executed command record, unchanged HEAD/index/sources/protected files; analysis has no model or training call; only existing CPU tests use tiny synthetic models |
| Report excluded measurements explicitly | Report opening and §4: no runtime, GPU memory, low-bit kernels, generation, KV cache, batching or vLLM measurement |
| Future requirements not passed off as completed | Report §6 separately lists fixed6 full-model evaluation, separately trained cheap pre-answer selector, probe-cost measurement and fair global-average-bit comparisons |

## Focused test coverage — actual tests, not just a count

All names below have prefix `QueryBudgetTests.test_`. Command output in
`R/focused-tests-final.log` is 16/16, exit 0. `R/all-tests-final.log` is 40/40,
exit 0: all 24 original CPU tests plus these 16. No existing test was edited.

| Explicit test requirement | Test suffix / inspected assertions |
|---|---|
| NLL direction; negative benefit | `nll_direction_negative_benefit_and_counts`: per-token benefits [4,2,0,−2], correct beneficial sign |
| MC transition classes; ordinary secondary | `mc_transition_categories_and_secondary_metric`: +1/0/0/−1, all three classes, acc_norm .75 and secondary acc .25 on SAME selection |
| Exact budgets; no dropped/duplicated queries | `all_five_budgets_exact_and_no_dropped_or_duplicated_query`: all budgets, exact rational averages, disjoint/exhaustive IDs; invalid nonintegral quota and repeated rank rejected |
| Deterministic ties | `deterministic_ties_use_task_then_integer_index_not_input_order`: reverse input, shared length/score/benefit and cross-task ID ties |
| Unequal token counts | `unequal_token_counts_are_ratio_of_sums_not_mean_of_means`: selected NLL32/tokens10=3.2; expected40/10=4; not unweighted mean; mismatched tokens rejected |
| Query-independent expected result | `query_independent_expectation_matches_all_exact_quota_subsets`: exhaustively enumerate every exact-quota subset at all five budgets for WT2 and MC |
| Outcome-informed limit / required harmful upgrade | `outcome_limit_does_not_skip_negative_benefit_to_meet_budget`: never skip a required upgrade when all observed benefits negative |
| Frozen score weighting/ranking | `frozen_router_parameter_weighted_ranking_not_block_mean`: weighted 1 vs1.8 reverses unweighted ranking; middle cost irrelevant, negative score retained; invalid cost/count rejected |
| Small fixed deterministic list | `hash_selections_are_predeclared_and_reproducible`: exact UTF-8 SHA256 recipe and all three frozen salts |
| Repeated IDs, missing queries, wrong order | `repeated_ids_missing_queries_and_wrong_order_fail`: duplicate/missing/reversed fixtures rejected; same index on separate tasks allowed; pooling rejected |
| Mismatched repeats | `mismatched_repeats_and_only_declared_timing_ignored`: ignores only allowed timing, rejects altered outcome/ID/peak bytes or query count |
| Missing paths / hash failures | `missing_and_hash_invalid_paths_are_reported`: hand fixture original digest accepted; mutated file and absent exact path rejected |
| Per-query raw consistency | `token_loss_validation_and_pair_identity`: incorrect NLL sum and cross-endpoint ID mismatch rejected |
| Paired resampling / exact budgets / repeated draw IDs | `paired_resampling_hand_draws_reselect_exact_half_and_preserve_tokens`: two explicit repeated-pair draws, deltas .8 and2/7, hand-interpolated95% bounds, zero-benefit interval [0,0] |
| Decision boundaries | `screening_exact_boundaries_negative_signal_and_ci_touching_zero`: equality .01/.25 passes, nextafter below fails, zero CI boundary fails, negative ratio preserved, NaN rejects |
| All frozen decisions / nonprimary cannot rescue | `all_decisions_and_descriptive_budgets_cannot_rescue_primary`: continue/revise-signal/revise-task/stop branches and exact primary-only extraction in synthetic full analysis |

## Determinism, arithmetic independence and commands

| Required check / actual command | Evidence and result |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python -m unittest discover -s tests -v` | `R/all-tests-final.log`: 40 tests, OK, exit0; earlier all-tests run also passes |
| Same prefix, `-m unittest discover -s tests -p test_query_budget_analysis.py -v` | `R/focused-tests-final.log`: 16 tests, OK, exit0 |
| Same prefix, `scripts/analyze_query_budget.py --out results/query-budget-feasibility-v1/run-1` | `R/run-1.log`: fresh process, exit0; no prior completed output reused |
| Same prefix, `scripts/analyze_query_budget.py --out results/query-budget-feasibility-v1/run-2` | `R/run-2.log`: fresh process, exit0; both `analysis.json` and `manifest.json` byte-identical |
| Independent CPU `R/independent_arithmetic_audit_v2.py` | `R/independent-arithmetic-v2.{json,log}`: passed; imports no `qaq`/checked implementation; raw-rational points/ranks, Python stable sorting of draw occurrences and manually interpolated percentiles |
| Independent audit coverage, not merely success flag | 576 individual ledgers and both endpoint metrics/repeats; 90 exact task/budget/method selections; all30 primary-budget method/metric interval pairs; every rank/selected/unselected ID; primary rules and decision reconstructed. Input/source/manifest/seal hashes and both run bytes checked |
| Arithmetic tolerances versus determinism | Independent floating sums may differ: metric/CI1e−13, NLL sums2e−11, score1e−14, recovery1e−12; all IDs/counts/decisions exact. Production fresh-run output bytes are EXACT, no tolerance |
| `R/cli_safety_audit.py` | `R/cli-safety.{json,log}`: five expected refusals (completed output, protected path, outside root, nonempty GPU visibility, bytecode env); no new forbidden directory and completed-output hashes unchanged |
| `git diff --check` | `R/preservation-final.json.whitespace_checks`: exit0, no diagnostics |
| `git diff --check qaq-baseline-closed-v1 --` | Same: exit0, no diagnostics; full tag diff and staged diff empty |
| `git diff --no-index --check /dev/null PATH` for each of seven additions | Same: no diagnostics, return1 is normal no-index difference, not whitespace error; all seven inspected, since normal git diff excludes untracked additions |
| CPU `R/preservation_audit.py` | `R/preservation-final.{json,log}`: passed; all protected inventories/metadata, seal, run source/input hashes, exact run files and allowed changes checked; no cleanup action |

Deterministic analysis SHA256:
`78eddf15e7e0ebb79957ac3c5ac5b6a1aa49347d71130631abd6d1fbfac57251`.
Deterministic manifest SHA256:
`7d7de0590aee3c34adf1060fb9f4635f289e24cdcc0889ff96d9f94b32286ebb`.
The final preservation record includes hashes for all seven additions and prior
raw outputs, without a circular self-hash. Run source hashes match the current
five protocol/code/test sources; report/audit were added afterward as planned.

## Failures and strengthened checks retained, no hidden replacement

1. Initial shell inspection used `python`, absent on PATH (exit127). Reissued
   using the already-installed `~/.venv/bin/python`; no installation or raw
   model rerun. This is recorded in the session tool output.
2. `R/preflight/focused-tests-1.log` records one failing hand-fixture expectation:
   sampled counts 1+1+2+3 total7, not5. Corrected only expected values/comment in
   the NEW test file before either real analysis run. Protocol and algorithm
   unchanged. Subsequent full and focused suites pass.
3. Independent arithmetic v1 passes but oracle ordering consumed benefit values
   already cross-checked against raw rational arithmetic. V2 strengthens
   independence by ranking only raw-derived values; no implementation import
   in either. Both scripts/logs/results remain, no overwrite or deleted failure.
4. CLI refusal cases are deliberate negative tests, not failed model experiments.
   No failed benchmark outcome was substituted with a new score.

## Statistical integrity and remaining uncertainty

11/11 methodological fallacy checks were considered: task pooling/Simpson and
ecological inference are avoided by separate task/query analyses; selection/
Berkson risk remains from small frozen subsets; no collider adjustment is fitted;
base rates and both-correct/both-wrong/harm counts are retained; regression-to-
mean / outcome-informed selection optimism is explicitly an analysis-only limit;
no survivorship filtering (all576 retained); look-elsewhere and forking-path
risks bounded by one frozen score, primary budget, salts and thresholds, but no
multiple-comparison correction is claimed; no observational-score association is
asserted to cause practical benefit; temporal pre-answer availability rules out
answer-scoring access for the router score but not surrogate mismatch.

The outcome-informed interval is NOT a generalization guarantee. WT2 window
independence is an assumption. All three router intervals include zero, and
ARC's point recovery barely clears25%; these are disclosed weaknesses, not
missing required artifacts or permission to change the decision rule. A frozen
screen can yield a definite **revise** decision despite weak predictive evidence.

There is no PR, push, CI, new-host installation, new benchmark, or deployment
requirement; none is claimed. The local ignored evidence bundle is not an
off-host backup. A future separately authorized study is required to address
cheap signal design, fixed6, probe cost and fair global average bits.

## Completion audit coverage boundary

Green tests verify particular code invariants, not dataset provenance or the
scientific claim. A manifest verifies listed hashes, not whether every requested
item was included. Exact repeats verify reproducibility, not usefulness. The
independent arithmetic audit verifies computations, not deployability. The
preservation gate verifies boundaries, not report honesty. This checklist and
inspection of protocol/report/source/raw fields close those separate surfaces.
All required deliverables/checks are present; the negative predictive finding is
an allowed completed outcome, not an unfinished implementation to keep tuning.

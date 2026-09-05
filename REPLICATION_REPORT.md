# QAQ core replication report

## Outcome

**The bounded functional reproduction is verified. The broader quality claim is
not reproduced.** Qwen/Qwen3-4B uses one signed/scaled quantized checkpoint, and
an actually trained, context-dependent router changes attention and FFN weights
independently at4/6/8 bits on unseen examples.

The adaptive policy narrowly beats our train-optimized static policy at exactly
six average bits under the preregistered test. **However, random six-bit routing
has substantially better WikiText perplexity than both learned policies; even
fixed4 beats adaptive on WikiText.** This is not evidence for a useful general
accuracy/efficiency advantage, nor for matching fixed8 quality. The experiment
ends here: no more seeds, router attempts, datasets or policy searches.

This report is NOT a reproduction of page4 Table1. It does not reproduce any
on-demand memory/latency claim. The table below contains only this repository's
fresh `results/core-v1/` runs on frozen subsets, never old/external results.

## 1. Paper statements (not our measurements)

Source: `references/QAQ.pdf`, five pages, SHA256
`f15f1d6cbaaaf7ee74e8300b9a9d2b46d180a3962019cf3a8acc7de252c37c8e`.
All pages and the five other required input files were read before code changes.

- Pages2–3 describe bit-plane weights, a lightweight MLP using block hidden
  representations, precision probabilities, and full-precision teacher/student
  knowledge distillation. Figure1 distinguishes attention and FFN blocks.
- Page4 Table1 compares FP16, fixed8, fixed4, QAQ on-demand off and on for
  Qwen3-4B/Qwen3-8B/LLaMA-3.1-8B. The paper reports near-fixed8 quality and
  memory/latency trade-offs. These are paper claims, not evidence about our code.
- The paper does not specify signed representation/scales, exact quantizer,
  router inputs/objective/data, precision candidates, checkpoint/data revisions,
  evaluation definitions or hardware. Its nonnegative bit-plane equation omits
  the sign and scale required by real transformer weights.
- `PAPER_AUDIT.md` and `scripts/audit_table.py` retain the arithmetic audit and
  ambiguities, including potential HellaSwag metric mixing. Historical
  paper-matching/Base-checkpoint proposals are superseded by our frozen protocol.

## 2. Our frozen choices

### Evaluation and environment

Preregistration: baseline `8fc7a05`, integration `71d1102`, router `c052ca3`.
Authoritative settings: `configs/core_protocol.json`, `EVALUATION_PROTOCOL.md`,
`ROUTER_PROTOCOL.md`, `configs/router_protocol.json`, `configs/router_data_lock.json`.

| Item | Fixed setting |
|---|---|
| Model/tokenizer | `Qwen/Qwen3-4B`, `1cfa9a7208912126459214e8b04321603b3df60c`, **not Base** |
| Model terms | Exact-revision model card declares Apache-2.0; no gated/license acceptance decision |
| WT2 revision | `Salesforce/wikitext`, `b08601e04326c79dfdd32d625aee71d232d685c3` |
| HellaSwag revision | `Rowan/hellaswag`, `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| ARC-C revision | `allenai/ai2_arc`, `210d026faf9955653af8916fad021475a3f00453` |
| Evaluation | Local causal likelihood scorer; batch1; zero-shot; no chat template, BOS/EOS insertion or generation |
| WT2 final | Raw test joined by two newlines;64 seeded nonoverlapping512-token windows; first128 context, last384 scored |
| MC final |256 seeded HellaSwag validation and256 ARC-Challenge test examples; all choices scored |
| Primary metrics | WT2 mean NLL; both MC `acc_norm`, fixed before any new scores |
| MC normalization | Total continuation log-likelihood / answer Unicode character count, excluding added leading space |
| Determinism | Seed1729, four CPU threads, FP16 SDPA, deterministic torch, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Runtime | Python3.12.3; torch2.5.1+cu121; Transformers5.16.1; NumPy1.26.4; SciPy1.17.1; PyArrow24.0.0 |
| Hardware | One idle, process-free NVIDIA RTX3090 per job,24GiB; new jobs used GPU6 except static repeat2 on GPU7 |

The original freeze contains full package lock, CPU/RAM/driver details, source
file hashes, tokenizer IDs, indices, prompts and revision manifests. Every GPU
job has its command, physical-device record and source snapshot. Runtime packages
were not changed during this continuation. `lm_eval`0.4.13 is only an attributed,
reviewed prompt-preprocessing reference; it did not execute these evaluations.

**WT2 here means prefix-conditioned token perplexity**, not harness word/byte
perplexity or the paper's unspecified WT2 definition. MC subsets and64 windows
limit power. Every returned metric is retained: WT2 NLL sum, scored tokens, mean
NLL, token PPL, window-NLL SE; MC acc/acc_norm, both SEs, and count. No metric was
chosen after seeing whether it matched a paper cell.

### Weight representation and block granularity

- All252 q/k/v/o/gate/up/down projection weights: groups of128 contiguous input
  columns, FP32 absmax/127 scales, ties-to-even round/clamp to[-127,127], signed
  int8 two's-complement codes. Embeddings/head/norms remain FP16.
- For precision b, s=8-b, reconstruction is
  `((q >> s) * 2**s + (2**s-1)/2) * scale`, then FP16. At8 it is exactly the
  chosen q×scale. Lower precisions depend only on retained signed high bits.
  Zero-valued groups have scale0; zero codes in nonzero groups shift at lower
  widths. Fixed4 is **nested midpoint4**, not separately scaled conventional RTN.
- One saved4,525,416,138-byte model contains q/scales, FP16 exclusions and actual
  rotary buffers. SHA256
  `b63aeeed85e11cea5d2790b1aae068920626e5d93117548358f76ae3b2171b85`.
  There are no saved4/6/8 model variants. The historical NumPy sign-magnitude toy
  is not the active representation.
- There are36 attention and36 FFN decisions. One transient FP16 reconstruction
  per projection is replaced on precision changes and never serialized.
  These are ordinary FP16 matmuls: **selected bits do not imply low-bit resident
  GPU memory, packed transfer volume or optimized kernel speed.**

### Router training and fair controls

- Freeze192 WT2 train and32 validation articles using seed2718; one512-token
  window per article, first128 tokens only for labels/features. No cross-article
  windows. Test-title/text and exact32-token-span exclusions cover all final
  examples/options; train/dev overlap is also excluded. Independent CPU audit
  rehashed source parquet files and retokenized every selected article window.
- A context-only fixed8 student prepass supplies normalized block inputs. Replay
  each local block at4/6/8 and in the same-revision frozen FP16 teacher at the
  **identical input/mask/positions**. Target: output MSE / FP16 output energy.
  Only original fixed8 outputs propagate. No backbone gradients or suffix labels.
- A1:72 independent68→32→3 ReLU MLPs,166,104 parameters. Inputs are32 channel
  mean/RMS groups and4 scalar summaries. Train-only standardization; constant
  coordinates masked in both fitting and inference. CPU FP32, seed31416, AdamW
  lr.003, weight decay.01,300 full-batch epochs, norm clip1, final epoch only.
  Regression predicts standardized log local NMSE; inverse transform gives costs.
- Exact assignment minimizes summed costs under12 each4/6/8 in attention and
  separately FFN, using installed SciPy's deterministic assignment solver.
  The static policy optimizes the same quotas from **mean raw train NMSE**.
  Random uses a frozen seed1618 plus context-token hash, with no gold/quality input.
- All three matched policies execute the same fixed8 context probe; static and
  random discard its features for decisions. One profile scores every suffix or
  answer option. This extra pass is a deliberate implementation difference from
  the paper's unspecified routing timing, not an efficiency achievement.
- Three attempts were declared. **A1 passed; A2/A3 were never fitted.** Selection
  used dev noncollapse, not final performance. Quotas enforce three levels;
  learned adaptation is established by between-query variation, not quotas alone.

## 3. Measured findings

### Gate evidence

- Baselines: all three modes have identical complete raw samples across two
  fresh processes. Internal-coherence thresholds pass. `BASELINE_RESULTS.md`.
- Integration:252/252 full-width weights and full8 logits exactly equal fixed8;
  all651 persisted tensors rehash exactly after reload. Actual lower4/6 and
  separate attention/FFN changes affect model logits. `INTEGRATION_RESULTS.md`.
- Training: standardized log-NMSE train loss1.516051→0.001761; dev1.544935→0.985984.
  **Substantial train/dev gap**; loss fitting alone is not useful adaptation.
- Unseen dev:31 profiles/32 queries, largest frequency6.25%,16 attention and19
  FFN blocks vary. Recomputed features and profiles exactly match cached fit
  results; constant-feature ablation yields one profile.
- Actual routed computation:304 distinct executed projection/precision checks
  match reconstructed weights exactly; on3 dev contexts adaptive logits differ
  from fixed4 and8. Real scoring API suffix/options/gold perturbations preserve
  the route; repeated scoring is exact. Integrated endpoint smoke equals the
  earlier fresh baseline raw losses. `ROUTER_RESULTS.md` and `router-verify/`.
- **20 CPU tests pass**, including quantizer nesting, exact serialization,
  independent blocks, local teacher replay, causal routing, constant-feature
  masking, optimal quota solver, weighted budgets, metric/prediction-tamper
  rejection and paired-bootstrap arithmetic. Tests alone are not completion.

### All six methods, unchanged final examples

All values below are fresh local measurements. Each mode has two identical
complete raw runs. Accuracy shown as fractions, not percentages.

| Mode | Scoring bits¹ | WT2 mean NLL | WT2 token PPL | HellaSwag acc | HellaSwag acc_norm | ARC-C acc | ARC-C acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 |16|2.842996|17.167127|.500000|.695313|.500000|.562500|
| Fixed8 |8|2.846345|17.224705|.496094|.695313|.503906|.558594|
| Fixed4 |4|2.938453|18.886606|.488281|.652344|.406250|.460938|
| Adaptive A1 |6|3.008898|20.265051|.488281|.667969|.429688|.531250|
| Static local-error policy |6|3.016863|20.427117|.488281|.667969|.441406|.503906|
| Random |6|2.879857|17.811721|.507813|.660156|.437500|.531250|

¹ Quantized projection weights only. Fixed4/8 are cost endpoints, **not** equal-bit
comparators. Adaptive/static/random each select21,799,895,040 bits over
3,633,315,840 projection weights **on every query**, exactly6.0 bits. All also
have389,152,256 excluded FP16 parameters and113,541,120 scale bytes. Including
excluded weights gives6.967446 bits/parameter; including scales gives7.193260.
These are logical accounting quantities, not measured packed GPU footprints.

The matched controls all process28,155 fixed8 probe tokens and147,626 scoring
forward-input tokens. Token-work-weighted precision on quantized projections is
6.320342 bits, identical across controls. This is a simple token accounting
measure, not FLOP/latency equivalence. Own-process measured evaluation peaks are
about12.53GB; job times include roughly220s of initialization and346–350s of
scoring/instrumentation. No performance claim is made.

Full metrics (including every SE/NLL total/count omitted from the compact table),
raw losses, per-query costs/features, all per-block distributions and comparison
CIs are in `results/core-v1/comparison-gate.json` and each run's result/sample files.

### Locked decision and negative controls

Paired bootstrap:10,000 draws, seed4242, percentile95% intervals; resample WT2
windows or MC examples, same indices per task across comparators. No alternative
resampling scheme was selected for a favorable outcome.

| Adaptive minus comparator | WT2 mean-NLL delta [95% CI] | HellaSwag acc_norm delta | ARC-C acc_norm delta |
|---|---:|---:|---:|
| Static |−.007966 [−.012750, −.003020]|.000000|+.027344|
| Random |+.129041 [+.108353, +.150026]|+.007813|.000000|
| Fixed4 |+.070445 [+.049858, +.091644]|+.015625|+.070313|
| Fixed8 |+.162553 [+.145667, +.179782]|−.027344|−.027344|
| FP16 |+.165901 [+.149098, +.183149]|−.027344|−.031250|

Lower NLL is better. The frozen primary criterion **passes narrowly against this
static comparator**: WT2 CI entirely below0, neither MC acc_norm falls by>.02.
Static-comparison MC acc_norm CIs are[−.019531,+.019531] for HellaSwag and
[+.007813,+.050781] for ARC-C. Raw ARC-C `acc`, also retained, decreases by.011719
versus static; we do not substitute that metric or hide it.

**Crucial limitation:** adaptive is substantially worse than random at the same
bits on WT2, and worse than fixed4 despite using more logical bits. The result
supports a small input-dependent improvement over this local-error static
assignment, not a reliable mixed-precision optimization method. The local-error
static control is optimal for its declared surrogate, **not** a demonstrated
strong downstream-language-model policy. Apparent nonmonotonic quality across
mixed profiles is compatible with interacting quantization errors; it is not
proof of the cause. The repeat, weight and scorer gates rule out the specific
implementation failures they test, not every possible methodological weakness.

### Unseen final routing

| Final task | Queries | Adaptive unique profiles | Attention blocks varying | FFN blocks varying |
|---|---:|---:|---:|---:|
| WT2 |64|63|19|21|
| HellaSwag |256|251|25|25|
| ARC-C |256|255|26|27|

Every query uses24 blocks each at4/6/8, split12/12 by block type. Every executed
projection's selected bits are traced; every option uses the same profile.
Static has one profile throughout. Random has64/256/255 distinct task profiles.
One duplicate ARC context receives the same deterministic random profile;
randomness is not evidence of learned adaptation. Adaptive profiles, features,
costs, predictions and raw token losses exactly repeat in fresh processes.

## 4. Failures, deviations and unresolved questions

### Recorded failures and repairs

1. Earlier baseline stage: an initial expected-whitespace test typo was corrected
   without changing prompt preprocessing. Original failed test log remains.
2. Earlier integration: nonpersistent rotary frequencies differed after reload;
   exact actual buffers are now saved. No tolerance relaxed. `doc/debug-report.md`.
3. Router review found an untrained constant log-length feature could explode on
   variable-length MC contexts. Train-constant masking was frozen and tested
   **before labels/fitting**. Full reviewer report is retained.
4. GPU preflight refused one earlier fixed4 launch and one new random-repeat
   launch (availability changed on immediate recheck, exit4). Neither launched
   a model. Clean rechecks and separate retry logs are retained; no process
   belonging to another user was signaled or reset.
5. Fresh PDF-reader commands found no `pdftotext`/reader in the runtime venv.
   The existing isolated `/tmp/qaq-pdf-reader` extracted the unchanged PDF; no
   evaluation-environment package installation was needed.

No successful result was substituted for a failed model run. No router attempt
was discarded based on final scores. No final-score-driven repair/search occurred.

### Open limitations (not silently filled in)

- Does local normalized output error predict downstream mixed-profile loss?
  These results give serious counterevidence to its adequacy as a surrogate.
- How much does training on128-token WT2-only prefixes transfer to different
  query lengths/domains? Constant masking fixes one bug, not distribution shift.
- Train/dev decontamination checks exact spans/titles/text, not paraphrases or
  the model's pretraining data. Final WT2 windows may share article context;
  window bootstrap does not remove that dependence. No population/full-split
  superiority or formal multiple-comparison guarantee is claimed.
- The paper's exact quantizer, data, router objective, checkpoints, metrics,
  loading implementation and hardware remain unknown. Our use of hard quotas,
  midpoint signed reconstruction and a separate probe are explicit choices.
- No on-demand loading,8B models,PTB,full table,dynamic batching,async loading,
  custom kernels or physical low-bit storage/speed benefit were attempted.
- Fresh-environment installation/portability is not tested. The frozen runtime
  and local ignored artifacts are required; `pyproject.toml` is the historical
  toy package manifest, not a complete inference environment lock.

These questions do not reopen the bounded search. Any new study needs a separate
protocol and user authorization, not further tuning against these final samples.

## 5. Artifact and command map

All paths below are local to this repository. Raw results/checkpoints and the PDF
are git-ignored but present on disk; preserve them with the code for handoff.

| Evidence | Location |
|---|---|
| Frozen inputs/model/data/software/hardware | `results/core-v1/frozen/` |
| Router split/raw articles/token IDs/rejections/seal | `results/core-v1/router-frozen/`, `configs/router_data_lock.json` |
| Baseline raw six runs | `results/core-v1/baseline-{fp16,fixed8,fixed4}-r{1,2}/` |
| Single model and original integration evidence | `results/core-v1/integration/` |
| New labels/features | `results/core-v1/router-collect{,-smoke}/` |
| A1 weights/normalizers/300-epoch losses/static policy | `results/core-v1/router-training/` |
| Actual routing/logits/causality/endpoint gate | `results/core-v1/router-verify/` |
| Matched raw results/routes/repeats | `results/core-v1/comparison-{adaptive,static,random}-r{1,2}/` |
| Aggregate metrics/paired CIs/recomputed audit | `results/core-v1/comparison-gate.json` |
| Exact shell commands/preflight/exit failures | `results/core-v1/logs/` |
| Tests/review/runtime decisions/audit logs | `results/core-v1/meta/` |
| Human prompt-to-artifact completion audit | `COMPLETION_AUDIT.md` |

Verified replay/audit commands from repo root after `source ~/.venv/bin/activate`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_baselines.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/check_router_results.py
python scripts/audit_table.py
```

New GPU evidence was produced with this prefix before **every** job:

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
 bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
 python scripts/router_job.py --stage evaluate --mode adaptive \
 --out results/core-v1/comparison-adaptive-r1
```

That exact completed output directory must **not** be reused; scripts refuse
collisions. The complete preparation/collect/train/verify/repeat invocation
sequence is recorded in `ROUTER_PROTOCOL.md`, per-job `command.json`, logs, and
`doc/runbook.md`. No new GPU experiment is needed for the completed audit.

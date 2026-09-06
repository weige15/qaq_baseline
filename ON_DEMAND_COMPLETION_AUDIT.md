# On-demand study completion audit

Status: **VERIFIED COMPLETE** for the separate `qaq-on-demand-v1` objective.
This audit was performed against current files and raw artifacts, not inferred
from implementation effort, a single green test, or the existing core audit.

## Concrete deliverables and success criteria

1. Preserve the completed Qwen3-4B checkpoint, quantizer, router, frozen examples,
   saved profiles and core conclusions; add only a separate study.
2. Store real two's-complement bit planes at one bit per code and reconstruct
   exactly the existing 4/6/8 FP16 weights.
3. Compare otherwise identical `resident` and `ondemand_sync` execution. Resident
   owns all packed planes/scales on GPU; on-demand owns CPU sources, transfers
   only current selected planes/scales into one block-sized GPU slot, and releases
   ownership after each attention/FFN block.
4. Use identical unpacking, reconstruction, FP16 matmuls, checkpoint, routes,
   samples and measurement boundaries.
5. Record requested/copied bytes, requests/loads/releases, transfer time,
   synchronized wall, allocated/reserved peaks, startup, hardware, commands,
   source and raw per-run results.
6. Pass exact CPU and tiny-Qwen gates, then one full-model smoke. Run three paired
   WikiText-2 repeats only after exact output, lower on-demand peak allocation and
   a <=30-minute projection.
7. Apply the stated revise/stop rules without paper-percentage targeting and
   without async prefetch, custom kernels, dynamic batching, router retraining,
   additional models or score-driven changes.
8. Produce evidence-separated `ON_DEMAND_PROTOCOL.md` and `ON_DEMAND_REPORT.md`.

## Prompt-to-artifact checklist

| Explicit requirement or gate | Concrete evidence inspected | Finding |
|---|---|---|
| Separate audited study; do not invalidate core | New namespace `results/on-demand-v1/`; new source/scripts/config/docs only. `git diff e317236 --name-only` excludes core quantization/model/router/configs, `REPLICATION_REPORT.md`, and original `COMPLETION_AUDIT.md`. Final artifact audit rehashes protected files. | Done |
| Preserve Qwen3-4B checkpoint/revision | `configs/on_demand_protocol.json`; every CPU/GPU command hash-checks 4.525GB checkpoint `b63aee…b85`; run `command.json` input hashes; model inventory. | Exact same checkpoint and revision |
| Preserve quantization rules | Protocol records group128/int8/FP32 midpoint rule; source hashes for `qaq.quantization` and `qaq.model`; packed path calls unchanged `reconstruct`. | Done; no requantization |
| Preserve router and saved query profiles | A1 checkpoint/selection and router source are hashed; both historical route files are hashed; canonical `(task,index,profile)` hash `b83498…58d`; runner replays profiles and never invokes router training/inference. | Done |
| Preserve frozen examples and samples | Frozen examples hash `5eba98…afd`; completed adaptive samples hash `10368e…acb`; every new profile/sample is aligned by task/index/context. | Done |
| Genuinely packed two's-complement planes | `pack_twos_complement` uses eight codes per uint8 plane byte, raw signed byte bits0..7; CPU test checks all256 codes and a255-code padded tensor. Full inventory is3,633,315,840 plane bytes for3,633,315,840 int8 codes, not eight unpacked bytes/code. | Done |
| 4/6/8 reconstruction exactly equals existing result | `tests/test_on_demand.py` uses `torch.equal` against `qaq.quantization.reconstruct` for all signed codes and padded tensor at all widths; targeted and full gate logs pass. | Exact, no tolerance |
| Tiny-Qwen exactness | Same test compares existing `NestedLinear`, resident and on-demand logits at fixed4/6/8 and mixed attention/FFN profile; exception release/accounting also checked. | Passed |
| Resident all planes/scales GPU | Each resident `storage` inventory:3,746,856,960 source bytes, all on `cuda:0`, no slot; startup copies exact storage bytes. | Done |
| On-demand source CPU; selected transfer only | Each on-demand inventory: same source bytes on CPU, zero source GPU bytes,77,045,760-byte sole max-block slot. Event bytes equal selected widths plus scales; code copies rows `8-b..7` only. | Done |
| One block slot and release | 4,608 requests/loads/releases per full run; `max_active_slots=1`, end active slots0; post-hook `always_call`; injected exception test releases. | Done; allocation persists, ownership releases as preregistered |
| Same unpack/reconstruct/model path | One `PackedLinear` and `reconstruct_packed` used by both modes; accepted pairs have identical executable source manifest `cafdf0…3d8`, model inventory, protocol/input hashes, profiles and physical GPU UUID. | Done |
| Same samples/routes and exact outputs | Smoke and all six accepted full outputs have exact sample/profile/metric equality; cross-repeat equality; exact to completed core WT2 samples; full sample SHA `d154bc…847`. | Passed |
| Requested/copied bytes | Full result/event traces:181,665,792,000 requested bytes per mode/run; on-demand copied same; resident execution copied0 and startup copied3,746,856,960. Raw block records retained. | Recorded and independently summed |
| Requests, loads, releases | Full:4,608 requests each; on-demand4,608 loads/releases; resident0 execution loads/releases and4,608 closed views. Smoke equivalents72. | Recorded |
| Transfer and total synchronized wall | Accepted raw results and report table: resident transfer0/wall57.290–57.362s; on-demand transfer42.926–43.081s/wall99.428–126.847s. | Recorded per run, variation retained |
| Peak allocated/reserved memory | Every accepted pair: resident5,246,962,176/5,878,317,056; on-demand1,511,877,120/2,124,414,976 bytes. Start/end values also in each raw result. | On-demand allocated peak strictly lower by3,735,085,056 bytes |
| Startup cost | Each result contains total and phase timings plus startup H2D and startup peaks. Accepted resident258.374–259.143s; on-demand250.354–265.342s. | Recorded separately |
| Hardware/software/commands | Per run `hardware.json`, `command.json`, UUID/start time, exact package versions, CUDA, complete `nvidia-smi`, source snapshot/diff. Accepted full pairs all physical RTX3090 UUID `daa3…`. | Recorded and pair-checked |
| Raw per-run results | Every run has profiles, samples/token NLL, transfer-event JSONL, result JSON, command/hardware/source manifest/snapshot and launch log. Final artifact audit finds all required files and no `failure.txt`. | Done |
| CPU/tiny before GPU | `cpu-gate/results.json` timestamp and logs precede smoke commands; runner refuses smoke without passing same-source gate. 4 targeted and24 total tests pass. | Gate order enforced |
| One full-model smoke before repeats | `smoke-gate.json`: exact outputs/profiles/metrics, lower allocated peak, same physical GPU/source; repeat authorization true. Timestamps precede all full accepted runs. | Passed |
| Continue only if memory falls | Smoke allocated peak5,246,962,176→1,511,877,120; source inventory excludes unintended on-demand GPU payloads. | Continuation condition passed |
| Pause above30-minute projection | Smoke projected fresh64-window jobs327.374s and381.140s versus1,800s limit; every actual process wall<434s. | No pause required; no run exceeded limit |
| Three paired WT2 repeats | `repeat-gate.json`: accepted IDs1b/2b/3, six unique run UUIDs, AB/BA/AB timestamp order,64 windows each, exact cross-repeat/core outputs, same physical GPU within/all accepted pairs. | Passed |
| Pairing deviations not hidden | First candidates1 and2 are retained and marked rejected solely for different physical GPU UUIDs; append-only1b/2b replacements and reasons appear in gate/report. | Audited, not result-selected |
| If output differs, revise | No output differed at any stage. Exact core oracle checks would fail each runner before a result completed. | Revision branch not triggered |
| If memory does not fall, stop | Memory fell after inventory proved zero unintended on-demand source bytes on GPU. | Stop branch not triggered |
| No paper percentage target | Protocol frozen at commit `6817f82` before GPU results; acceptance uses exact output, absolute allocated comparison and runtime cap only. Report labels paper values context only. | Preserved |
| Exclude async/prefetch/kernels/dynamic batching/router retraining/additional models/score changes | Source inspection: blocking `copy_` plus explicit synchronize, unpinned CPU, ordinary FP16 `F.linear`, batch1, saved routes, Qwen3-4B only. Protocol/report disclose exclusions. | Preserved |
| `ON_DEMAND_PROTOCOL.md` separates source claims and our choices | Sections1–6 explicitly separate paper statements, preserved inputs, our format/modes, measurement, gates and exclusions. Machine config committed pre-run. | Done |
| `ON_DEMAND_REPORT.md` separates statements/choices/measurements/questions | Sections1–5 plus outcome, deviations and artifact/command map; all raw numbers trace to result JSONs. | Done |
| Onboarding/runbook not stale | `README.md`, `doc/onboarding.md`, `doc/runbook.md` now link both closed studies and verified commands/known limits. | Done |

## Verification commands and coverage

Actually run on the current host:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
  scripts/check_on_demand_cpu.py --out results/on-demand-v1/cpu-gate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python scripts/check_baselines.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python scripts/check_router_results.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python scripts/check_on_demand.py --stage repeats
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
  results/on-demand-v1/meta/final_artifact_audit.py
git diff --check
```

Coverage boundaries were checked rather than treating green status as whole-goal
proof:

- CPU tests prove packing/reconstruction/tiny execution, not full-model CUDA
  memory or output behavior.
- The smoke gate proves one full-model pair and continuation conditions, not
  three-repeat stability.
- The repeat verifier recomputes source/hardware/profile/sample/event accounting,
  exact pairs, memory direction, repeat identity/order and core raw-output
  equality. It does not by itself prove paper interpretation or exclusions.
- The final artifact audit rehashes inherited inputs, inspects all12 successful
  model runs, checks required artifacts/no failure files and confirms protected
  tracked core files are unchanged. It does not make a general performance claim.
- Paper interpretation, implementation choices, scope exclusions, deviations and
  unresolved limitations were therefore inspected directly in the protocol,
  source and report as mapped above.

## Qualified conclusion

Every explicit requirement has concrete passing evidence. The result is narrow:
an audited synchronous storage comparison using uncached FP16 reconstruction on
one Qwen3-4B/WikiText setup. It does not reproduce paper percentages or establish
an optimized loader. No required item remains missing or uncertain. The separate
on-demand objective is complete; the completed core remains valid and unchanged.

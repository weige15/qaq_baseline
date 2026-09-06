# Audited synchronous on-demand bit-plane loading report

## Outcome

The separate `qaq-on-demand-v1` study passed its staged gates without changing
the completed QAQ core. Genuinely packed two's-complement planes reproduce the
existing 4/6/8-bit `qaq.quantization.reconstruct` weights exactly. On the same
frozen Qwen3-4B WikiText-2 examples and saved query-dependent profiles,
`resident` and `ondemand_sync` produced exactly identical raw token losses in
the smoke and all three accepted full pairs; they also equal the completed core
adaptive WikiText raw samples.

On-demand synchronous loading reduced measured peak allocated GPU memory from
5,246,962,176 to 1,511,877,120 bytes in every accepted full pair, an absolute
reduction of 3,735,085,056 bytes (3.479 GiB). It increased synchronized 64-window
wall time from a stable mean 57.318 s to 108.630 s; its measured synchronous
transfer component averaged 43.019 s. These are measurements of this deliberately
unoptimized pageable-CPU/PyTorch implementation on one RTX 3090, **not** targets,
reproductions, or explanations of the paper's percentages.

The earlier QAQ core conclusion remains unchanged: its narrow adaptive-versus-
static result and negative random/fixed4 controls are documented only in
`REPLICATION_REPORT.md`. This systems study neither retrained the router nor
changed any score, route, sample, checkpoint, or core report.

## 1. Paper statements

The five-page QAQ paper describes bit-plane weights and an on-demand mode that
stores weight data in CPU memory and transfers selected data to GPU memory. It
attributes additional latency to sequential transfer and reports a GPU-memory
trade-off. Figure 1 distinguishes attention and FFN blocks.

The paper does not specify a signed representation, scales/zero points, packing
order, selected widths, CPU pinning, stream use, prefetching, slot/cache size,
eviction, synchronization, warm-up, exact hardware, or peak-memory boundary.
Those omissions are retained in `PAPER_AUDIT.md`; the choices below are ours.
No comparison was optimized to or accepted against a paper percentage.

## 2. Our fixed choices

### Preserved inputs

- `Qwen/Qwen3-4B` revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Existing integrated checkpoint SHA256
  `b63aeeed85e11cea5d2790b1aae068920626e5d93117548358f76ae3b2171b85`.
- Existing group-128 FP32 scales, signed int8 codes, ties-to-even quantization,
  and midpoint 4/6/8 reconstruction. The core quantization/model/router source
  hashes were checked before every run and were not edited.
- The 64 frozen 512-token WikiText-2 windows, first 128 tokens as query context
  and remaining 384 scored, in their existing order.
- Saved adaptive `(task,index,profile)` records from both completed repeats.
  Their canonical SHA256 is
  `b83498862ef98b5b057896e2b20eb07bd9d414cd11beea31e8d05c623dd8c58d`.
  The A1 router was not run or retrained.
- Completed core adaptive raw-sample oracle SHA256
  `10368ead4efbab75dbec619b51e0f79a3fde5096ac31e63e6b2023657d5e3acb`.

All inherited checkpoint, example, route, sample, integration/comparison gate,
router-selection, and router-checkpoint hashes are recorded exactly in
`configs/on_demand_protocol.json` and the CPU/run command artifacts.

### Packed representation

For flattened int8 code `q[i]`, its raw two's-complement byte is split into
planes 0 through 7. Plane 7 is the sign plane. Each plane byte stores eight
codes, with code `i` in lane `i mod 8`; each projection's tail is zero padded.
Thus eight complete planes use one byte per code, while selecting 4 or 6 planes
uses physically 4 or 6 bits per byte-aligned code plus the unchanged scales.

At precision `b`, only planes `8-b` through 7 are read. Unpacking clears omitted
low bits and sign-extends bit 7, then calls the unchanged core `reconstruct`.
The midpoint and FP32 scale multiplication therefore remain identical, followed
by the same FP16 cast. Both storage modes use this one implementation and an
uncached ordinary FP16 `F.linear`; no integer/low-bit compute kernel is claimed.

### Storage and measurement

- `resident` keeps all 3,633,315,840 packed plane bytes and 113,541,120 scale
  bytes on GPU.
- `ondemand_sync` keeps those 3,746,856,960 source bytes on ordinary unpinned CPU
  memory. One reusable 77,045,760-byte maximum-block GPU slot has eight plane
  rows plus scales. A block request synchronously copies only selected high rows
  and current scales; post-hook ownership release occurs even on exceptions.
  The allocation and stale contents can remain, but no block can use them after
  ownership clears and overlapping active blocks are rejected.
- No reconstructed FP16 weight survives its projection call. Both modes have
  the same persistent model inventory: 778,304,512 GPU parameter bytes and 512
  rotary-buffer bytes. On-demand inventory found zero source-plane/scale bytes
  on GPU outside its sole slot after startup and after evaluation.
- Fresh processes use batch one, FP16 SDPA, seed 1729, deterministic Torch and
  TF32 disabled. After startup, garbage collection/`empty_cache`, synchronization,
  peak reset and counter reset establish the common measurement boundary.
  Synchronized wall includes route lookup, unpack/reconstruct/model/scorer work,
  CPU loss copies and raw sample writes. Transfer time synchronizes immediately
  before timing and after all blocking copies. Peak allocated and reserved are
  both retained; only allocated governed continuation.

Full protocol and byte definitions: `ON_DEMAND_PROTOCOL.md` and
`configs/on_demand_protocol.json`, committed at `6817f82` before GPU results.

## 3. Measurements

### CPU and tiny-Qwen gate

`results/on-demand-v1/cpu-gate/` records a clean-tree run at `6817f82`:

- 4/4 targeted tests and 24/24 complete tests passed;
- all signed int8 codes `-128..127` at 4/6/8 bits exactly match the unchanged
  reconstruction oracle;
- a non-byte-aligned 255-code tensor also reconstructs exactly at all widths and
  has zero padding;
- resident, on-demand and existing `NestedLinear` tiny-Qwen logits are exactly
  equal for fixed4/fixed6/fixed8 and one mixed attention/FFN profile;
- requested/copied/load/release counts, one-active-slot enforcement and release
  after an injected projection exception pass.

The gate's runnable-source manifest SHA256 is
`cafdf0631dd14d151b3e2f2c445f3fb2a59ffa0fea742be9b6ea290078cbd3d8`.
Every measured model run contains and rehashes the identical runnable snapshot.

### Full-model smoke gate

Both fresh smoke jobs used the same physical RTX 3090 UUID
`01fe0496-2d96-9bad-bca2-935ad704d1c7` and the first frozen WikiText window/profile.

| Mode | Startup s | Sync wall s | Transfer s | Requests / loads / releases | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|---:|---:|
| resident | 257.510 | 1.092 | 0 | 72 / 0 / 0 | 5,246,962,176 | 5,878,317,056 |
| ondemand_sync | 253.783 | 1.990 | 0.731 | 72 / 72 / 72 | 1,511,877,120 | 2,124,414,976 |

Both requested 2,838,528,000 bytes according to the selected logical block
payloads. On-demand physically copied exactly 2,838,528,000 bytes; resident made
no execution copy and copied all 3,746,856,960 storage bytes at startup. Raw
samples, profiles and metrics are exact between modes and exact to the completed
core. The conservative full-job projections were 327.374 s resident and 381.140
s on-demand, both below the 1,800 s pause threshold, so full repeats proceeded.

### Three accepted paired WikiText-2 repeats

All accepted jobs used the same physical RTX 3090 UUID
`daa3c5f3-ae58-28e9-e065-7ec92537ccb1`, CUDA 12.1, Python 3.12.3,
Torch 2.5.1+cu121 and Transformers 5.16.1. Pair order was resident→on-demand,
on-demand→resident, resident→on-demand. Every run scored all 64 windows and
24,576 tokens. Every mode/repeat returned exactly:

- NLL sum: `73946.67118498588`
- mean NLL: `3.008897753295324`
- token perplexity: `20.265050525066016`
- mean-window NLL standard error: `0.05310809249524012`
- raw sample SHA256:
  `d154bcb42911ffcfd209cff6e9e8108e4055da400a280c1ff725b7184f117847`

| Pair | Mode | Startup s | Sync wall s | Transfer s | Peak allocated bytes | Peak reserved bytes |
|---:|---|---:|---:|---:|---:|---:|
| 1 | resident | 259.131 | 57.362 | 0 | 5,246,962,176 | 5,878,317,056 |
| 1 | ondemand_sync | 250.354 | 126.847 | 43.081 | 1,511,877,120 | 2,124,414,976 |
| 2 | ondemand_sync | 265.342 | 99.615 | 42.926 | 1,511,877,120 | 2,124,414,976 |
| 2 | resident | 258.374 | 57.304 | 0 | 5,246,962,176 | 5,878,317,056 |
| 3 | resident | 259.143 | 57.290 | 0 | 5,246,962,176 | 5,878,317,056 |
| 3 | ondemand_sync | 251.841 | 99.428 | 43.050 | 1,511,877,120 | 2,124,414,976 |

Per full run, both modes recorded 4,608 block requests and
1,395,193,282,560 requested plane bits. Including repeated scales, requested
payload was 181,665,792,000 bytes (169.189 GiB). On-demand copied exactly that
amount in 4,608 loads and recorded 4,608 releases; resident made zero execution
loads/copies/releases. All 4,608 active views closed in both modes and the end
active-slot count was zero.

Across accepted repeats (arithmetic means; sample standard deviation in
parentheses):

| Measure | resident | ondemand_sync |
|---|---:|---:|
| Startup s | 258.883 (0.441) | 255.845 (8.257) |
| Synchronized wall s | 57.318 (0.038) | 108.630 (15.777) |
| Synchronized transfer s | 0 | 43.019 (0.082) |
| Peak allocated | 5,246,962,176 | 1,511,877,120 |
| Peak reserved | 5,878,317,056 | 2,124,414,976 |

The wall-time variability in on-demand pair 1 is reported rather than removed;
transfer timing itself was stable. No throughput or causal explanation is
claimed from three repeats.

### Pairing deviations retained

Two first pairing candidates completed successfully and had exact outputs, but
the shared-server preflight selected different physical RTX 3090 devices between
the two modes as availability changed. They were rejected from paired summaries
**solely before comparing their performance values** and remain intact:

- `repeat-1-resident` (GPU UUID `01fe…`) versus
  `repeat-1-ondemand_sync` (GPU UUID `daa3…`);
- `repeat-2-ondemand_sync` (GPU UUID `3000…`) versus
  `repeat-2-resident` (GPU UUID `daa3…`).

Append-only `1b` and `2b` runs on UUID `daa3…` replaced them as matched pairs.
`repeat-gate.json` verifies all accepted source/hardware/profile/sample hashes,
six distinct UUID-tagged run IDs/start times, declared order, exact raw repeats,
and the reasons for excluding the two candidate pairings. No failed quality or
memory result was discarded.

## 4. Failures, limits, and unresolved questions

- No loader output mismatch occurred at CPU, tiny-Qwen, smoke, or full scale; no
  loader revision or equality tolerance was needed.
- On-demand peak allocated memory fell and inventory found no unintended
  GPU-resident source payloads, so the protocol's negative stop rule did not fire.
- Physical-GPU availability changed between two initial pair candidates. The
  guard was never bypassed; no other process was signaled, reset, killed, or
  shared. All twelve measured model processes had fresh preflight logs and stayed
  below 30 minutes. The rejected candidates are provenance evidence, not repeats.
- The slot is a persistent maximum-block allocation whose **ownership**, not
  allocation, is released after each block. Reserved memory need not fall after
  release because PyTorch caches allocations. Both facts are reflected in the
  separate slot-capacity, active-slot, allocated and reserved fields.
- Transfer bytes count actual tensor payload, not PCIe protocol overhead,
  allocator traffic, pageable staging internals, or stale-slot zeroing. Transfer
  time uses synchronized host wall intervals, not CUDA events.
- Startup is dominated by loading the 4.5 GB core checkpoint on CPU. It is
  reported separately and is not included in synchronized evaluation wall.
- These results are for one model, one physical GPU for accepted full pairs,
  one frozen task/sample set, batch one, ordinary FP16 matmuls and a Python/
  PyTorch unpacker. They do not establish production latency, energy, throughput,
  multi-GPU behavior, or generality.
- No asynchronous prefetch, pinned memory, custom low-bit kernel, dynamic
  batching, router retraining, new profile, additional model, PTB/MC evaluation,
  quantizer change, or score-driven choice was attempted.
- The post-run verifier was adjusted only to select append-only hardware-matched
  `1b`/`2b` directories and to record the rejected candidates. The loader,
  runner, measured source snapshot and all raw run files remained unchanged.

## 5. Commands and artifact map

CPU prerequisite:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
  scripts/check_on_demand_cpu.py --out results/on-demand-v1/cpu-gate
```

Every GPU process used this exact guarded shape with a unique output path:

```bash
source ~/.venv/bin/activate
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
  bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
  python scripts/run_on_demand.py --mode MODE [--smoke] --out OUTPUT
```

Final audit:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
  scripts/check_on_demand.py --stage repeats
```

| Evidence | Path |
|---|---|
| Frozen protocol | `ON_DEMAND_PROTOCOL.md`, `configs/on_demand_protocol.json` |
| Packed runtime | `src/qaq/on_demand.py` |
| Guarded runner / verifier | `scripts/run_on_demand.py`, `scripts/check_on_demand.py` |
| CPU/tiny gate and logs | `results/on-demand-v1/cpu-gate/` |
| Smoke raw pair and authorization | `results/on-demand-v1/smoke-{resident,ondemand_sync}/`, `smoke-gate.json` |
| Accepted full raw pairs | `repeat-{1b,2b,3}-{resident,ondemand_sync}/` |
| Rejected pairing candidates | `repeat-{1,2}-{resident,ondemand_sync}/` |
| Aggregate audited result | `results/on-demand-v1/repeat-gate.json` |
| Every preflight/launch and audit log | `results/on-demand-v1/meta/` |
| Per-run provenance | each run's `command.json`, `hardware.json`, `source_manifest.json`, `source/`, `git-diff.patch` |
| Per-run raw evidence | each run's `samples.jsonl`, `profiles.json`, `transfer_events.jsonl`, `results.json` |

The entire `results/on-demand-v1/` tree is git-ignored like the preserved core
bundle and must be retained with it; a code-only clone is not the evidence.

# Synchronous on-demand bit-plane study protocol

Status: **frozen before GPU results**. Protocol ID: `qaq-on-demand-v1`.
Machine-readable settings: `configs/on_demand_protocol.json`.

This is a separate systems study. It neither changes nor reinterprets the
completed QAQ core result in `REPLICATION_REPORT.md` and `COMPLETION_AUDIT.md`.
All existing `results/core-v1/` artifacts are read-only inputs.

## 1. Paper statements

The five-page QAQ paper says that quantized weights are decomposed into bit
planes, selected planes can be transferred from CPU to GPU on demand, and the
reported synchronous transfer approach trades higher latency for lower GPU
memory. Figure 1 distinguishes attention and feed-forward blocks. The paper does
not specify its signed format, scale/zero point, packing order, CPU pinning,
slot/cache capacity, eviction, streams/prefetch, warm-up, synchronization, or
peak-memory boundary. Its percentages are context only and are not targets.
See `PAPER_AUDIT.md` for the source audit.

## 2. Preserved core inputs

Every run must hash-check and only read:

- model/checkpoint: `Qwen/Qwen3-4B` revision
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- integrated checkpoint:
  `results/core-v1/integration/quantized_model.pt`, SHA256 `b63aee…b85`;
- quantization: unchanged group-128 signed int8 codes, FP32 scales, midpoint
  reconstruction at 4/6/8, and final FP16 cast from
  `src/qaq/quantization.py`;
- all frozen examples and ordering from
  `results/core-v1/frozen/examples.jsonl`, SHA256 `5eba98…afd`;
- saved adaptive profiles from both completed repeats. Their timing-bearing
  files have different hashes, but `(task,index,profile)` is exactly equal and
  has canonical SHA256 `b83498…58d`;
- completed adaptive raw samples, SHA256 `10368e…acb`, as an exact output oracle;
- integration/comparison gates and the A1 selection/checkpoint, all protected by
  hashes in the machine-readable protocol. The router is **not executed or
  retrained** here; saved profiles are replayed unchanged.

The new code is additive in `src/qaq/on_demand.py`; it does not modify the core
checkpoint format, quantizer, router, scorer, source modules, routes, samples,
or reports.

## 3. Our packed format and storage choices

### Two's-complement planes

For flattened int8 code `q[i]`, let `u[i] = q[i] mod 256`. Plane `k` stores bit
`k` of each `u[i]`; planes are ordered 0 through 7, so plane 7 is the sign bit.
Within a packed byte, code `i` uses lane `i mod 8` (little bit order). Each
projection is padded with zero lanes to a whole byte, then projection segments
are concatenated inside its attention or FFN block.

Precision `b` transfers/reads only planes `8-b` through 7. Unpacking clears the
omitted low bits and sign-extends plane 7. It then calls the unchanged
`qaq.quantization.reconstruct(q, scale, b, torch.float16)`. Thus its arithmetic
right-shift term and midpoint are identical to the completed core method. Tests
must use `torch.equal`, not tolerance. The historical NumPy sign-magnitude toy
in `qaq.bitplanes` is not used.

### Otherwise identical modes

- `resident`: all eight packed planes and unchanged scales for all 72 blocks are
  kept on GPU. It performs no execution-time H2D weight load.
- `ondemand_sync`: all source planes/scales remain in ordinary **unpinned** CPU
  memory. It allocates one reusable maximum-block GPU slot: eight uint8 plane
  rows plus one FP32 scale vector. Before each attention/FFN block, blocking
  `copy_` operations fill only the selected high rows and current scales. CUDA
  is synchronized immediately before timing and after all copies. A post-hook
  with `always_call=True` clears slot ownership after the block, including when
  execution raises. Stale bytes can remain in the sole allocation but are not
  active or addressable by later projections until the next load.

Both modes use the same block ordering, packed layout, selected-plane unpacker,
core FP32-scale/FP16 reconstruction, uncached `F.linear`, model, SDPA path,
profile, samples, batch size one, and measurement boundary. No reconstructed
weight survives its projection call. This deliberately measures a correctness
reference, not an optimized low-bit kernel.

## 4. Accounting and measurement boundary

One **request** is one selected attention or FFN block payload. Requested bytes
are `ceil(sum(projection elements) * b / 8)` plus scale bytes. Copied bytes are
the actual selected packed projection segments (including any per-projection
byte padding) plus scales. Qwen's projection sizes are byte-aligned, but padded
CPU tests make the distinction auditable. `ondemand_sync` records one logical
load and one ownership release per request; resident records zero execution H2D
loads/releases and separately records its startup H2D bytes.

Each mode/repeat runs in a fresh process. Startup time is runner `main()` entry
(after interpreter/import overhead) through integrity checks, CPU
checkpoint load, packing/removal of original q/scales,
GPU move of only the FP16 remainder, resident placement or slot allocation, and
final synchronization. Phase times are also retained.

After startup: collect garbage, empty only this process's unused CUDA cache,
synchronize, reset peak stats and loader counters, then capture start allocated
and reserved bytes. Synchronized evaluation wall time begins immediately before
the frozen route/scorer loop and ends after a final CUDA synchronization. It
includes route lookup, model work, synchronous transfer, CPU loss copies, and
raw sample writes in both modes. Record start/end/peak allocated and reserved
memory, source/slot inventory, startup and process wall, transfer trace and
aggregate, hardware, versions, command, Git state, source snapshot, metrics, and
raw token losses.

Reserved memory may remain high because of PyTorch's caching allocator. The
continuation gate uses **peak allocated** memory and also requires zero
on-demand source-plane/scale bytes on GPU outside the one slot.

## 5. Staged gates and stop rules

1. **Exact CPU + tiny Qwen first**
   ```bash
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
     scripts/check_on_demand_cpu.py --out results/on-demand-v1/cpu-gate
   ```
   This gate runs both the targeted and full test commands, saves their complete
   logs plus a runnable-source manifest, and hash-checks every inherited oracle.
   A smoke runner refuses a missing/failed gate or changed source manifest.
   All 256 signed codes, padding, 4/6/8 reconstructions, invalid inputs,
   fixed/mixed tiny-Qwen outputs, request/copy/load/release counts, one-slot
   invariant, and exception release must pass.

2. **One full-Qwen smoke pair** on the first frozen WikiText-2 window/profile:
   ```bash
   PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false \
     bash scripts/gpu_preflight.sh --run timeout --signal=TERM 30m \
     ~/.venv/bin/python scripts/run_on_demand.py --mode MODE --smoke \
       --out results/on-demand-v1/smoke-MODE
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ~/.venv/bin/python \
     scripts/check_on_demand.py --stage smoke
   ```
   Continue only if raw token losses/profiles/metrics are exactly equal and
   on-demand peak allocated memory is strictly lower. Every run snapshots and
   hashes its executable source; the pair checker requires identical source,
   runtime and exact GPU identity. If outputs differ, revise the loader and
   repeat under new append-only directories by passing `--resident`, `--ondemand`,
   and `--gate-out` to the checker, then pass the selected CPU/smoke gate paths
   to later runners. If memory
   does not fall after inventory confirms no unintended GPU-resident source
   tensors, stop with the negative evidence.

3. **Runtime projection gate.** From each smoke's startup and synchronized
   one-window evaluation time, project a 64-window job conservatively as
   `startup + 64 * smoke evaluation`. Pause before any projected job above 30
   minutes.

4. **Three full paired WikiText-2 repeats**, only after both gates pass. Fixed
   order is resident→on-demand, on-demand→resident, resident→on-demand for pairs
   1–3. Each run uses all 64 frozen windows and a fresh preflight-selected single
   GPU. `scripts/check_on_demand.py --stage repeats` must require exact raw
   equality within every pair and retain every result. No percentage target or
   post-hoc score change is allowed.

Every GPU command must use `scripts/gpu_preflight.sh`; never kill, reset, signal,
or share another user's process. Output directories are append-only. Commands
carry UUIDs and start timestamps; the final verifier requires six distinct runs
and the declared resident→on-demand, on-demand→resident, resident→on-demand
pair order. Any failed run keeps `failure.txt` and its shell log.

## 6. Explicit exclusions and unresolved pre-run questions

Excluded: asynchronous prefetch, pinned-memory optimization, custom low-bit
kernels, dynamic batching, router retraining, new routes, additional models,
PTB/MC tasks, quantizer changes, score-driven changes, and matching paper
percentages.

Pre-run unknowns include actual pageable-memory transfer behavior, allocator
reserved-memory retention, and whether this unoptimized PyTorch unpacking stays
under 30 minutes for 64 windows. They will be measured, not silently assumed.

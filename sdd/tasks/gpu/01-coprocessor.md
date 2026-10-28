01 - GPU Co-processor

Status: done

**Closed.** The co-processor landed: the model, the transfer-aware offload, the per-dispatch `device`/`tile`, the VRAM budget and the CPU/GPU parity. The spillover is closed (`gpu/05`, the PCIe ROI is negative); the heterogeneous split is a **capability**, not a win here (`gpu/11`); the decode lever is the VRAM resident set (`gpu/06`).
Commits: -

## Goal

Define the GPU model: the GPU is a **co-processor** that extends the CPU's
processing unit, not a place to host the model. The CPU orchestrates the forward
and runs the ops the GPU should not; the GPU accelerates **specific ops** by
streaming weight tiles, so a **low-VRAM GPU still helps**. `--vram` is the
delegated budget of how much GPU the runtime may use. The plan (AOT) declares
where each op runs; the runtime re-decides nothing.

## Context — the earlier local prototype (its `src/gpu/`)

The earlier local prototype already proved this direction; carry the ideas, re-derive for zlm:

- **Per-op offload policy** (`policy.shouldOffload`): a supported quant type and
  a **payload threshold** decide offload, differing by device class:
  - **UMA** (integrated / Apple / unified RAM): no PCIe penalty → offload when
    `rows >= 32 and k >= 32` (mat_vec) / `tokens >= 2 and rows,k >= 32` (mat_mul).
  - **Discrete (PCIe)**: transfer not free → offload only when the work payload
    `tokens*rows*k` clears a threshold (`4M` for mat_vec, `1M` for mat_mul).
- **VRAM as a budgeted region set** (`VramMemoryMap` + `knapsackPackVramRoi`): a
  **ROI knapsack** (`time_saved / size`) packs regions of types
  `resident_weight | activation_scratch | kv_cache | packed_panel | virtual_vram`
  into `--vram` minus a **driver headroom** (anti-OOM). Overflow becomes
  **`virtual_vram` (DRAM spillover)** — the model never has to fit VRAM.
- **Chunked weight slicing** (`chunk_rows` / `dispatchMatVecRows`): when a weight
  exceeds the VRAM slot, it is sliced by output rows and dispatched in chunks.
- **AOT execution plan** (`ExecutionPlan`): per-layer DAGs of nodes
  `{ op, device, threads, tile_rows, micro_kernel, deps }` — device per op, plus
  `tile_rows`, enabling **dual-stream** (concurrent CPU SIMD + GPU) with
  dependency-driven overlap. Serialized in the artifact.
- **VRAM KV cache** (`VramKvCache`): the KV can be resident in VRAM when it fits
  the budget, else it stays in RAM.
- **Fallback**: any op not offloaded (or failing) runs on the CPU; metrics track
  `fallback_calls`, `bytes_staged`, `resident_hits/misses`, `cpu_gpu_overlap_ns`.

## Scope

- **In:** the model (co-processor, per-op offload, streaming/slicing, DRAM
  spillover, CPU fallback); the **transfer-aware cost**; the plan fields (per-op
  `device` + `tile`); the **VRAM budget** inside `planBudget`; the `Device`
  interface (`kernels/11`).
- **Out:** implementing the Vulkan/Metal backend (CPU-first; this is the design +
  the seam); GPU for the CPU-only ops; multi-GPU; network.

## Design

### The model

The CPU is the maestro: it runs the whole forward, executing on the GPU only the
dispatches the tuner marked (and only when profitable). Weights stay in RAM/mmap
and are **streamed/sliced** to the GPU per dispatch; the GPU's VRAM is a
**budgeted scratch/resident region**, never a mandatory weight store.

### Cost / ROI — the GPU tier on the `runtime/12` pipeline

`runtime/12` (the three stages + threshold-fitted stumps + the Scheduler) is the
foundation. The GPU adds a **device dimension** and a **transfer term**, and the
Scheduler's "pin/stream" grows into the **VRAM ROI**. This is the flow `gpu/01`
inaugurates; `runtime/12` itself is unchanged.

**Candidate space** (the Evaluator gains `device` and `tile`):
```text
candidates = { variant } × { parts } × { device: cpu|gpu } × { tile } × { prefill } × { window }
```
`argmin planScore` picks, per op, the device (and the tile) — deterministic.

**Step 1 — `staticCost(op, shape, variant, device, tile)`.** The logical work is
the same (`{ bytes, flops, complexity, deps }`) plus, for a GPU candidate, the
**staging bytes**: `staging = weight_tile + activation + result` (the earlier local prototype's
`bytes_staged`). `tile` (the earlier local prototype's `chunk_rows`) slices the weight when it exceeds the
VRAM slot — fewer resident bytes for more transfers.

**Step 2 — `roofline(..., device)`.** For a GPU candidate the ceilings are the
**GPU's** (`peak_flops_gpu`, `vram_bw`) plus the **transfer term**
`transfer_ns = staging_bytes / bw_host_device`, where `bw_host_device` is the
PCIe bandwidth (discrete) or treated as unbounded (UMA). The regime label uses
the GPU ceilings. This makes the trade-off explicit:
- **Prefill / batched** (weight read once per block of M tokens): the transfer is
  amortized → a GPU candidate;
- **Decode** (weight read once/token): `transfer ≈ a RAM read` and PCIe < RAM →
  not worth it **unless** the tile is **VRAM-resident** (see the Scheduler);
- **UMA**: no transfer → offload more.

**Step 3 — `scaling(..., device)`.** The GPU's scaling curve (SM occupancy /
device "parts"), calibrated per device, same shape as the CPU's `effective_ns`.

**Evaluator / `planScore`.** The score gains the transfer + sync terms
(`+ staging_bytes / bw_host_device + sync_ns`) and is scored across
`device × tile`; `gpu_ns + transfer_ns + sync_ns < cpu_ns` falls out of `argmin`.
No GPU candidate wins ⇒ the CPU kernel runs (fallback).

**Decision stump — the offload boundary.** Where the analytical cost is flat at
the device switch, a **stump** on `(op, work_payload = tokens·rows·k)` decides
offload; its thresholds are **fit by Threshold Search** over the calibration data
(the earlier local prototype's UMA/PCIe thresholds generalized: ~4M for matvec, ~1M for matmul, plus the
UMA free-transfer case), cached in the `.zat`. Analytical owns the continuous
cost; a small, auditable stump owns the boundary.

**Scheduler — VRAM ROI (the pin/stream boundary, generalized).** Residency grows
from one RAM window to a **VRAM budget**:
- a driver **headroom** reserved (anti-OOM) — RAM's `reserve`, for VRAM;
- a **ROI knapsack** over candidate regions
  (`resident_weight | activation_scratch | kv_cache`), ranked by
  `time_saved / size`, packed into `--vram − headroom`;
- overflow is **`virtual_vram`** (DRAM spillover) — the model never has to fit
  VRAM;
- the pinned set is recorded in the plan (`vram_map`), like the RAM residency
  `nodes`; a small `--vram` degrades the **GPU**, never the weights.

**Determinism / storage.** `applicable` + cost + stumps are pure functions of the
calibrated `caps`/`.zat`; the Evaluator is bounded and deterministic. The plan
stores **decisions** (`device`, `tile`, `vram_map`), never costs — the runtime
re-decides nothing (`runtime/12`).

### Plan fields

`Dispatch` already has `device`; add a **tile** (rows per slice) and keep the
`deps` DAG so the Executor can overlap CPU and GPU work (dual-stream). The plan
records the decisions; the runtime only applies them.

### `Device` interface

Per `kernels/11`: `alloc/free/upload/download/dispatch/sync` over tiles. The
`device/` runtime owns the context/VRAM/queues; the GPU AtomicOps
(`kernels/backends/gpu/`) use the interface. The Executor stays sequential per
dispatch and unchanged.

### Fallback

Within a device: the generic CPU kernel is always applicable (`kernels/11`).
Across devices: **plan-level first** (a plan targets one device; if the GPU lacks
an op, compile for CPU), **per-op later** (mixed dispatch + transfers). A dispatch
that the GPU rejects at run falls back to CPU (the earlier local prototype's `fallback_calls`).

## Flow / Arch

```text
tune:  per dispatch: cpu_ns vs (transfer + gpu_ns)  -> plan.device / tile
       --vram -> headroom + ROI knapsack (weights/scratch/kv) + DRAM spillover
run:   Session owns the Device (context/VRAM/queues)
       Executor: for each dispatch -> CPU kernel or GPU kernel (stream tile)
                 deps allow CPU/GPU overlap (dual-stream)
```

## Steps

1. Define the `Device` interface + the plan fields (`Dispatch.device` exists; add
   `tile`). [DONE] The `src/device/root.zig` `Device` API exists
   (`init/deinit/available/matVec/sync`, backend `.none` fallback); the plan
   gained `Dispatch.tile` (format v15), serialized/deserialized and covered by
   the round-trip test.
2. Extend the cost model with the transfer term (PCIe/UMA from `caps`) and make
   the tuner set `device`/`tile` per dispatch. [DONE] `tuner/gpu.zig:decideOffload`
   is a pure transfer-aware decision (`length_bytes / cpu_bw` vs
   `(resident ? 0 : length_bytes / PCIE_BW_GBPS) + compute + sync`); it replaces
   the fixed `rows*k >= 4M` thresholds and sets `Dispatch.device` for decode
   GEMV at `tune --gpu`. `tileRows` slices a streamed weight past
   `STAGING_LIMIT_BYTES` (128 MB); the executor honors `tile` via a per-slice
   streamed GEMV (`matVecTiled`).
3. VRAM in `planBudget`: headroom + ROI knapsack + spillover + slicing. [TODO]
4. `device/` runtime: context, VRAM allocator, queues, tile upload/download,
   sync; GPU AtomicOps in `kernels/backends/gpu/`. [PARTIAL] A working Vulkan
   backend exists: `src/device/{types,context,pipeline,shaders}.zig` +
   embedded SPIR-V; dynamic `libvulkan` load; streamed-weight GEMV for
   `q4_0/q4_1/q8_0/q6_k`; guarded GPU parity tests vs the current code CPU kernel
   (run on the RX 580). VRAM allocator/queues beyond scratch are TODO.
5. Executor: dispatch to the device per `Dispatch.device`; CPU fallback; optional
   dual-stream overlap via `deps`. [PARTIAL] The `Session` owns an optional
   `Device` (heap-owned, from `plan.config.gpu`); `executor.runMatvec` streams a
   GEMV to the GPU when `device != 0` and falls through to the CPU kernel when
   the device rejects it; `executor.prefillMatmul` runs the batched GEMM on the
   GPU via `Device.matMulBatch` and falls through to the CPU `matVecBatch` on any
   rejection. Dual-stream / `deps` overlap still TODO (the per-dispatch
   `vkQueueWaitIdle` remains; see the `device/pipeline.zig` TODO).
6. Parity: GPU kernel == CPU kernel (the earlier local prototype already had the shader oracle tests);
   e2e coherent; deterministic plan. [PARTIAL] Per-op GPU-vs-CPU parity for the
   four matvec quants and the three batched-GEMM quants (`q4_0/q4_1/q8_0`);
   e2e is coherent and the plan is deterministic (re-tune reuses the identical
   plan).

## Definition of Done

- [ ] The co-processor model, the transfer-aware offload, the VRAM budget and the
      `Device` interface are documented (this spec + `kernels/11`).
- [ ] The plan can record per-dispatch `device` + `tile`; the tuner sets them from
      the cost model (deterministic).
- [ ] `--vram` is accounted in `planBudget` (headroom + knapsack + spillover);
      the GPU degrades before the weights.
- [ ] A dispatch the GPU cannot run (or does not win) runs on the CPU.
- [ ] GPU numerics match the CPU kernel; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the co-processor framing for zlm; the mapping onto the AOT plan,
  `planBudget` and the `Device` interface.
- **consulted:** the earlier local prototype's `src/gpu/` (the PoC: `shouldOffload`, the VRAM
  ROI knapsack, `chunk_rows`, `ExecutionPlan`, `VramKvCache`) — reference; carry
  the ideas, re-derive.
- **spec:** Vulkan/Metal (compute pipelines); PCIe/UMA transfer models.
- **derived:** `kernels/11` (backend/device structure), `runtime/06` (budget),
  `runtime/12` (cost pipeline), `runtime/11` (performance).

## Comments

- **Not a llama.cpp-style offload.** The GPU is an extension of the CPU; the
  model need not fit VRAM (streaming + slicing + DRAM spillover).
- `--vram` = how much GPU the runtime may use to extend the CPU; it is a budget
  in `planBudget`, with a driver headroom (anti-OOM).
- The earlier local prototype proved the shape (per-op offload + VRAM ROI + slicing + dual-stream);
  zlm re-derives it as AOT plan decisions.
- Out of active scope (CPU-first); this locks the model and the seam so a GPU
  backend (`device/` + `kernels/backends/gpu/`) plugs in without touching the
  Executor.
- **As-built (backend slice).** Ported the earlier local prototype's Vulkan PoC into `src/device/`:
  dynamic `libvulkan` via `dlopen` (needs `link_libc`), embedded `.spv`,
  streamed-weight GEMV (`q4_0/q4_1/q8_0/q6_k`), and a guarded parity test vs
  `kernels/linear/matvec/root.zig` (runs on the AMD RX 580 / RADV). The earlier local prototype's
  q6_k shader was rewritten to match the current code's ggml `dequantQ6_K`; the earlier local prototype's
  `last_weight_ptr` upload cache was dropped (stale-weight bug). The CPU path
  and plan are untouched; the GPU is not yet wired into the tuner/executor, so
  it is inert by default (`.none` backend). `device/` imports only `utils/` and
  `kernels/` (the parity test).
- **As-built (decode-offload slice).** `tune --gpu` resolves a `Device` to learn
  the class (UMA/discrete) and marks each decode GEMV whose shape+quant the
  device supports (`tuner/gpu.zig`); the decision is the existing
  `Dispatch.device` (no plan-format change). `Session` brings the device up from
  `plan.config.gpu`; `executor.runMatvec` streams the weight and runs the GEMV
  on the GPU, falling through to the CPU kernel on any rejection. Verified on
  the RX 580: `tune --gpu` offloads 145/507 decode dispatches for qwen3-4B,
  `run` is coherent, and `gpu_busy_percent` shows the GPU is exercised.
  The KV arithmetic fingerprint now includes `Dispatch.device`, so a GPU plan
  never shares a `.zpc` with a CPU plan. Still TODO: the transfer-aware cost,
  `tile`, the VRAM budget/ROI, batched (prefill) GPU kernels, and dual-stream.
- **As-built (VRAM ROI / residency slice).** The plan gained `vram_resident`
  (tensor indices) + `vram_resident_bytes` (format v14). At `tune --gpu` the
  tuner builds a candidate set from every decode GEMV whose quant+shape the
  device supports and packs it with `tuner/gpu.zig:selectResident` — a greedy
  ROI knapsack ranked by per-token transfer saved (`length` desc, tensor-index
  tie-break) into `vram_bytes -| VRAM_DRIVER_HEADROOM` (64 MB). A budget too
  small for the headroom yields an empty set: the GPU degrades, the weights
  never do. `decideOffloadMatVec(is_uma, quant, rows, k, resident)` offloads on
  discrete when resident or `rows*k >= 4M`. `Session` uploads each resident
  tensor once into a `device.Resident` buffer and frees it on deinit;
  `executor.runMatvec` reads resident weights via `matVecResident`, else streams
  `matVec`, else falls through to the CPU. Verified on the RX 580: qwen3-4B at
  `--vram 4gb` offloads 217/507 decode dispatches with 1935 MB resident,
  `run` is coherent, and `gpu_busy_percent` reaches 100%. The CPU plan is
  unchanged and `tune` is idempotent (`reuse`). The synchronous per-dispatch
  queue wait still dominates; streaming and residency are comparable in wall
  time until dual-stream/async lands.
- **As-built (transfer-aware cost + `tile` slice).** The fixed `rows*k >= 4M`
  offload threshold is replaced by `tuner/gpu.zig:decideOffload`, a pure
  function of `(is_uma, quant, rows, k, length_bytes, resident, caps)`: UMA
  keeps the shape gate (no transfer); discrete compares the DRAM-bound CPU read
  `length_bytes / caps.dram_bw_gbps` against
  `(resident ? 0 : length_bytes / PCIE_BW_GBPS) + 2·rows·k / GPU_TFLOPS +
  caps.dispatch_latency_ns`. Because a quantized GEMV is DRAM-bound on the CPU,
  a streamed weight reads the same bytes over a slower bus and loses; only a
  resident weight (or a compute-heavy op) wins — no separate compute exception
  was needed. `Dispatch.tile` (plan format v15, `u16`, 0 = whole weight) records
  a streamed weight's row slice when it exceeds `STAGING_LIMIT_BYTES` (128 MB,
  mirroring the backend's streamed weight buffer); resident weights are never
  sliced. `executor.runMatvec` honors it via `matVecTiled`, staging each slice
  at its weight offset. The KV arithmetic fingerprint is unchanged (`tile` does
  not alter per-row arithmetic). Verified on the RX 580: qwen3-4B at `--vram
  4gb` offloads 217/507 decode dispatches with 1935 MB resident and `tile = 0`
  everywhere (no matvec weight reaches 128 MB); `run` is coherent,
  `gpu_busy_percent` reaches 100%, the plan is byte-identical across re-tunes,
  and a CPU plan (no `--gpu`) is untouched. Still TODO: the VRAM budget/ROI in
  `planBudget` and dual-stream (see the batched-prefill as-built below).
- **As-built (batched prefill + cold-start slice).** Ported the earlier local prototype's batched GEMM
  shaders (`matmul_q4_0/q4_1/q8_0.comp` + `.spv`, `glslc -O`) into
  `device/shaders/`; no `q6_k` shader, so that quant stays on the CPU. The
  `Device` API gained `matMulBatch(quant, weight, rows, k, m, x, out)` (over
  `VulkanPipeline.dispatchMatMul`, with a resident-weight twin
  `dispatchMatMulResident`), and a guarded GPU-vs-CPU parity test runs it on the
  RX 580 for the three quants. `tuner/gpu.zig:decideOffloadMatMul` is the pure
  AOT decision: eligible quant (`q4_0/q4_1/q8_0`) and (resident OR `m >= 2` and
  `rows,k >= 32`); `tune.selectPrefillDispatch` sets `device = 1, parts = 1` for
  each qualifying prefill `matmul` (`tile` stays 0). `prefillLmHead` (a `.matvec`)
  is never offloaded. `executor.prefillMatmul` runs the GPU batch and falls
  through to `matVecBatch` on any rejection. The daemon cold-start deadline is
  now `spawn_timeout_ms + vram_resident_bytes / cold_upload_gbps` (0.25 GB/s, a
  deliberately conservative host-visible staging rate); `residentUploadBytes`
  reads the plan so a CPU plan keeps the fixed 15 s deadline. Verified on the RX
  580: qwen3-4B at `--vram 4gb` offloads 252/252 prefill `matmul` dispatches
  (0 `matvec`, i.e. the lm_head stays CPU) with `tile = 0`; the first cold `run`
  succeeds (no `ServiceTimeout`) and is coherent; prefill is 9.7 tok/s vs 2.9
  tok/s on the CPU plan (62-token prompt) while decode is ~0.2 tok/s either way;
  `gpu_busy_percent` reaches 100%; the CPU plan is all `device = 0`; and re-tune
  reuses the identical plan. **Dual-stream was deliberately deferred:** the
  per-dispatch `submit + vkQueueWaitIdle` is documented in `device/pipeline.zig`
  and `runtime/executor.zig`; overlapping needs device-owned activation buffers,
  fence/timeline sync, and a dispatch deps DAG, not the sequential host-buffer
  executor.
- **Implementation log.** The delivered slices and the remaining work are tracked
  in [`02 - Co-processor Implementation`](02-co-processor-implementation.md);
  the as-built comments above are the per-slice detail.

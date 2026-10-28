03 - Async Dispatch and Dual-Stream

Status: done

**Closed (bounded).** The async landed: no per-dispatch `vkQueueWaitIdle` (a fence ring), `matVecResidentAsync`/`Handle`, the executor's `Inflight` tracker and the `dep`-driven overlap. The GPU decode beats the CPU (E2B ~24 vs ~15; 12B 8.7 vs 3.6 tok/s at VRAM 8 GB).
Commits:

## Goal

Remove the per-dispatch synchronous wait that makes the GPU slower than the CPU
on decode. Today every GPU op does `submit + vkQueueWaitIdle` (~200+ driver
round-trips per token). Submit asynchronously and synchronize only when a CPU op
actually needs a GPU result, and overlap independent CPU and GPU dispatches
(dual-stream) using the plan's dependency edges.

## Scope

- **In:** async submission in `device/pipeline.zig` (a command-buffer ring plus
  fence/timeline semaphores); device-owned activation buffers for GPU chains; a
  `dep`-driven executor that overlaps CPU and GPU work; the tuner emitting
  explicit deps.
- **Out:** splitting one op across devices (`gpu/04`); VRAM spillover (`gpu/05`);
  multi-GPU; a GPU compute graph beyond GEMV/GEMM.

## Design

### Async submission

- `device/pipeline.zig`: a small ring of command buffers; `dispatchMatVec` /
  `dispatchMatMul` (resident and streamed) **record and submit** without waiting,
  returning a fence handle. `Device.sync` (and a per-handle wait) waits on
  fences; the per-dispatch `vkQueueWaitIdle` is removed (the `TODO(dual-stream)`
  markers).
- Weights: keep the resident path (upload once); for streamed weights, stage on a
  host-visible buffer and record a copy plus the dispatch without a blocking
  wait.

### Device-owned activations

- For a chain of consecutive GPU dispatches, keep the activation in device-local
  memory and download only when a CPU dispatch consumes it (a boundary). This
  removes the per-dispatch upload/download.

### Deps DAG plus dual-stream

- `Dispatch.dep` exists; the tuner's scheduler must populate it (today the
  sequence is flat, deps implicit in-to-out). With deps, the executor can run
  independent dispatches in any order: submit GPU work and, while it runs, run
  CPU dispatches whose inputs are ready (e.g. q/k/v projections vs a CPU norm).
  Wait only at a dependency boundary.
- The Executor stays sequential per **dispatch**; overlap is between a GPU op in
  flight and the next CPU op.

### Plan / AOT

- The tuner decides the device per dispatch (already) and now the deps; the
  runtime re-decides nothing. No new CLI surface.

## Steps

1. `device/pipeline.zig`: fence-based async submit plus `sync`; keep the parity
   tests green.
2. `device/root.zig`: async `matVec`/`matMul` variants returning a handle;
   `sync`.
3. `session.zig`: in-flight tracking; wait only at boundaries; device-resident
   activation buffers. **Done (bounded).** The executor now owns an `Inflight`
   tracker (`executor.zig`): a resident GEMV is submitted async and its result
   stays in the device slot until a consumer drains it. Consecutive GPU GEMVs
   sharing one input keep that activation **device-owned** (staged once into a
   device-local `burst_x`, `pipeline.zig:submitResidentGemvShared`); the run's
   later members bind the same buffer.
4. `plan`/`tuner`: populate `Dispatch.dep` (scheduler); `executor.zig`: honor
   deps and overlap CPU/GPU. **Done (bounded).** `assignDeps` already populated
   `dep`; the executor now uses the output buffer binding to find the producer
   (`drainProducer`) and waits only at the point of first use, replacing the
   flat `runMatvecBatch` batch-and-wait. The tuner is unchanged.
5. Measure decode GPU vs CPU on E2B/qwen3; parity plus `zig build test`.
   **Measured; verdict below (see Comments).**

## As-built

- **1 (done).** `pipeline.zig` no longer calls `vkQueueWaitIdle`. The streamed
  path submits with a `VkFence` (`submitShared`); the shared staging/readback
  buffers are now persistently mapped (`ensureBuffer(..., mapped_ptr, ...)`), so
  the per-dispatch `vkMapMemory`/`vkUnmapMemory` is gone. A resident
  command-buffer **ring** (`RingSize = 8`) owns one command buffer + fence +
  descriptor set + activation/result buffer pair per slot; resident GEMVs are
  recorded and submitted without waiting. `sync()` stays as the global barrier.
- **2 (done).** `device/root.zig` exposes `matVecResidentAsync` -> `Handle`,
  `waitResident(handle, out)`, and `matVecResidentBatch(k, x, ops)`. The existing
  `matVec`/`matVecResident`/`matMulBatch`/`matMulResident` remain synchronous
  wrappers over the same fence machinery, so the parity tests and the CPU
  fallback are unchanged.
- **3 (done, bounded).** The executor owns an `Inflight` tracker of submitted
  resident GEMVs (`executor.zig`). `submitResidentAsync` submits a resident GEMV
  without waiting; `drainProducer` waits + downloads only when a dispatch reads
  its output buffer (the dep boundary); `drainAll` fences at the run end, a full
  ring, or a change of shared input. A run of consecutive GPU GEMVs sharing one
  input keeps that activation device-owned: `pipeline.zig:submitResidentGemvShared`
  stages `x` once into a device-local `burst_x` (the run's first member records
  the copy + transfer->shader barrier; later members bind the same buffer). The
  device API is `Device.matVecResidentBurstAsync(res, ..., first)`.
- **4 (done, bounded).** `tune.zig:assignDeps` populates `Dispatch.dep` (format
  unchanged). The executor no longer batches-and-waits: it submits each offloaded
  resident GEMV as soon as its dispatch is reached and drains the producer at the
  first consumer (`Dispatch.dep` is implicit in the output binding; no plan
  change). Verified on the RX 580 that the async path fires 135/step on E2B
  (matching the 135 offloaded decode dispatches). The streamed path and the
  CPU-only plan are unchanged.
- **5 (measured).** `zig build` / `zig build test` green (252 tests, including
  the four-quant GPU matvec parity, the new burst/async parity test, and the
  existing resident-batch/async test, all on the RX 580). See Comments for the
  decode numbers and the verdict.

## Definition of Done

- [ ] No per-dispatch `vkQueueWaitIdle`; the GPU path is fence-based.
- [ ] The GPU decode is **not slower** than the CPU on E2B (ideally faster for
      resident weights).
- [ ] CPU/GPU overlap is exercised (deps honored); correctness (parity) holds.
- [ ] A CPU plan is unchanged; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the async/dual-stream design on the current plan.
- **consulted:** the earlier local prototype's `ExecutionPlan` dual-stream; Vulkan fences/timelines.
- **derived:** `gpu/01` (model), `gpu/02` (as-built), `runtime/02` (executor),
  `plan/01` (deps).

## Comments

- **Delivered (bounded).** Fence-based async submission + a persistent,
  persistently-mapped readback; a resident command-buffer ring; the async device
  handle/batch API; `Dispatch.dep` populated at tune; and a real batching win
  (consecutive same-input resident GPU GEMVs, e.g. q/k/v, submit back-to-back and
  wait once). The `TODO(dual-stream)` markers are filled; the streamed path and
  the parity tests stay synchronous through thin wrappers.
- **Measurement (RX 580 / RADV, gemma4-E2B, `--threads 4 --ram 4gb --gpu
  --vram 4gb`, ReleaseFast).** decode **~5.0 tok/s** on the GPU plan vs **~13.8
  tok/s** on the CPU plan (`gemma4-e2b-4t4r`) — the GPU is still ~2.8x slower.
  (An earlier 0.18/0.4 tok/s reading was a Debug build, not representative.) The
  batch reduces GPU waits (170 submits/token -> 35 batches + 35 waits) and
  `gpu_busy_percent` peaks ~97%, but the GPU GEMV cost + the weight transfer
  still dominate, so the round-trip saving is not enough on this discrete GPU.
- **DoD 2 not met on this host.** GPU decode is still slower than the CPU. The
  dominant cost is not the per-dispatch wait but (a) the HDD-bound streamed
  weights on the CPU side and (b) the offloaded GEMV's GPU cost, which the
  transfer-aware `decideOffload` (constant `GPU_TFLOPS = 3.0`, no measured
  shader throughput) over-values. What remains:
  - **Device-owned activations + a dep-driven wait.** Submit a GPU dispatch and
    do not wait until a dispatch actually consumes its output (`Dispatch.dep`),
    so GPU work overlaps the *next independent CPU op*. The current batch still
    waits before the very next norm; the executor remains sequential elsewhere.
    This is the missing dual-stream half (step 3/4).
  - ~~**GEMV shader throughput.** The per-thread full-row dot is slow on
    Polaris; a row-parallel GEMV is needed before decode offload can win on a
    discrete GPU.~~ **Done (bounded).** All four `matvec_*` shaders are now one
    workgroup per row-group: lanes split the k-blocks (adjacent lanes read
    adjacent blocks, coalesced) and reduce in shared memory; q4_0 additionally
    reads 4 bytes at once (`read_word`) and decodes with `unpackUnorm4x8`/`dot`
    (vec4), reusing the `x` vector across 4 rows. Resident weights are now
    allocated **device-local (VRAM)** and staged with `vkCmdCopyBuffer` +
    transfer->shader barrier (was host-visible, read over PCIe); the resident
    GEMV activation is also device-local (staged per dispatch). This raises the
    synthetic resident q4_0 GEMV from ~8 to **~35 (E2B shape) / ~47 (12B shape)
    GB/s** sustained, and ~17-22 GB/s in the cold min-of-N the tuner records
    (still below the CPU's ~37). **But real offloaded decode is slower** on this
    host: with the warm figure the 12B tuner offloaded 202/868 decode dispatches
    and decode fell to **1.8 tok/s vs 4.0 CPU** (the tight probe loop holds boost
    clocks interleaved decode does not, and the large resident set is not kept
    hot). So the probe deliberately stays the conservative cold min-of-N and the
    tuner offloads **0/749** (E2B) and **0/868** (12B); decode is ~CPU (E2B 14.6
    vs 15.6; 12B 4.0 vs 3.8). The GPU does **not** yet beat the CPU on this
    discrete Polaris host. q4_1/q8_0/q6_k are row-parallel but not vec4-optimized.
  - ~~**Measured offload cost.** Replace the `PCIE_BW_GBPS`/`GPU_TFLOPS`
    constants with the `.zat` device probe so `decideOffload` stops offloading
    GEMVs the GPU loses.~~ **Done.** `device/root.zig:probeGpu` measures the
    resident + streamed q4_0 GEMV and the batched q4_0 GEMM (min-of-N, model
    FFN/block shape, bounded), and `tune --gpu` records them in the `.zat`
    (`Coeffs.gpu`, format v9; a CPU-only profile keeps them 0). `decideOffload`
    and `decideOffloadMatMul` compare the measured device time against the CPU
    time (falling back to the constants when unmeasured). On the RX 580
    (ReleaseFast, 4 threads / 4 GB) the probe reads gemv ~8.2, stream ~5.7,
    matmul ~1.7 vs cpu ~37 GB/s, so the tuner offloads **0/749** decode and
    **0/869** prefill dispatches for gemma4-E2B and **0/868** / **0/1060** for
    gemma4-12B (was 170/288 decode). GPU decode is no longer slower than the CPU
    (E2B 15.3 vs 14.2 tok/s; 12B 4.0 vs 3.8); prefill is left on the CPU because
    the measured GEMM loses there (21 vs 30 tok/s when forced).
- **Prerequisite** for `gpu/04`/`gpu/05` to pay off; the async seams they need
  now exist.
- **Dep-driven wait + device-owned activation landed (bounded).** The executor
  submits resident GEMVs async and drains them at the consumer; a run sharing an
  input stages that activation once (device-local). `zig build` /
  `zig build test` green (252/252). Decode is bit-identical to the CPU plan (the
  generated text matched on E2B and 12B). Interleaved warm A/B on the RX 580
  (ReleaseFast native, `--threads 4 --ram 4gb --gpu --vram 4gb`, 100 tok E2B /
  60 tok 12B): E2B GPU 14.1->14.4 tok/s (CPU plan 16.1), 12B GPU 4.0->4.1 tok/s
  (CPU plan 3.9). The change is ~2% on the GPU plan and neutral on the CPU plan;
  **offload still loses on E2B and only roughly ties/edges on 12B.**
- **mclk does not ramp from sustaining.** Polling `mem_busy_percent` /
  `pp_dpm_mclk` during a run (250 ms samples) shows `mem_busy` ~1% with brief
  spikes and `mclk` at 400 MHz for ~97% of samples (occasionally 1000, rarely
  1750) both before and after — e.g. before 400:3938/1000:134, after
  400:3957/1000:100/1750:10 over the same 200-token E2B run. Root cause: the
  offloaded decode work is a small serialized side path (135 resident GEMVs/token
  moving ~140 MB/token, ~2 GB/s at ~14 tok/s) while the CPU runs the fused
  `ffn_gate`/`inp_gate`/`proj`/norm/attention chain between GPU bursts, so the
  GPU duty cycle stays ~5%, far below the `MCLK_ACTIVE_LEVEL=25` needed to ramp.
  Scheduling alone cannot raise it: the decode DAG is a strict CPU<->GPU chain
  (every GPU output is consumed by the next CPU op), so there is no independent
  CPU work to overlap and no run of GPU ops long enough to bridge the CPU gaps.
  The remaining blocker is **how much work is offloaded**, not how it is
  scheduled.
- **Probe left conservative (decided).** The decode path offloads only
  **resident** weights (`resident-implies-offload`), so the measured streamed
  rate does not gate decode; a warm/batched probe would only change the streamed
  and prefill-GEMM gates and previously regressed 12B to 1.8 tok/s. The current
  cold min-of-N probe (RX 580: gemv 2.87, streamed 2.75, matmul 1.61 vs cpu
  10.71 GB/s) stays. Re-measure only if the tuner starts offloading streamed
  decode GEMVs or unfused FFN gate/up on the GPU.

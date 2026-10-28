11 - Performance

Status: done
Commits: 0fe3a8e 4115581 0c8780c

> Historical note: the comparison numbers below are from local development
> builds (an earlier local prototype vs the current code), not external
> benchmarks. See `sdd/roadmap.md`.

## Goal

The current code is **slower than the earlier local prototype** in full-CPU inference. Close
the gap and surpass the earlier local prototype — faster TTFT, prefill and decode — while preserving the
AOT-deterministic design ("the runtime decides nothing") and keeping the plan
deterministic.

## Baseline (measured)

Same model (`gemma-4-E2B-it Q4_0`), full CPU, `--threads 8 --ram 8gb`, daemon
warm. Machine: AMD Ryzen 7 5700X (8C/16T), Linux 6.18.

| metric | the earlier local prototype (warm) | the current code |
|---|---|---|
| TTFT | ~1.6 s | ~6.7 s (`token`) / ~12-15 s (`batch`) |
| prefill | ~20 tok/s | ~2.4 tok/s |
| decode | ~12 tok/s | ~3.8 tok/s |
| peak RSS | ~1.37 GB | — |

Findings:
- The current code has a **per-request fixed CPU cost** (~6.7 s wall / ~14 s CPU) even for a
  1-token prompt; **independent** of residency state (`cold`/`warm`/`hot`) and of
  prefill mode (`token` still ~7 s); `majflt = 0` (not I/O). A 400-token prompt
  is ~176 s ⇒ ~0.41 s/token prefill + the fixed cost.
- The prefill GEMM (`matmul`) and the decode GEMV are both well below the earlier local prototype's
  throughput. Attention is O(`pos`) (correct, not the culprit).
- **Bandwidth (root cause)**: the current code's calibrated `dram_bw` is **~9.6 GB/s**
  (single-thread `measureDramBw`), but the machine aggregates to **~24 GB/s @4T /
  ~49 GB/s @8T**. `cost.parEff` for memory-bound ops returns `min(1, parts/2)`,
  so `selectParts` picks ~2 parts and decode achieves only ~11.7 GB/s
  (3.03 GB × 3.8 tok/s) vs the earlier local prototype's ~36 GB/s (~90% of the ceiling). The cost model
  under-parallelizes: it treats the DRAM ceiling as single-threaded and assumes
  memory bandwidth does not scale with threads.

### Other investigation findings
- **Calibration (item 2).** `calibrate.measureDramBw` streams a 64 MiB buffer
  **single-thread** -> ~9.6 GB/s, far below the machine's aggregate (~49 GB/s @8T);
  `measureDispatchLatency` returns the per-call time of `tinyOp` (~**2 ns**), not a
  dispatch latency (should be tens of ns). Both feed the cost model, so it
  under-parallelizes and mis-sizes the prefetch lookahead.
- **Numerical parity (item 3).** The earlier local prototype and the current code produce **identical** greedy text on
  the same prompts (`The capital of France is Paris.`, a haiku) -> the current code is
  numerically correct; the only difference is the current code leaking `<end_of_turn>` into the
  output (existing open-item).
- **Affinity / SMT (item 5).** `cpu.affinityPlan` groups CPUs by physical core and
  fills one thread per core before SMT siblings — **SMT-aware and generic**
  (Linux topology), no per-CPU-architecture kernel. Fine as-is.

## Scope

- **In:**
  - A **measurement harness** so the current code reports TTFT / prefill / decode / RSS like the earlier local prototype
    (needed to verify any change).
  - Eliminate the **per-request fixed cost** (profile the daemon during one
    request; find what is redone per `generate`).
  - Speed up the **prefill GEMM** (SIMD, blocking, `parts` through the Pool).
  - Speed up **decode** (matvec SIMD / `parts`).
  - Residency nodes for **prefill** dispatches (the first block currently relies
    on mmap fault-in).
- **Out:** new user-facing features; sampling; `engine/04` (model optimization);
  GPU; changing the deterministic tuning design.

## Design

### Invariant
Optimizations must not add runtime heuristics: the plan still carries every
decision; the runtime only executes it faster. Any tuning change stays a pure
function of (declaration, calibrated `caps`).

### Bandwidth and parallelism (root cause)
The current code under-uses DRAM bandwidth. The calibration measures single-thread streaming
(~9.6 GB/s) and `parEff` assumes memory-bound ops saturate at 2 parts. On this
machine aggregate bandwidth scales with threads (6.3 -> 24 -> 49 GB/s), so the
fix is:
- calibrate the **aggregate** DRAM bandwidth (sweep 1..N threads, keep the best)
  and retain the per-thread figure;
- model the bytes term with `bw_eff(parts) = min(ceiling, per_thread * parts)`,
  so `opCost`/`selectParts` use enough threads to reach the ceiling.
This must stay deterministic (a pure function of the calibrated `caps`) and keep
the compute-bound (`matmul`) model intact.

### Prefetch thread
The current code reserves one of `--threads` for the residency prefetch worker
(`compute_workers = threads - 1` when `threads >= 3`); the earlier local prototype keeps residency inline
and uses **all** threads for compute. With `majflt = 0` in steady state (weights
resident), the dedicated prefetch thread may cost more than it saves — evaluate
giving it back to compute (or gating it on actual page-in pressure).

### 1. Measurement
The earlier local prototype reports `ttft`, `prompt_eval`, `generation`, `peak_rss`. The current code needs the same.
A timing/metrics surface is required; **a `--metrics` flag on `run` is a CLI
change and needs explicit approval** — otherwise instrument behind a debug build
or a hidden test harness first.

### 2. Per-request fixed cost (highest priority)
Profile the `__serve` daemon across one `GENERATE`. Candidates: something
re-parsed/re-built per request (plan/zmeta/tokenizer), a buffer reset, or the
first forward at `pos = 0` doing O(model) work. Establish the cause with a
profiler (`perf` if available; otherwise sampling or internal counters) **before**
changing code.

### 3. Prefill GEMM
The `matmul` `blocked` variant dequantizes a weight row once and reuses it across
`M` rows, but is likely scalar. Target: SIMD dot over the dequantized block,
register blocking, and `parts` executed through the worker `Pool` (open-item).
Must stay numerically equivalent to the per-row GEMV (existing test).

### 4. Decode
The matvec variants should use the widest SIMD; verify `parts` actually
parallelize and that `bytesEffective`-driven choices hold. Attention is fine.

### 5. Prefill residency
Add residency nodes for the prefill dispatch list so the first block pins/prefetches
instead of faulting in (open-item "Prefill has no residency nodes").

## Steps

1. Measurement: instrument `run`/the daemon to report TTFT / prefill / decode /
   RSS (approve the CLI surface first). [TODO]
2. **Bandwidth fix**: calibrate the aggregate DRAM bandwidth (multi-thread sweep)
   + per-thread; make the memory-bound bytes term scale with `parts`
   (`bw_eff = min(ceiling, per_thread * parts)`); verify `selectParts` uses more
   threads and decode reaches near the ceiling. [TODO]
3. Profile the per-request fixed cost; identify and fix it. [TODO]

   **Found (instrumented):** the fixed cost is **all in the prefill** — for a
   1-token prompt `executePrefill` spends ~**11.7 s**, of which `matmul` = ~6.9 s
   and every other op ~70 ms combined. `runBlocked` dequantizes a weight row to
   **f32** and uses a **scalar** `vector.dot`, while the matvec `_u4`/`_t2`
   kernels fuse dequant+dot with SIMD — so `blocked` is ~17x slower. The prefill
   dispatches are compiled with `M = prefill_block` (16) and select `blocked`,
   then run with the **actual** `m` (1 for a short prompt). Fix: optimize
   `blocked` (fused SIMD dequant+dot, or reuse the matvec kernels) and/or pick the
   prefill variant from the actual `m` (a plan-declared threshold), so a short
   prompt does not pay a batched GEMM.

4. Optimize the prefill GEMM (SIMD + blocking + pool); keep equivalence. [TODO]

   **Fixed (item 1).** The prefill now selects the **tuned matvec** variant +
   `parts` (`selectMatvec`) and runs a **batched GEMV**: one pool job per
   projection, each worker handling its output-row slice for **all M input rows**
   (`pool.Job.batch`/`k`, `Session.matVecBatch`). E2B prefill **11.9 s -> 1.77 s**
   (2.4 -> 15.8 tok/s), coherent, deterministic, 157/157 tests. The
   `matmul`/`blocked` GEMM is **retired from the runtime path** (kept in the
   catalog; a real SIMD GEMM is future work — the scalar `blocked` is not it).
5. Optimize decode (SIMD / `parts`). [TODO]
6. Prefill residency nodes. [TODO]
7. Evaluate the dedicated prefetch thread (give it back to compute when page-in
   pressure is low). [DONE] `cost.decidePrefetch` now decides inline vs a
   dedicated worker from the calibrated `madvise(WILLNEED)` cost and the streamed
   bytes, records `residency_workers` in the plan, and the runtime reads it
   (no hardcoded gate). On this machine the model picks **inline** for every
   tested model/RAM (the `madvise` work stays below one thread's compute share);
   the tuner also sizes `parts` to `threads - residency_workers`.
8. Re-benchmark vs the earlier local prototype; record the numbers. [TODO]
9. `zig build` / `zig build test` green; E2B coherent; plan deterministic. [TODO]

## Definition of Done

- [x] The current code reports TTFT / prefill / decode / RSS (`cli/04`).
- [x] Decode reaches the aggregate ceiling (the matvec's per-thread is now used;
      `parts` reaches 8).

**Final.** E2B (prod, 8T): prefill **21.2** / decode **15.5** tok/s vs the earlier local prototype's 20 / 12
— **both >= the earlier local prototype**. Remaining: prefill residency nodes (open-item).
- [x] The per-request fixed cost is gone (prefill runs a batched tuned GEMV).
- [x] Prefill 21.2 and decode 15.5 tok/s at 8T — **both >= the earlier local prototype** (20 / 12).
- [ ] Prefill dispatches carry residency (no per-block fault-in storm) — open-item.
- [x] The compute thread budget is not needlessly reduced by the prefetch worker
      (gated on `hasAsyncPrefetch`, `kernels/08`).
- [x] Deterministic plan; `zig build` / `zig build test` green (168/168); E2B coherent.

**As-built.** Prefill uses a batched tuned GEMV (one pool job/projection); decode
uses the tuned variant + the measured scaling curve; the prefetch worker is only
reserved for real async prefetch. E2B (prod): decode 4.4 -> 9.6 tok/s (near the earlier local prototype's
12), prefill 2.4 -> 11.5 tok/s (earlier local build: ~20). Remaining: the cost model's memory term
uses the **streaming** per-thread bandwidth (~11 GB/s), but the matvec is
compute-bound per thread (~3.7), so `bwAt` saturates at n=4 and caps `parts` —
see the open-item (per-op per-thread bandwidth).

## Arch Provenance

- **own:** the deterministic tuning design (kept intact).
- **consulted:** llama.cpp/ggml SIMD kernels — reference only.
- **derived:** `runtime/09`/`10`, `kernels/05`/`06`; benchmark vs the earlier local prototype.

## Comments

- Baseline captured on the machine above; a full sweep (other models/threads) is
  part of step 6.
- The `--metrics`/timing surface on `run` is a CLI change: ask before adding it.
- Related open-items: prefill residency, ISA-specialized AtomicOps,
  benchmark/selection beyond matvec.
- ~~**`--hot` pinned the whole model (no page-cache headroom).** Resolved: `tune`
  reserves a page-cache headroom separately from the pin window
  (`pageCacheHeadroom`) and caps the `hot` pin set below `want_window`, so the
  plan carries `pinned_bytes < weights_bytes` with `window_bytes - pinned_bytes
  >= headroom` (streamed tail + OS page cache keep room). Deterministic at
  `tune`; the runtime only applies it (`sdd/open-items.md` entry deleted).~~
  (the `weights/8` `pageCacheHeadroom` heuristic was later replaced by explicit
  `W_hot`/`W_reserved` layers, themselves removed by `runtime/15`.)
- **Ordering: `runtime/12` (the cost pipeline) is built first.** It defines the
  scaling/efficiency logic of the selected algorithms (AST -> Roofline ->
  measured scaling, + the Evaluator). This task's calibration (aggregate
  bandwidth, dispatch latency, scaling curve) is what feeds 12's stages; without
  12 the cost model keeps making the same wrong scaling decisions.
- **KV cache is f16.** The KV is stored as **f16** (was f32) to halve its RAM and
  free page cache for streamed weights. The forward/attention compute in f32; the
  f16 KV is nearly lossless (the greedy output matches the reference in our
  tests) but is **not bit-exact** vs an all-f32 baseline — a near-tie argmax
  could flip. Recorded so precision/parity work keeps it in mind.

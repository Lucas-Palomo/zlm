08 - Parallel Matvec Efficiency

Status: done
Commits: (see runtime session prefetch-worker fix)

## Goal

Close the decode gap vs the earlier local prototype. With a production build the two runtimes are
**identical per thread** but the current code's matvec (99% of decode time) scales poorly
across threads: it gets only ~1.7x from `parts=3`, while the earlier local prototype gets ~3.7x at 4
threads. Make the **matvec kernel** scale near-linearly for this memory-bound op,
without changing the deterministic tuning or "the runtime decides nothing".

## Baseline (measured, prod build)

Same model (`gemma-4-E2B Q4_0`), full CPU, clean `ReleaseFast` build.

| threads | the earlier local prototype | the current code |
|---|---|---|
| 1 | 2.62 tok/s | 2.6 tok/s |
| 4 | **9.61 tok/s** | 4.4 tok/s |
| 8 | — | 4.4 tok/s |

- Per thread the two are **equal** -> the kernel's single-thread codegen is fine
  (AVX2 `@Vector(8,f32)`), so the gap is **parallel scaling**.
- The decode is **~99% matvec** (`DECODE_OP`: matvec 251 ms vs attention 1.4 ms,
  activation 3 ms, everything else < 0.4 ms).
- The large projections (attn_q/ffn_gate/ffn_down/attn_output) use `parts=3`;
  the matvec scales only **1.7x** going 1->8 threads.
- Ruled out by test: the **variant** (`u4` vs `t2` — same), the **parts count**
  (forcing 8 — same), the **worker spin** (`max_spin_iterations=0` — same), the
  **affinity** (SMT-aware, distinct cores).

## Scope

- **In:**
  - Diagnose why the matvec kernel is slower per worker under parallelism than
    alone (~1.7x vs the expected ~3x for a memory-bound op).
  - Fix the **kernel** so a row-range slice scales near-linearly; keep numerics
    identical.
  - Coordinate with the runtime's split (`pool.rangeOf`) only where the fix
    requires it.
- **Out:** the cost model / tuning (`runtime/12`); the prefill (fixed in
  `runtime/11`); the variant catalog.

## Design

### Hypotheses to check (kernel-centric)
1. **Shared `x` traffic**: every worker re-reads the full `x` vector (K f32) per
   output row; if it is not held in L1, aggregate traffic is `N*K*4`, dwarfing the
   weights — the classic reason a matvec does not scale.
2. **Slice granularity / cache**: `pool.rangeOf` splits the output rows into
   contiguous ranges; a worker's weight working set may exceed L2/L3 and miss more
   than a single-threaded full pass.
3. **Aggregate bandwidth**: the kernel may not reach the machine's aggregate
   bandwidth (single-thread ~8 GB/s; machine streams ~24 GB/s @4T, ~49 @8T).
4. **Tiling within the kernel**: how the tiled variants (`_t2`/`_t4`) reuse the
   activation across rows — and whether that reuse survives a row-range split.

### Method
- A micro-benchmark of the matvec kernel alone at `parts = 1..8` (single tensor,
  1..8 threads) reporting achieved GB/s and per-worker time — isolates the kernel
  from the pool and the rest of the decode.
- Compare against the machine's streaming ceiling (6.3 / 23.7 / 49 GB/s at
  1/4/8 threads).
- Only then change the split/kernel.

## Steps

1. Micro-benchmark the matvec kernel at `parts = 1..8` (achieved GB/s per
   worker); reproduce the 1.7x. [TODO]
2. Identify the limiter (shared `x` / granularity / bandwidth / tiling) with
   evidence, not speculation. [TODO]
3. Fix it (e.g. keep `x` resident per worker, block the output rows so the shared
   activation fits cache, or tile across the row range); keep numerics identical. [TODO]
4. Re-measure the current code vs the earlier local prototype at 1/2/4/8 threads. [TODO]
5. Tests: parallel matvec == single-threaded (existing); scaling is monotone. [TODO]
6. `zig build` / `zig build test` green; E2B coherent; plan deterministic. [TODO]

**As-built.** A micro-benchmark of the matvec kernel **alone** showed it scales
near-linearly (3.5-3.9x @4T, ~6x @8T) — the kernel was **not** the limiter. The
gap was the **runtime reserving a compute worker for prefetch** even when the
plan has no `prefetch` nodes (`Session.open`: `compute_workers = threads-1`), and
`Pool.run` clamps `parts` to `spawned+1`. E2B's graph has 0 `prefetch` nodes (all
pinned), so one worker sat idle. Fix: gate the reservation on
`hasAsyncPrefetch(nodes)`. Decode @4T **7.4 -> 9.3 tok/s** (earlier local build: 9.6); 162/162.
Remaining 8T headroom is the cost-model `parts` saturation (`runtime/12`): the
calibrated per-thread figure (~11 GB/s streaming) overestimates the matvec's
per-thread (~3.7 GB/s, compute-bound), so `bwAt` saturates at n=4.

## Definition of Done

- [x] The matvec kernel scales near-linearly for the memory-bound projections
      (micro-benchmark: 3.5-3.9x @4T) — the kernel was already fine.
- [x] The current code decode is within ~10% of the earlier local prototype at 4T (9.3 vs 9.6 tok/s) — the gap was the
      runtime's prefetch-worker reservation, now gated on `hasAsyncPrefetch`.
- [x] Numerics unchanged (parallel == single-threaded); plan deterministic.
- [x] `zig build` / `zig build test` green (162/162); E2B coherent.

## Arch Provenance

- **own:** the diagnosis + fix of the parallel matvec kernel.
- **consulted:** the earlier local prototype's `execution/root.zig` (identical sync design; the difference
  is elsewhere) and llama.cpp/ggml multi-thread matvec.
- **derived:** `kernels/04`/`06`/`07` (catalog, selector, variants),
  `runtime/05` (parallel execution), `runtime/11` (performance).

## Comments

- **Ordering: `runtime/12` (the cost pipeline) is built first.** It defines the
  scaling/efficiency logic of the selected algorithms (AST -> Roofline ->
  measured scaling, + the Evaluator); the `parts` this task scales is chosen
  there. Build 12 first so the scaling decision and the kernel's scaling behavior
  are consistent — otherwise the kernel is optimized against the wrong `parts`.
- Per-thread, the earlier local prototype and the current code are identical, so the **cost/variant is not this task's cause** — this is
  the kernel's parallel efficiency, the implementation counterpart to 12's
  scaling logic.
- The runtime's `pool.rangeOf` split is the boundary: if the fix needs a different
  split (e.g. interleaved or blocked rows), coordinate with `runtime/`.
- `runtime/11` fixed the prefill; this is the decode counterpart.

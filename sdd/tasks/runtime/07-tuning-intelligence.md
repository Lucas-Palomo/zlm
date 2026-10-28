07 - Tuning Intelligence

Status: done
Commits:

## Goal

Give the `tune` phase measured intelligence, in two phases: **Phase 1** picks
kernel implementations and places threads; **Phase 2** sizes and admits the
memory budget. The runtime keeps executing recorded orders only.

## Scope

- **In:**
  - **Phase 1 (kernels & placement).** A generic per-family benchmark harness in
    the tuner; apply it to the **linear** family (`matvec`, `MatVecPlan`
    variants); record the chosen plan per dispatch; derive `grain`/`parts` from
    the measured throughput; a CPU affinity plan.
  - **Phase 2 (budget & memory).** A cold mmap I/O benchmark -> `read_cost` +
    `reserve`; value-based admission; KV/window inversion; derived lookahead.
  - **Phase 3 (autotune artifact).** Persist/reuse the measured results in a
    hardware-specific `.zat` file (`zlm autotune`); it may only be overwritten
    with `--force`.
- **Out:** sub-range partitioning and the embedding `row_cache` (stays in
  `open-items`); GPU tiers; batched prefill itself (only its benchmark hook);
  per-ISA atomics (open-items).

## Design

**Phase 1 — kernel selection & placement**

- **Benchmark harness.** The tuner enumerates a family's candidate
  implementations and times each on a scratch input of representative
  `(ggml_type, shape-class)` with warmup + N iterations; it picks the fastest and
  caches by `(device, cpu, op, type, shape-class)`.
- **Selector.** The chosen variant is recorded per dispatch in the plan (a
  `matvec_plan`/variant byte; `.zplan` VERSION bump, also needed by affinity).
  The runtime calls `matVecRowsWithPlan(..., chosen)` — the Selector stops being
  a stub.
- **Grain.** `parts = clamp(work / grain, 1, budget)` with `grain` derived from
  the measured per-op throughput (rows/s) instead of the fixed `1_000_000`
  (`runtime/05`).
- **Applies to the linear family.** `matvec` has interchangeable `MatVecPlan`
  variants; `matmul` (batched) reuses the harness when batched prefill lands.
  `quant` has a single atomic today; the other families' atomics are
  semantically distinct (chosen by params/architecture), so no benchmark.
- **Affinity.** Detect topology (physical cores, SMT siblings, cgroup CPU quota)
  and emit an affinity plan (worker -> CPU) sized by `--threads`; the runtime
  pins each worker with `sched_setaffinity` (no-op off Linux).

**Phase 2 — budget & memory**

- **I/O benchmark.** mmap-read a sample cold (`posix_fadvise(DONTNEED)` between
  samples) -> `read_cost(size) = fault + size/throughput` and a measured
  `reserve`; cache by device.
- **Value admission.** `density = reuse_per_token * (read_cost + compute_cost) /
  size`; greedy knapsack over **all** ranges (HOT included). **As-built:**
  `admissionDensity` is I/O-only (`(streamed − resident)/size`) and `reuse` is
  effectively 1. This is correct and **does not materialize**, even on the larger
  models (verified 2026-10-07 on E2B/12B/31B):
  - Per token a range's disk read amortizes to ~1: the first read faults, the rest
    hit the page cache; the value saved is reuse-independent.
  - The admission ranks by **per-byte saved**, which is reuse-independent, so the
    ordering is robust even when some ranges are read more than once.
  - Measured `reads/token` (dispatch references): E2B 467×1 + a few tiny shared
    norms (up to 17×); 12B 525×1, max 6×; 31B 645×1, max 8×. The `>1` cases are
    the content-dedup'd `attn_q/k_norm` and `layer_output_scale` (1–2 KB, already
    HOT), tied `token_embd` (2× but **different bytes** — one row vs the full
    table), and — only on 12B/31B — the `k_eq_v` `attn_k.weight` (the global
    layers read it for both the k and v projections).
  - Those `attn_k` re-reads are **adjacent and of the same bytes** (q→k→v in the
    block), so the second hit is page cache, not disk: disk I/O stays ~1. A
    `reuse` multiplier would therefore **over-count**; a correct model would need
    cache-eviction tracking between reads (the runtime has it; `tune` does not).
  - Batched prefill / prefix-cache hits change the workload-level read total, not
    the per-token residency ranking.
- **KV/window inversion.** Admit the pinned set first, then cap the context so
  `KV(context) + pinned + reserve + slots <= ram`; the window is never starved to
  `0` when the hot set fits (today KV-first gives `window = 0` at `--ram 1gb`).
- **Lookahead.** `prefetch_lookahead = clamp(read_latency / per_dispatch_time,
  1, N)`, recorded per plan (replaces the fixed `12`).

**Phase 3 — autotune artifact (`.zat`)**

- The measurements (Phase 1 variants, throughput/grain, Phase 2 `read_cost`/
  `reserve`/lookahead, affinity) are hardware-specific and expensive to redrive,
  so they are persisted in a `.zat` file (Zlm AuTOTune) in the store, keyed by
  the **image name + tuning configuration fingerprint**
  (`--ram --threads [--gpu --vram]`, i.e. `Config.fingerprint()`), at
  `.zlm/autotune/<image-name>_<fp>.zat`.
- **`tune`** loads the `.zat` when present and reuses it (no re-benchmark);
  otherwise it measures once and writes it.
- The `.zat` may **only be overwritten with `--force`** (a new flag on `tune`):
  without it, an existing `.zat` is used as-is; with it, measurements are
  redriven and the file replaced.
- The hardware-independent parts (image, plan structure) are unchanged; `.zat`
  is the hardware layer that a `.zplan` references/uses at tune.

## Flow / Arch

```text
tune:
  Phase 3  load .zat  (present and no --force -> reuse; else measure)
  Phase 1  candidates -> time -> choose variant -> Dispatch.matvec_plan
           throughput -> grain -> parts ; topology -> affinity
  Phase 2  cold mmap benchmark -> read_cost, reserve
           value density -> pinned ; admit pinned first -> cap context (KV)
           lookahead = f(read_cost, per_dispatch_time)
  write .zat (measurements) ; write .zplan { matvec_plan, budget, nodes, lookahead, affinity[] }
run:
  Session pins workers on spawn; executor uses the recorded variant; decides nothing
```

## Arch Provenance

- **own:** measured tuning/placement/budget as AoT decisions.
- **spec:** —.
- **consulted:** the earlier local prototype's `execution/root.zig`
  (`detectAffinityPlan`/`pinCurrentThread`) and `autotune/*` (benchmark/candidates/
  cache) — reference only.
- **derived:** from `runtime/05` (parts/grain), `runtime/06` (budget, nodes) and
  `sdd/open-items.md`.

## Steps

**Phase 1**
1. Benchmark harness: enumerate a family's candidates, time on a scratch input,
   cache by device/op/type/shape-class.
   - **As-built (1a):** `src/tuner/bench.zig` (`benchMatvec` ns/op,
     `bestMatvecPlan`); candidates per `ggml_type` (`q4_0` legacy/u4/t4/t2/
     production; `q4_1`/`q8_0`/`q6_k` legacy/u4/t2/production); native 2-D mapped
     views only.
2. Apply to `matvec` (`MatVecPlan`); record the chosen variant per dispatch
   (VERSION bump); runtime uses it.
   - **As-built (1a):** `matvec.Args.plan`; `Dispatch.matvec_plan` (VERSION 5→6);
     `tune.selectMatvecPlans` benches per `(ggml_type, m)` (in-tune cache) and
     records per dispatch; threaded executor → `Session.matVecPlan` → `pool.Job`
     → `Args.plan` → `matVecRowsWithPlan`.
3. Derive `grain`/`parts` from the measured throughput.
   - **As-built (1a):** `tune` derives `parts` from the measured `ns/row` with
     `target_part_ns = 200_000`; `compile.partsFor` stays as the no-measurement
     fallback.
4. Affinity: topology detect -> plan -> pin on spawn (Linux-only).
   - **As-built (1b):** `utils/cpu.zig` `affinityPlan` (allowed CPUs via
     `sched_getaffinity`, physical-first by `(package,core)`, capped by cgroup
     current/earlier quota; fallback `0..threads-1`; empty off Linux) and
     `pinCurrentThread` (`sched_setaffinity`); `AOTKernel.affinity: []u16`
     (VERSION 6→7); `tune` records it; `executor` pins the main thread to
     `affinity[0]` and the pool/prefetch pin their workers. Verified on E2B:
     worker `Cpus_allowed_list` = `0,1,2,3` with `--threads 4`.

**Phase 2**
5. Cold mmap I/O benchmark -> `read_cost`, `reserve` (cached; fallback).
   - **As-built:** `benchIoCost` (`IoCost{fault_ns, ns_per_byte}`) fits
     `cost(size)=fault+ns_per_byte*size` from cold samples (private mapping +
     `fadvise(DONTNEED)` + touch); `null` off Linux / on failure -> constants
     `{100_000, 2.0}`. `reserve` = computed scratch + 8 MiB margin, floor 32 MiB.
6. Value-based admission over all ranges using the measured cost.
   - **As-built:** `density = (read_cost(size) + compute_ns)/size`, greedy over
     all live ranges within `window`; embeddings drop out by low density; admitted
     ranges are marked `HOT` so `keep` nodes stay non-evictable.
7. KV/window inversion (admit pinned first, then cap context).
   - **As-built:** `want_window = max(value-admitted hot set, 8 MiB)`; then
     `context = clamp((ram - reserve - slots - want_window)/kv_per_token, 1, ...)`.
     E2B `--ram 1gb`: `fixed 983MB window 8MB pinned 7MB`, context
     `131072 -> 27952` (was `window 0`); `256mb` -> context 6107; `16mb` ->
     `window 0`, context 1 (`degraded`, never fails).
8. Derive the prefetch lookahead; record it in the plan.
   - **As-built:** `lookahead = clamp(fault_ns / median_matvec_ns, 1, 64)`
     (fallback 12), baked into node `trigger_step` (no plan field).

**Phase 3**
9. `.zat` artifact: persist the measured results; load and reuse on `tune`; write
   when absent.
   - **As-built:** `src/tuner/autotune.zig` (`Autotune{io, matvec[]}`,
     `save`/`load`, magic `"ZAT1"` + version + Wyhash checksum);
     `store.autotunePath`/`ensureAutotuneDir`. Path
     `.zlm/autotune/<image-name>_<fp_hex>.zat` (`fp_hex` = `Config.fingerprint()`
     hex). Stored: `IoCost` + matvec entries `{type, m, plan, ns}`; `reserve`/
     lookahead/affinity are cheap and re-derived each tune.
10. `--force` on `tune`: redrive the measurements and overwrite an existing
    `.zat`; without it, never overwrite.
    - **As-built:** new `--force` flag on `tune`; `!force && exists` -> load +
      reuse (skips `benchIoCost`/matvec benches); otherwise measure and save
      (overwrite only when `--force` or absent). Prints
      `autotune: reused|measured ...`. Verified: reuse 0.019 s vs `--force`
      1.56 s; file `e2b-test_940cef71011285f7.zat`.

11. `zig build` / `zig build test` green; E2B run with a small `--ram` (bounded
   RSS, coherent) and a measured speedup report.
   - **As-built:** build/test green (82 tests); E2B coherent, bounded
     (426/428 MB @ 1gb/256mb), 4 threads pinned to CPUs 0-3.

## Definition of Done

- [x] Phase 1: `matvec` variant chosen by measurement and recorded per dispatch.
- [x] Phase 1: `grain`/`parts` derived from the measured throughput.
- [x] Phase 1: affinity plan recorded and applied on spawn (no-op off Linux).
- [x] Phase 2: cold I/O benchmark -> `read_cost`/`reserve` (cached; fallback).
- [x] Phase 2: value admission + KV/window inversion (window not starved).
- [x] Phase 2: lookahead derived and recorded.
- [x] Phase 3: measurements persisted in a `.zat` and reused by `tune`.
- [x] Phase 3: `.zat` overwritten only with `--force`.
- [x] `zig build` / `zig build test` green; E2B run bounded and coherent.

## Comments

- Two phases, one deliverable: Phase 1 = decisions about *what/how to run*;
  Phase 2 = decisions about *where to keep it*.
- The harness is generic per family; only `matvec` (and later `matmul`) has
  interchangeable variants today.
- Sampling by `(type, shape-class)` and caching keep `tune` cheap.
- Affinity is Linux-only; macOS stays a no-op.
- `token_embd` row-cache / sub-ranges remain in `open-items`.
- The `.zlm/autotune/<image>_<fp>.zat` path is superseded by `engine/03`
  (Runtimes and Manifest), which moves the `.zat` to `.zlm/runtimes/`.
- **Prefetch worker decision (Phase 2 extension, supersedes the fixed
  reservation).** Step 8's lookahead is joined by `cost.decidePrefetch`: from the
  calibrated `madvise(WILLNEED)` cost (`calibrate.measureMadviseCost`) and the
  streamed bytes it decides **inline vs a dedicated worker**, recorded as
  `Config.residency_workers` (`.zplan` v10) and applied by the runtime (no
  hardcoded gate); `parts` is sized to `threads - residency_workers`. The
  decision and its coefficients are detailed in `runtime/09`; measured on this
  machine it always resolves to **inline** (the `madvise` work stays below one
  thread's compute share).

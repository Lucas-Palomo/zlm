06 - Residency and Budget

Status: done
Commits:

## Goal

`--ram` works de verdade, and the ZTEF stays on disk behind a **first-class,
AoT-compiled `ResidencyGraph`**. The Scheduler (tuning) decides where to
allocate, how much, what to pin, what to prefetch (async), and what to release;
the runtime only executes those orders. The result: the most useful data stays
resident without OOM; the rest is paged from the mmap.

## Scope

- **In:** a first-class `ResidencyGraph` in the plan (ranges + budget + ordered
  nodes with `sync|async`, deadlines, tiers); a Scheduler that computes the
  budget split from the workload (no fixed percentages) and emits the nodes; a
  runtime residency engine (byte accounting + budget guard) that runs async
  orders on the worker pool and provides a sync barrier to the executor; a
  computed safety reserve; graceful degradation (never fail) when the budget
  cannot fit the fixed set.
- **Out (now):** GPU execution and real tier moves (VRAM) — but the tier/device
  dimension and sync/async mode are part of the vocabulary from the start.

## Design

### The graph is the program

`ResidencyGraph` is a compiled artifact, produced by the Scheduler and stored in
the `.zplan`. It is **orders**, not hints:

- **Ranges** — the weight units: `{ ztef_kind, offset, length, flags }`. A range
  is **hybrid from the start**: a full tensor *or* a sub-range of one (offset +
  length into the ZTEF). Which one is a tuning decision (below), not a mode.
- **Budget** — the resident window in bytes, decided at tuning.
- **Nodes** — ordered orders over the **per-token** dispatch sequence:

  | field | meaning |
  | --- | --- |
  | `action` | `reserve` (grow/pin), `keep` (pin), `prefetch`, `release`, `evict`, `materialize`/`dematerialize` (bounded arena) |
  | `tier` | `ram` / `disk` (today) / `vram` (future) |
  | `mode` | `sync` (barrier: must be resident) / `async` (overlap with compute) |
  | `range` | the weight range |
  | `trigger_step` | when in the per-token sequence to fire |
  | `deadline_step` | for `async`: must be resident by this step |
  | `priority` | admission/eviction ordering |

The plan also stores `pinned_bytes` / `window_bytes` so the runtime knows the
split it must honor.

### Budget policy — computed, not percentages

The earlier local prototype split `--ram` into fixed X% hot / Y% dynamic / Z% safety. That is a proxy for
"what is actually needed", which the AoT *knows*. Compute the three sets
directly:

- **fixed (non-evictable):** activation buffers + KV caches. Known exactly at
  `tune`.
- **reserve (safety):** the worst-case transient (scratch, allocator
  granularity, headroom), **measured**/computed at `tune` — not a fixed Z%.
- **window = ram_bytes - fixed - reserve:** the evictable weight budget.

**Anti-OOM: degrade, never die.** The tuner always emits a feasible plan; it
never fails for a small `--ram`. The degradation ladder, in order:

1. **Shrink the window** (`window = max(0, ram - fixed - reserve)`): in the
   extreme, `window = 0` and every weight streams from the mmap.
2. **Shrink `fixed`**: cap the KV (context length; the sliding window already
   bounds sliding layers), so the KV fits what is left.
3. **Shrink the reserve** to its floor.
4. If even the minimal plan (tiny context + all-stream) does not fit the
   hardware, run the smallest feasible plan and **warn** — but do not abort.

Every step is recorded and printed (`tune` shows what was degraded), so the user
sees why the plan got smaller.

**Input validation.** `--threads >= 1` and `--ram > 0` (strictly positive),
rejected at parse with a clear message. `--ram = 0` is *not* a "full disk"
sentinel: full-disk is the degradation extreme (`window = 0`), reachable with any
positive budget below the floor. A future explicit `--no-weight-cache` flag can
name that mode, but 0 stays invalid. This removes the zero-budget / zero-thread
edge cases (no div-by-zero, no empty pool, no sentinel ambiguity).

Within the window, admit by **value density**, not "hot vs dynamic":

- reuse count (how often a range is read per token / over the schedule),
- liveness distance (first..last use),
- read cost (measured disk/mmap throughput at `tune`),
- size.

The always-hot set (norm weights, embeddings) is pinned when it fits; the rest
is streamed with async prefetch. Admission is a greedy/knapsack by
`value/byte`; ranges that do not fit stream from disk. No thrash: never pin more
than the window; whatever does not fit after degradation streams.

**Granularity is part of the same intelligence.** A range is not forced to be a
whole tensor or a fixed slice: the Scheduler partitions each weight by value.
A tensor whose whole content is reused enough is admitted **full** (one big
resident range, fewer nodes, better locality); a large tensor used once is
**streamed in sub-ranges** (page only what a dispatch needs, evict the rest).
The choice is made per tensor from `value/byte` (reuse × read cost vs size),
subject to the window — so "one full tensor" and "several sub-ranges" compete on
the same score.

**Measurement is incremental.** The first cut uses a small, cheap heuristic
benchmark at `tune` (sample sequential read throughput + first-touch latency) to
size `reserve` and estimate read cost. Later it grows into a real benchmark that
measures the model's tensors (per-tensor read/compute cost) and feeds the value
computation directly. Same inputs, better estimates; no format change.

### Asynchronous, using the thread budget

Because generation is a **repeated per-token sequence**, residency is a periodic
schedule: while token `t` computes layer `i`, the runtime can prefetch layer
`i+k`'s weights for the *same* token and keep hot ranges for token `t+1`.

- `async` nodes are queue from `advance(step)` (fire when due) and executed on
  the session's worker pool (from `runtime/05`) at low priority, so they overlap
  compute; they never displace a compute thread.
- `sync` nodes (and `keep`) are barriers: before a dispatch that reads a range,
  the executor ensures it is resident (waits for the pending async or pages it
  in). The executor decides nothing; it only hits the barrier.

### Heterogeneous (future) — reserved now

A node's `tier` + `mode` already express "put this here, move it there,
async/sync". Today only `ram`/`disk`. `vram` + `tier_move`/`upload`/`download`,
and compute placement (which device runs a dispatch), become Scheduler decisions
that reuse the same graph — no new runtime vocabulary.

## Flow / Arch

```text
tune (Scheduler):
  fixed   = slots + KV (exact)
  reserve = measured headroom -> shrinks to its floor under pressure
  window  = max(0, ram - fixed - reserve)  # anti-OOM: degrade, never fail
  if fixed + reserve > ram: shrink KV/context, then window -> 0
  ranges  = weight units (dedup)
  value   = f(reuse, liveness, read_cost, size)
  pin     = top value ranges while sum <= window (hot first)
  nodes   = keep(pin) + async prefetch(deadline) + release + evict(LRU) + reserve
  .zplan { ranges, budget{pinned,window}, nodes }        # VERSION bump

run (runtime, orders only):
  Session.open: mmap ZTEFs; Residency engine from the graph (byte accounting)
  executor(d): engine.barrier(d.residency_sync)   # ensure resident
               engine.advance(d.step)             # fire due async ops on the pool
               runDispatch(d)
  engine: never exceeds window; evicts LRU non-pinned before admitting
```

## Arch Provenance

- **own:** the computed budget split, value-based admission, async periodic
  schedule, first-class graph-in-the-core.
- **spec:** `.zplan`/`ResidencyGraph` (self).
- **consulted:** the earlier local prototype's `shadow_graph.zig` (actions, LRU, HOT,
  consume-accounting) and its percentage policy (as the anti-pattern to replace).
- **derived:** from CONSTITUTION ("the image is the source of truth"; anti-OOM),
  `runtime/05` (thread budget), `plan/ResidencyGraph`.

## Steps

1. Plan: first-class `ResidencyGraph` (ranges + `{pinned,window}` + nodes with
   action/tier/mode/range/trigger/deadline/priority); `Dispatch` references its
   sync/async nodes; VERSION bump; round-trip asserts.
2. Scheduler (`tune`): compute `fixed`/`reserve`/`window`; classify ranges
   (hybrid: full tensor vs sub-ranges, by value); value-based admission; emit
   nodes (keep/prefetch/release/evict); budget-aware KV/context; **degrade**
   (shrink window then KV/context, never fail); print the breakdown.
   `reserve`/read cost from the first-cut heuristic benchmark (a real
   per-tensor benchmark is a later refinement).
3. Runtime: `Residency` engine (byte accounting, budget guard, LRU eviction,
   `keep` pinning) + `madvise(WILLNEED/DONTNEED)` (or readahead); sync barrier;
   async execution on the worker pool.
4. Executor: hit barriers and advance the residency frontier per dispatch;
   decide nothing.
5. Tests: graph round-trip; engine accounting/eviction with a synthetic window;
   async prefetch + barrier ordering; `tune` degradation (window -> 0, KV cap)
   still produces a runnable plan.
6. `zig build` / `zig build test` green; run E2B with a small `--ram`, show
   bounded RSS (`/usr/bin/time -v`) and that async prefetch overlaps (metrics).

**As-built (stages 1-2).** Plan: first-class `ResidencyGraph` (`Range{kind,flags,offset,length}`,
`Node{action,tier,mode,range,trigger_step,deadline_step,priority}`,
`{ranges,nodes,pinned_bytes,window_bytes}`), `Dispatch.residency`, VERSION 5.
Scheduler: per-tensor ranges (HOT = norms/embd), `first_use`/`last_use` liveness,
HOT-that-fits pinned smallest-first, rest `prefetch`(async)+`release`, priorities
by size; budget `fixed`/`reserve`(32 MiB const)/`window`; context capped to fit
(`degraded: context X -> Y`); `--threads`/`--ram` must be `> 0`.
Runtime: `runtime/residency.zig` sync engine (byte accounting, `window` guard,
LRU eviction of non-HOT, `madvise(WILLNEED/DONTNEED)` via
`ztef.Reader.adviseRange`); `beginToken()` resets per-token node states;
executor applies `residency` + `advance` per dispatch; `generate` bounded by
`max_seq`.
Verified (ReleaseFast, E2B): `--ram 1gb` → fixed 992 MB, context capped
131072→28180, `run` 17 s for prefill+20 tokens, **peak RSS 415 MB, stable**;
`--ram 256mb` → fixed 223 MB, context→6334, 9 s, peak 419 MB; both coherent.
Bugs fixed on the way: `adviseRange` page alignment, the `Session` self-reference
(`Residency` must not hold `&self.ztefs` across the by-value return), and the
per-token node-state reset.

**As-built (stage 3).** `runtime/prefetch.zig`: one background worker over a
bounded ring (256) + in-flight set (`Io.Mutex`/`Condition`); the main thread
resolves each range to a page-aligned absolute `{addr,len}` (no `Reader`/
`Session` back-pointer, so the by-value `Session` return cannot dangle) and
enqueues; the worker only issues `madvise(WILLNEED)`. `Residency.prefetch` is
borrowed; `advance` enqueues `prefetch` nodes (issuing WILLNEED even at
`budget == 0` — readahead hint, not accounted residency) with an inline
fallback when the queue is full; `barrier` waits for in-flight then re-checks
residency inline; `applyDispatch` runs the node then barriers its range.
`tune` lookahead `prefetch_lookahead = 12`. `Session` owns a heap `Prefetch`
(only when `threads > 1`), joined in `deinit`. E2B (ReleaseFast): `--ram 1gb`
6-8 s / 415 MB, `--ram 256mb` 7 s / 417 MB, coherent, stable.
The thread budget is single-sourced from `config.threads`: the `Session` splits
it as main + compute helpers + residency workers. The count is no longer a fixed
`threads >= 3` gate — it is `Config.residency_workers` (0|1), decided by the
tuner's `cost.decidePrefetch` (calibrated `madvise` cost vs one thread's compute)
and read by the runtime; so `compute_workers = threads - residency_workers` and
`parts` is capped to it; verified process thread count equals `--threads`
(4 -> 4, 2 -> 2).

Still open: sub-range partitioning, value-based admission beyond
HOT-smallest-first, and the measured benchmark (grain/reserve).

## Definition of Done

- [x] The plan carries a first-class `ResidencyGraph`; the runtime only executes it.
- [x] `--threads >= 1` and `--ram > 0` are validated (0 rejected with a clear
      message); full-disk is the `window = 0` degradation, not a magic value.
- [x] Budget is computed (fixed/reserve/window), not percentage-based; a too-small
      `--ram` **degrades** (shrink window then KV/context) and never fails.
- [x] Weight residency never exceeds the window; LRU eviction of non-pinned
      ranges; admission is HOT-smallest-first (measured value deferred).
- ~~Ranges chosen by measured `value/byte`~~ / ~~hybrid sub-range partitioning~~
  / ~~measured `reserve` + read-cost benchmark~~ → deferred to
  `sdd/open-items.md` (budget intelligence).
- [x] Async prefetch runs on a background worker with a barrier (a dedicated
      worker rather than the compute pool; the pool is matvec-row-specific).
- [x] A `--ram` below the full model still runs (paging) with bounded RSS
      (415 MB @ 1gb, 419 MB @ 256mb, stable over tokens).
- [x] `zig build` / `zig build test` green.

## Comments

- **Measurement is incremental by design:** first cut = a small heuristic
  benchmark at `tune` (sample read throughput + first-touch latency) to size
  `reserve` and estimate read cost; later = a real benchmark over the model's
  tensors feeding the value computation. Fall back to conservative constants if
  the benchmark is unavailable.
- **Granularity is hybrid from the start:** full tensor and sub-range compete on
  the same `value/byte` score; neither is a special case.
- Async first cut runs on the existing compute pool; a dedicated residency
  worker (or I/O thread) is a refinement if contention shows up.
- Heterogeneity (`vram`, device placement) is vocabulary-only now; no GPU yet.
- **Decided:** `fixed + reserve > ram_bytes` **degrades** (shrink window, then
  KV/context; all-stream in the extreme) and never fails — anti-OOM means the
  program never dies.
- ~~Embedding tables (`token_embd`/`per_layer_token_embd`) pinned whole; add a
  `row_cache` directive (plan-decided budget, row-LRU at run)~~ — resolved: the
  tuner no longer marks embeddings HOT nor emits keep/prefetch nodes for them,
  and reserves a bounded `row_cache_bytes` (`rowCacheBytes`, capped by the L3
  and `window/4`); the runtime backs lookups with a `RowCache` (`plan.VERSION`
  13). Still open: partitioning a tensor into sub-ranges when its liveness
  differs (see `sdd/open-items.md`).

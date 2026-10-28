15 - Weight Ring (three-pillar residency buffer)

Status: done
Commits:

## Goal

Replace the adaptive multi-layer residency (`W_hot` + `W_reserved` + the staging
arena + the LRU window) with an explicit **three-pillar ring** --
`compute | reserved | prefetch` -- that holds the resident weight working set in
**consumption order**. The ring is opt-in, gated on `--residency` +
`--prefetch-threads` (`cli/09`): the tune sizes the shape, the runtime's prefetch
workers fill it, and the executor consumes it. On consume each pillar **rotates**
into the next role (`compute` -> the next `prefetch` target, `reserved` ->
`compute`, `prefetch` -> `reserved`), so the slot is **refilled, never discarded**
and the pipeline stays full while the workers work. Without `--residency` there
is **no** residency machinery: every weight streams on demand from the image mmap
(the `b25630b` behavior).

## Scope

- **In:** the three-pillar ring (states, producer/consumer, rotate/recycle) in the
  runtime; the tune's shape (the block size, `k = M`); the plan fields +
  `VERSION` bump; the executor's ring `advance` on consume; the **removal** of
  `W_hot` (`hotSpotBytes`, `pinned_bytes`, the `HOT` pin), `W_reserved`
  (`reservedSpotBytes`, `window_bytes`, the LRU eviction), the staging arena
  (`arena_bytes`, the `STAGED` flag, the `pread` staging) and the
  `pageCacheHeadroom` / `hotWindowBytes` heuristics (the space they reserved is
  the ring's buffer now).
- **Out:** changing the quantization; the image compression (`runtime/16`, since
  discarded); the
  GPU/VRAM residency (`gpu/`); a live residency control (`zlm state`, future); the
  KV precision; the embedding row cache (kept -- a separate mechanism).

## Design

### The ring

Three pillars; each pillar is a **block** -- a run of the streamed ranges in
**consumption order** (the ranges the token reads next), so the producer and the
consumer walk the same sequence:

```text
pillar:   [ compute  ]  [ reserved ]   [ prefetch ]
meaning:  in use by     the next block, being read (M workers,
          the layer     already read,   parallel pread) to become
                        waiting        `reserved`
```

- **Producer** (the prefetch pool, the `--prefetch-threads M` workers): fills the
  `prefetch` pillar with parallel `pread` -- `k = M` chunks, one per worker, so
  the concurrent streams equal M with no queued surplus (the staging arena's
  parallel-`pread` read path, reused). A filled pillar becomes `reserved`.
- **Consumer** (the executor): the compute reads the `compute` pillar's ranges;
  when the pillar's last range is consumed the ring rotates.

### The rotation (the fix for the first ring attempt)

```text
on consume (the `compute` pillar's last range):
  compute  -> becomes the next  `prefetch` target   (refill the slot, do not free)
  reserved -> becomes the new   `compute`
  prefetch -> becomes the new   `reserved`
```

The slot that leaves `compute` is **recycled as the next prefetch target**, so
the ring **oscillates**: the workers are always one block ahead and the same
three buffers cycle indefinitely (the consumption order repeats every token). No
block is freed on consume. The first attempt (the discarded ring) freed a
row as soon as the consumer left it, so the producer restarted the fill from
scratch and the next token re-read every weight -- it regressed (38.5 vs
28.9 GB/token); the recycle is the fix.

### The tune sizes the shape

The tune owns every shape decision (the runtime decides nothing):

```text
block_bytes = the pillar size -- the natural consumption unit (a layer) when
              three layers fit `--residency`, else step down (a projection, a
              run of ranges) so three pillars still fit; never more than R/3.
k           = M (the `--prefetch-threads` worker count): one chunk per worker.
```

The plan carries the block size (and `k`); the ring is always **three** pillars
(the minimum triple buffer). A tight `--residency` shrinks the block, never the
pillar count.

### The runtime

- `Session.open`: when the plan has a ring (`block_bytes > 0`) it allocates the
  three pillars (`3 * block_bytes`) and starts the M prefetch workers; a plan
  with no ring allocates nothing and starts no pool.
- The ring engine owns the pillars, the per-range block index and the
  producer/consumer state; `tensorView` returns the pillar slice for a ringed
  range and the mmap slice otherwise. The ring runs continuously across tokens
  (the consumption order repeats), so there is no per-token reset.
- The executor only calls the ring's `advance` when it consumes a range
  (`ring.onConsume(d)`); it decides nothing.

### The gate

The ring exists only with `--residency` (which requires `--prefetch-threads`;
`cli/09`): it needs the producer workers to fill the pillars. `--prefetch-threads`
alone is the **readahead-only** mode: the workers issue `madvise(WILLNEED)` for
the blocks ahead (the `b25630b` behavior), with no pillars. Neither flag present
=> the mmap + demand paging. The mmap still serves the un-ringed ranges (the row
cache, the ranges past the pillar) in every case; the ring never replaces the
mmap.

## Flow / Arch

```text
tune --residency R --prefetch-threads M:
  block_bytes = pick the pillar size (a layer | a projection | a run of ranges)
  k           = M
  plan: ring block_bytes + k                        # remove W_hot/W_reserved/arena

run:
  ring.init(3): pillars = { compute, reserved, prefetch }
  producer (M workers): prefetch -> parallel pread (k chunks) -> reserved
  consumer (executor):  compute -> onConsume(last range) -> rotate:
      compute -> prefetch (refill) ; reserved -> compute ; prefetch -> reserved
```

## Steps

> **As-built (uncommitted).** Steps 1-5 landed: the plan schema (`Range.block`,
> `ring_block_bytes`/`ring_k`, `VERSION` 31) with the dead layers removed; the
> tuner's ring sizing (block = layer, clamped to `R/3`) and the madvise-only mode
> for `--prefetch-threads` alone; the `Ring` engine (three pillars, `block % 3`
> rotate/recycle, async `pread` on the pool, inline fallback; `slice` falls back
> to the mmap) wired into `session`/`executor`; the old LRU engine, the nodes,
> the arena and the admission deleted. `zig build` green; `zig build test` 293
> pass + 1 skip. Step 6 (the 72B) is open.

1. **Spec + plan** (`plan/root.zig`): the ring fields (`block_bytes`, `k`); drop
   `pinned_bytes`, `window_bytes`, `arena_bytes` and the `HOT`/`STAGED` flags;
   serialize/deserialize + `fingerprint`; `VERSION` 30 -> 31.
2. **Tuner** (`cli/commands/tune.zig`): replace `hotSpotBytes`/`reservedSpotBytes`/
   `arena`/`admit` with the ring sizing (the block from the consumption order, `k`
   = M); build the per-range block index; drop the `keep`/`prefetch` node build.
3. **Runtime** (`runtime/residency.zig` a `Ring`, `runtime/session.zig`): the three
   pillars (alloc/free), the producer/consumer state, the rotate/recycle;
   `tensorView` serves a pillar slice; start the pool from the ring gate.
4. **Executor** (`runtime/executor.zig`): `ring.advance`/`onConsume` on consume;
   drop `advance`/`applyDispatch`/`beginToken` of the LRU engine.
5. **Tests:** the ring alloc/rotate/recycle (the slot returns to `prefetch`); a
   tight budget still yields three pillars; `block_bytes == 0` => no ring (the
   plan is the mmap stream); encode -> decode the ring fields.
6. **Verify:** `zig build` / `zig build test` green; the qwen3-72B decode +
   disk bandwidth with the ring on/off (the regression must not return).

## Definition of Done

- [x] The ring is exactly three pillars (`compute | reserved | prefetch`) in
      consumption order; a consumed pillar is recycled as the prefetch target.
- [x] The plan carries the ring shape; a plan with no ring is the mmap stream
      (`b25630b` behavior, no residency machinery).
- [x] `W_hot`/`W_reserved`/`arena`/`pinned`/`pageCacheHeadroom`/`hotWindowBytes`
      are gone from the plan, the tuner and the runtime.
- [x] The ring degrades the block size under a tight `--residency` (clamped to
      `R/3`, the overflow falls back to the mmap) and never OOMs.
- [x] `zig build` / `zig build test` green (293 pass, 1 skip).

## Arch Provenance

- **own:** the explicit three-pillar residency ring; the rotate/recycle (the
  `compute` -> `prefetch` oscillation); the removal of the multi-layer budget.
- **derived:** from `b25630b` (the no-flag stream, the `--prefetch-threads`
  madvise-only path); from the residency layers being replaced (the
  `W_hot`/`W_reserved` budgets, the lookahead/pin and the staging arena -- the
  parallel `pread` read path is reused); and from the first ring attempt's
  measurement (the discard-on-consume regression that the recycle fixes).

## Comments

- **Why three pillars.** The consumption order repeats (every token walks the
  layers in the same sequence), so a triple buffer is the minimum that keeps one
  block computing, one ready and one filling. Extra rows (prefetch depth) were
  the first attempt; they add buffer without a consumer, and the compute pillar's
  recycle already gives the depth.
- **What is removed and why.** The `W_hot`/`W_reserved` layers, the lookahead +
  pin and the staging arena each added a layer to the adaptive LRU. The ring
  makes the schedule **explicit and bounded**: the resident set is exactly
  `3 * block_bytes`, so the layers, the pin and the arena are subsumed by the
  ring's order.
- **The gate (resolved).** `cli/09` requires `--prefetch-threads` with
  `--residency`; `--prefetch-threads` alone is the readahead-only mode
  (madvise, no pillars) -- the `b25630b` behavior. Both flags absent => the plain
  mmap stream.
- ~~**Compressed images (`runtime/16`).** ...~~ **Discarded:** the blosc2 image
  showed no expressive gain on the 72B (see `runtime/16`); the compressed path
  was removed, so the ring only sees uncompressed `.ztef` ranges.
- **The row cache.** `row_cache_bytes` (the embedding row cache) is kept, but its
  sizing input was `window_bytes` (removed here); re-source it from `--residency`
  or the RAM window.
- **Supersedes.** This spec removes the multi-layer residency and replaces the
  first ring attempt; that discard is the lesson, not the target.
- **Open (encoding).** The exact plan encoding (a `block_bytes` + `k`, or a
  derived `pillar_bytes` + a per-range block index) and whether a block is a
  fixed byte count (tensors packed) or a run of whole tensors (no split).

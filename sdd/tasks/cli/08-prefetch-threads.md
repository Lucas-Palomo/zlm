08 - Prefetch threads (--prefetch-threads)

Status: done
Commits: (uncommitted)

## Goal

Separate the thread budget explicitly: `--threads N` is the **compute** pool
(never reduced by prefetch), and `--prefetch-threads M` adds **M dedicated**
residency-prefetch workers (the `madvise`/`WILLNEED` workers), **separate from
and additional to** the compute — the process runs `N + M` threads. Before this
the runtime had a **single** prefetch worker, decided automatically by the tuner
(`cost.decidePrefetch` → `Config.residency_workers` ∈ {0, 1}): there was no way
to ask for more, and `residency_workers = 2` reserved a thread that did nothing
because the second worker did not exist.

The user picks the prefetch count; the tuner records it; the runtime runs **M**
prefetch workers alongside the N compute threads.

## Scope

- **In:** the `--prefetch-threads M` flag (M ≥ 1; absent = 0) and the `--threads
  N` lower bound (N ≥ 1); `Config.prefetch_threads` (the plan records M; the
  compute stays N); the tuner's thread accounting with M; the runtime running
  **M** prefetch workers.
- **Out:** changing the prefetch **policy** (what to prefetch — that is
  `runtime/06`/`gpu/13`); GPU/device threads.

## Design

### The CLI

```
zlm tune <image> <alias> --threads N [--prefetch-threads M] --ram ...
```

- `--threads N`: N ≥ 1, the **compute** threads. It **never** reserves a thread
  for prefetch — the tuner forces **all N inline** (the residency `madvise` runs
  on the executor's dead pool time).
- `--prefetch-threads M`: M ≥ 1, the **dedicated prefetch** workers — **separate
  from and additional to** the compute. **Absent = 0** → no prefetch worker
  (inline).

The compute = `N`; the prefetch = `M`; the process has `N + M` threads. The
prefetch is **opt-in** via the flag; the tuner's automatic `cost.decidePrefetch`
is **removed**.

### The plan

`Config` records `threads` (N) and `prefetch_threads` (M) — replacing
`residency_workers` (a 0/1 flag) with an explicit count. The runtime reads both
and decides nothing. (`.zplan` version bump.)

The tuner's `cost.decidePrefetch` is **removed**: `prefetch_threads` is exactly
the flag's value (0 when absent). The prefetch is opt-in.

### The runtime (a prefetch pool)

`Session` spawns a **prefetch pool** of **M** workers (today a single `Prefetch`),
**additional to** the N compute threads. Each worker gets its own CPU affinity
(the plan's affinity list has `N + M` entries: the first N for the compute, the
last M for the prefetch). The residency graph's prefetch nodes go to a **shared
queue** drained by the M workers (an MPMC ring), so the M threads pull ranges in
parallel. The residency accounting (the window) is unchanged — the workers only
issue the I/O. `M = 0` → no pool, all N inline.

### The tuner

The thread accounting (`selectParts`, the matvec `parts`, the residency window)
uses `compute = N` — unchanged by `--prefetch-threads`. The M prefetch workers are
**extra** threads (the process runs `N + M`); the user adds I/O parallelism
without shrinking the compute pool.

## Steps

1. **CLI** (`cli/commands/tune.zig`, `cli/root.zig`): `--prefetch-threads M`
   parsing (integer ≥ 1; absent = 0; explicit `0` and negatives rejected) and
   the `--threads N ≥ 1` validation; `help_text` and the registry `usage`
   updated.
2. **Plan** (`plan/root.zig`): `Config.residency_workers` (a 0/1 flag) renamed
   `Config.prefetch_threads` (the count M, a `u8`); serialize/deserialize and
   `fingerprint` updated (the count is now a user knob, so it names the plan);
   `VERSION` 27 → 28.
3. **Tuner** (`cli/commands/tune.zig`, `tuner/cost.zig`): `cost.decidePrefetch`
   (and its test) removed; `cfg.prefetch_threads = M` exactly (0 when absent);
   the thread accounting uses `compute = N` — all `--threads` go to the matvec
   `parts`, the attention selection and the projection/prefill selection; the
   affinity plan is `affinityPlan(N + M)`.
4. **Runtime** (`runtime/prefetch.zig`, `runtime/session.zig`): `Prefetch`
   became a **pool** of M workers over the one shared ring (`init` takes a
   count + the prefetch CPU list; `deinit` joins them all); `Session` spawns it
   when the plan records M > 0 and the graph is non-empty, pinning worker k to
   `affinity[N + k]`. The compute pool keeps `compute_workers = N` and its
   helpers use `affinity[1..N]` so they never take a prefetch CPU. `M = 0` → no
   pool, all inline.
5. **Verify:** `zig build` / `zig build test` green (299/299); measured on the
   disk-bound 72B (`--threads 9 --ram 16gb --gpu --vram 6gb`): 9+0 vs 9+1 vs
   9+2.

Measured (decode seconds for 5 tokens, 3 runs each; real tok/s = 5 / seconds):

| config | decode runs (s) | mean | tok/s |
| --- | --- | --- | --- |
| 9 + 0 | 75.7, 82.0, 78.0 | 78.6 | 0.064 |
| 9 + 1 | 74.4, 80.3, 74.5 | 76.4 | 0.065 |
| 9 + 2 | 74.7, 73.2, 77.5 | 75.1 | 0.067 |

The differences are small (~2-4%) and inside run-to-run noise; the dedicated
workers are consistently a little ahead of inline and 9 + 2 was the fastest
mean. The disk is the bottleneck (30 GB of streamed weights over a 16 GB
budget), so the `madvise` walk is not on the critical path.

## Definition of Done

- [x] `--prefetch-threads M` (M ≥ 1; absent = 0) and `--threads N` (N ≥ 1).
- [x] The plan records (N, M); the compute stays N; the tuner uses it.
- [x] The runtime runs **M** prefetch workers (not one).
- [x] Measured on the 72B; the 9 + 2 result is reported honestly.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the explicit prefetch/compute split.
- **derived:** `runtime/06` (the residency prefetch), `gpu/13` (the warm-up), the
  `cost.decidePrefetch` rule.

## Comments

- **Why explicit.** The tuner's `decidePrefetch` (the `madvise` cost vs a compute
  thread) is a heuristic; on a disk-bound model it picked `inline` (0) but the
  8 + 1 split measured ~4% faster and more stable — the heuristic under-estimates
  the I/O benefit. An explicit count lets the user tune it.
- **`residency_workers` → `prefetch_threads`.** The 0/1 flag becomes an explicit
  count; an old plan re-tunes (the version bump).
- **Decided (absent = 0 / inline).** Without `--prefetch-threads` the prefetch is
  **disabled** and the runtime is always **inline**; `decidePrefetch` is removed.
  The prefetch is **opt-in**.
- **Decided (`prefetch_threads` + a pool).** The field is renamed
  `residency_workers` → `prefetch_threads`; the runtime runs a **prefetch pool**
  of M workers (not M ad-hoc instances), draining a shared range queue.

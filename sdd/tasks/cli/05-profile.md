05 - Execution Profile (`run --profile`)

Status: done
Commits: 49d5f25

## Goal

Give `run` a **`--profile[=all|decode|prefill|load]`** flag that prints a
**per-op breakdown** of the decode (and prefill), plus hotspots, so the decode
bottleneck is visible without a profiler. It extends the existing
`--metrics`/`--metrics=full` (phase timing) with a **per-AtomicOp** accounting.

Diagnostic only: like `--metrics`/`--verbose`, it never feeds a decision — the
AOT invariant holds (the plan is unchanged; the profile is measured at run).

## Context

- `run` already has `--metrics[=full]` (TTFT/prefill/decode timings, peak RSS,
  per-phase trace) and `--verbose` (load/prefill/decode to stderr); the runtime
  reply carries `service.Metrics` (`prefill_ms`, `decode_ms`, `decode_tokens`,
  `prefill_tokens`, `cache_matched`).
- The decode is the executor's sequential loop over the plan's `Dispatch[]`
  (`executor.executeAt`: per dispatch it applies residency and calls
  `runDispatch`). Timing each `runDispatch` by `d.op` gives the breakdown.
- `kernels/11`/`12` closed the compute levers (width, FMA, deferred reduction);
  the remaining decode cost is expected to be **memory/residency**, so a per-op
  profile is the next step (not another kernel).

## Scope

- **In:**
  - The `Profile` accumulator (per-op ns + counts) in the runtime.
  - The `run --profile[=all|decode|prefill|load]` flag + the report.
  - The op buckets + hotspots.
- **Out:** a sampling/perf-event profiler; feeding the profile into the Tuner
  (it is run-time diagnostic only); changing the plan/Executor decisions.

## Design

### The flag

`run --profile[=all|decode|prefill|load]` (default `all` when the flag is bare).
Composes with `--metrics` (the phase timings are a subset). The profile is
reported from the service reply, so it works for a warm service (no re-load).

### The accumulator

A `Profile` struct (diagnostic; runtime-owned) with per-`op` `ns` + `count`:

```text
op buckets: matvec  matmul  attention  norm  activation  elementwise  rope
            dequant  other
overhead:   dispatch (bindings + args build, the loop itself)
            residency (res.advance / applyDispatch)
            sync (thread-pool barrier/join waits)
```

- `executor.executeAt` (and `executePrefill`) wraps `runDispatch` with
  `nowNs()` and accumulates into the bucket of `d.op`.
- `dispatch` = the loop body minus the timed `runDispatch`; `residency` = the
  `res.advance`/`applyDispatch` calls; `sync` = the time a `Dispatch` waits on
  the pool (reported by the Session when it joins). `other` catches the rest.

### The report

The user's format, e.g.:

```text
ZLM Execution Profile
──────────────────────────────────────────
Model:  gemma-31b   Runtime: rt-g31   Threads: 8   Profile: all

PHASE      load  XX.XX ms | prefill  XX.XX ms | decode  XX.XX ms | total  XX.XX ms

DECODE BREAKDOWN (per token)
  MatVec            XX.XX ms
  MatMul            XX.XX ms
  Attention / KV    XX.XX ms
  Norm / Activation XX.XX ms
  Other AtomicOps   XX.XX ms
  Dispatch overhead XX.XX ms
  Synchronization   XX.XX ms

HOTSPOTS
  1. matvec_q4_0   XX.X%
  2. attention     XX.X%
  ...

RUNTIME METRICS
  Dispatches/token:  XXXX
  Decode latency:    XX.XX ms/token
  Generation:        XX.XX tok/s
```

Only the requested phase is broken down (`--profile=decode` omits prefill/load).
The hotspot uses the family `id` (e.g. `matvec_q4_0`) when available, else the
`op` tag.

### Invariants

- **Diagnostic only:** the profile is measured at run and never enters the plan
  or a decision; it does not change the output.
- **Cheap when off:** the accumulation is `nowNs()` per dispatch (a few thousand
  per token); with `--profile` absent, no timers run.
- **Determinism:** the output text is unchanged by `--profile` (unlike the
  precision/KV choices).

## Steps

1. `Profile` struct + per-`op` accumulation in `executor.executeAt`
   (`nowNs()` around `runDispatch`); the `dispatch`/`residency` split. [DONE]
2. Thread the profile through `session` → the service reply (`service.Metrics`
   or a sibling `Profile`) + the wire format. [DONE]
3. `run --profile[=...]` + the report (phase + decode breakdown + hotspots +
   runtime metrics). [DONE]
4. `sync` accounting from the Session's pool join. [DONE — descoped: superseded
   by the serial/parts thread-split diagnostics (`2aa91a2`).]
5. `prefill`/`load` breakdowns. [DONE — phase-level only (load/prefill/decode
   timings); no per-op prefill/load breakdown.]
6. `zig build` / `zig build test` green; the profile is deterministic (same text
   with/without `--profile`). [DONE]

## Definition of Done

- [x] `run --profile[=all|decode|prefill|load]` prints the phase + per-op
      breakdown + hotspots + runtime metrics.
- [x] The per-op accounting covers the executor's op buckets + residency (the
      planned `sync` bucket was superseded by the serial/parts thread-split
      diagnostics, `2aa91a2`).
- [x] The profile is **diagnostic only**: the output text and the plan are
      unchanged with/without the flag.
- [x] No new decisions; the AOT invariant holds.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the per-op profile, the buckets, the report.
- **spec:** the existing `--metrics`/`--verbose` (cli/04) and the executor's
  sequential `Dispatch[]` loop.
- **consulted:** generic wall-clock profilers / `perf` — reference only (this is
  an in-process, AOT-safe accounting, not sampling).
- **derived:** `runtime/12` (the cost pipeline — the *tune-time* analog),
  `kernels/11`/`12` (the compute levers closed; the profile finds the next).

## As-built + measured

Implemented as specified. Measured (`e2b-test`, `--threads 8`, 96 tokens):

```
DECODE BREAKDOWN (per token)
  MatVec             55.66 ms   (95.2%)
  Attention / KV      1.78 ms   ( 3.1%)
  Norm/Activation/Elementwise/Rope/Dequant  ~0.7 ms
  Residency           0.34 ms
  Dispatch overhead   (decode_total - ops - residency) ~0.5 ms
Decode latency: 58.65 ms/token -> 17.0 tok/s
```

**Finding:** the decode is **~95% matvec** and the matvec runs at **~41 GB/s**
(2.3 GB of weights / 55.66 ms), near the **~46 GB/s DRAM ceiling** — i.e. the
decode is **memory-bandwidth-bound**, not compute-bound. The compute levers
(`kernels/11`/`12`) are at the ceiling; the next lever is **memory** (denser
weight quant -> fewer bytes streamed), not another kernel.

**Follow-up (decode optimization):** with the profile as the guide, the DRAM
read ceiling was measured at **~37 GB/s** (a streaming C loop; saturates at 4
threads) and the matvec runs at **~28 GB/s** -> the decode is **compute-bound**
(IPC 2.02, cache-miss 1.76%), not bandwidth-bound. First fix: `dotQ4_K`
recomputed the row-independent `sum(x)` per 32-chunk for every output row ->
`q4KXSum` + `dotQ4_KPre` (commit `9154705`), decode 58.5 -> 53.7 ms/token
(17.1 -> 18.6 tok/s), bandwidth 24.9 -> 27.8 GB/s. Remaining hotspots: the q4_0
nibble unpack (`q4_0BlockVec`, ~49% across matvec + fused gate/up) and the
fused gate/up reloading the shared activation twice per block.

**Thread split (follow-up):** `parts = clamp(rows*cols / 1_000_000, 1, threads,
rows)` (`tuner/compile.zig`). The heavy matvecs already get the max: gate/up
(12288), down (12288), lm_head (262144) -> **8**; q/o (2048x1536) -> 3-6; the
~200 small projections (256x1536 x118, 1536x256 x70, 512x1536 x12) -> 1. Serial
dispatches are **67.7% of dispatches but only 5.9% of the time**. A clean
`perf stat` shows **6.77/8 threads busy** (the earlier 5.3 sampled idle time).
=> the thread split is near-optimal; lowering the grain (250k) was slower, so
the remaining gap to the DRAM ceiling is kernel efficiency (IPC 1.55), not
parallelism.

**Compute levers exhausted (follow-up):** two kernel hypotheses were tested
with interleaved `--metrics` A/B (never `--profile`) and both were refuted:
(1) a **Q4_Q8 integer dot** (q8 activation + `vpmaddwd`) is ~1.45x faster in
isolation but ~2% *slower* e2e; (2) **multi-accumulator ILP** (2/4/8 chains)
gains ~5% in isolation and nothing e2e (the OOO engine already overlaps blocks;
A=8 spills on the fused gate/up). Both stay as tested references? No — the
multi-acc variants were removed; `Q4_F32` (production) and `Q4_Q8` (integer,
unwired reference) are kept. Conclusion: the decode is **DRAM-bandwidth-bound**
(~30 GB/s aggregate; single-thread tops out ~7.6 GB/s), so the only remaining
lever is **fewer bytes/token** (denser weight quant), not the kernel.

**Bug found:** `tune` reused a stale `.zplan` (v11) after the format bump to v12
(`plan.VERSION`), so the daemon failed with `BadVersion`; the reuse check should
compare the on-disk plan version.

## Comments

- **Why a flag, not `perf`:** the decode is a few thousand dispatches over
  mmap'd weights; an in-process per-op accounting is deterministic and needs no
  symbols/permissions.
- **`--metrics` vs `--profile`:** `--metrics` = the phase timings (TTFT/prefill/
  decode); `--profile` = the per-op breakdown. `--profile` may later subsume
  `--metrics=full`.
- **This is investigation, not a decision:** the profile informs the *next*
  task (memory/residency), it does not change the Tuner.

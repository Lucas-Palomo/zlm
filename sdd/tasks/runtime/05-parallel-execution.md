05 - Parallel Execution

Status: done
Commits:

## Goal

The plan's `parts` actually parallelizes execution. The **tuner decides** how
many parts each dispatch may use; the **runtime supplies** the workers; the
**executor stays sequential** and only splits work *inside* a dispatch.

## Scope

- **In:** a per-dispatch `parts` decision in the tuner (op class + shape +
  budget + grain); a persistent worker pool in the runtime; row-range split for
  `matvec`/`lm_head`.
- **Out:** GPU; async residency/prefetch; CPU affinity/pinning (later);
  head/attention parallelism; changing the `.zplan` format (the `parts` field
  already exists).

## Design

Today `tuner/compile.zig` stamps `.parts = opts.threads` on **every**
instruction: `--threads` is a hardware *budget*, not a per-op decision. A 1536-wide
norm does not benefit from 4 threads; a 12288-row matvec does.

- **Roles (Tuner).** Per `docs/kernel-architecture.md`, the Tuner is three
  roles: the **Selector** chooses the `AtomicOp` (`atomic_id`); the **Composer**
  chooses the form (`single | fused | seq`); the **Scheduler** plans order,
  placement, residency **and parallelism (`parts`)**.
- **Axis is AtomicOp metadata (Registry).** The decomposition axis is scheduling
  metadata carried by the family/atomic (the doc's `api.Kernel`: `row_parallel`,
  later a `parallel_axis`), read by the **Scheduler** through the Registry — not
  a hardcoded list in the tuner. Param-dependent cases are already disambiguated
  by the Selector's `atomic_id`: `qk_norm_heads` is head-parallel, `rms_norm` is
  not. So `axis` follows the chosen atomic:

  | selected atomic | axis |
  | --- | --- |
  | `matvec.*`, `lm_head` | rows (output features) |
  | `fused_flash_head` | heads (query heads) |
  | `qk_norm_heads`, `rms_norm_no_weight_heads` | heads |
  | `rms_norm`, `rms_norm_no_weight`, `rope`, `gated_*`, `elementwise` | none |

- **Degree is the Scheduler.** For a materializable axis,
  `parts = clamp(units / grain, 1, budget)`; `none` => `parts = 1`.
  `units` = rows `m` (from `LayerDims`) for `rows`, or the head count for
  `heads`; `budget = Config.threads`; `grain` is a coordination-overhead
  threshold (first cut: a constant; later measured at `tune`). The Scheduler
  writes `Dispatch.parts`; the runtime reads it and decides nothing.
- **Materialization (runtime capacity).** The first cut implements only the
  `rows` axis (`matvec`/`lm_head`). The tuner resolves `heads` too, but records
  `parts = 1` for axes the runtime cannot execute yet, so the plan never asks
  for a split that cannot happen.
- **Composer boundary (out of scope here).** The Composer regroups adjacent
  model ops into a dispatch *form* (`single | fused | seq`) using the recipe
  catalog and the `fusable` metadata; it decides **which model ops become one
  dispatch**, not how that dispatch is implemented. It does not choose the
  hardware variant (Selector) nor order/place/thread (Scheduler). Task 05 leaves
  the block all-`single` (Composer pass-through); this note only fixes the
  boundary so fusion never migrates into the Selector or the Scheduler.

Phase order: **Composer (group) -> Selector (variant per dispatch) -> Scheduler
(order/placement/`parts`/residency)**. The three read the same catalog
(`api.Kernel`): the Composer reads `fusable`, the Selector the implementation
identity, the Scheduler `row_parallel`/`scratch`. The catalog (`registry.zig`)
currently exposes only `Family`; `api.Kernel` already declares
`row_parallel`/`fusable`/`scratch` but is not wired — task 05 wires only what the
Scheduler needs.
- **Resources (runtime).** The `Session` owns a persistent pool of
  `budget - 1` workers plus one job slot (mutex + condition). Created lazily on
  the first `parts > 1`; joined in `deinit`. The plan only describes sizes
  (buffers, KV) — threads are execution resources the runtime must materialize,
  exactly like buffers.
- **Execution (executor).** Sequential over dispatches. For a `matvec`/`lm_head`
  dispatch with `parts > 1`, split `[0, m)` into `min(parts, m, workers + 1)`
  equal ranges; the calling thread runs one range and signals the helpers for
  the rest, then joins. `matvec`/`lm_head` already accept `begin`/`end`
  (`kernels/linear/matvec/root.zig`), so no kernel change is required.
- **Determinism.** Matvec rows are independent (no reassociation), so the split
  does not change results.
- `TensorView.knee_threads` (data model) is currently unused; the knee is a
  tuner concern and stays out of the runtime.

## Flow / Arch

```text
AoT — tune (Tuner = Selector + Composer + Scheduler; no threads here)

  block instr
    ├─ resolveParams → params (norm kind, attention n_heads)
    └─ layerDims     → dims  (q/kv/ffn/hidden/vocab)

  Selector  : (op, params, cfg) ─▶ atomic_id       (registry.find + policy)
  Composer  : atomic_id         ─▶ form            (single | fused | seq)
  Registry  : atomic_id         ─▶ scheduling meta (row_parallel, scratch, ...)
  Scheduler : meta.axis + dims + budget + grain
                 ├─ rows  → units = m       → parts = clamp(units/grain, 1, budget)
                 ├─ heads → units = n_heads → (resolved; parts = 1 this cut)
                 └─ none  → parts = 1
              + order / placement / residency
  ──▶ Dispatch{ op, atomic_id, params, in_len/out_len, parts }  ──▶ .zplan

Run — runtime (resources; executor stays sequential)
  Session.open: budget = Config.threads
                pool = budget-1 workers (created lazily on first parts > 1)
  executeAt(pos):
    for d in Dispatch[]:
      if d.parts > 1 and d.op is matvec-family:
          k = min(d.parts, m, workers+1)
          split [0, m) into k ranges
          main thread runs range 0, helpers run the rest, join
      else:
          run once
```

Concrete trace (gemma4 E2B, sliding layer, `budget = 4`):

| instr | axis | units | parts |
| --- | --- | --- | --- |
| `attn_norm` (norm_rms) | none | hidden 1536 | 1 |
| `attn_q` matvec | rows | q = 8×256 = 2048 | per grain |
| `attn_k` / `attn_v` matvec | rows | kv = 1×256 = 256 | 1 (below grain) |
| `attn_q_norm` / `v_norm` | heads | 8 | 1 this cut (axis resolved) |
| `attention` | heads | 8 | 1 this cut |
| `ffn_gate` / `ffn_up` / `ffn_down` | rows | ffn / hidden | per grain |
| `lm_head` | rows | vocab | per grain |

The narrow K/V projections are the example of "rows but 1": independent rows,
yet too little work per part to pay for coordination.

## Arch Provenance

- **own:** the AoT split — tuner decides `parts`, runtime supplies workers.
- **spec:** —.
- **consulted:** the earlier local prototype's `src/execution/root.zig`
  (`ExecutionContext`, `partsForWithCap`, `runJob`) — reference only, not
  copied.
- **derived:** from CONSTITUTION "Parallelism is a plan property. The Dispatch
  declares it, the Runtime supplies the resources, and the Executor stays
  sequential."

## Steps

1. Tuner: `partsFor(op, dims, instr, budget)`; `emit` uses it instead of
   `opts.threads`. Test: `matvec`/`lm_head` get `> 1` for large shapes; small
   and non-row ops get `1`.
   - **As-built:** `api.Family.row_parallel` (+ `registry.rowParallel`);
     `tuner/compile.zig` `grain_work = 1_000_000` and `partsFor(...)`; `emit`
     sets `.parts = partsFor(instr, dims, opts.threads)`. With budget 4: ffn
     gate/up/down → 4, q proj → 3, k/v proj → 1 (below grain), norms/rope/
     attention → 1. Tests added; the old fixed `parts == 8` assertion updated
     to `1` (no dims).
2. Runtime: worker pool in `Session` (mutex/condition/job), lazy init, deinit.
   - **As-built:** new `src/runtime/pool.zig` (ported from the earlier local prototype
     `ExecutionContext`: `Sync` + `helperMain` + `rangeOf` + `runJob`,
     `Io.Mutex`/`Condition`); `Session.pool: ?Pool`, lazily created on the first
     `parts > 1` from `plan.config.threads`, joined in `deinit` before the
     buffers.
3. Executor: `runMatvec` splits by `d.parts` into row ranges and dispatches.
   - **As-built:** `Session.matVecPlan(view, x, out, rows, parts, atomic_id)`
     (fallback single-thread when no pool); `executor.runMatvec` delegates to it.
     The executor stays sequential; only the matvec-family row loop is split.
4. `zig build` / `zig build test` green; re-tune and measure e2b run time.
   - **As-built:** build/test green (64/64). Re-tuned `tune e2b-test rt-e2b
     --threads 4`; `run rt-e2b --prompt hello --max-tokens 2`:
     **5:12.77 (99% CPU) → 1:41.22 (339% CPU)** — ~3.1× on 4 threads (16-core
     host).
5. (later) measured `grain` via a tune-time bench; CPU affinity; `heads` axis.

## Definition of Done

- [x] Tuner emits per-dispatch `parts` (only `matvec`/`lm_head`; by grain/budget).
- [x] Runtime pool materializes `parts`; the executor stays sequential.
- [x] Row split is deterministic; `zig build` / `zig build test` green.
- [x] e2b `run` is measurably faster (5:12 → 1:41, ~3.1×).

## Comments

- First cut uses a heuristic `grain`; a measured knee at `tune` is the follow-up.
- Parallelism is recorded in the plan (`parts`), so a plan tuned with `--threads
  4` and run on a 4-worker runtime behaves as tuned; the runtime may cap below
  `parts` if resources are fewer, but never decides to add more.

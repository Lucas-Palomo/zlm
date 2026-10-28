09 - Residency Ceiling (--residency)

Status: done
Commits: (uncommitted)

## Goal

Give the resident weight working set an **explicit, user-set ceiling**. The
tuner derives the window from the RAM left after the KV/scratch
(`ram - fixed - reserve`), and the runtime's default `warm` state applied the
plan's window only in `hot`: in `warm` the residency budget was `0`, so the
`madvise(WILLNEED)` readahead ran **unaccounted** and the page cache grew to the
whole image. On a model that does not fit this is an OOM (observed on the 72B:
RSS 24 GB on a 31 GB host, no swap).

`--residency <mb|gb>` makes the resident set a **deliberate, bounded** choice:
the ceiling for the window plus the staging arena. It is **opt-in**; absent = no
residency machinery (every weight streams from the mmap). The default `warm`
state is bounded by the plan's window, so the unbounded `WILLNEED` is gone.

## Scope

- **In:** the `tune --residency <mb|gb>` flag (opt-in; absent = 0); the plan's
  `Config.residency_bytes`; the validation (`< --ram`; `--residency` requires
  `--prefetch-threads`, the KV-floor clamp); the split of
  the tuner's weight budget into
  "RAM reserved for the weights" (KV sizing) and "residency window"
  (pin/prefetch); the runtime `warm` bound.
- **Out:** changing the residency **policy** itself (what to pin, the lookahead,
  the arena) -- that is `runtime/06`/`15`/`16`/`17`; a live residency control
  (`zlm state`) -- future work in `sdd/roadmap.md`; the KV precision.

## Design

### The CLI

```
zlm tune <image> <alias> --threads N --ram <mb|gb> [--residency <mb|gb>] [--prefetch-threads M]
```

- `--residency <mb|gb>`: the explicit ceiling for the resident weight working
  set (the window plus the staging arena). **Opt-in:** absent = 0 = **no
  residency** (every weight streams on demand from the image mmap; no window, no
  arena, no prefetch, no graph).
- Must be **strictly smaller than `--ram`** (it has to leave room for the
  KV/scratch); `>= --ram` is rejected.
- `--residency` **requires** `--prefetch-threads` (the ring needs the workers to
  fill the pillars). `--prefetch-threads` alone is allowed: it is the
  readahead-only mode (the workers issue `madvise(WILLNEED)`, no ring buffer).
- A value above the **KV-floor ceiling** (`pre_kv - kv_floor`) is **clamped** to
  it with a warning: the ceiling is the RAM left after the KV floor, the slots
  and the scratch.

### The plan

`Config.residency_bytes` records the requested ceiling (bytes; `0` = no
residency). It enters the fingerprint and the serialization. `VERSION` 29 -> 30
(an old plan re-tunes).

### The tuner (two budgets)

The weight budget splits into two:

- **`weight_budget`** -- the RAM **reserved for the weights**, used only to size
  the KV (`planBudget`). It is what keeps the KV from absorbing the RAM that
  should back the mmap.
- **`window_budget`** -- the **residency window/pin** budget (`hot_target`,
  `reserved_bytes`, `admit`, `window_bytes`).

Without `--residency`:

- `weight_budget = weight_ceiling` -- the whole ceiling, the **pre-flag
  behavior**: the KV keeps its floor and the rest backs the **mmap page cache**,
  so the mmap gets the maximum.
- `window_budget = 0` -- no window, no pins, no prefetch, no `ResidencyGraph`
  nodes (the graph is empty).

With `--residency`:

- `weight_budget = window_budget = min(weight_ceiling, residency_ceiling)`.

### The runtime

- `Session` creates the `Residency` only when the graph has nodes (`has_graph`);
  a plan with no residency has `nodes == 0`, so **no `Residency`, no arena, no
  window, no prefetch pool**.
- `Residency.setState(state, plan_window)`: `warm` uses the plan's window but
  `pin_hot = false`, so it is a **bounded LRU** -- the streamed readahead can
  never exceed the window. `hot` re-enables the pin; `cold` disables the
  prefetch and the window. The old default (`budget = 0` -> `WILLNEED` with no
  accounting) is gone.
- The service's default state is `warm` (`service.Options.state = .warm`);
  `setResidencyState` is a no-op on a plan with no residency.

## Flow / Arch

```text
tune --residency R:
  weight_ceiling = min(pin_weights, pre_kv - kv_floor)
  weight_budget  = R == 0 ? weight_ceiling : min(weight_ceiling, R)   # KV sizing
  window_budget  = R == 0 ? 0            : weight_budget              # pin/prefetch
  window_bytes   = pinned + reserved                                  # 0 when R == 0
  nodes          = R == 0 ? 0 : ...                                   # the graph is empty

run:
  has_graph ? Residency.init(budget = window_bytes) : none
  warm -> budget = window_bytes, pin_hot = false   (bounded LRU)
  hot  -> budget = window_bytes, pin_hot = true    (the HOT set pinned)
```

## Steps

1. **CLI** (`cli/commands/tune.zig`): `--residency <mb|gb>` parsing; the
   `< --ram` validation; `--residency` ⇄ `--prefetch-threads` mutual
   requirement; `help_text` + the registry `usage`.
2. **Plan** (`plan/root.zig`): `Config.residency_bytes` (a `u64`);
   serialize/deserialize + `fingerprint`; `VERSION` 29 -> 30.
3. **Tuner** (`cli/commands/tune.zig`): the two budgets (`weight_budget` for
   `planBudget`, `window_budget` for `hot_target`/`reserved_bytes`/`admit`/
   `window_bytes`); the KV-floor clamp with the warning; the node-building guard
   (`residency_bytes > 0`).
4. **Runtime** (`runtime/residency.zig`, `runtime/session.zig`): `pin_hot` (the
   `warm` bound), `setState(state, plan_window)`, `isPinned`.
5. **Tests:** the two budgets (a model that does not fit: `window_budget = 0`
   still reserves the weight ceiling for the KV); the clamp; the `warm` bound.
6. **Verify:** `zig build` / `zig build test` green (310 pass, 1 skip); the tune
   end-to-end (below).

### As-built (qwen3-72b-blosc, `--threads 8 --ram 16gb`, ReleaseFast)

| invocation | residency | fixed | pinned | window | nodes | context |
| --- | --- | --- | --- | --- | --- | --- |
| (no flag) | 0 MB | 1297 MB | 0 MB | 0 MB | 0 | 4096 |
| `--residency 6gb --prefetch-threads 2` | 6144 MB | 10239 MB | 1525 MB | 6133 MB | 1410 | 32711 |
| `--residency 15gb --prefetch-threads 2` | 15360 MB | 1297 MB | 3736 MB | 15050 MB | 1410 | 4096 |

- No flag: `window 0`, `nodes 0`, no graph; `fixed` is the KV floor (the pre-flag
  budget) -- the freed RAM backs the mmap, not the KV (before this task `fixed`
  grew to 12817 MB).
- The clamp fires: `warning: --residency 15360 MB exceeds the 15086 MB ceiling
  (the RAM left for KV/compute/scratch); residency adjusted to 15086 MB`.
- `arena 0 MB` on this image because it is a compressed image (`runtime/16`,
  since discarded); the ceiling still covers the window.
- Error cases: `--residency` without `--prefetch-threads` -> "requires
  `--prefetch-threads`"; `--residency 16gb --ram 16gb` -> "must be smaller than
  `--ram`". `--prefetch-threads` alone passes validation (madvise-only).

## Definition of Done

- [x] `--residency <mb|gb>` (opt-in; absent = 0); must be `< --ram`;
      `--residency` requires `--prefetch-threads`; `--prefetch-threads` alone is
      the readahead-only mode.
- [x] The plan records `Config.residency_bytes` (VERSION 30); the fingerprint
      names the plan.
- [x] A `--residency` above the KV-floor ceiling is clamped with a warning.
- [x] Absent = no residency: `window 0`, `arena 0`, no nodes; the RAM stays
      reserved for the weights (the mmap page cache), the pre-flag budget.
- [x] `warm` is bounded by the plan's window; the unbounded `WILLNEED` is gone.
- [x] `zig build` / `zig build test` green (310 pass, 1 skip).

## Arch Provenance

- **own:** the explicit, opt-in residency ceiling; the "no residency = the
  pre-flag budget, the mmap gets the maximum" rule; the `warm` bound.
- **derived:** from `runtime/06` (the computed budget, the residency engine) and
  `cli/08` (the thread split); the hot/reserved layers, lookahead/pin and arena
  it derived from are removed by `runtime/15` (the ring).

## Comments

- **Why a ceiling.** The tuner's derived window (`ram - fixed - reserve`) is an
  upper bound, not a promise: on a model that does not fit it is ~15 GB, and the
  runtime's default `warm` issued `WILLNEED` for every streamed range with no
  accounting, so the page cache grew to the whole image (the OOM). An explicit
  ceiling makes the resident set deliberate.
- **Naming.** `--residency` over `--buffer`/`--resident-weights`/`--weight-cache`/
  `--weight-buffer`: it names the residency the runtime already models
  (`ResidencyGraph`, `Residency`), and it is the plan's own term.
- **Decided (opt-in, absent = no residency).** Without the flag there is **no**
  residency machinery: no window, no arena, no graph. The tuner still **reserves**
  the weight ceiling from the RAM (the KV keeps its floor), so the freed RAM
  backs the **mmap page cache** -- exactly the pre-flag behavior (mmap-first,
  `CONSTITUTION.md`).
- **Decided (`--residency` requires `--prefetch-threads`).** The ring needs the
  workers to fill its pillars; `--prefetch-threads` alone is the readahead-only
  mode (madvise, no ring), the `b25630b` behavior (`cli/08`).
- **Decided (the `warm` bound).** `warm` keeps the plan's window but pins
  nothing, so it is a bounded LRU; `hot` is the only state that pins. This is
  what removes the unbounded `WILLNEED` (`runtime/06`/`08`).
- **Uncommitted.** The work is verified but not committed (the plan format is
  v30).

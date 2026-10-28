15 - q4_0 Catalog: Dynamic Grid + Pruning

Status: done
Commits:

## Goal

Generate the q4_0 catalog **dynamically** at comptime from a five-axis grid —
`width` x `tile` x `unroll` x `int` x `prefetch` — so the catalog is one grid
definition (160 combinations) and the Selector/calibration picks. To keep the
benchmark bounded, the **plan prunes** the combinations that cannot win on the
current CPU *before* measuring, instead of generating/testing all 160.

## The grid (160 combinations)

```
width    ∈ { 4, 8, 16 }            // f32 lanes (or i32 lanes for `int`)
tile     ∈ { 1, 2, 4, 8, 16 }      // output rows per block sweep
unroll   ∈ { 1, 2, 4, 8, 16 }      // blocks per iteration
int      ∈ { 0, 1 }                // f32 dot vs the integer (q8-activation) dot
prefetch ∈ { 0, 1 }                // @prefetch the next blocks
budget   = tile * unroll * ceil(width/8) ≤ 16   // AVX2: 16 YMM accumulators
```
Base `(w,t,un)` = 40 (15 w4 + 15 w8 + 10 w16); x `int` (2) x `prefetch` (2) =
**160**.

## Names

`q4_0_w{W}t{T}un{U}_i{I}p{P}` — e.g. `q4_0_w8t2un4_i1p0`,
`q4_0_w8t2un4_i0p1`, `q4_0_w4t1un1_i0p0`.

## Pruning (before the benchmark)

The grid is the **definition**; the plan derives the **effective** candidate set
from the device caps, so the benchmark never sees a combination that cannot win.

### P1 — ISA prune, at **comptime** (the `int` width + the f32 width)

`simd.native` (`kernels/simd.zig`) already resolves the build target's native
width at comptime from `builtin.cpu.features` (the **`-Dcpu`** target): AVX-512
-> 16, AVX2 -> 8, else 4. Use it to **not generate** the widths that cannot win:

- `int = 1` (the integer dot) is generated only at `width == simd.native` — the
  integer path exists per ISA width (`pmaddwd`: SSE2 4, AVX2 8, AVX-512 16); a
  narrower one uses half the register file, a wider one does not exist. On AVX2
  this prunes `i4`/`i16` **at compile time** (the user's example).
- The f32 widths **wider than `simd.native`** are pruned too: `simd.zig` states a
  width wider than the target's native width is lowered by *splitting*, which the
  Tuner measures and avoids — so do not generate it.

The prune is **exact** for a `-Dcpu=native` build (the project's perf build; the
build fingerprint guards staleness). A generic `-Dcpu` build would over-prune, so
the runtime `applicable(caps)` gate stays as a belt-and-suspenders.

### P2 — prefetch prune (a heuristic, not a measurement)

Prefetch only pays when the op **streams from RAM** and is **memory-bound** — if
the working set fits the last-level cache the hardware prefetcher already covers
the sequential access. So the plan decides **before** the benchmark, from the
caps (cache size, bandwidth) + the roofline (`tuner/cost.zig`): if the op is not
a streaming, memory-bound case, the `p1` variants are pruned. The plan records
the outcome; the runtime re-decides nothing.

## Design

- One grid definition (the five value sets + the budget + the two prune rules).
  The catalog, the serialized ids (grid index, deterministic), the names, the
  parse/label, the registry kernels, the calibration walker and
  `bench.candidatesFor` are all generated at comptime.
- The two prunes run **before** generation/benchmark: **P1 is comptime** (the
  ISA, from `simd.native`); **P2 is a tune-time heuristic** (caps + roofline).
- The variant's `width` **replaces** the separate `Dispatch.simd_width`.
- `int` is an **algorithm** axis: it swaps the decode (f32 `q4_0BlockVec` -> i32
  `q4_0IntBlockVec`) and the meaning of `width` (f32 lanes -> i32 lanes), so the
  generator instantiates an **integer body per width**.
- `prefetch` is a **loop** axis (a `@prefetch` in the body).
- `VariantEfficiencies` sized to the catalog + the other quants (a `.zat` version
  bump + re-tune). The other quants keep their hand-written plans.
- **`scale` is out of scope** — it is a layout (the f16 repack), isolated and
  single, not a grid axis.

## Steps

1. **Done.** The five-axis grid + the generated catalog (list, ids, names,
   parse/label) in `types.zig` (`q4_0_grid_{widths,tiles,unrolls,prefetches}`,
   `q4_0WithinBudget`, the comptime `MatVecPlan`/`q4_0_candidates`).
2. **Done.** P1 (comptime: `q4_0IntGenerated`/`q4_0F32Generated` from
   `simd.native`) and P2 (`cost.prefetchWorthwhile` -> `SelectCtx.allow_prefetch`
   -> the Selector skips the `prefetch = 1` plans).
3. **Done.** `matVecQ4_0Gen` -> `matVecQ4_0GenF32` (with the `@prefetch`) /
   `matVecQ4_0GenInt` (the integer body per width).
4. **Done.** The registry dispatch (`matVecRowsWithPlan`), `planApplicable`, the
   calibration walker and `bench.candidatesFor` read the generated list.
5. **Done.** `VariantEfficiencies.max_variants` 32 -> 128; `.zat` VERSION 16.
6. **Done.** 265/265; parity for every generated candidate; the E2B selects
   `q4_0_w8t1un8_i0p0` (166 sites) and decodes ~15.8 (baseline); 12B ~3.7.

## Definition of Done

- [x] The q4_0 catalog is the generated five-axis grid (160 defined); no
      hand-listed entries/tags.
- [x] The **comptime** ISA prune does not generate the non-native `int` widths
      (i4/i16 on AVX2) nor the f32 widths wider than `simd.native`; the prefetch
      heuristic prunes `p1` when not streaming/memory-bound.
- [x] Names are `q4_0_w{W}t{T}un{U}_i{I}p{P}`; ids deterministic; the width folds
      in `simd_width`; the other quants unchanged.
- [x] The effective candidates are measured; the Selector picks; parity holds.
- [x] CPU decode measured; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the five-axis dynamic grid + the pre-benchmark pruning.
- **derived:** `kernels/13` + `kernels/14` (the generator + catalog),
  `kernels/11` (per-ISA width atomics), `kernels/10` (the integer dot).

## Comments

- **Flat on q4_0.** The q4_0 GEMV is memory-bound, so widening the catalog adds
  calibration cost without decode gain here; the value is generality (the `int`
  and `width` axes matter on compute-bound kernels).
- **The prunes are the point.** They encode the "don't benchmark what cannot
  win" rule the user asked for: the ISA prune is exact and **comptime** (from
  `simd.native`, the `-Dcpu` target), the prefetch prune is a tune-time heuristic
  (a roofline/caps decision, recorded in the plan).
- **As-built.** 160 defined -> **90 generated** after P1 on this AVX2 host (f32
  `w4`/`w8` x 15 in-budget `(t,u)` + `int` `w8` x 15, each x 2 prefetch); P2 then
  prunes the `prefetch = 1` plans at the Selector when the op is not a
  RAM-streaming, memory-bound case. The E2B selects `q4_0_w8t1un8_i0p0` (166
  sites); decode is flat (~15.8, baseline) — as expected for a memory-bound
  q4_0. `VariantEfficiencies.max_variants` 32 -> 128.

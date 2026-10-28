07 - Variant Cost Model

Status: done
Commits: 42eaadc

## Goal

The Selector does **not** choose the matvec/GEMM **variant** by cost today: the
`applicable` predicate hardcodes a shape rule (wide output -> tiled `_t2`/`_t4`,
narrow -> unroll `_u4`), and `cost` is identical across variants ("the variant
does not change the logical work"). So for a wide projection the only applicable
candidate is tiled, and **unroll never reaches the Selector**. The earlier local prototype measured each
plan at load and picked `unroll4`; the current code's static rule picks tiled and is slower.

Make the Selector **variant-aware**: rank all type-applicable variants by a cost
model that captures each variant's real efficiency, calibrated once per device —
without a per-candidate timing gate.

## Scope

- **In:**
  - `api.Kernel.cost` becomes **variant-aware** (per-variant efficiency), so
    `opCost` differs between u4/t2/t4 for the same logical bytes/flops.
  - `applicable` becomes **type/validity only** (drop the shape gate): all
    variants of a type are candidates.
  - **Measure** the per-variant efficiency coefficients once per device; cache
    them in the `.zat` (like bandwidth/latency).
  - The Selector ranks all type-applicable variants by `argmin opCost`.
- **Out:** the runtime (still executes the plan, decides nothing); GPU; recipes;
  the GEMM/attention families beyond the same mechanism.

## Design

### Why not the two extremes
- **Static rule** (today): no adaptation, and wrong here.
- **Per-candidate timing gate** (the earlier local prototype): rejected by `runtime/09` — "timing
  calibrates coefficients, never decides per candidate".

### Hybrid: variant-aware cost with calibrated coefficients
- Each candidate declares an **efficiency** `eff(variant, shape_class, lanes)`:
  the fraction of the tier bandwidth it reaches (or an equivalent per-variant
  constant). The bytes term becomes `bytes / eff`, so variants that move the same
  logical bytes but at different achieved bandwidth are ranked differently.
- `applicable` only checks type/variant **validity** (not shape); the cost ranks.
- **Measure** calibrates the efficiencies per device (a micro-bench per variant
  class, once) and caches them in the `.zat`; the decision stays a deterministic
  `argmin`, no per-candidate timing at decide time.

### Thread scaling (measure, don't assume)
The earlier local prototype measures a per-op **thread-scaling curve** (`effective_bw` at 1..8 threads)
and derives `knee_threads` (efficiency >= 0.75) and `saturation_threads`
(marginal gain < 5%); at run it caps `parts` by the measured knee
(`partsForWithCap`). The current code instead measures only single-thread bandwidth (~9.6 GB/s)
and assumes memory bandwidth does not scale (`cost.parEff = min(1, parts/2)`), so
it under-parallelizes (observed `parts` in {1,3,5} of 8). The variant cost should
use a **measured scaling curve** (knee/saturation) per op/variant, calibrated once
per device and cached in the `.zat`, so `selectParts` uses the right thread count.
Also consider a grain cap like the earlier local prototype's `min_rows_per_part` (the current code allows one
row per part).

### Flow
```text
Profiling   caps (ISA/cores/caches)
   ↓
Measure     coeffs (bw/latency) + per-variant efficiency + scaling curve -> .zat
   ↓
Selector    ALL type-applicable variants -> argmin opCost(variant-aware, parts)
   ↓
Composer    recipes
   ↓
Scheduler   order / parts / residency
```

## Steps

1. Add an efficiency input to the candidate cost (`api.Kernel` / the family's
   `cost` fn). [TODO]
2. Drop the shape gate from `planApplicable` (type/validity only); keep the
   `.production` fallback for types without variants. [TODO]
3. Extend **Measure** to calibrate per-variant efficiencies (device-keyed
   `.zat`). [TODO]
4. Selector: rank all type-applicable variants by the variant-aware cost. [TODO]
   - Calibrate a per-op/variant **thread-scaling curve** (effective_bw at 1..N
     threads) -> knee/saturation; use it to derive `parts` (replace
     `parEff = min(1, parts/2)`); consider a `min_rows_per_part` grain cap. [TODO]
5. Tests: the cost ranks variants differently by efficiency; selection is
   deterministic; existing equivalence tests hold. [TODO]
6. `zig build` / `zig build test` green; E2B coherent; measure vs the static rule. [TODO]

**As-built.** `calibrate` measures a per-variant efficiency (a representative
matvec per variant; `eff = best_time / variant_time`), stored in `ZAT2` v4
(`Caps.variant_eff`). On this machine: `q4_0_u4 = 1.0`, `q4_0_t2 = 0.91`,
`q4_0_t4 = 0.84` — **u4 wins for every interchangeable type** (matching the earlier local prototype).
`planApplicable` is now type/validity only (no shape gate; `_t4` still needs
`lanes >= 16`), so unroll is a candidate for wide output; the selector ranks
`base / eff(variant)`. E2B now picks `q4_0_u4` for the wide projections
(was `q4_0_t2`); deterministic; 166/166 tests.

## Definition of Done

- [x] `applicable` no longer gates by shape; unroll is a candidate for wide output.
- [x] `cost` differs per variant (efficiency); the Selector ranks by it.
- [x] Per-variant efficiency is calibrated per device and cached in the `.zat`.
- [x] `parts` comes from the measured scaling curve (`runtime/12`), not the fixed
      `parEff` cap.
- [x] Deterministic plan; `zig build` / `zig build test` green (166/166); E2B coherent.
- [x] The selected variants match the earlier local prototype (`q4_0_u4` for wide output).

## Arch Provenance

- **own:** the variant-aware cost + calibrated efficiency (deterministic
  selection; no per-candidate timing gate).
- **spec:** roofline; ISA widths.
- **consulted:** llama.cpp/ggml variant selection; the earlier local prototype's load-time bench
  (reference only, not mirrored).
- **derived:** `kernels/04`/`06` (candidate metadata, generic Selector),
  `runtime/09` (calibration), `runtime/11` (performance).

## Comments

- Continues `kernels/06` (the generic Selector) and `runtime/09` (calibration).
- The efficiency model is deliberately coarse: the goal is the correct
  **ranking**, not an exact time.
- Complements `runtime/11`: fixing the bandwidth/`parEff` there changes which
  variant wins, so the two should land together.
- **Ordering: `runtime/12` (the cost pipeline) is built first.** It owns the
  scaling/efficiency logic (Roofline + measured scaling + the Evaluator); this
  task's variant efficiency is one input to 12's stages. Building 12 first keeps
  the variant selection consistent with the rest of the cost model instead of
  patching a single `applicable` rule.

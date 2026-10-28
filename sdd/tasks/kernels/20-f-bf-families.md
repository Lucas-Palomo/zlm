20 - F/BF Families

Status: done
Commits:

## Goal

Extend the dynamic grid to the **F** (f32/f16) and **BF** (bf16) families — the
same pattern as the Q family (`kernels/17`): each member is an **L1 type**, the
catalog generates the kernel x variant grid, and the shared sweep
(`form/single.zig`) runs them. They are **fp-only** (no nibble -> no `int`). This
finishes the matvec families (Q done; `iq` later).

The current `matVecF32`/`matVecF16`/`matVecBF16` are hand-written (scalar / a
fixed 8-wide) and outside the grid; they become the families' `production_fp`.

## Scope

- **In:** the F/BF L1 types + their catalog entries; the sweep reuse; the test
  layout for the families (`kernels/19`).
- **Out:** `iq` (a later family); the GPU.

## Design

### The L1 types

`F32` / `F16` / `BF16`, contract `{ block_elems, block_bytes, has_min = false,
production_fp, fpBlock }`:
- `F32` — 4 bytes/elem; `fpBlock` reads a f32 chunk.
- `F16` — 2 bytes/elem (a f16 read + `@floatCast`).
- `BF16` — 2 bytes/elem (`u16 << 16`).
- **No `intBlock`** — there is no nibble, so no integer dot.

**The `block_elems` decision.** The F/BF have **no quant block** (the whole row is
the unit). Use a fixed **chunk** as the sweep's inner granularity (so the block
loop is uniform across families) — pick a sensible chunk (e.g. 16 or 32) and
justify it. The `block_bytes = block_elems * (4 | 2)`.

### The catalog

- `KernelTag += f32 / f16 / bf16`; the candidates `F32_fp_*`, `F16_fp_*`,
  `BF16_fp_*` (the same sweep axes; P1/P2 apply).
- `production_fp` = the current hand-written kernel (kept as the reference; the
  sweep's too-large-activation fallback).
- The dispatch (`form/single.zig`) routes the f32/f16/bf16 plans through the
  generated candidates; the calibration measures them.

### The test layout (per `kernels/19`)

- `kernels/float_test.zig`, `kernels/bfloat_test.zig` — the family tests.
- The shared fixtures in `matvec/fixtures.zig`; the cross-module parity in
  `matvec/test.zig`.
- The production files stay `test`-free.

## Steps

1. **Done.** `F32`/`F16` in `kernels/float.zig`, `BF16` in `kernels/bfloat.zig`:
   `{ block_elems = 32, block_bytes = block_elems * (4|2), has_min = false,
   production_fp, fpBlock }`. `fpBlock` reads one 32-element chunk (a `@Vector`
   load, `@floatCast` for f16, a 16-bit shift for bf16) and folds `w*x` into the
   `lanes`-wide accumulator; `corr` is unused. No `intBlock`.
2. **Done.** `KernelTag += f32/f16/bf16`; `kernel_catalog += F32_fp`/`F16_fp`/
   `BF16_fp` (fp-only, the same sweep axes + P1/P2); `f32_candidates`/
   `f16_candidates`/`bf16_candidates`. `tuner/calibrate.zig` measures them
   (`ProbeKind`/`fillProbe`/`gridCandidates`/`runGridGen`). `isPlanType` +
   `kernelGgmlType` in `root.zig`.
3. **Done.** `kernelType` in `form/single.zig` maps the tags to the L1 types; the
   existing generated-candidate loop + `dispatchCandidate` route the F/BF plans,
   so the `matVecF32`/`matVecF16`/`matVecBF16` fallbacks are now only the
   `.production` reference.
4. **Done.** `kernels/float_test.zig` + `kernels/bfloat_test.zig` (the L1
   `fpBlock` step), `fixtures.zig` (`fillF32Fixture`/`fillF16Fixture`/
   `fillBF16Fixture`), `matvec/test.zig` (the three cross-module candidate-parity
   tests + the catalog assertions). The production files stay `test`-free.
5. **Done.** Parity for every F/BF candidate; `zig build test` 275/275; `rg
   "^test "` on the production matvec files → 0; `zig fmt` clean on the changed
   files.

## Definition of Done

- [x] `F32`/`F16`/`BF16` are L1 types; the grid generates their variants.
- [x] The hand-written `matVecF32`/`matVecF16`/`matVecBF16` are the
      `production_fp` (not the dispatched path).
- [x] The test layout applies (the `_test.zig` + the fixtures + the aggregator).
- [x] Parity for every candidate; tests green; the production files `test`-free.

## Arch Provenance

- **own:** the F/BF families on the `kernels/17` pattern.
- **derived:** `kernels/17` (the grid), `kernels/18` (the structure),
  `kernels/19` (the test layout).

## Comments

- **fp-only.** No nibble -> no `int` for F/BF.
- **The `block_elems` is a chunk** (no quant block) — the one design decision.
- **Finishes the matvec families.** Q done (`kernels/17`), F/BF here, `iq` later.
- **Low perf value, high consistency:** the f32/f16 are rarely the bottleneck;
  the value is the same grid/variants for every family.
- **As-built.** `block_elems = 32` (no quant block): the Q families' granularity
  and a natural multiple of every 4/8/16-lane width (2-8 fma/chunk); it also
  keeps the sweep's `n % block_elems == 0` gate. The catalog is now **600
  generated plans** (the 4 Q families with both dots + q6_k `fp` + the three
  F/BF `fp`) plus `production` = **601**; `.zplan` VERSION 23 -> 24, `.zat`
  VERSION 20 -> 21 and
  `calibrate.max_variants` 512 -> 640 (the plan ordinal is the efficiency index).
  `zig build test` 270 -> 275. The six tuner tests that iterate
  `std.enums.values(MatVecPlan)` needed `@setEvalBranchQuota(1_000_000)` (the
  enum passed the default 1000-branch limit), and the two "unlisted type keeps
  production" tests moved `.f32` -> `.q5_0` (f32 is now a grid family).
- **E2B (as-built).** `gemma4-e2b` re-tuned (`e2b-after`, 4 threads, 4 GB):
  `inp_gate`/`proj` (70 tensors, f32) pick `F32_fp_w8t4un1_p0` and
  `per_layer_model_proj` (bf16, not f32 — the task's guess) picks
  `BF16_fp_w8t8un1_p0`; all were `.production` before. CPU decode (warm, 48
  tokens) **14.9 -> 15.2/15.4 tok/s** — noise. Coherent output
  ("The capital of France is **Paris**"). So BF16 is actually exercised by the
  model, not only synthetic.
- **Honest verdict.** **Low perf value, high consistency.** No measurable decode
  change (the Q families dominate and f32/bf16 are a small byte share); the
  value is that every matvec family now shares one grid + catalog + test layout,
  and `per_layer_model_proj` (bf16) gains the measured tile/unroll variants.

18 - matvec Structure (layout x kernel x form)

Status: done
Commits: a3be5dd

## Goal

Reorganize `kernels/linear/matvec/` by its three axes — **layout x kernel x
form** — so the directory reflects the taxonomy and no file mixes concerns. The
motivating smell: `scale_split.zig` *looked* like a kernel but is a **layout**,
and `root.zig` (1325 lines) mixed the dispatch, the test oracle/fixtures, the
family surface and the tests.

**This is a MECHANICAL refactor: no behaviour change.** The public surface
(`matvec/root.zig`) is unchanged (the facade re-exports the same names), and
`zig build test` stays green (268/268).

## The axes

- **layout** (how a block's bytes are read): `native | scale_split`.
- **kernel** (L1 the weight family x L2 the dot): `quantized` (Q) | `float` (F) |
  `bfloat` (BF).
- **form** (the Composer): `single | fused | batched`.
- **types**: the plan + the catalogs (L1/L2/L3).

## The layout

```text
matvec/
  root.zig          # facade (re-exports; the public surface)
  layout.zig        # native | scale_split  (the block readers)
  types.zig         # the plan + the catalogs (Variant/Dot/KernelTag/MatVecPlan)
  kernels/
    quantized.zig   # Q4_0 / Q4_1 / Q4_K / Q8_0 / Q6_K  (decode; layout = a comptime param)
    float.zig       # F16 / F32
    bfloat.zig      # BF16
  form/
    single.zig      # the single-weight matvec (the sweep L3)
    fused.zig       # the pair (gate_up + the activation)
    batched.zig     # the batch (the M dim)
```

## What goes where

| file | role | comes from |
|---|---|---|
| `layout.zig` | the block readers (native \| scale_split) | the layout parts of `native.zig` + `scale_split.zig` |
| `kernels/quantized.zig` | the Q family types (decode, layout param) | the L1 types in `native.zig` |
| `kernels/float.zig`, `bfloat.zig` | the F/BF family types (fp only) | the f16/f32/bf16 bodies in `native.zig` |
| `form/single.zig` | the sweep (tile/unroll/width/prefetch) + the dispatch | the sweep in `native.zig` + the dispatch in `root.zig` |
| `form/fused.zig`, `batched.zig` | the forms | `fused.zig` / `batched.zig` |
| `types.zig` | the plan + the catalogs | `types.zig` (unchanged) |
| `root.zig` | the facade + the family surface + the tests | the re-exports + the `surface` + the tests |

## Steps

1. **Done.** `layout.zig` (the native | scale_split readers, from the layout parts
   of `native.zig` + `scale_split.zig`).
2. **Done.** `kernels/quantized.zig` (the Q L1 types), `kernels/float.zig` (F32 /
   F16), `kernels/bfloat.zig` (BF16).
3. **Done.** `form/single.zig` (the sweep + the dispatch), `form/fused.zig`,
   `form/batched.zig`.
4. **Done.** `types.zig` (unchanged) + `root.zig` (the facade + the surface + the
   fixtures + the tests).
5. **Done.** `zig build test` 268/268; the public surface unchanged (12 importers
   compile); the diff is moves + wiring.

## Definition of Done

- [x] The matvec directory is organized by layout / kernel / form.
- [x] The public surface (`matvec/root.zig`) is unchanged (no importer changes).
- [x] No behaviour change; `zig build test` 268/268; `zig fmt` clean.

## Arch Provenance

- **own:** the layout x kernel x form organization.
- **derived:** `kernels/17` (the 3-level kernel), `kernels/11` (per-ISA).

## Comments

- **`scale_split` is a LAYOUT, not a kernel** — the motivating fix.
- **F and BF are separate families** (a distinct decode).
- **`root.zig` stays the facade** (the re-exports) + the family surface + the
  tests; the dispatch moves to `form/single.zig`.
- **As-built.** `native.zig` (1383) + `scale_split.zig` (151) are gone; the code
  lives in `layout.zig` (632), `kernels/quantized.zig` (735), `kernels/float.zig`
  (32), `kernels/bfloat.zig` (30), `form/single.zig` (398), `form/fused.zig`
  (391), `form/batched.zig` (533); `root.zig` 1325 -> 1091 (the facade + the
  surface + the fixtures + the tests). `readF16` (duplicated in
  native/scale_split) is now one definition in `layout.zig`. A **function-only
  import cycle** `kernels/quantized.zig <-> form/single.zig` (the L1
  `production_fp` calls the sweep; the dispatch calls the Q bodies) — Zig
  resolves it; no comptime loop. `tuner/calibrate.zig` reached into `native.zig`
  directly; rewired to the new modules (import wiring only).

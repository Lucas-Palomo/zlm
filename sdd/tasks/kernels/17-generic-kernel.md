17 - Generic Kernel (weight-family x dot x sweep)

Status: done
Commits:

## Goal

Replace the per-quant bodies (`matVecQ4_0Gen`, `matVecQ4_KGen`, ...) with **one
generic sweep template** and a **kernel catalog** of comptime types, organized
in three levels. No `switch (quant)`, no `int` branch: the decode and the dot are
resolved by monomorphization, so the runtime is a direct (inlined) call.

Scope: the **Q family first** (q4_0, q4_1, q4_k) — the ALU-bound K-quants are
where it pays.

## The hierarchy (3 levels)

```text
L1 - the weight decode family:   Q | F | BF | iq
L2 - the dot (the algorithm):    fp | int        (int only where there is a nibble: Q/iq)
L3 - the sweep (the variant):    { width, tile, unroll, prefetch }
```

- **kernel = L1 x L2** (a comptime type) — e.g. `Q4_0_fp`, `Q4_0_int`, `F16_fp`.
- **variant = L3** (the sweep), shared by every kernel.
- The specialization name: `<kernel>_w{W}t{T}un{U}_p{P}` — e.g.
  `Q4_0_fp_w8t1un8_p0`, `Q4_0_int_w8t2un4_p0`, `BF16_fp_w8t1un4_p0`.

| L1 | what it is | members |
|---|---|---|
| **Q** | a quantized block (scale/min + nibbles) | q4_0, q4_1, q4_k, q8_0, q6_k |
| **F** | a raw float | f16, f32 |
| **BF** | a raw bfloat16 | bf16 |
| **iq** | the ggml i-quants | iq4_nl, iq2_xxs, ... |

`fp` = the f32 dot (the decode produces f32); `int` = the integer dot (a q8
activation). The `int` only exists where the weight has nibbles (Q/iq), so F/BF
have `fp` only.

## Design

### L1 — the weight decode (a comptime type)

Each weight family member is a type with its family's contract:
```zig
Q4_0 = struct {                       // L1 = Q
    pub const block_elems = 32;
    pub const block_bytes = 18;
    pub inline fn blockVec(comptime lanes, block, x) @Vector(lanes, f32) { ... }   // fp decode
    pub inline fn blockAccum(comptime lanes, block, qx) @Vector(lanes, i32) { ... } // int decode
};
F16 = struct { ... };                 // L1 = F  (a raw float read; no int)
BF16 = struct { ... };                // L1 = BF
```
No switch — the body calls `K.blockVec` (fp) or `K.blockAccum` (int).

### L2 — the dot

- `fp` uses `K.blockVec` + a f32 accumulator.
- `int` uses `K.blockAccum` + an i32 accumulator (a q8 activation).
The `int` is only instantiated for the kernels that define `blockAccum` (Q/iq).

### L3 — the shared sweep template

The sweep is identical across every kernel, so it is **one comptime template**
parameterized by the inner op:
```zig
fn sweep(comptime K: type, comptime V: Variant, comptime dot: Dot, ...) !void
```
`matVecFpGen` / `matVecIntGen` are thin wrappers; only the dot differs.

### The catalogs + the generation

```text
kernel_catalog = { Q4_0_fp, Q4_0_int, Q4_1_fp, Q4_1_int, Q4_K_fp, Q4_K_int, ... }  // L1 x L2
variant_catalog = { {w,t,un,pf}, ... }                                             // L3
```
The generation is a comptime loop over (kernel_catalog x variant_catalog); no
branch anywhere. Adding a weight family member is a type + (for `int`) its
`blockAccum`.

### Prunes

- Reuse P1 (comptime ISA, from `simd.native`) and P2 (the prefetch heuristic).

## Steps

1. **Done.** The L1 contract (`block_elems`/`block_bytes`/`has_min` + the `fp`/`int`
   step); `Q4_0`/`Q4_1`/`Q4_K` as types in `native.zig` (the fp + int decodes;
   `Q4_1_int` is new).
2. **Done.** The L3 sweep template `sweep(K, V, dot, ...)` in `native.zig`;
   `matVecFpGen`/`matVecIntGen` are thin wrappers.
3. **Done.** The kernel catalog (`KernelSpec`/`kernel_catalog`) + the variant
   grid + the generation loop in `types.zig` (`candidates`, `plan_names`,
   `MatVecPlan`). The plan enum widened `u8` -> `u16` (the product is 275).
4. **Done.** The dispatch (`matVecRowsWithPlan` -> `dispatchCandidate`) +
   `planApplicable`; the P1/P2 prunes reused.
5. **Done.** Parity for every generated Q candidate (fp vs production, int vs
   the integer oracle / the q8 bound); `zig build` / `zig build test` green
   (268/268).

## Definition of Done

- [x] One sweep template; each weight family member is a type; **no switch, no
      dot branch** in the body (the `dot` is a comptime parameter).
- [x] The kernel catalog (L1 x L2) + the variant catalog (L3) drive the
      generation; a new weight family member is a type + a catalog entry.
- [x] The names are `<kernel>_w{W}t{T}un{U}_p{P}`; parity for every Q candidate;
      the prunes apply; CPU decode measured.
- [x] `zig build` / `zig build test` green (268/268).

## Arch Provenance

- **own:** the 3-level kernel catalog (weight-family x dot x sweep).
- **derived:** `kernels/13`-`16` (the grid/generator), `kernels/11` (per-ISA
  atomics), the constitution's Kernel / AtomicOp vocabulary.

## Comments

- **Scope:** the Q family first; F/BF next (fp only, a trivial decode); iq
  later.
- **The `int` is not an axis.** It is L2 (an algorithm) and only exists where
  the weight has nibbles — so it is a property of the kernel type, not the
  variant.
- **Why not a switch:** monomorphization over the type makes the decode a direct
  inlined call; a `switch (quant)` would be a hand-written dispatch table.
- **As-built.** `native.zig` holds the three L1 types and one `sweep`; the
  `fp`/`int` step is selected by a `comptime` parameter, so the body has no
  runtime branch. `types.zig` generates the kernel x variant product from
  `kernel_catalog` (L1 x L2) x the grid (L3); `root.zig` maps a plan to its
  kernel with one comptime `switch` over the 3-way tag (the dispatch, not the
  body). On this AVX2 host (native width 8) the effective catalog is **60 `fp`
  + 30 `int` variants per family**, so **270 generated plans** (3 families) and
  **275** total (`plan_count`), which forced the plan enum and
  `Dispatch.variant` to `u16` (`.zplan` VERSION 20 -> 21; `.zat` VERSION 17 ->
  18, `max_variants` 256 -> 384).
- **Honest verdict (does the refactor change anything measurable?).** **No.**
  The generic sweep is parity-identical to the four bodies it replaces, but the
  Q GEMV is memory-bound: E2B decode is 15.3 vs 15.4 tok/s and 12B is 3.8 vs
  3.6 tok/s (noise). The new q4_1 generated kernel does replace the hand-written
  `q4_1_u4` (ffn_down now selects `Q4_1_fp_w8t16un1_p0`), and q4_0 selects
  `Q4_0_fp_w8t1un4_p0` — both are calibration re-picks, not speedups. The value
  is structural: one body, a catalog a new family is a type + an entry for, and
  `Q4_1_int` now exists. Cost: the binary grew 18.5 -> 23.4 MB (+27%, mostly
  symbols; 270 instantiations vs 180) and a clean ReleaseFast build is slower.
- **As-built (q8_0 grid cleanup).** The L1 contract is now
  `{ block_elems, block_bytes, has_min, production_fp, fpBlock, intBlock }`: the
  spec's pure `blockVec`/`blockAccum` partials were removed (0 callers — the
  sweep calls `fpBlock`/`intBlock`, which inline the layout primitives, also
  fixing Q4_K's inconsistent `blockVec` signature). **q8_0 joined the grid** as a
  full family (`Q8_0`, `Q8_0_fp` + `Q8_0_int`); its hand-written matvec wrappers
  (`matVecQ8_0Lanes`/`TileRows`/`Width`/`U4`) and the `q8_0_u4`/`q8_0_t2` base
  plans are gone, so every q8_0 sweep variant is generated. The q8_0 `int` decode
  is a signed-`i8` dot (`q8_0IntBlockVec`: no nibble bias, no min term), so q8_0
  is no longer `fp`-only. The thin
  `matVecQ4_0Lanes`/`matVecQ4_0Width`/`matVecQ4_0TileRows` wrappers and the
  hand-written `matVecQ4_0Int` body were removed (the generic `matVecIntGen`
  covers it; a `q4_0IntOracle` fixture keeps the parity oracle). The Q-family
  tests moved from `kernels/quantized.zig` (735 -> 492) into
  `matvec/root.zig`. `.zplan` VERSION 21 -> 22 and `.zat` VERSION 18 -> 19 (the
  plan ordinals renumbered).
- **As-built (q6_k grid + thin productions).** q6_k joined the grid as an
  `fp`-only family (`Q6_K`, `Q6_K_fp`); its hand-written
  `matVecQ6_K`/`matVecQ6_KU4`/`matVecQ6_KT2` bodies and the `q6_k_u4`/`q6_k_t2`
  base plans are gone, so every q6_k sweep variant is generated and its
  `production_fp` is the native-width, tile-1, unroll-1 sweep. The q4_1/q4_k
  `production_fp` bodies (`matVecQ4_1U4`, `matVecQ4_K`) became thin
  `matVecFpGen` references exactly like q4_0/q8_0; the now-unused
  `matVecQ4_1`/`matVecQ4_1NoCache`/`matVecQ4_1TileRows` wrappers were removed.
  The `matVecQ4_KInt` integer oracle moved to `matvec/root.zig` as
  `q4_KIntOracle` (next to `q4_0IntOracle`). `quantized.zig` 492 -> 289 lines.
  `.zplan` VERSION 22 -> 23 and `.zat` VERSION 19 -> 20 (the plan ordinals
  renumbered); `calibrate.max_variants` 384 -> 512; 270/270 tests.
- **The L1 contract should have covered the dequant too (the `q5_k` gap).** It
  defined the matvec decode (`fpBlock`/`intBlock`) but not the **dequant**
  decode, so `q5_k` — a dequant-only weight (`per_layer_token_embd`, a lookup,
  no GEMV) — had nowhere to live and `dequantTo` stayed a per-type `switch`.
  The contract should have exposed the block dequant as well, so the matvec and
  the dequant resolve through one L1 type and share one decode. Closed by adding
  the `dequantBlock` step to every L1 type plus a generic `dequant(comptime K,
  ...)` block loop, with `dequantTo` as the type -> `K` dispatch; `q5_k` joins as
  a dequant-only member (`Q5_K`, no `fpBlock`).

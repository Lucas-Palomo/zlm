12 - Variant Grid for All GEMV Quants (spec constants)

Status: done
Commits: (uncommitted)

## Goal

`gpu/09` made the GPU GEMV variant a measured **grid** via Vulkan specialization
constants — but only for `q4_0`. Every other quant (`q4_1`, `q8_0`, `q6_k`,
`f32`, `f16`, `bf16`) and the fused `gate_up` still ship as fixed-`#define`
shaders, so their grid is a single `plain` entry: the probe cannot explore
`rows_per_group` / `unroll` / `reduction` / `workgroup_size`, the tuner cannot
select a variant, and `Quant.variant` has no `q4_1_u4` / `q8_0_u4` / … to record.
The grid is **not uniform across quants** — a latent inconsistency: the
mechanism exists but is applied to one quant.

Finish `gpu/09`: **one spec-constant GEMV shader per quant**, the same axes, the
same probe, the same per-dispatch variant.

## Scope

- **In:** convert `matvec_q4_1`, `matvec_q8_0`, `matvec_q6_k`, `matvec_f32`,
  `matvec_f16`, `matvec_bf16` and the fused `gate_up` (`q4_0`, `q4_0_u4`,
  `q8_0`) to the spec-constant axes; the grid entries; `grid.specConsts`; the
  `Quant` variants + `Quant.variant`; the Vulkan `createSpecializedPipeline` per
  quant; the probe (enumerate + measure every quant's variants) + the recorded
  choice (`q4_1_variant`, `q8_0_variant`, `q6_k_variant`, …); the tuner's
  per-quant variant; the parity tests.
- **Out:** the batched GEMM (`matmul_*`) axes — a separate, one-shot op with its
  own block structure; new quant support; `iq`; the `int` / vectorization axes
  (`gpu/09` marks them experimental, low return on a mem-bound GEMV).

## Design

### The axes (unchanged from `gpu/09`)

| `constant_id` | name | meaning |
|---|---|---|
| 0 | `ROWS_PER_GROUP` | the row tile (4 / 8) |
| 1 | `UNROLL` | the inner k-block unroll (independent accumulator chains) |
| 2 | `REDUCTION` | `0` shared tree / `1` subgroup (`subgroupAdd`) |
| 3 | `WORKGROUP_SIZE` | `local_size_x_id` (32 / 64 / 128) |

The GEMV body is the same shape for every quant — load a k-block, dot it against
the activation block, accumulate per row, reduce across the workgroup. **Only the
block decode differs** (`18`/`20`/`34`/`210` bytes, `f16` scale, `q4_1` min term,
`q6_k` super-block, element-wise floats). So the conversion factors the axes out
of `matvec_q4_0.comp` and re-applies them to each decode.

### One shader per quant (as `gpu/09` intended)

`matvec_q4_0.comp` already has the axes; the other `.comp` become the same
scaffold with their own decode body. The per-variant `.comp` shaders
(`matvec_q4_0_u4` is already gone; `matvec_q4_k_u4`/`_u8`,
`matvec_gate_up_q4_0_u4` are fixed variants) collapse into the spec-constant
shader + its `layout(constant_id=…)` values.

### The grid, the probe, the recorded choice

- The grid (`gpu/kernels/grid.zig`) lists one entry per `(op, quant, variant)`
  for every quant (as it already does for `q4_0`).
- `grid.specConsts(quant, variant)` returns the axis values for **every** quant
  (today it returns the base axes for `quant != .q4_0`).
- The backend creates a pipeline per `(quant, variant)` via
  `createSpecializedPipeline` (the `VkSpecializationInfo`), pruning variants the
  device rejects (the subgroup / workgroup cases).
- The probe measures each quant's variants **interleaved** (the RX 580 clock-ramp
  caveat) and records the winner in the coeffs (`q4_1_variant`, `q8_0_variant`,
  `q6_k_variant`, …). The tuner writes it onto every dispatch of that quant.

## Steps

1. **q8_0** (the first vertical slice — the quant in active use): `matvec_q8_0`
   gains the axes; parity for `plain`/`u4`/`sub`/`un2`/`un4`/`w32`/`w128`.
2. **q4_1**: same, with the `min` term (`dot = d·Σq·x + m·Σx`).
3. **q6_k**: the super-block decode wrapped in the axes.
4. **f32 / f16 / bf16**: element-wise; the axes over element groups.
5. **gate_up** (`q4_0`, `q8_0`): the two-weight body + the gated activation under
   the same axes; `q4_0_u4` collapses into the spec-constant shader.
6. **Wire**: `specConsts`, the grid, `Quant` + `variant`, the pipelines, the
   probe + the coeffs, the tuner.
7. **Verify**: parity for every `(quant, variant)`; `zig build` / `zig build test`
   green; no regression at `--vram` (the q4_0 plans unchanged).

## Flow / Arch

```text
grid.zig  ── (op, quant, variant) ──▶ specConsts(quant, variant)
                                          │  the axis values
                                          ▼
pipeline.zig  createSpecializedPipeline(shader[quant], axes)  ── N pipelines
probe (backend/root.zig)  measure each variant, interleaved  ── the winner
coeffs (calibrate.zig)  q4_1_variant / q8_0_variant / …        ── the plan
tune.zig  writes the winner onto every dispatch of the quant
```

## Definition of Done

- [x] Every GEMV quant ships **one** spec-constant shader (no fixed-`#define`
      variant shaders left for `q4_1`/`q8_0`/`q6_k`/`f32`/`f16`/`bf16`/`gate_up`).
      `matvec_q4_k_u4`/`_u8` and `matvec_gate_up_q4_0_u4` are deleted.
- [x] `grid.specConsts` covers every quant; the grid lists the variants for all.
- [x] `Quant` + `Quant.variant` carry the variants for every quant.
- [x] The probe measures every quant's variants; the coeffs record the winner
      (`.zat` v25); the tuner writes it per dispatch.
- [x] Parity tests cover every `(quant, variant)` (56/56 matvec + 14/14 gate_up on
      the RX 580, 0 pruned).
- [x] `zig build` / `zig build test` green (290/290).

## As-built

- The register arrays are sized by the **specialization constants**
  (`float acc[UNROLL][ROWS_PER_GROUP]` → `OpTypeArray %float %UNROLL`), so the
  driver sizes the footprint per pipeline. Fixed `MAX_UNROLL`/`MAX_ROWS` on the
  local accumulators were the first mistake (a 64-register `gate_up`).
- The generic `gate_up` then regressed the gemma-12B decode (~8.5 -> ~6.7
  tok/s): the dynamically-indexed temporaries (`dg[]`/`du[]`/`nbase[]`) spilled
  to local memory. **Fix:** move the row loop outside the group loop so those
  temporaries are **scalars** (bit-identical math); recompiled with
  `glslc -O`. 12B recovered to **8.3-8.4 tok/s**; the same pattern is applied to
  `matvec_q8_0`/`matvec_q4_1`.
- `.zat` **v24 -> v25 -> v26** (the per-quant variant fields); an old `.zat`
  re-measures.
- **Fused `gate_up` for `q4_1`** (the Qwen quant): a new
  `matvec_gate_up_q4_1.comp` (spec-constant, the optimized scalar structure, the
  q4_1 `min` term) + the full wiring (`gate_up_quants = {q4_0, q4_1, q8_0}`),
  `gate_up_q4_1_variant`, the probe. The tuner selects the gate_up variant **per
  quant** (`gpuGateUpVariant`) — a bug where every gate_up used the q4_0 winner
  was fixed. Measured on `qwen3-14b` Q4_1 (`--threads 8 --ram 8gb --vram 6gb`):
  decode offloads **27 -> 67**, context 9726 -> 36926, decode **3.7 -> 5.9-6.0
  tok/s** (+60%), output coherent.

## Arch Provenance

- **own:** the uniform grid across quants.
- **derived:** `gpu/09` (the axes + the spec-constant mechanism), `gpu/08` (the
  selector), `gpu/07` (the fused `gate_up`).

## Comments

- **Neutral on this host.** `gpu/09` measured the axes (reduction, unroll,
  workgroup) as **neutral on the RX 580** (the GEMV is memory-bound), so this is
  **consistency / completeness**, not a speedup here — the payoff is on a device
  whose axes win, and in the plan's uniformity.
- **The `.zat` format.** The recorded per-quant variant is a new coeffs field, so
  the calibration artifact's version bumps (an existing `.zat` is re-measured).
- **The `int` / vectorization axes** stay out (`gpu/09`: low return on a
  mem-bound GEMV).

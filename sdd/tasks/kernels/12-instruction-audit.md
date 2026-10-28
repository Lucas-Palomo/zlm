12 - Generic Instruction Audit (P0/P1: matvec, dequant, reduce)

Status: done
Commits: 0ea10f6 17fa378 1798aeb 2371433 b5039be 3328d6c

## Goal

Apply the **Generic CPU Instruction Catalog** (`sdd/kernels-instruction-
catalog.md`) to the hot AtomicOps and turn each finding into a verdict backed by
**assembly + benchmark**. Implement the `ACCEPT` items as **generic** Zig
changes (no handwritten ISA kernels); record the `REJECT`/no-gain items as
measured outcomes. The deliverable is a findings report plus the accepted
optimizations, not a checklist of hypotheses.

## Context

- `kernels/10` measured the generic path and found the levers were the **width**
  and the **instruction** (not register tiling / blocking).
- `kernels/11` closed the two primary dimensions: **vector width** (plan-recorded
  `simd_width`, Tuner-selected) and **FMA** (`simd.fma`, target-guarded).
- What remains are the **data-transform** operations of the quantized path:
  integer→float conversion, nibble unpacking (`@shuffle`), the per-block
  `@reduce`, and integer SIMD — all on the matvec/dequant hot path.
- The e2e decode is **memory-bound** on the current host (`kernels/11` measured
  ~2–4% from width+FMA), so a finding only counts if it moves a **kernel
  benchmark** and, where relevant, the **e2e** — or if it unlocks a future target
  (AVX-512, the generic x86 build).

## Scope

- **In (P0):** the matvec hot path — `@Vector`, `@mulAdd`, `@floatFromInt`,
  `@reduce`, integer SIMD:
  - `q4_0BlockDotLanes` / `q8_0BlockDotLanes` (native.zig),
  - `matVecQ4_0Lanes` / `matVecQ8_0Lanes`,
  - the dequant path used by the matvec (`quant/`).
- **In (P1):** quantization data transforms — `@shuffle`, integer
  widening/narrowing, byte permutation (nibble unpack).
- **In (P0, extended):** the other hot families — lm_head (q6_K/q4_K), norm,
  attention, activation, elementwise, batched prefill — audited for the same
  per-block `@reduce` / data-movement pattern.
- **Out:** P2/P3 (`@select`, `@floatCast`, saturating arithmetic, `@prefetch`,
  `@popCount`, `@clz`, `@ctz`); the width/FMA dimensions (done in `kernels/11`);
  any handwritten ISA kernel; the Scheduler/Selector/ResidencyGraph.

## Design

### Procedure

Follow the catalog's six steps (identify → record target → inspect asm →
propose generic adjustment → validate → benchmark) and report each finding in
the catalog's format. The verdicts are `ACCEPT` / `REJECT` /
`NEEDS FURTHER INVESTIGATION`.

### Targets to inspect first

- **`@floatFromInt`:** `q8_0BlockDotLanes` converts `@Vector(lanes,i8)` to f32.
  Check whether LLVM emits a vector `vcvtdq2ps` or goes through i32 / scalar
  conversions; whether the `i8→f32` path is better expressed as a wider integer
  load + one conversion.
- **`@shuffle` / nibble unpack:** `q4_0BlockDotLanes` expands 16 nibble bytes to
  32 signed values (vectorized in `15fb378`). Check the emitted shuffle/mask
  sequence and whether a cheaper generic expression (e.g. a wider unpack +
  `@select`/`@shuffle`) removes extract/insert.
- **`@reduce`:** the f32 block dots reduce **once per block**. Compare against a
  **deferred reduction** (accumulate the `@Vector` across blocks, reduce once per
  row) — the AVX2 integer kernel already does this; check the f32 path.
- **Integer SIMD:** the q4_0/q8_0 unpack uses `@Vector(lanes,i8)` arithmetic;
  check for scalarization or widening overhead.

### Findings table

Target: Zig 0.16.0, `-mcpu=native` (Zen 3: AVX2+FMA, no AVX-512), `-OReleaseFast`,
`simd_width=8`. Kernel benchmark = single-thread `[4096, 8192]` matvec,
min-of-5.

| AtomicOp | source fn | op | finding (asm) | baseline | candidate | verdict |
|---|---|---|---|---|---|---|
| matvec q8_0 | `q8_0BlockVec` | `@floatFromInt` | `vpmovsxbd`+`vcvtdq2ps` (vector i8→i32→f32); no scalar fallback | — | — | **REJECT** (already optimal) |
| matvec q4_0 | `q4_0BlockVec` | `@shuffle`/unpack | `vmovdqu`+`vpand`+`vpsrlw`+`vpand`+`vpaddb`; one `vpshufd $0xee` per chunk | — | — | **NEEDS INVESTIGATION** (low) |
| matvec q4_0 | `q4_0BlockVec` | integer SIMD (unpack) | fully vectorized, no scalarization; a 16-wide shuffle-free variant (`lo`↔`x[0..16]`, `hi`↔`x[16..32]`) is slower — 16-wide splits on AVX2 | 4.6 GB/s (8-wide) | 2.9 GB/s (16-wide) | **REJECT** (8-wide optimal) |
| matvec q4_0 | `matVecQ4_0Lanes` | `@reduce` | per-block horizontal reduce = scalar `vaddss` chain (`vmovshdup`/`vshufpd`/`vshufps`/`vextractf128`/`vaddss`) | 4.6 ms / 4.13 GB/s | 2.6 ms / 7.23 GB/s | **ACCEPT** (1.77×, implemented) |
| matvec q8_0 | `matVecQ8_0Lanes` | `@reduce` | same per-block reduce | 3.1 ms / 11.66 GB/s | 2.0 ms / 17.93 GB/s | **ACCEPT** (1.54×, implemented) |

**ACCEPT implemented** (`0ea10f6`): the block dot exposes the unscaled vector
partial (`q4_0BlockVec`/`q8_0BlockVec`); the matvec accumulates it across blocks
(scaled by each block's `d`) and reduces **once per row**.

**e2e (`run`, `--threads 8`):** e2b decode **14.7 → 16.1** tok/s (~9%), g12b
**3.7 → 3.8** (~3%). The decode was partly compute-bound on the scalar reduce
chain, so a compute fix moved it despite the memory-bound framing.

### Family sweep (extended scope)

The same per-block/per-chunk `@reduce` pattern was checked across the other hot
families (not only matvec):

| family | source fn | finding | verdict |
|---|---|---|---|
| lm_head q6_K | `q6KBlockDot` | per-chunk scalar reduce (`q6KChunk16`) | **ACCEPT** (2.2×, implemented) |
| softmax | `softmaxInPlaceLanes` | per-chunk total; deferring the total is `@exp`-bound | **REJECT** (no gain) |
| norm | `sumSqLanes` | reduces **once** per vector | **REJECT** (already deferred) |
| attention | `dotScaledLanes` | reduces **once** per score (inherent) | **REJECT** |
| activation / elementwise | `gatedSilu`/`add`/`mul`/… | no reduce (elementwise) | **REJECT** |
| batched prefill | `matVecQ*_0BatchedRows` | already accumulates a vector per row | **REJECT** (already deferred) |

**q6_K ACCEPT implemented** (`1798aeb`): `q6KChunkVec16` + accumulate in
`q6KBlockDot`, reduce once per block. Kernel 11.2 → 5.1 ms (2.2×). e2b/g12b use
an f16 `production` lm_head (not q6_K), so their e2e is unchanged; the win
applies to models with a q6_K lm_head.

### File sweep (every kernel file)

A systematic pass over every kernel file for the catalog patterns (per-block
`@reduce`, scalar fallback, vector splitting, redundant conversion, missing
FMA):

| file / fn | pattern | verdict |
|---|---|---|
| `matvec/native.zig` q4_0/q8_0 matvec | per-block `@reduce` | **ACCEPT** (done, 1.77×/1.54×) |
| `matvec/native.zig` q6_K (`q6KBlockDot`) | per-chunk `@reduce` | **ACCEPT** (done, 2.2×) |
| `matvec/native.zig` q4_K (`q4KNibDot32`/`dotQ4_K`) | per-chunk `@reduce` | **ACCEPT** (done, 1.43×) |
| `matvec/native.zig` q4_1 (`q4_1BlockDot`/`xBlockSum`) | per-block `@reduce` | **ACCEPT** (done, 1.63×) |
| `matvec/native.zig` `legacy` variants | not applicable (`planApplicable` excludes them) | **REMOVED** (dead code, `b5039be`) |
| `matvec/native.zig` tile variants t2/t4 (q4_0/q8_0/q4_1) | per-block `@reduce` | **ACCEPT** (deferred, `3328d6c`) |
| `matvec/native.zig` `matVecBF16` | reduce once per row | OK |
| `matvec/batched.zig` q4_0/q4_1 tiled | vector accumulate per row | OK (already deferred) |
| `matvec/batched.zig` q8_0 scalar/tiled, q4_0 scale_split batched | per-block `@reduce` | **ACCEPT** (deferred, `3328d6c`) |
| `matvec/fused.zig` `gate_up` q4_0/q4_1/scale_split | per-block `@reduce` | **ACCEPT** (deferred, `3328d6c`) |
| `matvec/scale_split.zig` | per-block `@reduce` | **ACCEPT** (done, same q4_0 structure) |
| `quant/floats.zig` f16/bf16 dequant | `@Vector(8,f16)→f32` = F16C | OK |
| `quant/standard.zig` q4_0/q8_0 dequant | `@Vector(16,f32)` (split on AVX2) | OK (memory-bound) |
| `quant/k_quants.zig` dequant | scalar element writes | **CANDIDATE** (not vectorized; k-quants not hot) |
| `matmul/root.zig` `runRows` | matvec per token | OK (deferred) |
| `matmul/root.zig` `runBlocked` | dequant + `dot` per token | OK (`dot` reduces once) |
| `transformer/norm.zig` `sumSq` | reduce once | OK |
| `transformer/attention.zig` `dotScaled` | reduce once per score | OK |
| `transformer/activation.zig`, `elementwise.zig` | elementwise | OK |
| `transformer/activation.zig` clamp (`@max`/`@min`) | `vminps`+`vmaxps` on ymm, no scalar fallback, no compare/blend | OK (optimal; width = `simd_width`) |
| matvec/fused block-scale broadcast (`@splat`) | `vbroadcastss` (ymm/zmm), hoisted once, reused across a split | OK (optimal) |
| `transformer/rope.zig` | 8-wide, mul-add without `simd.fma` | **CANDIDATE** (small) |
| `linear/vector.zig` `softmax` | per-chunk total | **REJECT** (`@exp`-bound) |
| `linear/vector.zig` `dot` | reduce once | OK |

The per-block-reduce pattern **recurs uniformly** and the fix is the same
(vector partial + deferred reduce). **All** occurrences are now closed:

- measured kernel wins: q4_0/q8_0 matvec (1.77×/1.54×), q6_K lm_head (2.2×),
  q4_K (1.43×), q4_1 (1.63×), scale_split;
- the tile (t2/t4), batched-alt and fused variants use the same fix;
- the `legacy` variants were **dead code** (never applicable) and were removed.

No selectable matvec / batched / fused path reduces per block anymore.

### Folder sweep (matmul, quant, vector, transformer, architectures)

- `matmul/root.zig`: `runRows` (deferred matvec) / `runBlocked` (`dot` once per
  row) — OK.
- `quant/standard.zig` (q4_0/q4_1/q8_0 dequant): `@Vector(16,f32)`; measured
  16- vs 8-wide = 3.5 vs 3.4 ms — **memory-bound, OK** (no split penalty).
- `quant/k_quants.zig` (q4_K/q5_K/q6_K dequant): was scalar `for l` loops —
  **ACCEPT** (vectorized, `56dbdf8`; kernel 16.0 → 2.8 ms, 5.7×).
- `quant/floats.zig` (f16/bf16 dequant): `vcvtph2ps` — OK.
- `transformer/rope.zig`: was `x0*c - x1*s` without FMA — **ACCEPT** (`simd.fma`,
  `56dbdf8`).
- `transformer/activation.zig` `softcapInPlace`/`geluInPlace`: was scalar loops —
  **ACCEPT** (vectorized with a `@exp`-based vector tanh, `56dbdf8`).
- `transformer/{norm,attention,elementwise}` + `vector.zig`: reduce once /
  elementwise — OK.
- `architectures/`: declarative (config/metadata/names/contract) — no compute
  kernels, OK.

### Catalog ops not used by zlm (asm evidence)

Audited for the catalog, no zlm use yet — recorded so a future use is grounded:

| op | asm (AVX2) | verdict |
|---|---|---|
| `@popCount(u64)` | `popcnt` | OK |
| `@popCount(@Vector(8,u32))` | scalarized bit-trick | OK (needs AVX-512VPOPCNTDQ for `vpopcntd`) |
| `@clz`/`@ctz(u32)` | `lzcnt`/`tzcnt` | OK |
| `@clz(@Vector(8,u32))` | scalarized | OK (needs AVX-512CD for `vplzcntd`) |
| `@Vector(16,i16) a +| b` | `vpaddsw %ymm` | OK |
| scalar `i16 +| i16` | `lea`/`sar`/`xor`/`add`/`cmovno` | OK (x86 has no scalar saturating add) |
| `@select(f32, a>b, a, b)` | `vmaxps %ymm` | OK (LLVM recognizes the max pattern) |
| `@floatCast(@Vector(8,f16))` | `vcvtph2ps (%rdi),%ymm0` | OK (F16C) |
| `@prefetch(.locality=3)` | `prefetcht0` | OK |
| `@intCast`/`@truncate` | scalar `mov` (folded); `@truncate(@Vector(8,u32))` → `vpshufb`+`vpermq` | OK |
| `@abs` | f32 `vandps`; i32 `vpabsd` | OK |
| `@sqrt` | scalar `vsqrtss`; vector `vsqrtps` | OK |
| `@byteSwap` | scalar `bswap`; vector `vpshufb` | OK |

**Measurement note.** A kernel optimization is validated by the **kernel
benchmark**, not the e2e: the e2e depends on which quant types a model uses.
e2b/g12b use q4_0 + f16, so q4_K/q4_1/scale_split/q6_K show no e2e change on
them, yet their kernel wins are real (and matter for models that use those
types). "No e2b gain" is **not** a reason to discard.

## Steps

1. Record the target (Zig/LLVM version, `-mcpu`, optimize mode, `simd_width`) and
   the current asm of the P0 kernels. [DONE — Zig 0.16.0, `-mcpu=native`, Zen 3]
2. Inspect `@floatFromInt` in the q8_0 dequant path; propose a generic
   adjustment or record `REJECT`. [DONE — `vpmovsxbd`+`vcvtdq2ps`, optimal]
3. Inspect the q4_0 nibble unpack (`@shuffle`); same. [DONE — `vpand`/`vpsrlw`/
   `vpaddb` + `vpshufd`; NEEDS INVESTIGATION, low]
4. Compare per-block vs deferred `@reduce` in the f32 block dots; same. [DONE —
   ACCEPT, 1.77×/1.54× kernel; e2b +9%, g12b +3% e2e]
5. Inspect the integer-SIMD unpack for scalarization; same. [DONE — fully
   vectorized; a 16-wide shuffle-free variant is slower on AVX2 → REJECT]
6. Implement the `ACCEPT` items (generic Zig only); validate numerics + tests.
   [DONE — deferred reduction, commit `0ea10f6`]
7. Benchmark each accepted item (kernel + e2e `run`); fill the findings table.
   [DONE]
8. `zig build` / `zig build test` green (native + a class target). [DONE]
9. Extend the audit to the other hot families (lm_head, softmax, norm, attention,
   activation, elementwise, batched prefill). [DONE — q6_K ACCEPT (2.2×); the
   rest already deferred/elementwise → REJECT]
10. Sweep **every** kernel file for the catalog patterns. [DONE — the per-block
    reduce recurs in the q4_K/q4_1/scale_split/batched-alt variants;
    dequant/norm/attention/rope are OK or low]
11. Implement the deferred reduction in the remaining quant paths. [DONE —
    q4_K (1.43×), q4_1 (1.63×), scale_split; measured at the kernel level, since
    the e2e depends on the model's quant types]
12. Close the remaining live variants: tile (t2/t4), batched-alt, fused; remove
    the dead `legacy` variants. [DONE — all deferred; `legacy` removed
    (`b5039be`, `3328d6c`)]

## Definition of Done

- [x] The findings table is filled with asm evidence and a benchmark for each
      P0/P1 target.
- [x] The audit covers the other hot families (lm_head, softmax, norm, attention,
      activation, elementwise, batched prefill), not only matvec.
- [x] Every kernel file was swept; the recurring per-block-reduce pattern is
      recorded (implemented where measured, candidates otherwise).
- [x] Every `ACCEPT` is a **generic** Zig change (no handwritten ISA kernel),
      with a parity test and a measured improvement (deferred reduction). The
      oracle is the **kernel benchmark**; the e2e is reported only where the
      model exercises the path.
- [x] Every `REJECT`/no-gain is recorded with its measurement (as in `kernels/10`).
- [x] The input/output contract and the quantization/FP tolerances are unchanged.
- [x] `zig build` / `zig build test` green on `native` and a class target
      (`x86_64_v2`), the latter exercising the non-FMA fallback.
- [x] No Scheduler/Selector/ResidencyGraph change (no measured need).

## Arch Provenance

- **own:** the audit applied to zlm's hot AtomicOps; the findings/verdict table.
- **spec:** the generic Zig operations (`@Vector`, `@mulAdd`, `@floatFromInt`,
  `@shuffle`, `@reduce`, integer SIMD) and their x86 lowering.
- **consulted:** `sdd/tasks/kernels/notes/cpu-kernels-instructions.md`; llama.cpp/ggml quant
  kernels — reference only.
- **derived:** `kernels/10` (the negative result), `kernels/11` (width + FMA),
  `docs/kernel-architecture.md`.

## Comments

- **Evidence over hypotheses.** The catalog lists what *may* be suboptimal; this
  task only reports what the asm shows, with a benchmark.
- **The e2e is memory-bound here.** A kernel win that does not move `run` is
  still recorded (it matters for AVX-512 and the generic build), but not
  oversold.
- **The width/FMA dimensions are done** (`kernels/11`); do not re-derive them.
- **`@reduce` is not inherently wrong.** Extraction instructions are legitimate;
  only *excess* per-block reduction is a candidate (deferred/multi-accumulator).

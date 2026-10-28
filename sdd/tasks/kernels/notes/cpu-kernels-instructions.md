# Generic CPU Instruction Catalog (reference)

A dev-facing reference for auditing zlm's AtomicOps: where a **generic Zig
operation** can make Zig/LLVM emit better native instructions for the selected
target, **without** a handwritten ISA-specific kernel. This is a guide, not a
task; the audit itself is `sdd/tasks/kernels/12-instruction-audit.md`.

## Purpose

Express each mathematical operation once, in generic Zig, and let the compiler
specialize it for the target. The goal is **not** to maximize the number of
AVX-512 instructions; it is to remove accidental inefficiency (scalar fallback,
vector splitting, redundant conversions, unnecessary data movement) that generic
code can avoid.

```
Generic Zig operation -> compile-time specialization -> Zig/LLVM
  -> target-specific assembly -> correctness validation -> benchmark
```

A change is only accepted with **three confirmations**:

1. Correct generic Zig semantics (contract unchanged).
2. Improved or appropriately specialized assembly (inspected, not assumed).
3. Measurable performance improvement (benchmarked, not inferred).

## Principles

- **Instruction selection belongs to Zig/LLVM.** Algorithmic variant selection
  belongs to the Tuner; deterministic execution belongs to the AOTKernel.
- **Do not assume the widest vector is optimal.** A 512-bit `@Vector` on AVX2 is
  split into 2 ymm (measured ~1.7× slower than 256-bit in `kernels/10`).
- **`@Vector(N,T)` does not enable ISA features — the target (`-mcpu`) does.**
- **No handwritten ISA kernels, no vendor-specific branches.** Prefer
  compile-time width, generic vector ops, better data-flow, fewer conversions,
  deferred reductions, independent accumulators.
- **Numerics matter.** FMA/`@mulAdd` changes rounding; saturating arithmetic
  changes behavior; reordering reductions changes results. Keep the contract and
  the tolerances.
- **Benchmark the full AtomicOp**, not instruction counts or assembly appearance.

## The catalog

Each entry: the generic Zig operation, the native instructions it may lower to,
what to inspect, and the zlm relevance. "Feature" names the ISA that unlocks the
vector form (LLVM may pick equivalents; mnemonics are illustrative).

### 1. Vector width — `@Vector(lanes, T)`

- Candidates: `lanes = 4` (SSE/xmm), `8` (AVX/AVX2/ymm), `16` (AVX-512/zmm).
- Inspect: vector splitting, scalar fallback, register allocation, spurious
  extract/insert, the actual SIMD width emitted.
- zlm: **primary dimension**; mechanism done in `kernels/11` (plan-recorded
  `simd_width`, Tuner-selected). Feature: AVX/AVX2/AVX-512.

### 2. Fused multiply-add — `@mulAdd(Vec, a, b, acc)`

- Lowers to `vfmadd231ps` (ymm on AVX2+FMA, zmm on AVX-512) / `fmla` on NEON;
  a non-FMA target lowers it to a **libcall** (`fmaf`).
- Inspect: FMA emission, xmm/ymm/zmm usage, residual `vmulps`+`vaddps`,
  contraction/rounding.
- zlm: done in `kernels/11` (`simd.fma`, target-guarded). Feature: FMA.

### 3. Integer → float conversion — `@floatFromInt`

- Lowers to `vcvtdq2ps` (ymm/zmm); can degrade to repeated scalar `vcvtsi2ss`.
- Inspect: scalar vs vector conversion, intermediate widening, register moves,
  conversions per quantization block, interaction with width.
- zlm: **q4_0/q8_0 dequant** (`q8_0BlockDotLanes` does `@floatFromInt(@Vector(lanes,i8))`).
  Evaluate together with unpack + widening + scale. Feature: AVX/AVX2/AVX-512F.

### 4. Byte permutation — `@shuffle(u8, a, b, mask)`

- Lowers to `vpshufb` (ymm, **within 128-bit lanes**); AVX-512VBMI adds `vpermb`.
- Inspect: shuffle sequences, cross-lane overhead, redundant extract/insert,
  constant-mask optimization.
- zlm: **q4_0 nibble unpack** (element `j` = low nibble of byte `j`, `j+16` =
  high nibble). AVX-512F alone is not enough for every byte permutation.
- **Audited (asm):** the q4_0 nibble unpack is **bit-ops, not `@shuffle`**:
  `vpand 0x0F` (lo) + `vpsrlw $4` (hi) + `vpand 0x0F` — the standard extraction
  (llama.cpp's `srli_epi16` + `and`). **`vpsrld`/`vpslld` are not used and would
  be wrong**: `vpsrlw` is the 16-bit word shift (a dword shift would mix bytes
  across nibbles). Then `vpaddb`(±8), `vpmovsxbd`, `vcvtdq2ps`, `vfmadd`.
  Verdict **OK**.

### 5. Vector conditional — `@select(T, a > b, a, b)`

- Lowers to `vcmpps`+`vblendvps`; AVX-512 uses mask registers (`vblendmps`).
- Inspect: scalar branches, mask materialization, redundant compares, blend
  efficiency, mask-register use. Preserve NaN semantics.
- zlm: activations, clamps. Feature: AVX/AVX-512.
- **Audited (asm):** `@select(f32, a>b, a, b)` on AVX2 is optimized by LLVM to
  **`vmaxps %ymm`** (it recognizes the max pattern) — better than
  `vcmpps`+`vblendvps`.

### 6. Vector reduction — `@reduce(.Add, acc)`

- Horizontal reduction costs several instructions (`vextractf128`/`vaddps`/
  `vaddss`, or `vextractf32x8` on AVX-512). Extraction is **not** a bug.
- Strategies: per-block vs deferred reduction, multiple independent accumulators,
  reduce after several blocks.
- Inspect: reduction frequency in hot loops, accumulator dependency chains,
  scalar transitions. Reordering changes results.
- zlm: **matvec/matmul/dot/norm**. The block dots reduce once per block; a
  deferred/4-lane accumulate can amortize (cf. the AVX2 integer kernel).

### 7. Integer SIMD arithmetic — `@Vector(lanes, i32)`

- `vpaddd`/`vpsubd`/`vpmulld` (ymm/zmm). Inspect scalarization, splitting,
  widening/narrowing overhead, register pressure. Preserve overflow semantics.
- zlm: quantized arithmetic, integer accumulation.

### 8. Saturating integer arithmetic — `a +| b`, `a -| b`

- `vpaddsb`/`vpaddsw`/`vpsubsb`/`vpsubsw` (and ZMM variants under AVX-512BW).
- Inspect scalar saturation sequences, unnecessary clamps, integer width.
- **Saturation changes behavior** — only where the spec requires it.
- zlm: none today. Priority LOW.
- **Audited (asm):** `@Vector(16,i16)` `a +| b` → **`vpaddsw %ymm`** (single);
  scalar `i16 +|` is a multi-instruction sequence (`lea`/`sar`/`xor`/`add`/
  `cmovno`) — x86 has no scalar saturating add.

### 9. f16 → f32 conversion — `@floatCast(@Vector(lanes, f16))`

- `vcvtph2ps` (F16C: ymm←xmm; AVX-512F: zmm←ymm). Inspect scalar fallback,
  vector width, intermediate memory ops, throughput.
- **FP16 ≠ BF16** — do not reuse FP16 conversion for BF16.
- zlm: f16 tensor/KV paths. Feature: F16C.
- **Audited (asm):** `@floatCast(@Vector(8,f16))` → **`vcvtph2ps (%rdi),%ymm0`**
  (F16C), single instruction. (`floats.zig` already uses this form.)

### 10. Population count — `@popCount`

- Scalar `popcnt`; AVX-512VPOPCNTDQ adds `vpopcntd`/`vpopcntq`. AVX-512F alone
  is not enough. zlm: none today. Priority LOW.
- **Audited (asm):** scalar `@popCount(u64)` → **`popcnt`**;
  `@popCount(@Vector(8,u32))` on AVX2 is **scalarized** to a bit-trick
  (`vpand`/`vpsrlw`/`vpshufb`/…) — no `vpopcntd` without AVX-512VPOPCNTDQ.

### 11. Leading/trailing zeros — `@clz`/`@ctz`

- `lzcnt` (LZCNT), `tzcnt` (BMI1); AVX-512CD adds `vplzcntd`. Preserve zero-input
  semantics. zlm: none today. Priority LOW.
- **Audited (asm):** scalar `@clz(u32)`/`@ctz(u32)` → **`lzcnt`**/**`tzcnt`**;
  `@clz(@Vector(8,u32))` on AVX2 is **scalarized** — no `vplzcntd` without
  AVX-512CD.

### 12. Memory prefetch — `@prefetch(ptr, .{ .rw, .locality })`

- `prefetcht0/t1/t2`, `prefetchnta`. Inspect locality, distance, redundant
  prefetches, cache pollution, hardware-prefetcher interaction.
- AVX-512 does not imply better prefetching; hints may be ignored. Benchmark
  memory-bound workloads separately.
- zlm: sequential tensor access, streamed weights, memory-resident processing.
- **Audited (asm):** `@prefetch(p, .{ .rw = .read, .locality = 3 })` →
  **`prefetcht0`**.

### 13. Vector min/max (clamp) — `@max`/`@min`

- `@max(a, b)` / `@min(a, b)` lower to `vmaxps` / `vminps` (ymm on AVX2, zmm on
  AVX-512) — the **direct** min/max form. Prefer it over `@select` for clamping:
  `@select` costs a compare + blend (`vcmpps`+`vblendvps`), min/max is one
  instruction.
- Inspect: scalar fallback, vector splitting, and whether a clamp is expressed
  as `@max`/`@min` (good) rather than a branch or `@select`.
- **Audited (`kernels/12`):** `activation.zig` clamps with
  `@max(@min(g, 15), -15)`; the asm is `vminps`+`vmaxps` on **ymm** (AVX2), no
  scalar fallback, no compare/blend. The width is the tuner-selected one
  (`*Lanes`), so 8-wide on AVX2 (a hard-coded `@Vector(16,f32)` would split into
  2 ymm). `attention.zig`/`vector.zig` use scalar `@max` (one lane). On AVX-512
  (`-mcpu=x86_64_v4`) the 16-wide is a single `zmm`: `vmaxps %zmm`, and the clamp
  is `vminps %zmm` + `vmaxps %zmm` with an embedded broadcast `{1to16}`. Verdict
  **OK** — already optimal, and the width follows `simd_width`.

### 14. Broadcast scalar → vector — `@splat`

- `@splat(d)` on a `@Vector(lanes, f32)` lowers to `vbroadcastss` (AVX2: ymm,
  AVX-512: zmm) — one instruction, hoisted once and reused across a split.
- Inspect: `vbroadcastss` vs `vpermilps`/`vshufps`/`vmovss`+shuffle; redundant
  broadcasts inside a loop (should be hoisted).
- **Audited (`kernels/12`):** the deferred-reduce kernels broadcast the block
  scale `d` per block (`simd.fma(lanes, @as(V, @splat(d)), …)`). Asm: AVX2
  `vbroadcastss %xmm0,%ymm0` + `vmulps %ymm` (8-wide) / one `vbroadcastss` + 2×
  `vmulps %ymm` (16-wide split); AVX-512   `vbroadcastss %xmm0,%zmm0` + `vmulps
  %zmm`. Verdict **OK** — correct instruction, hoisted.

### 15. Integer width — `@intCast` / `@truncate`

- `@intCast` (widening) → `movsx`/`movzx` (or free, folded into the ABI);
  `@truncate` (narrowing) → `movzx` / a narrow store. On vectors,
  `@truncate(@Vector(8,u32))` → `vpshufb` + `vpermq` (a pack).
- Inspect: scalar fallback, vector splitting, redundant widen/narrow.
- **Audited (asm):** the scalar forms are a plain `mov` (no explicit
  movsx/movzx needed); `@truncate(@Vector(8,u32))` → `vpshufb %ymm` +
  `vpermq`. Verdict **OK**. zlm: `quant` narrowing.

### 16. Absolute value — `@abs`

- f32 `@abs` → `vandps` with a `0x7FFFFFFF` mask (clear the sign bit); i32 →
  `vpabsd`. Inspect: a branch (bad) vs the and/abs form.
- **Audited (asm):** `@abs(@Vector(8,f32))` → `vandps %ymm` (broadcast mask);
  `@abs(@Vector(8,i32))` → `vpabsd %ymm`. Verdict **OK**.

### 17. Square root — `@sqrt`

- Scalar → `sqrtss`; vector → `vsqrtps` (ymm/zmm). Inspect scalar fallback.
- **Audited (asm):** `@sqrt(f32)` → `vsqrtss`; `@sqrt(@Vector(8,f32))` →
  `vsqrtps %ymm`. Verdict **OK**. zlm: `norm` (`sumSq` uses `@sqrt`).

### 18. Byte swap — `@byteSwap`

- Scalar → `bswap`; vector → `vpshufb` (per-lane byte reversal). Inspect scalar
  fallback and the shuffle sequence.
- **Audited (asm):** `@byteSwap(u32)` → `bswap`; `@byteSwap(@Vector(8,u32))` →
  `vpshufb %ymm`. Verdict **OK**. zlm: none today (endianness is handled by the
  ZTEF/GGUF readers).

## Audit procedure

For each hot AtomicOp:

1. **Identify** the generic Zig operations and their hot paths.
2. **Record target:** Zig version, LLVM backend/version, arch, feature profile,
   optimize mode, SIMD width.
3. **Inspect asm:** expected native vs scalar fallback, vector splitting,
   redundant conversions, spills, excessive data movement, unnecessary
   horizontal reductions.
4. **Propose a generic adjustment:** compile-time width, generic vector ops,
   better data flow, fewer conversions, deferred reductions, independent
   accumulators. **No** handwritten assembly or ISA intrinsics.
5. **Validate:** numerical correctness, quantization semantics, FP tolerances,
   supported features, identical I/O contract.
6. **Benchmark:** current vs candidate — latency, throughput, memory behavior,
   full AtomicOp. Do not rely on instruction counts or assembly appearance.

## Report format

```
AtomicOp:            <family / atomic>
Source function:     <file:fn>
Zig operation:       <@Vector / @mulAdd / @floatFromInt / @reduce / …>
CPU target:          <e.g. x86_64_v3 (AVX2+FMA)>
Required ISA:        <feature(s)>

Current assembly:    <snippet>
Candidate assembly:  <snippet>

Inefficiency:        <what and why>
Adjustment:          <generic Zig change>

Numerical impact:    <none / tolerance>
Correctness:         <test(s)>

Baseline:            <metric>
Candidate:           <metric>
Improvement:         <relative>

Verdict:             ACCEPT | REJECT | NEEDS FURTHER INVESTIGATION
```

A `REJECT` (or no measurable gain) is a valid, recorded outcome — as in
`kernels/10`.

## Priority order

- **P0 — matvec hot path:** `@Vector`, `@mulAdd`, `@floatFromInt`, `@reduce`,
  integer SIMD.
- **P1 — quantization data transforms:** `@shuffle`, integer widening/narrowing,
  byte permutation.
- **P2 — other numerical ops:** `@select`, `@floatCast`, saturating arithmetic.
- **P3 — memory/auxiliary:** `@prefetch`, `@popCount`, `@clz`, `@ctz`.

## Constraints

- **MUST:** preserve generic AtomicOps, CPU-first, Zig/LLVM target-aware
  specialization, deterministic AOTKernel execution; validate with asm **and**
  repeatable benchmarks.
- **MUST NOT:** introduce handwritten AVX2/AVX-512 kernels; hardcode
  vendor-specific logic; assume zmm is always faster than ymm; confuse wider
  vectors with better throughput; treat theoretical instruction wins as measured
  wins; touch Scheduler/Selector/ResidencyGraph without a measured need.

## Related

- `sdd/tasks/kernels/10-kernel-performance.md` — the negative result (generic
  `@Vector` did not emit the integer dot; width/instruction were the levers).
- `sdd/tasks/kernels/11-kernels-isa.md` — the width + FMA mechanism (done).
- `sdd/tasks/kernels/12-instruction-audit.md` — this catalog applied to the hot
  AtomicOps.
- `docs/kernel-architecture.md` — the vocabulary.

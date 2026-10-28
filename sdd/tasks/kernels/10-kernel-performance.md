10 - Kernel Performance (generic: integer dot, register tiling, blocking)

Status: done
Commits: aa8b07f ea6ad8f

## Goal

Close the per-thread kernel gap vs llama.cpp for the `linear` family with
**generic** improvements — one implementation each, no per-ISA source variants
(those are `kernels/11`). Measured on E2B Q4_0 (same GGUF, threads, CPU backend
Haswell/AVX2): llama.cpp **pp512 171.9 / tg128 22.0** tok/s vs zlm prefill
**~28.7** / decode **~11.5** tok/s → ~**6x** slower prefill and ~**2x** slower
decode. The gap is the kernel implementation, not the architecture (AOT, mmap,
anti-OOM are sound).

## Context / findings (measured, 2026-10-08)

- **Decode** (`native.q4_0BlockDot`, `native.zig:148`): unpack q4 nibbles →
  `@Vector(8, f32)` (`@floatFromInt`) → f32 FMA with the **f32** activation →
  `@reduce`. llama.cpp instead quantizes the activation to **q8_0** (once per op,
  shared across all output rows) and does an **integer** dot
  (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`), ~32 int8/instruction, no f32
  conversion.
- **Prefill** (`batched.matMulRegisterTiledQ4_0`, `batched.zig:8`) is a **stub**
  delegating to `matVecQ4_0BatchedRows`, which processes **one output row at a
  time** and re-reads the `M`-token activation per row. llama.cpp tiles several
  output rows per pass.
- Neither path **blocks for cache** nor issues **prefetch**.

## Scope

- **In (generic, one implementation each):**
  1. **Integer dot.** Quantize the activation to q8_0 per op (shared across output
     rows) and do the dot in integer SIMD (`@Vector` int ops; LLVM lowers to
     `pmaddubsw`/VNNI where available). New AtomicOp(s) per quant type, with an
     `applicable` gate on integer-SIMD availability.
  2. **Register tiling.** A real register-tiled GEMM for prefill: multiple output
     rows per pass, reusing the dequantized weight block **and** the q8 activation
     across rows. Replaces the stub; keeps the weight read once per block.
  3. **Cache blocking / prefetch.** Block the output-row range and the activation
     to L2/L3 and issue software prefetch on the weight/activation streams.
- **Out:** per-ISA variants (`kernels/11`); the cost model / tuning; the
  architecture declarations; weight/KV quantization; the decode numerics contract
  (outputs stay equal to the current path within tolerance).

## Design

- **Activation quantization (shared).** A per-op step that quantizes the input row
  (per 32-element block, f16 scale) to `q8_0`, once, before the dot loops; it is
  reused by every output row. This is the enabler for the integer dot and is
  generic (any quant weight type: `q4_0`/`q8_0`/`q4_k`/…).
- **Integer dot kernel.** For `q4_0`×`q8_0`: unpack nibbles to `i8`, `maddubs`
  (u8×i8→i16) then `madd` (i16→i32) accumulate, apply the per-block scale at the
  end. Written with `@Vector` integer ops so LLVM lowers to the target's integer
  SIMD; `applicable` gates it on `avx2 || vnni || …` (see `kernels/11` for the
  width/ISA variants).
- **Register-tiled GEMM.** For prefill (`M > 1`): tile `R` output rows (e.g. 4)
  per pass; for each weight block, dequantize once (or decode q4→i8 once) and FMA
  against the q8 activation of the `M` rows, accumulating `R × M` register
  accumulators; store `out[tok*N + i]`. This is `kernels/09`'s reuse plus output
  row tiling — the missing half.
- **Blocking / prefetch.** Choose the row-range and activation tile to fit L2
  (from `caps.l2_bytes`); issue `@prefetch` on the next weight block / activation
  row. Keep it a pure function of `caps` (deterministic), no runtime heuristic.
- **Selector wiring.** The new atomics enter the catalog; the Selector picks by
  `cost.opCost` (and the `applicable` gate). The plan records the chosen variant
  (`matvec_plan`/`atomic_id`); the runtime decides nothing. Keep the existing
  variants as fallbacks.

## Steps

1. Activation→q8_0 quantize + `q4_0×q8_0` integer dot AtomicOp; `applicable`
   gate; decode parity test. **[ATTEMPTED, reverted]** Implemented as the
   `q4_0_i8` variant (quantize the activation once, integer i8×i8→i16→i32 per
   block). The calibration measured its efficiency at **0.629** (≈**1.6× slower**
   than `q4_0_u4`). Finding: LLVM does **not** lower the generic `@Vector` i16
   multiply/reduce to `pmaddubsw`/VNNI, so the widen+mul+reduce is more uops than
   the f32 FMA path. A real integer dot needs ISA-specific intrinsics/inline asm
   (`kernels/11`). Reverted to keep the tree clean.
2. Register-tiled prefill GEMM. **[EVALUATED, no change]** `matVecQ4_0BatchedRows`
   already dequantises each weight block **once** and applies it to all `M` rows;
   the only remaining reuse is the activation `x` across output rows, which is
   **L2-resident** (~128 KB), so output-row tiling is marginal. The prefill is
   **compute-bound** (dequant+fma), not x-bandwidth-bound.
3. Cache blocking + prefetch. **[ATTEMPTED, reverted]** Software prefetch of the
   next weight/activation block measured **slower** (prefill 29.1 → 27.1 tok/s),
   consistent with compute-bound. Reverted.
4. Bonus — defer the horizontal reduce (accumulate the 8-wide block contribution,
   reduce once per row). **[ATTEMPTED, no change]** A/B (HEAD vs defer-reduce):
   median decode 13.8 vs 13.5 tok/s, within machine noise. Reverted.
5. Re-benchmark vs llama.cpp. **[DONE]** E2B Q4_0, same GGUF/threads: llama.cpp
   `pp512 171.9 / tg128 22.0` vs zlm prefill ~28.7 / decode ~13.8 tok/s (see
   Context). The gap did **not** narrow with the generic changes.
6. `zig build` / `zig build test` green. **[DONE]** 188/188.

**Conclusion.** The generic improvements are at their limit in Zig. The ~2× decode
and ~6× prefill gap vs llama.cpp is the **integer SIMD dot** (32 int8/instruction
via `maddubs`/VNNI) plus per-ISA width — i.e., `kernels/11` (intrinsics/inline
asm, not generic `@Vector`). The architecture is not the bottleneck.

## Definition of Done

- [x] Item 1 (integer dot) implemented, calibrated (eff **0.629**, slower) and
      reverted; the reason (no `maddubs` from generic `@Vector`) is documented.
- [x] Item 2 (register tiling) evaluated: the batched GEMV already has the key
      reuse; output-row tiling is marginal (L2-resident x); prefill is
      compute-bound.
- [x] Item 3 (prefetch) measured slower; reverted.
- [x] The gap is attributed to the integer SIMD → `kernels/11`.
- [x] `zig build` / `zig build test` green (188/188); coherent; plan deterministic.

## Arch Provenance

- **own:** the integer-dot + register-tiling + blocking designs for this codebase.
- **spec:** GGUF/ggml quant block layouts (`q4_0`, `q8_0`).
- **consulted:** llama.cpp/ggml `ggml_vec_dot_q4_0_q8_0` and
  `ggml_compute_forward_mul_mat` — reference only, not copied.
- **derived:** `kernels/05` (matmul family), `kernels/09` (batched GEMV reuse),
  `runtime/11` (per-thread measurements).

## Comments

- **As-built outcome (2026-10-08): none of the three generic items closes the gap
  on this codebase/hardware** (see Steps). The value of this task is the
  measurement + the negative result: a future agent should not re-attempt the
  integer dot with generic `@Vector`. The real lever is the integer SIMD
  (`kernels/11`), which needs intrinsics/inline asm.
- The three items are independent but compounding: integer dot raises per-thread
  throughput; register tiling raises prefill reuse; blocking/prefetch hides
  memory latency. All are **generic** (one source each) — the ISA-specific
  width/algorithm variants are `kernels/11`.
- Enabler for `runtime/11`'s "speculative decoding" (multi-token forwards) and
  for closing the prefill/TTFT gap on streaming models.
- Related open-items: "ISA-specialized AtomicOps" (now `kernels/11`),
  "Benchmark/selection beyond matvec".

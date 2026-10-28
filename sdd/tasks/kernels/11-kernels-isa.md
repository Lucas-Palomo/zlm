11 - Kernels and ISA (generic AtomicOps + compile-time width variants)

Status: done
Commits: bfdf878 de1fc38 02a8b6b 15fb378 5fd9e3a 9f4c7dd

## Goal

Make the compute layer **hardware-agnostic in source and hardware-optimized by
profiling**: one generic AtomicOps source, compiled into a small set of
**compile-time width variants** (SIMD4/8/16) for the build target, with the
**Tuner** benchmarking them on the machine and recording the winner in the
`.zplan`; the **Runtime** executes the recorded variant. No hand-written
per-ISA kernel.

## Context

- zlm is **CPU-first, hardware-agnostic, optimized by profiling**: the source is
  portable; the machine-specific choice is a *plan* property decided at `tune`.
- The mechanism already exists: `api.Kernel` (`applicable`, `variant`, `cost`),
  `Family.kernels`/`run`, and the plan's `matvec_plan` field record a family's
  sub-variant; the Runtime dispatches it. This task adds a **width** dimension
  to that same pattern.
- **The earlier premise was wrong.** `kernels/10` measured the generic `@Vector`
  path as slower and concluded it needed per-ISA intrinsics/inline asm. The real
  cause is that the source **hard-codes a width** (`@Vector(16,f32)`) that does
  not match the target: on an AVX2 host a 512-bit vector is lowered to 2 ymm
  (measured **~1.7× slower** than `@Vector(8,f32)`), and the compiler does not
  rewrite it. The fix is not a per-ISA kernel but a **generic source
  parameterized by width**, with the width chosen per machine.
- `@Vector(N,T)` does **not** enable ISA features — the **target** does. The
  variants must be compiled for the target (`-mcpu=native` or a class); each
  width then lowers to whatever the target has.

## Scope

- **In:**
  - The generic-width AtomicOps pattern (one source, `comptime lanes`).
  - The compile-time variant table (SIMD4/8/16) and the `Width` plan code.
  - The width as a **plan property**; the Tuner selection; the Runtime dispatch.
  - The build model: `-mcpu=native` per host (zlm's case), or a class
    (`x86_64_v3`/`_v4`) for a portable binary.
  - The **fused multiply-add** (`simd.fma`): `@mulAdd` when the target has FMA,
    plain `a*b+c` otherwise (a non-FMA target would lower `@mulAdd` to a libcall).
- **Out:** the `cpu/` folder (deferred; the `*Lanes` routines live flat in the
  families for now — see Comments); a single `variants` dispatch table (each
  family switches on `Args.simd_width`); the GPU device tier (`gpu/`); the
  cost-model internals (`runtime/12`); the architecture declarations;
  **multi-variant cross-compiled binaries** (a follow-up "generic x86 build").

## Design

### Generic AtomicOps (one source, parameterized by width)

A hot kernel is written once, parameterized by `comptime lanes`:

```zig
fn dot(comptime lanes: usize, a: []const f32, b: []const f32) f32 {
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    // stride, vector type and tail all derive from `lanes`
    ...
}
```

Each `lanes` value is a real specialization (distinct `@Vector` type, distinct
stride and tail).

### Compile-time variants

`kernels/simd.zig` declares the set and the plan code:

```zig
pub const Width = enum(u8) { s4 = 4, s8 = 8, s16 = 16 };
pub const all = [_]Width{ .s4, .s8, .s16 };   // plan codes
pub const native: Width = ...;                 // from the build target
pub fn fromCode(raw: u8) Width;                // 0 = native
```

Each kernel exposes a `*Lanes(comptime lanes, …)` plus a thin wrapper at the
native width (for direct callers), and each family's `run` switches on
`Args.simd_width` to the matching instantiation — a per-family dispatch, not a
single global table.

### Fused multiply-add (`simd.fma`)

`@Vector` alone does not fuse: under Zig's strict float mode `acc += a*b` stays
`vmulps`+`vaddps`. `simd.fma(lanes, a, b, c)` returns `@mulAdd` when the target
has FMA (`vfmadd` on AVX2/AVX-512, `fmla` on NEON) and `a*b+c` otherwise — a
non-FMA target would lower `@mulAdd` to a libcall. The width picks the register
(8 → ymm on AVX2; 16 → zmm on AVX-512, split on AVX2); the target decides
whether it fuses.

### The build (native per host, or a class)

`-mcpu` decides what each width lowers to:

- `-mcpu=native` → the **exact host** (zlm's default); the host's registers.
- `-mcpu=x86_64_v3` (AVX2+FMA) / `x86_64_v4` (AVX-512) / `baseline` → a **class**.

So on an AVX2 host the binary contains `dot(8)`=ymm and `dot(16)`=2 ymm; on an
AVX-512 host `dot(16)`=zmm. The Tuner measures and picks. A width wider than the
target's native width is correct but split; the Tuner will not pick it. Each host
builds its own binary; a class build serves a whole class (the follow-up generic
build).

### Width as a plan property

The `.zplan` `Dispatch` records the chosen width (`simd_width`, next to
`matvec_plan`), so the Runtime never re-decides. Same AOT pattern as the family
variant.

### Tuner selection

`tune` benchmarks the instantiated widths on the machine (min-of-N, like the
existing variant-efficiency calibration) together with the hardware, the model
and the policy, and records the fastest in the plan. Deterministic on a fixed
`(caps, model, policy)`.

### Runtime dispatch

The family `run` switches on the recorded width and calls the matching
instantiation. One branch per dispatch, not per element.

### Fallback

The narrowest/portable instantiation is always applicable; selection never
returns zero; a wider variant is never the only candidate.

### Determinism

`applicable`/`cost` are pure functions of `caps`; the selection is
deterministic. Every width is numerically equal within tolerance (the ISA/width
choice never changes the text).

### Honest notes

- Memory-bound ops (`add`/`mul`/`softmax`) gain little from width; the lever is
  compute-bound (`dot`/matvec/attention).
- **Measure, don't assume:** on some AVX-512 parts (downclock) a narrower width
  can win; the Tuner's benchmark decides.
- The e2e gain may be small where the kernel is not the decode bottleneck; the
  mechanism is still the right foundation and the prerequisite for the portable
  multi-variant build.

### As-built — the width variants

`kernels/simd.zig`: `Width { s4, s8, s16 }` (the plan code is the lane count),
`all` (the instantiation set), `native` (from the build target) and `fromCode`.

The `q4_0`/`q8_0` block dots and matvecs are written once with `comptime lanes`
(`q4_0BlockDotLanes`/`matVecQ4_0Lanes`, `q8_0BlockDotLanes`/`matVecQ8_0Lanes`)
plus a `matVec*_Width(width, ...)` dispatch; `matVecRowsWithPlan` routes the
recorded width to the type's dispatch. The same pattern covers norm
(`rmsNorm*`/`addRmsNorm`/`qkNorm*`), attention (`fusedFlash*`/`twoPass*`/
`dotScaled`), `vector` (`dot`/`add`/`mul`/`softmax`/…), activation
(`gatedSilu`/`gatedGelu`) and elementwise; each family's `run` switches on
`Args.simd_width`.

`simd.fma` is applied to the `q4_0`/`q8_0` block dots, `norm.sumSq`, `vector.dot`,
attention (`dotScaled`/`fusedFlash*`/`attendHeadReference`) and the batched
prefill. `activation` has no mul-add (only `mul*mul`), so nothing to fuse.

The plan's `Dispatch.simd_width` records the choice (`0` = build native). The
tuner sets it from `caps.simd_width`, which the calibration fills by timing the
`q8_0` matvec at each width (`measureSimdWidth` → `Coeffs.simd_width`; `.zat`
v7). The executor passes it (`session.matVecPlan` → `pool.Job` →
`Args.simd_width` → family). No per-ISA source.

**Measured (Zen 3, `e2b-test`/`g12b`, before = `01b3fd0`):** width + `simd.fma`
give decode **16.5** tok/s on e2b (before **15.8**) and **3.7** on g12b (before
**3.6**), same generated text. `-Dcpu=x86_64_v2` (SSE4.2) → `simd=4`, decode
9.2 tok/s (the tuner picks the width; no FMA on SSE). `zig build test` green on
both (190/190). The gains are modest because the decode is **memory-bound** and
the `q4_0` matvec already used the native width before this task.

## Steps

1. `kernels/simd.zig`: `Width` (plan code) + `all` + the native default + the
   variant-table helper. [DONE]
2. Parameterize the hot kernels by `comptime lanes`, starting with the `q8_0`
   block dot + `matVecQ8_0`; then `q4_0`, `dot`, norm, attention. [DONE —
   `q4_0`/`q8_0` matvec, norm (`rmsNorm*`/`addRmsNorm`/`qkNorm*`), attention
   (`fusedFlash*`/`twoPass*`/`dotScaled`), `vector` (`dot`/`add`/`mul`/
   `softmax`/…), activation (`gatedSilu`/`gatedGelu`), elementwise.]
3. Add `simd_width` to the plan `Dispatch` (+ serialization, format bump). [DONE]
4. Calibration: measure each instantiated width on the machine. [DONE]
5. Runtime: dispatch on the recorded width. [DONE]
6. Parity per width; native vs class (`x86_64_v2`) `run` comparison. [DONE —
   native `simd=8`/15.4 tok/s, the current code `simd=4`/9.2 tok/s, same text.]
7. Document the pattern in `docs/kernel-architecture.md`. [DONE]
8. `zig build` / `zig build test` green; coherent; measured. [DONE — 190/190 on
   both native and `x86_64_v2`.]

## Definition of Done

- [x] The generic-width pattern and the build model are documented (spec +
      `docs/kernel-architecture.md`).
- [x] A hot kernel is written once with `comptime lanes` and instantiated for
      SIMD4/8/16 (no per-ISA source) — `q4_0`/`q8_0` block dots + matvecs.
- [x] The chosen width is a plan property; the Runtime never re-decides.
- [x] The Tuner benchmarks the widths and the selection is deterministic on a
      fixed `(caps, model, policy)`.
- [x] Every width is numerically equal within tolerance; the fallback is always
      applicable; selection never returns zero.
- [x] A native vs class (`x86_64_v2`) build produces the same text on `run`.
- [x] No GPU code is added here (CPU-first; the structure accommodates it).
- [x] `zig build` / `zig build test` green (native + `x86_64_v2`).

## Arch Provenance

- **own:** the generic-width pattern, the compile-time variant table, the
  width-as-plan-property design for this codebase.
- **spec:** x86-64 microarchitecture levels (`x86_64_v2`/`_v3`/`_v4`), CPU ISA
  feature sets (SSE/AVX2/AVX-512/NEON/SVE), Zig `-mcpu`.
- **consulted:** Zig `std.simd.suggestVectorLength`; GCC/Clang function
  multi-versioning (FMV) — reference only (Zig has no per-function
  `target_feature`; zlm uses compile-time variants + a plan-recorded choice).
- **derived:** `kernels/01` (Family/Registry), `kernels/07` (variant cost model),
  `kernels/10` (the negative result), `runtime/12` (cost pipeline),
  `CONSTITUTION` layers.

## Comments

- **Replaced the per-ISA intrinsics/asm idea.** A PoC measured `@Vector(8,f32)`
  at ~1.7× `@Vector(16,f32)` on an AVX2 host; the width, not the ISA, was the
  problem. One generic source + compile-time variants is simpler and keeps the
  catalog as the single owner.
- **Width is orthogonal to the family variant** (`matvec_plan`): the plan records
  both, and the Runtime applies both.
- **Native build per host** is zlm's model (`-Dcpu=native`): the binary is
  specialized for the machine that tuned it. A **generic build** (a class, or
  multi-variant cross-compiled) is a follow-up for distribution — then the
  AVX-512 variant is compiled with `-mcpu=x86_64_v4` even on a non-AVX-512 host,
  and the AVX-512 host selects it without its own build.
- **Why not a single portable binary across ISAs:** a width wider than the target
  is only *split*, never *widened*; and Zig has no per-function `target_feature`.
  So a binary compiled for a class contains only that class's widths.
- **The `cpu/` folder is deferred.** The `*Lanes` routines live flat in their
  families (`native.zig`, `norm.zig`, `attention.zig`, `vector.zig`, …); a `cpu/`
  grouping (3+ siblings) is a follow-up. The `gpu/` tier is separate.
- **Width and FMA are orthogonal, target-driven:** the width is a *plan* property
  (tuner-selected); FMA is a *build* property (`simd.fma` picks `@mulAdd` from
  the target). Both are comptime monomorphizations of one source — no per-ISA
  kernel.
- **Continued by `kernels/12` (Generic Instruction Audit):** the width/FMA
  dimensions are closed here; `12` applies the generic-operation catalog
  (`sdd/tasks/kernels/notes/cpu-kernels-instructions.md`) to the remaining data-transform ops
  (`@floatFromInt`/`@shuffle`/`@reduce`, integer SIMD) on the matvec/dequant
  path.

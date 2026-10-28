# kernels

title: Kernels
reason: Define the compute layer's structure and assemble kernel families so a
        new family is only its AtomicOps. Establishes the KernelFactory,
        KernelSet and Registry before porting any family.

## Tasks

### [01 - KernelFactory and Registry](01-kernel-factory.md)
Status: done

Context: the `KernelFactory` (builds a family's `Kernel` catalog + `select` +
`run`), the `KernelSet` (product) and the `registry.zig` (aggregated catalog the
Tuner consumes). Then `quant` is the first family.

### [02 - Kernel API Simplification](02-kernel-api.md)
Status: done

Context: give `kernels/` one responsibility (routines + catalog). Remove the
duplicated `Dispatch`/`Op`/`OpId` and the global `Args` union; each family gets
a typed `AtomicId` + `run(id, args)`; the registry exposes
`run(op, atomic_id, args)`.

### [03 - Kernels Layout](03-kernels-layout.md)
Status: done

Context: group whole families by domain (`linear/`, `transformer/`), rename the
catalog to `registry.zig`, drop the redundant `matvec.zig` facade and the dead
`factory.zig` name (the Factory is dev-time only). Reserve `linear/matmul/` for
the prefill/GEMM family (placeholder: `root.zig` only — no GEMM exists yet).

### [04 - Candidates and Recipes](04-candidates-and-recipes.md)
Status: done

Context: declare per-`AtomicOp` cost/scheduling metadata on `api.Kernel` and a
**recipe catalog** as data (enumerable by the Composer), and explore new fusion
recipes (norm+projection, matvec+bias, norm+rope+store, add+norm, dequant+scale).
Prerequisite for `runtime/09`'s Selector/Composer.

### [05 - GEMM and Cost-Model Selection](05-matmul-and-selection.md)
Status: done

Context: add the `matmul`/GEMM family and select a projection by execution
**phase** (decode -> GEMV, prefill -> GEMM).

### [06 - Family-Agnostic Selector](06-family-agnostic-selector.md)
Status: done

Context: generalize the cost-driven Selector beyond `matvec`/`matmul` — candidate
`applicable(type,shape)` + per-candidate cost over `registry.kernelsFor`, and a
first non-`matvec` trade-off (attention).

### [07 - Variant Cost Model](07-variant-cost-model.md)
Status: done

Context: the Selector does not choose the variant by cost — `applicable`
hardcodes the shape rule and `cost` is variant-invariant, so unroll never
reaches the Selector. Make the cost variant-aware (calibrated per-variant
efficiency, cached in the `.zat`) and drop the shape gate, keeping the selection
deterministic.

### [08 - Parallel Matvec Efficiency](08-parallel-matvec.md)
Status: done

Context: per-thread, the earlier local prototype and the current code are identical, but the current code's matvec scales only ~1.7x with `parts`
(the earlier local prototype ~3.7x @4T) — the decode gap. Diagnose and fix the kernel's parallel scaling.

### [09 - Batched GEMV (prefill weight reuse)](09-batched-gemv.md)
Status: done

Context: the "batched" prefill is M GEMVs (re-reads the weights per prompt row),
so a streaming model's prefill/TTFT pays the model read per token. Add a real
batched GEMV (read each weight row once, SIMD, apply to the M rows).

### [10 - Kernel Performance (generic: integer dot, register tiling, blocking)](10-kernel-performance.md)
Status: done

Context: close the per-thread gap vs llama.cpp (E2B: prefill ~6x, decode ~2x)
with **generic** improvements. **Outcome:** all three were implemented and
measured; none helps — the kernels are compute-bound and generic Zig `@Vector`
does not emit the integer `maddubs`/VNNI dot (measured eff 0.629, slower), so the
lever moved to `kernels/11`. The task's value is the measurement + the negative
result.

### [11 - Kernels and ISA (backend structure + per-ISA atomics)](11-kernels-isa.md)
Status: pending

Context: lock the **backend structure** (compute vs device runtime;
family-primary `kernels/<family>/<op>/<device>/`; the `Device` interface; the
always-applicable fallback) **and** the **per-ISA atomics** (width + the
integer/VNNI/AMX algorithm variants via intrinsics/inline asm, gated by
`applicable(caps)`). Merges the former 11 (ISA) + 12 (structure); the integer dot
is the primary lever (`kernels/10`). The GPU device tier is the `gpu/` milestone.

### [13 - Comptime Kernel Variants (PoC: q4_0)](13-comptime-kernel-variants.md)
Status: done

Context: one generic kernel body per quant, instantiated at comptime into N
variants from a curated candidate list (unroll / tile / width), so a new variant
is one list entry instead of a hand-written body; the Selector measures and
picks. The CPU counterpart of the `gpu/08` AtomicOp catalog. PoC: q4_0. Supplies
the generation mechanism `kernels/11`'s per-ISA variants will use.

### [14 - q4_0 Catalog: Naming + Extension](14-q4-0-catalog-naming.md)
Status: done

Context: rename the q4_0 variants to `q4_0_tile{T}_unroll{U}` and extend the
curated catalog (tile x unroll), sizing `VariantEfficiencies` (already full at
16) with a `.zat` bump. Depends on `kernels/13`.

### [15 - q4_0 Catalog: Dynamic Grid + Pruning](15-q4-0-dynamic-grid.md)
Status: done

Context: generate the q4_0 catalog at comptime from a five-axis grid (`width` x
`tile` x `unroll` x `int` x `prefetch`, 160 combinations) named
`q4_0_w{W}t{T}un{U}_i{I}p{P}`, bounded by the AVX2 register budget; the plan
**prunes before benchmarking** — the ISA prune drops the non-native `int`
widths (i4/i16 on AVX2), and a prefetch heuristic drops `p1` when not
streaming/memory-bound. The width folds in the separate `simd_width`; `scale` is
out of scope. Depends on `kernels/14`.

### [16 - q4_k Catalog: Dynamic Grid](16-q4k-dynamic-grid.md)
Status: done

Context: apply the `kernels/15` machinery (the five-axis grid + the P1/P2 prunes)
to **q4_k**, replacing the hand-written `q4_k_u8` with a generated catalog. q4_k
is ALU-bound on the packed 6-bit scale decode, so the `unroll`/`int`/`width` axes
should actually pay here (unlike the memory-bound q4_0). Depends on `kernels/15`.

### [17 - Generic Kernel (weight-family x dot x sweep)](17-generic-kernel.md)
Status: done

Context: one generic sweep template + a kernel catalog of comptime types in three
levels — **L1** the weight decode family (Q | F | BF | iq), **L2** the dot (fp |
int), **L3** the sweep (the variant). `kernel = L1 x L2`, `variant = L3`, named
`<kernel>_w{W}t{T}un{U}_p{P}`; no `switch (quant)`, no dot branch. Scope: the Q
family first. Depends on `kernels/16`.

### [18 - matvec Structure (layout x kernel x form)](18-matvec-structure.md)
Status: done

Context: reorganize `kernels/linear/matvec/` by its three axes — layout
(native/scale_split) x kernel (quantized/float/bfloat) x form
(single/fused/batched) — so `scale_split` stops looking like a kernel (it is a
**layout**) and `root.zig` stops mixing the dispatch/oracle/surface/tests. A
**mechanical** refactor: no behaviour change, the public surface unchanged,
268/268. Depends on `kernels/17`.

### [19 - Test Layout](19-test-layout.md)
Status: done

Context: move the tests + the test-only fixtures out of the production modules:
`<mod>_test.zig` per module, one `test.zig` aggregator per directory (the
cross-module tests), and a shared `fixtures.zig`. Apply to the matvec first.
Depends on `kernels/18`.

### [20 - F/BF Families](20-f-bf-families.md)
Status: done

Context: extend the dynamic grid to the F (f32/f16) and BF (bf16) families — the
same pattern as Q: L1 types + the catalog + the shared sweep; **fp-only** (no
nibble -> no int). The hand-written `matVecF32/F16/BF16` become the
`production_fp`. Includes the test layout (`kernels/19`). Finishes the matvec
families. Depends on `kernels/17`.

### [21 - Generic Dots](21-generic-dots.md)
Status: done

Context: make the matvec's row-dot helpers generic — one `dot(comptime K, row, x,
xsum?)` on the L1 contract, replacing the per-quant `dotQ8_0`/`dotQ4_K`/
`dotQ4_KPre`/`dotQ6_K` used by `lmHeadRows` (the `lm_head`). Preserve the q4_k
`xsum` precompute (no regression). The `q5_k` is a **dequant**, not a matvec (out
of scope). Depends on `kernels/20`.

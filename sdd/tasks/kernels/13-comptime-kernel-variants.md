13 - Comptime Kernel Variants (PoC: q4_0)

Status: done
Commits:

## Goal

Instead of N hand-written kernel bodies per variant (unroll / tile / width),
write **one generic body per quant** and instantiate it at comptime into N
variants from a curated candidate list; the Selector measures and picks. This is
the "own" way to grow the kernel catalog — the CPU counterpart of what `gpu/08`
established for the GPU (a catalog of AtomicOps), instead of porting ggml's
hand-written variant zoo.

PoC: the **q4_0** matvec (the simplest family member), to validate the pattern
before extending to q4_k / q6_k / q8_0 / q4_1 and the other families.

## Scope

- **In:** a comptime `Variant` descriptor; a generic q4_0 matvec body; a curated
  candidate list; the generated registry kernels + `MatVecPlan` ids; the
  calibration/Selector integration.
- **Out:** the other quants; the GPU (GLSL source generation, later); new
  algorithms (the integer/VNNI path stays its own variant).

## Design

### Variant descriptor + generic body

- `Variant = struct { width: u8 = 8, tile: u8 = 1, unroll: u8 = 1 }` — the q4_0
  axes today (`width` {4,8,16}, `tile` {1,2,4}) plus `unroll` (blocks per
  iteration with U independent accumulators, a **new** axis).
- `fn matVecQ4_0Gen(comptime V: Variant, data, n, m, x, out, begin, end)` — one
  body; `inline for` over `tile` (rows) and `unroll` (blocks); `@Vector(V.width)`.
  The hand-written `matVecQ4_0Lanes` / `matVecQ4_0TileRows` become instantiations
  of this body (behaviour preserved).

### Curated candidate list

- `q4_0_candidates: []const Variant` — a **chosen** set (not the full cartesian;
  some combinations are pointless): the current variants plus the new unroll
  factors.
- The registry's q4_0 kernels and the `MatVecPlan` ids are **generated from the
  list** (deterministic order, stable ids).

### Selector

- The existing calibration (`variant_eff`) measures each generated variant; the
  Selector picks — no new selection code.

## Steps

1. **Done.** `Variant{width,tile,unroll}` + `matVecQ4_0Gen(comptime V, ...)` — one
   body; `matVecQ4_0Lanes`/`matVecQ4_0TileRows` are thin instantiations
   (bit-identical for those descriptors).
2. **Done.** A curated `q4_0_candidates` list drives `matVecRowsWithPlan`'s
   dispatch, `planApplicable`, `bench.candidatesFor` and the calibration walker.
   The enum tags stay hand-written (the ordinal is the serialized id): 5 appended
   after `q4_k_u8`.
3. **Done.** The calibration measures all 8 q4_0 candidates; the Selector picks
   (no new selection code).
4. **Done.** Parity for all 8 (1e-5); 263/263; CPU decode unchanged within noise
   (E2B ~15.5, 12B ~3.7). The E2B plan switched `q4_0_u4 -> q4_0_ur4` (a new
   variant).

## Definition of Done

- [x] One generic q4_0 body replaces the hand-written variants.
- [x] The candidate list drives the registry + the plan ids (the id still needs
      one hand-appended enum tag — see Comments); a new variant's body is one
      list entry.
- [x] Parity holds for every generated variant; the Selector picks the fastest.
- [x] CPU decode measured; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the comptime variant generator.
- **consulted:** — (the opposite of ggml's hand-written variant zoo).
- **derived:** `gpu/08` (the AtomicOp catalog), `kernels/11` (per-ISA atomics).

## Comments

- **PoC first:** q4_0 only. If the pattern holds, extend to the other quants and
  families, then the GPU via GLSL source generation (a template with
  `#define UNROLL N`).
- `kernels/11` already plans per-ISA AtomicOps gated by `applicable(caps)`; this
  task supplies the generation mechanism those variants will use.
- The serialized `MatVecPlan` id must stay deterministic/stable across builds
  (the build fingerprint guards staleness).
- **As-built (PoC works).** One body, a list-driven catalog, a genuinely new
  `unroll` variant shipped and selected. Friction found:
  - The **serialized enum** is the limit: a new id still needs one hand-appended
    `MatVecPlan` tag (the ordinal is the stable id; tags are used by name), so
    the enum cannot be generated without losing named ids.
  - `VariantEfficiencies.max_variants` is now **full (16)** — extending the
    catalog needs a `.zat` version bump + re-tune.
  - `variant_eff` clamps to `1.0`, so near-best candidates tie and catalog order
    decides (a noisy winner: E2B chose `ur4`, 12B `u4`); a measured-time
    tie-break would stabilise it.
  - `q4_0_t4` applicability shifted `lanes >= 16` -> `lanes >= width` (8): the old
    gate contradicted the module's "interchangeable variants are shape-free" rule.

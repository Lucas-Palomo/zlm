04 - Candidates and Recipes

Status: done
Commits:

## Goal

Give the deterministic tuner (`runtime/09`) something to *enumerate*: declare the
kernel families' **candidates** (cost/scheduling metadata per `AtomicOp`) and a
**recipe catalog** as data, and **explore new fusion recipes**. Today only
`matvec` has interchangeable variants (`MatVecPlan`); recipes exist in code
(`fusedGateUpRows`, `qk_norm_rope_heads`) but are neither declared nor used.

## Scope

- **In:**
  - **Candidate metadata** on `api.Kernel`: the cost/scheduling facts the Selector
    and Scheduler read — `row_parallel`, `scratch` (exist), plus `applicable
    (type, shape)`, a `cost` model (`bytes_effective`/`flops`), `tier`,
    `complexity` (passes / memory scaling).
  - **Recipe catalog as data**: named recipes (`single | fused | seq`) with their
    input/output op signatures, preconditions (shared inputs/slots, dims) and
    expected benefit (bytes/passes saved) — enumerable by the Composer.
  - **New recipes** — explore and implement fusions found in the gemma4 block.
- **Out:** the tuner pipeline (`runtime/09`); GPU kernels; measured selection.

## Design

### Candidate metadata (`api.Kernel`)
Extend the catalog entry so the Selector can rank without knowing the family:
`{ id, op, atomic_id, row_parallel, scratch, tier, applicable(type,shape),
cost: { bytes_fn, flops_fn, complexity } }`. `bytes_fn`/`flops_fn` are dev-time
functions (or a small enum + params) giving `bytes_effective`/`flops` for a
shape, so the cost model can compare candidates that move the same *logical*
bytes (tiling/unroll). Families fill this in at assembly (`api.erased`/`Family`).

### Recipe catalog (data)
Declare recipes alongside the families (dev-time), e.g.
`Recipe { name, form, ops: []OpSig, atomic_id, preconditions, bytes_saved }`.
The Composer enumerates applicable recipes over adjacent ops; the cost model
scores each. Start with the existing ones (`gate_up_fused`, `qk_norm_rope`) and
add as implemented.

### Recipes to explore (from the gemma4 block)
- **norm + projection** (`attn_norm`→q/k/v, `ffn_norm`→gate/up): fuse the RMSNorm
  into the matvec, avoiding the `normed` round-trip.
- **gate + up + activation**: `fusedGateUpRows` exists — declare + wire it.
- **q/k norm + rope** (and **rope + write_kv**): fuse the norm/rope/store path.
- **matvec + bias** (`matvec_bias`).
- **residual add + norm**: `add(hidden)` then `*_norm` in one pass.
- **dequant + scale**: embedding row dequant with the embed scale.
- (harder, later) **attention + o_proj**.

Each new recipe must be numerically equivalent to its unfused sequence (tests).

## Arch Provenance

- **own:** the candidate-metadata shape and the declared recipe catalog.
- **spec:** ggml layouts (for the cost models).
- **consulted:** llama.cpp fused kernels (reference only).
- **derived:** from `kernels/01..03` (families/registry) and the Selector/Composer
  contract in `docs/kernel-architecture.md`.

## Steps

1. Extend `api.Kernel` with the cost/scheduling metadata (applicable/cost/tier/
   complexity); expose it via `registry.zig`.
2. Declare a **recipe catalog** as data; enumerate it.
3. Annotate `matvec`'s `MatVecPlan` variants (the first interchangeable set).
4. Declare existing recipes (`gate_up_fused`, `qk_norm_rope`).
5. Explore/implement new recipes (norm+projection, matvec+bias, norm+rope+store,
   add+norm, dequant+scale), each verified equivalent.
6. Tests: every `AtomicOp` exposes metadata; recipes enumerate; fused ≡ unfused
   (numerically) per recipe.
7. `zig build` / `zig build test` green; a fused plan runs coherent (E2B).

**As-built.**
- `api.zig`: `Kernel` gains `tier`/`complexity`/`variant`; each family exposes
  `kernels: []const Kernel` (matvec exposes every `MatVecPlan` variant).
  `registry.kernels()`/`kernelsFor(op)` aggregate the catalog.
- `kernels/recipes.zig`: a `Recipe` catalog (data) — `gate_up_fused`,
  `qk_norm_rope`, `norm_proj`, `rope_write_kv`, `add_norm`, `dequant_scale`,
  `matvec_bias`; `registry.recipes()` returns it.
- New `AtomicOp`s (appended enums, so existing plan ids stay stable):
  `matvec.{gate_up_fused, norm_proj, matvec_bias}`,
  `attention.rope_write_kv`, `norm.add_norm`, `quant.dequant_scale`
  (`qk_norm_rope_heads` pre-existed).
- Each recipe has an equivalence test (fused vs separate ops, `1e-5`); fused
  routines: `fused.normThenMatVecRows`, `norm.addRmsNorm`,
  `attention.ropeWriteKv`, `quant.dequantScaledTo`.
- Verified: `zig build` / `zig build test` green (101/101).

## Definition of Done

- [x] Every `AtomicOp` exposes scheduling/complexity metadata (`tier`,
      `complexity`, `variant`, `row_parallel`, `fusable`, `scratch`) via the
      registry. *(`applicable(type,shape)` / `cost` fns live in `runtime/09`.)*
- [x] A recipe catalog is declared as data and enumerable.
- [x] Existing recipes are declared; `matvec` variants are annotated.
- [x] The norm+projection, matvec+bias and norm+rope+store recipes are
      implemented and verified equivalent to the unfused path.
- [x] `zig build` / `zig build test` green (101/101).

## Comments

- This is the prerequisite for `runtime/09`'s Selector/Composer; Phase 1 may ship
  with `matvec` only and grow the catalog incrementally.
- Recipe equivalence is the safety contract: a fused recipe must match the
  unfused result within tolerance; the tuner only *chooses*, it does not change
  semantics.

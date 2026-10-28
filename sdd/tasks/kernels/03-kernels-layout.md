03 - Kernels Layout and Family Split

Status: done
Commits:

## Goal

Give `kernels/` a layout that mirrors its parts — contract, catalog, families
grouped by domain — and make room for splitting the linear-algebra family into
the families the model needs (`matvec` for GEMV, `matmul` for GEMM), without
inventing algorithms that do not exist.

## Scope

- **In:** `linear/` + `transformer/` grouping; `registry.zig`; remove the
  `matvec.zig` facade; merge `quant/factory.zig`; add the `matmul` family as a
  placeholder (`root.zig` only); record the `softcap`/`elementwise` gaps.
- **Out:** writing real GEMM algorithms. None exist today: `matMulRegisterTiled*`
  are aliases of `matVec*BatchedRows` and `matMulRows` falls back to
  `matVecRows` for `M == 1` (`src/kernels/linear/matvec/{batched,root}.zig`).

## Design

- **Unit is the family** (`one Op = one Family`); group whole families, never
  split a family by technique.
- `linear/` — weight ops: `quant/`, `matvec/` (GEMV, the real algorithms),
  `matmul/` (GEMM; placeholder today), plus shared `types.zig`/`vector.zig`.
- `transformer/` — model ops: `norm`, `rope`, `activation`, `attention`
  (+ future `elementwise`).
- **Finding:** there is no GEMM implementation; `matmul` is batched GEMV inside
  `matvec/`. So `matmul/` starts as `root.zig` only and is filled when a real
  GEMM lands. It is **not registered** yet (no consumer until prefill).

Target:

```text
kernels/
  api.zig
  registry.zig
  linear/
    types.zig            # shared (future)
    vector.zig
    quant/
    matvec/              # family matvec (GEMV): native|scale_split|batched|fused|root
    matmul/              # family matmul (GEMM): root.zig only (placeholder)
  transformer/
    norm.zig  rope.zig  activation.zig  attention.zig
```

## Arch Provenance

- **own:** the domain grouping, the family-integrity rule, the placeholder.
- **spec:** —.
- **consulted:** —.
- **derived:** from `02` (one API) and `docs/kernel-architecture.md`.

## Steps

1. **(as built)** Created `linear/` and `transformer/`; moved `quant/`,
   `matvec/`, `vector.zig` and the four transformer families.
2. **(as built)** Deleted the redundant `matvec.zig` facade.
3. **(as built)** Merged `quant/factory.zig` into `quant/root.zig`.
4. **(as built)** `root.zig` -> `registry.zig` (thin catalog; dropped family
   re-exports); updated importers and `../` depths.
5. **(as built)** Updated `docs/kernel-architecture.md` (code map,
   family-assembly, registry, diagram).
6. **(as built)** Created `linear/matmul/root.zig` as a placeholder (TODO only):
   no family, no registration, no `api.Op.matmul` yet.
7. `zig build` / `zig build test` green.

## Definition of Done

- [x] `kernels/` is `api.zig`, `registry.zig`, `linear/`, `transformer/`.
- [x] No `matvec.zig` facade; no `factory.zig`.
- [x] `linear/matmul/` exists with only `root.zig` (placeholder).
- [x] `zig build` / `zig build test` green.

## Comments

- **Deferred:** fill `matmul/` and add `kernels.Op.matmul` when a real GEMM
  lands; then the Scheduler maps `graph.matvec` by phase (decode -> `matvec`,
  prefill -> `matmul`) and `graph.lm_head` -> `matvec`.
- **Deferred:** `softcap` as an AtomicId of `activation`; the residual `add` as
  the `elementwise` family (both are semantic ops without a kernel today).
- **Deferred:** move `matvec/types.zig` to `linear/types.zig` once `matmul`
  actually shares it.
- `linear/` has 3 immediate siblings (`quant/`, `matvec/`, `matmul/`), so it
  respects "directory only with 3+ siblings".

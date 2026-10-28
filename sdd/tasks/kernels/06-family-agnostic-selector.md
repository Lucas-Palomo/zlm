06 - Family-Agnostic Selector

Status: done
Commits: 740508e

## Goal

`kernels/05` added the `matmul`/GEMM family and a phase-aware projection chooser,
but selection is still per-op code (`selectMatvec`, `selectMatmul`). This task
makes the Selector **family-agnostic**: enumerate `registry.kernelsFor(op)`,
filter by an `applicable(type, shape)` predicate, and pick `argmin cost.opCost` —
realizing the `applicable`/cost metadata deferred in `kernels/04`. It also brings
a first **non-matvec trade-off** (attention) so the generic path is exercised
beyond `matvec`/`matmul`.

## Scope

- **In:**
  - **`applicable(type, shape)` + per-candidate cost** on `api.Kernel` (the
    metadata deferred in `kernels/04`), so a candidate declares when it applies
    and how to price it.
  - **Generic Selector** `select(op, ctx)`: enumerate `registry.kernelsFor(op)`,
    filter `tier = cpu` + `applicable`, `argmin opCost` over candidates, then
    `parts = argmin opCost (<= threads)`. `selectMatvec`/`selectMatmul` become
    per-family `applicable`/cost data.
  - **A non-matvec trade-off**: attention (e.g. fused-flash vs split-k/tiled),
    declared as candidates + cost so the generic Selector chooses by cost.
- **Out:** new kernels beyond attention; GPU; measured selection (still the
  deterministic cost model); recipes (unchanged).

## Design

### Candidate metadata
Extend `api.Kernel` (or its assembly) with:
- `applicable` — when the candidate applies (quant type, min/max dims), as data
  (a small enum/params), not family code;
- `cost` — the per-candidate cost inputs (`bytes`/`flops`) for `cost.opCost`.

The Selector reads these without knowing the family.

### Generic selection
```text
candidates = kernelsFor(op) where tier = cpu and applicable(type, shape)
best       = argmin over candidates of opCost(op, c.bytes(shape), c.flops(shape), parts, ...)
parts      = argmin opCost (<= threads)
```

`matvec`/`matmul` migrate their ISA×shape rule into `applicable`/`cost`; families
with a single candidate keep a trivial one.

### Attention trade-off (first non-matvec)
Declare attention candidates (fused-flash vs split-k/tiled) with their cost, so
the generic Selector can choose. Each must be numerically equivalent (tests).

### Determinism
Pure function of (declaration, calibrated `caps`); identical `.zplan` per device.

## Steps

1. Add `applicable`/`cost` to the candidate catalog (`api.Kernel`), data-only.
   [TODO]
2. Implement the generic `tuner.select(op, ctx)`; migrate `matvec`/`matmul`. [TODO]
3. Add the attention trade-off candidates + cost; select by cost. [TODO]
4. Tests: filtering by `applicable`; argmin by cost; equivalence per candidate;
   determinism; plans unchanged where the choice is unchanged. [TODO]
5. `zig build` / `zig build test` green; E2B coherent. [TODO]

**As-built.**
- `api.Kernel` gains `applicable(type, m, n, lanes)` + `cost(type, m, n, k, lanes)
  -> Cost { bytes, flops }` (primitives only, so `kernels/` never imports
  `tuner/`). `registry.zig` re-exports the types.
- `tuner.select(op, SelectCtx)` enumerates `registry.kernelsFor(op)`, filters
  `tier = cpu` + `applicable`, ranks by `cost.opCost`, derives `parts` with the
  strict-improvement loop. `selectMatvec`/`selectMatmul`/`selectProjection` are
  thin wrappers; `matvec`/`matmul` variant rules became candidate metadata
  (plan byte-identical).
- Attention trade-off: new `AtomicId.two_pass_head` (materialize scores +
  softmax + reduce V) alongside `fused_flash_head`; both declare `applicable`/
  cost; attention selection runs through the generic Selector in `tune.zig`.
  On this device the roofline keeps `fused_flash_head` (no churn); two-pass is
  selected on compute-bound caps and is executable.
- Verified: `two_pass_head` ≡ `fused_flash_head` (`1e-5`); 153/153 tests; E2B
  plan unchanged (749 decode / 869 prefill); run coherent.

## Definition of Done

- [x] `api.Kernel` carries `applicable`/cost; the Selector enumerates + filters.
- [x] `matvec`/`matmul` selection is expressed as candidate metadata.
- [x] >=1 non-matvec family (attention) is cost-selected among variants.
- [x] Every selected candidate is numerically equivalent to the baseline.
- [x] Deterministic plan; `zig build` / `zig build test` green; E2B coherent.

## Arch Provenance

- **own:** the family-agnostic Selector contract.
- **spec:** BLAS / attention tiling.
- **consulted:** llama.cpp/ggml attention variants — reference only.
- **derived:** `kernels/04` (candidate metadata), `kernels/05` + `runtime/09`
  (cost model).

## Comments

- Continues `kernels/05`, which landed the `matmul` family + phase selection.
- Realizes the `applicable(type, shape)`/cost metadata deferred in `kernels/04`.
- Keep plans unchanged where the cost choice does not change (no gratuitous plan
  churn).

05 - GEMM and Cost-Model Selection

Status: done
Commits: 819705c

## Goal

`runtime/09` shipped one cost-driven selector (`tuner.selectMatvec`, GEMV) and
`kernels/04` shipped the candidate/recipe catalogs. Everything else is either
picked by a fixed mapping (`compile.defaultAtomic`) or missing: there is no GEMM
(`linear/matmul/root.zig` is a TODO stub, and the runtime prefills token-a-token,
so prefill runs as batched GEMV inside `matvec/`). This task expands cost-model
selection:

1. add the `matmul` (GEMM) family;
2. select by execution **phase** (decode -> GEMV, prefill -> GEMM);
3. grow the candidate/cost coverage so the Selector has trade-offs to choose
   beyond `matvec`.

## Scope

- **In:**
  - **`matmul`/GEMM family**: `kernels.Op.matmul`, an `AtomicId` + `Args` +
    `run`, a `MatMulPlan` variant catalog, and registration in `registry.sets`.
  - **Real GEMM** routine(s): a blocked/tiled matmul with at least two
    interchangeable variants (tiling / vectorization) for the Selector.
  - **Phase-aware Scheduler**: map a projection dispatch to `matvec` (decode,
    M=1) or `matmul` (prefill, M>1) by the plan `phase`, and cost-score both.
  - **Family-agnostic Selector**: enumerate `registry.kernelsFor(op)`, filter by
    `applicable(type, shape)`, pick `argmin cost.opCost`. `matvec` becomes the
    first family using it; add `applicable`/cost where a real trade-off exists.
  - **Candidate metadata completion**: realize the `applicable(type, shape)` +
    cost fn deferred in `kernels/04`.
- **Out:** GPU kernels; measured/autotuned selection (still the deterministic
  cost model); arbitrary op-graph fusion (recipes stay declared); the runtime's
  batched prefill itself (a runtime task — see Comments).

## Design

### matmul family
Mirror the `matvec` family shape: typed `AtomicId` (`MatMulPlan`: blocked,
ISA-wide, naive, ...), an args struct (A/B/out views, dims M/N/K, quant type),
`run(id, args)`, and a `set` with a `kernels` catalog. Register `matmul` in
`registry.sets` and add `kernels.Op.matmul`. Quant (dequant-on-the-fly) mirrors
matvec's codecs; the value is GEMM-style reuse of the A-tile across N.

### Phase selects GEMV vs GEMM
A projection is one semantic op (`graph.matvec`) whose optimal kernel depends on
the rows to process:
- **decode** (one token, M=1): GEMV, memory-bound -> `matvec`;
- **prefill** (prompt batch, M>1): GEMM, compute-bound -> `matmul`.

The Scheduler sets `Dispatch.op` and `Dispatch.phase` accordingly. `cost.opCost`
gains the GEMM model (`bytes_effective` for GEMV vs flops/tiling for GEMM) so
the choice — and the tiling/`parts` — is cost-driven, not hard-coded.

### Family-agnostic Selector
`selectMatvec` generalizes to `select(op, ctx)`:

```text
candidates = registry.kernelsFor(op)   filtered by tier=cpu + applicable(type,shape)
variant    = argmin over candidates of cost.opCost(op, variant, ctx, caps)
parts      = argmin opCost (<= threads)
```

`applicable(type, shape)` lets a candidate declare when it applies (quant type,
min/max dims); `cost` gives its per-candidate bytes/flops. Families only add
these where there is a genuine trade-off (moving the same *logical* bytes
differently):
- `matvec` / `matmul`: variant + tiling by ISA×shape (exists -> generalize);
- `attention`: fused-flash vs split-k / tiling (candidates to add);
- `norm` / `quant`: only if a real alternative exists (e.g. fused vs separate).

No-gain / single-candidate families keep their fixed mapping.

### Determinism
Everything stays a pure function of (declaration, calibrated `caps`); no timing
gate. Two tunes on the same device produce the identical `.zplan`.

## Steps

1. Add `kernels.Op.matmul` and `kernels/linear/matmul/` (`AtomicId`/`Args`/`run`/
   `set` with a `MatMulPlan` catalog); register in `registry.sets`. [TODO]
2. Implement a real blocked/tiled GEMM with >=2 variants. [TODO]
3. Add the GEMM cost entry to `tuner/cost.zig` (`quad_flops`; flops + bytes) and
   teach `opCost`/`planScore` about matmul. [TODO]
4. Phase-aware Scheduler in `tune.zig`: choose `matvec` vs `matmul` by phase
   (decode vs prefill) via the cost model; set `Dispatch.op`/`phase`. [TODO]
5. Generalize the Selector to `registry.kernelsFor` + `applicable` + `argmin
   opCost`; keep `selectMatvec` as the matvec case. [TODO]
6. Extend `applicable`/cost to the next family with a real trade-off (attention
   tiling) and any norm/quant alternative. [TODO]
7. Tests: GEMM ≡ batched GEMV (within tolerance); phase mapping decode/prefill;
   Selector picks by cost and is deterministic; candidate filtering. [TODO]
8. `zig build` / `zig build test` green; E2B coherent; plan deterministic. [TODO]

**As-built (slice 1).**
- `Op.matmul` + `linear/matmul/` family: `AtomicId = { rows, blocked }`; `rows`
  loops the existing per-row GEMV, `blocked` dequantizes each weight row once and
  reuses it across the M input rows. Registered in `registry.sets`.
- `cost.zig`: `.matmul` is compute-bound (flops) so `opCost` prices it.
- `tuner.selectMatmul` + `Phase`/`Projection`/`selectProjection` (decode ->
  matvec, prefill -> matmul); `tune` routes projections via the chooser.
- Verified: GEMM ≡ M× GEMV per quant type (`1e-5`); decode plan byte-identical;
  run coherent; 146/146 tests.
- Follow-up: the family-agnostic `applicable`/cost enumeration and a non-`matvec`
  trade-off (attention) are `kernels/06`; the runtime batched-prefill side is
  `runtime/10` (done).

## Flow / Arch

```text
projection (graph.matvec)
        │  phase
        ├── decode (M=1) ──▶ matvec   (GEMV, bytes-bound)
        └── prefill (M>1) ─▶ matmul   (GEMM, flops-bound)
                 │
        Selector : kernelsFor(op) -> applicable -> argmin opCost
                 │
        Scheduler: op + variant + parts + residency
```

## Definition of Done

- [x] `matmul`/GEMM family registered with a variant catalog + cost metadata.
- [x] The Scheduler selects `matvec` (decode) vs `matmul` (prefill) by phase.
- [x] The Selector selects among `matvec`/`matmul` candidates and by phase
      (decode -> matvec, prefill -> matmul). *(The family-agnostic enumeration
      and a non-`matvec` trade-off moved to `kernels/06`.)*
- [x] GEMM ≡ batched GEMV numerically (within tolerance).
- [x] Deterministic plan; `zig build` / `zig build test` green; E2B coherent.

## Arch Provenance

- **own:** the phase-based `graph.matvec -> matvec|matmul` mapping and the
  family-agnostic Selector contract.
- **spec:** BLAS GEMM tiling; ggml quant layouts.
- **consulted:** llama.cpp/ggml GEMM (`ggml_compute_forward_mul_mat`) —
  reference only, not mirrored.
- **derived:** `kernels/04` (candidate metadata) and `runtime/09` (cost model,
  `selectMatvec`).

## Comments

- Prereq: `kernels/04` (candidate catalog) and `runtime/09` (cost model,
  `selectMatvec`); both done.
- Realizes the `applicable(type, shape)` / cost metadata deferred in `kernels/04`.
- **Dependency:** prefill is currently token-a-token in the runtime
  (`session.generate`), so `phase=prefill`/GEMM only pays off once prefill is
  batched (a runtime task). The plan can be phase-aware first; the runtime then
  feeds M>1. Track that runtime piece in `sdd/open-items.md` if out of scope.
- Cross-cutting: this is the successor of `runtime/09`'s single selector; the
  generic family-agnostic part continues in `kernels/06`.

09 - Batched GEMV (prefill weight reuse)

Status: done
Commits: 119c580

## Goal

Make the prefill's projections read each weight row **once** and apply it to all
`M` prompt rows (a real batched GEMV/GEMM), so the prefill is ~`M`x faster per
token. Today the "batched" prefill (`runtime/11`) is **M GEMVs**: `pool.run`'s
`runJob` loops the `M` rows calling the matvec per row, re-reading the weights
`M` times. Consequence: on a model that does **not** fit RAM (where each pass
costs a model read), prefill/token ~= decode/token — the 31B's TTFT is ~25 s for
a 32-token prompt instead of ~M x less.

## Baseline (measured, 31B 20gb warm)

- prefill **1.3 tok/s** (0.77 s/prompt-token); decode **1.35 tok/s** (0.74 s/token).
- The two are ~equal because both read the weights once per token — the prefill
  never reuses the weight load across its `M` rows.
- `runBlocked` (the retired `matmul`) *does* reuse (dequant once, dot M rows) but
  with a **scalar** `vector.dot`, so it is slower than the SIMD matvec — that is
  why it was retired to the (also M-GEMV) per-row path.

## Scope

- **In:**
  - A **SIMD batched GEMV** kernel: for each output (weight) row, dequantise its
    quant blocks **once** (SIMD, reusing the matvec `scale_split`/native codecs)
    and **fma into the M accumulators** (SIMD), writing `out[tok*N + i]`.
  - Wire the prefill projection path (`Session.matVecBatch` / `pool.runJob`) to
    use it when `batch > 1`; keep `batch == 1` exactly as today (decode).
  - Numerics equal to the per-row GEMV path within tolerance.
- **Out:** the cost model / tuning; weight/KV quantization; the decode path.

## Design

- The current `runJob` for `batch > 1` loops `tok` and runs a full per-row matvec;
  replace it with one batched kernel per output-row range so the weight bytes are
  read once.
- Kernel shape (per output row `i`): iterate the row's quant blocks; for each
  block `b`: decode the block (SIMD) -> f32 fragment; for `tok in 0..M`: fma the
  fragment against `x[tok, block]` into `acc[tok]`; store `out[tok*N + i] =
  acc[tok] * block_scale + ...`. This is `runBlocked`'s reuse with the matvec's
  SIMD dequant+dot.
- Reuse the existing codecs (`scale_split`/`native`) — no new quant layouts.
- The kernel is a `pipeline`/atomic in the matvec (or matmul) family; keep the
  per-row `batch == 1` kernels untouched.
- Determinism: same inputs -> same plan; the outputs are numerically equal to the
  per-row path (tests).

## Steps

1. Add a batched matvec atomic (or fix `runBlocked`) with a SIMD dequant-reuse
   loop over `M` rows. [TODO]
2. Wire `pool.runJob` (`batch > 1`) / `Session.matVecBatch` to it. [TODO]
3. Tests: batched == per-row GEMV per quant type (1e-5); `batch == 1` unchanged. [TODO]
4. Measure: the 31B TTFT (prefill) and the E2B/E4B/12B prefill before/after;
   confirm the prefill no longer pays `M` model reads. [TODO]
5. `zig build` / `zig build test` green; E2B coherent; plan deterministic. [TODO]

**As-built.** `pool.runJob` (`batch > 1`) now dispatches one **batched GEMV**
(`matvec.matMulRows` -> `batched.matVec*BatchedRows`) per worker row-slice: per
output row it dequantises each quant block **once** (SIMD, existing codecs) and
fma-applies it to all `M` rows into per-token accumulators (deferred `@reduce`).
`batch == 1` (decode) is untouched; a latent T-tiling race (writing outside
`[begin,end)`) was fixed. Prefill **~2x**: E2B 20.7->36.0, E4B 11.1->20.4,
12B 4.1->8.4, 31B 1.7->3.5 tok/s; 31B TTFT 23->11 s; batched == per-row (1e-5);
173/173 tests.

## Definition of Done

- [x] A batched GEMV reads each weight row once and applies it to all `M` rows.
- [x] Batched numerics == per-row GEMV (within tolerance); decode unchanged.
- [x] The streaming models show a prefill/TTFT improvement (31B TTFT 23->11 s;
      prefill no longer at parity with decode).
- [x] `zig build` / `zig build test` green (173/173); E2B coherent; deterministic plan.

## Arch Provenance

- **own:** the batched-GEMV weight-reuse loop.
- **spec:** ggml quant block layouts (existing codecs).
- **consulted:** llama.cpp `ggml_compute_forward_mul_mat` (batch reuse) —
  reference only.
- **derived:** `kernels/05` (`matmul`/`runBlocked`, the reuse idea),
  `runtime/10`/`11` (batched prefill), `runtime/06` (residency).

## Comments

- Directly attacks the streaming models' weak point (TTFT), with **no** quality
  cost — unlike KV/weight quantization.
- `runBlocked` (kernels/05) had the right structure but a scalar dot; this task is
  essentially "`runBlocked` with the matvec's SIMD kernels".
- Related: the KV remains f16 (`runtime/11`); int8 KV is a separate, lower-value
  item with a quality gate.

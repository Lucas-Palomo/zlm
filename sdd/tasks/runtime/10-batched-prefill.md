10 - Batched Prefill

Status: done
Commits: 02fea9b

## Goal

Prefill currently runs **token-a-token** (`session.generate` calls `stepAt(pos)`
per prompt token), so every projection is a GEMV (M=1) and the `matmul`/GEMM
family added in `kernels/05` is dormant. Batch the prompt into block forwards so
prefill uses the **GEMM** path — without breaking the invariant that the
**runtime decides nothing**: the *plan* declares whether prefill is batched and
with what block size; the runtime only executes it.

## Scope

- **In:**
  - **Plan carries the prefill policy**: `Config.prefill_mode` (`token` |
    `batch`) + `prefill_block` (max M), chosen by the tuner via the cost model
    (plan VERSION bump).
  - **Phase-tagged dispatches**: prefill projections use `matmul` (phase=prefill)
    while decode keeps `matvec` (phase=decode); M-aware forms for the other ops.
  - **Batched executor** (`executePrefill(t0, m)`): M-row norm/rope/activation/
    add/elementwise, block KV writes, block-causal attention, `matmul`
    projections.
  - **Buffer sizing / budget** for M rows; the residency/window accounts for it.
  - **Equivalence**: batched prefill ≡ token-a-token prefill (the next token /
    logits), within tolerance.
- **Out:** sampling (unchanged); the GEMM kernel itself (`kernels/05`);
  continuous batching / multi-request; GPU.

## Design

### Tuning decides, runtime obeys
The prompt length is unknown at tune, so the plan declares a **policy**, not a
length:
- `prefill_mode = token`: the runtime loops one token at a time (current
  behavior);
- `prefill_mode = batch`: the runtime processes the prompt in blocks of
  `prefill_block` (a plan-declared max), each block one `executePrefill(t0, m)`.

The tuner picks `prefill_mode`/`prefill_block` from the cost model (GEMM vs GEMV
roofline + the RAM cost of the M-row activations), deterministically. The runtime
reads the policy from the plan and applies it verbatim — it makes no choice.

### Phase-tagged plan
`Dispatch.phase` (exists) = `{ decode = 0, prefill = 1 }`. The plan carries the
dispatches for each phase; the executor runs the prefill subset for a prompt
block and the decode subset per generated token. Projections appear as `matvec`
(decode) and `matmul` (prefill); the other ops get an M-aware form (the same
routine looped over rows, or a batched variant). `prefill_block` is the M baked
into the prefill dispatches; the last block may be smaller (`m <= prefill_block`).

### Batched forward
`executePrefill(t0, m)` mirrors `executeAt` but over `m` rows: embeddings for the
`m` tokens, then per layer norm → qkv (`matmul`) → rope (`t0..t0+m`) →
write_kv (`m`) → block-causal attention → o_proj (`matmul`) → residual → ffn norm
→ gate/up (`matmul`) → activation → down (`matmul`) → residual; `lm_head` on the
last row only. Positions are `t0..t0+m`; the KV rows `[t_start, t0+i]` bound the
causal mask of query row `i`.

### Equivalence
The same prompt under `prefill_mode=batch` and `=token` must yield the same next
token (logits within tolerance). This is the safety contract.

## Steps

1. Plan: add `prefill_mode`/`prefill_block` to `Config` (fingerprint + VERSION
   bump); tuner sets them via cost. [TODO]
2. Tuner: emit phase-tagged dispatches (prefill projections `matmul`) and the
   M-aware op forms; keep the decode list unchanged. [TODO]
3. Executor: `executePrefill(t0, m)` with M-row ops + block-causal attention +
   block KV writes; route `matmul` to the `matmul` family. [TODO]
4. Session: `generate` reads the plan policy — `token` loops as today; `batch`
   runs the prompt in `prefill_block` blocks, then decodes. [TODO]
5. Buffers: size M-row activations; account them in the budget/window. [TODO]
6. Tests: batched ≡ token-a-token prefill (logits, tolerance); block boundaries
   (`m` = block, multiple blocks); KV correctness across the block. [TODO]
7. `zig build` / `zig build test` green; E2B coherent; deterministic. [TODO]

**As-built.**
- `plan.Config` gains `prefill_mode` (`token`|`batch`) + `prefill_block`, both in
  the fingerprint; VERSION 8 -> 9. `AOTKernel.prefill: []Dispatch` is the
  phase-tagged prefill list (the decode list is untouched).
- Tuner `choosePrefillPolicy(ram, dims)`: largest `M` in `{2,4,8,16}` whose
  M-row activation footprint fits `ram/8`; `M < 2` -> `token`. Prefill dispatches
  are compiled from the unfused block with projections as `matmul`; each GEMM's
  `MatMulPlan` + `parts` via `selectMatmul(..., M=block)`.
- Executor `executePrefill(t0, m)`: per-row loops for norm/rope/activation/
  elementwise; `matmul` projections; `write_kv` for `m` positions then per-row
  `fusedFlashAttentionHead` bounded to `[t_start, t0+i]` (block-causal);
  `lm_head` on the last row.
- `session.generate` reads `config.prefill_mode`: `token` loops (unchanged),
  `batch` runs the prompt in `prefill_block` blocks, then decodes.
- Verified: batched ≡ token-a-token (identical next token/text); E2B coherent;
  deterministic plan; 148/148 tests.

## Flow / Arch

```text
plan.prefill_mode = token | batch(block)
        │
generate: prompt ids
        ├── token : for id: executeAt(pos++)          (matvec / GEMV)
        └── batch : for block: executePrefill(t0, m)  (matmul / GEMM)
        │
        └── decode loop: executeAt(pos++)             (matvec / GEMV)
```

## Definition of Done

- [x] `Config.prefill_mode`/`prefill_block` in the plan; the tuner sets them
      deterministically. *(Footprint-fit rule; a cost/benefit rule is a
      follow-up — see `open-items`.)*
- [x] Prefill dispatches are phase-tagged; projections use `matmul` when batched.
- [x] `executePrefill(t0, m)` runs an M-row forward (matmul + block-causal
      attention) and `generate` applies the plan's policy without deciding.
- [x] Batched prefill ≡ token-a-token prefill (identical next token/text).
- [x] `zig build` / `zig build test` green (148/148); E2B coherent; plan
      deterministic.

## Arch Provenance

- **own:** the plan-declared prefill policy (tuning decides; runtime obeys) and
  the block-causal prefill forward.
- **spec:** causal block attention; ggml batching.
- **consulted:** llama.cpp/ggml batched mul_mat + attention — reference only.
- **derived:** `kernels/05` (matmul/GEMM + phase selection), `runtime/03`
  (forward execute), `runtime/04` (generation loop).

## Comments

- Depends on `kernels/05`; the `matmul` family + phase-aware selector landed, the
  runtime side is this task.
- The runtime must **not** choose mode/block: it reads `prefill_mode` from the
  plan (invariant "the runtime decides nothing").
- Resolved: the `open-items.md` entry "Batched prefill in the runtime" is struck.
- Follow-ups (see `open-items.md`): the policy is footprint-fit (almost always
  `batch`), not cost/benefit; ~~prefill dispatches are not yet routed through the
  worker `Pool`~~ (prefill projections now run through the pool: `executePrefill`
  → `prefillMatmul`/`prefillLmHead` → `Session.matVecBatch`/`matVecPlan`, consuming
  the plan's `parts`); prefill dispatches carry no residency nodes (first block
  relies on mmap fault-in).

05 - Per-Layer Embeddings

Status: done
Commits:

## Goal

gemma4's **per-layer embeddings (PLE)** are applied, so the gemma4 forward
produces coherent logits. E2B/E4B have `embedding_length_per_layer_input = 256`;
12B/31B have `0`, so the block must include or omit the PLE path by config. All
of it stays AoT: the tuner resolves slots/dims/params and the executor applies.

## Scope

- **In:** contract slots + a global PLE dim; a per-instruction gate
  (`Instr.gate`); the PLE head setup (per-layer token embedding, model
  projection, per-layer projection norm, combined input) and the post-FFN PLE
  branch (gate, gelu, mul, proj, post-norm, residual); three new atomics
  (ungated gelu, `a*b`, `(a+b)*s`); a scale op.
- **Out:** other architectures; GPU; withheld reasoning.

## Design

Reference: the earlier local prototype gemma4 (`forward.zig` `embd_lookup` PLE block +
`finishFfnAndLayer`; `session.zig` scales). Constants:
`per_layer_embed_scale = sqrt(plen)`, `per_layer_proj_scale = 1/sqrt(n_embd)`,
`per_layer_input_scale = 1/sqrt(2)`.

- **Dims.** `contract.LayerDims` gains `ple: u32 = 0` (plen) and
  `ple_all: u32 = 0` (`block_count * plen`). gemma4 `layerDims` fills them
  (`ple_all = block_count * plen` when `plen > 0`).
- **Slots.** Add, at the end of `graph.Slot`: `ple_token`, `ple_projected`,
  `ple_combined` (each `ple_all`) and `ple_a` (`ple`). `tune.slotBytes` sizes
  them; `compile.slotLen` maps them (`ple_a -> dims.ple`, the rest ->
  `dims.ple_all`).
- **Gate.** `Instr.gate: enum { always, ple, no_ple } = .always`; `compile.emit`
  skips a `.ple` instruction when the resolved layer's `dims.ple == 0` (and a
  `.no_ple` when `dims.ple != 0`). PLE presence is thus read from the arch dims;
  no new `Options` field.
- **Kernels.** Add `activation.AtomicId.gelu` (ungated `geluInPlace`), and
  `elementwise.AtomicId.mul` (`a*b`), `.scale` (`out *= s`), `.combine`
  (`(a+b)*s`); plus the matching `vector.zig` helpers.
- **Ops.** Add `graph.Op.gelu`, `.mul`, `.scale`, `.combine`; map `.gelu` ->
  kernel `.activation` (atomic `gelu`), the other three -> kernel `.elementwise`.
  Params: `mul { off: u32 = 0 }` (element offset into `in2`),
  `scale { s: f32 }`, `combine { s: f32 }`. `Params.code()` gains stable codes;
  `executor.runElementwise` switches on `sem_op` (as `runActivation` already
  does for softcap).
- **Embed scale by role.** `resolveParams` for `.embd_lookup` returns
  `sqrt(n_embd)` for role `token_embd`, `sqrt(plen)` for `per_layer_token_embd`.
- **Block.**
  - head (`.ple`): second `embd_lookup` (`per_layer_token_embd -> ple_token`),
    `matvec per_layer_model_proj` (`hidden -> ple_projected`), `scale` by
    `per_layer_proj_scale`, `norm` role `per_layer_proj_norm` (kind `norm_heads`,
    head_dim = plen, over `ple_all`), `combine` (`ple_projected + ple_token`,
    scale `per_layer_input_scale`) -> `ple_combined`.
  - layer, after the FFN residual `add` and before `layer_output_scale`
    (`.ple`): `matvec inp_gate` (`hidden -> ple_a`), `gelu` (`ple_a`),
    `mul` (`ple_a * ple_combined[layer*plen..] -> ple_a`, `off = layer*plen`),
    `matvec proj` (`ple_a -> normed`), `norm` role `post_norm` (`normed`),
    `add` (`hidden += normed`). `layer_output_scale` stays `.always`, so the
    PLE layer output is `(hidden + ple_out) * scale` and the non-PLE path is
    unchanged.
  - `opForRole`: add `.inp_gate`, `.proj` -> `.matvec`, `.post_norm` -> `.norm`;
    `per_layer_token_embd`/`per_layer_model_proj`/`per_layer_proj_norm` already
    map. `zcom` already whitelists `per_layer_proj_norm` and `post_norm`.

## Arch Provenance

- **own:** the AoT placement (slots/dims/params/gate) and the scale op.
- **spec:** gemma4 PLE metadata/dimensions.
- **consulted:** the earlier local prototype gemma4 `forward.zig`/`session.zig` (behavior,
  constants) — reference only.
- **derived:** from `architectures/04` and `runtime/03`.

## Steps

1. Contract: `LayerDims.ple`/`ple_all`; `Slot` PLE entries; `Instr.gate`.
   - **As-built:** `contract.LayerDims.ple`/`ple_all`; `graph.Slot` appended
     `ple_token/ple_projected/ple_combined/ple_a`; `graph.Instr.gate` +
     `graph.Gate{always,ple,no_ple}`.
2. Kernels: `activation.gelu`; `elementwise.mul/scale/combine`; `vector` helpers.
   - **As-built:** `activation.geluInPlace` + atomic `gelu`;
     `elementwise.mul/scale/combine` (`out=a*b`, `out=a*s`, `out=(a+b)*s`) +
     `vector.scaleBy` helper.
3. Graph ops/params/codes; `compile` `slotLen`/`kernelOp`/`defaultAtomic`/`emit`
   gate; `resolveParams` (scales, mul off, combine s, per_layer norms, roles).
   - **As-built:** `graph.Op.gelu/mul/scale/combine`; `Params.mul{off}`,
     `scale{s}`, `combine{s}` (codes 8/9/10); `compile` maps ops/atomics,
     `slotLen` for PLE slots, and `emit` drops `.ple` when `dims.ple == 0`
     (`.no_ple` when `!= 0`); gemma4 `resolveParams` role-based embed scale,
     `per_layer_proj_norm` -> `norm_heads` (`head_dim=plen`), `post_norm` ->
     `norm_rms`, `mul.off = layer*plen`, `scale.s = 1/sqrt(n_embd)`,
     `combine.s = 1/sqrt(2)`; `opForRole` adds `inp_gate`/`proj`/`post_norm`.
4. Executor: `.scale`, `.mul`, `.combine`, `.gelu` handling; `slotBytes` +
   per-layer dims in `tune.zig`.
   - **As-built:** `runActivation` handles the `gelu` atomic; `runElementwise`
     switches on `sem_op` (`mul` with the `in2` offset, `scale`, `combine`);
     `tune.zig` `Dims.ple/ple_all` + `slotBytes [15]u64`.
5. Tests: kernel atomics; a `compileBlock` test that `.ple` is dropped when
   `dims.ple == 0` and kept otherwise.
   - **As-built:** kernel atomics tested; PLE-gate `compileBlock` test added.
6. `zig build` / `zig build test` green; re-tune and `run` E2B -> coherent text.
   - **As-built:** build/test green (67/67); re-tuned (869 dispatches, 30
     buffers); `run` E2B:
     - "What is the capital of France?" -> "The capital of France is Paris."
     - "Count from one to five:" -> "One Two Three Four Five".

## Definition of Done

- [x] PLE head setup + post-FFN branch in the gemma4 block.
- [x] `Instr.gate`; PLE instructions dropped when `plen == 0`.
- [x] New atomics (gelu/mul/scale/combine) with tests.
- [x] e2b `run --prompt` emits coherent continuation.
- [x] `zig build` / `zig build test` green.

## Comments

- Locks the last known gemma4 correctness gap from `architectures/04`.
- If output is still incoherent after PLE, compare logits against
  the earlier local prototype / llama.cpp before assuming PLE is the only cause.

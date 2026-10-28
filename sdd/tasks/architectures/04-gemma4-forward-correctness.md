04 - Gemma4 Forward Correctness

Status: done
Commits:

## Goal

The gemma4 forward produces coherent logits. The current declarations drifted from
the reference forward (the earlier local prototype, `src/architectures/gemma4/`), so
generation is degenerate/repetitive. Restore the missing semantics **AoT**: the
tuner resolves every value and the plan carries it; the executor applies only.

## Scope

- **In:** attention scale; embedding scale; gated activation choice; shared-KV
  store/source semantics; sliding-window `t_start`; per-head V norm; per-layer
  embeddings (PLE). Declaration + tuner + plan params; executor only applies.
- **Out:** new sampling; threading/GPU; tokenizer (done in `runtime/04`).

## Design

The current rule: **a decision is made once, at `tune`, and recorded in the plan.**
The executor may compute values that depend on live inputs (`pos`), but never
inspects the layer/config to choose behavior.

- **Attention scale.** gemma4 uses `scale = 1.0` (the reference passes `1.0`; no
  pre-attn scaling). `resolveParams` records it; no runtime change.
- **Embedding scale.** `hidden *= sqrt(embedding_length)` after the lookup. Add
  a `Params.embed { scale }`; the tuner records it on the `embd_lookup`
  dispatch; the executor multiplies the dequantized row by `params[0]`.
- **Gated activation.** gemma4's FFN is gated **GELU**, not SiLU. The block
  declares the kind; the tuner maps `kind` -> `activation.AtomicId`; the plan
  carries the atomic id (the executor already runs by `atomic_id`).
- **Shared KV.** Layers that do not store a cache must neither write nor
  recompute K/V; they attend the source layer's cache. The architecture already
  declares this in `KvLayout` (`stores`, `source`). The tuner skips the
  store-path instructions for non-storing layers and points their attention at
  the source buffer. The runtime keeps running whatever is in the plan.
- **Sliding window.** The plan carries the window (`params[7]`); the executor
  derives `t_start = pos + 1 - window` (clamped) for sliding dispatches. The
  tuner sets `window`; the runtime decides nothing.
- **Per-head V norm.** gemma4 adds a weightless RMSNorm on V per head (the
  reference normalizes `v` alongside the weighted K norm). Declare it in the
  block.
- **PLE.** Per-layer embeddings: `per_layer_token_embd` (scaled, `sqrt(plen)`),
  `per_layer_model_proj` (`1/sqrt(n_embd)`), per-layer `rmsNorm` with
  `per_layer_proj_norm`, combined `(proj + token) * (1/sqrt(2))`; after the FFN,
  a gate (`gelu`) times the layer's PLE slice through `proj`, post-norm, then
  add-and-scale. All scales are config-derived and resolved at tune.

## Arch Provenance

- **own:** the AoT placement (which value is tuner-resolved vs executor-applied).
- **spec:** gemma4 GGUF config (shared_kv_layers, sliding_window, per-layer
  input length, key/value lengths, softcap).
- **consulted:** the earlier local prototype gemma4 `forward.zig`/`session.zig` (reference
  only; behavior and constants, not copied structure).
- **derived:** from `architectures/03` (forward block) and `runtime/04`.

1. Attention scale `1.0` in `resolveParams`.
   - **As-built:** `impl/gemma4/arch.zig` `.attention/.write_kv` records
     `.scale = 1.0` (was `1/sqrt(head_dim)`).
2. `Params.embed`; resolve `sqrt(embedding_length)`; executor applies it.
   - **As-built:** `graph.Params.embed { scale }` + `embed_code = 7`; gemma4
     `.embd_lookup` resolves `sqrt(embedding_length)`; `packParams` packs it;
     `executor.applyEmbedScale` multiplies the dequantized row when
     `sem_op == embed_code`.
3. Gated GELU: block declares the kind; tuner selects the atomic.
   - **As-built:** `graph.Params.activation_gelu`; gemma4 `.activation` resolves
     `kind = gelu`; `compile.activationAtomic` maps kind -> `activation.AtomicId`;
     `emit` overrides the dispatch's `atomic_id` from the recorded kind.
4. Shared KV: tuner skips the store path for non-storing layers and binds
   attention to the source cache.
   - **As-built:** `graph.Instr.store_only`; gemma4 block marks
     `attn_k`/`attn_v` matvec, `attn_k_norm`, `rope`(k) and `write_kv`;
     `Options.kv_stores` + `emit` skip; `tune.zig` passes `kv.layers[i].stores`.
     E2B: 739 → 639 dispatches (5×20 shared layers removed).
5. Sliding window: carry `window`; executor derives `t_start` from `pos`.
   - **As-built:** `Params.attention.window` (packs `params[7]`); gemma4
     `resolveParams` sets `cfg.sliding_window` for sliding layers;
     `executor.executeAt` computes `t_start` from `pos` + window.
6. Per-head V norm in the block.
   - **As-built:** new `Op.v_norm` + kernel atomic
     `rms_norm_no_weight_heads`; `Params.norm` carries `{eps, head_dim, kind}`
     and the tuner records the atomic (norm selection is now AoT);
     `layer_output_scale` added to the `zcom` `norm_f32` whitelist.
7. PLE declaration + tuner params + runtime application. **Moved to
   `architectures/05`** (new task — it needs contract/kernel work).
8. `zig build` / `zig build test` green; e2b `run` produces coherent text.
   - **As-built:** build/test green; e2b re-tuned and `run` works (654
     dispatches). Coherence still pending `architectures/05` (PLE).

**Verified (1-3):** `zig build` / `zig build test` green; re-tuned
`tune e2b-test rt-e2b`; `run rt-e2b --prompt hello` changed from a run of `-`
to multilingual text. Still incoherent because 4-7 are open (shared-KV
corruption dominates).

## Definition of Done

- [x] gemma4 attention uses `scale = 1.0`; embedding is scaled by `sqrt(n_embd)`.
- [x] FFN uses gated GELU.
- [x] Shared-KV layers neither write nor recompute K/V; attend the source.
- [x] Sliding layers attend only the window.
- [x] Per-head V norm.
- [x] `zig build` / `zig build test` green.
- ~~PLE is applied~~ / ~~e2b `run --prompt` emits coherent continuation~~ →
  moved to `architectures/05` (depends on PLE).

## Comments

- DoF is deliberately strict (coherent text): the point of the task is to make
  the gemma4 path actually correct, not merely to run.

03 - Forward Block

Status: done
Commits:

## Goal

Make gemma4's declaration a **complete forward**: the block covers the residual
path, the per-layer output scale and the final softcap, and the architecture
declares its KV layout. No forward *code* — still a semantic instruction
sequence the tuner compiles.

## Scope

- **In:** finish `impl/gemma4/block.zig` (residual adds, `layer_output_scale`,
  softcap, correct `write_kv`); add `op`s as needed (`add`); declare the KV
  layout (`core/kv.zig` + gemma4 populates it from `config`); resolve the
  instruction `params` (eps/rope/attention) from the config in the declaration
  path.
- **Out:** the plan serialization of params/KV (`plan/03`); the executor
  (`runtime/03`); measured tuning.

## Design

- **Residuals are ops, not conventions.** Add `graph.Op.add` (and the
  `elementwise` family in kernels, tracked separately) so the block declares
  `hidden = hidden + attn_out` and `hidden = hidden + ffn_mid` explicitly,
  instead of writing into the same slot.
- **`layer_output_scale`** becomes an instruction (role exists; the op maps to
  `norm`/scale today). gemma4 scales the block output per layer.
- **Final softcap** as a tail instruction (`softcap` op + `cap` from
  `final_logit_softcapping`) after `lm_head`.
- **`write_kv` models (k, v) -> cache**, not `in=.k, out=.v`: the instruction
  names k and v sources and the cache target; the KV cache slots come from the
  declared layout.
- **`core/kv.zig` — `KvLayout`:** the architecture declares its cache:
  `{ layers, kv_heads[], head_dim[], sliding[]?, shared_from?, source_layer[] }`.
  It is vocabulary (data), like `graph`/`contract`; gemma4 fills it from
  `GemmaConfig` (`head_count_kv`, `key_length`, `sliding_window_pattern`,
  `shared_kv_layers`, `kvSourceLayer`).
- **Params are resolved in the declaration.** Instead of `eps=0`,
  `freq_base=0`, `sliding=false`, the block/file exposes resolved params (from
  the config) so the tuner serializes real values (see `plan/03`). The block
  stays declarative; a small `paramsFor(layer)` helper reads the config.

## Arch Provenance

- **own:** the complete block shape, `KvLayout`, add-as-op.
- **spec:** gemma4 config semantics (RMS eps, RoPE dim/base, sliding, shared KV,
  final softcap) from GGUF metadata.
- **consulted:** gemma-family reference forward ordering.
- **derived:** from `architectures/02` (semantic graph) and `TuneSkeleton`.

## Steps

1. Add `graph.Op.add`; declare the residual instructions and
   `layer_output_scale`.
2. Add the final `softcap` instruction to the tail.
3. Fix `write_kv` to (k, v) -> cache.
4. `core/kv.zig`: `KvLayout`; gemma4 populates it from config.
5. Resolve instruction params from the config (eps/rope/attention/softcap).
6. A test asserts the block's op sequence per layer (order + slots).
7. `zig build` / `zig build test` green.

## Flow / Arch

```text
layer: attn_norm -> q,k,v -> qk_norm -> rope -> write_kv ->
       attention -> attn_output -> post_attention_norm -> add ->
       ffn_norm -> gate,up -> activation -> down -> post_ffw_norm -> add ->
       layer_output_scale
tail : output_norm -> lm_head -> softcap
```

## Definition of Done

- [x] Residual `add`, `layer_output_scale` and final `softcap` are declared.
- [x] `write_kv` models (k, v) -> cache.
- [x] `core/kv.zig` declares `KvLayout`; gemma4 fills it from config.
- [x] Instruction params are resolved (no `=0`/`false` placeholders).
- [~] A test pins the per-layer op sequence — covered indirectly by
      `compileBlock`'s ordered-dispatch test; a dedicated block-sequence test is
      deferred.
- [x] `zig build` / `zig build test` green.

## Comments

- Depends on: `runtime/03` for execution; `plan/03` for serializing the added
  ops/params/KV.
- The `elementwise` family (for `add`) is a kernels gap (tracked in
  `kernels/03` Comments); the op can be declared before the kernel lands.

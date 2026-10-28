04 - Per-Layer Dimensions

Status: done
Commits:

## Goal

Dispatch buffers are sized once, at the widest layer. gemma4 mixes full and
sliding layers with different `head_dim`/`ffn` (sliding layers are wider), so a
single size forces narrow layers to process a stale prefix and mis-sizes per-head
ops. Resolve each layer's activation sizes independently.

## Scope

- **In:** per-layer buffer sizing and binding offsets; how a `Dispatch` names the
  slice of a shared slot it uses; the tuner computing it from the config.
- **Out:** the forward execution itself (`runtime/03`); KV (already per layer).

## Design

- **Problem:** slots are indexed by `Slot` only (hidden/q/k/v/ffn_...), so all
  layers share one buffer. Today `tune` sizes each slot to the **max** over all
  layers (sliding layers are wider than full ones), and the executor passes a
  prefix. For per-head q/k norms and the attention head split this is wrong: a
  full layer sees the sliding layer's width.
- **Chosen — option 1 (per-dispatch lengths).** Slots stay one buffer per
  `Slot`, sized to the **max** over layers. Each `Dispatch` gains `in_len`/`out_len`
  (element counts in f32); the executor slices `buffer(ref)[0..len]`. No per-layer
  offsets are needed because every layer starts at the buffer base (a layer's
  q/k/v/ffn always begins at 0; only the *length* differs).
  - Why not option 2 (per-layer buffers): more `Buffer[]` entries and a bigger
    `.zplan` for no addressing benefit.
  - The KV cache is already per storing-layer (`d.kv`), unaffected.
- **What varies:** `head_dim` (full vs sliding), `n_kv_heads`, `ffn_length`
  (gemma4 E2B: ffn 6144 full, 12288 sliding; head_dim 512 full, 256 sliding).
  Hidden is uniform.
- **Resolver lives on the `Arch` contract:** the tuner is architecture-agnostic,
  so `Arch` gains `layerDims(cfg, layer) -> Dims` (like `resolveParams`); `tune`
  passes it into `compileBlock` via `Options`. Never import gemma4 in `tuner/`.
- **tuner:** for each instruction, resolve the layer's dims from the config and
  emit `in_len`/`out_len`; the executor must not compute them.
  - `matvec`: `x = dims[0]`, `out = dims[1]` (from the tensor).
  - `norm`/`activation`/`elementwise`/`rope`/`attention`: from the operand dims
    (hidden, q = n_heads*head_dim, kv = n_kv_heads*head_dim, ffn, vocab).
- **Defaults:** `in_len`/`out_len` = 0 means "whole buffer" (backward-compatible
  for ops that already matched, and global ops like embedding/lm_head).

## Arch Provenance

- **own:** the per-layer sizing scheme.
- **spec:** `.zplan` layout (`plan/01`), ggml/config dims.
- **consulted:** —.
- **derived:** from the "slots sized to the widest layer" note in `runtime/03`.

## Steps

1. Add `in_len`/`out_len: u32` to `Dispatch`; `.zplan` read/write; VERSION bump;
   round-trip asserts.
   - as-built: fields appended to `Dispatch`; `plan.VERSION` 3 -> 4; written/
     read after `rope_cfg`; round-trip asserts both set and default-0.
2. `tuner/compile.zig`: pass per-layer dims into `emit` (a `LayerDims` resolver
   from the config); set `in_len`/`out_len` per instruction.
   - as-built: `Arch.layerDims` + `contract.LayerDims`; `Options.layer_dims`/
     `layer_dims_ctx`; `emit` maps `graph.Slot` -> length (hidden/q/kv/ffn/vocab),
     leaving `embd_lookup`/`write_kv` at 0.
3. `tune.zig`: build the resolver from `model_cfg` (head_dim/n_kv_heads/ffn per
   layer) and pass it.
   - as-built: `LayerDimsCtx` wraps `arch.layerDims` and injects the tensor-
     derived `vocab` (the config lacks it); passed via `Options`.
4. `runtime/executor.zig`: slice `x`/`out` by `d.in_len`/`d.out_len` (0 = whole).
   - as-built: `bufN` helper; applied to matvec/norm/activation/elementwise/
     rope and q/out in flash-attention; KV and dequant untouched.
5. Test: a synthetic 2-layer block with differing ffn/head_dim yields distinct
   `in_len`/`out_len`.
   - as-built: `compileBlock` test with a fake `layerDims` (q 8/4, ffn 32/64).
6. `zig build` / `zig build test` green; re-run e2b (`tune` + `run`).
   - as-built: build/test green; `tune e2b-test rt-e2b` (739 dispatches) and
     `run rt-e2b --prompt hello` -> `token: 20248`.

## Definition of Done

- [x] Each layer's activation size comes from its own config dims.
- [x] No "stale prefix"/over-sized buffer; q/k norm + attention use exact widths.
- [x] `.zplan` round-trips the new field(s).
- [x] `zig build` / `zig build test` green.

## Comments

- Raised as an intentional first-cut deferral in `runtime/03` (slots sized to the
  widest layer). Not needed to prove the path; needed for correct logits.

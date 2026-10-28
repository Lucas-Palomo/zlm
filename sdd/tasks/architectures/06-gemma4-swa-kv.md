06 - Gemma4 SWA K/V (larger variants)

Status: done
Commits: 88eebfa

## Goal

The current code must run every gemma4 variant, not just E2B. Today the **12B fails at run
time**: the session opens but `generate` returns `error.Unsupported` (the daemon
logs `GENERATE_ERR Unsupported`; instrumented: `op=matmul tensor=NONE layer=5
in=3840 out=512`). The earlier local prototype runs the 12B warm at ~3.8 tok/s.

## Root cause (diagnosed)

The 12B's `gemma4.attention.head_count_kv` is **per layer**:
`[8,8,8,8,8,1, 8,8,8,8,8,1, …]` — the **sliding-window (SWA) layers**
(`head_count_kv = 1`, indices 5, 11, 17, …) have **no `attn_v` tensor** (the
`tensors` dump shows `blk.5.attn_k.weight [3840x512]` but no `blk.5.attn_v.weight`,
and `key_length_swa = value_length_swa = 256`). E2B is different: its
`head_count_kv` is the scalar `1` for all layers and it **does** have `attn_v`.

The gemma4 declaration emits an `attn_v` projection for every layer, so the SWA
layers get a `Dispatch` with `tensor = NONE`; the executor returns
`error.Unsupported`.

## Scope

- **In:**
  - Determine the exact SWA K/V semantics (V = K / k_eq_v, or K+V packed in the
    `attn_k` tensor) for gemma4 (12B/31B).
  - Make the gemma4 declaration emit the correct ops for the SWA layers: no
    separate `attn_v` tensor — reuse K as V, or split the packed `attn_k`.
  - Validate E4B and 12B end to end (coherent output; 12B comparable to the earlier local prototype).
- **Out:** performance; the KV/context budget (a separate open-item: the KV at
  the maximized context eats the RAM for large models).

## Design

- The declaration (`architectures/impl/gemma4/block.zig`) builds the layer ops
  from `Instr[]`; the `key`/`value` roles map to tensors via the tuner's LUT.
  For a SWA layer, the `value` role must resolve to the K tensor (k_eq_v) or to
  the second half of the packed `attn_k`, not to a missing `attn_v` name.
- The `head_count_kv`/`value_length_swa`/`key_length_swa` config already exist in
  `GemmaConfig`; the per-layer branch already exists for E2B's scalar form — the
  vector form (per-layer) must drive the SWA handling.
- Keep the E2B plan unchanged.

## Steps

1. Confirm the SWA semantics: compare a SWA layer's `attn_k` layout
   (`[3840x512]`, `key_length_swa=256`, `value_length_swa=256`) against the
   reference (the earlier local prototype) to decide V=K vs packed. [TODO]
2. Fix the gemma4 declaration/compile so the SWA layers emit no missing
   `attn_v` (k_eq_v or split the packed `attn_k`); the LUT resolves every
   `Dispatch.tensor`. [TODO]
3. Validate E4B and 12B: tune + run coherent; 12B comparable to the earlier local prototype's ~3.8 tok/s. [TODO]
4. Tests: the 12B's SWA layers resolve their K/V; a synthetic per-layer
   `head_count_kv` drives the right ops. [TODO]
5. `zig build` / `zig build test` green; E2B unchanged. [TODO]

**As-built.** The SWA layers use **k_eq_v** (V = K). Added `Arch.roleFallback`
(consulted by the tuner's `TensorLut` only after the primary lookup fails;
`attn_v -> attn_k` for gemma4). E2B/E4B have `attn_v` so the primary lookup
succeeds and the fallback is never consulted — their plan is unchanged. The 12B
now runs (no `error.Unsupported`), decode 3.3 / prefill 4.0 tok/s (earlier local build: 3.8/4.5);
170/170 tests. The 12B session no longer OOMs once the budget reserves the
weights (see `runtime/11` / the budget fix).

## Definition of Done

- [x] The SWA K/V semantics are confirmed (**V = K**, k_eq_v).
- [x] The 12B runs coherently end to end (no `Unsupported`); comparable to the earlier local prototype.
- [x] E4B runs; E2B plan unchanged.
- [x] `zig build` / `zig build test` green (170/170).

## Arch Provenance

- **own:** the gemma4 SWA K/V handling.
- **spec:** gemma4 attention config (`head_count_kv`, `*_swa`); gemma-family
  k_eq_v / packed-KV layout.
- **consulted:** the earlier local prototype's gemma4 handling (it runs the 12B).
- **derived:** `architectures/01`-`05` (gemma4 declaration + PLE), `plan/`
  (compile the block).

## Comments

- Blocks: the current code cannot run the 12B (and likely the 31B, which shares the pattern).
  Correctness, not perf.
- Found while testing larger models after the `runtime/11` perf work.
- Related open-item: "Budget window policy is coarse" (the 12B/E4B KV-vs-pinning
  issue) — the 12B is only *usable*, not just *runnable*, once both land.

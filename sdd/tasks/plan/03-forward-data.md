03 - Forward Data

Status: done
Commits:

## Goal

Carry everything the executor needs so the runtime **decides nothing**: the
per-dispatch params (norm eps, RoPE base/dim/rot, attention scale/sliding,
softcap cap) and the KV cache layout are serialized into the `.zplan`. Extends
`plan/02` (which emitted the `Dispatch[]` order/bindings) to a forward-complete
plan.

## Scope

- **In:** a params representation on `Dispatch` (read/write + `Reader`); KV
  `Buffer[]` entries (or cache descriptors) resolved from `KvLayout`; wiring
  the gemma4 params/KV (from `architectures/03`) through `tuner/compile.zig`
  into the plan; `zcom` RoPE tables (referenced by rope dispatches).
- **Out:** the block declaration (`architectures/03`); the executor
  (`runtime/03`); measured tuning.

## Design

- **Params live in the plan.** `Dispatch` gains a fixed params blob (or an index
  into a params table): a small, op-tagged record with the floats/ints an op
  needs. Suggested encoding: `params: [N]u32` per dispatch (bit-cast floats) or
  a `params_index` into an `Arena` of typed records. Keep it compact and
  versioned (`.zplan` VERSION bump).
- **One record per semantic op:** norm `{eps}`, rope `{pos, n_rot, freq_base}`,
  attention `{scale, sliding, window}`, softcap `{cap}`, add `{}`. The tuner
  fills them from the config (declaration path); the executor reads them and
  builds `Args`.
- **KV in the plan.** `KvLayout` -> concrete cache buffers: per-layer (or a
  single cache with layer offsets) with byte sizes from
  `kv_heads × head_dim × context_length × 4`; shared-KV layers reuse the source
  layer's slot. Registered in `plan.buffers` and referenced by `write_kv`/
  `attention` dispatches.
- **`zcom` RoPE tables.** Extend `engine/02` with `rope_cos_sin` regions
  (per `(freq_base, head_dim, n_rot)`); the rope dispatch references a region so
  the runtime reads it instead of computing.
- **Fingerprint/version:** bump `.zplan` VERSION; params+KV are hardware-agnostic
  parts of the plan (still one plan per config fingerprint).

## Arch Provenance

- **own:** the params-on-dispatch representation and KV-as-buffers scheme.
- **spec:** `.zplan` layout (self); GGUF config values.
- **consulted:** —.
- **derived:** from `plan/02`, `architectures/03`, `engine/02`.

## Steps

1. Add the params record to `Dispatch` + read/write + `Reader`; VERSION bump.
2. Resolve KV `Buffer[]`/descriptors from `KvLayout` in `tuner/compile.zig`.
3. Serialize gemma4 params (eps/rope/attention/softcap) per dispatch.
4. Add `zcom` `rope_cos_sin` regions (engine/02 extension) and reference them.
5. Round-trip test: a forward plan reads back with the same params + KV.
6. `zig build` / `zig build test` green.

## Flow / Arch

```text
architectures/03 (block + params + KvLayout)
  -> tuner/compile: Dispatch[] + params[] + buffers(KV) + zcom refs
  -> plan.write (.zplan VERSION+1)
runtime/03: plan.Reader -> Dispatch.params / buffers / zcom -> Args
```

## Definition of Done

- [x] `Dispatch` carries params; `.zplan` round-trips them (VERSION 3).
- [x] KV caches are declared in `plan.buffers` from `KvLayout`.
- [x] gemma4 params (eps/rope/attention/softcap) are serialized.
- [x] `zcom` has RoPE (`inv_freq`); rope dispatches reference the config id.
- [x] No runtime decision remains (params/KV/`max_seq`/`rope_cfg` in the plan).
- [x] `zig build` / `zig build test` green.

## Comments

- Encoding choice (inline `[N]u32` vs params table) decided in Step 1; prefer
  the smallest that keeps `Dispatch` fixed-size for the Executor loop.
- Depends on `architectures/03` (block/params/KvLayout) and extends `engine/02`
  (zcom).

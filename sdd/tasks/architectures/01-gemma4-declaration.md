01 - gemma4 Declaration

Status: done
Commits:

## Goal

Port gemma4 as **declaration only** on the new architecture contract: config
(from the image metadata) + name/role mapping + op-per-role. No forward logic —
the forward is the plan.

## Scope

- **In:** the generic `Arch` contract; `TensorRole`/`TensorRef`; gemma4
  `config.zig` (reads `zmeta`) and `arch.zig`; the `zmeta` reader; the
  architectures registry.
- **Out:** the semantic instruction graph and the engine (task 02); PLE/shared-KV
  recipes; qwen3.

## Design

- **`Arch`** (comptime bundle): `name`, `matches`, `parseTensorName`,
  `opForRole`, `loadConfig`. Declaration only.
- **Shared role space** (`model/contract.zig`): models map into one
  `TensorRole`; no per-model enum.
- **Config union** (`model/root.zig`): `Config = union { gemma4: GemmaConfig }`
  so the registry stays uniform and there is no import cycle.
- **Metadata source is the image**: `zmeta_reader.Zmeta` exposes the same
  accessors as `gguf.Gguf`, so the model reads from the image, not the GGUF.
- **Registry**: `model/root.zig` holds `archs[]` + `find`; `architectures/root.zig`
  is a thin facade (`find`/`isSupported`/`parse`/`opForRole`/`loadConfig`).
- **`tune`** generates `Dispatch`es via the architecture declaration
  (`parse` -> `opForRole`), not a hardcoded `ztef.Kind` map.

## Arch Provenance

- **own:** the `Arch` contract, the shared role space, the Config union, the
  zmeta-backed config.
- **spec:** GGUF metadata keys (`gemma4.*`) for `GemmaConfig`.
- **consulted:** llama.cpp/ggml gemma4 conventions — reference only.
- **derived:** `opForRole` mapping from the earlier `TensorRole`/role work.

## Steps

1. `image/zmeta_reader.zig`: `Zmeta.open` + getters (`getString/getUint/...`).
   **Done.**
2. `architectures/model/contract.zig`: `Arch`, `TensorRole`, `TensorRef`.
   **Done.**
3. `architectures/model/root.zig`: `Config` union + `archs[]` + `find`.
   **Done.**
4. `models/gemma4/config.zig`: `GemmaConfig` + `parseTensorName`/`formatLayeredName`
   on the shared role. **Done.**
5. `models/gemma4/arch.zig`: implements `Arch`. **Done.**
6. `architectures/root.zig` facade; `tune` uses it. **Done.**

## Flow / Arch

```text
zmeta (image metadata) ─▶ models/gemma4/config.GemmaConfig
models/gemma4/arch.zig ─▶ Arch { matches, parseTensorName, opForRole, loadConfig }
architectures/root.zig ─▶ find / parse / opForRole / loadConfig
tune ─▶ parse(name) -> opForRole(role) -> Dispatch
```

## Definition of Done

- [x] `Arch` contract exists; gemma4 implements it.
- [x] gemma4 uses the shared `TensorRole` (no local enum / conversion).
- [x] `Config` union + `loadConfig` on the contract (registry uniform).
- [x] `Zmeta` reader exposes the metadata; `config` reads from it.
- [x] `tune` maps tensor -> op via the declaration (E2B: 530 dispatch).
- [x] `zig build` / `zig build test` green.

## Comments

- `opForRole` is the bootstrap mapping; richer recipes come with the Composer
  (task 02+).
- `TensorRef.layer` is carried but not yet used by the scheduler.

01 - Plan Format

Status: done
Commits:

## Goal

Define the `.zplan` artifact: the compiled plan the runtime consumes — config +
fingerprint + tensor inventory + basic graph — plus the runtime store layout.
Originally delivered as `tune/01 - Tune Bootstrap`, together with the `tune`
command's flag handling; the format is owned by this milestone, the flags by
`cli/02`.

## Scope

- **In:** the `.zplan` format (config + fingerprint + tensor inventory + basic
  ranges/nodes); the runtime store (`.zlm/runtimes/<alias>/`).
- **Out:** the `tune` flag surface (see `cli/02`); measured autotuning
  (kernel/layout selection); GPU backend (flags reserved only); real forward
  execution (that is `runtime/`).

## Design

- **The user sets the limits; the plan carries them.** `run` decides nothing.
- **Quantized budgets:** `--ram`/`--vram` in `mb|gb`; stored as bytes (flag
  parsing lives in `cli/02`).
- **Fingerprint** = hash of `(threads, ram_bytes, vram_bytes, gpu, cpu_isa,
  gpu_id)`; stored in the plan (and would key multiple plans per image).
- **Runtime store:** `tune` creates `.zlm/runtimes/<alias>/plan.zplan`. The
  plan names the image; the runtime resolves it from `.zlm/images/<image>/`.
- **Graph (basic/mock):** one `Range` per `ztef` bucket (kind + payload bytes)
  and one `Node` (`page_in`) per range. No residency policy yet.
- **`load`** is a bypass (no daemon/session); **`run`** reads the plan and stops
  before execution (skeleton).

## Arch Provenance

- **own:** the `.zplan` skeleton, the config/fingerprint scheme, the runtime
  store layout.
- **spec:** GGUF/`ztef` offsets to size the ranges.
- **consulted:** —.
- **derived:** inventory/ranges from the image's `ztef` entries.

## Steps

1. `src/utils/cpu.zig` (cpu digest) + `src/plan/root.zig` (Config, fingerprint,
   Tensor/Range/Node, `write`/`Reader`). **Done.**
2. `store.zig`: runtime alias paths (`.zlm/runtimes/<alias>/`). **Done.**
3. `zlm tune` command: build the plan from the image and write it. **Done**
   (the flag parsing is now owned by `cli/02`).
4. Round-trip test for the plan; `zig build test`. **Done.**

Validated on E2B (600 tensors, 9 ranges); unknown image errors.

## Flow / Arch

```text
zlm tune <image> <alias> --threads N --ram X [--gpu --vram Y]
  config = {threads, ram, vram, gpu, cpu_isa, gpu_id}
  fp = fingerprint(config)
  inventory: read image ztef entries (kind/layer/name/length)
  graph: one range per bucket + one page_in node each        (mock/basic)
  write .zlm/runtimes/<alias>/plan.zplan
```

## Definition of Done

- [x] `zlm tune <image> <alias> --threads N --ram X` writes the plan.
- [x] `.zplan` round-trips (write -> read) with the same config/inventory/graph.
- [x] `--gpu`/`--vram` are reserved (gpu backend not implemented).
- [x] `zig build` / `zig build test` green.

## Comments

- GPU backend out of scope (flags reserved).
- Measured autotuning deferred (a later task).
- ~~Delivered as `tune/01 - Tune Bootstrap`.~~ Renamed and re-homed to the
  `plan` milestone; the CLI flag handling split out to `cli/02 - Tune Skeleton`.

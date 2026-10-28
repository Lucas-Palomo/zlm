01 - Runtime Skeleton

Status: done
Commits:

## Goal

Wire the minimal runtime plumbing so the compile -> tune -> run path exists end
to end: `load` is a bypass, `run` consumes the plan. No generation yet.

## Scope

- **In:** `zlm load <alias>` (validate the runtime; no-op); `zlm run <alias>
  --prompt ... [--max-tokens N]` reads the plan, resolves the image, prints the
  config and stubs at `TODO: Not Implemented yet!`. Top-level commands `tune`,
  `load`, `run`.
- **Out:** session/KV, tokenizer, kernels, gemma4 forward, sampler, GPU.

## Design

- **No daemon/session** in this version: the model does not stay alive on
  `load`. `run` is a one-shot process that reads the plan.
- **The plan carries the config** (threads/ram/vram/gpu) — `run` decides
  nothing; it only reads the plan and the named image.
- **Flags reserved:** `run` accepts only `--prompt` and `--max-tokens` for now.
- **GPU is not implemented**; `--gpu` in the plan is recorded but unused.

## Arch Provenance

- **own:** the runtime store (`runtimes/<alias>/`), the bypass/skeleton flow.
- **spec:** the `.zplan` layout from `tune`.
- **consulted:** —.
- **derived:** image resolution from the plan's image name.

## Steps

1. `zlm load <alias>`: validate `.zlm/runtimes/<alias>/plan.zplan` exists; print
   `loaded '<alias>'` (no-op). **Done.**
2. `zlm run <alias> --prompt ... [--max-tokens N]`: read the plan, print
   `image/config/tensors`, then `TODO: Not Implemented yet!`. **Done.**
3. Register `tune`/`load`/`run` in the CLI registry + help.
4. `zig build` / `zig build test`.

## Flow / Arch

```text
zlm load <alias>            -> validate runtimes/<alias>/plan.zplan; no-op
zlm run  <alias> --prompt P -> read plan; resolve image; print config; TODO
```

## Definition of Done

- [x] `zlm load <alias>` validates the runtime (error if missing); no-op
      otherwise.
- [x] `zlm run <alias> --prompt ...` reads the plan, prints the resolved config
      and image, and stops at `TODO: Not Implemented yet!`.
- [x] `tune`/`load`/`run` are top-level commands with help.
- [x] `zig build` / `zig build test` green.

## Comments

- Real execution (tokenizer/kernels/gemma4 forward/sampler) is the next runtime
  task.
- GPU backend out of scope (flags reserved).

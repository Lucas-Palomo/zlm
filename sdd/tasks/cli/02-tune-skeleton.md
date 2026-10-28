02 - Tune Skeleton

Status: done
Commits:

## Goal

The first command with real options: `zlm tune <image> <alias> --threads N --ram X
[--gpu --vram Y]`. Establishes the option-parsing/validation pattern (hand-rolled
for now) and the runtime store wiring that later commands follow.

## Scope

- **In:** `tune` flag parsing + validation (`--threads`/`--ram` required;
  `--gpu` requires `--vram`); `--ram`/`--vram` sizes in `mb|gb`; writing the
  runtime alias dir.
- **Out:** the plan content itself (that is `plan/01`/`plan/02`).

## Design

- **No defaults for required limits:** the user sets them; the plan carries them.
- **Validation at the edge:** missing/invalid/unknown flags error before any
  work.
- **Hand-rolled `while/else if`** over `ctx.args.next()` — the pattern task `03`
  replaces with declarative options.

## Arch Provenance

- **own:** the tune flag surface + validation messages.
- **spec:** n/a.
- **consulted:** CLI conventions.
- **derived:** from the original `tune/01 - Tune Bootstrap` (now `plan/01` +
  this task).

## Definition of Done

- [x] `--threads N` / `--ram X` required; `--gpu` requires `--vram`.
- [x] Missing/invalid/unknown flags error; `mb|gb` parsed to bytes.
- [x] The command creates the runtime alias dir.
- [x] `zig build` / `zig build test` green.

## Comments

- Delivered together with the original `tune/01 - Tune Bootstrap` (whose format
  half became `plan/01`); recorded here because the CLI option handling is the
  CLI milestone's concern.
- `--threads=4` (equals form) was not accepted by this hand-rolled parser — the
  motivating bug for `03`.

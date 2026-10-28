01 - CLI Bootstrap

Status: done
Commits: 91c29a0, 7387090

## Goal

Give `zlm` its command surface and mold, so later layers (importer, image,
runtime) plug in without re-implementing argv/help/dispatch.

## Scope

- **In:** project scaffold; the `Args` mold; the command registry + dispatch;
  generated/recursive help; the `zlm image` group (stubs); the identity.
- **Out:** any model logic.

## Design

- **Data-driven registry**, not an if/else chain: `help` is generated from it.
- **Nested commands** via `Command.subcommands` + `command.walk`; a group has no
  `run` and errors with "Missing subcommand".
- **`help`** is a command at the top level and a help request elsewhere
  (`zlm image help` == `zlm image --help`).
- **`Context`** carries `allocator`, `io`, `args`, `commands`.
- **Identity:** "you decide how your language model runs" + "Anti-OOM by design,
  RAM/VRAM-budgeted, CPU-first, mmap-first, tuned to your hardware."

### Layout

```text
src/cli/
  args.zig      # Args cursor
  command.zig   # Context, Command, find, walk
  usage.zig     # generated + path-aware help
  root.zig      # registry + dispatch
  commands/
    help.zig
    image.zig + image/{compile,verify,remove,list,info}.zig
```

## Arch Provenance

- **own:** the command registry/dispatch, `Command`/`Context`, the `Args` mold,
  recursive/generated help, the identity text.
- **spec:** n/a (no external format).
- **consulted:** general CLI conventions.
- **derived:** the `image` subcommand group from the store concept.

## Definition of Done

- [x] `zlm help` lists commands; `zlm help image compile` works.
- [x] `zlm image compile <name> <model.gguf>` dispatches.
- [x] `zlm <path> --help` / `-h` / `help` show the same command help.
- [x] Unknown command/subcommand -> message + help, exit 1.
- [x] `zig build` / `zig build test` green.

## Comments

- Moved here from `bootstrap/01 - CLI Skeleton`; the `bootstrap` milestone was
  renamed `engine` and now holds only the non-CLI foundation.
- The `Args.value` helper already accepts `--x=v` / `--x v`, but no command
  used it — the hand-rolled `while/else if` parsing is what `03` replaces.

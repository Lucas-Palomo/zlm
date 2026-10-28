03 - Improve Flags

Status: done
Commits:

## Goal

Replace per-command hand-rolled argv parsing with a single declarative option
layer, so a command declares its options and gets uniform parsing, validation and
help for free. Fixes the `--x=v` gap and removes the `while/else if` duplication.

## Scope

- **In:** `src/cli/options.zig` (`Spec`, `Parsed`, `parse`); migrate `tune`,
  `run`, `load`, `image/*`; option `--x=v` / `--x v` / `-x v`; typed accessors
  (`uint`, `size`); required/unknown/missing-value errors; optional help derived
  from the spec.
- **Out:** changing command behavior or the `.zplan`/image formats; a full
  argument-parsing library.

## Design

```text
src/cli/options.zig
  Kind   = flag | value | uint | size
  Spec   = { long, short?, kind, required? }
  Parsed = { flags, values, positionals }   + has/get/uint/size
  parse(alloc, *Args, specs) -> Parsed      (uniform errors)
```

Each command:

```zig
const specs = [_]options.Spec{
    .{ .long = "--threads", .short = 't', .kind = .uint, .required = true },
    .{ .long = "--ram", .kind = .size, .required = true },
    .{ .long = "--vram", .kind = .size },
    .{ .long = "--gpu", .kind = .flag },
};
var opt = try options.parse(ctx.allocator, &ctx.args, &specs);
defer opt.deinit();
const threads = opt.uint("--threads", u16).?;
const gpu = opt.has("--gpu");
const [image, alias] = opt.positionals;
```

- One parser owns: `--x=v` / `--x v` / `-x v`, unknown flag, missing value,
  required check, and typed conversion (`uint`, `size`).
- `Args` stays a pure cursor; `value`/`is` move into `options`.
- Help/`usage` can be generated from the specs (removes duplicated help text).

## Arch Provenance

- **own:** the declarative option layer and uniform error policy.
- **spec:** n/a.
- **consulted:** common CLI option patterns (getopt-style), reference only.
- **derived:** from `Args.value` and the per-command patterns it replaces.

## Steps

1. **(as built)** `options.zig`: `Spec`/`Parsed`/`parse` + typed accessors
   (`uint`/`size`); tests for `--x=v`/`--x v`/`-x=v`, unknown/missing/invalid/
   required.
2. **(as built)** Migrated `tune` (removed the `while/else if` and local
   `parseSize`/`missing`).
3. **(as built)** Migrated `run`, `load`, `image/{compile,verify,remove}`.
   `image/{list,info}` take no arguments.
4. Optional help-from-specs: not done (deferred).
5. `zig build` / `zig build test` green; `--threads=4` works end to end.

## Flow / Arch

```text
ctx.args ─▶ options.parse(specs) ─▶ Parsed ─▶ command logic
                                   (has/get/uint/size/positionals)
```

## Definition of Done

- [x] `src/cli/options.zig` with `Spec`/`Parsed`/`parse` + tests.
- [x] `--x=v`, `--x v`, `-x=v` all accepted; unknown/missing/invalid/required
      error.
- [x] `tune`/`run` no longer contain `while/else if` option chains.
- [~] (optional) `usage`/`help` derived from specs — deferred.
- [x] `zig build` / `zig build test` green.

## Comments

- `image compile` has only boolean flags today (`--check`/`--force`) but still
  gains uniform handling.
- Consider a `struct`-derived variant (schema -> typed struct) later; start with
  the table form.

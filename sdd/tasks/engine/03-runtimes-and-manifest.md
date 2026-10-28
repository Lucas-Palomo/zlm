03 - Runtimes and Manifest

Status: done
Commits:

## Goal

Fix the `.zlm` layout around the real workflow: `tune` creates the reusable
`(image, fingerprint)` plan + autotune artifacts, and a runtime is just an
**alias** pointing at them. Multiple aliases share one plan/`.zat`; a binary
`manifest.zrun` maps aliases -> artifacts (binary to avoid manual-edit
corruption).

## Scope

- **In:** the `runtimes/` layout (`<image>_<fp>.zplan`, `<image>_<fp>.zat`,
  `manifest.zrun` binary registry, with a checksum); registry
  read/upsert/remove/atomic write; the shared `utils/binary_writer.zig`
  factoring; `tune` reuse of an existing `(image, fp)` (+ `--force`); the
  `zlm list` and `zlm unload` (`--remove`/`--prune`/`--all`) commands;
  `run`/`load` alias resolution via the registry; orphan pruning; moving the
  `.zat` from `.zlm/autotune/` into `runtimes/`.
- **Out:** migration of old stores (prototype: re-tune); multi-process locking;
  `image remove` cascade; the daemon/session itself (landed in `runtime/08`;
  `zrun`'s reserved flags remain for future socket/session fields).

## Design

Layout:

```text
.zlm/runtimes/
  <image>_<fp>.zplan     # shared by N aliases
  <image>_<fp>.zat       # autotune for the same key
  manifest.zrun          # binary alias registry
```

`manifest.zrun` is a small **binary** file so manual edits cannot corrupt it. It
reuses the project's codec conventions — no new serialization approach, only a
new magic/payload. Layout (little-endian):

```text
MAGIC "ZRUN" | VERSION u32 | flags u32 (reserved, 0) | entry_count u32
  entries...   (each: id, plan, tune, image = u16-len + bytes; fp u64; flags u32, reserved 0)
checksum u64   (Wyhash over the body; load rejects a mismatch)
```

- **Room for expansion.** The `VERSION` plus the header/entry `flags` (reserved)
  and length-prefixed strings let future additions land additively — notably the
  **daemon/session** (the earlier local prototype's `load` ran a socket.io sub-daemon; `runtime/08` added
  the per-plan service with a socket/state). A larger change bumps `VERSION`;
  within a version, unknown reserved bits are ignored.

- Entry: `{ id: []const u8, plan: []const u8, tune: []const u8, image: []const u8, fp: u64 }`.
  `plan`/`tune` are the artifact filenames; `image`/`fp` are kept for display and
  validation (the plan filename also encodes them).
- **Reuse the codec, factor the helpers:** the `appendU16/32/64` functions are
  duplicated across `zman.zig`/`plan/root.zig`/`ztef.zig`; move them to a shared
  `utils/binary_writer.zig` and build `zrun` on that + the existing
  `utils/binary_reader.zig`. `zman` (the image manifest) cannot be reused as a
  payload: its fields are image-specific (content root, gguf size, tensor files).
- Read via `mmap` like the other artifacts; write atomically (`manifest.zrun.tmp`
  + rename).

The format of the registry is decoupled from the artifacts (`.zplan`/`.zat` do
not depend on it), so it can evolve independently.

- **`tune <image> <alias>`**: compute `fp = Config.fingerprint()`; the artifact
  base is `<image>_<fp>`. If `<image>_<fp>.zplan` exists and `--force` is not
  set, **reuse** it (and its `.zat`) and only upsert the alias row; otherwise
  produce the plan/`.zat` (tasks `plan`/`runtime/07`) and write them.
- **`run`/`load <alias>`**: read `manifest.txt`, find the `id`, open the
  referenced `<image>_<fp>.zplan`.
- **Atomicity**: upsert `manifest.txt` via `manifest.txt.tmp` + rename; the
  artifacts are written by the existing plan/`.zat` writers.
- **Alias uniqueness**: one row per `id`; re-tuning the same alias replaces its
  row (and may point at a new `(image, fp)`).
- **Multiple aliases, shared artifacts**: two aliases with the same
  `(image, fp)` point at the same `.zplan`/`.zat`; adding the second does not
  duplicate or rewrite the artifacts.
- **`zlm list`**: top-level command that reads `manifest.zrun` and prints the
  aliases, mirroring `zlm image list` (header + tab-separated). Columns:
  `ALIAS\tIMAGE\tFP` (the shared `(image, fp)`), one row per alias; `no runtimes`
  when empty.
- **`zlm unload`** (the store-mutating counterpart of `load`); `remove`/`prune`
  rewrite `manifest.zrun` (read -> modify -> atomic rewrite; not append-only).
  - `<alias>` (no flags): report; keeps the service online (`runtime/08`). The
    service (a per-plan daemon) is stopped by `--remove` when the alias was the
    last one for the plan, or by `--prune`/`--all --remove`.
  - `<alias> --remove`: delete the alias row. Shared `.zplan`/`.zat` survive while
    another alias references them; the immutable image is never touched.
  - `--prune` (no alias): delete orphan `.zplan`/`.zat` (referenced by no alias);
    keeps aliases.
  - `--all --remove [--prune]`: every alias (and orphans).
  - `--all --prune`: every orphan, keep aliases.
  - `--all --remove`: every alias, keep the `.zplan`/`.zat`.
  - **Validation**: `--remove` without `<alias>`/`--all` -> error; `<alias>` with
    `--prune` -> invalid (prune is store-wide, never per alias).
- **No migration**: the old `runtimes/<alias>/plan.zplan` is abandoned; a store
  is re-tuned. (`.zlm/autotune/` from `runtime/07` is superseded.)

## Arch Provenance

- **own:** the alias->artifact manifest and shared `(image, fp)` artifacts.
- **spec:** —.
- **consulted:** —.
- **derived:** from `runtime/06`/`07` (the `(image, fp)` `.zat` key) and the
  observation that `tune` owns plan creation.

## Steps

1. `src/image/store.zig`: `runtimes` artifact path
   (`<image>_<fp>.zplan`/`.zat`), `manifest.zrun` path + dir helpers.
2. `utils/binary_writer.zig` (factor `appendU16/32/64`); `src/image/zrun.zig`:
   the `ZRUN` writer/reader (entries + checksum), `find` by id, `upsert`, `remove`,
   atomic write; `load` rejects a magic/version/checksum mismatch.
3. `tune`: reuse an existing `(image, fp)` plan (skip re-plan/benchmark) unless
   `--force`; upsert the registry row; write artifacts under `runtimes/`.
   Adding a second alias for the same `(image, fp)` must not duplicate or rewrite
   the artifacts.
4. `zlm unload`: `<alias>` (no-op), `<alias> --remove`, `--prune`, `--all` with
   the combinations/validations above; rewrite `manifest.zrun` atomically and
   prune orphans as requested.
5. `zlm list`: print the registry (`ALIAS\tIMAGE\tFP`, mirroring `image list`);
   `run`/`load`: resolve the alias through the registry to the plan; error
   clearly when the alias or artifact is missing.
6. Move the `.zat` from `.zlm/autotune/` to `.zlm/runtimes/`.
7. Tests: `zrun` write/read/upsert/remove round-trip; two aliases -> one plan;
   reuse vs `--force`; removing one alias keeps the shared artifact.
8. `zig build` / `zig build test` green; e2e: tune two aliases with the same
   config -> one `.zplan`, both run; remove one -> the other still runs and the
   `.zplan` survives; remove the last -> the artifact is pruned.

**As-built.**
- `utils/binary_writer.zig` (shared `appendU16/32/64`; `zman`/`plan`/`ztef`
  refactored onto it).
- `src/image/zrun.zig` (`ZRUN` magic+version+flags+entries+Wyhash checksum;
  `load`/`write`/`upsert`/`remove`/`pruneOrphans`; atomic temp+rename).
- `store.zig`: flat helpers (`runtimesDir`, `runtimePlanPath`/`runtimeZatPath`,
  `registryPath`, `resolvePlan`, `aliasExists`, `loadRegistry`); the `.zat` moved
  from `.zlm/autotune/` to `.zlm/runtimes/`.
- `plan.Reader.openFile`; `Session.open` takes the plan path.
- `tune`: computes `(image, fp_hex)`; if the plan exists without `--force` ->
  reuse (skip compile/benchmarks/write) + `zrun.upsert`; else write
  `runtimes/<image>_<fp>.zplan`/`.zat` + upsert. Second alias never rewrites.
- `run`/`load`: resolve the alias via `manifest.zrun` to the plan path.
- New `zlm list` (`ALIAS\tIMAGE\tFP`) and `zlm unload`
  (`<alias>`, `<alias> --remove`, `--prune`, `--all` combinations; validations
  `--remove` needs alias/`--all`, `--prune`/`--all` reject `<alias>`).
- Verified (ReleaseFast, E2B): `tune rt-a` measured; `tune rt-b` reused one
  `.zplan`; `list` shows both; `unload rt-a --remove` keeps the shared artifact;
  after the last alias, `unload --prune` removes 2 orphans; error cases exit 1.
  `zig build test` 91/91.

## Definition of Done

- [x] `runtimes/<image>_<fp>.{zplan,zat}` + `manifest.zrun` are the layout.
- [x] `tune` reuses an existing `(image, fp)` plan; `--force` re-tunes.
- [x] Two aliases with the same config share one `.zplan`/`.zat`.
- [x] `unload <alias> --remove` leaves shared artifacts; `--prune` (or
      `--all --remove --prune`) removes orphans; `--all --remove` keeps artifacts.
- [x] `unload --remove` (no alias) and `unload <alias> --prune` are errors.
- [x] `zlm list` prints aliases (`ALIAS\tIMAGE\tFP`) like `image list`.
- [x] `zrun` rejects a corrupted file (bad magic/version/checksum).
- [x] `run`/`load` resolve aliases through the registry.
- [x] `zig build` / `zig build test` green; e2e shared-plan run works.

## Comments

- Reuses the codec conventions (magic+version+`appendU*`+`BinaryReader`), not a
  new serialization scheme; `zman` cannot be reused as a payload (its fields are
  image-specific).
- Supersedes the `.zlm/autotune/<image>_<fp>.zat` path from `runtime/07`.
- `unload` is the store counterpart of `load`; `--remove`/`--prune` rewrite the
  registry (read -> modify -> atomic rewrite). When a daemon/session lands, the
  reserved `zrun` fields carry the socket/session/refcount and `unload` becomes a
  real memory release (see `open-items`).
- `image remove` should later warn when aliases/artifacts reference the image;
  deferred.
- No locking yet: two concurrent `tune`/`unload` could race `manifest.zrun`;
  single-process assumption for now.

02 - Native Format and Compile

Status: done
Commits: 16310f4, 59cb458, 1037de2, 8027f09, fe3f95b, 0c098d8, ce8fe03

## Goal

`image compile <name> <model.gguf>` turns a GGUF into a self-contained,
deterministic **image** that no longer needs the GGUF. First real layer of the
rewrite: read an external raw format and store zlm's own.

## Scope

- **In:** `gguf/` importer; `image/` store and formats; the commands `compile`,
  `verify`, `remove`, `list` (`info` is a stub).
- **Out:** `tune` and the runtime; the tokenizer reader; a complete tensor map
  (unmapped tensors are dropped); `$ZLM_HOME`.

## Design

- **ZTEF is extraction, 1:1 native bytes.** No repack here (that is `tune`).
- **Bucketing is arch-agnostic:** tensor name -> `ztef.Kind` (`q_proj`…`embd`),
  no `ModelGeneric` dependency. The seven projection kinds are then merged into
  one `misc` file, ordered layer-major in decode consumption order, so the
  decode streams a single sequential file; `embd` and `norms` stay per-kind
  (the tuner's row-cache/window and HOT decisions key on those kinds).
- **Atomic + no overwrite:** build `.zlm/images/.tmp-<name>/`, then rename;
  error if `<name>` exists (`--force` removes and recreates).
- **One file per responsibility:**
  `zman` = manifest/identity (`content_root` + inventory); `zhash` = integrity
  (per-file sha + self-sha); `zmeta` = full GGUF metadata; `ztef` = tensor bytes.
  `verify <name> [gguf]` is self-contained via `zhash`; the GGUF only checks
  provenance.
- **Content-addressing (identity).** `meta_hash = sha256(gguf[0..data_offset])`;
  each tensor's payload is hashed **while it is copied**; `content_root =
  sha256(meta_hash || sorted(name || tensor_sha))`, stored in `zman`. No separate
  whole-file hash pass; `verify <gguf>` recomputes and compares the root.
- **`zmeta` belongs to `compile`** (hardware-independent, once per model).
- **Metadata is parsed eagerly** (all KVs) and written to `zmeta` in GGUF order
  (deterministic).
- **`--check`** parses and prints without writing.
- **`list` columns** `NAME | ARCH | IMAGE | SERVICE`; `IMAGE` = `building` /
  `ready` / `broken`, `SERVICE` = `online` / `offline` (offline until the
  runtime exists).
- **`info`** is a stub.

### Store

```text
.zlm/images/<name>/
  model.zman   # manifest/identity: content root + size, arch, file inventory
  model.zmeta  # full GGUF metadata, re-encoded
  model.zhash  # per-file sha256 + self-sha
  *.ztef       # tensor bytes: projections layer-major in misc.ztef, embd/norms per kind
```

No `source` reference and no separate registry — the directory is the registry.

### Formats (as built)

- **ZTEF**: `magic|version|kind|entry_count|reserved|entry_bytes|payload_bytes`,
  then entries `name(u16+bytes) layer type n_dims dims[4] offset length
  sha256[32]`, then payload (native bytes, zero-copy from the GGUF). Writer
  returns its sha. **format v2** persists the per-tensor `sha256` in the entry (content
  identity for `tune`'s shared-weight dedup and for per-tensor provenance).
- **ZMAN**: `magic|version|content_root[32]|gguf_size|arch|tensor_count|
  metadata_kv_count|ztef_count` + `(kind, entry_count, byte_length)` per file.
  No integrity hashes. 216 B for E2B.
- **ZHASH**: `magic|version|count|(kind, sha256[32])…` + trailing self-sha.
  407 B for E2B (was 11 entries: 9 ztef + zmeta + zman; the projection merge
  leaves 3 ztef + zmeta + zman).
- **ZMETA**: `magic|version|kv_count` + key + tagged value (arrays recursive).
  16.3 MB for E2B.

### Layout

```text
src/
  gguf/   format.zig reader.zig root.zig      # external raw (read-only)
  image/  store.zig ztef.zig zman.zig zmeta.zig zhash.zig
  utils/  io.zig mmap.zig binary_reader.zig
```

## Steps

1. `utils/{mmap,binary_reader}` + `gguf/{format,reader,root}`; `compile --check`
   prints `arch/version/tensors/metadata/data_off`. **Done** (`59cb458`).
   Fixtures parsed: gemma4 E2B 601, E4B 720, 12B 667; qwen3 4B 398, 8B 399; bad
   file -> `BadMagic`.
2. `image/store.zig` + `image/zman.zig`; `compile` writes `model.zman`.
   **Done** (`1037de2`).
3. `image/ztef.zig`; `compile` writes the 9 `*.ztef` buckets. **Done**
   (`1037de2`). Later: the 7 projection buckets are merged into `misc`, ordered
   layer-major in decode consumption order, so the decode streams one sequential
   file; `embd`/`norms` stay per-kind.
4. `image/zmeta.zig`; the full metadata. **Done** (`8027f09`).
5. `image/zhash.zig` + `verify`/`remove`/`list` + `info` stub; `list` columns
   split IMAGE/SERVICE and renamed to `online`/`offline`. **Done** (`8027f09`,
   `fe3f95b`, `0c098d8`).
6. Content-addressing: `ztef.write` returns the per-tensor sha (fused with the
   copy); `compile` builds `meta_hash` + `content_root`; `zman` carries it and
   `verify <gguf>` recomputes it. **Done.** Warm E2B compile ~30 s -> **~13 s**;
   `--check` 0.02 s.

## Flow / Arch

```text
GGUF (mmap, read-only)
  parse header + metadata + tensor directory     [gguf/reader]
  meta_hash = sha256(data[0..data_offset])       [compile]
  bucket tensors by kind (name-derived)          [ztef.bucketOfName]
  merge projection buckets -> misc, layer-major  [compile]
  write <bucket>.ztef (file sha + per-tensor sha) [image/ztef]
  content_root = sha256(meta_hash || sorted(name||tensor_sha))  [compile]
  write model.zman (content_root, returns sha)   [image/zman]
  write model.zmeta (returns sha)                [image/zmeta]
  write model.zhash (all shas + self-sha)        [image/zhash]
  temp dir -> rename -> .zlm/images/<name>/      [image/store]
```

## Definition of Done

- [x] `compile` writes `model.zman`, `model.zmeta`, `model.zhash` and the
      `*.ztef` files atomically under `.zlm/images/<name>/`.
- [x] `--check` parses and prints without writing.
- [x] `verify <name> [model.gguf]` validates the zhash self-sha and every file
      sha (GGUF only for provenance).
- [x] `compile` has no separate whole-file hash pass; `zman` carries
      `content_root`; `verify <gguf>` recomputes and compares it (a wrong source
      is detected).
- [x] `remove` deletes the image; `list` prints `NAME/ARCH/IMAGE/SERVICE`.
- [x] `info` is a stub.
- [x] `zig build` clean; `zig build test` passes (round-trip tests for
      `ztef`/`zman`/`zhash`); E2B verified end-to-end.

## Arch Provenance

- **own:** ZTEF bucket layout + entry encoding; the `zhash` scheme; the
  content-root scheme (metadata hash + per-tensor sha); the `zman`/`zmeta`
  shapes; store layout and `list` columns.
- **spec:** GGUF header/metadata/tensor directory; GGML type ids and block
  layouts (ggml quant spec).
- **consulted:** llama.cpp/ggml for format details (metadata keys, type ids,
  block sizes) — reference only, not copied.
- **derived:** bucketing by tensor role from GGUF names; `zmeta` as a
  re-encoding of the metadata; the CLI surface from the image concept.

## Comments

- ~~Compile is I/O bound: E2B (3 GB) cold ~62 s, warm ~30 s. Dominated by
  SHA-256 + page-in; fuse hashing with the copy pass later.~~ Resolved by
  content-addressing (identity fused with the copy): warm compile ~13 s.
- Gotcha: in `verify`, `ztef` entry names point into the `.ztef` mmap (unmapped
  on `deinit`); dupe before use — otherwise segfault (139).
- ~~One tensor is not bucketed (an auxiliary root); the runtime must not need
  it.~~ Resolved: `rope_freqs.weight` is kept in the `misc` bucket, and any
  unmapped tensor falls back to `misc`, so none is dropped.
- ~~`--force` removed the old image before the install rename; a failure there
  lost the old image.~~ Resolved by the `.bak-` swap: the old image is renamed
  aside and restored if the install fails (`store.swapIn`), then deleted on
  success.
- ~~`zmeta` has no reader yet; the tokenizer consumer comes with the runtime.~~
  Resolved by `runtime/04` (`src/runtime/tokenizer.zig`; `Session` reads the
  tokenizer from `zmeta`).
- Cross-cutting deferrals live in `sdd/open-items.md`.

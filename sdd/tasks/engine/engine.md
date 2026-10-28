# engine

title: Engine
reason: The non-edge foundation: the external importer (`gguf/`), the native
        image store and formats (`ztef|zman|zmeta|zhash|zcom`) and the `compile`
        step that turns a GGUF into a self-contained, deterministic image. The
        image is the source of truth; the runtime needs no GGUF.

## Tasks

### [01 - Native Format and Compile](01-native-format-and-compile.md)
Status: done

Context: `gguf/` importer + `image/` store and formats (`ztef`, `zman`,
`zmeta`, `zhash`); `compile` (GGUF -> image, identity fused with the copy via a
content root), `verify`, `remove`, `list`. Verified on gemma4 E2B. (Originally
`bootstrap/02`.)

### [02 - ZCOM (derived data)](02-zcom.md)
Status: done

Context: the image's derived data — bytes deterministically transformed from the
weights (norm f32, RoPE tables, embedding rows) computed once in `compile` and
read by the runtime. Sits with `ztef|zman|zmeta|zhash`; referenced by the (N)
`.zplan`s, which carry "how to consume". No `zat` yet. (Originally
`bootstrap/03`.)

### [03 - Runtimes and Manifest](03-runtimes-and-manifest.md)
Status: done

Context: the `.zlm/runtimes/` layout — shared `<image>_<fp>.zplan`/`.zat`
artifacts and a binary `manifest.zrun` mapping aliases to them. `tune` reuses an
existing `(image, fp)` (or `--force`); `run`/`load` resolve aliases through the
registry; `zlm unload` removes aliases and prunes orphans. N aliases share one
artifact.

## Notes

- The CLI surface for `compile`/`verify`/`remove`/`list` is delivered here but
  owned by the `cli` milestone (its bootstrap); this milestone owns the
  formats/store/compile logic.
- `bootstrap` was the original name; the CLI part moved to `cli/` and the
  plan/format part to `plan/`.

14 - Prefix-Cache Partial-Hit Correctness and Prompt-Format Invalidation

Status: done
Commits:

## Goal

Fix two defects exposed by the prefix cache (task 13):

1. **Partial-hit corruption.** When a prompt reuses only part of a cached
   segment (a *different* prompt that shares the chat-template prefix),
   `loadFromCache` loads the **wrong** K/V and the model degenerates (garbage
   tokens). Cache-cold runs and full hits are correct.
2. **Stale `.zpc` across a recompile.** The `.zpc` key is `Wyhash(content_root)`,
   and `content_root` depends only on the GGUF metadata + tensor bytes, **not**
   on `zmeta`. Recompiling an image (e.g. task 08's chat markers, same weights)
   leaves the old `.zpc` valid by key, so a prompt-format change reuses old
   cache state.

The fix makes the canonical K/V payload **prefix-contiguous** and folds the
prompt-format identity into the cache `Key`.

## Scope

- **In:**
  - The canonical K/V byte order (`Session.saveKvRange`/`loadKvRange`):
    buffer-major -> **position-major** so `[0, n)` is contiguous.
  - A prompt-format fingerprint field on `prefix_cache.Key`, derived from the
    effective chat template.
  - `.zpc` format `VERSION` bump (the payload byte order changed).
- **Out:** the radix-tree algorithm itself (lookup/commit/eviction are sound);
  the intra-execution K/V buffer layout (`write_kv`/`splitKv`); SWA per-layer
  reuse (still blocked, task 13).

## Root cause (evidence)

`nbuf = 15` K/V buffers (one per gemma-4 E2B layer group). `saveKvRange` writes
the canonical payload **buffer-major**: all positions of buffer 0, then all of
buffer 1, ... So a segment `[0, 15)` payload lays out as
`[buf0 pos0..14][buf1 pos0..14]...`.

`loadFromCache` loads a partial segment with a contiguous slice:

```zig
const n = (end - seg.token_begin) * per;   // per = bytes/token over ALL buffers
try self.loadKvRange(seg.token_begin, end, payload[0..n]);
```

For `[0, 4)` this takes the first `4 * per` bytes of a buffer-major payload =
"buffer 0's positions 0..~60, then buffer 1's ..." — not "positions 0..3 of
every buffer". Only a **full** segment (the whole payload) is correct.

Measured (debug build, same 4 tokens `<bos> <|turn> user \n`, gemma-4 E2B):

```text
cold A:  matched=0 len=15 nbuf=15 kv4hash=f21b9eee6bc9f542
warm C:  matched=4 len=16 nbuf=15 kv4hash=fc2655d6bd02eb5c   # differs -> wrong K/V
```

The loaded K/V differs from a freshly computed one -> garbage. Full hits load
the whole payload -> correct.

## Design

### Fix 1 - position-major canonical layout

Reorder `saveKvRange`/`loadKvRange` so the **position is the outer loop** and
the buffer/head the inner ones:

```text
for pos in [begin, end):
    for kb in kv_buffers:
        for half in {K, V}:
            for h in heads:
                copy head_dim
```

Then the bytes of `[0, n)` are exactly the first `n` positions across all
buffers, so `payload[0..n]` is a valid prefix and `loadKvRange` reads it
directly. Round-trip over the same `[begin, end)` is unchanged (both sides use
the new order); only the on-disk payload byte order differs -> `.zpc`
`VERSION` 2 -> 3, and `openBase` rejects old files (`BadVersion`, ignored).

The intra-execution K/V buffer layout is **not** touched; only the canonical
serialization for the cache changes.

### Fix 2 - prompt-format identity in the `Key`

Add `format: u64` to `prefix_cache.Key`, included in `eql` and `fingerprint`:

- `format` = `Wyhash` of the **effective chat template** (the turn markers plus
  the think decorations) that `Session.formatPrompt` uses - i.e. the stored
  `zlm.chat.*` when present, else the arch template, merged exactly as the
  prompt is built.
- Factor the merge into `Session.effectiveTemplate()` so `formatPrompt` and the
  key share one definition; `promptFormatFingerprint(self)` hashes it.
- `.zpc` header grows one `u64` -> `key_bytes = 8*4 + 1 + 3`; `writeBase` and
  `openBase` serialize/deserialize it. `VERSION` bump already covers the format
  change.

Result: a recompile that changes the prompt format (markers/think) yields a
different `.zpc` fingerprint, so the old artifact is ignored even though
`content_root` is unchanged.

## Steps

1. `session.zig`: reorder `saveKvRange`/`loadKvRange` to position-major. **Done.**
2. `prefix_cache.zig`: add `Key.format`; include in `eql`/`fingerprint`. **Done.**
3. `zpc.zig`: `VERSION = 3`; serialize/deserialize `format`. **Done.**
4. `session.zig`: factor `effectiveTemplate`; add `promptFormatFingerprint`;
   set `Key.format` in `cacheKey`. **Done.**
5. Tests: `saveKvRange`/`loadKvRange` multi-buffer **partial** round-trip
   (regression) + `Key.format` in `eql`/`fingerprint`. **Done.**
6. Verify: `zig build` / `zig build test` (227/227); e2e on gemma-4 E2B — cold
   prompt then a different prompt sharing the prefix, both coherent (the second
   previously degenerated). **Done.**

## Flow / Arch

```text
commit:    session KV[begin,end) --saveKvRange--> canonical payload (pos-major)
lookup:    radix tree -> matched prefix
load:      payload[0..matched*per] --loadKvRange--> session KV[0,matched)
prefill:   tokens[matched..] -> sample
Key:       { model=content_root, arithmetic, layout, dtype, format=chat-template }
```

## Definition of Done

- [x] A partial prefix hit produces the **same** K/V (and output) as a cold run;
      the previously degenerate second prompt is coherent.
- [x] Full hits and cold runs are unchanged.
- [x] `Key.format` invalidates the `.zpc` when the prompt format changes with an
      unchanged `content_root`.
- [x] `.zpc` `VERSION` bumped; an old `.zpc` is ignored, not misread.
- [x] Regression test covers the multi-buffer partial load.
- [x] `zig build` / `zig build test` green; e2e on gemma-4 E2B.

## Arch Provenance

- **own:** the position-major canonical layout and the prompt-format key field.
- **derived:** `runtime/13` (prefix cache + canonical layout), `runtime/10`
  (batched prefill), `architectures/08` (chat markers resolved at compile).

## Comments

- **Why the layout was wrong.** The canonical order was only exercised by full
  loads in the original e2e (single repeated prefix). The partial path (a
  divergent continuation) was untested with multiple K/V buffers.
- **Format is over-invalidation, by design.** A given token sequence's K/V does
  not depend on the template, and the token tree already isolates different
  formats. Including `format` in the key is hygiene: it guarantees a recompile
  never reuses cache entries from a different prompt shape, at the cost of not
  sharing across templates (acceptable).
- **In-memory journal.** The runtime's writable journal uses the same
  `saveKvRange`, so it is position-major too within a process; only the on-disk
  format version gates cross-process compatibility.
- **A stale `.zpc` is self-invalidating** once `format` is in the fingerprint;
  `unload --prune` / `image remove` still reap by image ownership.

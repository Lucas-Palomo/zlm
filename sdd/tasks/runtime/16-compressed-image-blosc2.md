16 - Compressed Image (blosc2)

Status: done
Commits: 4b402b8 (implemented; later removed -- see the discard note)

## Goal

An **opt-in compressed image**: `compile --blosc2=N` writes the ZTEF payload
compressed with Blosc2 (a byte-shuffle + a fast codec), so the disk reads fewer
bytes for a model that does not fit. The image records the instruction; the
`run` decompresses (today inline on the reader). The weights are **not**
re-quantized -- only their stored bytes shrink.

## Scope

- **In:** the `-Dblosc2` build option (`build_options.have_blosc2`) and the
  `--blosc2=N` compile flag (0-9, the Blosc clevel), **comptime-gated** (no link
  => no flag, no code); a compressed ZTEF payload (v3) + the per-tensor
  instruction; the runtime inline decompression + the per-dispatch cache.
- **Out:** changing the quantization; the decompression on the prefetch workers
  (`--prefetch-threads`; open, see `runtime/15`); a compressed image for a model
  that fits (the user opts in per image); the store duplication (the ZTEF *is*
  compressed).

## Design

### The flag is comptime-gated

`zig build -Dblosc2` links the system Blosc2 (`pkg-config`) and compiles in the
flag; without it the flag does not exist (the string, the parse and the help are
removed):

```zig
if (comptime build_options.have_blosc2) {
    if (opt.has("--blosc2")) { ... }   // N = 0..9, default 5; codec = zstd
}
```

`specs`/`help_text` themselves are `if (build_options.have_blosc2) ...`, so a
plain build is dependency-free and carries no blosc2 symbol.

### The image (ZTEF v3)

The compile compresses the payload **per tensor**: `blosc2_compress(clevel,
BLOSC_SHUFFLE, typesize, src, len, dst, cap)`. The `typesize` is the shuffle
stride = the quant's **block size** (`image/ztef.typesize` -> `ggml.blockLayout`;
q4_1 = 20, q4_0 = 18, q8_0 = 34, q6_k = 210, q4_k = 144, f16/bf16 = 2, f32 = 4);
it is what lifts the ratio from ~6% to ~11% (it separates the block scales from
the quants). A tensor only stores the compressed bytes when they are **smaller**
(`FLAG_COMPRESSED`); an incompressible tensor stays raw, so the image never
grows. The codec is **zstd** (`blosc1_set_compressor("zstd")`, `CODEC_ZSTD`).

Each entry records the stored `{offset, length}`, the original `raw_length` and
the `flags`; the header records the instruction `{ scheme: blosc2, clevel,
codec }` (the reserved word) and the version `VERSION_COMPRESSED = 3`. The
payload is staged in a temp file, so the whole payload is never held in RAM.
`VERSION = 2` (uncompressed) is unchanged: a `.none` instruction is
**byte-identical** to before.

### The runtime

- **`Reader.slice(e)`**: for a compressed entry, `blosc2_decompress` into a heap
  buffer of `raw_length` and return it; the buffer is **cached per entry**
  (`Reader.cache`) so a tensor decompresses once per window. For an uncompressed
  entry, the mmap slice (today).
- **`resetDecompCache`**: the executor calls it once per token / per prefill
  block (via `Session.resetDecompCaches`, `executor.zig`), so the decompressed
  working set stays bounded (the streamed weights are decompressed on demand,
  not held for the whole run).
- **Strategy**: the decompression is **inline** (on the consuming thread).
  Decompressing on the `--prefetch-threads` workers (a decompressed-buffer
  window / arena) is **open** -- it interacts with the `runtime/15` ring.

### The interaction with the read buffer (`runtime/15`)

The tune turns the staging arena **off** for a compressed image
(`image_compressed` -> `arena_bytes = 0`): the reader decompresses inline, so a
raw-`pread` arena adds nothing. Under `runtime/15` (the arena removed) this
becomes: a compressed image has **no ring** either, and the weights stream
through the mmap + the reader's decompression cache. The `--prefetch-threads`
decompress-ahead is the open path to a compressed buffer.

## Flow / Arch

```text
compile --blosc2=9  ->  per-tensor blosc2_compress(N, SHUFFLE, typesize) -> v3 .ztef
                        + the instruction {blosc2, clevel, zstd} + raw_length/flags
tune                ->  reads the instruction -> arena/ring off (compressed)
run                 ->  Reader.slice(e): blosc2_decompress -> cached -> the weight bytes
```

## Steps

1. Spec + build: the `-Dblosc2` link + the `have_blosc2` option (**done**,
   uncommitted); `image/blosc2.zig` (the FFI, `SHUFFLE`, `CODEC_ZSTD`,
   `typesize`); the `--blosc2=N` flag comptime-gated in `compile` (**done**,
   uncommitted).
2. Image: the per-tensor compression (`writeCompressed`, temp-file payload), the
   entry `{offset, length, raw_length, flags}` + the header instruction;
   `VERSION_COMPRESSED = 3` (**done**, uncommitted); a `.none` image stays format v2
   byte-identical.
3. Runtime: `Reader.slice` decompresses + the per-entry cache;
   `resetDecompCache`/`Session.resetDecompCaches` per dispatch (**done**,
   uncommitted).
4. Tune: read the instruction; the arena off for a compressed image (**done**,
   uncommitted). The `inline | workers` decompression strategy is **open**.
5. Tests: `ztef` "compressed round-trip" (compress -> decompress -> the original
   bytes) and "uncompressed image is byte-identical" (**done**, uncommitted).
6. Verify: `zig build` / `zig build test` green in both modes (**done**:
   plain 311 pass + 1 skip; `-Dblosc2` 312 pass). The qwen3-72B measurement was
   done and showed **no expressive gain**, so the feature was discarded (see
   Comments).

## Definition of Done

- [x] `--blosc2=N` exists only with `-Dblosc2` (comptime-gated); a plain build is
      unchanged and dependency-free.
- [x] A `--blosc2` image round-trips (the bytes are identical after
      decompression); a `.none` image is byte-identical to before.
- [x] `zig build` / `zig build test` green (both modes: plain and `-Dblosc2`).
- [x] The disk read drops and the decode improves on the disk-bound 72B (or the
      measurement shows it is not worth it). **Measured: not worth it** -- see
      the discard note.
- [x] The decompression strategy is decided (inline vs the `--prefetch-threads`
      workers), reconciled with the `runtime/15` ring. **Decided: discard** (the
      compressed path was removed).

## Arch Provenance

- **own:** the opt-in compressed image, the comptime-gated flag, the per-tensor
  shuffle + the zstd codec choice, the decompressed-buffer cache.
- **spec:** the Blosc2 API (`blosc2_compress`/`blosc2_decompress`,
  `blosc1_set_compressor`).
- **derived:** from the measurement that the Q4 quants are near-incompressible
  except with a byte-shuffle (8% -> 11.5%), and that a 4-thread Blosc2
  decompresses faster than the disk (2.23 GB/s vs 1.4 GB/s).

## Comments

- **Measured ratio.** With the shuffle, Blosc2 + zstd cl9 gives ~11.1% (0.889) on
  the Q4_1 projections; the decompression scales with threads (nt=4 = 2.23 GB/s
  > the 1.4 GB/s disk). Without the shuffle the weights are near-incompressible
  (<=8%, and lz4 expands).
- **The compression is a one-time compile cost** (zstd cl9 ~0.02-0.06 GB/s =>
  ~20 min for 45.7 GB); the decompression is per-run.
- **Discarded (measured).** The 72B test showed **no expressive gain**: on the
  disk-bound 72B (31 GiB host, 44 GB image) the blosc image decoded at
  **~63-82 s/tok** vs **~14 s/tok** for the uncompressed image, even with the
  ring and a parallel decompression pool. The disk saving is only ~7% (40.6 vs
  43.9 GB) and the zstd decompression dominates. The `--blosc2` / `-Dblosc2`
  flag, the compressed ZTEF (v3) and the ring's decompression path were removed.
- ~~**As-built (uncommitted).** The compile side (flag + v3 writer), the runtime
  inline decompression + the per-dispatch cache, and the tests are done; the two
  modes are green. The remaining work is the 72B measurement and the
  decompression-strategy decision.~~
- ~~**The cache window.** `resetDecompCache` is per token / per prefill block, so
  a token decompresses each streamed tensor once and the cache never grows past
  one block's worth.~~
- ~~**Interaction with `runtime/15`.** ...ring-less (mmap + decompression)...~~

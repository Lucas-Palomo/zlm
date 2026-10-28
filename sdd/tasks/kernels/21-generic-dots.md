21 - Generic Dots

Status: done
Commits:

## Goal

Make the matvec's row-dot helpers **generic**, closing the last per-quant residue.
Today `lmHeadRows` (the `lm_head` / logits, a single-row matvec) calls
per-quant dots:

```zig
dotQ8_0(row, x)
dotQ4_K(row, x)
dotQ4_KPre(row, x, xsum)   // the q4_k with the precomputed activation sums
dotQ6_K(row, x)
```

These are a **single-row matvec (m = 1)** — the `fpBlock` of the L1 type over the
row, then the reduce. Make them one generic `dot(comptime K, row, x, ...)` on the
L1 contract, exactly like the sweep, so **no per-quant code** remains in the
matvec.

## Scope

- **In:** a generic `dot`; the `lmHeadRows` routing; the removal of the per-quant
  dots; the parity/`lm_head` verification.
- **Out:** the `q5_k` (see Comments — it is a **dequant**, not a matvec);
  `iq`; the GPU.

## Design

### The generic dot

- `fn dot(comptime K: type, row: []const u8, x: []const f32) f32` — the L1
  `fpBlock` accumulated over the row's blocks + the horizontal reduce (the same
  step the sweep uses). Lives with the single form (`form/`, since it is the
  `single` form with m = 1).
- `lmHeadRows` calls `dot(K, row, x)` per row instead of `dotX`.

### The q4_k `xsum` (do not regress)

- `dotQ4_KPre` exists for **perf**: the `xsum` (the per-32-chunk activation sums)
  is row-independent, so `lmHeadRows` computes it **once** and reuses it across
  the M rows. The generic `dot` must preserve this — accept an optional
  precomputed `xsum` (or the L1 exposes a `xsumLen` / a precompute hook), so the
  q4_k path stays `dotQ4_KPre`-equivalent. Verify the `lm_head` time does not
  regress.

## Steps

1. **Done.** `form/dot.zig`: `dot(comptime K, row, x, xsum?)` — the L1 `fpBlock`
   accumulated over the row's blocks at the native width, then one reduce
   (`dotXsum`, shared). A new module (not `single.zig`) so both `single.zig` and
   `batched.zig` import it without an import cycle. The `has_min` `xsum` is
   optional: supplied (reused across rows) or, when omitted, computed once into
   a bounded cache on the cold `dotComputed` path.
2. **Done.** `lmHeadRows` (q6_k / q4_k / q8_0) routes through `dot`; the q4_k
   precompute is still done once per call and passed in. `batched.zig`'s
   `matVecQ6_KBatchedRows` also routes through `dot`. The per-quant `dotQ8_0` /
   `dotQ4_K` / `dotQ4_KPre` / `dotQ6_K` are removed (and their `root.zig`
   re-exports; nothing outside the module used them).
3. **Done.** `form/dot_test.zig`: `dot` equals the production single-row sweep
   for every quant (both `xsum` paths, bit-exact); `lmHeadRows` equals
   `matVecRows` (bit-exact).
4. **Done.** `zig build -Doptimize=ReleaseFast -Dcpu=native` green; `zig build
   test` 277/277 (was 275); changed files `zig fmt` clean; E2B decode unchanged
   (warm decode ~4.1-4.2 s / 64 tok, no regression).

## Definition of Done

- [x] One generic `dot` on the L1 contract; the per-quant `dotX` are gone.
- [x] The q4_k `xsum` precompute is preserved (no `lm_head` regression).
- [x] Parity holds; `zig build test` green.

## Arch Provenance

- **own:** the generic dot.
- **derived:** `kernels/17` (the L1 contract + the sweep), `kernels/20` (the F/BF
  families).

## Comments

- **The `q5_k` is NOT a matvec.** It is a **dequant** (`k_quants.dequantQ5_K`),
  used for `per_layer_token_embd` (a lookup), not a GEMV; `dequantTo` is a
  `switch` over the type. Making the *dequant* generic (on the L1 contract) is a
  separate task (`kernels/22`?) if wanted — out of scope here.
- **The dots are the `single` form with m = 1** — they belong with the single
  form, not in `kernels/*`.

02 - ZCOM (derived data)

Status: done
Commits:

## Goal

Add `zcom` — the image's **derived data**: bytes deterministically transformed
from the weights (norm weights already f32, RoPE cos/sin tables, embedding/PLE
rows dequantized). Computed once in `compile`; the runtime reads them instead of
re-deriving per run. This restores the fast path that a naive runtime would pay
for on every token.

## Scope

- **In:** the `zcom` format (`image/zcom.zig`); generating it in `compile`;
  reading it in the runtime; the minimum set the first decode needs.
- **Out:** autotune/layout selection (`zat`) — deferred; large derived payloads
  paging; GPU.

## Design

- **`zcom` is image data, not plan.** Deterministic and hardware-independent
  (no threads/RAM/VRAM): same image -> same `zcom`, on any machine. It sits with
  `ztef|zman|zmeta|zhash` and is referenced by every `.zplan` (there are **N**
  plans, one per config fingerprint — the derived bytes are not duplicated).
- **`zplan` is "how to consume the image".** It carries config + `Dispatch[]` +
  `ResidencyGraph` and the **selection** (which derived regions are used, and
  the layout/atomic chosen). Hardware decisions live there, never in `zcom`.
- **Deterministic generation:** derive from the image's own `ztef` bytes
  (f32 = dupe; quant = `dequantTo`), so `zcom` is reproducible and verifiable.
- **Criterion — what belongs in `zcom`:** small, reused derived data. Not the
  big weight tables: the source `ztef` stays mmap-backed and the runtime does a
  per-row lookup, so a full `token_embd` would only duplicate bytes (7.5 GB in
  f32 for E2B) with no decode-time gain. Quantizing/reducing an embedding table
  does not change that (size/duplication is the objection, not the format).
- **Minimum set (first cut):**
  1. **norm weights in f32** — the f32 `TensorView` for every norm-role tensor
     (`rms_norm` needs f32; avoids dequant per token);
   2. **RoPE `inv_freq`** — per `(freq_base, head_dim, n_rot)` config (the
      runtime rebuilds cos/sin rows per position). Full cos/sin tables were
      rejected: ~268 MB/config for E2B vs a few hundred B for `inv_freq`;
  3. ~~**embedding/PLE rows in f32**~~ **dropped from `zcom`**: the big table
     stays in `ztef`; the embedding is a per-token row lookup (dequantize the
     row in the run). Only *small* embedding-like tables (special tokens,
     positional/per-layer inputs) would qualify.
- **Format:** `magic|version|region_count` + region table
  `(id, kind, owner, offset, length)`, then the payload. `id` = a stable
  `(arch, role, layer, sub-kind)` key the runtime resolves; payload is
  bytes only. 64-byte aligned regions (matches the vector kernels).
- **`zhash` covers `zcom`** (per-file sha + self-sha) so `verify` validates it.

## Arch Provenance

- **own:** the `zcom`-as-image-data split (data vs consumption), the region
  table, the minimum derived set.
- **spec:** `ztef`/`zman`/`zmeta`/`zhash` shapes (from `02`); RoPE/norm math
  (from `kernels/`).
- **consulted:** the earlier local prototype's `zcom` (derived bytes precomputed at
  `compile`/`load`) — reference for the concept, re-derived here.
- **derived:** from `02` (image is deterministic) and the runtime path.

## Steps

1. **(as built)** `image/zcom.zig`: `magic|version|table_bytes|region_count` +
   region table `(kind, role, layer, offset, length)` + 64 B-aligned payload;
   `write`/`Reader` (`find` by kind/role/layer, `slice`); round-trip + alignment
   tests.
2. **(as built)** `compile` derives `norm_f32` regions from the image's `ztef`
   (f32 = copy, quant = `dequantTo`) and writes `model.zcom`. Embedding-to-f32
   was dropped from the minimum cut (whole-table f32 is too large; a
   row-indexed form comes later).
3. **(as built)** RoPE: `compile` loads the arch config from the already-written
   `model.zmeta`, collects the distinct `(freq_base, head_dim, n_rot)` configs
   across layers (first-appearance order), and writes one `rope_inv_freq`
   region per config. **Encoding: `inv_freq`, not full cos/sin** (a full table
   for E2B is ~268 MB/config; `inv_freq` is `n_rot/2` f32, a few hundred B).
   New `Kind.rope_inv_freq = 3`, keyed by `role = 0` and `layer` = **rope config
   id** (documented on `Reader.find`). Payload
   `inv[i] = base^(-2*i/head_dim)` via `zcom.ropeInvFreq`, bit-identical to
   `RopeTable.init` / `ropeProportional`. `zcom` gains round-trip + oracle
   tests. (E2B: 2 configs, 768 B total.)
4. **(as built)** `zhash` gains `zcom`; `verify` covers it (iterates the hashes).
5. **(as built)** `Runtime.Session` opens `zcom` and exposes `normWeight(role,
   layer)`; `runNorm` prefers the plan tensor's f32 weight.
6. `zig build` / `zig build test` green; validated on gemma4 E2B
   (~1.17 MB zcom, of which 768 B is RoPE `inv_freq`).
7. Fixed latent bugs found while wiring: `zmeta` writer counted all keys but
   skipped null ones, and array items were written with a per-item type tag the
   reader does not expect.

## Flow / Arch

```text
compile:  ztef bytes --(dequant/rotate)--> zcom regions --> model.zcom
verify:   zhash covers zcom (per-file sha)
run:      zplan (how to consume) --> zcom region(role,layer) --> kernels
```

## Definition of Done

- [x] `image/zcom.zig` round-trips (write -> read) byte-identical.
- [x] `compile` writes `model.zcom`; `zhash`/`verify` include it.
- [x] Minimum regions: **norm f32** and **RoPE `inv_freq`** (Kind 3) done;
      embedding f32 dropped (per-row lookup in the run).
- [x] A runtime path reads a `zcom` region instead of re-deriving it
      (`Session.normWeight`).
- [x] `zcom` is deterministic (pure transform of the `ztef` bytes).
- [x] `zig build` / `zig build test` green.

## Comments

- **N plans:** plans should live per fingerprint
  (`models/<image>/plans/<fp>.zplan`), not one `runtimes/<alias>/plan.zplan`;
  a `store`/`tune` change, tracked separately.
- **Not `zat`:** autotune/layout classes (repack choice, macro/micro plan) are a
  later concern and belong in the plan, not `zcom`.
- **Minimum-cut rationale:** norm + RoPE + embedding cover the first decode
  (`runtime/02`); other derived forms (fused scale/min caches) come later if the
  measurements justify.

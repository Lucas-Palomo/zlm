13 - Prefix Cache (K/V reuse across executions)

Status: done
Commits: 216760e b17b343 9d6a4de 9e9e53d ce40e7d

## Goal

Cache the K/V of prefixes **already processed in a previous execution** so a
repeated prefix skips the prefill: a multi-turn chat reuses the previous turns'
K/V and prefills only the delta, and the TTFT of the shared part drops to ~0.

It is a **persistent KV cache** (cross-execution), organized as a **radix tree**
over the causal token path, keyed by a **K/V-structure fingerprint shared across
plans/runtimes of the same image with the same K/V plan** (invariant to
`--threads`/`--ram`/`--vram`). Delivered in two phases: **phase 1** (radix tree +
canonical `.zpc` + Session wiring + warm) and **phase 2** (persistence/budget,
canonical layout, coding e2e, and the adjacent tune/runtime items). This spec
covers both as-built.

## What it is (and is not)

- **It is a KV cache.** It stores the actual `K`/`V` bytes per `(layer, position)`
  of prefixes and reuses them across executions. A HIT returns the **identical**
  K/V bytes, so the output is unchanged.
- **It is not the intra-execution KV buffer** (`runtime/03`, the session's live
  K/V): that one holds the K/V of the **current** generation (why decode is
  O(1)/token). This one persists **across** executions and skips the prefill.
  They are complementary.
- **It is not "memoization of the plan."** The plan and the Executor are
  untouched; the runtime re-decides nothing. The cache lives in the
  Service/Session, like the live KV buffer.
- **A HIT is exact** — no tolerance, no approximation, no quality gate.

| mechanism | scope | what it avoids | artifact |
|---|---|---|---|
| KV buffer | the current generation | recomputing past K/V per token | session memory |
| **KV cache (this)** | across executions | the prefill of a repeated prefix | `.zpc` |

## Context

Prefill is **re-computable waste** when the same prefix recurs (a chat
re-processes `1..N-1` every turn; the E2B prefill is ~1 s for 32 tokens and grows
with the prompt). The cache removes that waste **without touching quality** — it
reuses a deterministic K/V.

**Coding / agentic is the target scenario.** The canonical prefix is large and
stable (system prompt + tool definitions + repository context) while the
generation is short, so the **prefill dominates**. A HIT of the stable prefix
takes TTFT to ~0. Measured e2e: E2B, 126-token prompt — TTFT **9.4 s cold → 0.23 s
warm (~41×)**; 32-token — **2.3 s → 0.23 s (~10×)**. Decode is unchanged.

## Scope

- **In (phase 1):** the `PrefixCacheKey` fingerprint; the radix tree; the
  canonical `.zpc`; the Session wiring (lookup → resume or prefill+commit); the
  `tune --prefix-cache` warm.
- **In (phase 2):** the `.zpc` as a **persistent, disk-backed, mmap-managed**
  artifact (compaction, atomic flush); the cache ceiling from `planBudget`; the
  canonical, order-independent layout fingerprint; the coding e2e; and the
  adjacent **content-hash shared-weight detection** (compile persists, tune
  applies) and **prefill residency nodes**.
- **Out:** `--kv`/`--gate` (KV precision + quality gate); GPU/device migration;
  tolerant ("near") matching; cross-image (different weights) sharing; **SWA
  per-layer reuse** (blocked — see Design).

## Design

### Invariant

The K/V of a prefix is a pure function of `(image, arithmetic, tokens)`. A HIT
returns the **identical** bytes → the output is **unchanged**. The cache lives in
the Service/Session; the plan and the Executor are untouched.

### The cache fingerprint (`PrefixCacheKey`)

Scopes the whole tree; a mismatch invalidates every segment:

```text
PrefixCacheKey {
    model       // Wyhash of zman.content_root (GGUF metadata + every tensor sha)
    arithmetic  // { cpu_isa, gpu/gpu_id, each dispatch's atomic_id/matvec_plan/
                //   sem_op/in_len/out_len }, excluding parts/prefill
    layout      // canonical K/V shape list (n_kv_heads, head_dim), ordered by
                //   first layer; no plan slot, no max_seq
    dtype       // f16 (the KV dtype)
}
```

It is the **projection of the config onto the K/V structure**: shared across
plans/runtimes of the same image with the same K/V plan; **excludes**
`--threads`/`--ram`/`--vram` and per-dispatch `parts`/`prefill`; **includes the
device** (a kernel/device change changes the bytes). The matvec splits **output
rows** and the attention is **per-head, serial over positions**, so no reduction
is split across threads → the resource knobs do not change the K/V.

The artifact is `runtimes/<image>_<prefix_cache_fp>.zpc`, where
`prefix_cache_fp = Key.fingerprint()` — **not** the plan's config fingerprint, so
two runtimes of the image tuned with different `--threads`/`--ram` resolve to the
**same file**. (Verified: `--threads 8/--ram 8gb` and `--threads 4/--ram 6gb`
share one `.zpc`.)

### Canonical, stride-independent layout

The K/V of position `i` does not depend on the buffer's `max_seq` (capacity), so
the payload stores per position `K` then `V` (each `n_kv_heads × head_dim`,
head-major), **independent of `max_seq`**; loading into any `max_seq` re-strides.
The shape list is **canonical**: one entry per unique K/V buffer, ordered by
**first layer** (not the dispatch order), with **no plan `slot`** — so two plans
with the same K/V structure but different buffer numbering hash the same
(phase 2 fixed the order/slot fragility).

### Radix tree (variable-length segments)

A node's edge is the token range `[begin, end)`; its payload is the canonical K/V
of that range followed by the `u32` token ids (in mapped regions, zero-copy):

- **Lookup:** walk matching from the root, keeping the **maximal common prefix**.
  A divergent continuation is a **child starting at the divergence offset** (no
  node split, no byte copy); a matched node may be used partially
  (`[token_begin, min(token_end, matched))`).
- **MISS after L:** prefill `tokens[L..]`, then **commit** the new suffix as a
  child of the matched path.
- Attention is causal → the tree is the set of distinct prefixes; sharing is a
  shared **path** (a node is never shared across different parents).
- **LRU eviction** under the byte budget; `refcount > 0` (in-flight) segments are
  pinned.

### Storage — the `.zpc` artifact (persistent, mmap-managed)

Written by `tune --prefix-cache` (and re-written on compaction). Layout:
`magic | version | flags | Key | ... | records { parent, begin, end, off, len }`,
then per segment the canonical K/V + the token ids. Properties:

- **Phase 1:** the base `.zpc` is an immutable read-only mmap, referenced
  **zero-copy**; the runtime appends new segments to a separate writable,
  file-backed journal mmap, and the **segment index is itself a mapped region**
  (nothing on the heap).
- **Phase 2 (persistence):** the `Memo` carries a **persisted counter header**
  (`addMeta` restores the previous tree instead of resetting); the service flushes
  a dirty cache **on idle** (`writeCacheFile`, compacting base+journal into the
  `.zpc`), so the daemon's appends **survive a restart**. The flush writes a
  process-unique temp and `rename`s it (**atomic**), so a crash mid-flush cannot
  leave a torn `.zpc`.
- The fingerprint in the header makes stale files **self-invalidating**: a
  re-tune that changes the arithmetic changes the fingerprint → the file is
  ignored. A `.zpc` is **image-owned**: `unload --prune` reclaims it only when the
  **image no longer exists**, and `image remove` cascades the image's `.zpc`s.

### Budget integration (phase 2)

The **tuner** computes the cache ceiling as a slice of the plan's **window**
(`planBudget`'s leftover after `fixed` + `reserve` + the live K/V) and records it
in the plan; the runtime sizes the `.zpc` to it and never exceeds it. So a small
`--ram` degrades the **cache** before the weights (anti-OOM), not `ram/8`.

### Tune-time warmup (`tune --prefix-cache`)

Boolean flag, **default off** (no cache file). With it, `tune` prefills the
architecture's chat-template opening through the executor and writes the base
`.zpc`. The warm prefix is **known and recurring**. (Random/synthetic prompts are
a non-goal — they do not recur and are evicted before any HIT.)

### Coding target — a middle edit invalidates only the tail

Because the path is causal, a middle mutation keeps the maximal common prefix
(the stable head) and only the tail is new; committing the new tail reuses the
head. This is asserted by a `prefix_cache` test.

**SWA per-layer reuse is blocked (correctness).** The original design assumed a
sliding-window layer's K/V depends only on the local window. In gemma4 the
sliding/global pattern is `[1,1,1,1,0,...]`: the global layers at 4/9/14… mix the
full prefix into the residual stream, so a later sliding layer's K/V at position
`p` depends on every token `≤ p` through those layers. A middle mutation changes
those bytes → reusing them would be a **wrong HIT**. Only the leading all-sliding
run (layers 0–3 here) is provably local; the shared tail sources from layers
13/14. Not implemented (kept in `open-items.md`).

### Adjacent (phase 2)

- **Content-hash shared-weight detection.** The content identity is persisted at
  **compile**: ZTEF **format v2** stores each tensor's `sha256` (the writer already
  computed it for the content root). `tune` aliases byte-identical tensors to one
  tensor/range (`ContentKey.sha256` + type/dims/length; `applyAliases`), so the
  runtime reads and the residency prefetches one copy; reports `shared:`. On
  gemma4 the only duplicates are the shared `attn_q_norm`/`attn_k_norm` (48
  tensors, <0.1 MB). Memory/IO only — equal weights do **not** yield K/V reuse.
- **Prefill residency nodes.** Prefill dispatches carry residency nodes
  (`trigger_step = maxInt`), applied by `executePrefill` via `applyDispatch`, so
  the first prefill block pins/prefetches instead of faulting in.

### Parity

A HIT returns the identical K/V, so the greedy output equals the no-cache path
(round-trip: prefill → commit → reload → decode == decode).

## Steps

**Phase 1** (`216760e`):

1. `PrefixCacheKey` + canonical, `max_seq`-independent layout. [DONE]
2. Radix tree: lookup (maximal common prefix), commit, LRU, refcount. [DONE]
3. Session wiring: `GENERATE` lookup → resume or prefill+commit; `cache_matched`
   metric. [DONE]
4. `tune --prefix-cache` (default off) + the `.zpc` artifact + the runtime
   journal. [DONE]
5. Parity round-trip + chat e2e (warm hit and full hit; output unchanged). [DONE]

**Phase 2** (`b17b343` `9d6a4de` `9e9e53d` `ce40e7d`):

6. K/V byte-invariance across `--threads`. [DONE] A kernel-level test asserts the
   matvec row-split is **bit-identical** across `parts` (native +
   `q4_0_scale_split`); the e2e shares one `.zpc` across `--threads`.
7. Persistent `.zpc` (disk + mmap, commit generation, compaction, atomic flush).
   [DONE] `addMeta` restores the counter header; the service flushes a dirty
   cache on idle (`writeCacheFile`, temp + `rename`); a fresh daemon HITs a
   previous daemon's appends.
8. Tuner cache ceiling from `planBudget`. [DONE] A slice of the plan's window.
9. Canonical, order-independent layout fingerprint. [DONE] `KvBuffer`s ordered by
   first layer; the fingerprint drops the plan `slot`.
10. Coding e2e. [DONE] A middle edit keeps the maximal common prefix; only the
    tail is new.
11. Adjacent: content-hash shared-weight detection (ZTEF format v2) + prefill residency
    nodes. [DONE]
12. SWA per-layer reuse. [BLOCKED — correctness, see Design.]
13. `zig build` / `zig build test` green; e2e. [DONE] 188/188; `tune` + `run` on
    `e2b-test` coherent.

## Definition of Done

- [x] A repeated prefix HIT skips the prefill (TTFT of the shared part ~0) and the
      greedy output is identical to the no-cache path.
- [x] The radix tree matches the **maximal common prefix** (no fixed-grid loss).
- [x] The cache is bounded (LRU under the plan window), refcounted, evicts
      cleanly; mmap-only (payloads and index).
- [x] Plans of the same image that differ only in `--threads`/`--ram`/`--vram`
      **share one `.zpc`**; a device/kernel/shape change invalidates it.
- [x] The `.zpc` is **persistent** (survives a daemon restart) and disk/mmap-only;
      the flush is atomic.
- [x] The cache ceiling comes from the plan/`planBudget`, not `ram/8`.
- [x] The layout fingerprint is **order- and slot-independent**.
- [x] A middle edit invalidates only the tail (the radix). SWA per-layer reuse
      outside the window is **blocked** (interleaved global layers — Design).
- [x] `tune --prefix-cache` writes the `.zpc` and a later `run` HITs the warm
      prefix; default `tune` leaves no cache.
- [x] `zig build` / `zig build test` green; deterministic; the plan is unchanged.

## Arch Provenance

- **own:** the radix tree + per-plan cache fingerprint + canonical layout; the
  deterministic exact-path policy; the persistence/compaction design; the budget
  slice.
- **consulted:** vLLM (LRU-block prefix caching), SGLang RadixAttention, llama.cpp
  prompt cache, Prompt Cache / CacheBlend — reference only.
- **derived:** `runtime/03` (the intra-execution K/V buffer), `runtime/06`
  (residency/budget), `runtime/10` (batched prefill), `runtime/11` (prefill
  residency).

## Comments

- Distinct from the intra-execution K/V buffer (`runtime/03`); this is the
  **persistent, cross-execution** prefix cache.
- **"Default off" = no warm artifact.** The in-memory cache is always active for
  a plan with K/V, so multi-turn reuse works without a flag; `tune --prefix-cache`
  only (re)generates the `.zpc`.
- **Radix nodes are not split.** A divergent continuation is a child starting at
  the divergence offset; lookup matches the maximal common prefix without a copy.
- **mmap-only:** base `.zpc` zero-copy + a file-backed writable journal + the
  mapped index; nothing on the heap except the tiny region list. Nothing lives in
  anonymous RAM.
- **A HIT is exact** — no tolerance, no quality gate; the plan is unchanged.
- The `.zpc` is **the** durable artifact: the daemon's appends persist (subject
  to the ceiling + compaction), so a coding session's cache survives a restart.
- **SWA per-layer reuse is blocked (correctness)** — see Design; kept in
  `sdd/open-items.md`.
- Deferred: `--kv`/`--gate` (KV precision + quality gate); tolerant matching;
  cross-image sharing.

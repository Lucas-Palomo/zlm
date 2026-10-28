04 - Tokenizer and Generation

Status: done
Commits:

## Goal

Turn the decoded id into text: read the tokenizer from the image's `zmeta`,
encode the prompt to ids, run the autoregressive loop (one forward per token),
sample, and detokenize. Today `run` emits a single numeric token id for a fixed
BOS step and ignores `--prompt`/`--max-tokens`.

## Scope

- **In:** a `zmeta` tokenizer reader (vocab/merges/special ids); `encode`
  (text -> ids, with the arch chat template) and `decode` (ids -> text); the
  generation loop feeding each token through the forward; greedy sampling
  (argmax) first; wire `run --prompt/--max-tokens`.
- **Out:** tokenizer training; BPE/merge nuances beyond what gemma4 needs;
  sampling strategies (temp/top-k/top-p) as a later cut; threading; GPU.

## Design

- **Tokenizer lives in `zmeta` (metadata).** The tokenizer comes from GGUF keys
  (`tokenizer.ggml.tokens/merges/token_type`, `add_bos_token`, …), so it is
  metadata and already re-encoded into `zmeta` at `compile` (the reader keeps
  every KV). It stays there — `zcom` is only for *derived weight data* (norm,
  RoPE), which the tokenizer is not.
- **Chat template belongs to `architectures/`.** Keep a simple single-turn
  template (no "think") in `architectures/core` as the shared shape, with each
  `impl/<arch>` expanding its own; `run` must not build the template.
- **Two halves:** `Tokenizer` (encode/decode) and a generation driver
  (prefill + decode steps). Prefill and decode differ only in the number of
  positions; the plan dispatches already carry `pos`/`t_start` params.
- **Position and KV:** each step sets `pos` and the `t_start`/sliding window;
  `write_kv`/`attention` append to the KV cache. This exercises the shared-KV
  and sliding-window paths that are currently inert at `pos=0`.
- **Sampler:** greedy (argmax) first; the token id is decoded to text.
- **Prefill token-a-token:** the first cut processes the prompt one token per
  forward step (reusing the single-token path and advancing `pos`), not a batched
  prefill.
- **`--max-tokens` from the start:** the loop runs until `max_tokens` or EOS;
  tests stay tiny (8/16 tokens).
- **Edges stay edges:** `run` sends text and prints text; the runtime owns the
  tokenizer, template and loop (CONSTITUTION "an edge talks only to the
  runtime").

## Arch Provenance

- **own:** the tokenizer reader + generation loop shape.
- **spec:** GGUF/gemma tokenizer metadata (tokens/merges/specials), chat template.
- **consulted:** llama.cpp for BPE/tokenizer details — reference only.
- **derived:** from `zmeta` (written at compile) and the `runtime/03` Executor.

## Steps

1. `zmeta` tokenizer reader (vocab, merges, `bos`/`eos`/`unk`, `add_bos`).
   **As-built:** `src/runtime/tokenizer.zig` (`Tokenizer.fromZmeta` over
   `Zmeta.getArray`; vocab/merges/token_type/special ids + `add_bos`/
   `add_space_prefix`). Owns only the outer slices; the string bytes stay
   mmap-backed in `zmeta`.
2. `Tokenizer.encode`/`decode` + the gemma4 chat template (user turn wrap).
   **As-built:** SPM-BPE (`▁` space marker, greedy merge by rank, `<0xXX>` byte
   fallback) + registered/control special matching; `decode` (faithful) and
   `decodeText` (drops specials). Template shape in
   `architectures/core/chat.zig` (`Template.apply`), gemma4 strings in
   `architectures/impl/gemma4/chat.zig`; `Arch.chat` carries it and
   `architectures.root.applyChatTemplate` wraps a turn. Verified against
   `llama-tokenize` on the E2B GGUF (exact id match for the chat prompt).
3. Generation loop: prefill prompt ids, then one forward per step with `pos`.
   **As-built:** `Session.generate` formats + encodes + prefills token-a-token,
   `Session.token_id` selects the embedding row and `Session.stepAt(pos)`
   (`executor.executeAt`) overrides `params[0]` for rope/write_kv/attention.
4. Greedy sampler; wire `run --prompt/--max-tokens`; print text.
   **As-built:** argmax already in `executeAt`; `run` calls `session.generate`
   and prints the detokenized text. `--max-tokens` honored, stops at EOS.
5. Correctness: a prompt produces coherent continuation (needs `plan/04` for
   exact per-layer dims; without it, output may be degraded).
   **As-built:** `zig build` / `zig build test` green; e2b run prints text
   (degraded — see Comments).
6. `zig build` / `zig build test` green.
   **As-built:** both green.

## Definition of Done

- [x] `zinfo`/`zmeta` tokenizer reader round-trips (encode -> decode).
- [x] `run --prompt <text>` generates and prints text (not a numeric id).
- [x] `pos` advances per step; KV/shared-KV/sliding paths are exercised.
- [x] `zig build` / `zig build test` green.

## Comments

- Depends on `plan/04` for exact logits (per-layer dims). `plan/04` lands first;
  this task reads `in_len`/`out_len` and falls back to whole buffers when 0.
  **As-built:** the executor already slices by `in_len`/`out_len` (0 = whole).
- ~~**Open item:** sampling strategies beyond greedy (temperature, top-k, top-p)
  are deferred — recorded in `sdd/open-items.md` so they are not forgotten.~~
  Resolved: `run --temperature/--top-k/--top-p` + `runtime/sampler.zig`; greedy
  (`--temperature 0`) stays the default and is the exact argmax.
- ~~**Known-incorrect (output degraded):** sliding-window `t_start` is still the
  tuned 0 (the plan does not carry the window), so sliding layers attend the
  full cache; shared-KV layers still `write_kv` their source layer's cache. Both
  are pre-existing and explain the garbled e2b text; the loop itself is
  correct.~~ Picked up by `architectures/04` (gemma4 forward correctness).
- ~~**Open item:** `<end_of_turn>` (and other control/special pieces a GGUF may
  mislabel `NORMAL`) leaks into the displayed text — recorded in
  `sdd/open-items.md`.~~ Resolved: `decodeText` now also drops known control
  markers by piece, alongside the control/special-type and registered-id checks;
  encoding and the generation token stream are unchanged. Verified by a focused
  tokenizer test.

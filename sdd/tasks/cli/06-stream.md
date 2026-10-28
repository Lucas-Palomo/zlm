06 - Streaming Generation (run --stream)

Status: done
Commits:

## Goal

Let `run` print tokens **as they are generated** instead of waiting for the whole
decode. Today the daemon accumulates the full text and sends it in one reply, so
nothing appears until the last token. Add `run --stream` (flag, default off): the
daemon emits the decoded text in incremental chunks over the socket while
generating, and the client writes them live. The first token appears right after
the prefill (TTFT), and the rest stream in.

## Scope

- **In:** a `run --stream` flag; a `stream` field on the `GENERATE` request; a
  streaming reply (`DATA` frames plus the final metrics header); a streaming
  decode path in `Session` (per-token detokenize plus UTF-8-safe chunking plus
  special/control filtering); the client sink that prints live.
- **Out:** changing the plan/KV/prefix cache (streaming is presentation only); a
  non-blocking/async executor; a browser/SSE/HTTP streaming surface (`http/`); new
  sampling behavior.

## Design

### Protocol

Extend the canonical request to six fixed fields (the prompt stays the trailing
field, tab-safe):

```text
GENERATE  max_tokens  temperature  top_k  top_p  think  stream  prompt
```

Replies:

- `stream = 0` (default, unchanged): `OK  <text_len>  <metrics...>` then the text
  body.
- `stream = 1`: a sequence of `DATA  <chunk_len>` frames (each followed by the
  chunk bytes), then the normal metrics header `OK  <total_len>  <metrics...>`
  with **no body**. The client prints each `DATA` chunk immediately; on `OK` it
  stops, reads the metrics, and (if asked) prints them.

A stale daemon (no streaming support) is already handled by the build-fingerprint
handshake (it is restarted), and an old client sends `stream = 0`.

### Runtime - streaming decode

`Session` gains a streaming path that mirrors `generate`'s setup (format prompt,
encode, prefix-cache lookup, prefill, commit) but, in the decode loop, emits each
generated token's bytes to a sink instead of accumulating the whole text:

- **Detokenize per token.** Reuse the tokenizer's piece-to-text mapping and the
  same filtering the batch path applies (`isSpecialType` / `isRegistered` /
  `isControlMarker`), so specials/controls never reach the screen.
- **UTF-8-safe buffering.** A multi-byte character can span two tokens
  (byte-level BPE); accumulate emitted bytes and flush only a valid UTF-8 prefix,
  holding back an incomplete trailing sequence until the next token (flush the
  remainder at the end).
- **Stats.** The same `GenerateStats` (prefill/decode ms, token counts, profile,
  gen_ms); streaming does not change the forward or the accounting.

The sink is a small `{ ctx, write(bytes) }` callback (same shape as the tuner's
`Builder`), so `Session` stays decoupled from the socket.

### Edge

`run --stream` uses `service.generateStream(...)` with a sink that writes to
stdout; it prints text as it arrives and, when the final `OK` carries metrics,
optionally prints them (`--metrics`). Without `--stream`, the existing
`service.generate` path is unchanged.

## Steps

1. Spec plus `run --stream` flag plus help. **Done.**
2. `service.zig`: `stream` field on `GENERATE` (six fields; legacy default 0);
   `handleClient` streaming branch (`DATA` frames plus final `OK`); `generateStream`
   client. **Done.**
3. `session.zig`: streaming generation (per-token detokenize, UTF-8-safe
   chunking, special filtering) via a sink. **Done.**
4. `run.zig`: wire `--stream`; print live; metrics after. **Done.**
5. Tests: protocol parse with `stream`; streaming round-trip; UTF-8-boundary
   chunking; e2e `run --stream`. **Done** (246/246; stdout of stream == batch).
6. `zig build` / `zig build test` green. **Done.**

## Flow / Arch

```text
run --stream -> GENERATE ... 1 <prompt>
                 one Session streaming decode, per token: detokenize (specials
                 filtered), UTF-8 boundary -> DATA frame -> client prints
                 end -> OK <metrics...>
```

## Definition of Done

- [x] `run --stream` prints tokens live (the first well before the last).
- [x] The streamed text equals the batch text byte-for-byte (specials filtered).
- [x] No partial UTF-8 ever reaches the terminal.
- [x] `stream = 0` (default) is byte-identical to today; plan/KV/prefix cache
      unchanged; `--think` streams the reasoning turn too. (Note: `--show-think`
      to gate the reasoning block is `cli/07`.)
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the streaming protocol framing, the UTF-8-safe chunker, the session
  sink.
- **derived:** `cli/04` (metrics/TTFT), `runtime/04` (tokenizer/generation),
  `runtime/13` (prefix cache path), `architectures/09` (think).

## Comments

- **Presentation only.** Streaming does not touch the AOT plan, KV, or the prefix
  cache; batch stays the default so scripts/metrics stay reproducible.
- **Specials.** EOS/control markers are dropped exactly as `decodeText` does; only
  the timing of the bytes changes.
- **Sink shape** mirrors the tuner's `Builder` callback so `Session` has no socket
  dependency.

08 - gemma4 Chat Template (resolved at compile, stored in zmeta)

Status: done
Commits: 1037e2b

## Goal

The chat template is hardcoded in `architectures/impl/gemma4/chat.zig`
(`<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\n`), but gemma-4
variants differ: the unsloth E2B/E4B GGUF ships a `tokenizer.chat_template` that
uses `<|turn>`/`<turn|>` (and `<|...|>` markers), not `<start_of_turn>`. zlm
ignores the model metadata and hardcodes the format.

Resolve the template **at compile** from the model and store it in the image
(`zmeta`); the runtime reads it. The hardcoded `chat.zig` becomes only the
generic fallback.

## Scope

- **In:** derive `user_prefix`/`user_suffix`/`model_prefix` at `compile` from the
  model metadata; store them in `zmeta`; the runtime reads them and builds the
  prompt; the arch template stays as the fallback.
- **Out:** a full Jinja engine (the model template is complex); new CLI flags.

## Design

- **Source of truth = the image.** At `compile`, derive the three markers from
  the model's `tokenizer.chat_template` metadata (preferred) with a fallback that
  detects the format from the vocab special tokens:
  - `<|turn>` + `<turn|>` present → `"<|turn>user\n"`, `"<turn|>\n"`, `"<|turn>model\n"`.
  - else `<start_of_turn>` + `<end_of_turn>` present → the classic markers.
  - else omit (the runtime uses the arch fallback).
- **Store in `zmeta`** as `zlm.chat.user_prefix` / `zlm.chat.user_suffix` /
  `zlm.chat.model_prefix` (strings), appended after the GGUF metadata keys.
- **Runtime:** `Session.formatPrompt` reads the three keys from `self.zmeta`; if
  all present, apply `Chat.Template`; else `applyChatTemplate` (the arch's
  hardcoded template). Deterministic; the runtime decides nothing (the image
  carries the decision).
- Keep the layer boundaries: `image/` writes raw strings (no architectures
  import); the format knowledge may live in the arch layer, invoked by the
  compile.

## Steps

1. Add the derivation (from `tokenizer.chat_template` / the vocab) and store the
   keys in `zmeta`. [DONE]
2. Read the keys in `Session.formatPrompt`, falling back to the arch template.
   [DONE]
3. Tests: the derivation for both formats; the `zmeta` round-trip; prompt
   building from stored vs fallback markers. [DONE]
4. Verify e2e: E2B/E4B `run` coherent; `zig build`/`test` green. [DONE]

## Definition of Done

- [x] The template is resolved at compile and stored in `zmeta`.
- [x] The runtime uses the stored template; the arch template is the fallback.
- [x] `zig build` / `zig build test` green; E2B/E4B coherent.

## Arch Provenance

- **own:** the compile-time resolution + zmeta storage.
- **spec:** GGUF `tokenizer.chat_template` metadata.
- **consulted:** `llama.cpp` chat templating (reference only).
- **derived:** `architectures/04` (gemma4 forward correctness), `engine/02` (zcom).

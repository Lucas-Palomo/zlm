# 17 - OpenAI Tool Calling (Gemma 4 native dialect)

Status: done
Commits: none (uncommitted)

## Goal

OpenAI-compatible clients (OpenCode/AI SDK uses the `openai-compatible`
provider) could not use tools: the HTTP edge ignored `tools`, rendered no tool
declarations, and returned the model's raw call text as assistant content. The
goal is an end-to-end tool round — client declarations → prompt → model call →
OpenAI `tool_calls` → client executes → `role: "tool"` results → model answer —
using Gemma 4's native `<|tool_call>` dialect, with `think: true` supported on
every round.

## Scope

In:

- Gemma 4 tool declarations (`<|tool>…<tool|>`), calls (`<|tool_call>…`) and
  responses (`<|tool_response>…`), rendered/parsed from the model's chat
  template.
- OpenAI request fields `tools` and assistant `tool_calls`; `role: "tool"`
  results resolved by `tool_call_id`.
- OpenAI response `message.tool_calls` + `finish_reason: "tool_calls"`, both
  JSON and SSE (`delta.tool_calls`).
- `max_completion_tokens` as an alias of `max_tokens` (OpenCode sends the
  former).
- One system turn carrying `<|think|>`, the soul and the tool declarations.
- A fresh reasoning round after tool responses (`<|channel>thought\n`).

Out:

- Qwen 3's tool dialect (`Arch.tools` is null there; a request with tools errors
  with `UnsupportedArchitecture`).
- Persisting tool calls/results in `$ZLM_HOME/sessions` (the session record is
  still role+text only).
- True incremental streaming of tool-call arguments: a tool-enabled request is
  generated first and streamed as content/tool-call frames after parsing.

## Design

`architectures/core/tools.zig` defines the shared structured vocabulary
(`Tool`, `Call`, `Result`, `Parsed`, `Dialect`). The dialect is a set of
function pointers declared by the architecture (`Arch.tools`), so the runtime
and the edge never know the literal syntax. `architectures/impl/gemma4/tools.zig`
implements it from the model's chat template:

```text
<|tool>declaration:name{description:<|"|>…<|"|>,parameters:{…}}<tool|>
<|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>
<|tool_response>response:name{value:<|"|>…<|"|>}<tool_response|>
```

`Session.Turn` gained `role: tool`, `calls`, `tool_call_id` and `name`.
`Session.formatConversationWithTools` renders one system turn
(`<|turn>system\n<|think|>\nsoul<|tool>…<turn|>`), assistant calls before their
text, tool responses inside the open model turn, and opens
`<|channel>thought\n` after a tool response when thinking is ON. The context
budget truncation now drops the oldest user turn and its assistant/tool
replies.

Generation decodes with `Tokenizer.decodeToolText`, which keeps the
`<|tool_call>`, `<|tool_call|>` and `<|"|>` pieces (the normal display decoder
drops special tokens) while still suppressing the think channel. The prompt's
trailing `<|channel>` seeds the reasoning state, so the post-tool round is
suppressed like the first one. `max_think` is per reasoning round.

The HTTP edge maps the OpenAI JSON to turns, assigns `call_<n>` ids to the
model's calls, and answers with `tool_calls` (content is null when a call is
present: small models often hallucinate a pre-call result). SSE tool calls are
sent as full-argument deltas followed by a `finish_reason: "tool_calls"` frame
and `[DONE]`.

## Steps

1. Fixed the Gemma 4 system turn: `<|think|>` now lives at the top of the same
   system turn as the soul/tools instead of a second system turn. Think mode
   now actually engages.
2. Added `core/tools.zig` + `gemma4/tools.zig` (schema DSL, call/response
   rendering, call parsing with DSL or JSON arguments) and the `Arch.tools`
   declaration. Tests in `gemma4/tools_test.zig`.
3. Extended `Session.Turn` and added `formatConversationWithTools` /
   `extractToolCalls`; kept `formatConversation` as the no-tools wrapper.
4. Extended the tokenizer (`decodeToolText`, tool marker ids) and seeded the
   think state from the prompt for tool-response rounds.
5. Mapped the OpenAI fields in `http/server.zig` (tools, tool_calls, tool
   results, JSON + SSE), added `max_completion_tokens`, and dropped pre-call
   content when the model calls a tool.
6. Verified with OpenCode: `shell` tool call → real `pwd` output → final answer,
   with `think: true`.

## Definition of Done

- [x] Gemma 4 declarations/calls/responses match the model's chat template.
- [x] `tools` in the request influence the prompt; no tools → unchanged prompt.
- [x] Non-stream and SSE responses expose `tool_calls` and
      `finish_reason: "tool_calls"`.
- [x] `role: "tool"` results resolve their name through `tool_call_id` and the
      model answers after the round.
- [x] `think: true` round after a tool response starts in the think channel and
      is not leaked into `message.content`.
- [x] `zig build` and `zig build test` green (298 tests).
- [x] End-to-end with OpenCode against the real server.

## Arch Provenance

- own: structured `Tool`/`Call`/`Dialect`, runtime/edge split, SSE framing,
  pre-call content policy.
- spec: Gemma 4 chat template from the model's `tokenizer_config.json`
  (`<|tool>`, `<|tool_call>`, `<|tool_response>`, `<|think|>`, `<|channel>`).
- consulted: OpenAI Chat Completions tool-call shape (`tools`, `tool_calls`,
  `tool_call_id`, `finish_reason`); OpenCode/AI SDK provider behavior.
- derived: the Gemma DSL renderer/parser from the template above.

## Comments

- Tool calls and results are not persisted in `$ZLM_HOME/sessions`; the record
  remains role+text, so a `session`-field conversation cannot replay a tool
  round. See `sdd/open-items.md`.
- A tool-enabled SSE request is buffered until the reply is generated (the raw
  `<|tool_call>` block must not leak as content), so token-level streaming is
  lost on that path.
- Qwen 3 has no tool dialect yet; `tools` with a qwen3 image errors explicitly.

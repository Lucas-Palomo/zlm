# Using zlm from OpenCode

`zlm` serves an OpenAI-compatible HTTP API. OpenCode can use it as a provider
through `@ai-sdk/openai-compatible` — chat, streaming, and tool calling all work
for the Gemma 4 images. This is the known-good configuration and the traps that
cost the most time.

The examples use two tuned aliases:

| OpenAI id | zlm runtime | think | purpose |
| --- | --- | --- | --- |
| `gemma-12b` | `gemma-12b` | on | quality/reasoning |
| `gemma-e4b:fast` | `gemma-e4b` | off | fast chat |

Replace them with the aliases from `zlm list`; the OpenAI `id` is what the client
sends, the `runtime` is the tuned alias. They may differ (`gemma-e4b:fast` uses
the `gemma-e4b` runtime). One tuned runtime can back several OpenAI `id`s with
different `think`/limits; tune the runtime once.

## 1. Prerequisites

Build the **native release** binary. `zig build` alone is Debug and is roughly
6× slower on decode:

```bash
zig build -Doptimize=ReleaseFast
zlm list                       # aliases: ALIAS / IMAGE / FP
```

Compile the image and tune each alias once (`image compile`, then `tune …`); the
HTTP server loads the plan lazily on the first request. The store root defaults
to `$ZLM_HOME`, else `$HOME/.zlm`; from the repo, `./zlm.sh` sets it to
`<repo>/.zlm`.

The server must bind IPv4. Start it in its own terminal (the HTTP edge is
serial; while a generation runs it does not answer other requests):

```bash
./zlm.sh http serve 3001 --config config.json
```

Expected:

```text
http: 2 model(s) on http://127.0.0.1:3001/v1 (Ctrl-C to stop)
```

## 2. Server config: `config.json`

`zlm http config` writes a bootstrap with one entry per alias and default
generation controls. A two-model setup that matches the OpenCode example:

```json
{
  "version": 1,
  "host": "127.0.0.1",
  "default": "gemma-e4b:fast",
  "models": [
    {
      "id": "gemma-12b",
      "runtime": "gemma-12b",
      "soul": "You are a helpful assistant.",
      "temperature": 1,
      "top_k": 64,
      "top_p": 0.95,
      "think": true,
      "max_tokens": 256,
      "max_think": 256
    },
    {
      "id": "gemma-e4b:fast",
      "runtime": "gemma-e4b",
      "soul": "You are a helpful assistant.",
      "temperature": 1,
      "top_k": 64,
      "top_p": 0.95,
      "think": false,
      "max_tokens": 128,
      "max_think": 128
    }
  ]
}
```

Rules that matter:

- `default` must be a real `id`. If it is not, requests that omit `model` (or
  name an unknown one) return `400 {"error":{"message":"unknown model"}}`.
- `think` is a property of the **model entry**, not of the request. OpenCode's
  OpenAI-compatible client does not send a `think` flag, so a “fast” id only
  exists if the server declares one with `"think": false`.
- `max_tokens` bounds the answer and `max_think` the reasoning; they are
  separate. `max_think` applies per reasoning round, so a tool round gets a
  fresh budget. If the model does not close the thought channel, the runtime
  forces `<channel|>` at the limit and still gives the answer its `max_tokens`.
- `max_completion_tokens` is accepted as an alias of `max_tokens`; OpenCode
  sends it from the model's `limit.output`.

The server reads `config.json` at boot; restart `zlm http serve` after changing
ids, `think`, or generation limits.

## 3. OpenCode config: `~/.config/opencode/opencode.jsonc`

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "zlm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "zlm",
      "options": {
        "baseURL": "http://127.0.0.1:3001/v1"
      },
      "models": {
        "gemma-12b": {
          "name": "Gemma4 12B",
          "family": "gemma4",
          "limit": { "context": 48854, "output": 256 }
        },
        "gemma-e4b:fast": {
          "name": "Gemma4 E4B - Fast",
          "family": "gemma4",
          "limit": { "context": 131072, "output": 128 }
        }
      }
    }
  },
  "agent": {
    "chat": {
      "description": "Chat without tools (live stream)",
      "mode": "primary",
      "prompt": "You are a helpful assistant. Answer concisely.",
      "tools": { "*": false }
    },
    "dev": {
      "description": "Coding with shell/read",
      "mode": "primary",
      "prompt": "You are a coding assistant. Use the tools when needed.",
      "tools": { "*": false, "shell": true, "read": true, "write": true, "edit": true }
    }
  }
}
```

### Provider rules

- **`baseURL` must use `127.0.0.1`**, not `localhost`. `localhost` usually
  resolves to `::1` first; zlm binds `127.0.0.1`, so the client can get
  `ECONNREFUSED` even with the server up. Use `0.0.0.0` (and a firewall) only if
  you intentionally serve the LAN.
- The model keys are the OpenAI `model` ids and must **exactly** match the
  server's `id`s. A mismatch gives `Model unavailable: zlm/<id>` before any HTTP
  request.
- `limit` requires **both** `context` and `output`. A `limit` with only
  `output` makes OpenCode skip the provider (`$.provider.zlm kind=invalid`).
- `limit.output` should match the server entry's `max_tokens`. Without it,
  OpenCode can send `max_completion_tokens: 32000`, which on CPU means a very
  long generation.
- `limit.context` is the plan's real context. Read it from `inspect` — the
  tuner can degrade the context to fit `--ram`:

  ```bash
  zlm inspect gemma-e4b --plan | grep context
  ```

  On the example machine this is `131072` for the E4B and `48854` for the 12B
  (degraded by a 16 GB budget). Re-run `tune` with a different budget and update
  this value.

### Agent rules

- `agent` is a **top-level** key, a sibling of `provider`. Do **not** put it
  inside `provider.zlm`: `ProviderConfig` has `additionalProperties: false`, so
  the provider is considered malformed and skipped.
- One agent is one **behavior** (prompt + tools), not one model. `agent.model`
  is a single optional default; the same agent can run with any configured model
  via `-m`/the TUI model selector. The `-m` choice wins over `agent.model`.
- No tools → the server uses the per-token SSE path. Tools → the reply is
  generated first and tool-enabled SSE arrives at the end (the raw
  `<|tool_call>` block must not leak as content). The `chat` agent is the fast,
  live-streaming path; `dev` trades that for tool use.
- `tools` is marked deprecated in the OpenCode schema, but `{"*": false}` plus
  specific names is the practical way to disable/enable tools in v2.0.26.
- A custom agent prompt replaces the base system prompt but OpenCode still
  injects the applicable `AGENTS.md` (if any; its size depends on the project).
  Run chat from a directory without a large `AGENTS.md` if you want to shave
  that prefill as well.

OpenCode reads `opencode.jsonc` at boot; restart the TUI/CLI after editing it.

## 4. Running

TUI: select the agent (`chat`/`dev`) and the model (`gemma-e4b:fast` or
`gemma-12b`).

CLI:

```bash
opencode run --agent chat -m zlm/gemma-e4b:fast "hello"
opencode run --agent chat -m zlm/gemma-12b "explain this error"
opencode run --agent dev  -m zlm/gemma-e4b:fast "list the files"
```

`--agent`/`-m` are per run; the agent's `model` (when set) is only the default.

## 5. Verify

```bash
# the served ids (must match the OpenCode model keys)
curl -s http://127.0.0.1:3001/v1/models

# a direct, streaming completion (no tools: per-token SSE)
curl -N http://127.0.0.1:3001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-e4b:fast","messages":[{"role":"user","content":"count to 5"}],"stream":true,"max_tokens":32}'
```

An SSE reply ends with a `finish_reason` chunk followed by `data: [DONE]`; OpenAI
clients (the AI SDK used by OpenCode) reject a stream that ends without
`finish_reason`.

## 6. Troubleshooting

| symptom | likely cause | fix |
| --- | --- | --- |
| `ECONNREFUSED` | server down, wrong port, or `localhost` resolving to `::1` | start on `3001`, use `127.0.0.1` in `baseURL` |
| `Model unavailable: zlm/X` | `X` is not a key in `provider.zlm.models` | add/rename the model key to the server `id` |
| `unknown model` (HTTP 400) | `config.json` `default` is not a served `id` | set `default` to a valid id |
| `OpenAI Chat stream ended without finish_reason` | old zlm without the terminal SSE frame | update zlm; every stream must end with `finish_reason` |
| `configuration normalization … path=$.provider.zlm` | trailing comma, `limit` without `context`, or `agent` nested in the provider | move `agent` to the top level; add both limit fields; remove the comma |
| no tokens for minutes | cold prefill of the agent prompt, or a tool-enabled request (buffered), or a Debug build | use ReleaseFast, the `chat` agent (no tools), and let the prefix cache warm |
| `/v1/models` times out | the HTTP edge is serial and is busy generating | wait; a second request (e.g. the title turn) queues behind the first |

The first request after a server start is the expensive one: the whole system
prompt (agent + `AGENTS.md` + tools) must be prefilled. Later turns reuse the
prefix cache (`.zpc`) while the prompt prefix stays the same. A server restart,
a changed agent/tools/system prompt, or a different model entry is a cold start.

## 7. Known limitations

- The HTTP edge handles one request at a time. A long generation blocks
  `/v1/models` and OpenCode's title request; an abandoned client generation keeps
  running until its budget/EOS. Restart the server to clear it.
- Tool-enabled SSE is buffered (see above); the no-tools path streams per token.
- Tool calling is implemented for the Gemma 4 dialect. A Qwen 3 image returns
  `UnsupportedArchitecture` when `tools` are present.
- Sessions persisted with the HTTP `session` field store role+text only, so a
  tool round cannot be replayed from them.

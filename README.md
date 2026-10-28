# zlm

**You decide how your language model runs.**

Anti-OOM by design, RAM/VRAM-budgeted, CPU-first, mmap-first, tuned to your
hardware. The model is **mapped, not preloaded**; every runtime keeps memory
within the budget you set, so a model larger than your RAM still answers.

zlm compiles a GGUF into an immutable **image**, tunes a per-machine **plan**
(threads, RAM, VRAM, residency) and serves the model through a background
service: a one-shot `run`, an interactive `chat`, or an OpenAI-compatible
`http` API.

## Features

- **Budgeted memory** — the plan caps RAM/VRAM; the runtime never exceeds it.
- **mmap-first** — weights stream from the image on demand; the page cache is
  the cache.
- **CPU-first, GPU optional** — a Vulkan co-processor takes the heavy GEMV/GEMM
  ops when the tuned plan says it pays.
- **Tuned per machine** — one deterministic plan per (image, hardware), chosen
  by measured costs, not heuristics.
- **Persistent service** — the session stays loaded; `run`/`chat` are thin
  clients over one Unix socket. `http` keeps its sessions in-process.
- **Chat + OpenAI-compatible HTTP** — a REPL and an API (including Gemma 4 tool
  calls) that external tools can use as a backend.

## Build

Requires **Zig 0.16**.

| Platform | Status |
| --- | --- |
| Linux x86_64 | Supported; built and tested in CI |
| Linux other architectures | Build from source; not exercised in CI |
| macOS | **Experimental**; the base is POSIX and a local build may work, but it is not tested in CI and no prebuilt binary is provided |
| Windows | Out of scope (use WSL or Docker) |

CI runs on Linux only. Tagged releases (`v*`) publish a static Linux x86_64
(musl, `x86_64-v2`) archive from `.github/workflows/release.yml`.

```bash
zig build                        # debug
zig build -Doptimize=ReleaseFast # production (native CPU by default)
zig build test                   # unit + integration tests
```

The default target uses the native CPU. Pass `-Dcpu=<model>` when you need a
different CPU baseline (for example, the static Docker build in
[docs/docker.md](docs/docker.md), which also sets `-Dtarget`).

The store root is `$ZLM_HOME`, else `$HOME/.zlm`. For development,
`./zlm.sh <command>` runs `zig-out/bin/zlm` with `ZLM_HOME=<repo>/.zlm`
(`zig build run -- <command>` does the same). Install the binary by copying
`zig-out/bin/zlm` onto your `$PATH`.

For a Docker workflow (static binary, constrained container, benchmark
isolation), see [docs/docker.md](docs/docker.md).

## Quick start

Get a GGUF first: zlm does not ship weights. See
[docs/models.md](docs/models.md) for supported models, example files and
download pointers, then put the file under `models/` (ignored by Git).

```bash
# 0. Download/place a GGUF under models/, for example:
#    models/gemma-4-E2B-it-Q4_0.gguf

# 1. Compile a GGUF into an image (deterministic, hardware-independent).
zlm image compile gemma4-e2b models/gemma-4-E2B-it-Q4_0.gguf

# 2. Tune a runtime alias for this machine.
#    CPU:  --threads N --ram 8gb
#    GPU:  add --gpu --vram 4gb
zlm tune gemma4-e2b e2b --threads 8 --ram 8gb

# 3. One-shot generation.
zlm run e2b --soul "You are a concise assistant." --prompt "What is a monad?"

# 4. Interactive chat (the reply streams; /help lists the commands).
zlm chat e2b --soul "You are a concise assistant."

# 5. OpenAI-compatible HTTP API.
zlm http config --output http.json      # bootstrap from the tuned aliases
zlm http serve 8080 --config=http.json
```

## Supported models

zlm currently implements two GGUF architectures:

| GGUF `general.architecture` | Family | Chat | Think | Tool calls |
| --- | --- | --- | --- | --- |
| `gemma4` | Gemma 4 (E2B/E4B/12B/31B) | Gemma turn template | yes | yes |
| `qwen3` | Qwen 3 dense | ChatML | yes | no |

Example GGUF files, quantization support and download pointers are in
[docs/models.md](docs/models.md). Qwen 3 Mixture-of-Experts variants and other
architectures are not supported.

## Documentation

- [Supported models](docs/models.md) — Gemma 4 and Qwen 3, quantization support,
  example GGUFs and download pointers.
- [Architecture and code provenance](docs/arch-provenance.md) — own/spec/
  consulted/derived/port map for architectures, kernels and GPU.
- [Docker workflow](docs/docker.md) — static build, constrained containers,
  HTTP inside Docker.
- [OpenCode integration](docs/opencode.md) — provider/agent configuration and
  common pitfalls.
- [Kernel architecture](docs/kernel-architecture.md)
- [CPU benchmarks](docs/benchmark-cpu.md)

## Commands

| command | purpose |
| --- | --- |
| `zlm image compile <name> <model.gguf>` | GGUF → image |
| `zlm image verify <name> [model.gguf]` | verify the image (vs the source GGUF) |
| `zlm image list` / `info` / `remove` | manage images |
| `zlm tune <image> <alias> --threads N --ram <mb\|gb> [--gpu --vram <mb\|gb>]` | write a plan |
| `zlm list` | list the tuned aliases (`ALIAS`, `IMAGE`, `FP`) |
| `zlm inspect <alias> [--plan\|--image\|--profile\|--all]` | inspect a plan |
| `zlm load <alias> [--cold\|--warm\|--hot[=min]] [--timeout=min]` | load the service |
| `zlm run <alias> --soul … --prompt …` | one-shot generation |
| `zlm chat <alias> --soul …` | interactive REPL |
| `zlm http serve <port> --config=<file>` | OpenAI-compatible HTTP API |
| `zlm http config [--output <file>]` | write a bootstrap config |
| `zlm unload <alias>` / `--remove` / `--prune` / `--all` | unload / prune |

`--ram`/`--vram` accept `mb` or `gb`. `tune` also takes `--residency <mb|gb>`
with `--prefetch-threads M` (the resident weight ring) and `--prefix-cache`
(warm the chat template into `.zpc`).

## `run`

```bash
zlm run <alias> --soul <text|file.txt> --prompt <text>
       [--session <uuid>]
       [--max-tokens N] [--max-think N]
       [--temperature F] [--top-k N] [--top-p F] [--no-stream]
       [--metrics[=full]] [--verbose[=min|info|all]]
       [--cold | --warm | --hot[=minutes]] [--timeout=minutes]
```

- `--soul` (**required**) is the system turn: a literal string, `""` for none,
  or a path to an existing file.
- **Greedy by default** (`--temperature 0` is argmax); `--max-tokens` bounds the
  answer and `--max-think` the reasoning, as **separate** budgets (total
  `N + M`); `--max-think` applies per reasoning round, so a tool round gets a
  fresh one. `--think` enables the model's reasoning mode.
- Streaming is **on by default** (`--no-stream` waits for the whole decode);
  `--metrics` prints TTFT/prefill/decode and peak RSS; `--verbose` traces to
  stderr (stdout unchanged).
- `--session <uuid>` appends this prompt + reply to a persistent session
  (a fresh one if the id is new).
- The state/TTL flags apply to the auto-load: `warm` by default, 30 min idle
  timeout; `--timeout=0` holds until `zlm unload`.

## `chat` (REPL)

```bash
zlm chat <alias> [<uuid>] --soul <text|file.txt>
       [--max-tokens N] [--max-think N]
       [--temperature F] [--top-k N] [--top-p F]
       [--think] [--show-think] [--verbose[=min|info|all]] [--metrics[=min|all]]
       [--cold | --warm | --hot[=minutes]] [--timeout=minutes]
```

Each line is a user turn; the reply streams to stdout. The conversation is
**persisted** in `$ZLM_HOME/sessions/<uuid>.json`: an explicit `<uuid>` resumes
that session, none starts a new one. The prefix cache makes each turn cheap. Slash
commands (not sent to the model):

| command | effect |
| --- | --- |
| `/help` | list the commands |
| `/exit` | leave (Ctrl-D too) |
| `/clear` | forget the conversation |
| `/soul <text\|file.txt>` | replace the soul for later turns |
| `/enable think` · `/disable think` | toggle the reasoning |
| `/enable show-think` · `/disable show-think` | show/hide the reasoning |
| `/verbose <off\|min\|info\|all>` | diagnostic level |
| `/metrics <off\|min\|all>` | per-turn metrics level |
| `/temperature <F>` | sampling temperature (0 = greedy) |
| `/top-k <N>` · `/top-p <F>` | top-k / nucleus |
| `/max-tokens <N>` · `/max-think <N>` | answer / reasoning budgets |
| `/rename <name>` | rename the current session |
| `/sessions` | list sessions and switch to one (type its number) |

## `http` (OpenAI-compatible)

The server is driven by a declarative config:

```json
{
  "version": 1,
  "host": "127.0.0.1",
  "default": "e2b",
  "models": [
    {
      "id": "e2b",
      "runtime": "e2b",
      "soul": "You are a concise assistant.",
      "temperature": 0.7,
      "top_k": 40,
      "top_p": 0.9,
      "think": false,
      "max_tokens": 256,
      "max_think": 256
    }
  ]
}
```

`zlm http config --output http.json` writes a bootstrap with one entry per tuned
alias; `zlm http serve 8080 --config=http.json` serves it. Models load **lazily**
on the first request that names them; a request `model` selects the entry and
its `system` message overrides the entry's soul.

Endpoints:

- `GET /v1/models` — the configured ids.
- `GET /v1/sessions` — the persisted sessions (id, name, model, turns).
- `POST /v1/chat/completions` — OpenAI JSON, or **SSE** when `stream: true`
  (`data: {…}` frames followed by `data: [DONE]`). An extra `session` field
  persists the conversation under that id (list it via `/v1/sessions`).

Per-request `temperature`, `top_p`, `top_k`, `max_tokens` and `think` override
the config. `max_completion_tokens` is accepted as an alias of `max_tokens`.

The server also speaks OpenAI **tool calling**: it accepts `tools` (function
declarations), renders the model's native Gemma 4 dialect in the system turn,
and answers with `tool_calls` + `finish_reason: "tool_calls"` (JSON or SSE
`delta.tool_calls`). Send the result back as a `role: "tool"` message with its
`tool_call_id`; the model continues in the same turn. With `think: true` every
tool round starts a fresh reasoning round.

```bash
curl http://127.0.0.1:8080/v1/models
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"e2b","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

See [`docs/opencode.md`](docs/opencode.md) for a complete OpenCode
provider/agent configuration (models, limits, agents, streaming) and the common
pitfalls.

## Concepts

- **image** — the compiled model (`model.zman`, `model.zmeta`, `*.ztef`). Immutable
  and hardware-independent.
- **plan** (`.zplan`) — the tuned configuration for one (image, hardware): the
  dispatch program, the buffers, the residency graph and the device calibration.
  The runtime **decides nothing**; it executes the plan.
- **alias** — a name for a plan (`zlm tune <image> <alias>`).
- **service** — a detached, per-plan process holding the session and serving
  generation over a Unix socket. `run`/`chat` connect to it; `http` opens its
  sessions in-process.
- **state** — `cold` (mapped, nothing resident), `warm` (default) and `hot`
  (resident window), with a TTL.
- **residency** — the plan's resident weight window (and optional ring +
  prefetch workers); the rest streams from the mmap.
- **prefix cache** (`.zpc`) — cross-request K/V reuse for the common prefix,
  which is what makes a chat turn cheap.
- **session** — a persisted conversation (`$ZLM_HOME/sessions/<uuid>.json`) with
  its turns, a display name and the model; `/sessions`, `/rename` and the HTTP
  `session` field manage it.

## Layout

```text
src/
  cli/            terminal edge (image tune load run chat http unload list inspect)
  http/           network edge (OpenAI-compatible server + config)
  runtime/        engine (Session, Executor, service, cache, residency)
  architectures/  model declarations (core contract + per-arch impl)
  tuner/          compiles the plan from measured costs
  plan/           the AOT plan (Config + Dispatch[] + ResidencyGraph)
  kernels/        kernel families + AtomicOps
  image/  gguf/   the compiled image / the external raw format
  gpu/            Vulkan co-processor (device, pipelines, kernels, grid)
  data/           the shared TensorView
  utils/          io, mmap, binary_reader, cpu
```

## Status

The runtime, the tuner and the CLI are usable end to end. **Speculative
decoding** was prototyped and measured: on a compute-bound target it cannot
beat standalone decode (the batched verify does not amortize), so it is not
part of the default path — the experiment is not in the main tree. See the
documents above for the kernel architecture, CPU benchmarks and the OpenCode
integration guide.

## License

MIT. See [LICENSE](LICENSE).

The model images and GGUF weights are not distributed in this repository; they
keep their own model licenses.

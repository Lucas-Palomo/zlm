07 - Think Suppression (--show-think) and Verbose Levels

Status: done
Commits:

## Goal

Two output controls for `run`:

1. **`--show-think`** (flag, default off): the reasoning block (gemma4
   `<|channel>thought ... <channel|>`, qwen3 `thinking ...  response`) is
   **suppressed** by default and shown only with `--show-think`. Orthogonal to
   `--stream`: `--stream --think` streams only the answer; `--stream --think
   --show-think` streams the reasoning too.
2. **`--verbose[=min|info|all]`** (bare = `min`, default off): a stderr trace,
   additive levels:
   - **min** — the phase timing only (load / prefill / decode / generation);
   - **info** — min plus the **per-layer** decode timing and the **logits** (per
     generated token: id, chosen logit, piece);
   - **all** — info plus the **prompt tokenization** (ids) and the **think block**
     (the raw reasoning, even when stdout suppresses it).

`--verbose` becomes a value flag (`--metrics`-style), replacing the current
boolean.

## Scope

- **In:** the `--show-think` flag and the think filter in the tokenizer output
  path (batch + stream); `--verbose` levels (min/info/all); the runtime capture
  they need (per-layer time, per-token logit, prompt ids, raw reasoning); a
  `TRACE` frame in the service reply the client prints to stderr.
- **Out:** changing the plan/KV/prefix cache (output only); the HTTP surface;
  logging frameworks; a persistent log file.

## Design

### Think suppression

- The tokenizer resolves the think **open/close token ids** from the vocab at
  load: `<|channel>`/`<channel|>` (gemma4) and ` thinking`/`</think>` (qwen3),
  whichever are present (`think_open_id`/`think_close_id`, nullable).
- The decode output path (`decodeText` and the streaming `decodeTokenAppend`)
  tracks an "inside think" state: the open id enters, the close id leaves; while
  inside, nothing is emitted unless `show_think`. The open/close ids are control
  tokens and are dropped either way, so enabling it only re-adds the reasoning
  **content**.
- The state is per generation (reset each `generate`).

### Verbose levels

- `run` sends the level in the `GENERATE` request (`verbose` field: 0 off / 1 min
  / 2 info / 3 all) plus `show_think` (0/1).
- The daemon emits `TRACE\t<line>\n` frames (stderr-bound) before the final `OK`;
  the client prints each to stderr. `TRACE` frames may interleave with `DATA`
  frames in streaming; the client routes DATA->stdout, TRACE->stderr, `OK`->end.
- **min** trace: the phase line (load/prefill/decode/generation ms), emitted by
  the client (it alone knows `load`).
- **info** adds one line per transformer layer with its accumulated decode time
  (the executor accumulates a `layer_ns[]`; the runtime only reports it) and, per
  generated token, `tok <i> id=<id> logit=<f> piece=<text>` (the sampler exposes
  the chosen logit).
- **all** adds the prompt token ids (one line) and the think block (the raw
  reasoning text, regardless of `--show-think`).

Diagnostic only: the trace never feeds a decision and never changes stdout.

## Steps

1. `--show-think`: tokenizer think-id resolution + the filter in `decodeText` and
   `decodeTokenAppend`; wire the flag through `GENERATE` and `run`. **Done**
   (GENERATE is now seven fixed fields; verified on gemma4: `--think` hides the
   reasoning, `--think --show-think` shows it, batch and stream).
2. `--verbose` value flag (`min|info|all`); the `TRACE` frame in the protocol and
   the client's stderr printer; the `min` trace (phase only). **Done**
   (the phase line is client-side, which knows `load`; the daemon emits the
   per-layer lines at `info`).
3. The `info` trace (per-token logits) — capture the chosen logit in the decode
   loop. **Done.**
4. The `all` trace (prompt ids + raw think block). **Done** (`decodeThink`).
5. Tests + e2e (think on/off, each verbose level, stream + think combinations).
   **Done** (248/248; stdout byte-identical with/without `--verbose`).
6. `zig build` / `zig build test` green. **Done.**

## Flow / Arch

```text
run --think [--show-think] [--stream] [--verbose=...]
  GENERATE ... think show_think verbose <prompt>
    one decode: stdout <- answer (think content gated by --show-think)
                stderr <- TRACE frames (level-dependent)
    OK (metrics)
```

## Definition of Done

- [x] `--think` hides the reasoning; `--show-think` shows it (batch and stream).
- [x] `--stream --think` streams only the answer; adding `--show-think` streams
      the reasoning too, UTF-8-safe.
- [x] `--verbose` (min) prints only the phase timing; `info` adds the per-layer
      timing and per-token logits; `all` adds prompt ids + the think block.
- [x] stdout is unchanged by `--verbose`; batch stdout with/without the new flags
      is byte-identical when they are off.
- [x] plan/KV/prefix cache untouched; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the think-id resolution + output filter, the `TRACE` frame, the verbose
  levels.
- **derived:** `cli/06` (stream), `architectures/09` (think), `runtime/04`
  (tokenizer/generation), `cli/04`/`05` (metrics/profile).

## Comments

- **Default hides reasoning.** A user reading the answer should not see the
  chain-of-thought unless asked; `--show-think` opts in.
- **Diagnostic only.** Verbose is stderr; it never changes stdout or the plan.
- **Per-layer/logits capture** is runtime-side and additive; the plan is
  unchanged.

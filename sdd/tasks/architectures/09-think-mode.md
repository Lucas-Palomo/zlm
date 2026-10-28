09 - Think Mode Flag (`run --think`)

Status: done
Commits:

## Goal

Make "thinking" an explicit, per-request decision instead of an implicit,
per-model guess. Today the chat template is fixed: `formatPrompt` always emits
the same single-turn prefix, so a model that can reason (gemma-4's `<|think|>`
channel; Qwen3's ` thinking` block) has no way to be told whether to think, and
a caller has no way to ask. Guessing per architecture is the "chaos" this task
removes.

Add a `--think` flag to `run` (boolean; **absent = off**, **present = on**).
The architecture declares *how* a think turn is wrapped; the runtime applies it;
the flag is the caller's intent. gemma-4 supports thinking today
(`<|think|>` + `<|channel>`), so the flag is testable before Qwen3 lands.

## Scope

- **In:**
  - A think variant on the shared chat template (`architectures/core/chat.zig`):
    `Template.applyWith(allocator, text, think: bool)` and the two decoration
    fields the architectures use.
  - gemma-4 declares its `<|think|>` wrapping; the other arch (Qwen3) declares
    its ` thinking` pre-close (declaration only here; consumed by task 10).
  - Runtime: `Session.formatPrompt(text, think)` and
    `Session.generate(..., think)`.
  - Service/IPC: the `GENERATE` request carries the flag; the daemon passes it
    through (`runtime/service.zig`).
  - CLI: `run --think` (flag) + help text.
- **Out:**
  - A tri-state / `--no-think` / `--think=on|off` surface (binary flag only).
  - A new `serve` CLI flag; a `chat` command; multi-turn or tool templates.
  - The full Jinja `tokenizer.chat_template` (only the think decoration is
    modeled; the turn markers stay as resolved in task 08).
  - Qwen3 forward/session (task 10).

## Design

Two independent concerns, kept apart:

- **Intent** is per request -> the `--think` boolean.
- **Markup** is per architecture -> the arch's `chat.Template`.

The edge never builds a prompt (CONSTITUTION: "an edge talks only to the
runtime"); it ships a boolean and the runtime owns the template.

### Template (`architectures/core/chat.zig`)

Extend the shared `Template` with two decorations and a think-aware apply:

```zig
pub const Template = struct {
    user_prefix:   []const u8 = "",
    user_suffix:   []const u8 = "",
    model_prefix:  []const u8 = "",
    /// Emitted before the user turn when thinking is ON (gemma-4: the system
    /// turn that carries `<|think|>`). Empty when the arch has no think mode.
    think_prelude: []const u8 = "",
    /// Emitted after `model_prefix` when thinking is OFF (Qwen3: the pre-closed
    /// `" thinking\n\n response\n\n"`). Empty when not needed.
    think_prefill: []const u8 = "",

    pub fn apply(self: Template, a: Allocator, text: []const u8) ![]u8 {
        return self.applyWith(a, text, false);
    }

    pub fn applyWith(self: Template, a: Allocator, text: []const u8, think: bool) ![]u8 { ... }
};
```

The emitted prompt for one user turn:

```text
[think_prelude if think]  user_prefix  <trimmed text>  user_suffix  model_prefix
[think_prefill if !think]
```

`apply` keeps the existing non-think signature, so existing call sites and tests
are unchanged. `isEmpty` stays as-is; add `hasThink` (either decoration non-empty)
for diagnostics/tests.

### Architecture declarations

- **gemma-4** (`impl/gemma4/chat.zig`), matching the unsloth `<|turn>` template:
  - `think_prelude = "<|turn>system\n<|think|>\n<turn|>\n"` (the enable_thinking
    system turn, injected at the very top), `think_prefill = ""`.
  - think OFF emits no decoration (the template default with
    `enable_thinking = false`).
- **Qwen3** (declared in task 10, fixed here as the second concrete case):
  - `think_prelude = ""`, `think_prefill = " thinking\n\n response\n\n"`.
  - think ON emits nothing extra (the model opens ` thinking` itself).

### Runtime merge (`runtime/session.zig`)

`formatPrompt` currently prefers the compile-stored markers (`zlm.chat.*`) and
falls back to the arch template. Keep that, but the stored markers only describe
the *turn shape*; the think decoration always comes from the arch. New signature:

```zig
pub fn formatPrompt(self: *Session, allocator, text, think: bool) ![]u8
```

- Build the effective template as: turn markers = stored (if all three present)
  else the arch's; think fields = the arch's `Template.think_prelude` /
  `think_prefill`.
- Apply with `applyWith(text, think)`.
- `applyChatTemplate` in `architectures/root.zig` gains a think-aware sibling so
  the runtime can read the arch declaration (not just the fully-applied string).

`generate` threads `think` into `formatPrompt`; nothing else in the forward path
changes (think affects only the prompt string, so the plan/prefix-cache logic is
untouched).

### Protocol (`runtime/service.zig`)

Extend the canonical request with a fixed `think` field (`0|1`), placed before
the trailing prompt so the prompt may still contain tabs:

```text
GENERATE\t<max_tokens>\t<temperature>\t<top_k>\t<top_p>\t<think>\t<prompt>\n
```

- `parseGenerateCanonical` reads five fixed fields instead of four.
- `GenerateRequest` gains `think: bool`; `handleClient` passes it to
  `runtime.generate`.
- The legacy `prompt-first` and greedy shapes remain supported and default
  `think = false` (the same best-effort ambiguity that already exists between the
  canonical and previous shapes is accepted; documented in Comments).
- `service.generate` (client) gains a `think` parameter and writes the field.

### CLI (`cli/commands/run.zig`)

- Add `.{ .long = "--think", .kind = .flag }` to `specs`.
- Read `const think = opt.has("--think");` and pass it to `service.generate`.
- Update `help_text` (one line: `--think  enable the model's reasoning mode
  (default off; ignored by models without a think template)`).
- A model whose arch declares no think decoration is a silent no-op (the flag
  does not error); documented, not warned, so stdout stays clean.

## Steps

1. `core/chat.zig`: add `think_prelude`/`think_prefill`, `applyWith`, `hasThink`;
   keep `apply`. Unit tests for on/off/neutral. **Done.**
2. `impl/gemma4/chat.zig`: set the gemma-4 think prelude; test that think-off is
   byte-identical to today and think-on prepends the system/`<|think|>` turn.
   **Done.**
3. `runtime/session.zig`: `formatPrompt(text, think)` + `generate(..., think)`;
   tests for the stored-markers + arch-think merge and the arch fallback. **Done.**
4. `runtime/service.zig`: `GenerateRequest.think`; 5-field canonical parser;
   `handleClient`; client `generate` param; update protocol tests. **Done.**
5. `cli/commands/run.zig`: `--think` spec, read, pass, help. **Done.**
6. Verify: `zig build` / `zig build test` (225/225 pass); e2e on gemma-4 E2B
   (`rt-e2`) — non-think coherent, `--think` surfaces the `thought`/reasoning
   channel. Required recompiling `e2b-test` (it predated task 08; see Comments).
   **Done.**

## Flow / Arch

```text
run --think
  └─ service.generate(..., think)                       (cli/commands/run.zig)
       └─ GENERATE ... \t<think>\t<prompt>              (runtime/service.zig)
            └─ Session.generate(..., think)
                 └─ Session.formatPrompt(text, think)
                      ├─ turn markers: zmeta-stored, else arch Template
                      ├─ think decoration: arch Template (think_prelude/prefill)
                      └─ Template.applyWith(text, think)
```

## Definition of Done

- [x] `--think` accepted by `run`; absent = off, present = on; documented in help.
- [x] gemma-4: think-off output byte-identical to today; think-on emits the
      `<|turn>system\n<|think|>\n<turn|>\n` prelude.
- [x] Think decoration is an architecture-declared template field; the edge
      passes only a boolean.
- [x] Runtime merges compile-stored markers with the arch think decoration;
      fallback path covered by tests.
- [x] `GENERATE` carries the flag; legacy request shapes still parse.
- [x] e2e: `zlm run <gemma4-e2b> --prompt ...` and with `--think` both coherent.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the two-field think template (`think_prelude`/`think_prefill`), the
  binary `run --think` policy, the request-level intent vs arch-level markup
  split.
- **spec:** GGUF `tokenizer.chat_template` (`enable_thinking`, `<|think|>`,
  `<|channel>`, ` thinking`/`</think>`) for gemma-4 and Qwen3.
- **consulted:** the earlier local prototype's `architectures/*/chat.zig` (`PromptOptions.think`) and
  `cli/run.zig` (`--think`/`--no-think`) — precedent only; the V2 shape is a
  binary flag with off default, not the earlier local prototype's tri-state with on default.
- **derived:** `architectures/08` (markers resolved at compile), `architectures/10`
  (Qwen3 needs the flag).

## Comments

- **Default off vs model default.** Both templates default `enable_thinking` to
  false, so off matches the GGUF default. The earlier local prototype defaulted to on; this task
  deliberately follows "explicit intent, absent = off".
- **Stored-markers vs arch-decoration consistency.** The stored markers select
  the turn shape (`<|turn>` for unsloth gemma-4; `<start_of_turn>` only as the
  arch fallback). gemma-4's `think_prelude` is written against the `<|turn>`
  form, which is what the shipped unsloth fixtures resolve to. A hypothetical
  variant with a different turn marker and `<|think|>` would need its prelude
  parameterized by the stored markers — deferred (no such fixture).
- **Prefix cache.** think changes only the prompt bytes, so the token stream and
  the existing KV/prefix-cache keys already account for it; no new invalidation.
- **Protocol ambiguity.** Adding a 5th canonical field shifts the existing
  canonical/previous ambiguity by one position; mixed-version clients are not a
  supported case for this dev tool.
- **`--think` on a think-less model** is a silent no-op (no error, no notice).
- **Stale image / prefix cache (as-built finding).** The shipped `e2b-test`
  image predated task 08, so its `zmeta` had no `zlm.chat.*`; gemma-4 think
  then fell back to `<start_of_turn>` markers + a `<|turn>system` prelude, which
  the model did not accept. Recompiling the image (deterministic, ~36 s) stores
  the `<|turn>` markers and think works. A `.zpc` prefix cache written before a
  recompile is stale for the new image and must be cleared (`rm
  <store>/runtimes/<image>_*.zpc`); otherwise the first think run can reuse
  incompatible KV. Both are operational, not code, issues.

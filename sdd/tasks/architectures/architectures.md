# architectures

title: Architectures
reason: Model architectures declare how a model executes. gemma4 emits a
        semantic instruction graph; the tuner compiles it into the real
        directed plan; the runtime executes it.

## Tasks

### [01 - gemma4 Declaration](01-gemma4-declaration.md)
Status: done

Context: the generic `Arch` contract; gemma4 `config.zig`/`arch.zig` on the
shared role space; the `zmeta` reader; the architectures registry; `tune` maps
tensor -> op via the declaration.

### [02 - Semantic Graph and gemma4 Block](02-semantic-graph.md)
Status: done

Context: the semantic instruction graph (ops/roles/slots), gemma4's block
declaration, `Dispatch` bindings + buffer table, a minimal `tuner/scheduler`,
and the `runtime/engine` that executes the plan.

### [03 - Forward Block](03-forward-block.md)
Status: done

Context: make gemma4's declaration a complete forward — residual `add`s,
`layer_output_scale`, final softcap, correct `write_kv` (k,v -> cache), a
`core/kv.zig` `KvLayout` declared by the architecture, and instruction params
resolved from the config.

### [04 - Gemma4 Forward Correctness](04-gemma4-forward-correctness.md)
Status: done

Context: restore the gemma4 semantics the current declarations drifted from
(attention scale, embedding scale, gated GELU, shared-KV store/source, sliding
`t_start`, per-head V norm) — all decided AoT in the tuner/plan.

### [05 - Per-Layer Embeddings](05-per-layer-embeddings.md)
Status: done

Context: gemma4's PLE (E2B/E4B; plen>0): per-layer token embedding + model
projection, a per-layer combined input, and a post-FFN gate/proj/norm branch.
Adds contract slots/dims, a block gate (`.ple`), and three atomics.

### [06 - Gemma4 SWA K/V (larger variants)](06-gemma4-swa-kv.md)
Status: done

Context: the 12B fails in the current code (`error.Unsupported` at `generate`) because its
per-layer `head_count_kv` (`[8,8,8,8,8,1,…]`) makes the SWA layers lack an
`attn_v` tensor (V=K or K+V packed), which the gemma4 declaration still emits.
Handle the SWA K/V and validate E4B/12B.

### [07 - gemma4 Forward NaN (attention / KV)](07-gemma4-forward-nan.md)
Status: done

Context: the E2B/E4B decode produced all-NaN logits (empty output). The FFN
`gelu` (`geluInPlace`) computed `tanh` as `(e^{2x}-1)/(e^{2x}+1)` without
clamping, so large FFN values overflowed `e^{2x}` to `inf` -> `inf/inf = NaN`.
Clamp `tanhV` to ±15; the variants now generate coherent text.

### [08 - gemma4 Chat Template (resolved at compile, stored in zmeta)](08-gemma4-chat-template.md)
Status: done

Context: the chat template was hardcoded in the arch, but gemma-4 variants differ
(unsloth ships a `<|turn>`/`<turn|>` template). `compile` derives the markers from
the vocab special tokens and stores `zlm.chat.*` in the zmeta; the runtime reads
them and falls back to the arch template. E2B formats with `<|turn>`.

### [09 - Think Mode Flag (run --think)](09-think-mode.md)
Status: done

### [10 - Qwen3 (declaration + byte-level tokenizer)](10-qwen3.md)
Status: done

Context: the second concrete architecture after gemma4. A `qwen3` declaration
(config/names/block/chat) on the shared contract, registered in the image/tuner
path, plus the GPT-2 **byte-level BPE** the Qwen3 tokenizer needs. 4B (tied) and
8B (untied) run coherently; consumes the `think` prefill (task 09).

Context: make "thinking" an explicit per-request decision. An architecture
declares its think markup on `chat.Template` (`think_prelude`/`think_prefill`);
`run --think` (binary, default off) carries the intent through the service to
`Session.formatPrompt`. gemma-4 (`<|think|>`/`<|turn>system`) makes it testable
now; Qwen3 (task 10) consumes it.

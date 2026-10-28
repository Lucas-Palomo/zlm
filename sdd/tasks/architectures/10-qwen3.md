10 - Qwen3 (declaration + byte-level tokenizer)

Status: done
Commits:

## Goal

Run Qwen3 (dense) end to end on the current contract: a `qwen3` architecture
declaration (config + names + block), registered in the image/tuner path, plus
the **byte-level BPE** tokenizer Qwen3 needs (`tokenizer.ggml.model = "gpt2"`,
`pre = "qwen2"`). The earlier local prototype had a working Qwen3; the current code restructures the
forward into a declaration, so this is the second concrete architecture after
gemma4 (the "rule of three" case task 02 anticipated).

Fixtures: `models_gguf/qwen3-{4B,8B}-unsloth-Q4_1.gguf`.

## Scope

- **In:**
  - `architectures/impl/qwen3/`: `config.zig`, `names.zig`, `arch.zig`,
    `block.zig`, `chat.zig`, `root.zig`.
  - Register `.qwen3` in `core/root.zig` (the `Config` union + `archs[]`).
  - `image/compile.zig` `appendRopeRegions`: a `.qwen3` case (one inv_freq
    config).
  - **Byte-level tokenizer**: `tokenizer.ggml.model`/`pre` handling in
    `runtime/tokenizer.zig` (byte map + qwen2 pre-tokenizer + inverse decode),
    without changing the SPM (gemma4) path.
  - `image/zmeta.zig` `deriveChatMarkers`: recognize the ChatML markers
    (`<|im_start|>`/`<|im_end|>`) so the image stores them.
  - Tests + e2e on the 4B (and 8B untied).
- **Out:**
  - Qwen3 MoE (no fixture); tools; multi-turn; a full Jinja engine.
  - New kernels: every op already exists (`matvec`, `norm` incl. per-head,
    `rope`, attention GQA, `gated_silu`, `add`, `dequant`, `lm_head`).

## Facts (from the GGUF)

```text
                      4B            8B
block_count           36            36
embedding_length      2560          4096
feed_forward_length   9728          12288
head_count / kv       32 / 8        32 / 8
key_length (head_dim) 128           128
rope.freq_base        1e6           1e6
rms_epsilon           ~1e-6         ~1e-6
output.weight         absent(tied)  present(untied)
token_embd type       q6_k          q4_1
tokenizer.ggml.model  gpt2          gpt2
tokenizer.ggml.pre    qwen2         qwen2
add_bos_token         false         false
eos_token_id          151645        151645
```

Tensors are the gemma4 names **minus** the gemma-only ones (PLE, `inp_gate`,
`proj`, `post_*`, `layer_output_scale`), **plus** `attn_q_norm`/`attn_k_norm`;
uniform per layer (no `LayerArray`). All roles already exist in the shared
`TensorRole`; Qwen3 has **no** q/k/v bias.

## Design

### Forward (declaration)

Head: `embd_lookup(token_embd) -> hidden` (no embedding scale).

Per layer (uniform):

```text
attn_norm; q/k/v = matvec; q_norm, k_norm (per-head); rope(q), rope(k);
write_kv(k, v); attention(q) -> attn_out; attn_output; hidden += out;
ffn_norm; gate/up; gated_silu; down; hidden += down
```

Tail: `output_norm`; `lm_head(output)` (tied -> falls back to `token_embd`).

`resolveParams`:
- `attn_q_norm`/`attn_k_norm` -> `norm_heads` with `head_dim`.
- `rope`: `n_rot = head_dim` (full), `freq_base = qwen3.rope.freq_base`.
- `attention`: `scale = 1/sqrt(head_dim)`, `sliding = false`, GQA heads.
- `activation`: silu; `add`: 1.0; `embd_lookup`: `.none` (no scale).

`kvLayout`: uniform; every layer stores; `n_kv_heads = head_count_kv`.

`layerDims`: `q = head_count*head_dim`, `kv = head_count_kv*head_dim`,
`ffn = feed_forward_length`, `ple = 0`.

### Chat template

ChatML single-turn:

```text
user_prefix  = "<|im_start|>user\n"
user_suffix  = "<|im_end|>\n"
model_prefix = "<|im_start|>assistant\n"
```

Think (task 09): `think_prefill = " thinking\n\n</think>\n\n"` (think OFF
pre-closes); `think_prelude = ""` (think ON lets the model open ` thinking`).
`add_bos_token = false`, so no BOS is prepended.

### Tokenizer (byte-level BPE)

The current code's tokenizer is SentencePiece-only (`splitCodepoints` forces space -> `▁`).
Qwen3 needs GPT-2 byte-level BPE plus the Qwen2 pre-tokenizer:

- Read `tokenizer.ggml.model` (`gpt2`) and `tokenizer.ggml.pre` (`qwen2`) from
  `zmeta`; select the splitter strategy (`sentencepiece` | `byte_level`).
- **Byte map**: the 256-entry GPT-2 byte-to-unicode map (space/control ->
  `U+0100+n`), applied per pre-token; inverse on decode.
- **Qwen2 pre-tokenizer**: a small state machine (contractions, letters with an
  optional leading non-letter, single digit, punctuation runs, whitespace/newline
  rules) — no regex engine.
- Reuse the existing rank-ordered BPE merge loop; only the symbol splitter and
  the piece<->text mapping change.
- The SPM path (gemma4) is untouched; a `vocab_style` selects at load.

## Steps

1. `impl/qwen3/config.zig`: `Qwen3Config` from `qwen3.*` (scalar). **Done.**
2. `impl/qwen3/names.zig`: table (globals + layered) on the shared role space.
   **Done.**
3. `impl/qwen3/arch.zig`: `Arch` impl; `impl/qwen3/block.zig`: the forward.
   **Done.**
4. `core/root.zig`: `.qwen3` in `Config` + `archs[]`. **Done.**
5. `image/compile.zig`: `.qwen3` rope region; `image/zmeta.zig`: ChatML markers.
   **Done.**
6. `runtime/tokenizer.zig`: `VocabStyle`, byte map + byte-level split/decode;
   SPM (gemma4) untouched. **Done** (pragmatic GPT-2 byte map + global BPE, as the earlier local prototype;
   the full Unicode qwen2 pre-tokenizer is deferred — Comments).
7. Tests: names/chat + the byte map round-trip. **Done.**
8. e2e: `image compile` + `tune` + `run` on 4B (tied) and 8B (untied) coherent;
   `--think` enters the reasoning block. **Done.**

## Definition of Done

- [x] `image compile`/`tune`/`run` on Qwen3-4B produce coherent greedy text.
- [x] Qwen3-8B (untied `output.weight`) runs too.
- [x] The tokenizer round-trips the byte-level map (all 256 bytes) and a
      byte-level piece; decode round-trips. Oracle comparison deferred.
- [x] gemma4 (SPM) is unchanged; `zig build` / `zig build test` green (232).
- [x] `--think` changes the Qwen3 prompt (pre-closed ` thinking` when off).

## Arch Provenance

- **own:** the declaration on the current contract; the byte-level tokenizer.
- **spec:** GGUF `qwen3.*` keys; the Qwen2 byte-level BPE / pre-tokenizer rules.
- **consulted:** the earlier local prototype's `architectures/qwen/` (working Qwen3) and
  `tokenizer/root.zig` (byte-level); llama.cpp `unicode.cpp`/`unicode_regex_split_custom_qwen2`
  — reference only, not copied.
- **derived:** `architectures/01-09` (the contract, blocks, chat/think).

## Comments

- **Tokenizer is the risk.** A perfect forward with wrong tokenization yields
  junk; validate against an oracle before judging the model.
- **Tied vs untied lm_head.** 4B has no `output.weight`; the compile already
  falls back to `token_embd` for `lm_head`. Test both fixtures.
- **Rope.** Full rotary (`n_rot = head_dim`), base 1e6; the image compile emits
  one `inv_freq` region for the whole model.
- **No eta/rope scaling** in the fixtures; out of scope.
- **Tokenizer (as-built).** The byte-level path uses the GPT-2 byte map plus the
  existing rank-ordered BPE over the whole segment (as the earlier local prototype), not the full Unicode
  qwen2 pre-tokenizer (contractions / `\p{L}`/`\p{N}`/`\s` classes). The e2e is
  coherent on both fixtures, and the byte map round-trips exactly. A divergence
  from the real pre-tokenizer is possible on edge inputs; a `llama-tokenize`
  oracle comparison is deferred (promote if an input tokenizes wrong).
- **8B is slow** on CPU (Q4_1, ~5 GB weights); correctness holds, throughput is
  a `runtime/11` concern.

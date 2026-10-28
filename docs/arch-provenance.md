# Architecture and Code Provenance

Status: living document
Scope: `src/architectures/`, `src/kernels/`, `src/gpu/`
Last reviewed: 2026-11-09

This document records where the non-trivial parts of zlm come from. The policy
is the one in [`CONSTITUTION.md`](../CONSTITUTION.md): **llama.cpp and similar
projects are reference material, not a base to port**. Code is rewritten for
zlm's needs unless it is explicitly marked and attributed as a port.

The rest of the tree (CLI, HTTP, runtime, image formats, GGUF reader, tuner,
plan, data and utils) is zlm-owned. This document covers the three areas where
external formats, model definitions and GPU APIs make provenance non-obvious.

## Legend

| Term | Meaning | Attribution needed |
| --- | --- | --- |
| **own** | Original zlm design or implementation. | No |
| **spec** | Own implementation of an external specification or file format. | Usually a link to the spec |
| **consulted** | Used as reference for formats, algorithms or design shape; no code copied. | Optional note |
| **derived** | Algorithm or structure adapted from an external source, rewritten in zlm's architecture. | Yes: source and license |
| **port** | Direct or near-direct translation of third-party code. | Yes: copyright, license and source |

"Consulted" is not "derived" and "derived" is not "port". The distinction is the
point of this document.

## Summary

| Subsystem | Classification | External basis |
| --- | --- | --- |
| `src/architectures/` | Own declaration framework; specs for metadata, tensor names and chat/tool templates | GGUF metadata, Gemma 4 and Qwen 3 model definitions, ChatML, Gemma 4 tool DSL |
| `src/kernels/` | Own kernel architecture; derived quant codecs; own implementations of published transformer algorithms | ggml block layouts and reference dequantizers, RoPE, RMSNorm, GQA, SwiGLU, FlashAttention-style online softmax |
| `src/gpu/` | Own backend, grid and GLSL shaders; Vulkan API layer follows the Vulkan spec | Vulkan specification, ggml block layouts, zlm GEMV/GEMM design |

## 1. `src/architectures/`

### 1.1 Core contract and graph

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `core/contract.zig` | The `Arch` declaration contract: tensor names/roles, config loader, chat and tool hooks | **own** | zlm vocabulary; no upstream equivalent |
| `core/graph.zig` | Semantic `Block`/`Instr`/`Slot`/`Params` vocabulary | **own** | The forward is declared as a graph; this is a zlm design decision |
| `core/kv.zig` | Per-layer KV layout declaration (GQA, sliding window, shared layers) | **own** | Shapes follow the model configs; the declaration model is zlm's |
| `core/meta.zig` | Helpers to read scalar/per-layer GGUF metadata arrays | **own** | Reads `zmeta`, not a third-party parser |
| `core/names.zig` | Generic tensor-name table mapper (`blk.N.<suffix>`) | **own** | Table-driven; the tables themselves are spec data |
| `core/chat.zig` | Shared chat-template shape (user/model/soul/think decorations) | **own** framework, **spec** data | The literal markers come from each model's template |
| `core/tools.zig` | OpenAI-shaped tool declarations/calls and the `Dialect` contract | **own** framework, **spec** shape | The JSON shape follows the OpenAI API; the syntax is per architecture |
| `core/root.zig` | Architecture registry (`gemma4`, `qwen3`) | **own** | |

### 1.2 Gemma 4

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `impl/gemma4/config.zig` | Reads `gemma4.*` metadata: layers, heads, sliding window, shared K/V, PLE, RoPE configs | **spec + own** | Keys and semantics come from the Gemma 4 GGUF metadata; the parser is zlm's |
| `impl/gemma4/names.zig` | Gemma 4 tensor-name table | **spec** | GGUF tensor names (`blk.N.attn_q.weight`, etc.) |
| `impl/gemma4/arch.zig` | Binds the Gemma 4 config, block, KV layout, chat and tools to the contract | **own** | Includes the `attn_k`-as-`attn_v` fallback for 12B/31B shared-KV layers |
| `impl/gemma4/block.zig` | Gemma 4 decode forward as semantic instructions | **own** declaration | The architecture follows the model definition; no forward code is copied |
| `impl/gemma4/chat.zig` | Gemma turn template, system turn and think decorations | **spec/consulted** | Matches the Gemma 4 / Unsloth `<start_of_turn>` template; no template code is vendored |
| `impl/gemma4/tools.zig` | Gemma 4 native tool-call DSL renderer/parser | **spec + own** | Syntax comes from the Gemma 4 model definition; the parser/renderer are zlm code |

The Gemma 4 block uses standard transformer operations (RMSNorm, RoPE, GQA,
gated MLP). Their provenance is covered in [§2.4](#24-transformer-kernels).

### 1.3 Qwen 3

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `impl/qwen3/config.zig` | Reads `qwen3.*` dense config | **spec + own** | `qwen3.*` GGUF metadata; zlm parser |
| `impl/qwen3/names.zig` | Qwen 3 tensor-name table | **spec** | GGUF tensor names |
| `impl/qwen3/arch.zig` | Binds config, block, KV layout and chat to the contract | **own** | Dense, uniform layers; no tool dialect |
| `impl/qwen3/block.zig` | Qwen 3 decode forward as semantic instructions | **own** declaration | Follows the dense Qwen 3 model definition |
| `impl/qwen3/chat.zig` | ChatML template and think prefill | **spec/consulted** | ChatML markers and the pre-closed ` thinking` block follow Qwen 3's template |

### 1.4 Standard model math

These are published algorithms, implemented in zlm's own style:

| Algorithm | Used by | Reference |
| --- | --- | --- |
| RoPE (rotary position embedding) and proportional RoPE | Gemma 4, Qwen 3 | *RoFormer*, Su et al., arXiv:2104.09864 |
| RMSNorm | Gemma 4, Qwen 3 | *Root Mean Square Layer Normalization*, Zhang & Sennrich, arXiv:1910.07467 |
| GQA (grouped-query attention) | Gemma 4, Qwen 3 | *GQA*, Ainslie et al., arXiv:2305.13245 |
| SwiGLU / GeGLU | Qwen 3 / Gemma 4 | *GLU Variants Improve Transformer*, Shazeer, arXiv:2002.05202 |
| Online-softmax attention | attention kernel | *FlashAttention*, Dao et al., arXiv:2205.14135 |

The zlm code is an implementation of these methods, not a port of a specific
reference implementation.

## 2. `src/kernels/`

### 2.1 Kernel architecture and catalog

| Path | What it is | Provenance |
| --- | --- | --- |
| `api.zig` | `Kernel`/`Family`/`AtomicOp` vocabulary and type-erased `run` | **own** |
| `registry.zig` | Aggregated kernel catalog consumed by the tuner | **own** |
| `recipes.zig` | Composable execution recipes (`gate_up_fused`, etc.) | **own** |
| `simd.zig` | Compile-time SIMD width variants and FMA helper | **own** |
| `linear/matvec/types.zig` | Matvec plan/variant grid (width, tile, unroll, prefetch) | **own** |
| `linear/matvec/root.zig` | Matvec family surface | **own** |

The L1/L2/L3 split, the kernel × variant grid and the generated `MatVecPlan`
catalog are zlm design choices. llama.cpp/ggml informed what a quantized dot
looks like, not how this catalog is structured.

### 2.2 Quant block layouts

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `linear/quant/types.zig` | GGML block sizes and constants (`QK4_0`, `QK_K`, `Q4_K_BLOCK`, ...) | **spec** | GGML/GGUF quant block layouts |
| `linear/quant/floats.zig` | F32/F16/BF16 decode | **spec/own** | IEEE formats; trivial decode |
| `linear/quant/standard.zig` | Q8_0, Q4_0, Q4_1 dequant | **derived** | Decode loop follows the GGML block layout; rewritten with Zig vectors |
| `linear/quant/k_quants.zig` | Q4_K, Q5_K, Q6_K dequant | **derived + one direct translation** | See [§4](#4-direct-ports-and-attribution) |

The block layouts are a compatibility contract with GGUF. Any implementation
that reads a Q4_0 tensor necessarily follows the same byte layout and scale/min
semantics; that is spec compliance, not copied architecture.

### 2.3 Matvec, dot and GEMM

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `linear/matvec/layout.zig` | Block expand/dot helpers for q4_0, q4_1, q8_0, q4_K, q6_K | **own** | Integer dot pattern is informed by the ggml `vec_dot` family; no direct copy identified |
| `linear/matvec/kernels/quantized.zig` | Quantized L1 families (`fp` and `int` dots) | **own** | Uses the ggml block layouts as spec |
| `linear/matvec/kernels/float.zig`, `bfloat.zig` | F32/F16/BF16 L1 families | **own** | |
| `linear/matvec/form/single.zig` | Single-row sweep, bias and LM-head variants | **own** | |
| `linear/matvec/form/batched.zig` | Register-tiled prefill rows | **own** | |
| `linear/matvec/form/dot.zig` | Generic dot over an L1 family | **own** | |
| `linear/matvec/form/fused.zig` | Fused norm→matvec, gate/up and MLP forms | **own** | |
| `linear/vector.zig` | Vector helpers and softmax | **own** | |
| `linear/matmul/root.zig` | Blocked/register-tiled GEMM | **own** | Standard GEMM technique, not a BLAS port |

### 2.4 Transformer kernels

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `transformer/attention.zig` | GQA attention, sliding window, fused online-softmax head, two-pass head | **own implementation** | FlashAttention-style online softmax; the algorithm is published, the code is zlm |
| `transformer/norm.zig` | RMSNorm, per-head Q/K norm, fused QK-norm+RoPE | **own implementation** | RMSNorm is a published method |
| `transformer/rope.zig` | RoPE, interleaved RoPE, proportional RoPE, table-based RoPE | **own implementation** | RoPE/proportional RoPE from the model definitions/papers |
| `transformer/activation.zig` | SiLU, GELU-tanh, gated SiLU/GeGLU, softcap | **own implementation** | Standard activation formulas |
| `transformer/elementwise.zig` | Add/scale/combine/softcap families | **own** | |

## 3. `src/gpu/`

### 3.1 Backend contract, grid and fused paths

| Path | What it is | Provenance |
| --- | --- | --- |
| `root.zig` | GPU-facing API and build-time backend selection | **own** |
| `backend/root.zig` | Backend contract: `Device`, `Quant`, `Resident`, `Handle`, required ops | **own** |
| `kernels/grid.zig` | GPU variant grid (tile, unroll, reduction, workgroup size) and specialization constants | **own** |
| `backend/vulkan/root.zig` | Vulkan backend implementation of the contract | **own** |
| Fused `gate_up`/MLP paths | Single-workgroup and burst variants | **own** |

### 3.2 Vulkan API layer

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `backend/vulkan/types.zig` | Vulkan structs, enums, flags and function-pointer types | **spec** | Transcribed from the Khronos Vulkan headers/specification. No Khronos source is vendored |
| `backend/vulkan/context.zig` | Dynamic `libvulkan` loading, instance/device/queue creation | **own implementation** | Uses the Vulkan API; no link-time `libvulkan` dependency |
| `backend/vulkan/pipeline.zig` | Shader modules, descriptor sets, compute pipelines, buffers, command buffers, fences | **own implementation** | Follows the Vulkan API contract |

The Vulkan specification and headers are published by Khronos under
Apache-2.0. If a future change copies a non-trivial block of header code, the
license notice must be added; today the file is an API transcription, not a
vendored header.

### 3.3 Shaders

| Path | What it is | Provenance | Notes |
| --- | --- | --- | --- |
| `kernels/shaders/matvec_*.comp` | GEMV shaders for Q4_0, Q4_1, Q8_0, Q4_K, Q6_K, F32, F16, BF16 | **own GLSL**, **spec layouts** | The shaders implement zlm's variant grid over GGML block layouts |
| `kernels/shaders/matmul_*.comp` | Batched GEMM shaders for Q4_0, Q4_1, Q8_0, Q4_K | **own GLSL**, **spec layouts** | Comments describe a rewrite of the earlier per-lane shader; no upstream shader is named |
| `kernels/shaders/matvec_gate_up_*.comp` | Fused two-weight gated GEMV | **own** | |
| `kernels/shaders/matvec_mlp_*.comp` | Fused three-weight MLP | **own** | |
| `kernels/shaders/*.spv` | Compiled SPIR-V | **generated** | Checked in so the build does not require `glslc`; record the compiler version in release notes |

**Validated (2026-11-09).** Compared with llama.cpp commit
`a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5`
(`ggml/src/ggml-vulkan/vulkan-shaders/`). Upstream uses a different binding
layout (`ncols`/`stride_*` push constants and `mul_mat_vec_base.glsl`), while
zlm uses `InputVec`/`WeightMat`/`OutputVec`, `groups_x`,
`ROWS_PER_GROUP`/`MAX_LANES` and specialization constants 0–3. No matching
shader body was found in llama.cpp/ggml. The shaders are zlm-owned. The code
comments now use "batched GEMM shader"; the earlier "ported" wording referred to
porting the GEMV memory layout into the batched GEMM shader, not to copying an
upstream shader.

## 4. Direct ports and attribution

### 4.1 `getScaleMinK4` — high-confidence direct translation

- **File:** `src/kernels/linear/quant/k_quants.zig`
- **Upstream:** `get_scale_min_k4` in
  [`ggml/src/ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
- **License:** MIT (llama.cpp/ggml)
- **Status:** validated against llama.cpp commit
  `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5`. The branches and bit operations
  match the upstream function line-for-line in logic; this is effectively a
  direct translation.
- **Action:** add a source/attribution header to the function and a third-party
  notice (or a `NOTICE`/`THIRD_PARTY` section) preserving the upstream
  copyright and MIT text.

### 4.2 Q4_K, Q5_K and Q6_K dequantizers — derived

- **File:** `src/kernels/linear/quant/k_quants.zig`
- **Upstream reference:** `dequantize_row_q4_K`, `dequantize_row_q5_K`,
  `dequantize_row_q6_K` in `ggml/src/ggml-quants.c`
- **License:** MIT (llama.cpp/ggml)
- **Status:** validated against the same llama.cpp commit. The block unpacking,
  scale/min use, high-bit handling and output order follow the reference
  algorithms, but the loops are rewritten with Zig vectors and
  `storeScaled`/`storeScaledI`. The code is **derived**, not verbatim.
- **Action:** add a header citing the reference. Treat the code as **derived**,
  not verbatim, unless a future diff shows otherwise.

### 4.3 Q4_0, Q4_1 and Q8_0 dequantizers — derived from the layout

- **File:** `src/kernels/linear/quant/standard.zig`
- **Status:** the decode is dictated by the GGML block layout; the Zig vector
  implementation is zlm's.
- **Action:** cite the GGML layout/spec in the file header if the file grows
  beyond trivial format decoding.

### 4.4 No direct copies identified in `src/architectures/`

The architecture declarations, config readers, chat wrappers and Gemma 4 tool
parser are zlm code. Their external inputs are metadata keys, model templates
and the Gemma tool syntax; those are specifications, not copied implementation.

## 5. Validation status

Validated on 2026-11-09 against:

- llama.cpp commit `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5`
  (`ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/arch/x86/quants.c` and
  `ggml/src/ggml-vulkan/vulkan-shaders/`);
- the local `gemma-4-E2B-it-unsloth-Q4_0.gguf` metadata
  (`general.base_model.0.repo_url`, `general.quantized_by` and
  `tokenizer.chat_template`);
- the local Vulkan declarations and their use sites.

| Point | Result | Evidence |
| --- | --- | --- |
| Vulkan shaders | **own (no llama.cpp/ggml match)** | Upstream uses `ncols`/`stride_*` push constants and `mul_mat_vec_base.glsl`; zlm uses `InputVec`/`WeightMat`/`OutputVec`, `groups_x`, `ROWS_PER_GROUP`/`MAX_LANES` and spec constants 0–3. No matching shader body was found in llama.cpp/ggml. |
| Q4_K/Q5_K/Q6_K dequant | **derived; one direct translation** | The block unpacking, scale/min order and high-bit handling match `dequantize_row_q4_K` (line 1539), `_q5_K` (1741) and `_q6_K` (1949); the Zig loops are vectorized. `getScaleMinK4` matches `get_scale_min_k4` (line 889) line-for-line in logic. |
| Integer matvec | **own** | Compared with `ggml_vec_dot_q4_0_q8_0` in the x86 backend (line 765). Upstream uses `mul_sum_i8_pairs_float`/`maddubs`; zlm uses widening i16 products, `@shuffle` grouping and deferred bias folding. No direct copy found. |
| Gemma 4 tool DSL | **spec/consulted** | The local GGUF identifies `general.base_model.0.repo_url = https://huggingface.co/google/gemma-4-E2B-it` and `general.quantized_by = Unsloth`; the embedded `tokenizer.chat_template` contains `<|tool>`, `<|tool_call>` and `<|tool_response>`. |
| Vulkan types | **spec** | `types.zig` is a 569-line zlm-specific subset with no Khronos header markers or copyright block; it declares only the needed `Vk*` structs/enums/function pointers. It follows the Vulkan ABI, not a vendored header. |

This review compared only llama.cpp/ggml. Other upstream projects were not
surveyed for the shaders or kernels; the "own" results above are relative to
that comparison.

No open classification questions remain. The remaining work is attribution:

- Add the upstream URL and MIT notice to `getScaleMinK4` and the derived
  `k_quants.zig` dequantizers, plus a `NOTICE` or `THIRD_PARTY` section.
- Record the llama.cpp commit/version used for the comparison in those headers.

## 6. Rules for future changes

- New kernel, architecture or GPU code must state its provenance in the file
  header when it is not entirely zlm-owned.
- A direct port must preserve the upstream copyright, name the source and
  commit/version, and add the license text to `NOTICE` (or an equivalent
  third-party file).
- A derived implementation must cite the source and license, even when the code
  is rewritten.
- "Consulted" is not enough to justify copied code. If code is copied, call it a
  port.
- Update this document when a new architecture, kernel family, shader family or
  GPU backend is added.
- The SDD `Arch Provenance` field remains the per-task record; this document is
  the cross-cutting map.

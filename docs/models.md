# Supported models

zlm reads GGUF v2/v3 files. The architecture is selected from the GGUF
`general.architecture` key; the current tree implements two decoder families.

| GGUF `general.architecture` | Family | Chat template | Think | Tool calls |
| --- | --- | --- | --- | --- |
| `gemma4` | Gemma 4 | Gemma turn template | yes | yes (native Gemma 4 dialect) |
| `qwen3` | Qwen 3 **dense** | ChatML | yes | no |

Anything else is rejected by the runtime. In particular, Qwen 3
Mixture-of-Experts variants are not supported.

## Gemma 4

The `gemma4` implementation covers the Gemma 4 sizes that use the standard
GGUF tensor layout:

| Model | Example GGUF | Quant | GGUF size | Source | Notes |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B | `gemma-4-E2B-it-Q4_0.gguf` | Q4_0 | ~3.0 GB | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | per-layer embeddings |
| Gemma 4 E4B | `gemma-4-E4B-it-Q4_0.gguf` | Q4_0 | ~4.8 GB | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | per-layer embeddings |
| Gemma 4 12B | `gemma-4-12b-it-Q4_0.gguf` | Q4_0 | ~6.7 GB | [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | shared K/V on global layers |
| Gemma 4 31B | `gemma-4-31B-it-Q4_0.gguf` | Q4_0 | ~17.3 GB | [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) | shared K/V on global layers |

The benchmarks in [`benchmark-cpu.md`](benchmark-cpu.md) use E2B. E4B, 12B and
31B share the same `gemma4` implementation and differ in per-layer embeddings
(E2B/E4B) and shared K/V on the global layers (12B/31B).

Gemma 4 is the only architecture with a native tool-call dialect. It supports
the OpenAI `tools` flow over HTTP, including streaming `delta.tool_calls`.

## Qwen 3

The `qwen3` implementation targets **dense** Qwen 3 models:

| Model | Example GGUF | Quant | GGUF size | Source |
| --- | --- | --- | --- | --- |
| Qwen 3 4B | `Qwen3-4B-Q4_1.gguf` | Q4_1 | ~2.6 GB | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Qwen 3 8B | `Qwen3-8B-Q4_1.gguf` | Q4_1 | ~5.2 GB | [unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF) |
| Qwen 3 14B | `Qwen3-14B-Q4_1.gguf` | Q4_1 | ~9.4 GB | [unsloth/Qwen3-14B-GGUF](https://huggingface.co/unsloth/Qwen3-14B-GGUF) |

Qwen 3 uses ChatML and supports thinking, but it has no tool dialect yet: a
request with `tools` against a Qwen 3 image returns
`UnsupportedArchitecture`. Other dense Qwen 3 sizes should follow the same
`qwen3.*` GGUF metadata; `zlm image compile --check` prints the detected
architecture before writing anything.

## Quantization support

The runtime supports the following GGML tensor types. Use a GGUF whose weight
tensors are one of these types; other types are not decoded by the CPU kernels.

| GGML type | CPU | Vulkan matvec | Vulkan GEMM / fused |
| --- | --- | --- | --- |
| `F32` | yes | yes | no |
| `F16` | yes | yes | no |
| `BF16` | yes | yes | no |
| `Q8_0` | yes | yes | yes (GEMM; fused gate/up/MLP) |
| `Q4_0` | yes | yes | yes (GEMM; fused gate/up/MLP) |
| `Q4_1` | yes | yes | yes (GEMM; fused gate/up/MLP) |
| `Q4_K` | yes | yes | yes (GEMM only; no fused MLP shader) |
| `Q6_K` | yes | yes | no (no GEMM shader) |
| `Q5_K` | yes | no | no (CPU only) |

`Q4_0` (Gemma 4) and `Q4_1` (Qwen 3) are the recommended starting points. The
GPU column is a best-effort acceleration path: the tuner decides per op whether
the GPU actually pays, and an unsupported type simply stays on the CPU.

## Getting a GGUF

zlm does not distribute model weights. The example files above are Unsloth GGUF
conversions of the base models; the tables link the Hugging Face repository for
each size. The Gemma 4 E2B conversion is based on
[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), and the
Qwen 3 4B conversion is based on
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

Minimal path with the Hugging Face CLI (download into `models/`, which is
ignored by Git):

```bash
mkdir -p models

# Gemma 4 E2B, Q4_0.
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  --include 'gemma-4-E2B-it-Q4_0.gguf' --local-dir models

# Qwen 3 4B, Q4_1.
huggingface-cli download unsloth/Qwen3-4B-GGUF \
  --include 'Qwen3-4B-Q4_1.gguf' --local-dir models
```

The downloaded names match the examples below. If you already have a GGUF with
a different name, pass its path to `zlm image compile` instead. Other sizes are
in the repositories linked in the tables above; pick a quant from the support
table.

Model weights keep their own licenses. Check the model card before using or
redistributing a GGUF (Gemma models use the Gemma Terms of Use; Qwen 3 models
are published under Apache-2.0, but always confirm the specific release).

## Compile, tune, run

```bash
# Gemma 4 example (CPU; add --gpu --vram <mb|gb> to tune for Vulkan).
zlm image compile gemma4-e2b models/gemma-4-E2B-it-Q4_0.gguf
zlm tune gemma4-e2b e2b --threads 8 --ram 8gb
zlm run e2b --soul "You are a concise assistant." --prompt "Hello"

# Qwen 3 example.
zlm image compile qwen3-4b models/Qwen3-4B-Q4_1.gguf
zlm tune qwen3-4b qwen --threads 8 --ram 8gb
zlm run qwen --soul "You are a concise assistant." --prompt "Hello"
```

`image compile` is deterministic and hardware-independent, so an image can be
built once and reused across machines. `tune` is the per-machine step: it
records the plan that matches the current CPU (and GPU, if requested).

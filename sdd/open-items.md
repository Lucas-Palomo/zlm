# Open Items / Known Issues

Deferred work, each linked to the task that raised it. Promote to a task spec
when picked up.

## engine

### Composer recipes beyond gemma4's set
from: sdd/tasks/runtime/09-deterministic-tuning.md
reason: the applicable gemma4 fusions are implemented (`qk_norm_rope`,
`gate_up_fused`). The rest need more: `norm_proj` needs a fused `qkv` (the attn
norm feeds q/k/v — multi-consumer); `add_norm` needs a two-output kernel
(residual + normed); `rope_write_kv` competes with `qk_norm_rope` (one wins);
`matvec_bias`/`dequant_scale` are unused in gemma4 (no biases; embed scale already
folded into the lookup dispatch).

### Plan reuse is keyed by config, not image content
from: sdd/tasks/gpu/12-spec-constant-grid-all-quants.md
reason: `tune` reuses a `.zplan` by `(image name, config fingerprint)`, which
does not fold in the image's content hash. Replacing an image's content under the
same name (recompiling `qwen3-14b` from a Q8_0 to a Q4_1 GGUF) silently reuses
the stale plan, so the dispatch tensors no longer match the image and `run`
fails. Fix: fold the image content hash (the `.zhash`) into the plan fingerprint,
or validate the plan against the image at load.

## kernels

### Prefill policy is footprint-fit, not cost-based
from: sdd/tasks/runtime/10-batched-prefill.md
reason: `choosePrefillPolicy` picks the largest block in `{2,4,8,16}` that fits
`ram/8`, so it almost always chooses `batch` (the `token` branch is effectively
dead). A real cost/benefit rule (GEMM vs GEMV roofline) would decide block size
and when token wins.

### Matvec ~10% below llama.cpp (with enough RAM)
from: sdd/tasks/gpu/11-device-scheduler.md
reason: on the gemma 12B Q4_0 at 8 threads the zlm CPU-only decode is 4.1-4.4
tok/s vs llama.cpp's 4.7 (`llama-bench`; its build has asserts enabled, so the
figure may be conservative). Most of the earlier 3.6-vs-4.7 gap was the **RAM
budget**, not the kernels: at `--ram 4gb` the 6.4 GB of weights do not fit and
stream from the disk (the window is the bottleneck). The residual ~7-13% is
kernel/variant tuning (the variant choice or the thread split at this shape),
not architecture.

## runtime

### Sub-ranges for tensors with differing liveness
from: sdd/tasks/runtime/06-residency-and-budget.md
reason: ranges are still whole tensors. The `row_cache` directive (plan-decided
budget, row-LRU for `token_embd`/`per_layer_token_embd`) is resolved and those
tables are no longer pinned whole; what remains is partitioning a tensor into
sub-ranges when its liveness differs (value-based admission beyond
HOT-smallest-first).

### Cost pipeline: Evaluator (bounded search)
from: sdd/tasks/runtime/12-cost-pipeline.md
reason: the plan-level Evaluator is not built. The plan-level knobs are limited —
the prompt length is unknown at tune (so the prefill policy cannot be scored per
prompt) and the RAM window's max is trivially optimal. A meaningful Evaluator
needs a richer candidate space (e.g. per-op variant x parts) scored by
`planScore`. `runtime/12` built the three stages + the measured scaling curve;
the search is deferred.

### Cost pipeline: decision stumps + threshold search
from: sdd/tasks/runtime/12-cost-pipeline.md
reason: the empirical decision layer is not built. Fitting per-variant thresholds
(u4 vs t2/blocked, the scaling knee) needs a per-variant efficiency measurement
in the calibration — the concrete mechanism for `kernels/07`.

### Cost pipeline: the four ★ algorithms
from: sdd/tasks/runtime/12-cost-pipeline.md
reason: Critical Path (serial fraction from the `deps` DAG), I/O Complexity (a
cache floor; `bytesEffective` already approximates it), Space-filling design
(calibration sampling) and Little's Law (formalize the prefetch `lookahead` as
`in_flight = bw x latency`) are not implemented. Self-contained but marginal
until the kernel scaling (`kernels/08`) and the variant measurement land.

### Tool rounds are not persisted in chat sessions
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: `$ZLM_HOME/sessions` records are still `role` + `text`. An HTTP request
with the `session` field and tools loses the assistant `tool_calls` (and the
`tool_call_id` pairing) on the next load, so the round cannot be replayed. The
runtime has the structured `Turn.calls`/`tool_call_id`; only the session file
format and its loader need the fields.

### Qwen 3 has no tool dialect
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: `Arch.tools` is declared only by gemma4; a request with `tools` against a
qwen3 image returns `UnsupportedArchitecture`. Qwen 3 uses its own
`<tool_call>` JSON format and would need a second `Dialect` implementation.

### Tool-enabled SSE is buffered
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: the raw `<|tool_call>` block must not reach the client as content, so a
request with `tools` generates the whole reply before emitting the SSE content
and `delta.tool_calls` frames. Live token streaming is lost only on that path;
the no-tools path still streams per token.

### The HTTP edge is serial and does not cancel abandoned generations
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: `serve` accepts one connection at a time and `handle` blocks for the
whole prefill/decode, so a second request (OpenCode's title turn) waits behind
the first and `GET /v1/models` stalls during a generation. A client disconnect
is only noticed on the next write; a buffered (tools) or non-stream reply keeps
computing after the client is gone. Fixing this needs a per-model work queue and
a cancellation token, not a config change.



## gpu

### GPU duty cycle bounds the RX 580 memory clock (resolved, residual remains)
from: sdd/tasks/gpu/06-vram-budget-saturation.md
reason: the Polaris memory clock only ramps (`3D_FULL_SCREEN`,
`MCLK_ACTIVE_LEVEL=25`) when memory activity is sustained above ~25%. Our decode
was bursty and offloaded only a small fraction, so the duty cycle was ~5% and
the clock stayed at 400 MHz — not a DPM lock (a game holds 1750 MHz).
**Resolved for the fused FFN:** offloading the fused `gate_up` (`gpu/07`) raised
the duty cycle enough that the clock ramps to 1750 MHz during decode (E2B 20.1
vs CPU 16.2 tok/s; 12B 5.0 vs 4.1). **Resolved for the secondary quants at a
non-saturated budget:** the `q4_k`/`f32`/`bf16` projections (`gpu/07` secondary)
are offloaded; E2B at 4 GB goes 16.7 -> 20.9 tok/s (CPU 16.3). **Residual:** the
12B at 4 GB is budget-saturated, so the 540 MB `q4_k` lm_head displaces ~20
per-token `q4_0` weights and is decode-neutral (its GPU GEMV measures ~40 GB/s,
~break-even with the CPU's ~38 GB/s because the packed 6-bit scale decode is
ALU-bound). The remaining CPU-side decode ops (norms, attention, elementwise)
still bound the duty cycle; the next levers are a faster `q4_k` shader
(vectorized nibble/scale decode) and/or a warm device state.

### GPU selection stays split from the Selector
from: sdd/tasks/gpu/08-co-processor-selector.md
reason: the GPU AtomicOps are catalogued (`tier = .gpu`), but the variant choice
(the device probe) and the placement (`decideOffload`) stay outside the Selector:
`ApplicableFn`/`CostFn` are pure shape functions with no device/residency state.
A single Selector scoring CPU+GPU by cost needs those signatures widened. A new
**quant** additionally needs a `GpuCoeffs` field, the tuner's quant->variant
switch and the `decideOffload`/`eligibleResident` gate.

### Device scheduler: the split is a capability; the VRAM resident set is the lever
from: sdd/tasks/gpu/11-device-scheduler.md
reason: the chain scheduler's model (a Viterbi over `cpu | gpu | split`), the
RAM/VRAM policy with shadow prices and the **`split_efficiency`** probe are built;
a split is priced `bytes / (gpu_rate * split_efficiency)` and the probe measures
**0.70** on the RX 580 (the memories contend), so the tuner keeps every op whole
on the GPU. The split is a **capability** — a machine where the bands add (> 1)
would split with no code change. Two items remain unbuilt and are **off the
critical path**: the measured `hop_ns` dispatch-setup constant, and the `runs`
telemetry (`profile:` reports whole-gpu/split/cpu-only, not the runs). The
performance lever is the **VRAM resident set** (`gpu/06`): the 12B at 8 threads
goes **5.8 tok/s at VRAM 4 GB** to **8.7 at VRAM 8 GB** (2.4x the CPU-only 3.6),
with the RAM as **free context** (49856 tokens at 8.7).

### `gate_up` pairs of quants without a fused shader fill the VRAM resident set
from: sdd/tasks/gpu/06-vram-budget-saturation.md
reason: `addResidentPairCandidate` makes a fused `gate_up` pair resident for any
`eligibleResident` quant, but only q4_0 and q8_0 have a fused `gate_up` shader
(`decideOffloadGateUp`), so a q4_1/q6_k pair is resident yet never offloaded in
decode — dead weight that also keeps the weights out of the RAM window. The Q8_0
case is **fixed**: a q8_0 fused `gate_up` shader now ships (`gpu/07`), so on
`qwen3-14b` the pair offloads (decode offloads 6 -> 46, context 4096 -> 13600).
Fix for the rest: skip a pair whose quant the device has no fused shader for (or
add the shader).


### HTTP request headers and slow clients are not bounded
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: the body is now capped at 16 MiB, but headers are read into an 8 KiB
buffer and there is no per-connection read timeout; a slow or oversized request
can block the single-threaded edge indefinitely, and `Transfer-Encoding:
chunked` / `Expect: 100-continue` are not implemented. The server also has no
authentication, so a non-loopback `host` exposes it to the network as-is.

### `sessions.load` silently drops records above 1 MiB
from: sdd/tasks/runtime/08-service-and-states.md
reason: `readFile` uses a fixed 1 MiB stack buffer; a long conversation that
grows past it returns an error that `load` maps to `null`, so the session
disappears instead of failing loudly or streaming the JSON. The parser should
read dynamically with a configured cap, or use a file reader.

### `sessions.save` truncates the session file in place
from: sdd/tasks/runtime/08-service-and-states.md
reason: `createFile` + `writeStreamingAll` is not atomic. A crash or full disk
during the write leaves a truncated `.json`; the next load silently returns
`null`. Write to a temporary sibling and rename it.

### A client can request an unbounded answer budget
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: `max_tokens`/`max_completion_tokens` is clamped only by the plan's
sequence limit, so a client can ask for a very long CPU generation. The body cap
prevents oversized requests, not an oversized answer; cap the effective answer
budget (or expose a server-side ceiling) for a shared/remote server.

### SSE errors after the headers are not framed as a terminal chunk
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: a generation/extractor error after `HTTP/1.1 200` writes
`data: {"error":…}`, but no `finish_reason` frame or `data: [DONE]` follows.
Some OpenAI clients treat a stream without the terminal frame as a transport
error and retry; emit a terminal error frame (or close the stream explicitly)
on that path.

### HTTP config is not validated for duplicate/empty model ids
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: `config.parse` accepts duplicate `id`s, empty ids, and a `default` that
does not name a model. Requests then select the first match or fall back to an
invalid default (`400 unknown model`). Validate the table at boot and fail with
a clear message.

### Live SSE and buffered tool decoding differ on a stray `<|tool_call>`
from: sdd/tasks/runtime/17-openai-tool-calling.md
reason: the no-tools live sink uses `decodeTokenAppend`, which drops tool
markers; the buffered/tool path uses `decodeToolText`, which preserves them. A
model that emits a call without `tools` would produce different streamed and
saved text. Both paths should share the same tool-marker policy.

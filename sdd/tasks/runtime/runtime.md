# runtime

title: Runtime
reason: Execute a tuned image and decide nothing. The Session owns the plan,
        image, `zcom` data, KV and the tokenizer; the Executor runs each
        `Dispatch` from the plan. Status: the full gemma4 forward executes end
        to end and `run --prompt` prints generated text (greedy loop with
        advancing `pos`).

## Tasks

### [01 - Runtime Skeleton](01-runtime-skeleton.md)
Status: done

Context: `zlm load <alias>` is a bypass (no daemon/session); `zlm run <alias>
--prompt ... [--max-tokens ...]` reads `.zlm/runtimes/<alias>/plan.zplan`,
resolves the image, prints the config, and stops before generation
(`TODO: Not Implemented yet!`).

### [02 - Executor (minimal run)](02-executor.md)
Status: done

Context: `src/runtime/` (Session + Executor) executes the AOTKernel once — one
decode step: resolve the image/ZTEF, allocate buffers + KV, build each family's
args from the plan's bindings, call `kernels.registry.run`, `argmax` the logits.
No tokenizer/sampler/threads/GPU yet.

### [03 - Forward Execute](03-forward-execute.md)
Status: done

Context: execute the forward-complete plan — the Executor builds each op's
`Args` from bindings + params, uses the `zcom` tables and a `KvCache`, and
produces logits. No decisions at run.

### [04 - Tokenizer and Generation](04-tokenizer-generation.md)
Status: done

Context: read the tokenizer from `zmeta`, encode the prompt, run the
autoregressive loop (one forward per token with advancing `pos`), greedy sample,
and detokenize — so `run --prompt` prints text instead of a numeric id.

### [05 - Parallel Execution](05-parallel-execution.md)
Status: done

Context: make the plan's `parts` real. The tuner decides per-dispatch `parts`
(op shape + hardware budget + grain); the `Session` owns a worker pool; the
executor splits `matvec`/`lm_head` rows inside a dispatch while staying
sequential.

### [06 - Residency and Budget](06-residency-and-budget.md)
Status: done

Context: make `--ram` enforced around a first-class, AoT-compiled
`ResidencyGraph` (where/how much to allocate, pin, prefetch async, release,
evict; `sync|async`, tiers for future GPU). The Scheduler computes the budget
split (fixed/reserve/window, no fixed percentages) and the admission by value;
the runtime only executes the orders with byte accounting.

### [07 - Tuning Intelligence](07-tuning-intelligence.md)
Status: done

Context: two phases of measured decisions — Phase 1 picks kernel
implementations (a per-family benchmark harness; `matvec` `MatVecPlan`) and
derives `grain`/`parts`, plus a CPU affinity plan; Phase 2 sizes the memory
budget (cold I/O `read_cost`/`reserve`, value admission, KV/window inversion,
derived lookahead). Follow-up to `runtime/06`.

### [08 - Service and Residency States](08-service-and-states.md)
Status: done

Context: a per-alias background service holding the Session, over a Unix socket;
residency states (`cold`/`warm`/`hot`) with `--hot=minutes` (pin TTL) and
`--timeout=minutes` (auto-unload); `load` preloads it and `run` delegates
(auto-loading when offline, printing a notice); `image list` reports
online/offline; `unload` stops it. A minimal step toward the earlier local prototype's sub-daemon/session.

### [09 - Deterministic Tuning](09-deterministic-tuning.md)
Status: done

Context: replace timing-based autotune with a deterministic, capability-driven
loop — Profile (capabilities) -> Selector (ranked candidates) -> Composer
(recipes/form) -> Evaluator (roofline cost, bounded loop) -> `.zplan`, under the
`resident <= ram` and `parts + residency_workers <= threads` constraints. GPU
tiers modeled and pruned. Retires `bench.zig`/`autotune.zig` timing; the `.zat`
becomes a capability profile.

### [10 - Batched Prefill](10-batched-prefill.md)
Status: done

Context: prefill runs token-a-token today, so the `matmul`/GEMM family is
dormant. The **plan** declares the prefill policy (`token` | `batch(block)`,
chosen by the tuner via cost); the runtime batches the prompt into block forwards
(`executePrefill`) and obeys. Projects use `matmul` (prefill) vs `matvec`
(decode). The runtime still decides nothing.

### [11 - Performance](11-performance.md)
Status: done

Context: the current code is slower than the earlier local prototype in full CPU — a
per-request fixed CPU cost plus slow prefill GEMM and decode. Add a measurement
harness, eliminate the fixed cost, and optimize the kernels, keeping the plan
deterministic and "the runtime decides nothing".

### [12 - Cost Pipeline](12-cost-pipeline.md)
Status: done

Context: redesign the tuning cost into three stages — **AST static cost**
(hardware-independent, in the plan-compile) → **Roofline** → **Amdahl/Gustafson**
(decode/prefill) — feeding a bounded, deterministic **Evaluator** that scores
candidates and picks the plan. Replaces the single-pass `opCost`/`parEff`. The
stumps/Evaluator/★ remain in `open-items.md`; the GPU cost/ROI is `gpu/01`.

### [13 - Prefix Cache (K/V reuse across executions)](13-prefix-cache.md)
Status: done

Context: a **persistent, cross-execution prefix cache**: store the K/V of
prefixes already processed in a previous execution so a repeated prefix skips the
prefill. A **radix tree** over the causal token path (maximal common prefix),
keyed by a per-plan fingerprint `PrefixCacheKey` shared across runtimes of the
image with the same K/V plan (invariant to `--threads`/`--ram`/`--vram`), with a
canonical, `max_seq`-independent layout. The `.zpc` is disk/mmap-managed and
**persistent** (the daemon appends + compacts on idle, atomic flush), its ceiling
a slice of `planBudget`'s window. Also: content-hash shared-weight dedup (ZTEF format v2)
and prefill residency nodes. A HIT is exact. **SWA per-layer reuse is blocked**
(interleaved global layers). Out: `--kv`/`--gate`.

### [14 - Prefix-Cache Partial-Hit Correctness and Prompt-Format Invalidation](14-prefix-cache-partial-hit.md)
Status: done

Context: a partial prefix hit (a different prompt sharing the chat-template
prefix) loaded the wrong K/V and degenerated, because the canonical payload was
**buffer-major** while `loadFromCache` sliced it as if prefix-contiguous; and a
recompile that changed the prompt format did not change `content_root`, so the
old `.zpc` stayed valid. Fix: **position-major** canonical layout + a
`Key.format` fingerprint of the effective chat template; `.zpc` `VERSION` bump.

### [15 - Weight Ring (three-pillar residency buffer)](15-weight-ring.md)
Status: done

Context: replace the multi-layer residency (`W_hot`/`W_reserved`/arena/LRU) with
an explicit three-pillar ring (`compute | reserved | prefetch`) in consumption
order, opt-in on `--residency` + `--prefetch-threads`. On consume the pillars
rotate -- the `compute` slot is recycled into the next `prefetch` target (never
discarded) -- so the pipeline stays full while the workers work. Without the
flags every weight streams from the mmap (`b25630b`).

### [16 - Compressed Image (blosc2)](16-compressed-image-blosc2.md)
Status: done

Context: an opt-in compressed image -- `compile --blosc2=N` (comptime-gated on
`-Dblosc2`) writes the ZTEF payload compressed per tensor with Blosc2 (a
byte-shuffle by the quant block size + zstd), so a model that does not fit reads
fewer disk bytes. The weights are not re-quantized; the runtime decompresses
inline with a per-dispatch cache. **Discarded:** on the 72B the disk saving was
~7% and the decompression dominated (63-82 s/tok vs ~14 uncompressed), so the
flag and the code were removed.

### [17 - OpenAI Tool Calling (Gemma 4 native dialect)](17-openai-tool-calling.md)
Status: done

Context: the HTTP edge ignored OpenAI `tools` and echoed the model's raw call
text. A structured `Tool`/`Call`/`Dialect` layer (declared by the architecture)
now renders Gemma 4's `<|tool>`/`<|tool_call>`/`<|tool_response>` blocks, maps
`tool_calls` + `tool_call_id` both ways (JSON and SSE), and starts a fresh
`<|channel>thought` round after tool results.

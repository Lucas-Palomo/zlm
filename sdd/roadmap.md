# Roadmap

zlm — a CPU-first runtime for Language Models. Identity and rules:
`CONSTITUTION.md`; deferred work: `sdd/open-items.md`.

> **Historical measurements.** Some task specs compare the current code with an
> earlier local prototype. Those were local development/test builds, not
> published releases or external benchmarks. The numbers are historical
> measurements kept for design context; they are not reproducible external
> sources. `format vN` / `plan vN` labels are versions of zlm's own formats, not
> project releases.

## Phases

The path from a raw model to running tokens:

```text
compile   raw model -> image      deterministic, hardware-independent
tune      image -> plan           hardware-specific measurement
run       image + plan -> tokens  executes the plan, decides nothing
```

Dependencies: `compile -> tune -> run`.

## Platform

**POSIX-first:** Linux + macOS (same POSIX code). **Windows is out of scope** —
run under WSL or Docker. OS tuning (madvise/fadvise, huge pages, `io_uring`) is a
`tune` concern, with fallbacks and a **cgroup-aware** budget for containers.

## Epics

- [**engine**](tasks/engine/engine.md) — done
  - [01 - Native Format and Compile](tasks/engine/01-native-format-and-compile.md) — done
  - [02 - ZCOM (derived data)](tasks/engine/02-zcom.md) — done
  - [03 - Runtimes and Manifest](tasks/engine/03-runtimes-and-manifest.md) — done
- [**cli**](tasks/cli/cli.md) — done
  - [01 - CLI Bootstrap](tasks/cli/01-cli-bootstrap.md) — done
  - [02 - Tune Skeleton](tasks/cli/02-tune-skeleton.md) — done
  - [03 - Improve Flags](tasks/cli/03-improve-flags.md) — done
  - [04 - Inspect and Metrics](tasks/cli/04-inspect-and-metrics.md) — done
  - [05 - Execution Profile (`run --profile`)](tasks/cli/05-profile.md) — done
  - [06 - Streaming Generation (`run --stream`)](tasks/cli/06-stream.md) — done
  - [07 - Think Suppression (`--show-think`) and Verbose Levels](tasks/cli/07-verbose-and-show-think.md) — done
  - [08 - Prefetch Threads (`--prefetch-threads`)](tasks/cli/08-prefetch-threads.md) — done
  - [09 - Residency Ceiling (`--residency`)](tasks/cli/09-residency.md) — done
- [**plan**](tasks/plan/plan.md) — done
  - [01 - Plan Format](tasks/plan/01-plan-format.md) — done
  - [02 - Compile the Block](tasks/plan/02-compile-block.md) — done
  - [03 - Forward Data](tasks/plan/03-forward-data.md) — done
  - [04 - Per-Layer Dimensions](tasks/plan/04-layer-dim-dispatch.md) — done
- [**runtime**](tasks/runtime/runtime.md) — done
  - [01 - Runtime Skeleton](tasks/runtime/01-runtime-skeleton.md) — done
  - [02 - Executor (minimal run)](tasks/runtime/02-executor.md) — done
  - [03 - Forward Execute](tasks/runtime/03-forward-execute.md) — done
  - [04 - Tokenizer and Generation](tasks/runtime/04-tokenizer-generation.md) — done
  - [05 - Parallel Execution](tasks/runtime/05-parallel-execution.md) — done
  - [06 - Residency and Budget](tasks/runtime/06-residency-and-budget.md) — done
  - [07 - Tuning Intelligence](tasks/runtime/07-tuning-intelligence.md) — done
  - [08 - Service and Residency States](tasks/runtime/08-service-and-states.md) — done
  - [09 - Deterministic Tuning](tasks/runtime/09-deterministic-tuning.md) — done
  - [10 - Batched Prefill](tasks/runtime/10-batched-prefill.md) — done
  - [11 - Performance](tasks/runtime/11-performance.md) — done
  - [12 - Cost Pipeline](tasks/runtime/12-cost-pipeline.md) — done
  - [13 - Prefix Cache (K/V reuse across executions)](tasks/runtime/13-prefix-cache.md) — done
  - [14 - Prefix-Cache Partial-Hit Correctness and Prompt-Format Invalidation](tasks/runtime/14-prefix-cache-partial-hit.md) — done
  - [15 - Weight Ring (three-pillar residency buffer)](tasks/runtime/15-weight-ring.md) — done
  - [16 - Compressed Image (blosc2)](tasks/runtime/16-compressed-image-blosc2.md) — done (discarded)
  - [17 - OpenAI Tool Calling (Gemma 4 dialect)](tasks/runtime/17-openai-tool-calling.md) — done
- [**kernels**](tasks/kernels/kernels.md) — done
  - [01 - KernelFactory and Registry](tasks/kernels/01-kernel-factory.md) — done
  - [02 - Kernel API Simplification](tasks/kernels/02-kernel-api.md) — done
  - [03 - Kernels Layout](tasks/kernels/03-kernels-layout.md) — done
  - [04 - Candidates and Recipes](tasks/kernels/04-candidates-and-recipes.md) — done
  - [05 - GEMM and Cost-Model Selection](tasks/kernels/05-matmul-and-selection.md) — done
  - [06 - Family-Agnostic Selector](tasks/kernels/06-family-agnostic-selector.md) — done
  - [07 - Variant Cost Model](tasks/kernels/07-variant-cost-model.md) — done
  - [08 - Parallel Matvec Efficiency](tasks/kernels/08-parallel-matvec.md) — done
  - [09 - Batched GEMV (prefill weight reuse)](tasks/kernels/09-batched-gemv.md) — done
  - [10 - Kernel Performance (generic: integer dot, register tiling, blocking)](tasks/kernels/10-kernel-performance.md) — done
  - [11 - Kernels and ISA (generic AtomicOps + compile-time width variants)](tasks/kernels/11-kernels-isa.md) — done
  - [12 - Generic Instruction Audit (P0/P1: matvec, dequant, reduce)](tasks/kernels/12-instruction-audit.md) — done
  - [13 - Comptime Kernel Variants (PoC: q4_0)](tasks/kernels/13-comptime-kernel-variants.md) — done
  - [14 - q4_0 Catalog: Naming + Extension](tasks/kernels/14-q4-0-catalog-naming.md) — done
  - [15 - q4_0 Catalog: Dynamic Grid + Pruning](tasks/kernels/15-q4-0-dynamic-grid.md) — done
  - [16 - q4_k Catalog: Dynamic Grid](tasks/kernels/16-q4k-dynamic-grid.md) — done
  - [17 - Generic Kernel (weight-family x dot x sweep)](tasks/kernels/17-generic-kernel.md) — done
  - [18 - matvec Structure (layout x kernel x form)](tasks/kernels/18-matvec-structure.md) — done
  - [19 - Test Layout](tasks/kernels/19-test-layout.md) — done
  - [20 - F/BF Families](tasks/kernels/20-f-bf-families.md) — done
  - [21 - Generic Dots](tasks/kernels/21-generic-dots.md) — done
- [**gpu**](tasks/gpu/gpu.md) — done
  - [01 - GPU Co-processor](tasks/gpu/01-coprocessor.md) — done
  - [02 - Co-processor Implementation (as-built + pending)](tasks/gpu/02-co-processor-implementation.md) — done
  - [03 - Async Dispatch and Dual-Stream](tasks/gpu/03-async-and-dual-stream.md) — done
  - [04 - Heterogeneous Compute + GPU-aware RAM Budget](tasks/gpu/04-heterogeneous-compute.md) — done
  - [05 - VRAM Extended by RAM (spillover / virtual_vram)](tasks/gpu/05-vram-spillover.md) — done (closed; ROI negative)
  - [06 - VRAM Budget Saturation](tasks/gpu/06-vram-budget-saturation.md) — done
  - [07 - Fused FFN gate_up Offload](tasks/gpu/07-fused-ffn-offload.md) — done
  - [08 - Co-processor Selector](tasks/gpu/08-co-processor-selector.md) — done
  - [09 - GPU Sub-calculator + Variant Axes](tasks/gpu/09-gpu-sub-calculator.md) — done
  - [10 - gpu Structure (backend-ready)](tasks/gpu/10-gpu-structure.md) — done
  - [11 - Device Scheduler (heterogeneous chain scheduling)](tasks/gpu/11-device-scheduler.md) — done (the split is a capability; the VRAM resident set is the lever)
  - [12 - Variant Grid for All GEMV Quants (spec constants)](tasks/gpu/12-spec-constant-grid-all-quants.md) — done
  - [13 - Heavy-duty GPU Warm-up (sustained clock ramp)](tasks/gpu/13-heavy-duty-gpu-warmup.md) — done
  - [14 - Fused SwiGLU MLP (gate_up + activation + down in one dispatch)](tasks/gpu/14-fused-mlp-swiglu.md) — done
- [**architectures**](tasks/architectures/architectures.md) — done
  - [01 - gemma4 Declaration](tasks/architectures/01-gemma4-declaration.md) — done
  - [02 - Semantic Graph and gemma4 Block](tasks/architectures/02-semantic-graph.md) — done
  - [03 - Forward Block](tasks/architectures/03-forward-block.md) — done
  - [04 - Gemma4 Forward Correctness](tasks/architectures/04-gemma4-forward-correctness.md) — done
  - [05 - Per-Layer Embeddings](tasks/architectures/05-per-layer-embeddings.md) — done
  - [06 - Gemma4 SWA K/V (larger variants)](tasks/architectures/06-gemma4-swa-kv.md) — done
  - [07 - gemma4 Forward NaN (attention / KV)](tasks/architectures/07-gemma4-forward-nan.md) — done
  - [08 - gemma4 Chat Template (resolved at compile, stored in zmeta)](tasks/architectures/08-gemma4-chat-template.md) — done
  - [09 - Think Mode Flag (run --think)](tasks/architectures/09-think-mode.md) — done
  - [10 - Qwen3 (declaration + byte-level tokenizer)](tasks/architectures/10-qwen3.md) — done
- **product** — planned

## Future work / experimentation (no spec yet)

Ideas recorded for later, not part of the active plan. Promote to a task spec
when picked up.

### Macro

The big, cross-cutting bets.

- **GPU accelerator.** Offload dispatches to a local GPU co-processor: the plan
  records a per-dispatch `device` + `tile`, the tuner sets them from the cost/ROI
  model, and `--vram` enters `planBudget` (headroom + knapsack + spillover). Spec:
  `gpu/01` (draft).
- **Speculative decoding (decode "turbo").** A draft model proposes K tokens; the
  target verifies them in one batched forward (the `kernels/09` batched GEMV
  already enables multi-token forwards) and accepts the greedy-matching prefix —
  exact for greedy (same output, faster). Amortizes the per-token model read by
  ~K. Variants: Medusa/EAGLE (self-speculation), lookahead decoding.
- **Remote accelerator.** Offload dispatches to a **remote host / remote OS** over
  the network to accelerate processing — a distributed co-processor. The plan
  records a remote `device` per dispatch (cf. `gpu/01`); the runtime ships the
  activations/weights and accounts for the transfer in the budget. Needs a
  transport, a serialization format and a trust/security model.

### Smaller

Focused improvements and feature proposals.

- **Model optimization (GGUF/encoder).** Quantization/optimization of the
  compiled image — e.g. denser weight formats (q3/q2) to reduce bytes moved, and
  a low-bit KV (int8) with a quality gate. Cuts both the streaming traffic and the
  RAM footprint.
- **KV precision control (`tune --kv=f16|f32|i8|i4`).** `f16` default. The image
  carries a `kv_type` (compile writes it — a GGUF hint if present, else `f16`);
  `--kv` overrides; the plan bakes it (and it enters the fingerprint). `i8` needs
  per-head scales, `i4` per-block; the budget accounts for the KV size per
  precision. f32 is the parity baseline. (KV is currently f16;
  `docs/kernel-architecture.md`.)
- **KV quality gate (`tune --gate`).** A deterministic check run at `tune`: decode
  a fixed reference prompt greedily with the candidate KV precision vs the `f32`
  baseline and require token equivalence (or a logit/KL tolerance); on failure
  fall back to a higher precision. Lets `i8`/`i4` be validated without a runtime
  decision.
- **`verify --fast`.** A lighter `verify` mode that skips the full per-file sha
  recomputation (validate the `zhash` self-checksum and the manifest, not every
  byte).
- **`zlm state` (live residency control).** `load` already sets an initial state
  (cold/warm/hot) with `--hot=minutes` (pin TTL) and `--timeout=minutes`; a live
  `zlm state <alias> [--cold|--warm|--hot=minutes] [--timeout=minutes]` to
  change/report a running service is future work. (New CLI surface — ask first.)
- **Benchmark/selection beyond matvec.** Extend the tuning harness from the
  `linear` `matvec` family to `matmul` (batched prefill; not exercised yet) and to
  `quant` if it gains interchangeable variants.
- **Prefix cache: per-layer reuse on SWA layers.** **Blocked on correctness.**
  The premise "an SWA layer's K/V depends only on the local window" is false for
  gemma4's interleaved pattern: the global layers give later sliding layers a
  full-context receptive field, so a middle mutation changes its K/V bytes and a
  reuse would be a wrong HIT. Needs a provable local receptive field (an
  all-sliding leading run, or a per-layer dataflow analysis).

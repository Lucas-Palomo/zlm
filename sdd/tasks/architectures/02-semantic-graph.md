02 - Semantic Graph and gemma4 Block

Status: done
Commits:

## Goal

Make the forward *be the plan*. gemma4 declares a **semantic instruction
graph** (abstract ops over roles and activation slots); the tuner compiles it
into the **real directed plan** (`Dispatch[]` with bindings + deps + buffer
table); the runtime executes it. No forward code in the model.

```text
SEMANTIC   gemma4 emits an op-instruction graph   (abstract: op/role/slots)
    │
PLAN       tuner compiles it into a directed graph (Dispatch[] + bindings +
    │       deps + buffer table + ResidencyGraph)         = AOTKernel
    │
EXECUTE    runtime executes (Executor: resolve binding -> kernel.run)
```

## Scope

- **In:** semantic instruction types; gemma4 block declaration; `Dispatch`
  bindings (tensor | buffer) + deps; the buffer table; a minimal
  `tuner/scheduler.zig`; `runtime/engine.zig`.
- **Out:** `Selector`/`Composer` recipes (atomic_id stays 0); GPU; prefill
  batching; KV/sampler beyond what decode needs; qwen3.

## Design

- **Semantic instruction** (abstract): `op`, `role` (to find the tensor),
  `in`/`out` activation **slots**, params (eps, dims, pos).
- **List, not DAG (first cut).** gemma4 declares the ordered sequence
  (embed -> per-block -> final); deps are implicit (in -> out) and materialized
  by the scheduler.
- **`Dispatch` binding** is either a **tensor** (ZTEF index) or an activation
  **buffer** (session slot). Deps are indices into `Dispatch[]`.
- **Buffer table:** named slots (hidden, normed, q, k, v, attn_out, ffn_a,
  ffn_b, ffn_mid, logits, ...) with sizes, emitted into the plan; the runtime
  allocates them in the session.
- **Scheduler (minimal):** walk the semantic list, resolve role->tensor and
  slots->buffers, emit `Dispatch[]` (atomic_id = 0), append the ResidencyGraph.
- **Engine:** iterate `Dispatch[]`; resolve the binding (tensor view from ztef,
  or buffer slice); build `Args`; call `kernels.run(d, args)`.

## Arch Provenance

- **own:** the semantic-instruction layer, the binding model, the buffer table.
- **spec:** GGUF/ZTEF for tensors; gemma4 config for shapes.
- **consulted:** llama.cpp/ggml gemma4 layout — reference only.
- **derived:** Dispatch/ResidencyGraph from the earlier plan work.

## Steps

0. **Cleanup (done).** Split gemma4 responsibilities: `config.zig` -> params only
   (342 -> 145 lines); `names.zig` -> `parseTensorName`/`formatLayeredName`;
   `model/meta.zig` -> shared `LayerArray`/`BoolArray`/readers; `model/graph.zig`
   -> semantic types. gemma4 uses the shared role space.
1. Semantic instruction types in `architectures/model/graph.zig`
   (`Instr`, `Slot`, `Params`, `Block`) + gemma4 `block.zig` emitting the
   sequence. **Done.**
2. `plan/`: `Dispatch.binding` (tensor | buffer) + `dep`; buffer table.
   **Done** (plan: `Binding`, `Buffer`, `Dispatch.in/out/dep/sem_op`).
3. `tuner/scheduler.zig`: semantic list -> `Dispatch[]` + buffers + graph.
4. `runtime/engine.zig`: execute the plan (resolve binding -> `kernel.run`).
5. Wire `tune` to the scheduler; `run` to the engine.

## Flow / Arch

```text
gemma4.block (semantic: op/role/slots)
  -> tuner/scheduler: resolve role->tensor, slots->buffers, deps
  -> AOTKernel (Dispatch[] + buffers + ResidencyGraph)
  -> runtime/engine: binding -> view/slice -> kernels.run
```

## Definition of Done

- [x] gemma4 declares a semantic instruction sequence (`block.zig`).
- [x] `Dispatch` carries a binding (tensor | buffer) + deps; the plan has a
      buffer table.
- [x] The compiler (`tuner/compile.zig`, as-built; named `scheduler` here)
      compiles the semantic list into `Dispatch[]`.
- [x] The Executor (`runtime/executor.zig`, as-built; named `engine` here)
      executes every dispatch sequentially.
- [x] `zlm tune` emits the full plan; `zlm run` executes it (`token:` output).
- [x] `zig build` / `zig build test` green.

## Comments

- Recipes (Composer) and per-context Selector are a later task; here
  `atomic_id = 0`.
- **Generalization stops here.** `model/names.zig` (table-driven) and
  `model/meta.zig` are the molds with evidence. `arch`/`config`/`block` stay
  model-specific until qwen3 gives a second concrete case (rule of three) —
  avoid over-abstraction on a single implementation.
- The buffer table + bindings are the missing link that lets matvec/norm/rope/
  attention run (they need activation memory).

02 - Executor (minimal run)

Status: done
Commits:

## Goal

Make `zlm run` actually execute the tuned `AOTKernel` once — one decode step —
and print the next token id. No tokenizer, sampler, threading or GPU yet; the
point is to prove the `image + plan -> kernels` path works end to end.

## Scope

- **In:** `src/runtime/` (Runtime/Session + Executor); resolving the image from
  the plan (`store.openImage` + ZTEF mmap); building `TensorView` from the
  image; allocating the session buffers (`plan.buffers`) and the KV cache;
  building each family's `Args` from the `Dispatch` bindings; the sequential
  Executor calling `kernels.registry.run`; `run.zig` calling the session API.
  Decode of **one** token, `argmax` as the "sampler".
- **Out:** tokenizer / chat template / detokenizer, real sampling, threads
  (`parts`), GPU, `ResidencyGraph`/`ShadowGraph` paging, sliding-window /
  KV-sharing / PLE, `softcap`/residual correctness.

## Design

Layers: `run.zig` (edge) talks only to the session-level API; the Executor is
sequential and decides nothing (CONSTITUTION "Layers and boundaries").

```text
run.zig  ->  runtime.Session.step()  ->  Executor: for d in plan.dispatch
                                          build args(d) -> registry.run(op, atomic_id, args)
```

- **`Runtime`/`Session`** owns the plan (`plan.AOTKernel`), the resolved image
  readers, the activation buffers and the KV cache. `Session.open(...)` reads
  the plan, resolves the image, mmaps the ZTEF files, allocates buffers + KV.
- **`Executor`** is a sequential loop: for each `Dispatch`, resolve the weight
  `TensorView` (tensor binding) and the `x`/`out` buffer slices, build the
  family's typed `Args`, and call `kernels.registry.run(op, atomic_id, args)`.
- **TensorView** comes from the image: `plan.Tensor` -> ZTEF entry
  (`dims`, `ggml_type`) -> `TensorView{ mapped_slice = ztef.slice(e), ... }`.
  The ZTEF does not carry `layout`; for this task assume `.native_quant` and
  keep scale-split out (open item).
- **Buffers/KV**: activation buffers from `plan.buffers`; KV cache sized from
  the arch config (`head_count_kv`, `head_dim`, `context_length`).
- **One token**: `head` (embd_lookup) -> `layer` x `block_count` -> `tail`
  (final norm, lm_head) -> `argmax(logits)` -> print id. `--prompt` and
  `--max-tokens` are accepted but ignored (no tokenizer yet).

## Arch Provenance

- **own:** the session-level API, the Executor shape, the minimal decode slice.
- **spec:** the `.zplan` (tune) and ZTEF (image) layouts.
- **consulted:** —.
- **derived:** from `plan.Reader`, `image/ztef.Reader`, `data/tensor_view`.

## Steps

1. **(as built)** `src/runtime/`: `root.zig` (facade), `session.zig`,
   `executor.zig`; wired into `main.zig`.
2. **(as built)** `Session.open`: read plan, open the image's ZTEF readers +
   `zcom`, allocate activation buffers from `plan.buffers`.
3. **(as built)** `Executor.execute`: per `Dispatch`, resolve the weight
   `TensorView`, build the family `Args`, call
   `registry.run(op, atomic_id, args)`. `run.zig` calls `Session.step`
   (`argmax` of the logits buffer).
4. **(as built)** Minimal decode skips `rope`/`attention`/`write_kv` (no KV/head
   context yet).
5. `zig build` / `zig build test` green; `zlm run rt` executes a real E2B plan
   end to end (returns a token; not yet a correct forward — see Comments).

## Flow / Arch

```text
run <alias> --prompt P
  -> Session.open(runtime_dir, image_dir)
       plan.Reader.open ; manifest ; ztef.Reader.open(kind)* ; alloc buffers+KV
  -> Executor.run(plan.dispatch)
       for d: sense TensorView + buffers -> registry.run(d.op, d.atomic_id, args)
  -> argmax(logits) -> print token id
```

## Definition of Done

- [x] `src/runtime/` exists (`root.zig`, `session.zig`, `executor.zig`).
- [x] `zlm run` executes a **real** plan end to end and prints a token.
- [x] `Session.open` resolves the image (ZTEF + `zcom`) and allocates buffers.
- [x] `Executor` dispatches `dequant`/`matvec`/`norm`/`activation` via the
      registry.
- [x] No tokenizer/sampler/threads/GPU (deferred).
- [x] `zig build` / `zig build test` green.

## Comments

- **Blocked by tune:** today `tune` emits one `Dispatch` per weight tensor
  (`atomic_id = 0`, no order, no `in`/`out`, empty `buffers`) — not a forward.
  This task can proceed with a **synthetic** plan in tests; a real `run` needs
  `plan/02 - Compile the Block`, which produces the executable plan this
  Executor consumes.
- **Open:** where `layout` (native vs scale-split) comes from for `TensorView`.
- **Not a correct forward yet:** attention/`write_kv` are skipped, activations
  are treated as hidden-space, and there is no tokenizer/KV; the token is a
  placeholder. Correctness needs the remaining ops + KV + a complete block.
- **Deferred:** softcap/residuals make logits wrong; not needed to prove path.

02 - Compile the Block

Status: done
Commits:

## Goal

Turn the architecture's declared forward (`arch.block`) into an **executable**
`Dispatch[]`: ordered by the forward, with resolved bindings (weight tensor +
activation buffers), a chosen `atomic_id`, a `phase` and `parts`. This is the
"Selector + Scheduler" cut of the Tuner (declaration -> plan), replacing the
`01` stopgap that emitted one `Dispatch` per weight tensor.

## Scope

- **In:** compiling `arch.block` (`head` + `layer` x `block_count` + `tail`) into
  `Dispatch[]`; mapping `graph.Op` -> `kernels.Op`; choosing `atomic_id`; solving
  `role` -> tensor index and `in`/`out` -> buffer bindings; declaring
  `plan.buffers` from `block.slots`; setting `phase`/`parts`; the runtime store
  keeps its layout.
- **Out:** measured autotuning; multi-plan/fingerprint keying; real GEMM/prefill;
  GPU; residency policy. Also out: changing the `.zplan` format (see Comments).

## Design

- **The block is the program; the plan is its encoding.** `tune` reads
  `arch.block` and walks the instructions in order; each `Instr` becomes one
  `Dispatch` (or a small sequence, e.g. attention = write_kv + attn).
- **Op mapping (Selector, minimal):** `graph.Op -> kernels.Op` +
  `atomic_id`. Bootstrap rules:
  - `embd_lookup`/`dequant` -> `dequant`;
  - `matvec` + `lm_head` -> `matvec` (`atomic_id` from `role`/`op`: `mat_vec`
    vs `lm_head`; `mat_mul` reserved for prefill);
  - `norm` -> `norm` (`rms_norm`, or `rms_norm_in_place` for `in == out`);
  - `rope` -> `rope`; `write_kv`/`attention` -> `attention`;
  - `activation` -> `activation` (`gated_*` by `params.activation.kind`);
  - `softcap` -> (none yet; defer).
- **Bindings:** `Instr.role` -> tensor index (via `arch.parse` + inventory);
  `Instr.in`/`.out` -> `Binding{ .kind = .buffer, .ref = slot }` using `Slot`
  indices declared by `block.slots`. Fused ops use `ref2` for the second operand.
- **Buffers:** one `plan.Buffer` per `Slot`; byte size from the arch config
  (embedding/hidden/head/ffn dims) — filled by the Scheduler, not hardcoded.
- **Params:** resolve placeholders from `config` (`norm.eps` =
  `layer_norm_rms_epsilon`; `rope.{pos,n_rot,freq_base}` from `ropeBase`/
  `ropeRotDim`/`headDim`; `attention.sliding = isSliding(layer)`). The plan
  already stores `sem_op`; params may ride in `Args` at run time or be encoded
  per dispatch.
- **Phase / parts:** `phase = decode`; `parts = config.threads`.
- **KV:** `write_kv`/`attention` need a KV cache; the buffer slots for k/v are
  per-layer (or a single cache) — declared here so the runtime allocates.

## Arch Provenance

- **own:** the compile-the-block step, the minimal op/atomic mapping, the
  bindings-from-slots scheme.
- **spec:** the `.zplan` layout (`01`).
- **consulted:** —.
- **derived:** from `arch.block`, `graph.Op`/`Slot`, `kernels.Op`.

## Steps

1. **(as built)** `architectures/compile.zig`: `compileBlock(allocator, block,
   opts) -> { dispatch, buffers }`. Walks `head` -> `layer` x count -> `tail`;
   `kernelOp(graph.Op) -> kernels.Op`; `sem_op = Instr.params.code()`;
   `in`/`out` -> buffer `Binding` (slot ref); `role` -> tensor via the arch's
   name tables + a name lookup; `Buffer[]` from `opts.slot_bytes`.
2. **(as built)** `model/contract.zig`: `Arch` gains `globalName`/`layeredName`;
   gemma4 `arch.zig` wires them (`names.globalName`/`formatLayeredName`).
3. **(as built)** `graph.Params.code()` (stable `sem_op`); `plan.NONE`; a
   `parse`/`write`-safe try helper.
4. **(as built)** `tune.zig`: replaced the per-tensor walk with `compileBlock`
   (arch found via `architectures.root.find`, `TensorLut` name->index, layout
   mock kept). Prints `dispatch`/`buffers` counts.
5. **(as built)** `compileBlock` unit test: head + 2 layers + tail -> 6
   dispatches, forward order, buffer bindings, parts, buffers.
6. `zig build` / `zig build test` green.

### As-built deltas (deferred)

- ~~`layer_count` hardcoded `1`~~ **done**: `tune` loads the arch config via
  `zmeta` + `arch.loadConfig` and uses `block_count` (35 on E2B -> 598
  dispatches).
- ~~`slot_bytes` placeholders~~ **done**: sized from
  `embedding_length`/`head_count*key_length`/`feed_forward_length`.
- `atomic_id` uses the bootstrap map (`kernels` first AtomicOp per op);
  the real Selector refines by layout/params.
- `phase`/`dep`/KV cache are not emitted yet (attention runs are skipped by the
  runtime until then).

## Flow / Arch

```text
arch.block (head/layer/tail)
  -> Selector   : graph.Op -> kernels.Op + atomic_id
  -> Scheduler  : order + role->tensor + slot->buffer + params + phase + parts
  -> Dispatch[] + Buffer[] -> plan.write(.zplan)
```

## Definition of Done

- [x] `tune` compiles `arch.block` into `Dispatch[]` (ordered, bound).
- [x] `plan.buffers` is declared from `block.slots` (via `slot_bytes`).
- [x] Tensor indices and buffer bindings are resolved (tensor via lookup;
      buffers via slot).
- [x] A synthetic-block test asserts the `Dispatch[]` shape (count/order/op).
- [x] Old tensor-walk removed (no duplicate dispatches).
- [x] `zig build` / `zig build test` green.

## Comments

- **Block completeness:** residuals, `layer_output_scale` and final `softcap`
  are missing from `block.zig`; the compile step emits whatever the block
  declares. Correctness of logits waits on a complete block (separate task).
- **`.zplan` info:** the runtime needs `dims`/`ggml_type`/`layout` per tensor to
  build a `TensorView`. Plan: read dims/type from the ZTEF and `layout` from the
  zmeta at run time (format unchanged); storing them in the plan is option B.
- **KV shape:** whether the plan declares one KV cache or per-layer slots is an
  open decision; it affects `Buffer[]` and the runtime allocator.
- **Unblocks:** `runtime/02` (Executor) — it needs an executable plan, not the
  `01` tensor list.

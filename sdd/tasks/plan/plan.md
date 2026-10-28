# plan

title: Plan
reason: The compiled artifact the runtime consumes — the `.zplan` (`AOTKernel`)
        and how `tune` compiles it. The plan is data ("how to consume the
        image"), produced ahead of time; the runtime executes it and decides
        nothing.

## Tasks

### [01 - Plan Format](01-plan-format.md)
Status: done

Context: the `.zplan` format (config + fingerprint + tensor inventory + basic
ranges/nodes) and the runtime store layout. (Originally `tune/01 - Tune
Bootstrap`.)

### [02 - Compile the Block](02-compile-block.md)
Status: done

Context: compile `arch.block` into an executable `Dispatch[]` — ordered by the
forward, with `graph.Op -> kernels.Op`/`atomic_id` (Selector), role->tensor and
slot->buffer bindings, `Buffer[]` from `block.slots`, params from the config,
`phase`/`parts`. Replaces the per-tensor stopgap; unblocks `runtime/02`.

## Notes

- `tune` is the phase that shapes this artifact; it is not a milestone (the CLI
  part is `cli/`, the format/compilation is here).

### [03 - Forward Data](03-forward-data.md)
Status: done

Context: serialize the forward data the executor needs — per-dispatch params
(norm eps, RoPE base/dim/rot, attention scale/sliding, softcap) and the KV
cache layout (`KvLayout` -> `plan.buffers`), plus `zcom` RoPE tables — so the
runtime decides nothing.

### [04 - Per-Layer Dimensions](04-layer-dim-dispatch.md)
Status: done

Context: activation buffers are sized once (widest layer); gemma4 mixes full and
sliding layers with different head_dim/ffn, so narrow layers process a stale
prefix. Resolve each layer's sizes independently (per-dispatch offset/len or
per-layer buffers). Deferred from `runtime/03`.

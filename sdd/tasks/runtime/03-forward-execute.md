03 - Forward Execute

Status: done
Commits:

## Goal

Execute the forward-complete plan: the Executor builds each op's `Args` from the
plan's bindings + params, uses the `zcom` tables and the KV cache, and produces
logits — **deciding nothing** (no variant selection, no param computation at
run). Turns `run` from "a token" into a correct single-token gemma4 forward.

## Scope

- **In:** the executor's per-op `Args` construction (matvec x/out, norm with the
  weight+eps and the `atomic_id`-chosen variant, rope from the `zcom` table,
  `write_kv`/`attention` with KV + heads, `add`, `activation` gate/up, `lm_head`),
  a `KvCache` over the plan's KV buffers, `argmax`; a BOS/id-only token path
  (no tokenizer yet).
- **Out:** the plan format/compilation (`plan/03`); the block declaration
  (`architectures/03`); sampling/threading/GPU; tokenizer/detokenizer.

## Design

- **No decisions at run.** Variants come from `atomic_id`; params come from the
  `Dispatch` (plan/03); KV sizes from `plan.buffers`. The executor only maps
  bindings -> `Args` and calls `registry.run`.
- **`KvCache`:** a live view over the plan's KV buffers; `write_kv` stores k/v at
  `pos` for a layer, `attention` reads the rows `[t_start, pos]`. Shared-KV
  layers read the source layer's slot (from the plan's KV descriptors).
- **Ops:**
  - `dequant` (embedding row -> hidden), `matvec`/`lm_head`, `norm`
    (`rms_norm`/`in_place` per `atomic_id`, weight from tensor/zcom, eps from
    params), `rope` (cos/sin from the `zcom` region), `write_kv` + `attention`
    (heads/kv_heads/dims from the plan; scale/sliding from params), `add`
    (residual), `activation` (gate/up), `softcap`.
- **Buffers are typed by slot**, not "hidden-space": each `Dispatch.in/out`
  already points at the right slot; the executor sizes `Args` from the buffer
  and the op's params.
- **One token:** token id = BOS (0) for now; `head` -> `N` layers -> `tail` ->
  `argmax(logits)`. Prompt/max-tokens ignored (no tokenizer).

## Arch Provenance

- **own:** the args-from-plan executor, `KvCache`.
- **spec:** `.zplan` params/KV (plan/03); `zcom` regions (engine/02).
- **consulted:** llama.cpp/ggml for attention/KV shapes — reference only.
- **derived:** from `runtime/02` (Executor) and `plan/03`.

## Steps

1. `KvCache` over `plan.buffers` (slots per layer, `pos`).
2. Per-op `Args`: fix `matvec` x/out, `norm` (variant via `atomic_id`, eps +
   weight), `rope` (zcom table), `write_kv`/`attention`, `add`, `activation`
   (gate/up), `lm_head`.
3. Read params from `Dispatch`; read RoPE from `zcom`.
4. One-token path + `argmax`; keep the `token:` output.
5. A test runs a small synthetic forward plan (2 layers, tiny dims) end to end.
6. `zig build` / `zig build test` green.

## Flow / Arch

```text
Session.open: plan + ZTEF + zcom + buffers(+KV)
Executor.execute:
  for d in dispatch: build Args(d, params, KV, zcom) -> registry.run(op, d.atomic_id, args)
  argmax(logits) -> token
```

## Definition of Done

- [x] Executor builds `Args` per op from the plan; no runtime decisions.
- [~] `KvCache` backs `write_kv`/`attention`; shared-KV layers still emit
      `write_kv` (correct at `pos=0`, wrong across tokens — deferral).
- [x] RoPE reads the `zcom` inv_freq; norm uses the plan's eps/weight.
- [x] Executor tests (rope oracle, 1-head attention) pass; e2e on E2B.
- [x] `zig build` / `zig build test` green.

## Comments

- Depends on `plan/03` (params/KV in the plan) and `architectures/03` (complete
  block). Can start the `KvCache` + args plumbing against a synthetic plan.
- Correctness still needs the `elementwise` family (add) and a complete block;
  listed as dependencies.

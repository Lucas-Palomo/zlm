02 - Kernel API Simplification

Status: done
Commits:

## Goal

Give `kernels/` a single, clear responsibility: the **routines** (AtomicOps)
and the **catalog**. Remove the duplicated/ambiguous shapes that crept in
(`Dispatch`, `Op`, `OpId`, the global `Args` union) and put dispatch/args where
they belong (the plan/engine).

## Scope

- **In:** `kernels/api.zig` (the family contract), `kernels/registry.zig`
  (catalog), per-family `run(atomic_id, args)` typed by family, per-family
  `AtomicId` enum; retire `kernels.Dispatch`, `OpId`, the `Args` union.
- **Out:** changing the AtomicOps' math; the tuner/engine (they adapt in the
  following tasks).

## Design

Current confusion (to remove):

1. **Two `Dispatch`** — `kernels.factory.Dispatch` and `plan.Dispatch`.
2. **Two `Op`** — `kernels.factory.Op` and `model/graph.Op`.
3. **Dead `OpId` union** (no payload).
4. **Global `Args` union** — one mega-struct with fields for every family.
5. **Select/find in two layers** (`factory.findKernel/selectOne` vs the
   registry `select`).
6. **`atomic_id: u32`** untyped; only `run` knows how to read it.

Target:

```text
kernels/
  api.zig       # contract: Family{ op, name, run }, AtomicId per family
  registry.zig  # sets[] + find/run (the Tuner's surface)
  quant/ norm.zig rope.zig attention.zig activation.zig matvec/
  vector.zig    # utility (not a family)
```

Rules:

- **`kernels` has no `Dispatch`.** The dispatch is the plan's. Kernels expose
  `run(op, atomic_id, args)`.
- **One `Op` for kernel families** (dequant/matvec/norm/rope/attention/
  activation). The **semantic `Op`** (embd_lookup/lm_head/write_kv/softcap)
  stays in `model/graph.zig`; the scheduler maps `graph.Op -> kernels.Op`.
- **`Args` is per family**, not a union. The caller (engine) builds the family's
  `Args` and calls `registry.run(op, atomic_id, args)`; the registry switches on
  `op` and forwards to `family.run`.
- **`AtomicId` is an enum per family** (typed); the plan stores the number
  (`u8`); the engine/scheduler converts.
- **`KernelError`** in `api.zig`: minimal (`Unsupported`, `UnsupportedType`,
  `ShapeMismatch`, `TruncatedData`, `OutOfMemory`); layout errors live with the
  op that has layouts.

## Arch Provenance

- **own:** the family contract (`Family`/`AtomicId`), the removal of the
  duplicated dispatch/args.
- **spec:** ggml types/layouts per family.
- **consulted:** —.
- **derived:** from the earlier `KernelSet`/registry work.

## Steps

1. `api.zig`: `Op`, `KernelError`, `Family` (op + name + run), `Kernel`
   (id/atomic_id/supports/…). **(as built)** `Family` is the common contract
   struct (homogeneous so `sets[]` works); the typed `run(id, args)` is adapted
   by `api.erased(IdT, ArgsT, runFn)`, which validates `atomic_id` and casts the
   args. `KernelError` is the minimal set; the registry junction carries
   `anyerror` so each family keeps its own (possibly wider) error set.
2. Each family: a typed `AtomicId` enum + `run(id, args)`, published as
   `api.Family{ .op = …, .name = …, .run = api.erased(AtomicId, Args, run) }`.
   **(as built)** quant/matvec/norm/rope/attention/activation; `matvec` extends
   `KernelError` with its layout errors (`NoMappedData`, `InvalidRange`, …).
3. Registry (kept in `kernels/root.zig`, as in `01`): `sets[]` + `find(op)` +
   `run(op, atomic_id, args)`. **Removed** `select`/`SelectCtx`; they return with
   the real Selector.
4. **Removed** `kernels/factory.zig` and folded it into `api.zig`;
   `quant/factory.zig` stays (it is the quant family's surface, not the old
   KernelFactory). Updated importers: `plan/root.zig`,
   `architectures/{root,model/contract,models/gemma4/arch}.zig`, `main.zig`,
   `quant/types.zig`.
5. Updated `kernels/root.zig` exports and tests (`run` dispatches a dequant;
   `api.erased` rejects an unknown id).
6. `zig build` / `zig build test` green.

## Flow / Arch

```text
plan.Dispatch { op, atomic_id, ... }
  -> engine builds the family's args
  -> kernels.registry.run(op, atomic_id, args) -> family.run(id, args)
```

## Definition of Done

- [x] `kernels/api.zig` holds `Op`/`KernelError`/`Family`/`Kernel`.
- [x] No `Dispatch`, `OpId`, or global `Args` union in `kernels/`.
- [x] Each family exposes a typed `AtomicId` + `run(id, args)`.
- [x] `registry.run(op, atomic_id, args)` dispatches to the family.
- [x] `zig build` / `zig build test` green.

## Comments

- The plan/tuner/engine adapt in `architectures/02` Step 3+ (scheduler maps
  `graph.Op -> kernels.Op`; engine builds the args). The engine decodes the
  plan's numeric `atomic_id` and builds the family's `Args`, then calls
  `registry.run(op, atomic_id, args)`.
- `vector.zig` is a utility, not a family (documented as such).
- The registry junction is dynamic, so `Family.run`/`registry.run` carry
  `anyerror`; each family keeps its own typed error set (`api.KernelError` plus
  layout errors where it has layouts).
- `Kernel` (the catalog metadata) is defined in `api.zig` but not yet populated
  by families; the Selector task fills it. `select`/`SelectCtx` are gone until
  then.

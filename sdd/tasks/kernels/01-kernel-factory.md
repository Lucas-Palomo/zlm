01 - KernelFactory and Registry

Status: done
Commits:

## Goal

Lay down the compute-layer structure so each family is only its `AtomicOp`s.
Establish `KernelFactory` (builds a family's surface), `KernelSet` (the product)
and the thin `Registry` (the Tuner's surface). `quant` is the first family.

## Scope

- **In:** the factory/set/registry scaffolding; the `Dispatch` plumbing that the
  `run` uses; `quant` (dequant/encode) as the first family.
- **Out:** the matvec family and the other ops; the real `Tuner`/`Selector`
  (only the surface they consume); measured autotuning.

## Design

Rules every family follows (the factory enforces/encodes them):

1. `AtomicOp` receives **slices + params** and writes `out`; **no allocation**.
2. `Kernel` **does not allocate or map**; it describes/supports only.
3. `select(ctx)` returns a `Dispatch` or `error.Unsupported`.
4. `run(dispatch, args)` executes the resolved choice; it **decides nothing**.
5. Explicit `begin..end` ranges on every per-row op.
6. One `KernelError`; no global state.
7. No `architectures`/`runtime`/`gguf` imports (only `gguf/format` + `utils`).
8. One oracle/test per `AtomicOp`.

Shape (indicative):

```zig
// kernels/factory.zig
pub const SelectCtx = struct { op: Op, quant: ?, layout: ?, dims: ?, budget: ? };
pub const Kernel = struct {
    id: []const u8,
    op: Op,
    supports: *const fn (ctx: SelectCtx) bool,
    // scheduling metadata the Scheduler reads later
    row_parallel: bool = false,
    fusable: bool = false,
    scratch: u64 = 0,
};
pub const Dispatch = struct { op: Op, atomic_id: u32, /* bindings/device/phase/parts */ };
pub const KernelSet = struct {
    kernels: []const Kernel,
    select: *const fn (ctx: SelectCtx) KernelError!Dispatch,
    run: *const fn (d: Dispatch, args: anytype) KernelError!void,
};
pub fn Factory(comptime defs: anytype) KernelSet { ... } // dev-time builder

// kernels/registry.zig  (the Tuner's surface; thin)
pub const sets = [_]*const KernelSet{ &quant.set, ... };
pub fn find(op: Op) ?*const KernelSet;
pub fn select(op: Op, ctx: SelectCtx) KernelError!Dispatch;
pub fn run(d: Dispatch, args: anytype) KernelError!void;
```

## Arch Provenance

- **own:** the factory/set/registry split, the family rules, the AOT surface.
- **spec:** ggml quant block layouts (for `quant`).
- **consulted:** llama.cpp/ggml quant codecs — reference only.
- **derived:** `Kernel`/`AtomicOp` from the earlier `KernelDescriptor`.

## Steps

1. `kernels/factory.zig`: `Kernel`, `SelectCtx`, `Dispatch`, `KernelSet`, and the
   `Factory` builder. **Done.**
2. `kernels/registry.zig` (as `kernels/root.zig`): the static list +
   `find/select/run`. **Done.**
3. `quant` family: `kernels/quant/{types,floats,standard,k_quants,factory}.zig`
   via the factory; `dequantTo` as the AtomicOp. **Done.**
4. Oracle tests per AtomicOp (11 dequant tests ported); `zig build test`. **Done.**

## Flow / Arch

```text
quant AtomicOps ─→ KernelFactory ─→ KernelSet ─┐
                                               ├─→ registry ─→ Tuner/Selector
                                  (later families) ┘
```

## Definition of Done

- [x] `kernels/factory.zig` exposes `Kernel`/`SelectCtx`/`Dispatch`/`KernelSet`
      and a `findKernel`/`selectOne` helper.
- [x] `kernels/root.zig` (registry) aggregates sets with `find/select/run`.
- [x] `quant` is a family built via the factory; `dequantTo` returns the same
      results as before (oracles pass).
- [x] No family imports `architectures`/`runtime`/`gguf` beyond `gguf/format`.
- [x] `zig build` / `zig build test` green.

## Comments

- The `args: anytype` in `run` is a placeholder; a typed args union per op may
  replace it once matvec lands.
- The real `Tuner`/`Selector` (measurement) comes in the `tune` epic; here we
  only build the surface they consume.

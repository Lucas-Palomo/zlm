# Kernel Architecture

The compute vocabulary shared by `kernels/`, the tuner, and the plan
(`.zplan`).

## 🏗️ Compile and Optimize (Ahead-of-Time)

- **Kernel** — catalog / abstract definition of a compute operation.
- **AtomicOp** — concrete, optimized routine inside a Kernel (SIMD, tiling,
  codec).
- **Selector** — chooses the ideal `AtomicOp` variant/implementation for the
  hardware and the context (quant, layout, shape, budget).
- **Composer** — groups operations into an execution form: single, fused or
  sequential.
- **Scheduler** — plans the order, the device placement, data residency and the
  phase.
- **Tuner** — the compilation pipeline that runs the phases above and produces
  the final compiled artifact.

## 📦 Deliverable Artifact (the Plan)

- **AOTKernel (`.zplan`)** — the static macro artifact: the deterministic,
  compiled execution plan.
- **Dispatch** — the unit item inside the plan: the chosen operation, memory
  bindings, device, phase, and **how to parallelize** it.
- **ResidencyGraph** — part of the `AOTKernel`: how the ZTEF files must be read
  (ranges + `page_in`/`page_out`/`keep` nodes). Static data.

## 🚀 Runtime

- **Runtime** — manages the session, I/O buffers, cache, and loads the `.zplan`.
  Owns the resources (workers/thread pool, memory).
- **ShadowGraph** — the **live copy** of the `ResidencyGraph`, kept by the
  Runtime during execution: it admits/evicts ZTEF within the budget, following
  the graph and deciding nothing.
- **Executor** — a **sequential** loop that consumes and dispatches each
  `Dispatch` in order, on the hardware. It decides nothing.
- **KV cache** — the session's key/value cache is stored in **`f16`** (the
  forward/attention compute in `f32`): it halves the KV's RAM and frees page
  cache for streamed weights. A deliberate RAM/precision trade-off — nearly
  lossless, not bit-exact.

## Notes

- **atomic = not decomposed by the Executor**, not CPU/concurrency atomicity.
- **AOT = ahead-of-time**: everything is resolved during `tune`; the Runtime
  executes and never re-selects or re-decides.
- **Parallelism is a plan property.** The `Executor` is sequential; a `Dispatch`
  invokes parallelism (row ranges, parts, threads, device) as recorded in the
  plan. The Runtime supplies the resources.
- Only two names carry "Kernel": **`Kernel`** (the op definition, catalogued as
  `api.Kernel`) and **`AOTKernel`** (the plan).
- `Selector` / `Composer` / `Scheduler` are roles of the **Tuner**, not kernels.

## Family assembly (dev-time)

Each kernel family declares its surface in `kernels/api.zig`:

- **`Family`** — the family's product: `{ op, name, row_parallel, kernels, run }`,
  where `kernels` is the candidate catalog and `run` is type-erased
  (`run(atomic_id, args)`).
- **`api.erased(AtomicId, Args, run)`** — binds the family's typed
  `run(id, args)` (and validates the id) into `Family.run`.

This assembly is **dev-time only**: it does not run at tune or run time. What
flows onward is the static `Family` catalog. The `Selector` (tuner) maps a model
op to a `kernels.Op`/`atomic_id`; the `Executor` calls the resolved `atomic_id`
through the family's `run`.

## Candidate catalog and recipes

Each family also declares *what* the Selector may pick and *which* compositions
the Composer may form:

- **`api.Kernel`** (candidate) — one entry per selectable AtomicOp: `id`, `op`,
  `atomic_id`, `row_parallel`, `fusable`, `scratch`, `tier` (`cpu`/`gpu`),
  `complexity` (`linear`/`quad_mem`/`quad_flops`) and `variant` (the family
  sub-code, e.g. `MatVecPlan`). `registry.kernels()` / `kernelsFor(op)` expose
  the aggregated catalog; a `gpu` entry is simply never selected without a
  backend.
- **`Recipe`** (`kernels/recipes.zig`) — a named composition the Composer may
  emit: `name`, `form` (`fused`/`seq`), the owning `op` + `atomic_id` that
  implements it, and the model `ops` it replaces. `registry.recipes()` returns
  the catalog (data only; a family wires the atomic once implemented).
- The catalog declares seven recipes today (`gate_up_fused`, `qk_norm_rope`,
  `norm_proj`, `rope_write_kv`, `add_norm`, `dequant_scale`, `matvec_bias`); the
  Composer emits the ones gemma4's graph uses. `bytes_saved` is a coarse count
  of eliminated round-trip passes, not a shape-specific byte figure.

**Compile-time width variants.** A hot kernel is written **once**, parameterized
by `comptime lanes`, and the build instantiates a small fixed set
(`kernels/simd.zig`: `Width { s4, s8, s16 }`) for the target. `@Vector(N,T)`
does not enable ISA features — the target (`-mcpu=native` or a class like
`x86_64_v3`) does; a width wider than the target's native width is lowered by
splitting. The Tuner benchmarks the instantiations and records the fastest in
the plan (`Dispatch.simd_width`); the Runtime dispatches the recorded one. No
per-ISA source: the machine choice is a *plan* property, decided by profiling.

## AOT, not JIT

The pipeline is **ahead-of-time**: every decision is made at `tune` and written
into the plan. Consequences:

- **No interpreter.** The `Executor` calls a resolved function directly per
  `atomic_id`; the plan is data, not interpretable bytecode. (This is the
  deliberate break from ART/JIT.)
- **Composition is recipes, not IR.** The `Composer` chooses among fusion
  *recipes* declared by the families (e.g. `gate_up_fused`, `matvec_bias`,
  `qk_norm_rope`); it does not assemble an arbitrary op graph at run time.
  A new fusion is dev-time family code.
- **One `atomic_id` per AtomicOp.** A family = one `Op` = one `Family`; its
  variants (primitives and fused recipes) are `AtomicOp`s, each with an
  `atomic_id`. The `Dispatch` carries the chosen `atomic_id`.

## Selector / Composer / Scheduler contract

```text
Registry.find(op) -> Family                 per-op catalog (tune)

Selector (raw)     : model op      -> atomic_id + bindings
Composer           : model ops     -> grouping (single | fused | seq)
Selector (re-select): group/recipe -> atomic_id + bindings
Scheduler          : Dispatch[]    -> AOTKernel (order, device, phase, parts,
                                                 ResidencyGraph)
```

- The **op list** comes from the architecture declaration (`architectures/`,
  `Block`/`Instr[]`), not from `kernels/`.
- The **Selector** is the **only** role that chooses an `atomic_id`. It runs per
  *unit*: first a raw pass over the model ops, then again whenever the Composer
  forms a new unit — a fused recipe is a *different operation*, so it needs its
  own selection. The Selector never changes the grouping.
- The **Composer** owns **grouping**: which adjacent model ops become one
  dispatch (`fused`), which stay alone (`single`), and which must stay ordered
  as a group (`seq`), using the recipe catalog and the `fusable` metadata.
  Because a fused/seq group is a new unit, the Composer **drives a
  re-selection** (asks the Selector for that unit's atomic); it does not pick the
  atomic itself.
- The **Scheduler** owns order, placement, phase, `parts` and the
  `ResidencyGraph`; it never chooses an atomic nor a grouping.
- `Dispatch.bindings` references the tensor to read (ZTEF tensor id/offset); the
  `Executor` does not resolve it.
- `Dispatch` fields: `op`, `atomic_id`, `device`, `phase`, `parts`, `bindings`.

```text
model ops ──▶ Selector (raw) ──▶ per-op candidates
                        │
        ┌───────────────▼───────────────┐
        │ Composer                      │
        │  ops -> single | fused | seq  │
        └───────────────┬───────────────┘
                        │ triggers re-selection
        ┌───────────────▼───────────────┐
        │ Selector (re-select)          │
        │  group -> atomic_id           │
        └───────────────┬───────────────┘
                        │ atomic_id per final unit
        ┌───────────────▼───────────────┐
        │ Scheduler                     │
        │  order / placement / parts    │
        └───────────────┬───────────────┘
                        ▼
             Dispatch[] ──▶ AOTKernel
```

## Registry (the Tuner's surface)

The **Registry** is the aggregated catalog of every family. It is the single
surface the **Tuner** consumes: the Tuner does not know the families, it asks
the Registry.

- **Built statically:** `registry.zig` holds a slice of `*const Family`
  (comptime). No runtime/plugin loading.
- **Surface:**
  - `sets` — enumerate;
  - `find(op) -> ?*const Family`;
  - `run(op, atomic_id, args)` — validation/benchmarking in the Tuner;
  - `kernels()` / `kernelsFor(op)` — the aggregated **candidate catalog** (one
    `api.Kernel` per selectable AtomicOp): `row_parallel`, `fusable`, `scratch`,
    `tier`, `complexity`, `variant`. The Selector/Scheduler enumerate these
    without knowing the family;
  - `recipes()` — the declared **recipe catalog** (`kernels/recipes.zig`),
    data-only, consumed by the Composer.
- **Not here:** `select(op, ctx) -> Dispatch` lives in the Tuner's Selector.
- **Boundaries:** built at dev-time (`Family` -> Registry); read at **tune**
  (Tuner/Selector); **not** used at **run** (the `Executor` uses `dispatch.op`,
  already resolved in the plan).

`registry.zig` is intentionally thin: the family list plus `find/run`, plus the
aggregated candidate/recipe catalogs (`kernels`/`recipes`). Adding a family is
one line.

## Composition (general flow)

```text
 DEV-TIME (code)                          TUNE (per machine)                    RUN
 ┌───────────────────────────┐
 │ architectures/ impl/<arch>│ declares
 │  block = Instr[]          │───────────┐  op list (model)
 │  resolveParams/layerDims  │           │
 └───────────────────────────┘           │
 ┌───────────────────────────┐           │
 │ kernels/ families         │───────────┤
 │  AtomicOps + api.Kernel   │           ▼
 │  (+ recipes catalog)      │  ┌─────────────────────────────────────┐
 └───────────────────────────┘  │ Registry = Family[] + find/run      │
                                │ + kernels()/recipes() (api.Kernel)  │
                                └───────┬─────────────────────────────┘
                                        ▼
                      ┌──────────────────────────────────────────────┐
                      │ Tuner (tune)                                 │
                      │  Profiling        (caps: ISA/cores/caches)   │
                      │        │                                     │
                      │        ▼                                     │
                      │  Measure          (DRAM/disk bw, dispatch ns)│
                      │        │                                     │
                      │        ▼                                     │
                      │  Calibrate/Decide (apply coeffs; cost ranks) │
                      │        │                                     │
                      │        ▼                                     │
                      │  Selector (raw)    op -> atomic              │
                      │        │                                     │
                      │        ▼                                     │
                      │  Composer    single | fused | seq            │
                      │        │ triggers re-selection               │
                      │        ▼                                     │
                      │  Selector (re-select) group->atomic          │
                      │        │                                     │
                      │        ▼                                     │
                      │  Scheduler   order/placement/parts           │
                      │              budget via cost.zig             │
                      └─────────────────┬────────────────────────────┘
                                       ▼ writes
                    ┌─────────────────────────────────────────────────┐
                    │ AOTKernel (.zplan)                              │
                    │   Dispatch[]     (operations, bindings, parts)  │
                    │   ResidencyGraph (how to read the ZTEF:         │
                    │                   ranges/nodes page_in/out/keep)│
                    └───────────────────────┬─────────────────────────┘
                                            │ loads
                                            ▼
                             ┌──────────────────────────┐
                             │ Runtime (daemon)         │
                             │  load spawns daemon      │
                             │   Session + Executor     │
                             │  ResidencyGraph -> Shadow│
                             │  dispatch -> run(…)      │
                             └──────────────────────────┘
```

## Tuning flow (profile → plan)

`tune` is **deterministic**: every decision comes from the analytical cost model
(`cost.zig`) over the machine's capabilities (`caps.zig`). Timing is used **only
to calibrate the cost model's coefficients** (native measurement), never as a
per-candidate decision gate. The device profile is a *phase of `tune`*, not a
peer of the plan, and its `.zat` is a **device-keyed cache** (never read at run).

```text
 DEV-TIME (code)                          TUNE (per machine)                                    RUN
 ┌───────────────────────────┐
 │ architectures/ impl/<arch>│ declares
 │  block = Instr[]          │───────────┐  op list (model)
 │  resolveParams/layerDims  │           │
 └───────────────────────────┘           │
 ┌───────────────────────────┐           │
 │ kernels/ families         │───────────┤
 │  AtomicOps + api.Kernel   │           ▼
 │  (matvec MatVecPlan, …)   │  ┌─────────────────────────────────────────────┐
 └───────────────────────────┘  │ Registry = Family[] + find/run              │
                                │ + kernels()/recipes() (api.Kernel)          │
                                └───────────────────────┬─────────────────────┘
                                                        ▼
                   ┌──────────────────────────────────────────────────────────────┐
                   │ Tuner  (`cli/commands/tune.zig`)                              │
                   │                                                              │
                   │  (1) CALIBRATE (device)  ── caps + measure coeffs ─┐       │
                   │        caps.zig: ISA/cores/caches/cgroup/freq      │       │
                   │        calibrate.zig: DRAM/disk bw + dispatch ns   │       │
                   │        (min-of-N; reuse <digest>.zat unless --force│       │
                   │                                                  ▼ │       │
                   │                                          ┌──────────────┐  │
                   │                                          │ .zat (device)│  │
                   │                                          │ Caps + Coeffs│  │
                   │                                          └──────┬───────┘  │
                   │  (2) DECIDE  ◀── caps (+coeffs) + declaration ──┘          │
                   │        Selector : variant (ISA×shape) + parts=argmin cost  │
                   │        Composer : recipes -> single | fused | seq          │
                   │        Scheduler: budget/admission/lookahead via cost.zig, │
                   │                   KV/window inversion, residency           │
                   └───────────────────────────────┬──────────────────────────────┘
                                                   │ writes
                                                   ▼
                                     ┌──────────────────────────────────┐
                                     │ AOTKernel  (`.zplan`)            │
                                     │   Config + fingerprint           │
                                     │   Tensor[] / Buffer[]            │
                                     │   Dispatch[] (atomic, params,    │
                                     │     in/out_len, parts, residency)│
                                     │   ResidencyGraph + affinity[]    │
                                     └───────────────┬──────────────────┘
                                                     │ .zlm/runtimes/
                                                     │   <image>_<fp>.zplan
                                                     │   <device_digest>.zat (shared)
                                                     │   manifest.zrun : alias -> …
                                                     ▼
                                           ┌──────────────────────────┐
                                           │ Runtime  (`runtime/`)    │
                                           │   daemon: load spawns it │
                                           │   Residency engine       │
                                           │   Executor: sequential   │
                                           │     dispatch -> run(…)   │
                                           └──────────────────────────┘
```

Chain:

```text
calibrate ─▶ .zat (device) ─▶ decide (Tuner) ─▶ .zplan ─▶ run
(caps+coeffs)               Select/Comp/Sched  (AOTKernel)  (Executor)
```

- **Calibrate** (`caps.zig` + `calibrate.zig`) introspects the device
  (deterministic: ISA lanes/features, cores/topology, caches, frequency,
  cgroup/affinity, NUMA) and measures the cost model's coefficients — DRAM
  bandwidth, disk streaming bandwidth, per-dispatch latency, and the
  `madvise(WILLNEED)` cost (`base + per_page*pages`, capped) (min-of-N). It caches
  a **device profile** (`caps` + `coeffs`) as `ZAT2` at
  `.zlm/runtimes/<device_digest>.zat` (keyed by the CPU digest, shared across
  images); it is measured on the first tune and reused, and `--force`
  re-measures it.

  Measurement is a **min-of-5** per coefficient (the best pass is kept, so it
  estimates the **peak**, not the average): DRAM streams a 64 MiB buffer; disk
  cold-reads up to 128 MiB (`fadvise(DONTNEED)` before each pass); dispatch
  latency runs 200k serial `tinyOp` calls. A failed sample falls back to a
  conservative constant; tuning the count is only worth it for noisy/shared I/O.
- **Decide** (`Selector` + `Composer` + `Scheduler`) consumes the caps and the
  declaration. The Selector picks the AtomicOp variant by a **documented ISA×shape
  rule** and `parts = argmin cost.opCost(…) (≤ threads)`; the Composer applies the
  recipe catalog; the Scheduler sizes the budget via `cost.zig`
  (`admissionDensity`, `lookahead`, `opCost`, `planScore` — a TTFT+generation
  objective weighted toward generation). No timing, no per-candidate `.zat`.
- **`.zat`** is a device-keyed calibration cache, upstream of the decisions and
  never read at run; it carries no per-image or per-decision data.
- **Run** reads only the `.zplan` (plus the store registry to resolve an alias).

## Code map

| Term | Where |
|---|---|
| `Kernel` | `kernels/{linear,transformer}/<op>.zig` (+ `kernels/registry.zig` catalog) |
| `AtomicOp` | `kernels/linear/quant/`, `kernels/linear/matvec/{native,scale_split,batched,fused}.zig` |
| `Width` (compile-time SIMD variants) | `kernels/simd.zig` |
| `Family` (dev-time assembly) | `kernels/api.zig` + each family root |
| `Registry` | `kernels/registry.zig` (the Tuner's surface) |
| `Selector` / `Composer` / `Scheduler` / `Tuner` | `tuner/` + `cli/commands/tune.zig` |
| `Caps` / `Coeffs` (device profile, `.zat`) | `tuner/caps.zig` + `tuner/cost.zig` + `tuner/calibrate.zig` |
| `Composer` (recipes) | `tuner/composer.zig` (+ `kernels/recipes.zig` catalog) |
| `Dispatch` / `AOTKernel` | `plan/` (the `.zplan`) |
| `ResidencyGraph` | `tuner/compile.zig` (built at tune, stored in the plan) |
| `Store` / alias registry | `image/store.zig`, `image/zrun.zig` (`manifest.zrun`) |
| `ShadowGraph` / `Runtime` / `Executor` | `runtime/residency.zig`, `runtime/session.zig`, `runtime/executor.zig` |

## Relationship to ART/JIT (not followed)

The shape echoes ART (an ahead-of-time artifact + an instruction executor +
profile-guided compilation), but zlm is **AOT-deterministic and has no VM**: no
bytecode interpreter, no JIT, no GC, no deopt. The `.zplan` is not an
interpretable bytecode; it is a fully-resolved plan, and the Executor never
re-decides.

## Relationship to XLA/MLIR (reference)

**XLA** and **MLIR** are graph/IR compilers and are a reference for the *shape*
of the tuner, not a base we build on.

- **XLA** compiles a whole op-graph ahead-of-time/JIT (HLO -> passes -> native
  kernels), fusing ops and picking tiling/layout. This maps to our `Tuner`
  (`Selector` + `Composer` + `Scheduler`) producing an `AOTKernel`; XLA's
  "executable" is our `.zplan`.
- **MLIR** is an IR framework (multi-level dialects + lowering). That is the
  canonical answer to "an IR between `Kernel` and `AtomicOp`" — we do **not**
  need it today: our IR is the `AOTKernel`/`Dispatch` sequence itself.

Where we differ: XLA/MLIR are general-purpose (arbitrary graphs, JIT, GPU/TPU,
aggressive fusion). zlm is **AOT-deterministic, LLM/mmap-focused**: no VM/JIT,
no dynamic re-lowering; everything is resolved at `tune`.

## Relationship to BLIS (not followed)

BLIS has no `Selector`/`Composer`/`Tuner` split and fixes macro<->micro
statically. zlm selects per context/budget and composes dynamically at tune time.

## Provenance

- **own:** the vocabulary, the AOT framing, the `Dispatch`/`AOTKernel` split, and
  parallelism-as-a-plan-property.
- **consulted:** llama.cpp/ggml for quant layouts and operation details —
  reference only, not mirrored.
- **derived:** `Kernel`/`AtomicOp` mapping from the earlier `KernelDescriptor`
  work.

The module-level record, including the quant/GPU attribution questions, is in
[`arch-provenance.md`](arch-provenance.md).

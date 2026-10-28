# CONSTITUTION

The rules this project does not violate. When something conflicts with this
file, this file wins. Changes here are deliberate and rare.

## Identity

`zlm` is a CPU-first runtime for Language Models. It compiles a GGUF model into
a self-contained **image** (`ztef | zman | zmeta | zhash`) and runs it with
bounded, explicit memory.

Identity: **"You decide how your language model runs."**
Anti-OOM by design, RAM/VRAM-budgeted, CPU-first, mmap-first, tuned to your
hardware. `zlm` is just a name (no acronym).

## Reference, not a port

**llama.cpp (and similar projects) are reference material only.** Use them to
look up formats, quant layouts, and architecture details — then implement what
*this* project needs.

This is **not** a port of llama.cpp to Zig. Do not mirror its file layout, APIs,
naming, or design decisions unless this codebase independently arrives at the
same need. When a decision here differs, the reason must stand on its own (see
`docs/`/`sdd/`), not on "llama.cpp does it this way".

## Principles

- **mmap-first.** Never read a whole model into RAM; access lazily.
- **Bounded memory.** Explicit budgets, predictable scratch; no full-weight
  preload, no full dequantized matrices.
- **The image is the source of truth.** After `image compile`, the GGUF is not
  needed to run.
- **Three phases:** `compile` (deterministic, hardware-independent) →
  `tune` (hardware measurement + plans) → `load/run` (execute the plan).
- **File-per-module; a directory only with 3+ real siblings.** No facade file
  unless there are many importers.
- **Dependency direction is one-way.** External formats (`gguf/`) and our own
  formats (`image/`) do not import architecture/model-specific code.
- **AOT (ahead-of-time).** Everything is resolved at `tune`; the runtime
  re-decides nothing.
- **Atomic = not decomposed by the Executor** (not CPU/concurrency atomicity).
- **Parallelism is a plan property.** The `Dispatch` declares it, the `Runtime`
  supplies the resources, and the `Executor` stays sequential.

## Platform

- **POSIX-first:** Linux and macOS (same POSIX code, `std.posix.mmap`).
- **Windows is out of scope** (mmap/OS limitations). Windows users run it under
  **WSL or Docker**.
- OS-specific tuning is a `tune` concern: capability boosts with fallbacks, and
  a cgroup-aware budget for containers.

## Kernel Architecture

Vocabulary (full doc: `docs/kernel-architecture.md`):

- **Kernel** — abstract definition of an operation (catalog).
- **AtomicOp** — concrete routine (SIMD/tiling/codec).
- **Selector** — picks the best AtomicOp for hardware + context.
- **Composer** — execution form: single | fused | seq.
- **Scheduler** — order, placement, residency, phase.
- **Tuner** — compiles (Selector + Composer + Scheduler) -> AOTKernel.
- **AOTKernel** (`.zplan`) — the compiled plan: `Dispatch[]` + `ResidencyGraph`
  (how to read the ZTEF); **Dispatch** — one item in it.
- **Runtime** — session/buffers/cache/resources; copies the `ResidencyGraph`
  into a live `ShadowGraph`; **Executor** — a sequential loop that dispatches
  each `Dispatch` and decides nothing.
- **KernelFactory** (dev-time) — builds a family's `Kernel` catalog + `select` +
  `run` from its `AtomicOp`s; **KernelSet** — the product.
- **Registry** — the aggregated `KernelSet` catalog; the single surface the
  **Tuner** consumes (`find/select/run` + scheduling metadata). Not used at run.

## Layers and boundaries

Five layers, from the outside in:

```text
EDGE        cli/    http/
              # expose the runtime to a user or a client; no model logic
              # examples: cli/commands/run.zig, cli/commands/chat.zig, http/

RUNTIME     runtime/
              # engine (Executor) + session (KV/sampler) + memory (ShadowGraph)
              #   + model + infra; owns the AOTKernel and all state

DECLARATION architectures/   tuner/
              # architectures/ declares (core/ vocabulary + impl/<arch>);
              # tuner/ compiles the plan (Selector + Composer + Scheduler)

ARTIFACT    plan/   kernels/   image/   gguf/   gpu/
              # plan = AOTKernel; kernels = definitions + AtomicOps;
              # image/gguf = our/external formats;
              # gpu = the co-processor (gpu-facing API + backend contract +
              #       GPU kernel grid); a backend is a build-time selection

BASE        data/   utils/
              # shared data vocabulary (TensorView); io, mmap, binary_reader, cpu
```

Flow:

```text
user/client ─▶ EDGE ─▶ RUNTIME ─▶ kernels.run(dispatch)
                              ▲
architectures (declare) ─▶ tuner ─▶ plan (AOTKernel) ─┘
```

Rules:

1. **An edge talks only to the runtime.** It sends text and receives text; the
   runtime owns tokenizer, chat template and generation. An edge never imports
   `kernels`/`plan`/`image`, never touches a `Dispatch`, and never assembles a
   model call.
2. **`runtime/` executes the plan and owns state** (session, KV, memory, infra).
3. **`architectures/` only declares.** No forward logic: the forward *is* the
   `plan`.
4. **Dependencies point inward only.** `kernels/`, `plan/`, `image/`, `gguf/`
   and `gpu/` never import `architectures/`, `runtime/` or `cli/`.
5. **`gpu/` is the co-processor's implementation.** It may import `kernels/`
   (the catalog vocabulary), `gguf/`, `data/` and `utils/`; nothing in
   `kernels/`, `plan/`, `image/`, `gguf/`, `data/` or `utils/` imports `gpu/`.
   `gpu/root.zig` is the gpu-facing API, `gpu/backend/root.zig` the backend
   contract and `gpu/backend/<name>/` a backend (a build-time selection, no
   runtime plugin). The GPU kernel grid lives in `gpu/kernels/grid.zig`.

## Conventions

- No inline (`//`) comments unless asked. Doc comments (`///`) on public APIs
  and `// TODO:` markers are allowed.
- All stdout/stderr goes through `utils/io.zig`.
- Error sets live in the module that defines them.
- Before adding a dependency/library, check that the project already uses it.
- **Tests live beside their module.** Whenever possible, a module's tests and its
  private fixtures go in a sibling `<mod>_test.zig`; the production file holds
  **no** `test` block. A directory's cross-module tests go in its `test.zig`
  aggregator, and the fixtures shared by more than one `<mod>_test.zig` go in a
  `fixtures.zig` (test-only, never imported by production). The aggregator is
  pulled by `main.zig` so `zig build test` covers it.

## Documentation (SDD)

Specs live in `sdd/tasks/<milestone>/NN-slug.md`; `sdd/roadmap.md` holds the
phases and the epic/task catalog; cross-cutting deferrals go in
`sdd/open-items.md`.

```markdown
NN - Title

Status: <draft|pending|in progress|done>
Commits: <hashes>

## Goal        (history / why)
## Scope       (what is / is not in)
## Design      (how; prose and/or bullets)
## Steps       (detailed; may use bullets and subsections)
## Flow / Arch (when applicable; diagrams help)
## Definition of Done   (checkboxes; checkpoints for big tasks)
## Arch Provenance  (only if applicable: own / spec / consulted / derived)
## Comments    (open/pending/blocked; list the cross-cutting ones in open-items)
```

- `draft` = being written; `pending` = design done, not started; `in progress`;
  `done` = all DoD checked.
- `Steps` is the history: annotate each step **in place** as it lands (as-built
  result + commit). Never pre-check DoD.
- `Comments` holds open/pending/blocked points; promote the cross-cutting ones
  to `sdd/open-items.md`. When a point is resolved, **strike it** (`~~...~~`) in
  place instead of deleting it, and **delete** the matching `sdd/open-items.md`
  entry.

`Arch Provenance` records where each decision comes from, so a task never
silently becomes a port: **own** (original to zlm) / **spec** (external
format/spec) / **consulted** (llama.cpp/ggml or similar; reference only, not
copied) / **derived** (from own/spec/consulted).

### Epics (milestones)

`sdd/tasks/<milestone>/<milestone>.md` is the epic: the objective of the group,
generic over its tasks. A sub-epic adds `from:`.

```markdown
# <Title>

Status: <draft|pending|in progress|done>

Reason: <why this group exists>

## Tasks

### [NN - <title>](NN-slug.md)
Status: <status>

Context: <one-line overview>
```

### Roadmap (catalog)

`sdd/roadmap.md` holds the phases and the epic/task catalog.

```markdown
# Roadmap

## Phases
<compile -> tune -> run>

## Epics
- [**<epic>**](<milestone>/<milestone>.md) — <status>   (link the name when it exists)
  - [NN - <title>](<milestone>/NN-slug.md) — <status>
```

### Open items

`sdd/open-items.md`, grouped by milestone; each item links back to the task that
raised it.

```markdown
### <title>
from: <task file>
reason: <why it is open / cause>
```

When an item is resolved, delete it here and strike (`~~...~~`) the matching
line in the task's `Comments`.

10 - gpu Structure (backend-ready)

Status: done
Commits:

## Goal

Rename `device/` -> `gpu/` and restructure it so a new backend is a **drop-in**,
not a rewrite: Vulkan today, Metal/... tomorrow. Three explicit layers — the
gpu-facing API, the **backend contract**, and the backend implementation — plus
the GPU kernels.

## Scope

- **In:** the rename; the backend contract; the Vulkan impl under
  `gpu/backend/vulkan/`; the kernels dir; the importers; the constitution.
- **Out:** implementing a second backend; the `gpu/09` catalog.

## The structure

```text
gpu/
  root.zig            # the gpu-facing API (Device/Resident/Handle/Probe) -- what the runtime uses
  backend/
    root.zig          # the CONTRACT: the shared types + the ops a backend provides
    vulkan/
      root.zig        # the Vulkan backend (init/device)
      context.zig     # the instance/queue
      pipeline.zig    # the pipelines + the ring
      types.zig       # the Vulkan FFI
  kernels/
    grid.zig          # the GPU kernel GRID: quant x variant (the catalog)
    shaders.zig       # the embedded shaders
    shaders/          # the .comp/.spv (the impl)
```

## Design

### The kernels grid

- `gpu/kernels/grid.zig` holds the **GPU kernel grid** — the quant x variant
  catalog (the `GpuVariant` + the `Kernel` entries) — moved out of the CPU's
  `kernels/linear/matvec/root.zig` (today the GPU grid lives on the CPU side).
- The shaders are the **impl** (`shaders/` + the `shaders.zig` embed); the grid
  references them. A backend (the Vulkan) executes the grid.
- This is the exact parallel of the CPU: the definition + the grid live together,
  the executor (the backend) is separate. GLSL has no comptime, so the GPU
  "generic" is the grid-as-data (today) + the `gpu/09` specialization constants
  (later) — not a comptime instantiation.

### The backend contract

- `gpu/backend/root.zig` declares the **ops** a backend must provide (init/deinit,
  the resident upload/free, the GEMV/GEMM submit, sync, the probe) + the shared
  types (`Resident`, `Handle`, `Quant`, `Probe`). A backend **implements** it.
- The backend is selected at **build** (Vulkan on Linux; Metal on macOS) — a
  comptime selection; **no runtime plugin**. `gpu/root.zig` exposes the
  gpu-facing API and delegates to the selected backend.
- A new backend = a new `gpu/backend/<name>/` implementing the contract; the
  gpu-facing API, the kernels grid and the runtime are untouched.

### The test layout

- Per `kernels/19` + the CONSTITUTION's Conventions: the tests move out of the
  production modules into `<mod>_test.zig` siblings, one `test.zig` aggregator
  per directory, and a shared `fixtures.zig` for the test-only helpers. The
  production files hold no `test` block; the aggregator is pulled by `main.zig`.

## Steps

1. The rename `device/` -> `gpu/` (+ the importers: `cli/{inspect,tune}`,
   `main.zig`, `runtime/{executor,session}`, `tuner/calibrate`; + the
   constitution's boundary). **Done.** `src/gpu/` replaces `src/device/`; the
   CONSTITUTION (`ARTIFACT` layer + rules 4/5) and AGENTS `Layout` carry the
   `gpu/` boundary.
2. The backend contract (`gpu/backend/root.zig`) from the current `Device`'s ops.
   **Done.** The shared types (`Quant`, `GateUpAct`, `Resident`, `Handle`,
   `ResidentOp`, `RingSize`, `Probe`) + `required_ops`/`check` + the generic
   probe live there.
3. The Vulkan impl under `gpu/backend/vulkan/`. **Done.**
   `backend/vulkan/root.zig` is the Vulkan `Device` (the ops); `context.zig`,
   `pipeline.zig`, `types.zig` are the instance/queue, pipelines/ring and FFI.
4. The kernels: the grid (`gpu/kernels/grid.zig`) moved from the CPU side + the
   shaders (`gpu/kernels/shaders/` + `shaders.zig`). **Done.** `GpuVariant` +
   the catalog moved off `kernels/{api,linear/matvec/root}.zig`; the CPU family's
   catalog no longer carries GPU entries.
5. The test layout (`<mod>_test.zig` + `test.zig` + `fixtures.zig`). **Done.**
   `gpu/{root_test,backend/root_test,kernels/grid_test}.zig` + `gpu/test.zig`
   (pulled by `main.zig`); the production files are `test`-free. No shared
   `fixtures.zig` was needed (the only shared helper, `fillWeights`, is
   production: the probe uses it).
6. Verify: build/test green; the GPU plan unchanged; the probe. **Done.**
   `zig build` green, `zig build test` 280/280 (same as before), `zig fmt`
   clean on `src/gpu/`.

## Definition of Done

- [x] `gpu/` replaces `device/`; the importers + the constitution updated.
- [x] The backend contract is explicit; the Vulkan impl lives under
      `gpu/backend/vulkan/`.
- [x] The GPU kernel grid lives in `gpu/kernels/grid.zig` (moved off the CPU
      side); the shaders are the impl.
- [x] The test layout applies (the `<mod>_test.zig` + the aggregator + the
      fixtures); the production files are `test`-free.
- [x] A new backend would be a new `gpu/backend/<name>/` implementing the
      contract, with the gpu-facing API, the kernels grid and the runtime
      untouched.
- [x] Build/test green; the GPU behaviour unchanged.

## Arch Provenance

- **own:** the backend-ready gpu structure.
- **derived:** `gpu/02` (the as-built), `gpu/09` (the sub-calculator).

## Comments

- **The backend is a build-time selection** (a comptime type), not a runtime
  plugin — a new backend is a new directory, not a loaded library.
- **`gpu/kernels/`** = the GPU kernel grid (`grid.zig`: quant x variant) + the
  shaders (the impl) — the exact parallel of the CPU's `kernels/` (definition +
  grid together, the executor separate). It is **not** the CPU `kernels/`.
- **The test layout applies** (the CONSTITUTION Conventions): `<mod>_test.zig` +
  one `test.zig` aggregator + a shared `fixtures.zig`; the production files are
  `test`-free.
- **Mechanical first where possible:** the rename + the moves, then the contract
  extracted from the current `Device` and the grid moved off the CPU side.
- **As-built boundary decision:** `gpu/` imports `kernels/` (the `api.Kernel`
  catalog vocabulary + the matvec family's `AtomicId`), `gguf/` (the ggml type
  codes) and `data/`; `kernels/`/`gguf/`/`data/`/`utils/` do not import `gpu/`.
  `gpu/kernels/grid.zig` records the family's `AtomicId` rather than mirroring
  it, so the grid and the CPU family cannot drift.
- **As-built `Kernel` type:** the generic catalog entry stays in
  `kernels/api.zig`; the grid is a list of `api.Kernel` with `tier == .gpu`.
  `GpuVariant` moved from `kernels/api.zig` to the grid (it is GPU-specific).
- **As-built probe:** the probe logic is backend-agnostic, so it lives in the
  contract (`gpu/backend/root.zig:probeGpu`, generic over the device type) and
  the Vulkan `Device.probeGpu` wires it — the probe stays a backend op without
  duplicating the timing/enumeration code per backend.
- **As-built `Resident`:** the contract's `Resident` is a backend-neutral handle
  (`bytes` + an opaque `device` pointer); the Vulkan storage lives behind it.

09 - GPU Sub-calculator + Variant Axes

Status: done
Commits: (uncommitted)

## Goal

Two halves of the same thing:

1. **The sub-calculator.** The GPU is the runtime's **sub-calculator** — a small
   **op library** of **quant-aware** domain ops the runtime composes, submitted
   back-to-back and drained on demand (the `gpu/03` ring). Not dumb
   `sum`/`mult`: the quant decode is inherent to the format, so the primitive
   must decode on the GPU (`gemv(weight, x, quant)`).
2. **The variant axes.** Today a GPU variant is **baked per shader**
   (`plain`/`u4`/`u8` = separate `.comp` with a fixed `#define`). Make the
   variant a measured **grid** via Vulkan **specialization constants**: one
   shader, the axes as `layout(constant_id = N)`, the driver specializing at
   `vkCreateComputePipelines` — N variants, 0 extra shaders.

## Scope

- **In:** the op library; the `gpu/03` ring as the transaction; the variant axes
  (rows / unroll / reduction / workgroup) as specialization constants; the grid +
  the probe + the pre-creation prune.
- **Out:** a second backend (the `gpu/10` seam is the prerequisite); `iq`.

## Design

### The op library (quant-aware)

```text
gpu.gemv(w_q4_0, x, .q4_0)       // one Dispatch; the quant decode on the GPU
gpu.gate_up(g_q4_0, u_q4_0, x)   // the fused pair + the gated activation
gpu.gemm(w, x, m)                // the batched (the prefill)
gpu.drainProducer(ref)           // wait the in-flight op that wrote `ref` (the dep)
gpu.drainAll()                   // drain everything (token end / ring full / input change)
```
There is **no explicit `Transaction()`/`Commit()`**: the transaction **spans two
layers** — the **ring** (the gpu: the command-buffer slots + the fences) and the
**drain** (the **runtime's** executor: `drainProducer`/`drainAll` live in
`runtime/executor.zig`, waiting when a CPU op reads a GPU output). The gpu
submits; the runtime drains. Ops are submitted immediately (to overlap) and
drained on demand. The pseudo-API above is therefore **illustrative of the
composition**, not a gpu-only surface.

### The variant axes (the GPU grid)

| axis | what it is | note |
|---|---|---|
| `rows_per_group` | the tile (rows/workgroup) | the activation reuse (the CPU's `tile`) |
| `unroll` | the inner block loop | the ILP (the CPU's `unroll`) |
| **`reduction`** | shared-memory tree **vs** **subgroup/wave** (`subgroupAdd`/`subgroupBallot`) | **GPU-specific**; the shared tree + `barrier()` is expensive — the highest-value axis |
| `workgroup_size` | the lanes (32/64/128) | ~the CPU's `width` (the wavefront) |
| `int` / vectorization | the integer dot / the load width | experimental (low return on a mem-bound GEMV) |

There is **no `prefetch`** (the GPU hides latency with threads) and **no `width`**
in the CPU sense (the wavefront is fixed).

### The specialization constants

- `layout(constant_id = 0) const uint ROWS_PER_GROUP = 4;` etc. — one shader per
  (op, quant); the axes are constants the driver specializes.
- The pipeline is per **(op, quant, the axis values)**; the existing per-quant
  shaders become one shader each.

### The grid + the probe + the prune

- The grid = the op x quant x the axes; the **probe** measures each variant
  (min-of-N, the `gpu/08` selector) and the tuner records the choice.
- **Prune before pipeline creation**: creating N specialized pipelines costs (the
  driver compiles each); do not create a pipeline that cannot win.

## Steps

1. Convert one shader (`matvec_q4_0`) to specialization constants; parity.
   **Done.** `matvec_q4_0.comp` now declares `layout(constant_id = 0)
   ROWS_PER_GROUP`, `1 UNROLL`, `2 REDUCTION`, and `layout(local_size_x_id = 3)`
   (the workgroup size); the per-variant `matvec_q4_0_u4.comp/.spv` is deleted.
   The parity tests (`plain`/`u4`/`sub`) pass against the current code CPU kernel.
2. **The reduction axis first** (shared vs subgroup) — the GPU-specific lever;
   measure it. **Done.** `GpuVariant.sub` + the grid entry + `Quant.q4_0_sub`;
   the shader's `REDUCTION == 1` path uses `subgroupAdd` (with a cross-subgroup
   combine). Measured: neutral on the RX 580 (see Comments).
3. The axes `rows_per_group` / `unroll` / `workgroup_size` as constants.
   **Done.** `grid.SpecConsts` + `grid.specConsts(quant, variant)`; the
   `u4` variant is the same shader specialized to `ROWS_PER_GROUP = 8` (fixing
   the old tile-vs-unroll naming drift). `workgroup_size` is a real
   `local_size_x_id` constant (64); the backend accepts it and parity holds.
   **The remaining axis values are instantiated and measured:** `un2`/`un4`
   (inner k-block unroll 2/4, each unroll lane its own accumulator chain) and
   `w32`/`w128` (workgroup size 32/128 vs the base 64); the RX 580 accepts all
   three sizes and every variant is parity-correct.
4. The pipeline per (op, quant, constants) + the grid + the probe + the
   prune. **Done.** `pipeline.zig:createSpecializedPipeline` sets
   `VkSpecializationInfo`; the q4_0 shader yields the `plain`/`u4`/`sub`/
   `un2`/`un4`/`w32`/`w128` pipelines (one shader, N pipelines). The probe
   enumerates the grid; the prune skips the subgroup pipeline when the device
   lacks Vulkan 1.1 subgroup arithmetic and leaves a rejected workgroup size's
   pipeline null.
5. Verify: parity; the clock/offload behaviour; no regression at `--vram 4gb`.
   **Done.** `zig build` green; `zig build test` 283/283 (Debug; the 8
   pre-existing `prefix_cache`/`zpc` ReleaseFast crashes are unrelated).
   E2B decode at `--vram 4gb` ~23.5 tok/s (matches the ~23.8 baseline);
   12B ~5.1 tok/s (matches ~5.4).

## Definition of Done

- [x] The GPU axes are specialization constants (rows/unroll/reduction/workgroup);
      one shader, N pipelines.
- [x] The **reduction-by-subgroup** variant exists and is measured.
- [x] The **unroll** (2/4) and **workgroup-size** (32/128) axis values exist, are
      parity-correct and are measured (neutral on the RX 580).
- [x] The grid is probed; the Selector picks; the prune avoids dead pipelines.
- [x] No regression at `--vram 4gb`.

## Arch Provenance

- **own:** the sub-calculator + the GPU variant axes.
- **derived:** `gpu/03` (the ring), `gpu/08` (the selector), `gpu/10` (the
  structure + the seam), `kernels/17`–`21` (the CPU grid).

## Comments

- **The reduction (subgroup) is the highest-value axis** — GPU-specific and pure
  ALU/synchronization, which the CPU grid has no analogue for.
- **The spec constants are the GPU-native "comptime"** — GLSL has none; the
  driver specializes per pipeline.
- **The `u4` name is inconsistent today.** The q4_0's `u4` is a **tile**
  (`ROWS_PER_GROUP 8`); the gate_up's `u4` is an **inner unroll** (the tile stays
  4). The axes here (`rows_per_group` + `unroll`) cover both, so converting to
  spec constants also **fixes the naming**.
- **`workgroup_size`**: the current shaders are 64 lanes (the AMD wavefront);
  verify the backend accepts 32/64/128 before making it an axis. **As-built:**
  it is a real `local_size_x_id` spec constant, specialized to 64 (the base);
  parity holds, so the backend accepts and honors it. 32 and 128 are now
  catalogued (`w32`/`w128`) and the RX 580 accepts both.
- **As-built (the reduction axis is neutral on this host).** The subgroup
  reduction is correct (parity) and measured, but on the RX 580 it is **not
  faster** than the shared-memory tree: interleaved min-of-N gives plain (shared)
  **22.2 GB/s** vs `sub` (subgroup) **22.2 GB/s** (E2B) and **43.4 vs 43.4**
  (12B). The GEMV is memory-bound, so the tree's `barrier()` cost is hidden; the
  reduction is a tiny fraction of the time. Decode is likewise neutral (E2B
  23.6 plain vs 23.8 sub tok/s).
- **The probe ordering bias was real and is fixed.** The old q4_0 variant probe
  measured `plain`/`u4`/`sub` **sequentially**, so the RX 580's slow memory-clock
  ramp flattered the last-measured variant — the first `sub` reading looked +32%
  faster, which was pure ordering. `probeResidentVariants` now measures the
  q4_0 variants **interleaved** (like the gate_up probe), which also demotes the
  8-row `u4` tile (16.5 vs 22.2 GB/s) to where it belongs.
- **As-built (the UNROLL and WORKGROUP_SIZE axes are neutral on this host).**
  The shader's `UNROLL` now drives **independent accumulator chains** over
  consecutive k-blocks (the CPU sweep's `unroll`, the ILP lever) and
  `WORKGROUP_SIZE` is the `local_size_x_id`; `un2`/`un4`/`w32`/`w128` are
  catalogued, parity-correct and measured interleaved. The q4_0 GEMV is
  memory-bound, so neither axis wins: warm interleaved min-of-N gives base
  (wg64, unroll 1) **34.3 GB/s** vs `un2` 33.4, `un4` 33.3, `w32` 32.2, `w128`
  33.2 GB/s (E2B); the probe keeps `plain`. **Cold-start caveat:** on the very
  first probe after an idle period the memory clock is still ramping across the
  timed pass, so the variants measured later within each interleaved round (the
  appended `w32`/`w128`) are flattered (first run: base 21.9 vs `w128` 23.6
  GB/s) — the same ordering artifact the sequential->interleaved fix addressed,
  residual because the ramp spans the whole pass; re-running warm removes it.
  **Tile interaction:** the unroll multiplies the accumulator chains with the
  row tile, so the unroll variants stay at the 4-row tile (an 8-row tile x
  unroll 4 = 32 chains would spill, per the gate_up `u4` note).
- **As-built (one shader, N pipelines).** `matvec_q4_0.spv` alone yields the
  `plain`, `u4`, `sub`, `un2`, `un4`, `w32` and `w128` pipelines via
  `VkSpecializationInfo`; the `matvec_q4_0_u4.comp/.spv` shaders are gone. The
  subgroup pipeline is pruned (not created) when the device lacks Vulkan 1.1
  subgroup arithmetic; a workgroup size the device rejects would likewise leave
  its pipeline null (the RX 580 accepts 32/64/128).

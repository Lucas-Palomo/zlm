08 - Co-processor Selector

Status: done
Commits: 0c8f11c, b1c8624, 01a522d (the consolidated variant work), 9f479e5, 30a3082

## Goal

Unify how a GPU (co-processor) kernel is chosen with how the CPU chooses one: a
variant is an **AtomicOp** selected by the **Selector** and recorded **per
dispatch** — not a plan-level bitmask. This consolidates `gpu/08` + `gpu/09` +
`gpu/10` (the ad-hoc unroll variants + per-quant ROI) into one mechanism that
lives in the Kernel Architecture vocabulary (`CONSTITUTION.md`: Kernel /
AtomicOp / Selector / Composer).

Why: `gpu/08`–`gpu/10` grew a *second* variant mechanism beside the CPU's
`MatVecPlan` / `Dispatch.matvec_plan`: a plan-level `AOTKernel.gpu_unroll`
bitmask plus a per-op special slot, and every new variant bumped the plan format
(v15→v18) and `.zat` (v9→v12). Two mechanisms for one concept ("which AtomicOp
runs in this dispatch") is the wrong growth — and it drifts toward porting the
ggml kernel zoo, against "reference, not a port".

## Scope

- **In:** a device-agnostic **AtomicOp id per `Dispatch`**; the GPU shaders
  modelled as AtomicOps of the GEMV Kernel; one Selector (calibration + device
  probe) choosing for both CPU and GPU; dropping the plan-level bitmask.
- **Out:** new kernels/features; the CPU kernel catalog beyond the variant
  plumbing.

## Design

### One variant field per dispatch

- The chosen kernel variant is a property of the `Dispatch` (the role
  `Dispatch.matvec_plan` already plays for the CPU), covering the device too.
  Replace `AOTKernel.gpu_unroll` with a per-dispatch AtomicOp id, so a new
  variant **does not** change the plan format.

### GPU shaders are AtomicOps

- Each GPU GEMV shader (per quant, per variant) is an AtomicOp of the GEMV
  Kernel in the Registry, with the same `select`/`run` contract as the CPU
  AtomicOps. `device/` becomes the *implementation backend* of those AtomicOps,
  not a parallel policy layer.

### One Selector

- The tuner's Selector picks the AtomicOp per (quant, shape, device) from the
  calibration + the device probe — a single code path. The probe is just a
  measurement source for the GPU AtomicOps.

## History (consolidated from gpu/08/09/10)

- **`gpu/08` (0c8f11c).** Unrolled GEMV variants + per-quant ROI:
  `matvec_q4_k_u4` (+27% sustained), `matvec_q4_0_u4` (8-row tile; rejected —
  slower). The probe measured plain vs u4 and recorded a 1-bit-per-quant mask.
- **`gpu/09` (b1c8624).** q4_k 8x on both devices: CPU `q4_k_u8` (+6% over
  `production`), GPU `matvec_q4_k_u8` (+6% over u4, +73% over plain). The mask
  went to 2 bits/quant.
- **`gpu/10` (01a522d).** Unrolled fused `gate_up` (`matvec_gate_up_q4_0_u4`):
  inner-loop unroll, tile kept at 4 (an 8-row tile spills). Fused-op +13–16%;
  decode E2B +4.7% / 12B +5.8%. Its own 2-bit slot.
- **Outcome.** At `--vram 4gb`, E2B 14.4 → **23.3** tok/s, 12B 3.6 → **5.4**
  tok/s. The end-to-end gain tracked the *heaviest* dispatch (the fused op), not
  each shader — the per-op variant zoo has diminishing returns; the mechanism
  should be uniform.

## Steps

1. **Done.** `Dispatch.matvec_plan` -> `Dispatch.variant` (device-scoped: CPU
   `MatVecPlan`, GPU `device.GpuVariant`); the Selector writes it per dispatch.
2. **Done.** The GPU GEMV shaders are catalogued as `tier = .gpu` AtomicOps of
   the matvec family (`gpu_kernels`: one entry per op x quant x `GpuVariant`,
   with `applicable`/`cost` as data). `GpuVariant` now lives once in
   `kernels/api.zig`; `device/` re-exports it. The probe enumerates the catalog
   (`catalogGpuVariants`) instead of a literal variant list, so a new variant of
   an existing op/quant is a catalog entry plus a shader. The Selector stays
   CPU-only; `decideOffload` remains the transfer-aware placement gate (the pure
   `cost`/`applicable` signatures carry no device/residency state). The runtime
   still maps `(quant, variant)` to the Vulkan pipeline via `gpuQuant`. CPU plan
   bit-identical; 260/260; E2B ~23 / 12B 5.4 tok/s at `--vram 4gb` (unchanged).
3. **Done.** `AOTKernel.gpu_unroll` removed (plan v18->v19; `.zat` v12->v13);
   one bump, ending the per-variant churn.
4. **Done.** CPU plan **bit-identical**; 258/258; E2B ~23.3 / 12B ~5.5 tok/s at
   `--vram 4gb` (unchanged). The probe picks q4_0 plain / q4_k u8 / gate_up u4.

## Definition of Done

- [x] A variant is a per-dispatch AtomicOp id; no plan-level variant bitmask.
- [~] GPU shaders are AtomicOps selected by the same Selector as the CPU —
      **partial**: they are catalogued (`tier = .gpu`); the Selector stays
      CPU-only and `decideOffload` remains the placement gate (see Comments).
- [x] Adding a variant no longer bumps the plan format.
- [x] E2B/12B decode at `--vram 4gb` measured; CPU plan unchanged; tests green.

## Arch Provenance

- **own:** the unified co-processor Selector.
- **consulted:** the ggml kernel-variant zoo (reference only; we unify instead).
- **derived:** `gpu/08` + `gpu/09` + `gpu/10` (consolidated here), `gpu/02`.

## Comments

- Consolidates `gpu/08` / `gpu/09` / `gpu/10`; those files are removed and their
  as-built recorded above (git history: 0c8f11c, b1c8624, 01a522d).
- **Residual (DoD 2).** The GPU AtomicOps are catalogued, but the variant
  *choice* (the probe) and the *placement* (`decideOffload`) stay separate from
  the Selector. A single Selector scoring CPU+GPU by cost needs
  `ApplicableFn`/`CostFn` to carry device/residency state — deferred. A new
  **quant** (not just a variant) additionally needs a `GpuCoeffs` field, the
  tuner's quant->variant switch, and the `decideOffload`/`eligibleResident`
  gate. See `sdd/open-items.md`.

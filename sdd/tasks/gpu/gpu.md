# gpu

Status: done

Reason: the GPU is a **co-processor** — an **extension of the CPU's processing
unit**, not a model host. Unlike llama.cpp (which offloads the maximum and needs
VRAM ≥ the offloaded weights), zlm keeps the CPU as the maestro and uses the GPU
for the ops where it pays, so **old / low-VRAM GPUs still help**. `--vram`
delegates how much GPU the runtime may use to extend the CPU. CPU-first, POSIX.

```text
Processing unit
    +-------------------+
    |  CPU              |
    |      +-------+    |
    |      |  GPU  |    |
    |      +-------+    |
    +-------------------+
```

## Tasks

### [01 - GPU Co-processor](01-coprocessor.md)
Status: done

Context: the model — per-op offload (transfer-aware), VRAM as a budgeted
scratch/resident region with DRAM spillover, chunked weight slicing, CPU
fallback, and the AOT plan (`device` + tile per dispatch). Grounded in the earlier local prototype
(its `src/gpu/`).

### [02 - Co-processor Implementation (as-built + pending)](02-co-processor-implementation.md)
Status: done

Context: the delivered implementation — Vulkan backend, AOT offload, VRAM ROI
residency, transfer-aware cost + `tile`, batched prefill, cold-start fix — and
what remains (dual-stream/async, VRAM KV + activation scratch, `--vram` in
`planBudget`, GPU calibration, `q6_k` batched GEMM).

### [03 - Async Dispatch and Dual-Stream](03-async-and-dual-stream.md)
Status: done

Context: remove the per-dispatch `vkQueueWaitIdle` (fence-based async submit,
device-owned activations) and overlap independent CPU and GPU dispatches via the
plan's deps. Fence async + a resident command-buffer ring + the q/k/v batch are
landed; device-owned activations and the dep-driven wait remain.

### [04 - Heterogeneous Compute + GPU-aware RAM Budget](04-heterogeneous-compute.md)
Status: done

Context: run one op on both the CPU pool and the GPU over disjoint output rows
(concurrently, the `gpu/03` overlap), **and redesign the RAM budget** when `--gpu`
so it holds the **split ops' weights** instead of the KV — the RAM buys
**throughput** (the heterogeneous) rather than **context**. The duplication is
**precise** (only the split ranges). **Measured**: the split wins at a large
context; the per-op device oracle (`decideDevice`), the row-range residency, the
4-profile telemetry and the re-scoped budget landed. The prefill GEMM split is
not pursued (one-shot); the chain scheduler continues in `gpu/11`. Depends on
`gpu/03` + `gpu/05` (the exclusion).

### [05 - VRAM Extended by RAM (spillover / virtual_vram)](05-vram-spillover.md)
Status: done (closed — the ROI does not pay)

Context: **closed by the ROI finding.** The spillover streams the RAM to the VRAM
over the PCIe (~8 GB/s), **slower than the CPU reading the RAM (~34 GB/s)** — a
negative ROI at any model size. The real ROI is the **budget**: one tier per
weight (VRAM -> GPU, RAM -> CPU, disk -> the rest), no duplication. **The bind
finding (measured)** closes the direct-RAM-read variant too: the shader reading a
host-visible buffer is **5.43 GB/s** (vs the CPU's 35), the GPU reads **only
weights** (no attention kernel, so the K/V is not a bind candidate), and a bigger
KV is a **decode cost** (`--ram` 4 -> 12 GB: 5.9 -> 1.1 tok/s). No RAM segment
pays as "bind only".

### [06 - VRAM Budget Saturation](06-vram-budget-saturation.md)
Status: done

Context: the budget split — **RAM is an efficiency budget** (spend what is
useful), **VRAM is a saturation budget** (use the maximum, proactively resident,
never on-demand). Saturate `--vram` with the best resident regions and make a
resident weight offloaded by default (the GPU executes it from VRAM); the
measured cost gates only streamed ops. Depends on `gpu/02` + `gpu/03`.

### [07 - Fused FFN gate_up Offload](07-fused-ffn-offload.md)
Status: done

Context: the decode's fused `gate_up` (ffn_gate + ffn_up, ~half the decode GEMV
traffic) is CPU-only; offload it (a fused two-weight GEMV shader) so the GPU
runs ~all the decode GEMV traffic. This raises the GPU duty cycle past the ~25%
the RX 580's SMU needs to ramp the memory clock (a game ramps it to 1750 MHz;
our compute sits at 400). Then the GPU (~75 GB/s) beats the CPU (~38). Also the
remaining `q4_k`/`f32`/`bf16` projections. Depends on `gpu/06`.

### [08 - Co-processor Selector](08-co-processor-selector.md)
Status: done

Context: unify the GPU kernel-variant mechanism with the CPU's — a variant is an
**AtomicOp** selected by the **Selector** and recorded **per dispatch**, not a
plan-level bitmask. Consolidates the ad-hoc unroll/ROI work (`gpu/08`+`09`+`10`,
whose as-built is recorded in the spec) and drops `AOTKernel.gpu_unroll`. Depends
on `gpu/07`.

### [09 - GPU Sub-calculator + Variant Axes](09-gpu-sub-calculator.md)
Status: done

Context: two halves — (1) the GPU as the runtime's **sub-calculator** (a small
op library of **quant-aware** domain ops submitted on the `gpu/03` ring, drained
on demand; no explicit Transaction/Commit) and (2) the **variant axes** as
Vulkan **specialization constants** (one shader, the driver specializes per
pipeline), making the variants a measured **grid**: `rows_per_group` x `unroll` x
**`reduction` (shared vs subgroup — the GPU-specific lever)** x
`workgroup_size`. Today a variant is baked per shader (plain/u4/u8). Depends on
`gpu/08` + `gpu/10`. **Done:** the q4_0 GEMV is one spec-constant shader
(`plain`/`u4`/`sub`/`un2`/`un4`/`w32`/`w128` pipelines); the reduction, unroll
and workgroup-size axes are parity-correct but neutral on the RX 580 (the GEMV
is memory-bound).

### [10 - gpu Structure (backend-ready)](10-gpu-structure.md)
Status: done

Context: rename `device/` -> `gpu/` and restructure so a new backend is a
drop-in: the gpu-facing API (`gpu/root.zig`), the **backend contract**
(`gpu/backend/root.zig`), the Vulkan impl (`gpu/backend/vulkan/`), and the GPU
kernels (`gpu/kernels/`). A new backend (Metal, ...) = a new `gpu/backend/<x>/`,
the runtime untouched. Depends on `gpu/02`.

### [11 - Device Scheduler (heterogeneous chain scheduling)](11-device-scheduler.md)
Status: done

Context: turn `gpu/04`'s **per-op** device oracle into a **chain scheduler** —
the whole decode chain decided at once so the plan does not ping-pong between
the devices for a marginal per-op gain. It minimizes the makespan with the
**switch cost** (the activation transfer + the drain), the **setup cost** (a
measured constant + a per-op model), the **dependencies**, and a **dynamic
look-ahead** (a receding-horizon DP whose window opens at the next heavy op).
The output is the **runs**: `cpu` / `gpu` / `split` / `mixed`. The executor's
pre-record is **deferred** (the `gpu/03` async already overlaps the execution).
Depends on `gpu/04`; continues `runtime/12`'s Scheduler.

**Done + findings:** the model (a Viterbi over `cpu | gpu | split`, the RAM/VRAM
policy with shadow prices), the **`split_efficiency`** probe (the concurrent
CPU+GPU rate over the GPU-only rate — **0.70** on the RX 580, so the tuner keeps
every op whole on the GPU), and the tuner wiring. The split is a **capability**
(a machine where the bands add would split), and the performance lever is the
**VRAM resident set** (12B: VRAM 4 GB = 5.8, 8 GB = **8.7** tok/s; the RAM is
free context). Not built: the measured `hop_ns` and the `runs` telemetry.

### [12 - Variant Grid for All GEMV Quants (spec constants)](12-spec-constant-grid-all-quants.md)
Status: done

Context: `gpu/09` made the GPU GEMV variant a measured grid via specialization
constants, but only for `q4_0`; every other quant (`q4_1`, `q8_0`, `q6_k`,
`f32`, `f16`, `bf16`) and the fused `gate_up` still ship fixed-`#define`
shaders, so their grid is a single `plain` entry. Finish it: one spec-constant
shader per quant, the same axes, the same probe, the same per-dispatch variant —
a uniform grid. Depends on `gpu/09` + `gpu/07`.

### [13 - Heavy-duty GPU Warm-up (sustained clock ramp)](13-heavy-duty-gpu-warmup.md)
Status: done

Context: a discrete device's memory clock only ramps under sustained activity,
so a `--gpu` plan records `Config.gpu_warmup` and the runtime runs a sustained
resident-GEMV load at load, before the first token. Depends on `gpu/06` + `gpu/07`.

### [14 - Fused SwiGLU MLP (gate_up + activation + down in one dispatch)](14-fused-mlp-swiglu.md)
Status: done

Context: the FFN MLP is two dispatches (`gate_up_fused` -> `ffn_down`), so the
`ffn_mid` round-trips and, in the `mixed`, crosses the PCIe every layer. Fuse
the whole SwiGLU MLP into one dispatch (the activation + Hadamard in
registers/shared, fed to the `down`) — like vLLM/TensorRT-LLM/FlashInfer. New
`mlp_fused` op (three weights), a Composer recipe, the CPU + GPU kernels.
Depends on `gpu/07` + the Composer.

### ~~08 - GPU Shader Variants (unroll) with ROI Selection~~ — consolidated into `gpu/08`
### ~~09 - q4_k 8x Variant (CPU + GPU)~~ — consolidated into `gpu/08`
### ~~10 - Fused gate_up Unroll Variant~~ — consolidated into `gpu/08`

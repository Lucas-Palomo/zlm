02 - Co-processor Implementation (as-built + pending)

Status: done

**Closed.** The as-built landed: the Vulkan backend with per-op parity, the `device`/`tile` recorded by the tuner, the ROI VRAM residency, the batched-prefill offload and the cold-run deadline. The dual-stream/async landed (`gpu/03`); the "VRAM KV + activation scratch" is closed with the spillover (`gpu/05`) and handled by the saturation (`gpu/06`).
Commits: 56c0306 c265d33 44b81e7 be347d6 b38a92c

## Goal

Implement the `gpu/01` model on the current seams: a working Vulkan co-processor
backend, the AOT per-op offload decision, VRAM-resident weights selected by ROI,
a transfer-aware cost, batched prefill on the GPU, and a robust cold start. This
spec is the as-built record and the list of what remains; `gpu/01` is the frozen
model.

## Scope

- **In (delivered):**
  - `src/device/`: a Vulkan backend — dynamic `libvulkan` (`dlopen`, needs
    `link_libc`), embedded SPIR-V, a backend-agnostic `Device`
    (`init/deinit/available/matVec/matVecResident/matMulBatch/matMulResident/
    sync`, `.none` fallback), streamed + resident GEMV and batched GEMM.
  - Plan fields: `Dispatch.device`, `Dispatch.tile`, `AOTKernel.vram_resident` +
    `vram_resident_bytes`.
  - Tuner: `tuner/gpu.zig` (offload policy, ROI residency knapsack, `tile`).
  - Executor: per-`device` routing with CPU fallback (decode GEMV + prefill
    GEMM); `Session` owns the `Device` and uploads the resident set.
  - Cold-start deadline scaled by the plan's resident bytes.
- **Out (pending, see Comments):** dual-stream/async; VRAM KV cache + activation
  scratch; `--vram` fully inside `planBudget`; GPU calibration in the `.zat`;
  `q6_k` batched GEMM; UMA hardware; multi-GPU/network.

## Design (as-built)

```text
tune --gpu:  Device class (UMA/discrete) ->
             ROI knapsack: resident weights packed into --vram -| 64 MiB
             per decode GEMV: decideOffload(cpu vs gpu+transfer+sync) -> device
             per prefill GEMM: decideOffloadMatMul -> device
             exceeds a staging slot -> tile rows
run:         Session owns the Device; uploads the resident set once
             executor: device!=0 -> resident|streamed|batched GPU, else CPU
```

Measured on the AMD RX 580 (RADV Polaris10):

| | CPU | GPU |
|---|---|---|
| prefill (qwen3-4B, ~60 tok) | ~2.9 tok/s | ~9.7–13 tok/s |
| decode (qwen3-4B) | ~0.8 tok/s | ~0.2 tok/s |
| decode (gemma4-E2B) | ~14 tok/s | ~5 tok/s |

Prefill wins (~4×, weights amortized over the block). Decode loses because each
GPU dispatch pays a `submit + vkQueueWaitIdle` (dual-stream pending).

## Steps (delivered)

1. **Backend** (`56c0306`). Vulkan context/pipeline/types + embedded SPIR-V;
   streamed GEMV for `q4_0/q4_1/q8_0/q6_k`; guarded GPU-vs-CPU parity tests run on
   the RX 580. The earlier local prototype's `q6_k` shader was rewritten to match the current code's `dequantQ6_K`;
   the earlier local prototype's `last_weight_ptr` cache was dropped (stale-weight bug).
2. **Offload** (`c265d33`). `Dispatch.device` set at `tune --gpu`; `Session`
   device; `executor.runMatvec` routes + CPU fallback; the KV arithmetic
   fingerprint includes `device`.
3. **VRAM ROI** (`44b81e7`). `vram_resident` (plan v14); ROI knapsack (resident
   weights, largest transfer first, `--vram -| 64 MiB`); resident GEMV upload once
   (`device.Resident`); qwen3-4B: 217/507 decode offloaded, ~1.9 GB resident.
4. **Transfer-aware cost + `tile`** (`be347d6`). `tuner/gpu.zig:decideOffload`
   (CPU `len/bw_cpu` vs GPU `transfer(0 if resident)/pcie + compute + sync`) and
   `tileRows` + `matVecTiled`; plan v15.
5. **Batched prefill + cold start** (`b38a92c`). `matmul_{q4_0,q4_1,q8_0}`
   shaders + `matMulBatch`/`matMulResident`; `decideOffloadMatMul`;
   `selectPrefillDispatch` offloads 252/252 prefill GEMM; the daemon deadline
   scales with `vram_resident_bytes`.

## Definition of Done

- [x] A working Vulkan backend with per-op GPU-vs-CPU parity (matvec + batched
      GEMM); `zig build` / `zig build test` green.
- [x] The plan records `device` + `tile`; the tuner sets them deterministically.
- [x] VRAM-resident weights selected by ROI within `--vram -| headroom`; uploaded
      once; `--vram` too small degrades the GPU, not the weights.
- [x] A dispatch the GPU cannot run (or does not win) runs on the CPU.
- [x] Batched prefill offload (transfer amortized) with a CPU fallback.
- [x] A GPU plan's first cold `run` succeeds (deadline scaled).
- [ ] Dual-stream/async (decode GPU ≥ CPU).
- [ ] `--vram` fully in `planBudget`; VRAM KV cache + activation scratch regions.

## Arch Provenance

- **own:** the current seam mapping (`Device`, plan fields, tuner policies, executor
  routing), the transfer-aware cost, the ROI packer.
- **consulted:** the earlier local prototype's `src/gpu/` (the PoC: `shouldOffload`, the ROI
  knapsack, `chunk_rows`, `ExecutionPlan`) — reference; re-derived.
- **spec:** Vulkan compute; PCIe/UMA transfer models.
- **derived:** `gpu/01`, `kernels/11`, `runtime/06`, `runtime/10`, `runtime/12`.

## Comments

- **Delivered round.** Backend + AOT offload + VRAM ROI + transfer-aware cost +
  `tile` + batched prefill + cold-start fix; verified on the RX 580 with qwen3-4B
  and gemma4-E2B. The CPU path is untouched (all GPU branches are `device != 0` /
  `plan.config.gpu`).
- **Pending — dual-stream / async (highest value).** The per-dispatch
  `submit + vkQueueWaitIdle` costs 200+ driver round-trips/token on decode, which
  is why decode is slower on the GPU. Overlap needs device-owned activation
  buffers, fence/timeline sync, and a dispatch deps DAG — not the sequential
  host-buffer executor. Marked `TODO(dual-stream)` in `device/pipeline.zig` and
  `runtime/executor.zig`.
- **Pending — VRAM regions beyond resident weights.** The earlier local prototype also packed KV cache and
  activation scratch into the VRAM ROI; here only resident weights are packed.
  `--vram` currently drives the resident set, not the full `planBudget`.
- **Pending — GPU calibration.** The cost uses documented constants
  (`PCIE_BW_GBPS = 8`, `GPU_TFLOPS = 3`) rather than a measured `.zat` profile;
  a device probe (peak flops, VRAM/PCIe bandwidth) would make the transfer term
  exact.
- **Pending — `q6_k` batched GEMM.** `gemma4`'s `token_embd` is `q6_k`; the
  batched GEMM covers `q4_0/q4_1/q8_0` only, so those stay CPU for prefill.
- **Note (operational).** Daemons are detached and persist across rebuilds; a
  stale daemon can be reused by `run` after a re-tune/rebuild. A guard (self-exit
  on plan-artifact change) is not yet implemented.

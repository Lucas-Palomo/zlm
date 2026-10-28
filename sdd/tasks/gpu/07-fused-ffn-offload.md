07 - Fused FFN gate_up Offload (raise the GPU duty cycle)

Status: done

**Closed.** The fused `gate_up` offloads (parity) and raises the duty cycle so the RX 580 `mclk` ramps to 1750 MHz during decode; the E2B/12B decode is measured (GPU vs CPU).
Commits:

## Goal

The decode's fused `gate_up` matvec (`ffn_gate` + `ffn_up`, q4_0) is **CPU-only**
because the tuner and executor skip a two-weight dispatch (`tensor2 != NONE`).
It is the single biggest CPU GEMV: **~560 MB/token** on gemma4-E2B and **~3
GB/token** on gemma4-12B — about **half the decode GEMV traffic**. Offload it so
the GPU runs ~all the decode GEMV traffic.

Why it matters: the RX 580's memory clock stays at 400 MHz under our compute
(duty cycle ~5%) but ramps to **1750 MHz** under a game (GPU ~100% busy). The
power profile (`3D_FULL_SCREEN`) needs memory activity sustained above
`MCLK_ACTIVE_LEVEL=25` to ramp. With the clock ramped the GPU reads ~75 GB/s and
beats the CPU's ~38 GB/s. Offloading the fused FFN is the lever that raises the
duty cycle.

## Scope

- **In:** a fused two-weight GEMV shader (`matvec_gate_up_*`) with the gated
  activation; the tuner treating the fused `gate_up` as an offload/resident
  candidate (both tensors); the executor's async path handling the fused op.
- **Secondary (same task, after the fused FFN):** the remaining CPU-only GEMV
  quants — `q4_k` (`token_embd` output projection, 216 MB E2B / 540 MB 12B) and
  `f32`/`bf16` (`inp_gate`/`proj`, `per_layer_model_proj`).
- **Out:** prefill fused GEMM (`gpu/02`); splitting an op across devices
  (`gpu/04`); VRAM spillover (`gpu/05`).

## Design

### Fused GEMV shader

- New `src/device/shaders/matvec_gate_up_q4_0.comp` (base it on `matvec_q4_0.comp`,
  row-parallel/coalesced). For each output row `i`: `g = dot(gate_row_i, x)`,
  `u = dot(up_row_i, x)`, `out[i] = gelu_tanh(g) * u` — matching the CPU
  `fusedGateUpRows` (`src/kernels/linear/matvec/root.zig`, the `gate_up_fused`
  atomic; `activation.gatedGelu`).
- Add the other quants (`q4_1`/`q8_0`/`q6_k`) only if a model's `ffn_gate`/`up`
  use them; gemma4 uses q4_0.

### Tuner

- The resident packer must place the `gate_up` **pair** together (both tensors
  resident) — extend `Candidate`/`selectResident` or add the pair as one
  candidate carrying both tensor indices.
- `decideOffload`/`eligibleResident`: admit the fused pair (resident => offload,
  as in `gpu/06`).

### Executor

- `asyncResidentGemv` currently requires `tensor2 == NONE`. Extend the async
  burst path to submit the fused `gate_up` (two resident handles, one dispatch),
  keeping the shared input activation device-owned.

## Steps

1. Shader + `.spv` (glslc -O) + the pipeline dispatch (`gate_up_fused` atomic).
2. Tuner: the pair as a resident candidate + the offload decision.
3. Executor: the async fused submit.
4. Parity (CPU vs GPU) for the fused op; `zig build` / `zig build test` green.
5. Measure: the GPU duty cycle (`mem_busy_percent`) and `mclk` during a run, and
   the decode tok/s (E2B, 12B), ReleaseFast.

6. **Secondary scope (as-built).** New `matvec_q4_k` (one row per 64-lane
   workgroup, 144-byte block / packed 6-bit scale decode, matching
   `k_quants.dequantQ4_K`), `matvec_f32`, `matvec_f16` and `matvec_bf16`
   (one row per workgroup, unquantized weights) shaders + `.spv`; `device.Quant`
   gains `q4_k/f32/f16/bf16`; `pipeline` registers the pipelines and the GEMV
   switch/matvec-only dispatch; `gpuQuant`, `eligibleResident` and
   `decideOffload` accept them. The `f32`/`bf16` weights (`inp_gate`/`proj`,
   `per_layer_model_proj`) and the `q4_k` tied `token_embd` lm_head are now
   residency candidates (`bf16`/`f32`/`q4_k` were previously CPU-only).
   CPU-vs-GPU parity added for all four quants on the RX 580; 255/255 tests.
   Results at `--vram 4gb` (ReleaseFast): E2B 170/749 -> 242/749 decode
   offloaded (all +72 new weights resident), resident 986 -> 1334 MB, decode
   16.7 -> 20.9 tok/s (CPU 16.3). 12B is budget-saturated: 84/868 -> 64/868
   (the 540 MB q4_k displaces ~20 per-token q4_0 weights), resident unchanged
   4031 MB, decode 4.8 -> 4.8 tok/s (CPU 4.0). The 12B q4_k is ~break-even:
   measured GPU q4_k GEMV ~40 GB/s vs q4_0 ~50 GB/s (ALU-bound packed-scale
   decode), so it only matches the CPU's ~38 GB/s DRAM read. mclk still ramps
   to 1750 MHz on both.

## Definition of Done

- [ ] The fused `gate_up` offloads (`device = 1`) and is parity-correct.
- [ ] The GPU duty cycle rises and the RX 580 `mclk` ramps above 400 MHz during
      a `run` (sysfs-measured) — or a precise reason why not.
- [ ] Decode tok/s (GPU vs CPU) measured for E2B and 12B.
- [ ] CPU plan unchanged; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the fused-offload + duty-cycle rationale.
- **consulted:** the CPU `fusedGateUpRows`/`gatedGelu`; the `matvec_q4_0` shader.
- **derived:** `gpu/02` (as-built), `gpu/03` (async), `gpu/06` (resident).

## Comments

- **Why the fused FFN first.** It is the largest CPU-only GEMV and the only way
  to move ~half the decode traffic to the GPU; the q4_k/f32/bf16 projections are
  smaller and follow the same pattern.
- **Environment.** The lever is the duty cycle; if offloading the bulk still
  leaves the mclk at 400, the decode DAG's CPU ops dominate and the answer is a
  later "warm" device state (deferred).

14 - Fused SwiGLU MLP (gate_up + activation + down in one dispatch)

Status: done
Commits:

## Goal

The FFN MLP is two dispatches today: `gate_up_fused` (gate + up + activation ->
the `ffn_mid` buffer) and `ffn_down` (`ffn_mid` -> the output). The intermediate
`ffn_mid` **round-trips** (written and re-read), and — when the two land on
different devices — **crosses the PCIe** every layer (the 31B's `mixed`: ~90
crossings/token).

Fuse the whole SwiGLU MLP into **one dispatch**:

```
y = (swish(x·W_gate) ⊙ (x·W_up)) · W_down
```

like vLLM / TensorRT-LLM / FlashInfer: the activation + Hadamard product stay in
registers/shared, fed directly to the `down` matvec — no global intermediate.

## Scope

- **In:** a new `mlp_fused` op (three weights `gate`/`up`/`down` + the gated
  activation); the CPU AtomicOp; the GPU shader (the `ffn_mid` in shared
  memory); the Composer recipe (fuse `gate_up` + `ffn_down`); the plan
  (`Dispatch` with `tensor`/`tensor2`/`tensor3`); the residency/offload for the
  fused op; the parity tests.
- **Out:** the MoE variant; fusing the `down` residual/norm; the PLE block.

## Design

### The op

One dispatch over `(gate, up, down)` weights and the activation `x`:

- phase 1: `m_j = act(Σ_k gate[j,k]·x_k) · (Σ_k up[j,k]·x_k)` for every ffn row
  `j` — the `gate`/`up` in parallel, the activation in registers;
- phase 2: `y_r = Σ_j down[r,j]·m_j` — the `down` over `m` (shared/registers, no
  global write).

The `ffn_mid` never touches global memory (the whole point). The `ffn` dim
(15360 on the 12B, ~21504 on the 31B, in f32) bounds the shared use
(`ffn·4` bytes): feasible on the RX 580 (64 KiB) for the 12B; the 31B needs a
two-pass or a `m`-tiled phase 2.

### The plan

`Dispatch` gains a third weight (`tensor3`, `plan.NONE` by default) and the op is
the new `mlp_fused` (a matvec-family AtomicOp). The resident/offload treats the
three weights as one candidate (like the `gate_up` pair today): a resident
`mlp_fused` is transfer-free; a streamed one is priced by the combined bytes.

### The Composer

The Composer emits the **separate** pair (`gate_up` + its consuming `ffn_down`);
it does **not** fuse the whole MLP. Whether the fusion pays depends on the
device (a discrete GPU's single-workgroup fused shader loses to the
multi-workgroup pair), so the choice is made **per instruction** by the tuner's
cost model (`gpu_policy.mlpFusionPays` + `gpu_policy.fuseMlps`), not by the
hardware-independent Composer. Both kernels stay: a layer may use the fused
`mlp`, another the separate pair. A plan that fuses nothing is byte-for-byte
the pre-`mlp` decode program.

### The devices

- **CPU:** a fused `fusedMlpRows` (the `gate`/`up` per row, the activation, then
  the `down` over the computed `m`) — the `m` in a scratch buffer (the CPU has no
  register-resident requirement; the win is one pass + no dispatch split).
- **GPU:** a fused shader (phase 1 the `gate`/`up`, the `m` in shared, phase 2 the
  `down`); the `gate`/`up` may reuse the existing fused-`gate_up` decode.

## Steps

1. The plan: `Dispatch.tensor3` + the `mlp_fused` op/atomic; serialize.
   **As-built:** `Dispatch.tensor3` (serialized after `tensor2`), plan `VERSION`
   26 -> 27; `graph.Op.mlp` + `Instr.role3`; `matvec.AtomicId.mlp_fused`.
2. The Composer recipe + the test (the gemma4 block fuses its MLP).
   **As-built (revised):** the Composer recipe was **removed** — it fused
   unconditionally, which forced the fused form on every device (the regression).
   The Composer now emits the separate pair; the tuner's per-op cost pass fuses
   it when it pays. Gemma4 stays at 3 recipes/layer (norm+rope, gate_up).
3. The CPU AtomicOp + parity.
   **As-built:** `fusedMlpRows` (the fused `gate_up` into a scratch `m`, then
   the `down`); parity vs gate_up+down in `form/fused_test.zig`.
4. The GPU shader + parity.
   **As-built:** `matvec_mlp_{q4_0,q4_1,q8_0}.comp` (the `m` in shared, phase 2
   the `down`), the grid entries, the pipeline (a 5th binding for the `down`),
   `matVecMlpResidentBurstAsync`; parity in `gpu/root_test.zig` (RX 580, all 7
   variants × 3 quants).
5. The tuner: the residency/offload for the fused op (one candidate, 3 weights).
   **As-built:** `decideOffloadMlp` + `costOptions(.mlp)`, `Candidate.tensor_index3`,
   `addResidentTripleCandidate`, the residency graph now tracks `tensor3`.
   **As-built (revised, the review items):** `gpu_policy.mlpFusionPays` (a real
   cost comparison of the fused op vs the pair) + `gpu_policy.fuseMlps` (the
   per-op merge over the decode dispatch); `tune.zig` builds the weight metadata
   and applies it before device selection. The fused `.mlp` never splits
   (`costOptions(.mlp).split_ns == inf`, `decideDevice(.mlp) != .split`) and its
   three weights are one all-or-nothing residency candidate, ranked first by the
   packer's `value`.
6. Verify: parity; the e2e decode; `zig build` / `zig build test` green.
   **As-built:** 297/297 tests green; gemma-12B decode on the RX 580 (discrete):
   7.6 tok/s decode (the always-fused form was 4.4). The fused `mlp` is correctly
   **not** used on a discrete GPU (0 `mlp_fused`, 48 `gate_up_fused` in the
   plan); it is selected on UMA.
7. The discrete two-kernel form (the last review item): the atomic `mlp` runs as
   the existing fused `gate_up` then the existing `ffn_down` GEMV on **one
   device**, the `ffn_mid` in a device-local buffer (VRAM), so both kernels keep
   full workgroup parallelism and the intermediate never crosses the PCIe.
   **As-built:** `gpu/backend/vulkan` gains `matVecMlpPairResidentBurstAsync` /
   `submitResidentMlpPairShared` (a second descriptor set per ring slot, a
   device-local `burst_mid`, a compute-write -> compute-read barrier between the
   two dispatches); the shaders are unchanged. `executor.submitMlpAsync` picks
   the fused shader on UMA and the two-kernel form on a discrete device. The
   tuner allows a discrete `mlp` (`decideOffloadMlp`/`costOptions(.mlp)`) and the
   `down` may keep its own GEMV quant; `mlpFusionPays` fuses a discrete pair only
   when it would cross devices, and `fuseDecodeMlps` iterates the fusion to a
   fixed point (the packer's choice moves as pairs fuse). 299/299 tests green
   (the two-kernel and mixed-quant parity added). gemma-12B: plan **byte-identical**
   (no crossing there, so no fusion) and 7.5-7.6 tok/s decode; gemma-31B
   `g31b-t8r16v6`: 60/60 MLPs fused (28 whole-GPU, 32 whole-CPU, **0** crossings)
   at 2.1 tok/s decode vs 2.0 unfused.

## Definition of Done

- [x] The tuner fuses `gate_up` + `ffn_down` into `mlp_fused` **per op**, when
      the cost model says it pays (both kernels stay).
- [x] The CPU + GPU kernels are parity-correct.
- [x] The `ffn_mid` does not round-trip through global memory.
- [x] The 31B `mixed` no longer crosses the PCIe for the MLP.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the zlm fused MLP.
- **consulted:** vLLM / TensorRT-LLM / FlashInfer fused SwiGLU (the technique).
- **derived:** `gpu/07` (the fused `gate_up`), the Composer, `kernels/`.

## Comments

- **The gain is the dispatch count + the intermediate, not the VRAM bandwidth on
  the decode.** For a single token the `ffn_mid` is ~60-86 KB vs the ~187 MB of
  weights/camada — the saved round-trip is ~0.05% of the bandwidth. The real wins:
  one dispatch instead of two (the setup), and — in the `mixed` — keeping the
  whole MLP on one device (no `ffn_mid` crossing).
- **Shared bound.** The `m` in shared is `ffn·4` bytes (60 KiB on the 12B,
  86 KiB on the 31B) — the 31B exceeds the RX 580's 64 KiB; a `m`-tiled phase 2
  (or a two-pass) is needed there.
- **The fused GPU shader is single-workgroup (shared `m`).** The `m` in shared
  forces one workgroup to read all three weights through one CU; on the RX 580
  that loses badly to the multi-workgroup `gate_up`+`down`. The shader stays the
  **UMA** path (parity-correct; exercised by `gpu/root_test.zig`); on a discrete
  device the same atomic `mlp` now runs as the two multi-workgroup kernels with
  the `ffn_mid` in VRAM (step 7). An **unconditional** fusion through the single
  shader regressed gemma-12B decode to ~4.5 tok/s (the MLP moved off the device);
  the per-op choice keeps the separate pair where it does not cross, so the 12B
  plan is byte-identical and the decode stays ~7.6 tok/s.

### Review items

- ~~**`mlp` is atomic — never cross-device.**~~ **Done.** The fused op is one
  dispatch and never splits: `costOptions(.mlp)` leaves `split_ns` infinite and
  `decideDevice(.mlp)` never returns `.split` (the `ffn_mid` is recomputed per
  row slice, so a split would duplicate the `gate`/`up` work); the three weights
  are one all-or-nothing residency candidate (`addResidentTripleCandidate`).
  Covered by `gpu.zig`'s "fused mlp is atomic" test.
- ~~**The Composer is opportunistic, not generalist.**~~ **Done.** The Composer
  emits the separate pair; the tuner's `fuseMlps` fuses **per op** when
  `mlpFusionPays` (a fused-vs-pair cost comparison) says it pays. Both kernels
  stay; on a discrete GPU a pair is kept unless it would cross devices, so a
  plan with no crossing is the pre-`mlp` one (no regression).
- ~~**Device by ROI.**~~ **Done.** When fused, the atomic MLP is one residency
  candidate whose combined `value` (the 3 weights) ranks it first in the
  packer; the smaller ops fill the residual budget. On a discrete GPU the fused
  triple is packed as one candidate too (both forms run there now).
- ~~**The shader with `m` in VRAM**~~ **Done (step 7).** The `m` is kept in a
  device-local buffer with the `down` as a **second multi-workgroup kernel** on
  the same command buffer (the shared-`m` shader is untouched). The atomic `mlp`
  therefore runs efficiently on a discrete GPU: the ROI device choice is no
  longer UMA-only, and the `mixed` MLP no longer crosses the PCIe.

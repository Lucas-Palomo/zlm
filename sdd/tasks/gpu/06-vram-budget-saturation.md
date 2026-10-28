06 - VRAM Budget Saturation

Status: done

**Closed.** The saturation pays and is the decode lever: the 12B at 8 GB VRAM uses 6408/8128 MB resident with 289/868 ops offloaded -> **8.7 tok/s** (2.4x the CPU-only 3.6); the RAM is free context (49856 tokens at 8.7).

**Evidence (`gpu/11`, measured):** the saturation pays, and it is the performance
lever. The 12B at 8 threads: `--vram 4gb` uses 4031/4032 MB resident with 64/868
decode ops offloaded (**5.8 tok/s**); `--vram 8gb` uses **6408/8128 MB** with
**289/868** ops (**8.7 tok/s** — 2.4x the CPU-only 3.6). The RAM buys **free
context** (49856 tokens at 8.7). The heterogeneous split does not pay (the
memories contend, `split_efficiency` 0.70 — `gpu/11`), so the VRAM budget is
where the decode gain lives.
Commits:

## Goal

Treat the two memory budgets differently:

- **RAM** is an **efficiency** budget: spend what is useful, the most efficient
  way (the current residency window; it need not be full).
- **VRAM** is a **saturation** budget: use the **maximum** of `--vram`,
  proactively **resident** (never on-demand/lazy), the most useful way.

Concretely: `--vram 4gb` must end up using ~4 GB of resident data (weights plus
KV, best-first), and those resident weights must actually be **executed** on the
GPU (a resident weight is transfer-free, so it is offloaded). Today a GPU plan
uses only ~708 MB of a 4 GB budget and offloads nothing.

## Scope

- **In:** `selectResident` saturates `--vram − headroom` (a fill pass after the
  greedy ROI pass); the candidate set covers every offloadable weight (and the
  KV); the offload policy makes a **VRAM-resident weight offloaded by default**
  (the GPU reads it from VRAM), with the measured cost (task 03) gating only
  streamed/non-resident ops.
- **Out:** exceeding VRAM / spillover (`gpu/05`); heterogeneous split
  (`gpu/04`); multi-GPU.

## Design

### VRAM saturation (tuner)

- The ROI knapsack's objective becomes "saturate the VRAM budget": after the
  greedy best-first pass, a **fill pass** packs the largest remaining candidates
  that fit, so the resident set reaches `--vram − headroom` (or all candidates
  are resident). Best-first by value, but the goal is a **full** budget, not
  "only what clearly wins".
- Candidates: every weight whose quant/shape the device supports (the same
  `eligibleResident` gate), across all layers; optionally the KV when it fits.
  **First investigate why the E2B candidate set stops at ~708 MB of a 4 GB
  budget** (only decode `matvec` dispatch tensors are collected today — the
  model's other 2D weights, or the KV, may be missing). Widen the candidate set
  so the budget can actually be reached; report the candidate total vs the
  budget.

### Resident implies offload (tuner + plan)

- A weight that is VRAM-resident is **offloaded**: its dispatch carries
  `device = 1`, because the GPU reads it from VRAM with no host transfer (the
  device-local resident buffers from task 02). The measured offload cost is
  consulted only for **streamed** (non-resident) ops, so the resident set is
  always used.
- This also keeps the GPU busy across the token, which helps the DPM hold the
  memory clock (the interleaved-idle clock drop is why a warm probe overvalued
  and a conservative one undervalued offload).

### Budgets

- RAM: unchanged (the residency window; efficiency, not saturation).
- VRAM: the resident set is filled to the budget; the KV/scratch may join once
  the weights are placed.

## Steps

1. **(done)** tuner `selectResident`: `Candidate` gained a per-token `value`
   tier; the packer orders by `value` then `length` (decode ROI first, then the
   amortized prefill fill) and continues past non-fitting candidates to saturate
   `--vram -| headroom`. `tune` now prints `resident/budget MB`. See the
   investigation and measurements in Comments.
2. **(done)** `decideOffload`/`decideOffloadMatMul` return `true` for a resident
   weight after the quant/shape gate; the measured rate gates only the streamed
   path.
3. **(done)** No plan-format change: `vram_resident` + `Dispatch.device`.
4. **(done)** Measured on the RX 580, ReleaseFast, 4 threads / 4 GB; E2B and 12B
   (Comments).
5. **(done)** `zig build` / `zig build test` green (251/251); CPU `.zplan`
   bit-identical under a fixed `.zat`; GPU `.zplan` deterministic (identical
   hashes across two tunes).

## Definition of Done

- [~] `--vram 4gb` yields a resident set close to the budget (best-first), not
      ~708 MB. **12B: 4001/4032 MB.** E2B is candidate-limited: its device-
      runnable weights total 986 MB (990 MB with the headroom), because the big
      tensors are quants the backend does not run (Q5_K/Q4_K/F32/BF16) and
      `ffn_up`/`ffn_gate` are fused in decode; the set is the whole candidate
      set, not a partial one.
- [x] Every resident weight is offloaded (`device = 1`); the GPU executes it
      from VRAM. (E2B 135/749 decode + 205/869 prefill; 12B 240/868 + 277/1060.)
- [x] The decode is measured (GPU vs CPU) and reported honestly; the CPU plan is
      unchanged; the plan is deterministic.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the RAM-efficiency vs VRAM-saturation split; resident-implies-offload.
- **consulted:** the earlier local prototype's `VramMemoryMap` (ROI packing) — reference.
- **derived:** `gpu/01` (model), `gpu/02` (as-built), `gpu/03` (measured cost +
  device-local residents).

## Comments

- **Why the E2B candidate set stopped at ~708 MB (investigation).** The old
  loop collected only decode `matvec` dispatch `tensor`s with `eligibleResident`.
  Three gaps: (1) the model's big tensors are quants the backend does not run —
  `per_layer_token_embd` Q5_K 1614 MB (reached by `dequant`, not a matvec),
  `token_embd` Q4_K 226 MB, `inp_gate`/`proj` F32 110 MB, `per_layer_model_proj`
  BF16 27 MB; (2) `ffn_up` (and, in decode, `ffn_gate`) are fused into
  `gate_up_fused`, so `ffn_up` is only reachable as a prefill `matmul` tensor —
  the old loop never looked at prefill or at `tensor2`; (3) the KV is never a
  candidate. The device-runnable decode set is 743 MB (242 unique matvec
  tensors, all fitting the 4 GB budget), hence ~708 MB selected. Widening to the
  union of decode (non-fused) `matvec` and prefill `matmul` eligible weights
  raises E2B to 986 MB and 12B to 6.15 GB (the budget binds).
- **Two tiers, not one.** Decode weights save a per-token transfer (value =
  bytes); prefill GEMM weights amortize their transfer over the block (value =
  0). Ranking decode-first stops large prefill-only weights (`ffn_up`) from
  crowding on-policy decode weights out of a saturated budget (12B went from
  36 to 240 decode offloads when the tier was added).
- **Why "fill" and "offload" are one decision.** Filling VRAM with weights the
  CPU will run wastes the VRAM; the budget is only worth saturating if the
  resident set is executed on the GPU. So saturation and offload land together.
- **Fused `gate_up` has no device path.** The decode `gate_up_fused` dispatch
  runs `runGateUp` on the CPU (its weights are not offloaded even when resident);
  a `device` bit there would be ignored, so the tuner leaves it at 0. Its
  `ffn_gate`/`ffn_up` are still residents, executed on the GPU in prefill.
- **KV not placed.** `vram_resident` is tensor-index based and there is no
  device-KV path (still `gpu/02` pending), so the KV cannot join the resident
  set. Left for the KV/scratch task.
- **Environment (confirmed).** The RX 580's memory clock is pinned at its lowest
  DPM level (400 MHz of 400/1000/1750) throughout GPU work; the core clock ramps
  to ~1019–1284 MHz. So a transfer-free resident GEMV is memory-bound at the
  measured ~16 GB/s (E2B) / ~21 GB/s (12B) versus the CPU's ~36–38 GB/s, and the
  GPU is the limiter, not the policy.

### Measured (RX 580 / RADV Polaris, ReleaseFast, `-Dcpu=native`, 4 threads, 4 GB)

Aliases: `gemma4-e2b-4t4r4vr` (GPU) / `gemma4-e2b-4t4r` (CPU);
`g12-gpu` / `g12-cpu`.

| plan | resident | decode offload | prefill offload | decode tok/s | prefill tok/s |
|---|---|---|---|---|---|
| E2B CPU | – | – | – | **17.0–17.5** | ~25.8 (cold) |
| E2B GPU | 986/4032 MB | **135/749** | 205/869 | **14.0–14.4** | ~21.2 (cold) |
| 12B CPU | – | – | – | **3.9–4.0** | – |
| 12B GPU | 4001/4032 MB | **240/868** | 277/1060 | **4.0** | – |

- **E2B: ~15–20 % decode regression** (14.2 vs 17.1 tok/s); prefill also slower.
  The 4 GB RAM budget holds all E2B weights, so the CPU plan is fully RAM-resident
  and the persisted ~350 MB of decode weights run ~2× slower on the 400 MHz-
  memory-clock GPU than on the CPU.
- **12B: a wash** (4.0 vs 4.0 tok/s); the CPU plan is HDD-bound at 4 GB RAM, so
  moving ~4 GB resident to VRAM and executing it offsets the slower device.
- **Verdict.** Saturating VRAM + resident-implies-offload does **not** help on
  this host: it is neutral on 12B and a regression on E2B. The blocker is the
  device (VRAM clock stuck at 400 MHz), not the policy; on a GPU whose memory
  clock boosts, the transfer-free resident GEMV should win. The plan, offload
  and saturation mechanics are correct and CPU-identical.
- **Spillover is `gpu/05`.** This task saturates within `--vram`; exceeding it is
  the separate spillover task.

09 - Deterministic Tuning

Status: done
Commits:

## Goal

One tuning pipeline whose **spine is an analytical cost model**: the hardware is
*profiled* (capabilities), its coefficients are optionally *calibrated* (native,
robust), candidates are *enumerated* and ranked by `cost`, an active **Composer**
composes them, and an **Evaluator** loops (bounded) to the best plan. Timing is
used **only to calibrate the model's coefficients** — never to pick an algorithm
per candidate (that is what drifts and throws good algorithms away). Phase 1
targets CPU; GPU tiers are modeled and pruned.

```text
Profile (caps) ─▶ Calibrate (native) ─▶ Cost model ─▶ Residency ─▶
                  Selector ─▶ Composer ─▶ Evaluator (loop) ─▶ .zplan
```

## Scope

- **In:**
  - **Profile** — detailed, deterministic hardware introspection (ISA, cores/SMT,
    caches, memory channels, NUMA, cgroup, tiers).
  - **Calibrate** — native, cached: robust relative measurement that refines the
    **coefficients** (`bw` per tier, `par_eff`/knee, `dispatch_latency`, startup),
    persisted as a **device-keyed `.zat`** in `.zlm/runtimes/`; measured and
    saved on the first tune, reused afterwards; `--force` re-measures it.
  - **Cost model** (`cost`, plan-global) — roofline over tiers
    (`disk < ram < cache`, `vram`), `bytes`/`flops`, `par_eff`, per-op overhead,
    `model_bytes`, with the **RAM** and **thread** constraints.
  - **Residency-first** — build the `ResidencyGraph` *structure* (ranges,
    liveness, working set vs cache) **before** the cost decision, so caching
    informs `cost`.
  - **Selector / Composer / Evaluator** — enumerate ranked candidates; compose
    (`single | fused | mixed | seq`); bounded deterministic loop.
  - Retire the timing decisions (`bench.zig` per-candidate pick → calibration).
- **Out:** GPU backend/materialization (tier modeled, GPU-only candidates pruned);
  per-candidate timing gates; a decision cache (the `.zplan` already carries the
  steps).

## Design

### Profile — detailed hardware introspection (deterministic)
Read everything stable: **CPU/ISA** (`builtin.cpu` + CPUID: SIMD width, AVX-512
subfeatures, VNNI/BF16/AMX, freq, hybrid P/E); **topology/NUMA**
(`/sys/devices/system/{cpu,node}`); **cache** sizes/associativity/line per level;
**memory** channels/speed/type via SMBIOS/DMI -> theoretical peak bandwidth; and
the budget/quota (cgroup, allowed CPUs, `--ram`/`--vram`). No timing here.

### Calibrate — coefficients, not decisions (native)
Measure **once per device**, robustly, to refine the model's coefficients:
- **A/B interleaved** (alternate order) cancels slow drift; **min-of-N** is the
  least-contaminated estimator; **pinned** threads avoid migration.
- Targets: `bw` per tier (DRAM, and cache where useful), `par_eff`/knee (compute
  vs memory-bound), `dispatch_latency`, startup/`disk`, and the
  `madvise(WILLNEED)` cost model (`base + per_page*pages`, capped) that lets the
  prefetch-worker decision scale with the model size.
- **Persisted** in a device-keyed `.zat` under `.zlm/runtimes/`
  (`caps + coefficients`; new magic/version). Reused across images and configs;
  `--force` re-measures; absent -> measure and write. Always on: the plan is a
  pure function of the declaration + the calibrated caps.
- Calibration shifts the *model*, never flips an algorithm by a raw time gate.

### Cost model
```text
tiers:  disk << ram << L3 << L2 << L1        (+ vram; pcie for GPU moves)
per op/group:
  cost = max( bytes_effective(tier, cache_fit)/bw(tier),
              flops / (peak_flops(hw) * par_eff(op, parts)) )
       + k_ops * dispatch_latency
bytes_effective = weight_bytes + activation_bytes, adjusted by cache fit
                  (working set vs L1/L2/L3) so variants that move the same
                  *logical* bytes are still differentiated
flops  = 2*n*m (matvec) | O(n^2) naive vs O(n) flash (attention) | ...
objective (Phase 1) = w_ttft * cost_ttft + w_gen * cost_generation,  w_gen > w_ttft
score = sum_ops cost(op | tier) + streamed_live_bytes/bw_disk + startup/bw_disk
subject to  resident_bytes <= ram (<= vram)
        and parts(d) + residency_workers <= threads    # never steal compute
```
- **`model_bytes`** decides the regime (fits -> RAM/cache; else the streamed term
  dominates and fusion/residency matter).
- **Cache-aware `bytes_effective`** is what differentiates candidates that move
  the same logical bytes (tiling/unroll) — the crux; without it the Selector is
  blind.
- **CPU vs GPU** differ by `(bw, peak_flops)`; GPU adds transfer `bytes` (PCIe)
  unless resident in VRAM -> else pruned.
- **Residency prefetch worker**: `cost.decidePrefetch` compares the per-token
  `madvise(WILLNEED)` work (from the calibrated `madvise` cost and the
  non-resident ranges) with one thread's compute share. The result
  (`residency_workers` 0|1) is recorded in the plan and `parts` is chosen with
  `threads - residency_workers`; the runtime only applies it.

### Residency-first
Build the `ResidencyGraph` **structure** up front — the weight ranges, their
`first_use`/`last_use`, and the working-set-vs-cache classification — so `cost`
can account for what is cache-resident vs streamed. The *admission* (which ranges
to pin) is part of the cost optimization (knapsack under `ram`), so it lives in
the Evaluator, not before.

### Selector -> Composer -> Evaluator
```text
candidates = Selector: per op, the finite applicable AtomicOps/recipes, ranked by cost
composed   = Composer: recipes + form (single|fused|mixed|seq), compatible adjacents
loop (<= MAX_PASSES, deterministic):
    score = Evaluator(candidates/composition, hw, residency)
    if no neighbor strictly improves: break        # no-gain => keep current
    else re-compose (swap candidate / fuse / unfuse)
winner -> .zplan (Dispatch[]: atomic, params, parts, residency)
```

### Prerequisite — kernel expansion (candidate catalog as data)
The Selector/Composer need something to enumerate. Today only `matvec` has
interchangeable variants (`MatVecPlan`); recipes exist in code
(`fusedGateUpRows`, `qk_norm_rope_heads`) but are **not declared**. A prerequisite
(kernels task) must extend `api.Kernel` with cost metadata (`bytes_effective`
model, `flops` model, `applicable(type,shape)`, `tier`, `complexity-class`) and
declare **recipes** as data. Phase 1 can start with `matvec` only.

### Artifacts
- **`.zplan`** — the plan; already carries every step the runtime executes.
- **`.zat`** — the **device** calibration profile (`caps + coefficients`),
  `.zlm/runtimes/<device_digest>.zat`; no decisions, no per-candidate timings.
  Supersedes the old `<image>_<fp>.zat` (VERSION bump; old files ignored).

## Flow / Arch

```text
 ┌──────────┐   ┌───────────────┐   ┌───────────┐   ┌───────────┐
 │ PROFILE  │──▶│ CALIBRATE     │──▶│ COST      │──▶│ RESIDENCY │
 │ caps     │   │ (native,.zat) │   │ model     │   │ structure │
 └──────────┘   └───────────────┘   └─────┬─────┘   └─────┬─────┘
                                          ▼               ▼
                              ┌───────────────────────────────────┐
                              │ SELECTOR → COMPOSER → EVALUATOR    │
                              │ (bounded loop; min score s.t. RAM, │
                              │  parts+residency<=threads)         │
                              └────────────────┬──────────────────┘
                                               ▼
                                   .zplan (steps)   ; .zat = device profile
```

## Arch Provenance

- **own:** the unified pipeline (cost spine + native coefficient calibration),
  plan-global cost model, residency-first.
- **spec:** —.
- **consulted:** the earlier local prototype's `src/autotune/*` (rounds/gates): kept the
  robust *relative* measurement idea for calibration, **rejected** as a
  per-candidate decision gate.
- **derived:** `docs/kernel-architecture.md` (Selector/Composer/Scheduler),
  `runtime/05` (thread budget), `runtime/06` (residency mechanics), `runtime/07`
  (Profile/plan split).

## Steps

1. **Prereq (kernels):** cost metadata on `api.Kernel` + a declared recipe
   catalog; at least `matvec`. (May be its own `kernels/` task.)
2. `caps` + `calibrate` (new): detailed introspection + native, robust
   coefficient measurement; device-keyed `.zat` (caps+coefficients, new
   magic/version) under `.zlm/runtimes/`; `--force` re-measures.
3. `cost` (new): tiers, cache-aware `bytes_effective`, `flops`,
   `par_eff(op,parts)`, overhead; `model_bytes`; the TTFT/generation objective;
   the `resident<=ram` and `parts+residency<=threads` constraints.
4. Residency-first: build the `ResidencyGraph` structure before `cost`.
5. Selector/Composer/Evaluator: enumerate+rank; compose; bounded loop
   (`MAX_PASSES`) minimizing the score.
6. Wire `tune`: Profile -> Calibrate -> Cost -> Residency -> Select/Compose/
   Evaluate -> `.zplan`; retire the per-candidate timing picks.
7. Tests: cache-aware ordering (same-bytes variants differentiated); calibration
   is robust/IDEMPOTENT-ish; loop terminates; GPU pruned without backend;
   device `.zat` round-trip.
8. `zig build` / `zig build test` green; e2e: E2B tune (default and `--measure`),
   coherent, bounded RSS, reproducible run-to-run.

## Definition of Done

- [x] Profile is capability-only; `Calibrate` refines coefficients (native,
      device-keyed `.zat`), never picking algorithms by timing.
- [x] `cost` is cache-aware (`bytes_effective`), plan-global, with the RAM and
      thread constraints and a TTFT+generation objective (generation weighted
      more).
- [x] Residency structure is built before the cost decision.
- [~] Selector/Composer/Evaluator — `Selector` (ranked candidates) and `Composer`
      (`qk_norm_rope`, `gate_up_fused` via multi-binding) done; the "loop" is a
      deterministic **greedy collapse** (no iterative hill-climb) — see
      `open-items`.
- [x] GPU-only candidates are modeled and pruned without a backend.
- [x] `.zplan` alone carries the steps; no decision cache.
- [x] `zig build` / `zig build test` green; E2B coherent and bounded.

**As-built (stages 1-3 + calibration).**
- `caps.zig` (deterministic introspection: ISA lanes/features, topology, caches,
  frequency, cgroup, NUMA) and `cost.zig` (tiers, cache-aware `bytesEffective`,
  `parEff`, `opCost`, `planScore` TTFT/gen 0.3/0.7).
- `tuner.zig`: `selectMatvec` — variant by a documented ISA×shape rule, `parts`
  = argmin `opCost` (≤ threads). Wired into `tune` (replacing the timing bench).
- `calibrate.zig`: min-of-N pinned measurement of DRAM/disk bandwidth +
  dispatch latency; `Profile` (caps+coeffs) as `ZAT2` at
  `.zlm/runtimes/<device_digest>.zat`; `tune` load/reuse/measure/`--force`
  (always calibrated); the old per-image timing `.zat` retired.
- Verified: deterministic plan (identical sha256 across tunes), coherent E2B run
  (~425 MB), 119/119 tests.

**As-built (Composer + budget).**
- Composer (`tuner/composer.zig`): `qk_norm_rope` (`Instr.fuse_rope`) and
  `gate_up_fused` (multi-binding: `Dispatch.tensor2`, `Instr.role2`,
  `Op.gate_up`, `.zplan` VERSION 7→8); deterministic greedy collapse. E2B
  869 → 749 dispatches.
- Budget via `cost` (iii): `cost.admissionDensity` + `cost.lookahead` replace
  `admitByDensity`/`deriveLookahead`; `reserve`/window sourced from `caps`;
  removed the ad-hoc constants/`bench` from the decision path.
- Verified: deterministic plan, coherent E2B run, 129/129 tests.

## Comments

- Phase 1 = CPU; GPU tier modeled/pruned. Follow-ups: GPU backend, real
  measurement-of-the-winner validation, kernel catalog breadth.
- The unified positioning: **measure to calibrate the model, decide by the
  model** — O(1) per device, robust to drift, reproducible.
- **Composer status:** the applicable gemma4 fusions are implemented
  (`qk_norm_rope`, `gate_up_fused`) via multi-binding (`Dispatch.tensor2`,
  `Instr.role2`, `Op.gate_up`: 869 -> 749 dispatches). The remaining recipes
  (`norm_proj`, `add_norm`, `rope_write_kv`, `matvec_bias`) need a fused `qkv`, a
  two-output kernel, or are unused — see `sdd/open-items.md`.
- **Remaining (iii):** replace the last 06/07 budget heuristics
  (`admitByDensity`, `deriveLookahead`, `reserveBytes`) with the cost model.
- `dram_bw` calibration reads low (~9.7 GB/s) — the stream-sum microbench is
  compute-bound; refine with independent accumulators / wider loads / multiple
  threads (a caliberation-quality follow-up).
- **`--no-calibrate` removed (always calibrate).** The flag produced plans whose
  fallback per-thread bandwidth equalled the aggregate (flat `bwAtOp`), so
  `selectParts` stalled at `parts≈3` even at 8 threads — decode ran on ~3 threads
  (~4.3 vs ~14 tok/s calibrated). Since the plan's decisions depend on the
  calibrated caps, calibration is no longer optional; the sweep's failure
  fallback is now `aggregate / cores` so it still scales to the core count.

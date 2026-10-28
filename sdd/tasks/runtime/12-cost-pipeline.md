12 - Cost Pipeline

Status: done
Commits: e9640cd 5f1a3fd

## Goal

**Complete refactor of the tuning cost model.** `cost.zig` today is a single
formula (`max(bytes/bw, flops/(peak*parEff)) + dispatch`) that mixes three
different questions, and the tuner is a **single linear pass** — no candidate
evaluation. Replace it with three stages, each measuring a different kind of
complexity, feeding a bounded **Evaluator**:

1. **AST static cost** — the logical work per op `{ bytes, flops, complexity,
   deps }` (hardware-independent).
2. **Roofline** — the machine ceiling `max(bytes/bw, flops/peak)`; also decides
   the op's **regime** (memory- vs compute-bound).
3. **Measured scaling** — the empirical `bw(n)`/scaling curve per op/variant
   (memory-bound), with Amdahl/Gustafson as the compute-bound reading of it.

The three stages are the *cost function*; the Evaluator is the *search*. This
supersedes `cost.zig` entirely, plus `runtime/09`'s `parEff`, `runtime/11`'s
scaling and `kernels/07`'s variant cost.

## Scope

- **In:**
  - `staticCost(op, shape, variant, parts) -> LogicalCost` as a **shared function**
    used twice: **1a** over the declaration (the plan-compile, hardware-independent,
    inside `tune`) and **1b** per candidate (variant + parts).
  - The **Roofline** step (`ideal_ns`) and the **Amdahl/Gustafson** step
    (`effective_ns(parts)`) as distinct functions.
  - A bounded, deterministic **Evaluator** over a small candidate set
    (`variant × parts × prefill policy × window/recipes`), scored by `planScore`,
    `argmin`.
  - A per-variant efficiency / thread-scaling input calibrated once per device
    (the `.zat`), so stages 2-3 are accurate.
  - A **decision layer** of **decision stumps** with thresholds fit by
    **Threshold Search** over the calibration data (in `Measure`), for the
    boundaries where the analytical cost is flat (variant, knee, pin/stream).
  - Four auxiliary algorithms (★): **Critical Path** (Step 1), **I/O Complexity**
    (Step 2), **Space-filling design** (Measure), **Little's Law** (Scheduler).
- **Out:** the runtime (unchanged, decides nothing); GPU; changing the plan format
  (the plan still stores decisions, not costs); the GEMM/kernel implementations
  themselves.

## Design

### The three stages
```text
staticCost(op, shape, variant, parts) -> LogicalCost { bytes, flops, complexity, deps }
  Step 2  roofline(LogicalCost, caps)             -> ideal_ns + regime
  Step 3  scaling(LogicalCost, parts, regime, curve) -> effective_ns   # measured curve
```
- **Step 1a** runs once, over the declaration: the logical cost per op. It is
  hardware-independent, so it may be cached. It belongs to the **plan-compile**
  inside `tune` (it needs the declaration + the image; it must not leak the
  declaration into `image/`).
- **Step 1b** re-runs the same function per candidate, because the cost depends on
  **how the selected algorithm is used** (variant + `parts`). This is what the
  Evaluator scores.
- **Step 2** is the theoretical ceiling (bandwidth vs flops), variant-aware
  (efficiency), calibrated. It also labels the op **memory- or compute-bound**.
- **Step 3 is the measured scaling model**, not a closed-form law. The model is
  the **empirical curve** (knee/saturation) per op/variant:
  - **memory-bound** (the LLM hot path): `bw(n) = min(ceiling, per_thread * n)`
    -> `effective_ns = bytes / bw(n)`;
  - **compute-bound**: Amdahl/Gustafson is the *reading* of the same curve
    (saturating vs scaling) — the shape, with the curve's coefficients.

  Amdahl/Gustafson are analytical approximations; the measured curve is the real
  model (it captures bandwidth saturation, which the laws do not). The earlier local prototype's
  `autotune` (knee/saturation) is exactly this model.

### Evaluator (bounded search)
```text
candidates = { variant } × { parts } × { prefill policy } × { window/recipes }   (small)
for c in candidates:
    for op: lc = staticCost(op, shape, c.variant, c.parts)   # Step 1b
            t  = scaling(roofline(lc, caps), c.parts, phase, caps)  # Steps 2-3
    score(c) = planScore(t, c, caps)                          # weighted TTFT+gen
plan = argmin score
```
Deterministic (no timing gate); the search space is bounded (a few dimensions,
greedy local refinement per op).

### Decision layer (stumps + threshold search)
Where the analytical cost is **flat** (variant-invariant) or a measured threshold
is more reliable than a formula, a small set of **decision stumps** (one feature,
one threshold) makes the discrete boundary:
- **variant** (`kernels/07`): `variant = stump(K, N, lanes; thresholds)` — the
  analytical cost cannot rank variants, so a measured stump does;
- **knee / saturation** (Step 3): the thread count where the curve flattens — a
  stump on the scaling axis;
- **pin / stream** (Scheduler): the residency boundary.

The thresholds are **fit by Threshold Search** over the calibration data (in
`Measure`) and cached in the `.zat`. Deterministic: the thresholds are calibrated
once, then only applied. Keep the stump set **small and auditable** (no opaque
ML); the analytical model still owns the continuous/quantitative decisions.

### The four auxiliary algorithms (★)
Each removes a *guess* in the pipeline; all are analytical, deterministic,
auditable:
- **Critical Path** (Step 1) — the longest path in the `deps` DAG is the serial
  lower bound `s`; it feeds Amdahl (Step 3) instead of a guessed serial fraction.
- **I/O Complexity** (Step 2) — the minimum bytes moved between cache levels
  (Aggarwal-Vitter / Hong-Kung) is the **floor** of the cache-aware roofline,
  replacing the hand-picked `l1/l2/l3` multipliers.
- **Space-filling design** (Measure) — a Latin-Hypercube / D-optimal design picks
  **which** calibration points to measure (deterministic), reducing measurements.
- **Little's Law** (Scheduler) — `in_flight = bw × latency` formalizes the
  prefetch lookahead (bytes in flight to hide the disk latency).

### Flow
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPILE  raw model (GGUF) ──▶ image (ZTEF) + manifest   (no cost analysis)   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TUNE  (per machine)                                                          │
│   declaration (Block/Instr[]) + image                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ① AST · Step 1a   staticCost(declaration) → LogicalCost[]  (HW-indep.)     │
│      ★ Critical Path (longest path in the `deps` DAG) → serial fraction s    │
│        │                                                                     │
│   MEASURE   caps + coeffs + variant-eff + scaling curve                      │
│        │   ★ Space-filling design (LHS / D-optimal) → WHICH points to measure │
│        │   └─▶ ④ THRESHOLD SEARCH → thresholds (knee/sat, variant, pin/stream)│
│        │              └─▶ .zat  (coeffs + curve + thresholds)                │
│        │                                                                     │
│   ┌────┴───────────────────────┐   ┌────────────────────────────┐            │
│   ▼                            │   ▼                            │            │
│  ANALYTICAL (cost)             │  EMPIRICAL (decision)          │            │
│   ① AST (1b) → LogicalCost     │   ⑤ DECISION STUMPS            │            │
│   ② ROOFLINE → ideal + regime  │     variant = stump(K, N,      │            │
│     ★ I/O Complexity (AV/HK) → │              lanes; thresh.)   │            │
│        per-cache-level floor   │     (thresholds from ④)        │            │
│   ③ SCALING  → effective(parts)│                                │            │
│     memory: bw(n)=min(cap,pt·n)│                                │            │
│     compute: AMDAHL (uses ★ s) │                                │            │
│   └───────────┬────────────────┘   └────────────┬───────────────┘            │
│               └─────────────┬───────────────────┘                            │
│                             ▼                                                │
│   EVALUATOR (bounded, deterministic)                                         │
│     candidates = { parts } × { prefill } × { window/recipes }                │
│     score(c) = planScore(analytical(c), stumps(c));  plan = argmin           │
│                             │                                                │
│                             ▼                                                │
│   Scheduler: order / residency                                               │
│     ★ Little's Law  in_flight = bw × latency → prefetch lookahead → .zplan   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUN  executes the plan, decides nothing                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

| # | Component | Where | Role |
|---|---|---|---|
| ① | **AST** | tune · plan-compile (1a) + per candidate (1b) | logical work: bytes, flops, complexity, deps |
| ★ | **Critical Path** | tune · Step 1 (AST) | serial fraction `s` from the `deps` DAG (feeds Amdahl) |
| ② | **Roofline** | tune · analytical | machine ceiling + regime label |
| ★ | **I/O Complexity** | tune · Step 2 (Roofline) | per-cache-level byte floor (Aggarwal-Vitter / Hong-Kung) |
| ③ | **Scaling / Amdahl** | tune · analytical | effective time with `parts` (curve; Amdahl = compute-bound, using `s`) |
| ★ | **Space-filling design** | tune · Measure | which calibration points to measure (LHS / D-optimal) |
| ④ | **Threshold Search** | tune · Measure | fits the stump thresholds from measured data |
| ⑤ | **Decision Stumps** | tune · empirical | applies thresholds -> discrete boundaries (variant, knee, pin/stream) |
| ★ | **Little's Law** | tune · Scheduler (residency) | prefetch depth: `in_flight = bw × latency` |

### Where costs are stored
- Step 1a's logical table is **transient** in `tune` (hardware-independent; may be
  cached in the future).
- The `.zplan` stores **decisions** (variant / `parts` / residency / prefill
  policy), not costs. The runtime never sees a cost.

## Steps

1. Define `LogicalCost` + `staticCost(op, shape, variant, parts)`; move the
   per-op bytes/flops/complexity metadata (`kernels/07`) behind it. [TODO]
2. **Step 1a**: `analyze(declaration, dispatch) -> []LogicalCost` in the
   plan-compile (hardware-independent); keep it in `tune`. [TODO]
   - ★ **Critical Path**: longest path in the `deps` DAG -> serial fraction `s`. [TODO]
3. **Step 2**: `roofline(LogicalCost, caps) -> ideal_ns`, variant-aware. [TODO]
   - ★ **I/O Complexity**: per-cache-level byte floor (Aggarwal-Vitter /
     Hong-Kung) as the cache-aware roofline floor. [TODO]
4. **Step 3**: `scaling(LogicalCost, parts, regime, curve) -> effective_ns` — the
   **measured curve** (`bw(n) = min(ceiling, per_thread*n)` for memory-bound);
   Amdahl (compute-bound) uses `s` from the Critical Path. [TODO]
5. **Measure**: calibrate the per-variant efficiency **and the thread-scaling
   curve** (effective_bw at 1..N threads -> knee/saturation) into the `.zat`. [TODO]
   - ★ **Space-filling design** (LHS / D-optimal): pick the calibration points. [TODO]
6. **Threshold Search + stumps**: fit the thresholds (knee/saturation, variant
   boundaries, pin/stream) from the calibration data; expose a small, auditable
   stump set. [TODO]
7. **Evaluator**: enumerate the bounded candidate set, score with `planScore`
   (analytical + stumps), pick `argmin`; keep it deterministic. [TODO]
8. **Scheduler**: order / residency; ★ **Little's Law** (`in_flight = bw × latency`)
   sets the prefetch lookahead. [PARTIAL] `cost.lookahead` sets the node
   `trigger_step`, and `cost.decidePrefetch` compares the calibrated
   `madvise(WILLNEED)` work (`calibrate.measureMadviseCost`) against one thread's
   compute share, recording `Config.residency_workers` (plan v10); `parts` is
   sized to `threads - residency_workers`. The full order/residency search (and
   the stump/threshold layer) remains TODO.
9. Replace the old single-pass `opCost`/`parEff` with the pipeline; keep the plan
   deterministic and E2B coherent. [TODO]
10. Tests: the stages are monotone/ordered; Critical Path / I/O Complexity are
    correct on known shapes; Threshold Search fits known thresholds; the Evaluator
    picks by score and is deterministic; equivalence vs the old where unchanged. [TODO]
11. `zig build` / `zig build test` green; re-benchmark vs the earlier local prototype. [TODO]

**As-built (Stage A — faithful refactor).** `cost.zig` now exposes the three
stages: `LogicalCost` (Step 1) -> `roofline()` -> `Roofline { bytes_ns, flops_ns,
dispatch_ns, regime }` (Step 2) -> `scaling()` (Step 3). `opCost` is a thin
composition of the two, so the plan is **byte-identical** (sha `9592b13…`) and
158/158 tests pass. Next: the measured scaling curve, the Evaluator, the stumps
and the ★ algorithms.

**As-built (measured scaling curve — Step 3).** `calibrate.measure` now measures
DRAM bandwidth **multi-threaded**: the per-thread figure (~11 GB/s) and the
aggregate ceiling (~36 GB/s @8T), stored in the `ZAT2` v3 `.zat`. `cost.scaling`
prices the memory term with `bwAt(n) = min(aggregate, per_thread * n)`, so
`selectParts` reaches the real ceiling. E2B decode **4.4 -> 9.7 tok/s** (2.2x),
near the earlier local prototype's ~12; plan deterministic; 161/161 tests. Remaining: the Evaluator, the
stumps (+threshold search), the ★ algorithms.

## Definition of Done

- [x] `staticCost`/`LogicalCost` is a shared function (Step 1) used by the
      Selector and the plan-compile.
- [x] Roofline (Step 2) and the **measured scaling** (Step 3) are distinct;
      Step 3 uses the calibrated curve (`bwAt(n) = min(ceiling, per_thread*n)`).
- [ ] (deferred → `open-items.md`) A **decision layer** of stumps resolves the
      variant/knee/pin boundaries from **Threshold Search**-fitted thresholds.
- [ ] (deferred → `open-items.md`) The Evaluator enumerates candidates and picks
      `argmin planScore`, deterministically.
- [ ] (deferred → `open-items.md`) The ★ algorithms: **Critical Path**, **I/O
      Complexity**, **Space-filling design**; **Little's Law** is partial
      (`cost.lookahead` + `cost.decidePrefetch`).
- [x] The plan stores decisions only; the runtime still decides nothing.
- [x] `zig build` / `zig build test` green; coherent; deterministic plan.

## Arch Provenance

- **own:** the three-stage split, the measured-scaling model, the decision-stump
  layer (threshold search over calibration data), and the bounded deterministic
  Evaluator.
- **spec:** Roofline (Williams et al.); Cache-Aware Roofline; Amdahl's law;
  Gustafson's law (the compute-bound reading of the curve); decision stumps
  (one-level tree); **Critical Path** (longest path in a DAG); **I/O Complexity**
  (Aggarwal-Vitter / Hong-Kung); **Little's Law**; **space-filling design** (LHS).
- **consulted:** the earlier local prototype's `autotune` (per-op scaling / knee / saturation) — reference
  for the measured scaling curve and the threshold search, not the timing gate.
- **derived:** `runtime/09` (deterministic tuning), `runtime/11` (performance),
  `kernels/06`/`07` (generic Selector, variant cost), `plan/` (compile the block).

## Comments

- Absorbs the intended-but-unbuilt `runtime/09` **Evaluator** (its DoD marked the
  loop `[~]`), and supersedes the single `opCost`/`parEff`.
- Depends on accurate calibration: `runtime/11` (aggregate bandwidth, dispatch
  latency) and `kernels/07` (variant efficiency) are prerequisites — a search over
  a wrong cost picks garbage.
- Step 1a stays in `tune` (the plan-compile) to respect the `image/` boundary.
- **Built:** the three stages (Stage A) and the **measured scaling curve**
  (Step 3). **Deferred to `open-items.md`:** the Evaluator (bounded search), the
  decision stumps + threshold search, and the four ★ algorithms — see the
  "Cost pipeline" entries there. `runtime/12` is paused after the measured curve
  (which already delivered a 2.2x decode win); we will return to it.
- **Evaluated and discarded** (they do not fit): **ILP/CP solvers** for the
  Evaluator (the candidate space is small and the tune runs once per device;
  a solver is an unnecessary dependency) and the **Polyhedral model** (the AST is
  an op sequence, not affine loop nests; zlm deliberately rejects IR/loop-nest
  lowering — "Composition is recipes, not IR"). zlm's weights are **dense**, so
  sparse-indexing rationales do not apply.

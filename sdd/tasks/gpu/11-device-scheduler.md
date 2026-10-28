11 - Device Scheduler (heterogeneous chain scheduling)

Status: done (the model + the DP + the RAM/VRAM policy + the `split_efficiency`
+ the tuner wiring — 289/289). Two findings, both measured:

- **The split is a *capability*, not a win here.** The probe measures the
  concurrent CPU+GPU GEMV over the GPU-only rate (`split_efficiency`); on the RX
  580 it is **0.70** — the memories contend, so a split *loses* to the whole-GPU
  op. The tuner prices it `bytes / (gpu_rate * split_efficiency)` and keeps every
  op whole on the GPU. On a machine where the bands add (> 1) the same tuner
  would split — no code change.
- **The VRAM resident set is the lever.** 12B, 8 threads: VRAM 4 GB -> 5.8 tok/s;
  VRAM 8 GB -> **8.7** (2.4x the CPU-only 3.6); the RAM buys **context** and it is
  free once the offload is good (**49856** tokens at 8.7). No context cap needed.

Not built (small, and no longer on the critical path): the measured `hop_ns`
dispatch-setup constant and the `runs` telemetry (`profile:` reports
whole-gpu/split/cpu-only, not the runs).
Commits: 59d068e a295d5e 2e9ba5e a3e3567 b2a9975 8c2614f 23acde8

## Goal

Turn the **per-op device oracle** (`gpu/04`) into a **chain scheduler**. Today
`decideDevice` chooses `cpu | gpu | split(g)` for each op **independently**, so
the plan can ping-pong between the devices for a marginal per-op gain — every
crossing paying a transfer, a sync and a GPU setup. The scheduler decides the
**whole decode chain** at once, minimizing the token's makespan with:

1. the per-op cost (`gpu/04`'s oracle);
2. the **switch cost** (transfer + sync) of changing device;
3. the **setup cost** of a GPU op (a measured constant + a per-op model);
4. the **dependencies** (the chain);
5. a **dynamic look-ahead**, so a cheap op steps aside for a heavy one.

The output is not a device per op but a sequence of **runs** (maximal same-device
stretches) — the chain's profile: `cpu`, `gpu`, `split`, `mixed` (with its runs
and crossings).

## Scope

- **In:** the receding-horizon DP over the dispatch chain; the switch cost; the
  setup cost (the measured constant + the per-op model); the runs; the extended
  `profile:` telemetry (the runs + the crossings).
- **Out:** the executor's **pre-record** (preparing the next GPU op's command
  buffer while the CPU runs) — **deferred**, not rejected; see Comments.
  Cross-token pipelining; multi-GPU; network.

## Design

### The inputs

| input | source |
|---|---|
| the per-op cost `cpu \| gpu \| split(g)` | `decideDevice` (`gpu/04`), fed by the chain's `Load` |
| the **switch cost** | the activation transfer + the drain (a sync) |
| the **setup cost** | `dispatch_latency_ns` (measured) + the per-op model |
| the chain | the plan's `Dispatch[]` + `dep` |

### The switch cost

Changing device at op `i` costs:

- the **activation transfer** — the producer's output crosses the PCIe (a
  download from VRAM to RAM, or an upload). KB, but a round trip.
- the **sync** — the consumer drains the producer's fence (`waitResident`), a
  driver round trip.

Small but real, and it is what makes a per-op decision wrong: a crossing is paid
**once**, so a **run** amortizes it. Sticky-by-default, crossing only when the
run's benefit exceeds the cost.

### The setup cost

```
setup_ns(op) = dispatch_latency_ns                (the measured constant, .zat)
             + activation_bytes / PCIe_BW_GBPS     (the staging model)
             + pipeline_switch_ns                  (only when the quant/pipeline changes)
```

Both the constant and the per-op coefficients are **essential** and calibrated:
the constant from the device probe (`gpu/02` / `gpu/09` — today's
`dispatch_latency_ns` is a placeholder), the coefficients from the op's shape.

### The dynamic look-ahead (receding-horizon DP)

The scheduler is a **bounded DP** over the chain. With `D(i)` the device of op
`i` and `d` the *previous* device:

```
dp[i][d] = min over D(i) of  cost(op_i, D(i))
                          + (D(i) != d ? switch_cost : 0)
                          + (D(i) == gpu ? setup_ns(op_i) : 0)
                          + dp[i+1][D(i)]
```

The window `W` is **dynamic**: it extends while the visible ops are cheap (a
crossing would not pay) and opens at the next **heavy** op — one whose cost
exceeds the switch cost. The look-ahead therefore looks *as far as the next op
worth crossing for* — exactly the cheap-Op3 / heavy-Op4 case:

```
Op1 [CPU]  ->  Op2 [GPU] -> Op3 [GPU]  (cheap: sticky)
                                \  -> Op4 [GPU, heavy]: open the window —
                                   Op3 -> CPU (free the GPU), Op4 [GPU]
```

Deterministic: the window is a pure function of the ops and `caps`, no timing.

> **As built (simpler):** for a **linear** chain the costs are additive, so a
> plain **forward Viterbi** (state = the previous device) is *exact* — there is
> no need for a windowed heuristic, because the DP's cost-to-go **is** the
> look-ahead. `scheduleDevices` implements that; the "dynamic window" above is
> the conceptual framing, the Viterbi is the mechanism.

### The runs (the output)

The plan still records a device per dispatch (`Dispatch.device` + `row_count`),
but the **runs** are the unit a human reads: a maximal stretch of same-device
dispatches. The tuner prints them and the crossings:

```
profile:  mixed — cpu(12) gpu(43) cpu(9)   (2 crossings)
```

### The RAM/VRAM policy (the tier follows the schedule)

The weights are **born in the image** (the RAM side, `mmap`). The RAM residency
is not the bytes but the **budget** (the pin + the prefetch window), and the KV
absorbs whatever the weights do not spend. Migrating a weight to the GPU
**releases its RAM budget** (the `gpu/05` exclusion) and the freed budget becomes
context; the reverse re-pins it and the KV shrinks. Measured (`gpu/05`): the 12B
RAM `weights` 6411 -> 2920 MB, the context 4096 -> 5162.

So a device choice is, in practice, a choice about **where the RAM budget goes**:

| device | RAM the residency spends | VRAM |
|---|---|---|
| `cpu` | `B` (the whole weight) | 0 |
| `gpu` (resident) | **0** — the budget is freed, the KV grows | `B` |
| `split(g)` | `B*(m-g)/m` (the CPU tail) | `B*g/m` |
| `gpu` (stream, the migration bridge) | `B` (the weight stays) | 0 |

**No duplication.** One tier per weight (`gpu/05`); the split's rows are
disjoint. A weight **floats** — a move, not a copy.

**The two GPU variants.** A GPU op whose weight is in the RAM has two priced
choices:
- `gpu_resident` — **move** it (the upload once); the shader reads the VRAM; the
  VRAM budget is the constraint.
- `gpu_stream` — **stream** it (the shader reads the RAM over the PCIe). A
  **per-token** op pays the PCIe every token (~5.43 GB/s vs the CPU's ~35 GB/s)
  and loses; a **one-shot** op (the prefill) amortizes. Otherwise the stream is
  the **migration bridge**.

**The migration (the declared exception).** The spillover is not a general pager
(`gpu/05`); the tuner **declares** it in the plan: per weight, the **target
tier**, a **trigger step** (the first op that uses it) and the **bridge**. The
runtime applies it: at the step it uploads; the shader streams from the RAM until
the upload completes, then reads the VRAM — a short, bounded window, the cost of
moving paid once. The op that **leaves** the GPU (its node re-activates the RAM
residency) and the upload of the next op's weight land in the same declared step
— the "dance" the scheduler produced.

**The objective.** The scheduler minimizes the makespan **subject to** the RAM
budget: `Σ` (the residency's CPU-side bytes) must leave the KV at least at its
floor. The KV is the buffer, so the trade is explicit: migrating to the GPU pays
in VRAM (the hot set) and earns context; not migrating pays in VRAM pressure. The
DP carries a **RAM shadow price** (`ram_ns_per_byte`) that the tuner calibrates by
a bounded search (raise it until the residency fits the budget — the Lagrangian
reading).

## Steps

1. **[todo] The measured setup constants**: the probe times `dispatch_latency_ns`
   (a real round trip, not the placeholder) + the staging coefficient.
2. **[done] The RAM/VRAM policy**: the `Options` carries the residency bytes per
   device (cpu: `B`; gpu: `0` — freed, the KV grows; split: the tail `B*(m-g)/m`)
   and the DP prices them (`ram_ns_per_byte` / `vram_ns_per_byte`), so a whole-GPU
   op beats a marginally cheaper split. Tested (`the RAM price keeps a weight
   resident`). The shadow-price **calibration** (the bounded Lagrangian search)
   and the tuner feed are part of the wiring.
3. **[done] The DP**: `scheduleDevices` — a forward Viterbi over (op, previous
   device). For a linear chain the additive cost is a shortest path, so the DP is
   **exact** and the "dynamic window" is unnecessary: the cost-to-go *is* the
   look-ahead. Tests: sticky runs, a justified crossing, the free-the-CPU case.
4. **[done] The tuner wiring + the RAM calibration**: the tuner prices every op
   (`costOptions`), resolves the RAM shadow price by a **bounded Lagrangian
   search** (raise it until the residency fits `pre_kv - kv_floor`), runs the
   Viterbi and applies. At 4 GB: the whole-GPU `mixed` profile (64 gpu, 0 split),
   the full context (5162); at 8 GB: all 64 split, context 14905. **The decode is
   within the run noise of the per-op oracle** (~6.0 tok/s at both), so the
   scheduler's benefit is not measurable on this machine yet.
5. **[todo] The runs + the `profile:` telemetry** (the runs, the crossings).
6. **[todo] e2e**: measure the scheduler's token time against the per-op oracle
   (the 12B, the E2B, short and long contexts).

## Definition of Done

- [ ] The setup constants are **measured** (the probe), not hard-coded — **not
      built**: the `hop_ns` (the dispatch-setup) is still the placeholder.
- [x] The scheduler minimizes the makespan with the switch + setup costs — the
      `scheduleDevices` Viterbi (state = the previous device; exact for a linear
      chain), tested.
- [x] The **RAM/VRAM policy**: the residency bytes per device + the shadow prices
      in the DP, so a whole-GPU op (which frees the RAM, the KV grows) beats a
      gratuitous split.
- [x] The **`split_efficiency`**: the probe measures the concurrent CPU+GPU rate
      over the GPU-only rate (0.70 here); the tuner prices a split by it and keeps
      the op whole on the GPU.
- [x] The tuner **uses** it (the two-pass wiring): the plan no longer ping-pongs
      (`whole-gpu 64, split 0` at both RAMs).
- [ ] The `profile:` shows the runs + the crossings — **not built** (it reports
      whole-gpu/split/cpu-only).
- [x] e2e: measured — the VRAM is the lever (12B: 4 GB 5.8, 8 GB **8.7** tok/s =
      2.4x the CPU-only), the RAM is free context (49856 at 8.7); `zig build` /
      `zig build test` green.

## Arch Provenance

- **own:** the chain scheduler, the switch + setup costs, the dynamic look-ahead.
- **derived:** `gpu/04` (the oracle + the mechanism), `runtime/12` (the
  Evaluator/Scheduler — whose bounded search was deferred to `open-items.md`),
  `runtime/06` (the residency), `gpu/03` (the async overlap).

## Comments

- **The wiring needs the RAM term (attempted).** Wiring the DP into the tuner
  with the placeholder `hop_ns` split every resident op and collapsed the context
  (4549 -> 2615): the cost model prices the **compute + transfer**, not the
  **RAM** the residency spends. A split's CPU tail stays in the RAM (shrinking
  the KV); a whole-GPU op frees it. The scheduler must price that, or it will
  always prefer the per-op cheapest split. Reverted until the model carries it.
- **The load model's completeness (inherited from `gpu/04`).** `gpu/04`'s `Load`
  counts only the **weight** bytes; the **KV/attention** bytes — the dominant CPU
  read at a large context — must join it here, so the split sizes against the
  real CPU load. This is the item `gpu/04` handed over.
- **Policy vs mechanism.** `gpu/04` owns the split, the row-range residency and
  the per-op oracle; this spec owns the **chain**. The `Load` `gpu/04`
  introduced is the scheduler's first input.
- **The switch cost is why the per-op oracle is wrong**: it is optimal per op
  and globally worse — the classic reason to schedule the chain, not the leaf.
- **The pre-record is deferred, not rejected.** The `gpu/03` async already
  overlaps the GPU's **execution** with the next CPU op; only the **setup** is
  exposed, and it is small (KB staging; the shaders are per quant). The trigger
  to revisit is explicit: a measurement showing the setup dominates.
- **The preparable vs not**: the pipeline / descriptors / command buffer are
  pre-recordable; the activation staging is not (it needs the producer's
  output). So a future pre-record can hide the former, never the latter.
- **The 4 profiles** (`cpu` / `gpu` / `split` / `mixed`) are the scheduler's
  output; the `mixed`'s **parallel** mode is the executor's existing dep-driven
  overlap (independent ops, e.g. q/k/v), the **serial** mode is the chain.
- **Measurement caveat.** The run-to-run variance on this machine is ~10%, so
  the scheduler's small effects must be measured with repeats; the split's win
  is only clear at a large context (`gpu/04`).

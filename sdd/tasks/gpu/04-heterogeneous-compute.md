04 - Heterogeneous Compute + GPU-aware RAM Budget

Status: done (the device oracle, the split — plain and `gate_up` —, the row-range
RAM residency, the global load, the 4-profile telemetry and the GPU-aware budget
are implemented and measured). The prefill GEMM split is **not pursued** (a
one-shot prefill, low ROI); the KV bytes in the load model move to `gpu/11`.

**Outcome (measured, `gpu/11`):** the split is a **capability**, not a win on
this machine — the concurrent CPU+GPU rate is **0.70x** the GPU-only rate (the
memories contend), so the oracle keeps the op whole on the GPU. The performance
lever is the **VRAM resident set** (the `gpu/06`), not the split (12B: VRAM 4 GB
= 5.8, 8 GB = 8.7 tok/s).
Commits:

## Goal

Three things that only make sense together:

1. **One device oracle.** Every dispatch is decided by a **single cost-based
   oracle** — **CPU**, **GPU**, or **split (CPU + GPU)** — and the chain's
   profile **emerges**: `cpu`, `gpu`, `split` (intra-op) or `mixed` (inter-op,
   serial or parallel). It serves any op (single, fused, seq), replacing the
   per-path binary `decideOffload*` calls with one decision.
2. **Heterogeneous compute.** Let one op run on the **CPU + the GPU** over
   disjoint output rows, concurrently, so the two processing units **add up**
   instead of choosing one.
3. **A GPU-aware RAM budget.** When `--gpu`, **redesign the RAM budget** (its
   layers) to hold the ops that give the GPU the most return — the **split ops'
   weights** — instead of the KV. The premise: with `--gpu` the RAM should buy
   **throughput** (the heterogeneous compute) rather than **context** (the KV).

## Scope

- **In:** the **device oracle** (eligibility + partition + global allocation; the
  4 profiles); the row split; the executor running the CPU slice (the pool) and
  the GPU slice (the device) with the `gpu/03` overlap; the GPU-aware RAM budget
  (the KV at the floor, the split weights prioritized, decided by cost); the
  **precise** duplication (only the split ops' read ranges).
- **Out:** the `gpu/05` spillover (closed — the PCIe ROI is negative); multi-GPU;
  network; cross-token pipelining (overlapping *dependent* ops between tokens).

## Design

### Measure first (the go/no-go)

Before wiring anything, **measure** the heterogeneous case: run the CPU and the
GPU on the **same op over disjoint rows, concurrently**, and compare the combined
throughput against the CPU-only and the GPU-only (per shape / model). The split
makes sense only where the combined **beats both** — the premise is that the two
memory systems add (the CPU's DDR4 ~34 GB/s + the GPU's GDDR5 ~75 GB/s). **Report
it; if it loses, stop.**

**Measured — GO.** A q4_0 GEMV (30720 x 3840, 63 MB) on the RX 580, 4 CPU threads,
min-of-7:

| | rate |
|---|---|
| CPU-only | 23.8 GB/s |
| GPU-only (resident) | 46.3 GB/s |
| **split** (GPU rows first) | **49.1 GB/s** (~2.06x the CPU-only) |
| split (CPU first) | 32.2 GB/s |
| sum (ideal) | 70.1 GB/s |

The split reaches **~70% of the ideal sum** (the memory controller is shared) and
~2x the CPU-only. Parity is **exact** (disjoint rows). The **order matters**: the
GPU slice submitted **first** beats the CPU first (49.1 vs 32.2) — so the tuner
emits the GPU dispatch before the CPU one. A separate proxy (a CPU-only decode
and a GPU decode running at once) shows heavy contention (the GPU decode halves
when the CPU is saturated), so the split's end-to-end gain is bounded by the
CPU's remaining bandwidth — measure the real decode too.

**Real decode — implemented on `the current implementation`.** The split is implemented: the
plan (`Dispatch.row_start`/`row_count`, `VramResident.rows`, format **v25**), the
residency (a partial upload of the GPU rows only), the executor (the row range
plus the concurrent CPU slice) and the tuner (`g = m * gpu_rate/(gpu_rate +
cpu_rate)` on the resident decode GEMVs). 285/285 tests green. The 12B
(`--threads 4 --gpu --vram 4gb`, a fresh calibration):

| `--ram` | plan | context | decode |
|---|---|---|---|
| 4 GB | no split | 5162 | 5.9 tok/s |
| 4 GB | split | 4096 | 5.1 tok/s |
| 4 GB | **split + tail residency** | 4549 | 5.6 tok/s |
| 8 GB | no split | 17645 | 2.6 tok/s |
| 8 GB | **split** | 16263 | **3.2 tok/s** |

The **row-range RAM residency** landed: a split tensor's range covers only the
CPU tail `[g, m)`, so the RAM budget never accounts for the GPU rows (context
4096 -> 4549 at 4 GB, decode 5.1 -> 5.6). At the large context the split wins
**~23%**; at the small one it is still slightly below the no-split (5.6 vs 5.9),
where the CPU is already the bottleneck. The GPU-aware budget (the last piece)
would cap the KV at the floor for the split ops.

### The device oracle (the 4 profiles)

Every dispatch gets **one** decision, and the chain's profile **emerges** from it.
That decision is a **single cost oracle**, not a special case per path:

```
decideDevice(op) -> { cpu | gpu | split(g) }        -- the per-dispatch state (3)
```

**Eligibility.** Only ops with a ported GPU kernel can leave the CPU (`matvec`,
`matmul` today); the rest (`norm`, `rope`, `attention`, `elementwise`, `dequant`)
are CPU-only.

**Partition.** `cpu` / `gpu` take the whole op; `split(g)` divides the *output
rows* of one op across the devices (`[0, g)` on the GPU, `[g, m)` on the CPU).
The slices are disjoint, so the partition is **concurrent by construction**.

**Allocation.** The devices are a *global* resource — the CPU slices of every op
share one memory bandwidth, and the GPU link is shared too. So the oracle is not
per-op independent; the chain decides globally:

```
minimise  max( gpu_bytes/gpu_rate , cpu_bytes/cpu_rate )   s.t. VRAM, RAM
```

**The 4 profiles** (the chain aggregates the per-dispatch states):

| profile | how it arises |
|---|---|
| **cpu** | every dispatch `cpu` |
| **gpu** | every dispatch `gpu` (needs a GPU kernel for every op — not yet) |
| **split** | some op is partitioned (intra-op) |
| **mixed** | ops land on **different** devices (inter-op) |

`mixed` schedules in one of two **modes**:
- **serial** — the ops are dependent (a decode chain): only the per-op
  **attribution** (each op on its fastest device);
- **parallel** — the ops are **independent** (q/k/v, gate/up): they overlap.

**Attribution vs partition vs parallel — what needs what:**

| gain | mechanism | needs independence? |
|---|---|---|
| attribution | each op on its best device | **no** — a serial chain benefits (Op1 CPU -> Op2 GPU -> Op3 CPU) |
| partition (`split`) | one op's rows on both devices | **no** — the slices are independent by construction |
| parallel (inter-op) | different ops overlap | **yes** — the ops must be independent |

So the common "offload the heavy op" case is `mixed`/**serial** with the
**attribution**; the **partition** spends the CPU that would otherwise idle while
a heavy GPU op runs; the **parallel** is the extra win of the independent groups.
The attribution and the partition are why the split can pay even in a serial
decode.

### The split as first-class (row-range residency)

The one idea: make the **output-row range** a first-class citizen of the plan and
the residency, so a weight's rows can live in **two tiers** and one op can run on
**two devices**:

```
y[0..g)  = W[0..g) * x     (GPU, reads the VRAM)
y[g..m)  = W[g..m) * x     (CPU, reads the RAM)      -- concurrent, same x
```

Disjoint output rows → no shared reduction → trivial parity. The two slices read
**different memories** (GDDR5 vs DDR4), which is what lets them add. The split's
PCIe cost is the **activation** (the weights are VRAM-resident), small — but the
ROI must **pay it**.

**Plan (`Dispatch`).** Add `row_start: u32 = 0` and `row_count: u32 = 0` (`0` =
the whole tensor, today's behaviour). A split op is **two dispatches** sharing
`tensor`/`in`/`out`, the GPU one first:

```
[i]    device=1  row_start=0  row_count=g      tensor=W in=x out=y
[i+1]  device=0  row_start=g  row_count=m-g    tensor=W in=x out=y
```

Two dispatches (not a `device=2`): the flat `Dispatch[]` stays the single source,
the executor already handles `device 0/1`, and the runtime still decides nothing.

**Residency (the precise duplication).** `vram_resident` becomes `(tensor,
gpu_rows)`: `openResident` uploads only `data[0 .. g*row_bytes]`. The `gpu/05`
exclusion becomes row-range aware: drop rows `[0,g)` from the RAM residency (they
are in the VRAM) and **keep `[g,m)`** (the CPU reads them). So VRAM = the GPU rows,
RAM = the CPU rows — only the split ops' ranges, never the whole tensor.

**Executor.** `runMatvec` honours `row_start`/`row_count`: the CPU slice calls
`matVecPlan(..., begin=row_start, end=row_start+row_count)` (the `Args.begin/end`
and the `Pool` row split already exist; the pool only needs the `g` offset); the
GPU slice calls `matVecResident(res, q, row_count, k, x, out[0..row_count])`. The
overlap is the existing `gpu/03` async: `submitResidentAsync` submits the GPU
slice, the loop runs the CPU slice on the pool, and the consumer (or `drainAll`)
drains the GPU. `asyncResidentGemv` must use `row_count` (not `view.dims[1]`) and
the partial resident handle.

### The cost model (balance the two devices)

Per op, `cpu | gpu | split`, and the split point `g`:

- `cpu`: `t = B / cpu_rate`; `gpu` (resident): `t = B / gpu_rate`.
- `split`: the slices run concurrently → `t = max(g*r/gpu_rate, (m-g)*r/cpu_rate)`.
  The optimum **balances** them: `g/m = gpu_rate/(gpu_rate+cpu_rate)`, giving
  `t_min = B/(gpu_rate+cpu_rate)`.

The global objective is to **balance the two devices' byte totals** so neither
idles:

```
minimise  max( gpu_bytes/gpu_rate , cpu_bytes/cpu_rate )   s.t. VRAM, RAM
```

The split's real lever is the **VRAM**: it costs only `g*r = B*gpu_rate/(gpu_rate
+cpu_rate)` (~half) of VRAM, so more ops fit resident — and each is faster (the
CPU adds its bandwidth). The tuner must also budget the **CPU's total** (the sum
of the CPU slices plus the streamed weights and the KV) against `cpu_rate`.

### The GPU-aware RAM budget

Today the RAM layers are: the **KV**, the **pinned weights**, the **window**, the
**row-cache** — designed for the CPU. With `--gpu`:

- The objective shifts to the **GPU's ROI**: the RAM holds the **split ops'
  weights** (the CPU-side copies) at the expense of the **KV**, which stays at its
  **floor** (the min context).
- **The KV is a decode cost, not free** (measured, `gpu/05`): growing the context
  slows the decode ~linearly (the CPU attention reads the growing KV). The 12B
  sweep `--ram` 4 -> 8 -> 12 GB gives context 5162 -> 17645 -> 30128 at **5.9 ->
  2.6 -> 1.1 tok/s**, with the weights residency **unchanged**. So the **leftover
  RAM** should go to the **hot set / prefetch** (throughput), **not** to a bigger
  KV: **cap the context** at the floor instead of letting the leftover grow it.
  Today the tuner inverts the KV window to fit the weights and lets the context
  grow with the leftover; with `--gpu` the leftover should buy throughput.
- The tuner decides by **cost**, automatically: the split's throughput gain (the
  CPU bandwidth added) vs the KV's context value. The same Evaluator/cost idea
  `runtime/12` sketches.
- The **duplication is precise**: only the **split ops' read ranges** are
  duplicated — the VRAM (the GPU slice) + the RAM (the CPU slice) — **never the
  whole tensor**, and **never** for a non-split op (there the `gpu/05` exclusion
  holds: one tier per weight).

## Steps

1. **[done] Measure** the heterogeneous case (the CPU+GPU vs the CPU-only / the
   GPU-only) — **GO** (the isolated GEMV, ~2x).
2. **[done] The plan + the row sub-range**: `Dispatch.row_start`/`row_count`, the
   partial `VramResident`, the executor's concurrent CPU + GPU slice, the pool's
   row offset.
3. **[done] The precise duplication**: the row-range RAM residency (a split
   tensor's range covers only the CPU tail `[g, m)`; context 4096 -> 4549 at
   4 GB).
4. **[done] The device oracle**: one `decideDevice(op) -> {cpu | gpu | split(g)}`
   in `tuner/gpu.zig`, the single decision the tuner calls for **every** op
   (`matvec`, `matmul`, `gate_up`). The per-kind cost stays internal
   (`decideOffload*` are helpers now); the plan is byte-identical (behavior-
   preserving refactor).
5. **[done] The global allocation**: the oracle takes the chain's running `Load`
   (`gpu_bytes`/`cpu_bytes`) so a split balances the *global* totals. The verdict
   also reports the bytes each device keeps, which the tuner accumulates. (The
   model counts the weight bytes; the KV/attention bytes are a `gpu/11` item.)
6. **[done] The `gate_up` split**: the fused pair splits like a plain GEMV (the
   GPU rows `[0, g)` of both weights, the CPU `[g, m)`). The prefill GEMM split
   is **not pursued** (one-shot, low ROI).
7. **[done] The 4 profiles (telemetry)**: the tuner prints the profile the oracle
   produced — `cpu` / `gpu` / `split` / `mixed` — with the gpu/split/cpu dispatch
   counts. The `mixed`'s parallel mode is the executor's existing dep-driven
   overlap (no new work).
8. **[done] The GPU-aware budget (re-scoped)**: with a split, the RAM prioritizes
   the weights the CPU reads over growing the KV (`want_window =
   pin_weights_bytes`; the KV keeps its floor). The original "cap the KV" rested
   on the stale sweep and is dropped — the split *wins* at a large context.
9. **[done] Parity + e2e**: parity is exact; the e2e is measured — the split
   wins at a large context; the small-context results are within the run noise
   (~10%).

## Definition of Done

- [x] The heterogeneous measurement is reported (the CPU+GPU vs either alone).
- [x] The split is parity-correct; the cost selects it (`splitRows`).
- [x] The row-range RAM residency (only the CPU's tail `[g, m)`), so a split does
      not shrink the KV.
- [x] The **device oracle**: one `decideDevice` the tuner calls for every op; the
      per-path `decideOffload*` **calls** are gone (the functions are internal
      cost helpers). The plan is unchanged.
- [x] The **global allocation** feeds the oracle (the chain's running `Load`);
      the split sizes against the global byte totals.
- [x] The **`gate_up` split** (the prefill GEMM split is **not pursued** — a
      one-shot prefill, low ROI; the per-token decode is where the split pays).
- [x] The **4 profiles** surfaced as telemetry (`profile:` in `tune`); the
      `mixed`'s parallel mode is the executor's existing overlap.
- [x] The **GPU-aware budget**, re-scoped to prioritize the CPU's weights over
      the KV (the KV keeps its floor).
- [x] The CPU plan unchanged; deterministic; `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the split + the GPU-aware budget.
- **derived:** `gpu/03` (the overlap), `gpu/05` (the exclusion), `runtime/12`
  (the cost), `runtime/06` (the RAM residency).

## Comments

- **The oracle:** one `decideDevice` decides every op; the **4 profiles**
  (`cpu`, `gpu`, `split`, `mixed`) emerge from it. The **attribution** and the
  **partition** need no independence (they pay even in a serial decode); only the
  **parallel** (inter-op) does.
- **The premise:** with `--gpu`, the RAM buys **throughput** (the heterogeneous)
  rather than **context** (the KV). This could beat keeping the KV in RAM.
- **Measured backing (`gpu/05`):** the bind (the GPU reading the RAM directly) is
  **5.43 GB/s** vs the CPU's 35, and the GPU reads **only weights** (no attention
  kernel — the K/V has no GPU reader). So the RAM here goes to the **hot set**,
  and the KV stays at the floor; a bigger KV is a **decode cost**, not a win.
- **Precise duplication:** only the split ops' read ranges; never the whole model;
  never a non-split op.
- **Measure first:** if the heterogeneous does not beat either device, stop — do
  not redesign the budget for a split that loses.
- **Coherent with `gpu/05`:** `single` stays one-tier (the exclusion); `split`
  re-duplicates **deliberately** (only the ranges it reads on both sides).

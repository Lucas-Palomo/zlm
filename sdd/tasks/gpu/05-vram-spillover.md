05 - VRAM Extended by RAM (spillover / virtual_vram)

Status: done (closed — the ROI does not pay; see the ROI finding)
Commits:

## ROI finding (why this is closed)

The spillover ("virtual_vram": a resident region lives in RAM and is streamed
into VRAM on demand) is a **negative ROI**, structurally and **independent of the
model size**, because the tiers are fixed:

| tier | who reads it | rate |
|---|---|---|
| VRAM (memory clock ramped) | GPU | ~75 GB/s |
| RAM | CPU | ~34 GB/s |
| **RAM -> VRAM (PCIe/staging)** | GPU | **~8 GB/s** |
| disk (HDD) | CPU | ~0.2 GB/s |

The order is **VRAM > RAM > PCIe > disk**: the spillover reads the RAM over the
**PCIe (~8)**, which is **slower than the CPU reading the same RAM (~34)**. So a
weight that fits the RAM is **better on the CPU**, and the spillover only wins if
the PCIe beat the CPU — it cannot (the GPU link is slower than the CPU's RAM).

- **Larger models** do not change this: most of the weights live on the **disk**
  (~0.2), so the bottleneck is the disk, not the tier; the spillover is neutral
  at best.
- The only exceptions are **speculative and load-specific**: the GPU duty cycle
  (a busier GPU ramps the memory clock, speeding the VRAM-resident part) and CPU
  bandwidth contention (the spillover relieving a saturated CPU) — neither is a
  reliable lever.

The **real ROI is in the budget, not the spillover**: a weight should live in
**one** tier (the fastest that holds it) — VRAM -> GPU, RAM -> CPU, disk -> the
rest — with **no duplication**.

**The cleanup landed with this close (commit `69d8996`).** A `vram_resident`
weight leaves the RAM residency (`ranges`/`weights`/`pinned`/`nodes`) **iff the
CPU never reads it** (no `device == 0` dispatch references it, `tensor` or
`tensor2`; the prefill dispatches count only when the plan runs prefill). Effect
at `--vram 4gb`:

| | E2B | 12B |
|---|---|---|
| RAM `weights` | 2885 -> **1898 MB** | 6411 -> **2920 MB** |
| `ranges` | 553 -> 348 | 610 -> 499 |
| context | 131072 (max) | 4096 -> **5162** |

The CPU plan is **byte-identical**; the decode is unchanged (E2B ~24, 12B ~5.75
tok/s); the excluded set is provably GPU-only (a guard test). The measured RSS
barely moves (the image is mmap'd; the freed budget is reallocated to the KV) —
the win is the **12B context (+26%)**, not a smaller RSS.

## Bind finding (the shader reads the RAM directly / GART) — measured

The spillover's cousin is the **bind**: keep a region in the RAM and let the
**shader read it over the PCIe** (a `HOST_VISIBLE` storage buffer, or the AMD
GART / ReBAR mapping) instead of staging it into VRAM. It is the same tier
question, and it is now **measured**, not modelled.

The backend's **streamed** path *is* the bind: `ensureBuffer` allocates a
`HOST_VISIBLE | HOST_COHERENT` buffer, `stageMapped` copies the weight in, and
the shader reads that buffer **directly**. So the probe's two rates are a direct
A/B on the same q4_0 GEMV:

| who reads | rate | source |
|---|---|---|
| GPU <- **VRAM** (resident) | **33.53 GB/s** | `gpu_gemv_gbps` |
| GPU <- **RAM** (bind, host-visible) | **5.43 GB/s** | `gpu_gemv_streamed` |
| **CPU** <- RAM | **35.00 GB/s** | `dram_bw_gbps` |

The bind loses to the CPU by **~6.5x**; even the PCIe peak (~8) loses. A true
zero-copy bind (skipping the `stageMapped` RAM->RAM copy) removes only the extra
copy — the **PCIe read stays the ceiling**. The probe is deliberately cold (no
sustained warm-up, see `gpu/backend/root.zig`), so 33.53 is the **unramped** VRAM
figure; the ramp (`gpu/07`) raises the *resident* rate toward the ROI table's 75,
but the **bind's PCIe ceiling does not move**. The conclusion of the ROI finding
is unchanged.

**Only the weights are GPU-read.** `zlm`'s GPU is GEMV-only, and the plan
confirms it: the GPU dispatches (**64 `matvec` decode + 336 `matmul` prefill** in
the 12B plan) reference **only `.weight` tensors** — every non-`.weight` tensor
(`attention`/`write_kv`/`norm`/`rope`/`embed`) stays on the CPU; the shader
catalog (`gpu/kernels/shaders/`) has only `matvec_*`/`matmul_*`. So the **GPU
never reads the K/V**, and the K/V is never resident in VRAM to be "freed". A bind
can only "free" what is resident, and the **only** resident thing is the weights —
binding a weight does not free VRAM for another weight, it just puts that weight
on a slower tier (5.43 < 35). So the weight bind is **strictly worse**.

**Break-even.** The bind pays only when the VRAM freed outweighs the slower read
of the bound segment:

- gain (1 GB of weight, CPU -> VRAM) = `1/CPU - 1/VRAM`
- cost (1 GB of segment read over the bind) = `1/PCIe - 1/VRAM`
- pay iff `traffic/bytes < (1/CPU - 1/VRAM) / (1/PCIe - 1/VRAM)`

| rates | gain | cost | break-even `traffic/bytes` |
|---|---|---|---|
| VRAM **ramped** (~75) | +15.2 ms/GB | 170.8 ms/GB | **< 0.089** |
| VRAM **measured** (33.53) | **-1.25 ms/GB** | 154.3 ms/GB | **< 0 -> nothing pays** |
| PCIe peak (8) | +15.2 ms/GB | 111.7 ms/GB | < 0.137 |

With the **measured** (unramped) rate the VRAM is no faster than the CPU, so the
gain term is **negative** — nothing pays, not even in principle. The bind only
becomes interesting if the memory clock is ramped **and** the segment is read
rarely relative to its size.

**Per-segment verdict.**

| segment | GPU-read? | traffic/bytes | resident in VRAM? | bind pays? |
|---|---|---|---|---|
| weights (matvec/matmul) | **yes** | 1.0 | yes | **no** (5.43 < 35; frees no VRAM) |
| activations x/y | yes (staged) | ~1 | no (ephemeral, KB) | no |
| K/V sliding | **no** | ~0.06 (window/context) | no | moot |
| K/V global | **no** | 1.0 | no | moot |
| norms / rope / elementwise | **no** | ~1 | no | moot |
| embeddings (lookup) | **no** | ~1/rows | no | moot |
| row-cache (prefix) | **no** | low | no | moot |

The only segments with a favourable ratio (the **sliding K/V** and the **embedding
lookup**) are **CPU-only**, so the bind is moot for them; the only GPU-read segment
(the weights) has ratio 1, so it loses. **No RAM segment pays as "bind only".**

**Caveat (not a law).** This holds for today's **GEMV-only** GPU. If the GPU ever
gains an **attention kernel** (then it would read the K/V) *and* the memory clock
is ramped, the **sliding K/V** at a large context (>= ~11x the window, where the
ratio crosses 0.089) becomes the one bind candidate — the only segment whose
traffic decouples from its size.

**And the K/V is a decode cost, not free.** Sweeping `--ram` on the 12B
(`--threads 4 --gpu --vram 4gb`; the weights residency is **identical** across
all three — pinned 2064 / window 2462 MB):

| `--ram` | context | decode |
|---|---|---|
| 4 GB | 5162 | **5.9 tok/s** |
| 8 GB | 17645 | **2.6 tok/s** |
| 12 GB | 30128 | **1.1 tok/s** |

(The 8 GB row is re-measured with `--force`, a fresh calibration; the earlier
1.6 tok/s came from a stale plan. The 12 GB row is from an earlier pass and still
needs a `--force` re-check.)

Only the KV grows; the decode falls **~3.7x** (the CPU attention reading the
growing KV, the per-layer time rising ~3.8x). So a bigger `--ram` that grows the
KV **buys context at the cost of throughput** — see `gpu/04` for the budget
implication.

## Goal

Let the resident set (weights plus KV) exceed physical VRAM: the overflow
("virtual_vram") lives in RAM and is streamed/paged into VRAM on demand, so the
model never has to fit VRAM and a small `--vram` degrades the GPU, not the
weights.

**Not pursued** — the ROI finding above closes it.

## Scope

- **In:** a VRAM budget in the tuner (headroom plus an ROI knapsack over resident
  vs virtual regions); the runtime's VRAM residency (prefetch/evict/stream
  between RAM and VRAM); the plan recording the VRAM map; the KV optionally
  resident in VRAM.
- **Out:** multi-GPU; network; a full GPU pager (keep it a bounded window).

## Design

- Tuner: the ROI knapsack packs regions into `--vram − headroom` as **resident**;
  the overflow is **virtual** (RAM-backed), streamed to VRAM per use. Recorded in
  the plan (`vram_map`), like the RAM residency `nodes`.
- Runtime: a VRAM window with the same discipline as the RAM residency
  (`prefetch`/`keep`/`evict`), but the target tier is VRAM; staged from the mmap
  (RAM/page cache) into a device buffer. A small `--vram` shrinks the resident
  set first.
- KV: when it fits, keep the K/V resident in VRAM (the earlier local prototype's `VramKvCache` idea);
  otherwise it stays in RAM.
- Depends on `gpu/03` (async) for the streaming to overlap.

## Steps

1. tuner: VRAM budget (headroom plus resident/virtual knapsack); plan `vram_map`.
2. device/runtime: VRAM residency (prefetch/evict/stream), staged from RAM.
3. optional VRAM KV.
4. e2e: `--vram` below the resident set works; measure.

## Definition of Done

- [ ] `--vram` smaller than the resident set runs (spillover to RAM), with the
      budget honored (anti-OOM headroom).
- [ ] The GPU degrades before the weights; correctness holds.
- [ ] Deterministic plan; CPU plan unchanged; `zig build` / `zig build test`
      green.

## Arch Provenance

- **own:** the current VRAM residency plus spillover.
- **consulted:** the earlier local prototype's `VramMemoryMap`/`virtual_vram`, `VramKvCache`.
- **derived:** `gpu/01`, `gpu/02`, `runtime/06` (RAM residency).

## Comments

- **Closed — the ROI finding (above).** The spillover streams the RAM to the VRAM
  over the PCIe (~8 GB/s), **slower than the CPU reading the RAM (~34 GB/s)**; it
  cannot pay, at any model size. The real ROI is the **budget**: one tier per
  weight (VRAM -> GPU, RAM -> CPU, disk -> the rest), no duplication.
- **Closed — the bind finding (above), measured.** The GPU reading the RAM
  directly (host-visible/GART) is the same tier question: **5.43 GB/s**, losing to
  the CPU's 35. The K/V is **not** a bind candidate (the GPU has no attention
  kernel, so it never reads the K/V), and a bigger KV is a **decode cost**, not a
  win. So no RAM segment pays as "bind only".
- The `Tier.vram` already exists in the plan's `ResidencyGraph`; this task would
  have given it a runtime — **not pursued**.
- Depends on `gpu/03` for overlap; benefits from `gpu/04` on UMA.
- **Reopened as a *declared exception* (`gpu/11`).** The **general** runtime
  spillover stays closed (the ROI above), but the new RAM/VRAM policy
  reintroduces it as a **tune-declared** exception: a weight migrating RAM -> VRAM
  streams (the shader reads the RAM) until the upload completes, then reads the
  VRAM. The tuner declares the target tier + the trigger step + the bridge; the
  runtime applies. Bounded, paid once — the cost of *moving*, not a general
  pager. The exclusivity rule here (one tier per weight) still holds.

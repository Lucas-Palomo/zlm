# CPU Benchmark: zlm vs llama.cpp

Measurements of `zlm` against `llama.cpp` (the reference) on CPU, same model and
same thread counts — first with no RAM limit, then under Docker memory caps. The
goal is to place `zlm`'s throughput against the reference and to locate its
**operational RAM floor**.

All numbers are decode (`tok/s`) unless stated otherwise, and come from repeated
runs (median/best of N) with the model warm in the page cache. They are local
measurements from the development host, not vendor claims.

## Proposal

Compare `zlm` with `llama.cpp`, the reference implementation for CPU inference.
The comparison answers two questions:

1. **Speed** — what `zlm` delivers in decode, at the same thread settings, with
   plenty of RAM.
2. **Memory resilience** — down to what RAM floor each runtime runs the same
   model, which is `zlm`'s value proposition ("anti-OOM by design").

## Info

### Hardware + OS

- **CPU:** AMD Ryzen 7 5700X (Zen 3), 8 cores / 16 threads; AVX2 + FMA, **no
  AVX-512**; 32 MiB L3; up to ~4.66 GHz.
- **RAM:** 32 GB DDR4 dual-channel. Measured read bandwidth: **~23 GB/s**
  (1 thread), **~37.8 GB/s** (4 threads, peak), **~34.8 GB/s** (8), ~34 GB/s
  (16).
- **Storage:** the model store lives on an **HDD** (Seagate ST2000DM008, 2 TB,
  7200 rpm). Cold reads are disk-bound; warm reads come from the OS page cache.
- **OS:** Manjaro Linux, kernel `6.18.49-1-MANJARO`, x86_64.
- **Containers:** Docker 29.7.2, cgroup v2, `--memory` + `--network none`.

### Model

- **Gemma-4 E2B**, **Q4_0** quant (unsloth), GGUF of **3.04 GB**
  (`gemma-4-E2B-it-unsloth-Q4_0.gguf`).
- The `zlm` image `e2b` was compiled from that GGUF (byte-verified against it).
- Native context **128K**; the tuner's `min_context` is **4096**.
- Roofline reference: bytes read per decode token = **1334.2 MiB** (~1.399 GB).
- **llama.cpp:** image `ghcr.io/ggml-org/llama.cpp:full` (build 11176, AVX2 CPU
  backend). The host also has `llama-cli`/`llama-bench` (build 10809).

## Benchmark Notes

Important caveats for interpreting the numbers above.

### Machine drift

Every number is subject to drift. CPU frequency scaling (thermal/turbo), the
page-cache state, and single-thread micro-benchmarks introduce run-to-run
variance: **~3%** in isolated kernel benchmarks and up to **~5–10%** in
end-to-end decode. Comparisons use repeated runs (median/best of N) and
interleaving where possible to control for it; treat small differences as noise.

### zlm build (host / Docker)

`zlm` is built once with
`zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseFast -Dcpu=native`, i.e.
codegen targeted at this exact CPU (Zen 3, AVX2 + FMA). The musl target makes
the binary statically linked, so it is **the same one** run on the host and
inside the container (`docker/Dockerfile` copies `zig-out/bin/zlm`). There is no
host/container build difference — only the memory cap and the cgroup differ.

### llama.cpp version/build (host / Docker)

- **Docker (used for all reported numbers):** `ghcr.io/ggml-org/llama.cpp:full`,
  `llama-completion` — version `0.5.0-dev`, build **11176** (commit `f805c57a7`),
  built with GNU 14.2.0, AVX2 CPU backend.
- **Host (sanity only):** `llama-cli`/`llama-bench` — version `0.4.0-dev`, build
  **10809** (commit `5266f24da7`), built with GNU 16.2.1, loading the `haswell`
  CPU backend (AVX2 + FMA).

The reported `llama.cpp` numbers come from the **Docker image (build 11176)**;
the host build is a different version and is not directly comparable.

### Host mmap cache inside Docker

The model files live on the host filesystem and are bind-mounted into the
container. `run` mmaps them, and the OS **page cache is shared with the host**:
pages already resident (faulted by the host or another cgroup) are not
re-charged to the container's cgroup. Two consequences:

- The cgroup memory numbers (`cgroup_peak`, `memory.stat`) reflect only what the
  container faulted; a process's `VmRSS` can be much larger, because mmap pages
  charged elsewhere still count in the process's RSS.
- **Cold vs warm dominates.** With the store on an **HDD**, a cold run is
  disk-bound (a cold E2B container dropped to ~0.7 tok/s), while a warm run is
  DRAM-bound. All decode numbers here are warm.

## zlm Pipeline

v2 flow: `image compile` → `tune` → `run`.

```text
zlm image compile <gguf>                       # deterministic, hardware-free
zlm tune <image> <alias> --threads N --ram X   # measures and writes the .zplan
zlm run <alias> --prompt ... --max-tokens M --metrics
```

Two setups were used:

- **A) Everything in the container** — `tune` and `run` inside the capped
  container. Measures the total cost under the restriction (includes the tuner
  cost).
- **B) `tune` on the host + `run` in the container** — the plan is generated on
  the host (free RAM) and only `run` executes in the container. Isolates the
  **runtime** footprint from the tuning cost, and is the most precise test for
  the execution floor.

See [`docker.md`](docker.md) for the container workflow and the `docker run`
equivalent of both setups.

On the `llama.cpp` side:

```text
llama-completion -m <gguf> -p <prompt> -n M -t N -c 1024 -no-cnv --ignore-eos
```

## Thread Scaling

No RAM limit, E2B, **32 decode tokens**, median of 3 runs. Only the thread count
changes.

> **zlm plan budget:** the plans were tuned with `--ram 8gb` (an 8 GB window
> budget). The `--ram` value is an important plan parameter — it decides the
> residency window and the context; here it is fixed at 8 GB so only threads
> vary.

| threads | zlm (tok/s) | llama.cpp (tok/s) |
|---|---|---|
| 4 | 18.8 | **22.2** |
| 6 | **20.5** | 21.8 |
| 8 | 19.7 | 21.3 |
| 12 | 20.1 | 19.7 |
| 16 | 14.9 | 12.0 |

**Reading:**

- **zlm scales up** with threads to ~12 and **degrades at 16** (SMT contention).
- **llama.cpp is best at 4 threads** and **declines** monotonically with more
  threads (memory-bound); the two curves **cross around `threads=8–12`**, and at
  12/16 **zlm is ahead**.
- At each runtime's best setting the gap is small: `llama.cpp` ~22.2 (4 threads)
  vs `zlm` ~20.5 (6 threads) — **~8%**.
- **Ceiling (roofline):** with ~37.8 GB/s of DRAM, E2B sits at ~**27 tok/s**
  theoretical; measured ~21 tok/s (~79% of the ceiling). The matvec kernel
  sustains ~31.5 GB/s = **~83% of pure streaming**; the rest is the
  compute/instruction cost inside the loop.

## RAM Survival (Docker)

`threads=8`, plan generated **inside** the container, RAM from 4 GB down to
512 MB (Docker `--memory`; the plan's `--ram` matches the cap). **8 decode
tokens.** This is a **survival** test — does each runtime keep running (and how
fast) as the cap tightens — not a performance scaling test.

| cgroup | zlm (tok/s) | zlm peak | llama.cpp (tok/s) | llama peak |
|---|---|---|---|---|
| 4 GB | 18.1 | 239 MB | **21.4** | 1.4 GB |
| 2 GB | 18.9 | 244 MB | **21.4** | 1.4 GB |
| 1 GB | **18.6** | 234 MB | OOM (SIGKILL 137) | 1 GB |
| 512 MB | **15.6–16.4** | 229 MB | OOM (SIGKILL 137) | 512 MB |

**Reading:** with plenty of RAM `llama.cpp` is ~15% faster; below **1 GB it is
killed by the kernel** (OOM) while `zlm` keeps running. `zlm`'s RSS stays
**flat (~230 MB)** regardless of the cap — it never materializes the whole
model.

## Extreme Survival

`threads=8`, plan generated **inside** the container, RAM from 256 MB down to
32 MB. **4 decode tokens.** Pushes the survival limit further.

| cgroup | zlm | llama.cpp |
|---|---|---|
| 256 MB | 16.0 tok/s (peak 227 MB) | OOM |
| 128 MB | `tune` ok, `run` fails | OOM |
| 64 MB | `window=0`, fails | OOM |
| 32 MB | `tune` is OOM-killed | OOM |

**Reading:** in this sweep `zlm`'s floor shows up at **~256 MB**, and
`llama.cpp` dies at every point.

> **Note:** the first `zlm` failures below 256 MB are the **runtime's own
> footprint on E2B** — the tokenizer + metadata (~128 MB, see *Refined floor*) —
> **not** the model weights or the KV: those are streamed through mmap and do not
> count against the cap.

### Refined floor (host-generated plan, minimal context)

With setup **B** (tune on the host), low-budget plans degrade the **context** and
zero the weight window, and the floor drops further:

| plan | context | min cgroup | tok/s |
|---|---|---|---|
| `--ram 32mb` | 1642 tok | ~160 MB | 15.2 |
| `--ram 16mb` | 806 tok | 128 MB | 14.5 |
| `--ram 8mb` | 351 tok | **128 MB** | 14.0 |

**E2B operational floor ≈ 128 MB.** Below that (96/64/32 MB) the process is
OOM-killed even with the minimal-context plan.

**Where the floor comes from** (RSS breakdown per `load` step):

| step | VmRSS | delta |
|---|---|---|
| open the ZTEF (mmap of the weights) | 2.0 MB | — |
| `Zmeta.open` (metadata) | 55.9 MB | **+54 MB** |
| `Tokenizer.fromZmeta` | 131.9 MB | **+76 MB** |
| zcom / plan buffers / scratch | 132.0 MB | ~0 |
| prefix cache | 134.1 MB | +2 MB |
| residency / prefetch | 134.1 MB | 0 |

So the floor **is not inference**: it is the **tokenizer** (~76 MB: `token_to_id`
+ the BPE merge map) and the **metadata** (~54 MB). The **weights are mmap** (not
charged to the cgroup) and the **KV is small** at the degraded context. That is
why the floor is fixed and model-dependent (vocab + merges), not a function of
the plan budget.

## Adaptive RAM Capacity

The plan budget (`--ram`) is a **residency-window target**, not an RSS
requirement. This shows up in two ways:

**1. Budget larger than the container (setup B).** With `tune` on the host, a
plan with a **1 GB** budget runs inside much smaller containers — the runtime
adapts the resident window to the available cgroup:

| plan (budget) | cgroup | tok/s |
|---|---|---|
| 1 GB | 216 MB | 18.0 |
| 1 GB | 208 MB | 11.8 |
| 1 GB | 200 MB | OOM |
| 512 MB | 208 MB | 16.0 |
| 256 MB | 208 MB | 11.1 |

`zlm` **does not read the memory cgroup** (only `cpu.max`, for CPU). The
adaptation is emergent: `mmap` + `madvise` let the **OS reclaim the page cache**
under the cgroup, shrinking the effective window on its own.

**2. Low budget, host with lots of RAM.** Running the `--ram 8mb` plan on the
**host** (no cgroup), the controlled footprint is:

```text
Rss:           1,514,636 kB  (1.48 GB)
Anonymous:       131,644 kB  (128.6 MB)   <- runtime base (tokenizer+metadata)
Private_Clean: 1,382,472 kB  (1.32 GB)    <- model mmap (page cache, clean)
```

- The **runtime base (anonymous) = 128.6 MB**, matching the tokenizer + metadata
  floor.
- The **weights sit in reclaimable page cache (~1.32 GB here)**: with RAM
  available, the OS keeps the weight window resident; under pressure, it
  discards it. That is, **a low plan budget does not stop `zlm` from using the
  available RAM as its window** — the effective window is whatever the OS
  allows, not what the budget dictates.

**Memory model summary:**

```text
footprint = runtime base (tokenizer+metadata, ~128 MB on E2B)
          + plan budget (fixed + window)
          + weights in reclaimable page cache (sized by available RAM)
```

## Conclusions

- **Speed:** with plenty of RAM, `llama.cpp` wins **~10–15%** in decode; the gap
  comes from the matvec kernel's efficiency (~83% of the streaming peak).
- **RAM:** `zlm` runs the 3 GB E2B with a **~128 MB floor**; `llama.cpp` needs
  ~2 GB and is OOM-killed below ~1 GB. **~16× less RAM** for the same model.
- **zlm's floor is the tokenizer + metadata**, not inference. Making them
  compact/lazy (strings in the mmap, a denser BPE structure, on-demand loading)
  is the path below ~128 MB.
- **The weight window is reclaimable page cache:** the plan budget sets the
  target, but the OS sizes the effective window by the available RAM.

16 - q4_k Catalog: Dynamic Grid

Status: done
Commits:

## Goal

Apply the `kernels/13`–`15` pattern to **q4_k**: one generic body (the q4_k
super-block decode) instantiated at comptime from the same five-axis grid
(`width` x `tile` x `unroll` x `int` x `prefetch`) with the same prunes, so the
q4_k catalog is one grid definition and the Selector picks — replacing the two
hand-written plans (`q4_k` production, `q4_k_u8`).

Why q4_k: it is **ALU-bound** on the packed 6-bit scale decode (measured earlier:
the `u8` gave +6% on the CPU; on the GPU it only ties the CPU), so the
`unroll`/`int`/`width` axes matter **more** here than on the memory-bound q4_0 —
this is where the catalog should actually pay.

## Scope

- **In:** the q4_k generic body + the grid; the prunes (reuse P1/P2); the
  q4_k `int` (integer) decode.
- **Out:** the other quants; the GPU (its own catalog); `scale`/layout.

## Design

### The grid (same axes, q4_k budget)

```
width    ∈ { 4, 8, 16 }
tile     ∈ { 1, 2, 4, 8, 16 }
unroll   ∈ { 1, 2, 4, 8, 16 }
int      ∈ { 0, 1 }
prefetch ∈ { 0, 1 }
budget   = tile * unroll * ceil(width/8) <= 16
name     = q4_k_w{W}t{T}un{U}_i{I}p{P}
```
Reuse `kernels/15`'s machinery: the grid value sets, `q4_0WithinBudget` (rename
generic), the P1 comptime prune (`simd.native`) and the P2 prefetch heuristic
(`cost.prefetchWorthwhile`).

### The generic q4_k body

- Generalize `dotQ4_KPre` (the per-row super-block decode with the precomputed
  `xsum`) into `matVecQ4_KGen(comptime V, ...)` with the `tile`/`unroll` sweep and
  the `@prefetch`, mirroring `matVecQ4_0GenF32`.
- The q4_k `int` axis: the integer (q8-activation) decode — the q4_k block's
  scale/min folded like `matVecQ4_0GenInt`, using the `xsum`/min-term trick the
  scalar path already has.

### Prunes

- Reuse P1 (comptime ISA) and P2 (prefetch) verbatim.

## Steps

1. **Done.** The generic q4_k f32 body (`matVecQ4_KGenF32`) + the grid generation
   (`q4_k_candidates`), reusing the `kernels/15` machinery (the axes/budget/P1/P2
   are shared).
2. **Done.** The q4_k `int` body (`matVecQ4_KGenInt`), the q8-activation integer
   decode with the scale/min folded.
3. **Done.** The prunes wired (P1 comptime, P2 the `prefetchWorthwhile`
   heuristic); `q4_k_u8` subsumed (kept as a parse alias).
4. **Done.** `VariantEfficiencies.max_variants` 128 -> 256; `.zat` VERSION 17;
   plan VERSION 20.
5. **Done.** 267/267; parity for every generated q4_k candidate; effective count
   **90** on this AVX2 host (7 + 90 + 90 = 187 plans).

## Definition of Done

- [x] The q4_k catalog is a generated grid (`q4_k_w{W}t{T}un{U}_i{I}p{P}`); the
      hand-written `q4_k_u8` is subsumed.
- [x] The prunes apply (P1 comptime, P2 tune-time).
- [x] Parity for every generated candidate; the Selector picks; CPU decode
      measured.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the q4_k grid on the `kernels/15` machinery.
- **derived:** `kernels/13`–`15` (the generator), `kernels/09` (the q4_k scalar
  path + the `xsum` trick), `kernels/10` (the integer dot).

## Comments

- **Where it should pay.** q4_k is ALU-bound; the `unroll`/`int` axes are the
  levers, so unlike q4_0 the catalog is expected to move the decode.
- **Reuse, don't fork.** The grid/budget/prune machinery is shared with
  `kernels/15`; only the decode body differs.
- **As-built verdict (honest).** The generated body is ~4% faster than the old
  production per-row path (inlining), but the `unroll`/`int`/`width` axes **do
  not systematically pay** — all within ~2-4% and the winner flip-flops. Worse,
  the selected q4_k variant is **inert at runtime**: the only q4_k tensor (the
  tied `token_embd`, the lm_head) is dispatched via `lmHeadRows`, which calls
  `dotQ4_KPre`/`dotQ4_K` directly and **ignores the plan variant**. So this task
  delivers the generated q4_k catalog (generality, `q4_k_u8` subsumed) but **no
  decode gain**; a real gain needs the lm_head routed through the plan (a
  separate change), and the binary grows (~+45% unstripped, mostly symbols; the
  stripped code cost is ~7.8 KB/instantiation). `kernels/17` generalizes this
  into one generic body.

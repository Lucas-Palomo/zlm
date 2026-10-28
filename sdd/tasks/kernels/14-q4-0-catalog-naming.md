14 - q4_0 Catalog: Naming + Extension

Status: done
Commits:

## Goal

Extend the comptime q4_0 candidate catalog (`kernels/13`) and rename its
variants to a readable `q4_0_tile{T}_unroll{U}` scheme (the current
`q4_0_u4`/`_t2`/`_t4`/`_ur2`/`_t2ur4` names leak the hand-written history).

## Scope

- **In:** the q4_0 variant names; a larger curated candidate list; the
  `VariantEfficiencies` sizing the larger catalog needs.
- **Out:** the other quants' names (they follow when generated); new algorithms.

## Design

### Naming

- `q4_0_tile{T}_unroll{U}` — `T` = output rows advanced per block sweep, `U` =
  blocks processed per iteration. The width stays the build-native width (a
  separate concern; a `w{W}` prefix can be added if width variants are wanted).
  E.g. `q4_0_tile1_unroll1`, `q4_0_tile2_unroll4`, `q4_0_tile4_unroll8`.
- The serialized ids stay stable: the tags are **renamed in place** (the ordinal
  is the id), and a new id is appended after every existing plan.

### Extended candidate list

- A curated set (not the full cartesian) over `tile` x `unroll`, e.g.
  `tile` in {1,2,4,8} x `unroll` in {1,2,4,8}, minus the pointless combos.
- The `VariantEfficiencies` tables are sized to the catalog (a `.zat` version
  bump + re-tune); `max_variants` (16) is already full.

## Steps

1. Rename the q4_0 tags + update their references (tests/tuner/CLI).
2. Extend `q4_0_candidates`; size `VariantEfficiencies` to fit (`.zat` bump).
3. Re-tune; measure the new variants' `variant_eff` and the CPU decode.
4. `zig build` / `zig build test` green; parity for every candidate.

## Definition of Done

- [x] The q4_0 variants use `q4_0_tile{T}_unroll{U}`; ids stable.
- [x] The extended catalog is calibrated and the Selector picks from it.
- [x] Parity for every candidate; CPU decode measured; tests green.

## Arch Provenance

- **own:** the naming + extension.
- **derived:** `kernels/13` (the generator).

## Comments

- **Naming caveat.** Only the generated q4_0 family is renamed; the hand-written
  quants keep their names until they are generated too.
- **As-built.** Tags renamed in place (ids 1/2/3/11-15 unchanged); the catalog
  grew 8 -> 13 by appending (1,8), (2,8), (4,4), (8,1), (8,2) (dropped
  (4,8)/(8,4)/(8,8): `tile*unroll` > 16 live `[T][U]Vec` accumulators spill).
  `VariantEfficiencies.max_variants` 16 -> 32, `.zat` `VERSION` 13 -> 14.
  Re-tuned: best `q4_0_tile1_unroll4` (eff 1.0), with `q4_0_tile1_unroll8` a
  near-tie at 0.9959; all other new candidates land 0.77-0.93. The Selector
  picks `tile1_unroll4` for both E2B and 12B. CPU decode is **flat**: E2B's plan
  is byte-identical (same variant id 12), 12B moved `tile1_unroll1 ->
  tile1_unroll4` and a controlled A/B measured 3.7-3.8 vs 3.8 tok/s (noise).
  The kernel is memory-bound; more tile/unroll does not beat unroll-4.

04 - Inspect and Metrics

Status: done
Commits: a3aad86

## Goal

The current code has no way to see **what the tuner chose** (variants, atomic_ids, `parts`,
phase, residency) or to **measure** a run (TTFT / prefill / decode / RSS). The earlier local prototype has
both (`inspect`, `run --metrics`). Add `zlm inspect` and `run --metrics[=full]` /
`--verbose` so selection and performance are observable — the basis for the
performance work (`runtime/11`, `kernels/07`).

## Scope

- **In:**
  - `zlm inspect <alias> [--plan|--image|--profile|--all]` — print the resolved
    plan: tensors, dispatches (op, atomic_id/variant, `parts`, phase, residency),
    the residency graph summary, and the device profile (caps + coeffs).
  - `run --metrics[=full]` — print TTFT, prompt-eval tok/s, generation tok/s,
    peak RSS (with `full`, per-phase timings).
  - `run --verbose` — human-readable trace of the phases (load/prefill/decode).
- **Out:** changing tuning/execution; new formats; changing default output.

## Design

- `inspect` reads the `.zplan` (+ `.zat`) and formats it (the same parse done
  ad-hoc during analysis). It must **not** change the plan.
- `--metrics` is gated so the normal `run` output is unchanged; `full` adds
  timing collected by the daemon (per phase). Timing is **diagnostic only** — it
  never feeds a decision, so the AOT invariant holds.
- These add CLI surface; explicitly requested (see Comments).

## Steps

1. `inspect`: plan dump (tensors / dispatches / variants / `parts` / phase /
   residency). [as-built: `cli/commands/inspect.zig`; default = plan section,
   `--plan`/`--image`/`--profile`/`--all` select sections]
2. `inspect`: profile dump (caps + coeffs from the `.zat`). [as-built:
   `calibrate.load` via the registry entry's `tune` digest; "no profile" when absent]
3. `run --metrics[=full]`: TTFT / **generation** / prefill / decode / RSS
   (+ per-phase for full). [as-built: `Session` times prefill/decode
   (`GenerateStats`); the daemon also wall-clocks the whole `GENERATE`
   (`gen_ms`, tokenize+prefill+decode+detokenize) and the reply carries
   `prefill_ms`/`decode_ms`/`decode_tokens`/`rss_kb`/`prefill_tokens`/`gen_ms`.
   `generation` prints just below `ttft`; `--profile` shows it as a phase.]
4. `run --verbose`: phase trace. [as-built: load/prefill/decode to stderr]
5. Tests: `inspect` on a synthetic plan; metrics parse. [as-built: inspect
   render tests + `parseHeader` old/extended tests]
6. `zig build` / `zig build test` green; help updated. [as-built: 157/157 tests;
   `inspect` registered + `run` help/usage updated]

## Definition of Done

- [x] `zlm inspect <alias>` shows the resolved selections (variant/`parts`/residency).
- [x] `zlm inspect` shows the device profile (caps + coeffs).
- [x] `run --metrics[=full]` reports TTFT / prefill / decode / RSS.
- [x] Normal `run` output is unchanged without the flags.
- [x] `zig build` / `zig build test` green (157/157).

## Arch Provenance

- **own:** the inspect/metrics surfaces.
- **consulted:** the earlier local prototype's `inspect` / `run --metrics` (reference for the fields).
- **derived:** plan format (`plan/`), calibration (`runtime/09`).

## Comments

- Explicitly requested; adds CLI surface. Timing is diagnostic only.
- Unblocks `runtime/11` (measure) and `kernels/07` (see selected variants).
- The `GENERATE` reply appends `prefill_ms\tdecode_ms\tdecode_tokens\trss_kb\tprefill_tokens`;
  `OK\t<len>` (old daemon) is still parsed. `prefill_tokens` is the extra field
  needed for prompt-eval tok/s.
- `inspect` imports `plan`/`kernels`/`tuner` at the edge (as `tune` already
  does); it is read-only formatting for an operator surface.

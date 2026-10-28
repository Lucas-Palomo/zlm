13 - Heavy-duty GPU Warm-up (sustained clock ramp)

Status: done
Commits:

## Goal

The RX 580's memory clock stays at 400 MHz under zlm's decode but ramps to 1750
MHz under a game (`gpu/06`): the Polaris DPM needs **sustained** memory activity
(`MCLK_ACTIVE_LEVEL=25`, `3D_FULL_SCREEN`) to ramp. `gpu/07` raised the decode's
duty cycle by offloading the fused `gate_up`, but the ramp is **not
deterministic** — when the decode's duty cycle is low (few offloads, a short
token) the clock never leaves 400 and the GPU decode runs ~4x slow (gemma-12B:
7.4 -> 1.7 tok/s). The tune-time calibration already warms the clock
(`calibrate.zig`: 32 resident GEMVs "so the device clock is ramped"), but the
**run** does not.

Add a **run-time heavy-duty warm-up**: when `--gpu`, the tuner records it in the
plan and the runtime runs a **sustained** GPU load at load, before the first
token, so the memory clock is ramped deterministically.

## Scope

- **In:** `Config.gpu_warmup` (a duration in ms, decided by the tuner when
  `--gpu`); the `.zplan` serialization (version bump); the runtime's warm-up
  (a sustained resident-GEMV load for the duration, then `sync`).
- **Out:** forcing the clock via sysfs (root); a device probe of the clock; the
  decode-time duty cycle (that is `gpu/07`).

## Design

### The plan records it (the tuner decides)

`Config.gpu_warmup: u16` = the warm-up duration in ms (0 = off). The tuner sets
it when `--gpu` (a discrete device whose clock ramps under load) — the same
condition that enables the offload. The runtime reads it and decides nothing.

### The runtime runs a **sustained** load

At `Session.load`, after `openResident`, if `gpu_warmup > 0`:

- Pick the **largest resident weight** (the biggest VRAM buffer — the most
  memory traffic per dispatch); if there are no residents, skip.
- Run a **continuous** stream of resident GEMVs against it for `gpu_warmup` ms:
  submit **asynchronously** and keep the ring full (submit while the GPU runs;
  drain only when the ring is full), so the memory bus stays busy the whole
  window — a **sustained** load, not N discrete bursts (a sync per dispatch
  leaves the bus idle between them and does not ramp the clock).
- `sync()` at the end.

This is what a game does to the GPU: a continuous, high-duty-cycle memory load.

## Steps

1. `plan/root.zig`: `Config.gpu_warmup` + serialize/deserialize + `VERSION`
   bump.
2. `tune.zig`: set `gpu_warmup` when `--gpu`.
3. `session.zig`: the sustained warm-up at load.
4. Verify: `zig build` / `zig build test` green; the gemma-12B GPU decode back to
   ~7.4 tok/s with the clock at 1750 (vs 1.7 at 400).

## Definition of Done

- [x] `--gpu` records the warm-up in the plan.
- [x] The runtime runs a **sustained** load (the ring stays full), not N bursts.
- [x] The gemma-12B GPU decode is deterministic (the clock ramped) — ~7.4 tok/s.
- [x] `zig build` / `zig build test` green.

## Arch Provenance

- **own:** the run-time heavy-duty warm-up.
- **derived:** `gpu/06` (the DPM/duty-cycle finding), `gpu/07` (the decode duty
  cycle), `calibrate.zig` (the tune-time warm-up).

## Comments

- **Sustained, not bursts.** The whole point: the DPM ramps on *sustained*
  activity; N sync GEMVs leave the bus idle between dispatches and do not ramp
  it. The warm-up must keep the GPU's queue non-empty (the async ring).
- **Why the tuner.** Whether the clock ramps is a device property the tuner
  already models (`--gpu` + a discrete device); the runtime applies the plan.

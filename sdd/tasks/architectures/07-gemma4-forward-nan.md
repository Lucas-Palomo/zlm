07 - gemma4 Forward NaN (attention / KV)

Status: done
Commits: 193a74d

## Goal

The gemma-4 E2B decode produced **all-NaN logits** (`nan=262144/262144`) → the
argmax was index 0 → empty/degenerate output. Find the source of the NaN and fix
it, so the gemma-4 variants generate coherent text.

The model files are fine: `llama.cpp` generates coherent text for the same
`gemma-4-E2B-it-unsloth-Q4_0.gguf` (and E4B). The defect was in zlm's forward.

## Root cause

The FFN `gelu` activation (`activation` AtomicOp `gelu` → `geluInPlace`)
computed `tanh` as `(e^{2x} - 1) / (e^{2x} + 1)` **without clamping**. For large
`inner = 0.79788456*(v + 0.044715*v^3)` (a handful of large FFN values, `|v| ≳
5`), `e^{2x}` overflows to `inf` → `inf/inf = NaN`. The NaN then propagated
through the residual stream → all-NaN logits.

`gated_gelu` already clamped its `inner` to ±15 before the `@exp`; `geluInPlace`
and `softcapInPlace` (which share `tanhV`) did not. The scalar tails use
`std.math.tanh` (stable), so only the vector path was affected.

## Scope

- **In:** the gemma-4 forward data path — the `activation` AtomicOps (`tanhV`,
  `geluInPlace`, `softcapInPlace`).
- **Out:** the model files; the chat template (separate, `open-items`).

## Design

Clamp `tanhV`'s input to ±15: `tanh` saturates to ±1 to f32 precision for
`|x| > 15`, so the clamp is exact and removes the overflow for every caller
(`geluInPlace`, `softcapInPlace`, and any future `tanhV` user). `gated_gelu`
keeps its own clamp (unchanged).

## Steps

1. Instrument the forward (per dispatch, prefill + decode) to localize the first
   NaN. **[DONE]** — logits all NaN; first NaN at the `activation` op
   (`gelu`, `out_nan=14/4096`); the KV/attention NaN was downstream.
2. Identify the root cause. **[DONE]** — `tanhV` overflow in `geluInPlace`
   (`e^{2x}=inf` → `inf/inf=NaN`).
3. Implement the smallest correct fix. **[DONE]** — clamp `tanhV` to ±15.
4. Verify. **[DONE]** — `zig build`/`test` 216/216; E2B `run` → "The capital of
   France is Paris."; E4B `run` → "Paris"; 0 NaN.

## Definition of Done

- [x] The NaN source is identified with evidence.
- [x] The fix is implemented and generic (no per-ISA hack).
- [x] `zig build` / `zig build test` green (216/216).
- [x] E2B/E4B `run` (greedy) produce coherent text.

## Arch Provenance

- **own:** the localization and the fix.
- **consulted:** `llama.cpp` gemma-4 forward (reference only).
- **derived:** `architectures/04` (gemma4 forward correctness), `runtime/11`.

## Comments

- **Pre-existing.** The pre-session binary (`0274d50`) reproduced the same
  degenerate output, so this was not caused by the recent open-items work.
- **The chat template** (`<start_of_turn>` vs the model's `<|turn>`) is a
  separate, non-fatal mismatch: the E2B still generated coherent text with the
  current template. Tracked in `sdd/open-items.md`.

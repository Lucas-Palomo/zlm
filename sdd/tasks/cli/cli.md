# cli

title: CLI
reason: The terminal edge. Owns argv parsing, the command registry/help, and the
        per-command option surface. Originally spread across `bootstrap/01` and
        `tune/01`; this milestone gathers the CLI as its own concern and improves
        it (declarative options) so commands stop re-implementing argv handling.

## Tasks

### [01 - CLI Bootstrap](01-cli-bootstrap.md)
Status: done

Context: the command surface and mold (`Args`, `Command`/`Context`, registry +
`walk`, generated/path-aware help, the `image` group). Originally delivered as
`bootstrap/01 - CLI Skeleton`; recorded here as the CLI milestone's baseline.

### [02 - Tune Skeleton](02-tune-skeleton.md)
Status: done

Context: the first command with real options (`tune`): flag parsing, validation
(`--threads`/`--ram` required, `--gpu` requires `--vram`), and the runtime store
wiring. Delivered with `tune/01 - Tune Bootstrap`; flagged here because the CLI
option handling it hand-rolled is what task `03` replaces.

### [03 - Improve Flags](03-improve-flags.md)
Status: done

Context: a reusable declarative option layer (`options.zig`: `Spec` + `Parsed` +
`parse`) so commands declare options instead of `while/else if`. Accepts
`--x=v`, `--x v`, `-x v`; validates required/unknown/missing uniformly; help can
be derived from the spec.

### [04 - Inspect and Metrics](04-inspect-and-metrics.md)
Status: done

Context: add `zlm inspect <alias>` (the resolved plan: variants/atomic_ids/parts/
phase/residency + the device profile) and `run --metrics[=full]` / `--verbose`
(TTFT/prefill/decode/RSS), so selection and performance are observable. The earlier local prototype has
both; the current code has neither.

### [05 - Execution Profile (`run --profile`)](05-profile.md)
Status: done

Context: extend the `--metrics` phase timing with a per-AtomicOp breakdown of the
decode (matvec/matmul/attention/norm/other + dispatch/residency/sync) + hotspots,
via `run --profile[=all|decode|prefill|load]`. Diagnostic only (never feeds a
decision); the tool to find the decode bottleneck after `kernels/11`/`12` closed
the compute levers.

### [06 - Streaming Generation (`run --stream`)](06-stream.md)
Status: done

### [07 - Think Suppression (`--show-think`) and Verbose Levels](07-verbose-and-show-think.md)
Status: done

Context: hide the reasoning block by default (`--show-think` opts in; batch and
stream); turn `--verbose` into levels (`min` phase + per-layer, `info` +
per-token logits, `all` + prompt ids + think block) delivered as `TRACE` frames to
stderr. Output only; the plan is untouched.

### [08 - Prefetch Threads (`--prefetch-threads`)](08-prefetch-threads.md)
Status: done

Context: separate the thread budget explicitly — `--threads N` (N ≥ 1) is the
compute pool (never reduced) and `--prefetch-threads M` (M ≥ 1; absent = 0) adds
M dedicated residency-prefetch workers, separate from and additional to the
compute (the process runs N + M). The plan records (N, M); the runtime runs **M**
prefetch workers in a pool (today one). Replaces `cost.decidePrefetch` with an
explicit opt-in; measured on the disk-bound 72B: 9 + 2 and 9 + 1 are marginally
ahead of 9 + 0, within noise.

Context: print tokens as they are generated. A `stream` field on `GENERATE`; the
daemon emits `DATA` frames while decoding (per-token detokenize, UTF-8-safe
chunking, special/control filtered) and the final metrics header; the client
writes them live. Presentation only — the plan/KV/prefix cache are unchanged and
batch stays the default.

### [09 - Residency Ceiling (`--residency`)](09-residency.md)
Status: done

Context: give the resident weight working set an explicit, opt-in ceiling
(`tune --residency <mb|gb>`). Absent = no residency (every weight streams from
the mmap; the tuner still reserves the weight ceiling so the freed RAM backs the
page cache). `--residency` and `--prefetch-threads` are mutually required; a
value above the KV-floor ceiling is clamped with a warning. The runtime's
default `warm` state is bounded by the plan's window, removing the unbounded
`WILLNEED` that OOM'd the 72B.

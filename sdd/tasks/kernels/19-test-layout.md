19 - Test Layout

Status: done
Commits: b8ba00a

## Goal

Adopt a test-layout pattern so the production modules hold **no** tests or
fixtures — the inline `test` blocks and the test-only helpers (the oracles, the
`fill*`/`split*` fixtures) move out of the production files. This keeps each
module's code focused and makes the test surface explicit.

Apply it to `kernels/linear/matvec/` first, then the rest.

## The pattern

| file | role |
|---|---|
| `<mod>_test.zig` | a module's **tests + its private fixtures** (the `<mod>` holds only production) |
| `test.zig` | the **directory aggregator**: imports the `<mod>_test.zig` files + the **cross-module** tests |
| `fixtures.zig` | the **shared** test-only helpers (the fixtures/oracles used by more than one `<mod>_test.zig`) |

Rules:
- **One module, one `<mod>_test.zig` sibling.**
- A test that crosses modules (e.g. "every generated candidate matches
  production", the catalog test) goes in the directory's `test.zig`, not in a
  `<mod>_test.zig`.
- The `<mod>_test.zig` files are **private** (not API; never imported by
  production code or from outside the directory).
- **One aggregator per directory** (not one per submodule).

## The matvec layout

```text
matvec/
  root.zig            # facade (+ a `test { _ = @import("test.zig"); }`)
  layout.zig          layout_test.zig
  types.zig           types_test.zig
  kernels/
    quantized.zig     quantized_test.zig
    float.zig         float_test.zig
    bfloat.zig        bfloat_test.zig
  form/
    single.zig        single_test.zig
    fused.zig         fused_test.zig
    batched.zig       batched_test.zig
  test.zig            # the aggregator + the cross-module tests
  fixtures.zig        # the shared test-only helpers (the oracles, fill*/split*)
```

## Design

- **Discovery.** Zig runs the tests of the files that are *referenced*. A
  `test { _ = @import("test.zig"); }` in the module root (or `main.zig` pulls the
  directory aggregator) guarantees `zig build test` covers everything.
- **Fixtures.** A fixture used by one module lives in its `<mod>_test.zig`; one
  used by several goes to `fixtures.zig` (importable, test-only). The oracles
  (`q4_0IntOracle`, `q4_KIntOracle`) are shared → `fixtures.zig`.
- **No behaviour change**: this is a move, not a rewrite. `zig build test` stays
  green with the same count.

## Steps

1. **Done.** `<mod>_test.zig` per module with tests: `layout_test.zig`,
   `kernels/quantized_test.zig`, `form/{single,fused,batched}_test.zig`.
2. **Done.** `fixtures.zig` — the shared helpers (the oracles, `fill*`/`split*`,
   `expect*`).
3. **Done.** `test.zig` — the aggregator + the 8 cross-module tests (the
   generated-candidate parity for q4_0/q4_1/q4_k/q8_0/q6_k, the catalog, the gpu
   catalog, the parts-invariance).
4. **Done.** The discovery lives in `main.zig` (`_ = @import(".../matvec/test.zig")`),
   so `root.zig` (a production file) stays `test`-free.
5. **Done.** `zig build test` 270/270; every production matvec file has `rg
   "^test "` → 0; `zig fmt` clean.

## Definition of Done

- [x] The matvec production modules hold **no** `test` block.
- [x] `<mod>_test.zig` per module; one `test.zig` aggregator; the shared fixtures
      in `fixtures.zig`.
- [x] `zig build test` green (270/270); no behaviour change.

## Arch Provenance

- **own:** the test layout.
- **derived:** `kernels/18` (the matvec structure).

## Comments

- **One aggregator per directory** — not per submodule.
- The `<mod>_test.zig` files are private; the shared fixtures go to
  `fixtures.zig` so no `_test.zig` imports another.
- Apply to the other milestones (gpu, runtime, ...) after the matvec proves it.
- **As-built.** The matvec is the first application: `fixtures.zig` (the shared),
  `test.zig` (the aggregator + the 8 cross-module tests) and the per-module
  `_test.zig` files. The discovery is in `main.zig` (not a `test` block in
  `root.zig`) so the production files stay test-free. A few private symbols went
  `pub` (`kernelGgmlType`/`planApplicable`/`q4_0Expand`) so the sibling tests can
  reach them — additive, no behaviour change. `types.zig`/`float.zig`/
  `bfloat.zig` have no tests, so no placeholder `_test.zig`.

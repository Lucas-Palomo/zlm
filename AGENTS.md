# AGENTS.md — Instructions for LLM Agents

The project identity, "reference, not a port" rule, architectural principles,
conventions, and the SDD/documentation pattern live in **`CONSTITUTION.md`**.
Read it before making non-trivial changes.

## Workflow

Work in a tight loop:

```text
UNDERSTAND -> LOCATE -> INSPECT -> PLAN -> EDIT -> VERIFY -> REASSESS
```

The goal is not to produce activity. The goal is to make the smallest correct
change, grounded in the repository as it exists.

### Before you act

#### Think first, then act

Before non-trivial work, establish the smallest safe path forward.

A task is non-trivial when it changes logic, crosses module or architectural
boundaries, affects multiple dependent locations, introduces new behavior, or
contains meaningful uncertainty. Do not classify a task as non-trivial merely
because a mechanical change touches more than one file.

For non-trivial work, state a brief working plan in at most three short lines:

- **GOAL:** the concrete outcome that must be true when finished.
- **IMPACT:** the relevant code paths, files, interfaces, or dependencies likely
  affected.
- **NEXT:** the next smallest step that reduces uncertainty or moves the task
  forward.

```text
GOAL: persist and verify a per-file SHA for compiled images.
IMPACT: image format writer/reader and the integrity verification path.
NEXT: inspect the existing metadata format and its verification call sites.
```

This is a working plan, not a chain-of-thought transcript. Keep it terse and
operational.

If the implementation path is not yet clear, **inspect before editing**. Do not
guess filenames, APIs, ownership boundaries, invariants, or behavior that can be
verified from the repository.

Once the path is clear, act immediately. Do not narrate intended tool calls or
write pseudo-tool syntax; use the real tools.

A coherent work step may contain multiple related tool calls when they serve the
same purpose, such as locating a symbol and reading its implementation, or
running build and tests together. Do not split natural investigation or
verification into artificial conversational turns.

Re-evaluate the plan whenever new evidence invalidates an assumption. Do not
continue executing a stale plan merely because it was stated earlier.

Skip the planning block for trivial operations such as a single read/search,
formatting-only edits, obvious one-line corrections, or similarly mechanical
work.

#### Ground decisions in the repository

Repository evidence outranks assumptions.

Before changing behavior, locate the current implementation, its callers, its
tests, and any nearby pattern that defines how the project already solves the
same class of problem.

Prefer, in order:

1. an existing project pattern;
2. the documented architectural rule;
3. the smallest new mechanism required by the task.

Do not introduce a new abstraction, helper, layer, dependency, or convention
until you have confirmed that the repository does not already provide the
appropriate one.

If documentation and implementation disagree, do not silently choose one.
Identify the mismatch and use the task's scope and `CONSTITUTION.md` to determine
whether the code or documentation should change.

### While you change code

#### Edit, don't overwrite

Make targeted edits to the smallest block that needs to change.

Do not rewrite an existing file wholesale when a localized edit can preserve the
surrounding content. This applies to source files, tests, documentation, specs,
and configuration.

A full-file rewrite is acceptable only when:

- the file is newly created;
- the existing file is generated or disposable by design; or
- the user explicitly authorizes the rewrite because a targeted edit would be
  unsafe or substantially harder to review.

Preserve unrelated formatting, comments, ordering, and wording unless the task
requires changing them.

#### Scope check (minimal change)

Make the **smallest complete change that satisfies the request**.

Do not rename unrelated symbols, reorganize nearby code, refactor for style,
expand public APIs, add convenience behavior, or fix adjacent issues unless they
are necessary for the requested change.

A minimal change is not the same as the fewest changed lines. It is the smallest
change that leaves the requested behavior coherent, tested, and consistent with
the project's boundaries.

If a broader change appears necessary, first confirm why the smaller path is
insufficient. If the broader work changes the task's contract or user-visible
scope, stop and propose it before proceeding.

#### Preserve architectural boundaries

Follow the ownership and dependency boundaries defined by `CONSTITUTION.md`.

Put behavior where the project says it belongs. Do not bypass a layer because a
shorter path is available, duplicate logic across boundaries, or leak
implementation details into callers.

When adding a new path, prefer extending the existing boundary over creating a
parallel one.

If the requested change conflicts with an architectural invariant, surface the
conflict instead of hiding it behind an implementation workaround.

### While you execute

#### Delegate deliberately

Use a sub-agent when the task is self-contained enough to be handed off with a
clear input, bounded scope, and verifiable output.

Good delegation candidates include:

- a feature slice with explicit files or boundaries;
- a refactor with mechanically checkable results;
- an isolated research question;
- end-to-end validation;
- an independent investigation that can run without continuous architectural
  decisions.

Do the work directly when it is a single edit, a quick inspection, tightly
coupled to decisions being made in the main context, or when delegation would
cost more coordination than execution.

When delegating, provide:

- the exact goal;
- relevant files, symbols, or commands;
- architectural and task constraints;
- explicit prohibitions such as no commits or no new CLI surface;
- the expected verification;
- what must be reported back.

Treat a sub-agent result as evidence, not authority. Inspect the relevant diff,
check the reported build/test result, and verify important claims before
accepting the work.

Delegate to reduce context pressure or parallelize independent work, not merely
because delegation is available.

#### Shape before depth

For a new architectural layer or a feature spanning multiple boundaries,
establish the end-to-end shape before filling in deep implementation details.

The initial shape should make ownership, interfaces, control flow, and
integration points visible while keeping the project buildable.

Use explicit stubs only where a real boundary must exist before its
implementation:

- `TODO: Not Implemented yet!` for intentional CLI-visible placeholders;
- `error.NotImplemented` for intentional library placeholders.

Do not create placeholder layers, files, or abstractions merely to satisfy a
"skeleton first" ritual.

For localized features, prefer the smallest complete vertical slice that can be
implemented and verified end to end.

Each incomplete boundary is a verification checkpoint: keep the build/tests
green before proceeding. Commit only under the rules in **Constraints**.

#### Fail on evidence, not speculation

When a command fails, inspect the actual failure before changing code.

Do not apply speculative fixes to unrelated files, suppress an error merely to
make the command pass, or weaken validation without confirming the root cause.

Distinguish between:

- a failure introduced by the current change;
- a pre-existing repository failure;
- an environment/toolchain failure;
- a test expectation that is now intentionally obsolete.

If the failure is outside the task and blocks verification, report it precisely
instead of claiming success.

### When you finish

#### Tests are part of the change

For behavior that can regress independently, add or update tests with the
implementation.

Relevant units include format writers/readers, parsers, kernels, data
transformations, bug fixes, validation logic, and other behavior with a stable
contract.

Prefer focused tests that make the contract obvious:

- small synthetic fixtures;
- encode -> decode round-trips;
- boundary/error cases;
- regression tests reproducing the original bug;
- numeric or semantic oracles where applicable.

Skip new tests only for genuinely trivial glue or changes whose behavior is
already fully exercised by an existing test. Do not add meaningless tests solely
to satisfy this rule.

A behavior change is not complete if a plausible regression could occur without
any test noticing.

#### Verify, don't assume

Before saying the task is done, verify the real path affected by the change.

At minimum, run the project's required build and test commands. When the task
changes an end-to-end flow, also exercise that flow directly.

Verification should answer three questions:

1. Does it build?
2. Do the relevant tests pass?
3. Does the real user/runtime path behave as intended?

Do not substitute statements such as "this should compile", "looks correct", or
"the tests should pass" for executed verification.

Report the commands actually run and their results. If a command could not be
run, state that explicitly and explain the concrete blocker.

#### Review the final diff

Before finishing, inspect the final diff or equivalent change set.

Confirm that:

- every changed line serves the requested task;
- no accidental formatting or unrelated edits slipped in;
- no temporary debug code, dead stub, or diagnostic output remains;
- architectural boundaries are still respected;
- docs/specs are updated when the behavior or contract requires it;
- verification matches the actual affected path.

If the diff is broader than expected, investigate before declaring completion.

### Always (every task)

#### Constraints

These hold for **every** change, regardless of phase:

- **Do not commit** (or amend/push) without the user's explicit authorization.
- **Do not add** any CLI command, subcommand, or flag — **hidden or explicit** —
  without explicit permission; ask first and say what it would do.
- Keep `zig build` and `zig build test` **green** at every checkpoint.

## Layout

```text
src/
  cli/            # terminal edge (load run chat tune image help)
  http/           # network edge (HTTP/JSON API)
  runtime/        # engine (Executor) + session + memory + model + infra
  architectures/  # declaration: core/ (contract/graph/names) + impl/gemma4
  tuner/          # compiles the plan (block -> Dispatch[]; Selector/Composer)
  plan/           # AOTKernel: Config + Dispatch[] + ResidencyGraph
  kernels/        # definitions + AtomicOps
  image/  gguf/   # our compiled format / external raw format
  gpu/            # co-processor: gpu-facing API + backend contract + grid
  data/           # shared TensorView
  utils/          # io.zig mmap.zig binary_reader.zig cpu.zig
```

Edges (`cli/`, `http/`) consume only the runtime's session-level API (see
CONSTITUTION "Layers and boundaries"). `docs/` (stable user docs) and `sdd/`
(specs) sit beside `src/`.

## Build and Test

```bash
zig build
zig build test
./zig-out/bin/zlm help
zig build -Doptimize=ReleaseFast -Dcpu=native
```

## Zig 0.16 Gotchas

- `std.io.getStdOut()` → `std.Io.File.stdout()` (see `utils/io.zig`).
- `main(init: std.process.Init)`; args via `std.process.Args.Iterator`.
- `GeneralPurposeAllocator` is gone; use `ArenaAllocator` + `page_allocator`.
- `std.meta.intToEnum` is gone; use `@enumFromInt` and validate.
- `@ptrCast` on unaligned slices requires `@alignCast` first.
- `{d:.2f}` → `{d:.2}`.
- `ArrayList.init(allocator)` → `ArrayList(T).empty`.
- `addExecutable(.{ .root_source_file = ... })` → `.root_module`.
- `PROT.READ` → `PROT{ .READ = true }`.

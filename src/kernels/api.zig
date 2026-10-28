const std = @import("std");

/// The kernel layer: AtomicOps (routines) + a catalog. It has no `Dispatch`
/// (that is the plan's) and no global args union (args are per family).
pub const KernelError = error{
    Unsupported,
    UnsupportedType,
    ShapeMismatch,
    TruncatedData,
    OutOfMemory,
};

/// Kernel family (op). Distinct from the *semantic* op in `model/graph.zig`;
/// the scheduler maps one to the other.
pub const Op = enum {
    dequant,
    matvec,
    norm,
    rope,
    attention,
    activation,
    elementwise,
    matmul,
};

/// A family's surface: its op, name and the type-erased `run` the registry
/// calls. Each family keeps a typed `AtomicId` enum and `run(id, args)` and
/// wires them through `erased`.
pub const Family = struct {
    op: Op,
    name: []const u8,
    /// Scheduling metadata the Scheduler reads: is this op's work independent
    /// per output row?
    row_parallel: bool = false,
    /// Candidate catalog: one `Kernel` per selectable `AtomicOp` variant of
    /// this family. The Selector/Scheduler enumerate these without knowing the
    /// family.
    kernels: []const Kernel = &.{},
    /// Execute one AtomicOp. `atomic_id` is decoded by the family (via
    /// `erased`); `args` points at the family's own args struct. The registry
    /// junction is dynamic, so it carries `anyerror`; each family keeps its own
    /// (possibly wider) error set.
    run: *const fn (atomic_id: u16, args: *const anyopaque) anyerror!void,
};

/// Adapt a family's typed `run(id, args)` to the registry's type-erased
/// signature. Validates `atomic_id` against `IdT`; the family's own error set
/// flows through.
pub fn erased(
    comptime IdT: type,
    comptime ArgsT: type,
    comptime runFn: anytype,
) *const fn (u16, *const anyopaque) anyerror!void {
    const S = struct {
        fn run(atomic_id: u16, args: *const anyopaque) anyerror!void {
            const id = std.enums.fromInt(IdT, atomic_id) orelse return error.Unsupported;
            const typed: *const ArgsT = @ptrCast(@alignCast(args));
            return runFn(id, typed);
        }
    };
    return &S.run;
}

/// Execution tier of a candidate.
pub const Tier = enum(u8) { cpu, gpu };

/// How a candidate's cost scales (passes / memory / flops).
pub const Complexity = enum(u8) { linear, quad_mem, quad_flops };

/// A candidate's cost inputs for a shape, consumed by the analytical roofline
/// (`tuner.cost.opCost`). Data only: the family fills these at assembly.
pub const Cost = struct {
    /// Bytes the candidate moves for the shape (DRAM-equivalent).
    bytes: u64 = 0,
    /// Arithmetic work (mul+add counted as 2).
    flops: u64 = 0,
};

/// When a candidate applies: `code` is the raw ggml type code, `m`/`n` are the
/// family's shape dims and `lanes` the SIMD width. `null` means always.
pub const ApplicableFn = *const fn (code: u32, m: u64, n: u64, lanes: u16) bool;

/// A candidate's cost for a shape: `m`/`n`/`k` are family dims and `lanes` the
/// SIMD width. `null` means the family default (zero) cost.
pub const CostFn = *const fn (code: u32, m: u64, n: u64, k: u64, lanes: u16) Cost;

/// Catalog entry (a kernel definition). Used by the Selector.
pub const Kernel = struct {
    id: []const u8,
    op: Op,
    atomic_id: u16,
    row_parallel: bool = false,
    fusable: bool = false,
    scratch: u64 = 0,
    tier: Tier = .cpu,
    complexity: Complexity = .linear,
    /// The family's sub-variant code (e.g. `MatVecPlan`); `0` when the family
    /// has no sub-variants.
    variant: u16 = 0,
    /// When the candidate applies (the family's ISA×shape rule), as data.
    /// `null` = always applicable.
    applicable: ?ApplicableFn = null,
    /// The candidate's cost inputs for a shape. `null` = family default.
    cost: ?CostFn = null,
};

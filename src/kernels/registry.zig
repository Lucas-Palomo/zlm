const api = @import("api.zig");
const recipes_mod = @import("recipes.zig");
const quant = @import("linear/quant/root.zig");
const activation = @import("transformer/activation.zig");
const norm = @import("transformer/norm.zig");
const rope = @import("transformer/rope.zig");
const attention = @import("transformer/attention.zig");
const elementwise = @import("transformer/elementwise.zig");
const matvec_mod = @import("linear/matvec/root.zig");
const matmul_mod = @import("linear/matmul/root.zig");

pub const Op = api.Op;
pub const KernelError = api.KernelError;
pub const Family = api.Family;
pub const Kernel = api.Kernel;
pub const Tier = api.Tier;
pub const Complexity = api.Complexity;
pub const Cost = api.Cost;
pub const ApplicableFn = api.ApplicableFn;
pub const CostFn = api.CostFn;
pub const Recipe = recipes_mod.Recipe;
pub const Form = recipes_mod.Form;

/// Aggregated catalog the Tuner consumes: one `Family` per op.
pub const sets = [_]*const Family{
    &quant.set,
    &activation.set,
    &norm.set,
    &rope.set,
    &attention.set,
    &elementwise.set,
    &matvec_mod.set,
    &matmul_mod.set,
};

/// Every family's candidate kernels, concatenated (comptime).
const all_kernels = blk: {
    var total: usize = 0;
    for (sets) |s| total += s.kernels.len;
    var arr: [total]Kernel = undefined;
    var i: usize = 0;
    for (sets) |s| {
        for (s.kernels) |k| {
            arr[i] = k;
            i += 1;
        }
    }
    break :blk arr;
};

/// The aggregated candidate catalog: every `AtomicOp` of every family.
pub fn kernels() []const Kernel {
    return &all_kernels;
}

/// The candidate catalog of one family.
pub fn kernelsFor(op: Op) []const Kernel {
    const s = find(op) orelse return &.{};
    return s.kernels;
}

/// The declared recipe catalog (data only).
pub fn recipes() []const Recipe {
    return &recipes_mod.catalog;
}

pub fn find(op: Op) ?*const Family {
    for (sets) |s| {
        if (s.op == op) return s;
    }
    return null;
}

/// Scheduling metadata: is this op's work independent per output row?
pub fn rowParallel(op: Op) bool {
    const f = find(op) orelse return false;
    return f.row_parallel;
}

/// Execute one AtomicOp of `op`. `args` points at the family's own args struct;
/// `atomic_id` is the family's `AtomicId` as a number. Errors flow from the
/// family.
pub fn run(op: Op, atomic_id: u16, args: *const anyopaque) anyerror!void {
    const s = find(op) orelse return error.Unsupported;
    return s.run(atomic_id, args);
}

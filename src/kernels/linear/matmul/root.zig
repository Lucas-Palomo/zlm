const std = @import("std");
const TensorView = @import("../../../data/tensor_view.zig").TensorView;
const dequant = @import("../quant/root.zig");
const vector = @import("../vector.zig");
const matvec = @import("../matvec/root.zig");
const ggml = @import("../../../gguf/format.zig");

pub const KernelError = matvec.KernelError;

/// AtomicOps of the matmul (GEMM) family. `rows` is the naive per-row GEMV
/// composition; `blocked` dequantizes a weight row once and reuses it across
/// the `M` input rows.
pub const AtomicId = enum(u16) { rows, blocked };

/// Alias kept for the Selector's variant catalog naming (`MatMulPlan`).
pub const MatMulPlan = AtomicId;

/// `x` is `M x K` f32, `out` is `M x N` f32, the weight is `[K, N]`.
pub const Args = struct {
    view: *const TensorView,
    x: []const f32,
    out: []f32,
    m: usize,
    n: usize,
    k: usize,
    begin: usize = 0,
    end: usize = 0,
    scratch: []f32 = &.{},
};

fn validate(args: *const Args) KernelError!struct { k: usize, n: usize } {
    const view = args.view;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const k: usize = @intCast(view.dims[0]);
    const n: usize = @intCast(view.dims[1]);
    if (args.k != k or args.n != n) return error.ShapeMismatch;
    if (args.x.len != args.m * k) return error.ShapeMismatch;
    if (args.out.len != args.m * n) return error.ShapeMismatch;
    if (args.begin > args.end or args.end > n) return error.InvalidRange;
    return .{ .k = k, .n = n };
}

/// Naive GEMM: one existing per-row GEMV (`matVecRows`) per input row. Keeps
/// every layout the matvec family supports.
pub fn runRows(args: *const Args) KernelError!void {
    const dims = try validate(args);
    var tok: usize = 0;
    while (tok < args.m) : (tok += 1) {
        try matvec.matVecRows(
            args.view,
            args.x[tok * dims.k ..][0..dims.k],
            args.out[tok * dims.n ..][0..dims.n],
            args.begin,
            args.end,
        );
    }
}

/// Blocked GEMM: dequantize each weight row once into `scratch` and reuse it
/// for all `M` input rows. Native layouts only; transformed layouts fall back
/// to `runRows`.
pub fn runBlocked(args: *const Args) KernelError!void {
    const dims = try validate(args);
    if (!args.view.layout.isNative()) return runRows(args);
    if (args.m == 0) return;
    if (args.scratch.len < dims.k) return error.ShapeMismatch;
    const data = args.view.mapped_slice orelse return error.NoMappedData;
    const row_bytes: usize = @intCast(ggml.tensorByteLength(args.view.ggml_type, dims.k) catch return error.UnsupportedType);
    if (data.len < dims.n * row_bytes) return error.TruncatedData;

    const row = args.scratch[0..dims.k];
    var i = args.begin;
    while (i < args.end) : (i += 1) {
        const raw = data[i * row_bytes ..][0..row_bytes];
        try dequant.dequantTo(args.view.ggml_type, raw, row);
        var tok: usize = 0;
        while (tok < args.m) : (tok += 1) {
            args.out[tok * dims.n + i] = vector.dot(row, args.x[tok * dims.k ..][0..dims.k]);
        }
    }
}

fn familyRun(id: AtomicId, args: *const Args) KernelError!void {
    switch (id) {
        .rows => return runRows(args),
        .blocked => return runBlocked(args),
    }
}

/// Quant types whose rows `dequantTo` can expand for the blocked variant.
fn hasDequantCodec(t: ggml.GGMLType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q8_0, .q4_0, .q4_1, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

/// The blocked variant dequantizes a weight row once and reuses it across the
/// `M` input rows: it applies to a batched, wide, dequantizing, >=256-bit ISA
/// shape. The naive per-row variant is its complement.
fn blockedApplicable(code: u32, m: u64, n: u64, lanes: u16) bool {
    const t = ggml.GGMLType.fromInt(code) orelse return false;
    return m >= 2 and n >= matvec.tile_min_rows and lanes >= 8 and hasDequantCodec(t);
}

fn rowsApplicable(code: u32, m: u64, n: u64, lanes: u16) bool {
    return !blockedApplicable(code, m, n, lanes);
}

/// Roofline inputs for either variant: the weight bytes and `2*k*n*m` flops.
/// The variant changes how those bytes are moved, not the logical work.
fn matmulCost(code: u32, m: u64, n: u64, k: u64, lanes: u16) api.Cost {
    _ = lanes;
    const t = ggml.GGMLType.fromInt(code) orelse return .{};
    const elems = std.math.mul(u64, k, n) catch return .{ .flops = std.math.maxInt(u64) };
    const bytes = ggml.tensorByteLength(t, elems) catch 0;
    const total = std.math.mul(u64, elems, m) catch std.math.maxInt(u64);
    const flops = std.math.mul(u64, total, 2) catch std.math.maxInt(u64);
    return .{ .bytes = bytes, .flops = flops };
}

const plan_kernels = [_]api.Kernel{
    .{
        .id = "matmul_rows",
        .op = .matmul,
        .atomic_id = @intFromEnum(AtomicId.rows),
        .row_parallel = true,
        .complexity = .quad_flops,
        .variant = @intFromEnum(AtomicId.rows),
        .applicable = &rowsApplicable,
        .cost = &matmulCost,
    },
    .{
        .id = "matmul_blocked",
        .op = .matmul,
        .atomic_id = @intFromEnum(AtomicId.blocked),
        .row_parallel = true,
        .complexity = .quad_flops,
        .variant = @intFromEnum(AtomicId.blocked),
        .applicable = &blockedApplicable,
        .cost = &matmulCost,
    },
};

pub const set = api.Family{
    .op = .matmul,
    .name = "matmul",
    .row_parallel = true,
    .kernels = &plan_kernels,
    .run = api.erased(AtomicId, Args, familyRun),
};

const api = @import("../../api.zig");

const std = @import("std");
const TensorView = @import("../../../data/tensor_view.zig").TensorView;
const dequant = @import("../quant/root.zig");
const dot = @import("../vector.zig").dot;
const ggml = @import("../../../gguf/format.zig");
const activation = @import("../../transformer/activation.zig");
const api = @import("../../api.zig");
const simd = @import("../../simd.zig");

pub const types = @import("types.zig");
pub const layout = @import("layout.zig");
pub const quantized = @import("kernels/quantized.zig");
pub const float = @import("kernels/float.zig");
pub const bfloat = @import("kernels/bfloat.zig");
pub const single = @import("form/single.zig");
pub const batched = @import("form/batched.zig");
pub const fused = @import("form/fused.zig");

pub const ActivationKind = types.ActivationKind;
pub const max_activation_cache_blocks = types.max_activation_cache_blocks;
pub const KernelError = types.KernelError;
pub const MatVecPlan = types.MatVecPlan;
pub const parseMatVecPlan = types.parseMatVecPlan;
pub const planLabel = types.planLabel;
pub const Variant = types.Variant;
pub const q4_0_candidates = types.q4_0_candidates;
pub const q4_1_candidates = types.q4_1_candidates;
pub const q4_k_candidates = types.q4_k_candidates;
pub const q8_0_candidates = types.q8_0_candidates;
pub const q6_k_candidates = types.q6_k_candidates;
pub const f32_candidates = types.f32_candidates;
pub const f16_candidates = types.f16_candidates;
pub const bf16_candidates = types.bf16_candidates;
pub const candidates = types.candidates;
pub const Candidate = types.Candidate;
pub const isPrefetchPlan = types.isPrefetchPlan;
pub const q4_1ScaleSplitPlan = types.q4_1ScaleSplitPlan;

pub const xBlockSum = layout.xBlockSum;

pub const fusedGateUpRows = fused.fusedGateUpRows;
pub const fusedMlpRows = fused.fusedMlpRows;
pub const normThenMatVecRows = fused.normThenMatVecRows;

pub const matVec = single.matVec;
pub const matVecRows = single.matVecRows;
pub const matVecWithBiasRows = single.matVecWithBiasRows;
pub const matVecRowsWithPlan = single.matVecRowsWithPlan;
pub const matMul = single.matMul;
pub const matMulRows = single.matMulRows;
pub const matMulWithBiasRows = single.matMulWithBiasRows;
pub const lmHeadRows = single.lmHeadRows;

pub fn matVecReference(view: *const TensorView, allocator: std.mem.Allocator, x: []const f32, out: []f32) !void {
    const data = view.mapped_slice orelse return error.NoMappedData;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const m: usize = @intCast(view.dims[1]);
    if (x.len != n or out.len != m) return error.ShapeMismatch;

    const flat = try allocator.alloc(f32, n * m);
    defer allocator.free(flat);
    try dequant.dequantTo(view.ggml_type, data, flat);

    for (0..m) |i| {
        out[i] = dot(flat[i * n .. (i + 1) * n], x);
    }
}

// --- family surface ---

pub const AtomicId = enum(u16) { mat_vec, mat_mul, lm_head, gate_up_fused, mlp_fused, norm_proj, matvec_bias };

/// Row count used by the matmul family to gate its row tiling. Matvec's own
/// variant choice is now type/validity only and ranked by the variant-aware
/// cost, so this is retained only for the GEMM applicability rule.
pub const tile_min_rows: u64 = 1024;

/// Candidates the projection Selector must never pick (non-plan atomics of the
/// family, e.g. the fused recipes).
fn neverApplicable(_: u32, _: u64, _: u64, _: u16) bool {
    return false;
}

fn isPlanType(t: ggml.GGMLType) bool {
    return switch (t) {
        .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 => true,
        else => false,
    };
}

/// The ggml type a catalog kernel tag decodes.
pub fn kernelGgmlType(comptime tag: types.KernelTag) ggml.GGMLType {
    return switch (tag) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q4_k => .q4_k,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

/// Validity rule for one `MatVecPlan`, as a pure predicate: all interchangeable
/// variants of a type are applicable regardless of shape (the Selector ranks
/// them by the variant-aware cost). The generated kernel x variant catalog is
/// the single source, so a new catalog entry is applicable automatically; a
/// variant wider than the machine's SIMD lanes is rejected. Types without
/// variants fall back to `.production`.
pub fn planApplicable(comptime plan: MatVecPlan) api.ApplicableFn {
    @setEvalBranchQuota(1_000_000);
    inline for (types.candidates) |cand| {
        if (plan == cand.plan) return quantApplicable(kernelGgmlType(cand.kernel.tag), cand.variant.width);
    }
    return &struct {
        fn f(code: u32, m: u64, n: u64, lanes: u16) bool {
            _ = m;
            _ = n;
            _ = lanes;
            const t = ggml.GGMLType.fromInt(code) orelse return false;
            return switch (plan) {
                .production => !isPlanType(t),
                // The generated kernel x variant plans are handled above.
                else => false,
            };
        }
    }.f;
}

/// Generated-candidate applicability: the quant type matches and the variant's
/// f32 width fits the machine's SIMD lanes (`0` = the build's native width,
/// always fits).
fn quantApplicable(comptime t: ggml.GGMLType, comptime width: u8) api.ApplicableFn {
    return &struct {
        fn f(code: u32, m: u64, n: u64, lanes: u16) bool {
            _ = m;
            _ = n;
            const got = ggml.GGMLType.fromInt(code) orelse return false;
            if (got != t) return false;
            if (width == 0) return true;
            return lanes >= width;
        }
    }.f;
}

/// Roofline inputs shared by every matvec plan: the weight bytes and `2*n*m`
/// flops. The logical work is variant-independent; the Selector charges the
/// calibrated per-variant efficiency on top of this cost.
fn matvecCost(code: u32, m: u64, n: u64, k: u64, lanes: u16) api.Cost {
    _ = k;
    _ = lanes;
    const t = ggml.GGMLType.fromInt(code) orelse return .{};
    const elems = std.math.mul(u64, n, m) catch return .{ .flops = std.math.maxInt(u64) };
    const bytes = ggml.tensorByteLength(t, elems) catch 0;
    const flops = std.math.mul(u64, elems, 2) catch std.math.maxInt(u64);
    return .{ .bytes = bytes, .flops = flops };
}

const plan_kernels = blk: {
    const plan_values = std.enums.values(MatVecPlan);
    var arr: [plan_values.len + 6]api.Kernel = undefined;
    for (plan_values, 0..) |plan, i| {
        arr[i] = .{
            .id = planLabel(plan),
            .op = .matvec,
            .atomic_id = @intFromEnum(AtomicId.mat_vec),
            .row_parallel = true,
            .fusable = true,
            .variant = @intFromEnum(plan),
            .applicable = planApplicable(plan),
            .cost = &matvecCost,
        };
    }
    arr[plan_values.len] = .{
        .id = "mat_mul",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.mat_mul),
        .row_parallel = true,
        .applicable = &neverApplicable,
    };
    arr[plan_values.len + 1] = .{
        .id = "lm_head",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.lm_head),
        .row_parallel = true,
        .applicable = &neverApplicable,
    };
    arr[plan_values.len + 2] = .{
        .id = "gate_up_fused",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.gate_up_fused),
        .row_parallel = true,
        .fusable = true,
        .applicable = &neverApplicable,
    };
    arr[plan_values.len + 3] = .{
        .id = "norm_proj",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.norm_proj),
        .row_parallel = true,
        .fusable = true,
        .applicable = &neverApplicable,
    };
    arr[plan_values.len + 4] = .{
        .id = "matvec_bias",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.matvec_bias),
        .row_parallel = true,
        .fusable = true,
        .applicable = &neverApplicable,
    };
    arr[plan_values.len + 5] = .{
        .id = "mlp_fused",
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.mlp_fused),
        .row_parallel = true,
        .fusable = true,
        .applicable = &neverApplicable,
    };
    break :blk arr;
};

pub const Args = struct {
    view: *const TensorView,
    x: []const f32,
    out: []f32,
    begin: usize = 0,
    end: usize = 0,
    scratch: []f32 = &.{},
    plan: u16 = @intFromEnum(MatVecPlan.production),
    /// Chosen SIMD width (f32 lanes 4/8/16; `0` = build's native).
    simd_width: u8 = 0,
    view_up: ?*const TensorView = null,
    view_down: ?*const TensorView = null,
    weight: []const f32 = &.{},
    bias: []const f32 = &.{},
    eps: f32 = 1e-6,
    act: ActivationKind = .gelu_tanh,
};

fn familyRun(id: AtomicId, args: *const Args) KernelError!void {
    switch (id) {
        .mat_vec => return matVecRowsWithPlan(args.view, args.x, args.out, args.begin, args.end, std.enums.fromInt(MatVecPlan, args.plan) orelse .production, args.simd_width),
        .mat_mul => return matMulRows(args.view, args.x, args.out, args.begin, args.end),
        .lm_head => return lmHeadRows(args.view, args.x, args.out, args.begin, args.end, args.scratch),
        .gate_up_fused => {
            const view_up = args.view_up orelse return error.ShapeMismatch;
            return fusedGateUpRows(args.view, view_up, args.x, args.out, args.begin, args.end, args.act);
        },
        .mlp_fused => {
            const view_up = args.view_up orelse return error.ShapeMismatch;
            const view_down = args.view_down orelse return error.ShapeMismatch;
            return fusedMlpRows(args.view, view_up, view_down, args.x, args.out, args.scratch, args.begin, args.end, args.act);
        },
        .norm_proj => return normThenMatVecRows(args.view, args.x, args.weight, args.eps, args.out, args.begin, args.end, args.scratch),
        .matvec_bias => return matVecWithBiasRows(args.view, args.x, args.bias, args.out, args.begin, args.end),
    }
}

pub const set = api.Family{
    .op = .matvec,
    .name = "matvec",
    .row_parallel = true,
    .kernels = &plan_kernels,
    .run = api.erased(AtomicId, Args, familyRun),
};

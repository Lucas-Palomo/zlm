const std = @import("std");
const api = @import("../../kernels/api.zig");
const ggml = @import("../../gguf/format.zig");
const matvec = @import("../../kernels/linear/matvec/root.zig");

/// Co-processor shader variant of a GPU GEMV AtomicOp: the family sub-variant
/// code for `tier == .gpu` candidates. `plain` is the base shader; `u4`/`u8`
/// are the unrolled/tiled variants; `sub` selects the **subgroup/wave**
/// reduction instead of the shared-memory tree. `un2`/`un4` are inner k-block
/// unroll factors (the ILP lever, the CPU grid's `unroll`); `w32`/`w128` are
/// the workgroup-size axis (`local_size_x` 32/128 vs the base 64). It lives in
/// the GPU kernel grid so the catalog and the backend's AtomicOps share one
/// enum — no parallel variant list.
pub const GpuVariant = enum(u8) { plain = 0, u4 = 1, u8 = 2, sub = 3, un2 = 4, un4 = 5, w32 = 6, w128 = 7 };

/// Shader-variant slots in a per-variant table (one per `GpuVariant`).
pub const variant_slots = std.enums.values(GpuVariant).len;

/// How a GPU GEMV workgroup reduces its per-lane partials into one output row.
/// `shared` is the shared-memory tree with `barrier()`; `subgroup` is the
/// GPU-native `subgroupAdd`/wave reduction (no shared round-trip). This is the
/// highest-value GPU-specific variant axis: the CPU grid has no analogue.
pub const Reduction = enum(u32) { shared = 0, subgroup = 1 };

/// The GPU variant axes as Vulkan specialization constants: the driver
/// specializes **one** shader into N pipelines at `vkCreateComputePipelines`
/// (GLSL has no comptime). `rows_per_group`/`unroll` are the tile and inner
/// unroll (the CPU grid's `tile`/`unroll`); `reduction` is GPU-specific;
/// `workgroup_size` is the local `local_size_x` (the wavefront lanes, 64 here).
pub const SpecConsts = struct {
    rows_per_group: u32 = 4,
    unroll: u32 = 1,
    reduction: Reduction = .shared,
    workgroup_size: u32 = 64,
};

/// The specialization-constant values a `(quant, variant)` maps to. Every GEMV
/// quant ships one spec-constant shader, so the axes are uniform across quants:
/// the variant alone selects the values (the base quant is the same scaffold,
/// only its block decode differs).
pub fn specConsts(t: ggml.GGMLType, v: GpuVariant) SpecConsts {
    _ = t;
    return switch (v) {
        .plain => .{ .rows_per_group = 4, .unroll = 1, .reduction = .shared },
        .u4 => .{ .rows_per_group = 8, .unroll = 1, .reduction = .shared },
        .sub => .{ .rows_per_group = 4, .unroll = 1, .reduction = .subgroup },
        .u8 => .{ .rows_per_group = 4, .unroll = 1, .reduction = .shared },
        .un2 => .{ .rows_per_group = 4, .unroll = 2, .reduction = .shared },
        .un4 => .{ .rows_per_group = 4, .unroll = 4, .reduction = .shared },
        .w32 => .{ .rows_per_group = 4, .unroll = 1, .reduction = .shared, .workgroup_size = 32 },
        .w128 => .{ .rows_per_group = 4, .unroll = 1, .reduction = .shared, .workgroup_size = 128 },
    };
}

/// The matvec family's atomic ids the grid implements. The grid is the GPU
/// side of the family's AtomicOps, so it records the family's ids (one source,
/// no mirror enum).
pub const AtomicId = matvec.AtomicId;

/// GPU candidate validity: the base quant matches and the shape is runnable by
/// the device. The transfer-aware placement (UMA/discrete, residency) is the
/// Scheduler's (`tuner/gpu.zig`), not the Selector's, so it is not encoded here.
fn gpuApplicable(comptime t: ggml.GGMLType) api.ApplicableFn {
    return &struct {
        fn f(code: u32, m: u64, n: u64, lanes: u16) bool {
            _ = lanes;
            return code == @intFromEnum(t) and m > 0 and n > 0;
        }
    }.f;
}

/// Roofline inputs of a GPU GEMV: the weight bytes and `2*n*m` flops. The
/// logical work is variant-independent, like the CPU matvec plans'.
fn matvecCost(code: u32, m: u64, n: u64, k: u64, lanes: u16) api.Cost {
    _ = k;
    _ = lanes;
    const t = ggml.GGMLType.fromInt(code) orelse return .{};
    const elems = std.math.mul(u64, n, m) catch return .{ .flops = std.math.maxInt(u64) };
    const bytes = ggml.tensorByteLength(t, elems) catch 0;
    const flops = std.math.mul(u64, elems, 2) catch std.math.maxInt(u64);
    return .{ .bytes = bytes, .flops = flops };
}

/// Roofline inputs of a fused two-weight GPU GEMV: both weights' bytes and the
/// same `2*n*m` flops as the single-weight op.
fn gpuGateUpCost(code: u32, m: u64, n: u64, k: u64, lanes: u16) api.Cost {
    const c = matvecCost(code, m, n, k, lanes);
    return .{ .bytes = std.math.mul(u64, c.bytes, 2) catch std.math.maxInt(u64), .flops = c.flops };
}

/// One catalogued GPU GEMV AtomicOp (a single-weight `mat_vec` shader).
fn gpuMatvec(comptime id: []const u8, comptime t: ggml.GGMLType, comptime v: GpuVariant) api.Kernel {
    return .{
        .id = id,
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.mat_vec),
        .row_parallel = true,
        .fusable = true,
        .tier = .gpu,
        .variant = @intFromEnum(v),
        .applicable = gpuApplicable(t),
        .cost = &matvecCost,
    };
}

/// One catalogued GPU fused `gate_up` GEMV AtomicOp (two weights, gated act).
fn gpuGateUp(comptime id: []const u8, comptime t: ggml.GGMLType, comptime v: GpuVariant) api.Kernel {
    return .{
        .id = id,
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.gate_up_fused),
        .row_parallel = true,
        .fusable = true,
        .tier = .gpu,
        .variant = @intFromEnum(v),
        .applicable = gpuApplicable(t),
        .cost = &gpuGateUpCost,
    };
}

/// Roofline inputs of a fused three-weight GPU MLP: all three weights' bytes and
/// the `2*n*m` flops of the gate/up plus the `2*hidden*ffn` of the down (the
/// activation is negligible). Same logical work as the three separate GEMVs.
fn gpuMlpCost(code: u32, m: u64, n: u64, k: u64, lanes: u16) api.Cost {
    const c = matvecCost(code, m, n, k, lanes);
    return .{ .bytes = std.math.mul(u64, c.bytes, 3) catch std.math.maxInt(u64), .flops = c.flops };
}

/// One catalogued GPU fused MLP AtomicOp (three weights: gate, up, down).
fn gpuMlp(comptime id: []const u8, comptime t: ggml.GGMLType, comptime v: GpuVariant) api.Kernel {
    return .{
        .id = id,
        .op = .matvec,
        .atomic_id = @intFromEnum(AtomicId.mlp_fused),
        .row_parallel = true,
        .fusable = true,
        .tier = .gpu,
        .variant = @intFromEnum(v),
        .applicable = gpuApplicable(t),
        .cost = &gpuMlpCost,
    };
}

/// The GPU kernel grid: one catalog entry per (op, quant, shader variant), the
/// single source of "which GPU kernels exist". The shaders under `shaders/` are
/// the implementation a backend executes; a new variant is a grid entry plus a
/// shader. `tier == .gpu` entries are never chosen by the CPU Selector — the
/// transfer-aware placement stays in `tuner/gpu.zig`, and the probe enumerates
/// this grid instead of a literal variant list.
pub const gpu_kernels = [_]api.Kernel{
    gpuMatvec("gpu_matvec_q4_0_plain", .q4_0, .plain),
    gpuMatvec("gpu_matvec_q4_0_u4", .q4_0, .u4),
    gpuMatvec("gpu_matvec_q4_0_sub", .q4_0, .sub),
    gpuMatvec("gpu_matvec_q4_0_un2", .q4_0, .un2),
    gpuMatvec("gpu_matvec_q4_0_un4", .q4_0, .un4),
    gpuMatvec("gpu_matvec_q4_0_w32", .q4_0, .w32),
    gpuMatvec("gpu_matvec_q4_0_w128", .q4_0, .w128),
    gpuMatvec("gpu_matvec_q4_1_plain", .q4_1, .plain),
    gpuMatvec("gpu_matvec_q4_1_u4", .q4_1, .u4),
    gpuMatvec("gpu_matvec_q4_1_sub", .q4_1, .sub),
    gpuMatvec("gpu_matvec_q4_1_un2", .q4_1, .un2),
    gpuMatvec("gpu_matvec_q4_1_un4", .q4_1, .un4),
    gpuMatvec("gpu_matvec_q4_1_w32", .q4_1, .w32),
    gpuMatvec("gpu_matvec_q4_1_w128", .q4_1, .w128),
    gpuMatvec("gpu_matvec_q8_0_plain", .q8_0, .plain),
    gpuMatvec("gpu_matvec_q8_0_u4", .q8_0, .u4),
    gpuMatvec("gpu_matvec_q8_0_sub", .q8_0, .sub),
    gpuMatvec("gpu_matvec_q8_0_un2", .q8_0, .un2),
    gpuMatvec("gpu_matvec_q8_0_un4", .q8_0, .un4),
    gpuMatvec("gpu_matvec_q8_0_w32", .q8_0, .w32),
    gpuMatvec("gpu_matvec_q8_0_w128", .q8_0, .w128),
    gpuMatvec("gpu_matvec_q6_k_plain", .q6_k, .plain),
    gpuMatvec("gpu_matvec_q6_k_u4", .q6_k, .u4),
    gpuMatvec("gpu_matvec_q6_k_sub", .q6_k, .sub),
    gpuMatvec("gpu_matvec_q6_k_un2", .q6_k, .un2),
    gpuMatvec("gpu_matvec_q6_k_un4", .q6_k, .un4),
    gpuMatvec("gpu_matvec_q6_k_w32", .q6_k, .w32),
    gpuMatvec("gpu_matvec_q6_k_w128", .q6_k, .w128),
    gpuMatvec("gpu_matvec_q4_k_plain", .q4_k, .plain),
    gpuMatvec("gpu_matvec_q4_k_u4", .q4_k, .u4),
    gpuMatvec("gpu_matvec_q4_k_sub", .q4_k, .sub),
    gpuMatvec("gpu_matvec_q4_k_un2", .q4_k, .un2),
    gpuMatvec("gpu_matvec_q4_k_un4", .q4_k, .un4),
    gpuMatvec("gpu_matvec_q4_k_w32", .q4_k, .w32),
    gpuMatvec("gpu_matvec_q4_k_w128", .q4_k, .w128),
    gpuMatvec("gpu_matvec_f32_plain", .f32, .plain),
    gpuMatvec("gpu_matvec_f32_u4", .f32, .u4),
    gpuMatvec("gpu_matvec_f32_sub", .f32, .sub),
    gpuMatvec("gpu_matvec_f32_un2", .f32, .un2),
    gpuMatvec("gpu_matvec_f32_un4", .f32, .un4),
    gpuMatvec("gpu_matvec_f32_w32", .f32, .w32),
    gpuMatvec("gpu_matvec_f32_w128", .f32, .w128),
    gpuMatvec("gpu_matvec_f16_plain", .f16, .plain),
    gpuMatvec("gpu_matvec_f16_u4", .f16, .u4),
    gpuMatvec("gpu_matvec_f16_sub", .f16, .sub),
    gpuMatvec("gpu_matvec_f16_un2", .f16, .un2),
    gpuMatvec("gpu_matvec_f16_un4", .f16, .un4),
    gpuMatvec("gpu_matvec_f16_w32", .f16, .w32),
    gpuMatvec("gpu_matvec_f16_w128", .f16, .w128),
    gpuMatvec("gpu_matvec_bf16_plain", .bf16, .plain),
    gpuMatvec("gpu_matvec_bf16_u4", .bf16, .u4),
    gpuMatvec("gpu_matvec_bf16_sub", .bf16, .sub),
    gpuMatvec("gpu_matvec_bf16_un2", .bf16, .un2),
    gpuMatvec("gpu_matvec_bf16_un4", .bf16, .un4),
    gpuMatvec("gpu_matvec_bf16_w32", .bf16, .w32),
    gpuMatvec("gpu_matvec_bf16_w128", .bf16, .w128),
    gpuGateUp("gpu_gate_up_q4_0_plain", .q4_0, .plain),
    gpuGateUp("gpu_gate_up_q4_0_u4", .q4_0, .u4),
    gpuGateUp("gpu_gate_up_q4_0_sub", .q4_0, .sub),
    gpuGateUp("gpu_gate_up_q4_0_un2", .q4_0, .un2),
    gpuGateUp("gpu_gate_up_q4_0_un4", .q4_0, .un4),
    gpuGateUp("gpu_gate_up_q4_0_w32", .q4_0, .w32),
    gpuGateUp("gpu_gate_up_q4_0_w128", .q4_0, .w128),
    gpuGateUp("gpu_gate_up_q4_1_plain", .q4_1, .plain),
    gpuGateUp("gpu_gate_up_q4_1_u4", .q4_1, .u4),
    gpuGateUp("gpu_gate_up_q4_1_sub", .q4_1, .sub),
    gpuGateUp("gpu_gate_up_q4_1_un2", .q4_1, .un2),
    gpuGateUp("gpu_gate_up_q4_1_un4", .q4_1, .un4),
    gpuGateUp("gpu_gate_up_q4_1_w32", .q4_1, .w32),
    gpuGateUp("gpu_gate_up_q4_1_w128", .q4_1, .w128),
    gpuGateUp("gpu_gate_up_q8_0_plain", .q8_0, .plain),
    gpuGateUp("gpu_gate_up_q8_0_u4", .q8_0, .u4),
    gpuGateUp("gpu_gate_up_q8_0_sub", .q8_0, .sub),
    gpuGateUp("gpu_gate_up_q8_0_un2", .q8_0, .un2),
    gpuGateUp("gpu_gate_up_q8_0_un4", .q8_0, .un4),
    gpuGateUp("gpu_gate_up_q8_0_w32", .q8_0, .w32),
    gpuGateUp("gpu_gate_up_q8_0_w128", .q8_0, .w128),
    gpuMlp("gpu_mlp_q4_0_plain", .q4_0, .plain),
    gpuMlp("gpu_mlp_q4_0_u4", .q4_0, .u4),
    gpuMlp("gpu_mlp_q4_0_sub", .q4_0, .sub),
    gpuMlp("gpu_mlp_q4_0_un2", .q4_0, .un2),
    gpuMlp("gpu_mlp_q4_0_un4", .q4_0, .un4),
    gpuMlp("gpu_mlp_q4_0_w32", .q4_0, .w32),
    gpuMlp("gpu_mlp_q4_0_w128", .q4_0, .w128),
    gpuMlp("gpu_mlp_q4_1_plain", .q4_1, .plain),
    gpuMlp("gpu_mlp_q4_1_u4", .q4_1, .u4),
    gpuMlp("gpu_mlp_q4_1_sub", .q4_1, .sub),
    gpuMlp("gpu_mlp_q4_1_un2", .q4_1, .un2),
    gpuMlp("gpu_mlp_q4_1_un4", .q4_1, .un4),
    gpuMlp("gpu_mlp_q4_1_w32", .q4_1, .w32),
    gpuMlp("gpu_mlp_q4_1_w128", .q4_1, .w128),
    gpuMlp("gpu_mlp_q8_0_plain", .q8_0, .plain),
    gpuMlp("gpu_mlp_q8_0_u4", .q8_0, .u4),
    gpuMlp("gpu_mlp_q8_0_sub", .q8_0, .sub),
    gpuMlp("gpu_mlp_q8_0_un2", .q8_0, .un2),
    gpuMlp("gpu_mlp_q8_0_un4", .q8_0, .un4),
    gpuMlp("gpu_mlp_q8_0_w32", .q8_0, .w32),
    gpuMlp("gpu_mlp_q8_0_w128", .q8_0, .w128),
};

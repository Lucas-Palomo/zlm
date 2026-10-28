const std = @import("std");
const grid = @import("grid.zig");
const api = @import("../../kernels/api.zig");
const ggml = @import("../../gguf/format.zig");
const matvec = @import("../../kernels/linear/matvec/root.zig");

/// GPU grid entries for one AtomicOp and quant that pass `applicable` at a
/// representative shape. Used by the probe-enumeration test.
fn countGpuVariants(atomic: grid.AtomicId, t: ggml.GGMLType) usize {
    var n: usize = 0;
    for (grid.gpu_kernels) |k| {
        if (k.atomic_id != @intFromEnum(atomic)) continue;
        if (k.applicable) |f| {
            if (f(@intFromEnum(t), 64, 64, 0)) n += 1;
        }
    }
    return n;
}

test "gpu grid: one entry per (op, quant, variant), enumerated by the probe" {
    const testing = std.testing;
    try testing.expect(grid.gpu_kernels.len > 0);
    for (grid.gpu_kernels) |k| {
        try testing.expectEqual(api.Op.matvec, k.op);
        try testing.expectEqual(api.Tier.gpu, k.tier);
        try testing.expect(std.enums.fromInt(grid.GpuVariant, k.variant) != null);
        try testing.expect(k.atomic_id == @intFromEnum(grid.AtomicId.mat_vec) or
            k.atomic_id == @intFromEnum(grid.AtomicId.gate_up_fused) or
            k.atomic_id == @intFromEnum(grid.AtomicId.mlp_fused));
        try testing.expect(k.applicable != null);
        try testing.expect(k.cost != null);
    }
    // The grid is the source of the variant list the probe enumerates, and it
    // is uniform across quants: every GEMV family ships the same seven variants.
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mat_vec, .q4_0));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mat_vec, .q4_k));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mat_vec, .q8_0));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.gate_up_fused, .q4_0));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.gate_up_fused, .q4_1));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.gate_up_fused, .q8_0));
    try testing.expectEqual(@as(usize, 0), countGpuVariants(.gate_up_fused, .q4_k));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mlp_fused, .q4_0));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mlp_fused, .q4_1));
    try testing.expectEqual(@as(usize, 7), countGpuVariants(.mlp_fused, .q8_0));
    try testing.expectEqual(@as(usize, 0), countGpuVariants(.mlp_fused, .q6_k));
    // The grid moved off the CPU side: the CPU family's candidate catalog
    // carries no GPU entries (the Selector only ever reads the CPU tier).
    var in_set: usize = 0;
    for (matvec.set.kernels) |k| {
        if (k.tier == .gpu) in_set += 1;
    }
    try testing.expectEqual(@as(usize, 0), in_set);
}

test "gpu spec constants: the q4_0 variants map to the axes" {
    const testing = std.testing;
    const plain = grid.specConsts(.q4_0, .plain);
    try testing.expectEqual(@as(u32, 4), plain.rows_per_group);
    try testing.expectEqual(@as(u32, 1), plain.unroll);
    try testing.expectEqual(grid.Reduction.shared, plain.reduction);

    // `u4` is a tile (rows), not an unroll: the axis that names it is
    // `rows_per_group`, fixing the old `u4`/tile naming drift.
    try testing.expectEqual(@as(u32, 8), grid.specConsts(.q4_0, .u4).rows_per_group);

    // The reduction axis: the same shader, specialized to the subgroup/wave.
    const sub = grid.specConsts(.q4_0, .sub);
    try testing.expectEqual(grid.Reduction.subgroup, sub.reduction);
    try testing.expectEqual(@as(u32, 4), sub.rows_per_group);

    // The axes are uniform across quants: the base quant does not enter the
    // mapping, so every GEMV family specializes the same way.
    try testing.expectEqual(@as(u32, 8), grid.specConsts(.q6_k, .u4).rows_per_group);
    try testing.expectEqual(@as(u32, 8), grid.specConsts(.q4_k, .u4).rows_per_group);
    try testing.expectEqual(grid.Reduction.subgroup, grid.specConsts(.q4_1, .sub).reduction);
    try testing.expectEqual(@as(u32, 2), grid.specConsts(.q8_0, .un2).unroll);
    try testing.expectEqual(@as(u32, 128), grid.specConsts(.f32, .w128).workgroup_size);

    // The UNROLL axis: the inner k-block unroll factor.
    const un2 = grid.specConsts(.q4_0, .un2);
    try testing.expectEqual(@as(u32, 2), un2.unroll);
    try testing.expectEqual(@as(u32, 4), un2.rows_per_group);
    try testing.expectEqual(grid.Reduction.shared, un2.reduction);
    try testing.expectEqual(@as(u32, 64), un2.workgroup_size);
    try testing.expectEqual(@as(u32, 4), grid.specConsts(.q4_0, .un4).unroll);

    // The WORKGROUP_SIZE axis: the local_size_x.
    try testing.expectEqual(@as(u32, 32), grid.specConsts(.q4_0, .w32).workgroup_size);
    try testing.expectEqual(@as(u32, 128), grid.specConsts(.q4_0, .w128).workgroup_size);
    try testing.expectEqual(@as(u32, 1), grid.specConsts(.q4_0, .w32).unroll);
    try testing.expectEqual(@as(u32, 64), grid.specConsts(.q4_0, .plain).workgroup_size);
}

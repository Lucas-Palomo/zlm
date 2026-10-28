const std = @import("std");
const contract = @import("root.zig");
const grid = @import("../kernels/grid.zig");

test "probe enumerates the gpu variant list from the catalog" {
    const testing = std.testing;
    const mat_vec = @intFromEnum(grid.AtomicId.mat_vec);
    const gate_up = @intFromEnum(grid.AtomicId.gate_up_fused);

    const q4_0 = contract.catalogGpuVariants(mat_vec, .q4_0, 64, 64);
    try testing.expectEqualSlices(grid.GpuVariant, &.{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 }, q4_0.slice());

    // The grid is uniform across quants: q4_k ships the same seven variants.
    const q4_k = contract.catalogGpuVariants(mat_vec, .q4_k, 64, 256);
    try testing.expectEqualSlices(grid.GpuVariant, &.{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 }, q4_k.slice());

    const gu = contract.catalogGpuVariants(gate_up, .q4_0, 64, 64);
    try testing.expectEqualSlices(grid.GpuVariant, &.{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 }, gu.slice());

    // A quant with no shader enumerates nothing.
    const none = contract.catalogGpuVariants(mat_vec, .q5_k, 64, 64);
    try testing.expectEqual(@as(usize, 0), none.len);
}

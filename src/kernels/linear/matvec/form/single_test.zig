const std = @import("std");
const single = @import("single.zig");
const quantized = @import("../kernels/quantized.zig");
const fixtures = @import("../fixtures.zig");
const TensorView = @import("../../../../data/tensor_view.zig").TensorView;

test "matVecWithBiasRows matches matVec plus bias" {
    const allocator = std.testing.allocator;
    const n: usize = 4;
    const m: usize = 3;
    var weights = [_]f32{
        1,  2,   3, 4,
        -1, 0.5, 2, -2,
        0,  1,   0, 1,
    };
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    const x = [_]f32{ 1, -1, 0.5, 2 };
    const bias = [_]f32{ 0.25, -0.5, 3 };
    var got = [_]f32{0} ** m;
    var want = [_]f32{0} ** m;

    try single.matVecWithBiasRows(&view, &x, &bias, &got, 0, m);
    try single.matVecRows(&view, &x, &want, 0, m);
    for (&want, bias) |*w, b| w.* += b;

    _ = allocator;
    try std.testing.expectEqualSlices(f32, &want, &got);
}

test "matMulWithBiasRows matches matMul plus per-token bias" {
    const n: usize = 4;
    const m: usize = 2;
    const M: usize = 2;
    var weights = [_]f32{
        1,  0, 0, 1,
        -1, 2, 1, 0,
    };
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    const x = [_]f32{
        1,  2, 3, 4,
        -1, 0, 1, 2,
    };
    const bias = [_]f32{ 0.5, -2 };
    var got = [_]f32{0} ** (M * m);
    var want = [_]f32{0} ** (M * m);

    try single.matMulWithBiasRows(&view, &x, &bias, &got, 0, m);
    try single.matMulRows(&view, &x, &want, 0, m);
    for (0..M) |tok| {
        for (0..m) |row| want[tok * m + row] += bias[row];
    }

    try std.testing.expectEqualSlices(f32, &want, &got);
}

test "matvec_bias atomic matches matvec plus bias" {
    const n: usize = 4;
    const m: usize = 3;
    var weights = [_]f32{ 1, 2, 3, 4, -1, 0.5, 2, -2, 0, 1, 0, 1 };
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{ .name = "w", .dims = &dims, .ggml_type = .f32, .offset = 0, .byte_length = data.len, .mapped_slice = data };
    const x = [_]f32{ 1, -1, 0.5, 2 };
    const bias = [_]f32{ 0.25, -0.5, 3 };
    var got = [_]f32{0} ** m;
    var want = [_]f32{0} ** m;

    try single.matVecWithBiasRows(&view, &x, &bias, &got, 0, m);

    try single.matVecRows(&view, &x, &want, 0, m);
    for (&want, bias) |*w, b| w.* += b;
    for (0..m) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

test "q4_0 u4/t2/t4 variants match production within 1e-5" {
    const n: usize = 64;
    const m: usize = 9;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 32) * 18]u8 = undefined;
    fixtures.fillQ4_0Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m);

    var got_u4: [m]f32 = [_]f32{0} ** m;
    try quantized.matVecQ4_0U4(&data, n, m, &x, &got_u4, 0, m);
    var got_t2: [m]f32 = [_]f32{0} ** m;
    try single.matVecFpGen(quantized.Q4_0, .{ .width = 8, .tile = 2, .unroll = 1 }, &data, n, m, &x, &got_t2, 0, m);
    var got_t4: [m]f32 = [_]f32{0} ** m;
    try single.matVecFpGen(quantized.Q4_0, .{ .width = 8, .tile = 4, .unroll = 1 }, &data, n, m, &x, &got_t4, 0, m);

    for (0..m) |i| {
        try std.testing.expectApproxEqAbs(want[i], got_u4[i], 1e-5);
        try std.testing.expectApproxEqAbs(want[i], got_t2[i], 1e-5);
        try std.testing.expectApproxEqAbs(want[i], got_t4[i], 1e-5);
    }
}

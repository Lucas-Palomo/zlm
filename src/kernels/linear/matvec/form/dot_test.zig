const std = @import("std");
const dot_mod = @import("dot.zig");
const single = @import("single.zig");
const quantized = @import("../kernels/quantized.zig");
const fixtures = @import("../fixtures.zig");
const layout = @import("../layout.zig");
const types = @import("../types.zig");
const simd = @import("../../../simd.zig");
const ggml = @import("../../../../gguf/format.zig");
const TensorView = @import("../../../../data/tensor_view.zig").TensorView;

/// The generic dot against the production single-row sweep (the reference the
/// per-quant `dotX` helpers used to mirror). Both `xsum` paths are checked: the
/// inline one and, for `has_min`, the precomputed one.
fn expectDotMatchesSweep(comptime K: type, comptime fill: fn ([]u8, usize, usize) void, comptime n: usize, comptime m: usize) !void {
    const testing = std.testing;
    var data: [m * (n / K.block_elems) * K.block_bytes]u8 = undefined;
    fill(&data, n, m);

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }

    var want: [m]f32 = undefined;
    try single.matVecFpGen(K, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, &data, n, m, &x, &want, 0, m);

    var xsum: [types.max_activation_cache_blocks]f32 = undefined;
    if (comptime K.has_min) layout.q4KXSum(&x, xsum[0 .. n / 32]);

    const row_bytes = (n / K.block_elems) * K.block_bytes;
    for (0..m) |i| {
        const row = data[i * row_bytes ..][0..row_bytes];
        try testing.expectEqual(want[i], try dot_mod.dot(K, row, &x, null));
        if (comptime K.has_min) {
            try testing.expectEqual(want[i], try dot_mod.dot(K, row, &x, xsum[0 .. n / 32]));
        }
    }
}

test "generic dot matches the production sweep for every quant" {
    try expectDotMatchesSweep(quantized.Q4_0, fixtures.fillQ4_0Fixture, 32 * 33, 7);
    try expectDotMatchesSweep(quantized.Q4_1, fixtures.fillQ4_1Fixture, 32 * 33, 7);
    try expectDotMatchesSweep(quantized.Q8_0, fixtures.fillQ8_0Fixture, 32 * 33, 7);
    try expectDotMatchesSweep(quantized.Q6_K, fixtures.fillQ6_KFixture, 256 * 5, 7);
    try expectDotMatchesSweep(quantized.Q4_K, fixtures.fillQ4_KFixture, 256 * 5, 7);
}

fn expectLmHeadMatchesMatVec(comptime K: type, comptime t: ggml.GGMLType, comptime fill: fn ([]u8, usize, usize) void, comptime n: usize, comptime m: usize) !void {
    const testing = std.testing;
    const row_bytes = (n / K.block_elems) * K.block_bytes;
    var data: [m * row_bytes]u8 = undefined;
    fill(&data, n, m);

    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = t,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }

    var want: [m]f32 = undefined;
    try single.matVecRows(&view, &x, &want, 0, m);
    var got: [m]f32 = undefined;
    try single.lmHeadRows(&view, &x, &got, 0, m, &.{});
    try testing.expectEqualSlices(f32, &want, &got);
}

test "lmHeadRows routes every dot quant through the generic dot" {
    try expectLmHeadMatchesMatVec(quantized.Q8_0, .q8_0, fixtures.fillQ8_0Fixture, 32 * 33, 5);
    try expectLmHeadMatchesMatVec(quantized.Q6_K, .q6_k, fixtures.fillQ6_KFixture, 256 * 5, 5);
    try expectLmHeadMatchesMatVec(quantized.Q4_K, .q4_k, fixtures.fillQ4_KFixture, 256 * 5, 5);
}

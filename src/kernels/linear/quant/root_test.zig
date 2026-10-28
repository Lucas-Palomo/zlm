const std = @import("std");
const root = @import("root.zig");
const types = @import("types.zig");
const k_quants = @import("k_quants.zig");
const quantized = @import("../matvec/kernels/quantized.zig");

const dequantTo = root.dequantTo;
const dequantScaledTo = root.dequantScaledTo;
const encodeQ8_0Block = root.encodeQ8_0Block;
const encodeQ4_0Block = root.encodeQ4_0Block;
const encodeQ4_1Block = root.encodeQ4_1Block;

/// A deterministic q5_k super-block: d, min, 12 packed scales, the high-bit
/// plane and the 128 quant bytes.
fn fillQ5_KBlock(block: *[types.Q5_K_BLOCK]u8, seed: u32) void {
    @memset(block, 0);
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.5)), .little);
    const scales = [_]u8{ 0x41, 2, 3, 4, 5, 0x45, 7, 8, 0x21, 0x13, 0x0F, 0x32 };
    @memcpy(block[4..16], &scales);
    var s = seed;
    for (16..types.Q5_K_BLOCK) |i| {
        s = s *% 1664525 +% 1013904223;
        block[i] = @truncate(s >> 16);
    }
}

test "dequant f32" {
    const vals = [_]f32{ 1.0, -2.5, 3.25, 0.0 };
    var raw: [16]u8 = undefined;
    for (vals, 0..) |v, i| {
        std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(v), .little);
    }
    var out: [4]f32 = undefined;
    try dequantTo(.f32, &raw, &out);
    try std.testing.expectEqual(vals, out);
}

test "dequant f16" {
    const h: f16 = 1.5;
    var raw: [2]u8 = undefined;
    std.mem.writeInt(u16, &raw, @bitCast(h), .little);
    var out: [1]f32 = undefined;
    try dequantTo(.f16, &raw, &out);
    try std.testing.expectEqual(@as(f32, 1.5), out[0]);
}

test "dequant bf16" {
    const vals = [_]u16{ 0x3F80, 0xC000, 0x0000, 0x3F00 };
    const expected = [_]f32{ 1.0, -2.0, 0.0, 0.5 };
    var raw: [8]u8 = undefined;
    for (vals, 0..) |v, i| {
        std.mem.writeInt(u16, raw[i * 2 ..][0..2], v, .little);
    }
    var out: [4]f32 = undefined;
    try dequantTo(.bf16, &raw, &out);
    try std.testing.expectEqual(expected, out);
}

test "dequant q8_0" {
    var qs: [32]i8 = undefined;
    for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i)) - 16);
    const block = encodeQ8_0Block(0.5, &qs);
    var out: [32]f32 = undefined;
    try dequantTo(.q8_0, &block, &out);
    for (&qs, 0..) |q, i| {
        const expected = 0.5 * @as(f32, @floatFromInt(q));
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-6);
    }
}

test "dequant q4_0" {
    var nib: [16]u8 = undefined;
    for (&nib, 0..) |*n, i| n.* = @intCast((i << 4) | (i & 0xF));
    const block = encodeQ4_0Block(1.0, &nib);
    var out: [32]f32 = undefined;
    try dequantTo(.q4_0, &block, &out);
    for (0..16) |j| {
        const x0: i32 = @as(i32, nib[j] & 0x0F) - 8;
        const x1: i32 = @as(i32, nib[j] >> 4) - 8;
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(x0)), out[j], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(x1)), out[16 + j], 1e-6);
    }
}

test "dequant q4_1" {
    var nib: [16]u8 = undefined;
    for (&nib, 0..) |*n, i| n.* = @truncate(i * 17 + 5);
    const block = encodeQ4_1Block(0.5, -1.25, &nib);
    var out: [32]f32 = undefined;
    try dequantTo(.q4_1, &block, &out);
    for (0..16) |j| {
        const x0: f32 = @floatFromInt(nib[j] & 0x0F);
        const x1: f32 = @floatFromInt(nib[j] >> 4);
        try std.testing.expectApproxEqAbs(x0 * 0.5 - 1.25, out[j], 1e-6);
        try std.testing.expectApproxEqAbs(x1 * 0.5 - 1.25, out[16 + j], 1e-6);
    }
}

test "dequant q4_k block layout" {
    var block: [types.Q4_K_BLOCK]u8 = [_]u8{0} ** types.Q4_K_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.5)), .little);
    const scales = [_]u8{ 0x41, 2, 3, 4, 5, 0x45, 7, 8, 0x21, 0x13, 0x0F, 0x32 };
    @memcpy(block[4..16], &scales);
    for (0..128) |i| block[16 + i] = @truncate(i * 17 + 3);

    var out: [types.QK_K]f32 = undefined;
    try dequantTo(.q4_k, &block, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), out[32], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), out[64], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 101.0), out[128], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.5), out[160], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), out[192], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 26.5), out[255], 1e-5);
}

test "dequant q5_k block layout" {
    var block: [types.Q5_K_BLOCK]u8 = [_]u8{0} ** types.Q5_K_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.5)), .little);
    const scales = [_]u8{ 0x41, 2, 3, 4, 5, 0x45, 7, 8, 0x21, 0x13, 0x0F, 0x32 };
    @memcpy(block[4..16], &scales);
    for (16..48) |i| block[i] = 0x01;
    for (48..176) |i| block[i] = @truncate((i - 48) * 17 + 3);

    var out: [types.QK_K]f32 = undefined;
    try dequantTo(.q5_k, &block, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 35.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), out[32], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), out[64], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 101.0), out[128], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.5), out[160], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), out[192], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 26.5), out[255], 1e-5);

    for (16..48) |i| block[i] = 0xFF;
    try dequantTo(.q5_k, &block, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 35.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 61.5), out[32], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 110.5), out[64], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 645.0), out[128], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 111.5), out[160], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 570.0), out[192], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.5), out[255], 1e-5);
}

test "generic dequant q5_k is bit-identical to the codec" {
    const blocks = 3;
    var data: [blocks * types.Q5_K_BLOCK]u8 = undefined;
    for (0..blocks) |b| {
        fillQ5_KBlock(data[b * types.Q5_K_BLOCK ..][0..types.Q5_K_BLOCK], 0xABCD_0000 + @as(u32, @intCast(b)));
    }

    var want: [blocks * types.QK_K]f32 = undefined;
    try k_quants.dequantQ5_K(&data, &want);

    var got: [blocks * types.QK_K]f32 = undefined;
    try root.dequant(quantized.Q5_K, &data, &got);

    for (want, got) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

test "dequant q8_0 partial tail" {
    var qs: [32]i8 = [_]i8{1} ** 32;
    const block = encodeQ8_0Block(2.0, &qs);
    var out: [10]f32 = undefined;
    try dequantTo(.q8_0, &block, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
}

test "dequant_scale matches dequant then scale" {
    var qs: [32]i8 = undefined;
    for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i)) - 16);
    const block = encodeQ8_0Block(0.25, &qs);
    const scale: f32 = 2.5;
    var got: [32]f32 = undefined;
    var want: [32]f32 = undefined;

    try dequantScaledTo(.q8_0, &block, scale, &got);
    try dequantTo(.q8_0, &block, &want);
    for (&want) |*v| v.* *= scale;

    for (0..32) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

test "dequant q6_k block layout" {
    var block: [types.Q6_K_BLOCK]u8 = [_]u8{0} ** types.Q6_K_BLOCK;
    std.mem.writeInt(u16, block[208..210], @bitCast(@as(f16, 0.5)), .little);
    for (0..128) |i| block[i] = 0x11;
    for (128..192) |i| block[i] = 0x00;
    for (192..208) |i| block[i] = @bitCast(@as(i8, 2));

    var out: [types.QK_K]f32 = undefined;
    try dequantTo(.q6_k, &block, &out);

    for (0..32) |l| {
        const q1: i32 = @as(i32, 0x11 & 0x0F) | @as(i32, (0x00 >> 0) & 0x03) << 4;
        const expected = 0.5 * 2.0 * @as(f32, @floatFromInt(q1 - 32));
        try std.testing.expectApproxEqAbs(expected, out[l], 1e-5);
    }
}

test "dequant q6_k rejects non-multiple" {
    var block: [types.Q6_K_BLOCK]u8 = [_]u8{0} ** types.Q6_K_BLOCK;
    var out: [100]f32 = undefined;
    try std.testing.expectError(error.ShapeMismatch, dequantTo(.q6_k, &block, &out));
}

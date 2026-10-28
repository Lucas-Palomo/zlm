const std = @import("std");
const dequant = @import("../../quant/root.zig");
const k_quants = @import("../../quant/k_quants.zig");
const types = @import("../../quant/types.zig");
const quantized = @import("quantized.zig");
const fixtures = @import("../fixtures.zig");

test "q4_0 integer matvec agrees with the f32 path within q8 activation error" {
    const testing = std.testing;
    const standard = @import("../../quant/standard.zig");
    const n: usize = 128;
    const m: usize = 17;
    const row_blocks = n / 32;

    // Synthetic q4_0 matrix: pseudo-random nibbles, small per-block scales.
    var data: [m * row_blocks * 18]u8 = undefined;
    var seed: u32 = 0x1234_5678;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var nib: [16]u8 = undefined;
            for (&nib) |*b| {
                seed = seed *% 1664525 +% 1013904223;
                b.* = @truncate(seed >> 16);
            }
            const d: f16 = @floatCast(0.02 + 0.003 * @as(f32, @floatFromInt((row + rb) % 11)));
            const block = dequant.encodeQ4_0Block(d, &nib);
            @memcpy(data[(row * row_blocks + rb) * 18 ..][0..18], &block);
        }
    }

    // Activation spanning several magnitudes so the quantization error is real.
    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        v.* = (t * 2.0 - 1.0) * (0.25 + 0.25 * @as(f32, @floatFromInt(i % 7)));
    }

    var want: [m]f32 = undefined;
    var got: [m]f32 = undefined;
    try quantized.matVecQ4_0U4(&data, n, m, &x, &want, 0, m);
    try fixtures.q4_0IntOracle(&data, n, m, &x, &got, 0, m);

    // Reference: f32 dot against the block-dequantized weights.
    for (0..m) |row| {
        var row_ref: f32 = 0;
        var scale: f32 = 0;
        for (0..row_blocks) |rb| {
            var deq: [32]f32 = undefined;
            standard.dequantQ4_0Block(data[(row * row_blocks + rb) * 18 ..][0..18], &deq);
            for (0..32) |j| {
                const term = deq[j] * x[rb * 32 + j];
                row_ref += term;
                scale += @abs(term);
            }
        }
        // The f32 kernel is exact here; the integer path only differs by the q8
        // activation quantization, whose per-term error is <= 1/254, so a 2%
        // bound relative to `sum |w*x|` is comfortable.
        try testing.expectApproxEqAbs(row_ref, want[row], 1e-4);
        try testing.expect(@abs(got[row] - row_ref) <= 0.02 * scale + 1e-6);
    }
}

test "q8_0 block dot is equal across widths within tolerance" {
    const testing = std.testing;
    var qs: [32]i8 = undefined;
    for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((i * 7 + 1) % 251)) - 125);
    const block = dequant.encodeQ8_0Block(@as(f16, 0.03125), &qs);
    var xb: [32]f32 = undefined;
    for (&xb, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.25;

    const r4 = quantized.q8_0BlockDotLanes(4, &block, &xb);
    const r8 = quantized.q8_0BlockDotLanes(8, &block, &xb);
    const r16 = quantized.q8_0BlockDotLanes(16, &block, &xb);
    try testing.expectApproxEqRel(r4, r8, 1e-5);
    try testing.expectApproxEqRel(r8, r16, 1e-5);
}

test "q4_k integer matvec agrees with the f32 path within q8 activation error" {
    const testing = std.testing;
    const n: usize = 2048;
    const m: usize = 3;
    const row_blocks = n / 256;

    var data: [m * row_blocks * 144]u8 = undefined;
    var seed: u32 = 0x2468_1357;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const block = data[(row * row_blocks + rb) * 144 ..][0..144];
            const d: f16 = @floatCast(0.03 + 0.002 * @as(f32, @floatFromInt((row + rb) % 7)));
            const dm: f16 = @floatCast(0.01 + 0.001 * @as(f32, @floatFromInt((row + rb) % 5)));
            std.mem.writeInt(u16, block[0..2], @bitCast(d), .little);
            std.mem.writeInt(u16, block[2..4], @bitCast(dm), .little);
            for (4..144) |i| {
                seed = seed *% 1664525 +% 1013904223;
                block[i] = @truncate(seed >> 13);
            }
        }
    }

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        v.* = (t * 2.0 - 1.0) * (0.5 + 0.25 * @as(f32, @floatFromInt(i % 9)));
    }

    var want: [m]f32 = undefined;
    var got: [m]f32 = undefined;
    try quantized.matVecQ4_K(&data, n, m, &x, &want, 0, m);
    try fixtures.q4_KIntOracle(&data, n, m, &x, &got, 0, m);

    // The integer path differs from the f32 path only by the q8 activation
    // quantization, so bound the error relative to `sum |w*x|`.
    for (0..m) |row| {
        var scale: f32 = 0;
        for (0..row_blocks) |rb| {
            var deq: [256]f32 = undefined;
            try dequant.dequantTo(.q4_k, data[(row * row_blocks + rb) * 144 ..][0..144], &deq);
            for (0..256) |j| scale += @abs(deq[j] * x[rb * 256 + j]);
        }
        try testing.expect(@abs(got[row] - want[row]) <= 0.02 * scale + 1e-6);
    }
}

test "q5_k L1 block dequant matches the codec bit-exactly" {
    const testing = std.testing;
    var block: [types.Q5_K_BLOCK]u8 = undefined;
    var seed: u32 = 0x0BAD_F00D;
    for (&block) |*b| {
        seed = seed *% 1664525 +% 1013904223;
        b.* = @truncate(seed >> 15);
    }
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 0.75)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.25)), .little);

    var want: [types.QK_K]f32 = undefined;
    try k_quants.dequantQ5_K(&block, &want);

    var got: [types.QK_K]f32 = undefined;
    try quantized.Q5_K.dequantBlock(&block, &got);

    for (want, got) |a, b| {
        try testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

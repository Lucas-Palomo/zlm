const std = @import("std");
const float = @import("float.zig");

test "F32/F16 fpBlock folds a 32-element chunk into the accumulator" {
    const testing = std.testing;
    const lanes = 8;

    var x: [32]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25;

    var f32_block: [128]u8 = undefined;
    var want32: f32 = 0;
    for (0..32) |j| {
        const w: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(j % 5)) - 2)) * 0.5;
        std.mem.writeInt(u32, f32_block[j * 4 ..][0..4], @bitCast(w), .little);
        want32 += w * x[j];
    }
    var acc32: @Vector(lanes, f32) = @splat(0);
    var corr32: f32 = 0;
    float.F32.fpBlock(lanes, &f32_block, .{ .x = &x }, 0, &acc32, &corr32);
    try testing.expectApproxEqAbs(want32, @reduce(.Add, acc32), 1e-4);
    try testing.expectEqual(@as(f32, 0), corr32);

    var f16_block: [64]u8 = undefined;
    var want16: f32 = 0;
    for (0..32) |j| {
        const w: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(j % 5)) - 2)) * 0.5);
        std.mem.writeInt(u16, f16_block[j * 2 ..][0..2], @bitCast(w), .little);
        want16 += @as(f32, w) * x[j];
    }
    var acc16: @Vector(lanes, f32) = @splat(0);
    var corr16: f32 = 0;
    float.F16.fpBlock(lanes, &f16_block, .{ .x = &x }, 0, &acc16, &corr16);
    try testing.expectApproxEqAbs(want16, @reduce(.Add, acc16), 1e-4);
    try testing.expectEqual(@as(f32, 0), corr16);
}

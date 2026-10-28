const std = @import("std");
const bfloat = @import("bfloat.zig");

test "BF16 fpBlock folds a 32-element chunk into the accumulator" {
    const testing = std.testing;
    const lanes = 8;

    var x: [32]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25;

    var block: [64]u8 = undefined;
    var want: f32 = 0;
    for (0..32) |j| {
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(j % 5)) - 2)) * 0.5;
        const bits: u32 = @bitCast(v);
        const b: u16 = @truncate(bits >> 16);
        std.mem.writeInt(u16, block[j * 2 ..][0..2], b, .little);
        want += @as(f32, @bitCast(bits & 0xFFFF_0000)) * x[j];
    }
    var acc: @Vector(lanes, f32) = @splat(0);
    var corr: f32 = 0;
    bfloat.BF16.fpBlock(lanes, &block, .{ .x = &x }, 0, &acc, &corr);
    try testing.expectApproxEqAbs(want, @reduce(.Add, acc), 1e-4);
    try testing.expectEqual(@as(f32, 0), corr);
}

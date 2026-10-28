const std = @import("std");
const dequant = @import("../quant/root.zig");
const layout = @import("layout.zig");

test "q4_0Expand matches the reference q4_0 dequant layout" {
    const testing = std.testing;
    var nibbles: [16]u8 = undefined;
    for (&nibbles, 0..) |*b, i| b.* = @truncate((i * 37 + 11) & 0xFF);
    const block = dequant.encodeQ4_0Block(@as(f16, 0.5), &nibbles);
    const q = layout.q4_0Expand(&block);
    for (0..16) |j| {
        try testing.expectEqual(@as(i8, @intCast(@as(i32, nibbles[j] & 0x0F) - 8)), q[j]);
        try testing.expectEqual(@as(i8, @intCast(@as(i32, nibbles[j] >> 4) - 8)), q[16 + j]);
    }
}

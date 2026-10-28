const std = @import("std");
const simd = @import("simd.zig");

test "width lanes and native are sane" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 4), simd.Width.s4.lanes());
    try testing.expectEqual(@as(usize, 8), simd.Width.s8.lanes());
    try testing.expectEqual(@as(usize, 16), simd.Width.s16.lanes());
    for (simd.all) |w| {
        const n = w.lanes();
        try testing.expect(n == 4 or n == 8 or n == 16);
    }
    try testing.expect(simd.native == .s4 or simd.native == .s8 or simd.native == .s16);
}

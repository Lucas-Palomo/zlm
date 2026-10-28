const std = @import("std");
const rope = @import("rope.zig");

test "rope rotates pairs" {
    var head = [_]f32{ 1, 0, 0, 1 };
    rope.rope(&head, 0, 10000);
    try std.testing.expectApproxEqAbs(@as(f32, 1), head[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), head[1], 1e-6);

    var h2 = [_]f32{ 1, 0 };
    rope.rope(&h2, 1, 1.0);
    const angle = @as(f32, 1.0);
    try std.testing.expectApproxEqAbs(@cos(angle), h2[0], 1e-5);
    try std.testing.expectApproxEqAbs(@sin(angle), h2[1], 1e-5);

    var h3 = [_]f32{ 1, 0, 0, 1 };
    rope.ropeInterleaved(&h3, 0, 10000);
    try std.testing.expectApproxEqAbs(@as(f32, 1), h3[0], 1e-6);
}

test "rope table matches compute path" {
    const allocator = std.testing.allocator;
    var t = try rope.RopeTable.init(allocator, 4, 8, 10000.0, 8.0);
    defer t.deinit(allocator);
    t.ensure(3);

    var h1 = [_]f32{ 1, 0.5, -0.3, 2, 0.1, 0.2, -1, 1.5 };
    var h2 = h1;
    rope.ropeProportional(&h1, 3, 10000.0, 8, 8);
    rope.ropeProportionalFromTable(&h2, t.cosRow(3), t.sinRow(3), 8, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(h1[i], h2[i], 1e-4);

    var g1 = [_]f32{ 1, 0.5, -0.3, 2, 0.1, 0.2, -1, 1.5 };
    var g2 = g1;
    rope.rope(&g1, 3, 10000.0);
    rope.ropeFromTable(&g2, t.cosRow(3), t.sinRow(3));
    for (0..8) |i| try std.testing.expectApproxEqAbs(g1[i], g2[i], 1e-4);
}

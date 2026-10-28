const std = @import("std");
const elementwise = @import("elementwise.zig");

test "mul scale combine" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    var out: [3]f32 = undefined;

    var args_mul = elementwise.Args{ .a = &a, .b = &b, .out = &out };
    try elementwise.set.run(@intFromEnum(elementwise.AtomicId.mul), &args_mul);
    try std.testing.expectEqual([_]f32{ 4, 10, 18 }, out);

    var args_scale = elementwise.Args{ .a = &a, .out = &out, .scale = 0.5 };
    try elementwise.set.run(@intFromEnum(elementwise.AtomicId.scale), &args_scale);
    try std.testing.expectEqual([_]f32{ 0.5, 1.0, 1.5 }, out);

    var args_combine = elementwise.Args{ .a = &a, .b = &b, .out = &out, .scale = 0.5 };
    try elementwise.set.run(@intFromEnum(elementwise.AtomicId.combine), &args_combine);
    try std.testing.expectEqual([_]f32{ 2.5, 3.5, 4.5 }, out);
}

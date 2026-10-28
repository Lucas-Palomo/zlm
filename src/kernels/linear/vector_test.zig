const std = @import("std");
const vector = @import("vector.zig");
const activation = @import("../transformer/activation.zig");

test "dot add mul silu" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    try std.testing.expectApproxEqAbs(@as(f32, 32), vector.dot(&a, &b), 1e-6);

    var out: [3]f32 = undefined;
    vector.add(&a, &b, &out);
    try std.testing.expectEqual([_]f32{ 5, 7, 9 }, out);
    vector.mul(&a, &b, &out);
    try std.testing.expectEqual([_]f32{ 4, 10, 18 }, out);

    var s: [1]f32 = undefined;
    activation.silu(&[_]f32{0}, &s);
    try std.testing.expectEqual(@as(f32, 0), s[0]);
    activation.silu(&[_]f32{1}, &s);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), s[0], 1e-5);
}
test "maxReduce sum" {
    const v = [_]f32{ -1, 3, 2 };
    try std.testing.expectEqual(@as(f32, 3), vector.maxReduce(&v));
    try std.testing.expectApproxEqAbs(@as(f32, 4), vector.sum(&v), 1e-6);
}

test "softmax stable and sums to one" {
    const in = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;
    vector.softmax(&in, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1), vector.sum(&out), 1e-6);
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);

    const big = [_]f32{ 1000.0, 1001.0 };
    var big_out: [2]f32 = undefined;
    vector.softmax(&big, &big_out);
    try std.testing.expectApproxEqAbs(@as(f32, 1), vector.sum(&big_out), 1e-6);
    try std.testing.expect(!std.math.isNan(big_out[0]));

    var inplace = in;
    vector.softmaxInPlace(&inplace);
    try std.testing.expectApproxEqAbs(@as(f32, 1), vector.sum(&inplace), 1e-6);
}

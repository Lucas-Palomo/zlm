const std = @import("std");
const activation = @import("activation.zig");

test "gatedSilu multiplies silu(gate) by up" {
    const gate = [_]f32{ 0.0, 1.0 };
    const up = [_]f32{ 2.0, 3.0 };
    var out: [2]f32 = undefined;
    activation.gatedSilu(&gate, &up, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(0.7310586 * 3.0, out[1], 1e-5);
}

test "geluInPlace" {
    var v = [_]f32{ 0.0, 1.0, -1.0 };
    activation.geluInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 0), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(activation.geluTanh(1.0), v[1], 1e-6);
    try std.testing.expectApproxEqAbs(activation.geluTanh(-1.0), v[2], 1e-6);
}

test "geluInPlace is finite for large inputs (no tanh overflow)" {
    // `tanhV` computes `(e^{2x}-1)/(e^{2x}+1)`; without the clamp, large `inner`
    // overflows `e^{2x}` to `inf` -> `inf/inf = NaN`. The gemma-4 `gelu` FFN
    // path hit this and produced all-NaN logits.
    var v = [_]f32{ 100, -100, 50, -50, 20, -20, 8, -8, 7, -7, 6, -6, 5.5, -5.5, 30, -30 };
    activation.geluInPlace(&v);
    for (v) |x| try std.testing.expect(std.math.isFinite(x));
    try std.testing.expectApproxEqAbs(@as(f32, 100), v[0], 1e-1);
    try std.testing.expectApproxEqAbs(@as(f32, 0), v[1], 1e-1);
}

test "softcapInPlace" {
    var v = [_]f32{ 0.0, 30.0, 60.0, -60.0 };
    activation.softcapInPlace(&v, 30.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(30.0 * std.math.tanh(@as(f32, 1.0)), v[1], 1e-4);
    try std.testing.expect(v[2] < 30.0);
    try std.testing.expect(v[2] > 0);
    try std.testing.expect(v[3] < 0);
}

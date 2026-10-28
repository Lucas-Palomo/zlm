const std = @import("std");
const types = @import("types.zig");
const single = @import("form/single.zig");
const dequant = @import("../quant/root.zig");
const fixtures = @import("fixtures.zig");
const TensorView = @import("../../../data/tensor_view.zig").TensorView;

test {
    _ = @import("layout_test.zig");
    _ = @import("kernels/quantized_test.zig");
    _ = @import("kernels/float_test.zig");
    _ = @import("kernels/bfloat_test.zig");
    _ = @import("form/dot_test.zig");
    _ = @import("form/single_test.zig");
    _ = @import("form/fused_test.zig");
    _ = @import("form/batched_test.zig");
}

test "every generated q4_0 candidate matches production within 1e-5" {
    const testing = std.testing;
    // 33 blocks (unroll 16 main sweep + tail) and 35 rows (tile 16 + remainder),
    // so the grid's tail takes the tiled and unrolled paths.
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 32) * 18]u8 = undefined;
    fixtures.fillQ4_0Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m); // production: width-dispatched reference
    // The integer path's oracle: the reference integer kernel.
    var want_int: [m]f32 = [_]f32{0} ** m;
    try fixtures.q4_0IntOracle(&data, n, m, &x, &want_int, 0, m);
    // The integer path's `sum(w*x) = d_w*d_x*(sum(nibble*qx) - 8*sum(qx))` is a
    // difference of larger terms, so its rounding is proportional to `sum|w*x|`
    // (the same scale the q8 test uses), not to the result. Measure that scale.
    const standard = @import("../quant/standard.zig");
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n / 32) |rb| {
            var deq: [32]f32 = undefined;
            standard.dequantQ4_0Block(data[(row * (n / 32) + rb) * 18 ..][0..18], &deq);
            for (0..32) |j| s += @abs(deq[j] * x[rb * 32 + j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.q4_0_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        if (cand.kernel.dot == .int) {
            // The integer body differs from the reference only by reduction
            // reordering (the unrolled accumulators), so the bound is far below
            // the q8 quantization error.
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want_int[i]) <= 1e-3 * scale[i] + 1e-6);
            }
        } else {
            for (0..m) |i| {
                try testing.expectApproxEqAbs(want[i], got[i], 1e-5);
            }
        }
    }
}

test "every generated q4_1 candidate matches production within the q8 bound" {
    const testing = std.testing;
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 32) * 20]u8 = undefined;
    fixtures.fillQ4_1Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_1,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m); // production: `matVecQ4_1U4`
    const standard = @import("../quant/standard.zig");
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n / 32) |rb| {
            var deq: [32]f32 = undefined;
            standard.dequantQ4_1Block(data[(row * (n / 32) + rb) * 20 ..][0..20], &deq);
            for (0..32) |j| s += @abs(deq[j] * x[rb * 32 + j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.q4_1_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        if (cand.kernel.dot == .int) {
            // The integer body differs from the f32 path only by the q8
            // activation quantization, so bound the error relative to
            // `sum|w*x|` (cf. the q4_0 integer test).
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want[i]) <= 0.02 * scale[i] + 1e-6);
            }
        } else {
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
            }
        }
    }
}

test "every generated q4_k candidate matches production" {
    const testing = std.testing;
    // 17 super-blocks (unroll 16 main sweep + tail) and 35 rows (tile 16 +
    // remainder), so the grid's tail takes the tiled and unrolled paths.
    const n: usize = 256 * 17;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 256) * 144]u8 = undefined;
    fixtures.fillQ4_KFixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_k,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    var seed: u32 = 0x2468_1357;
    for (&x, 0..) |*v, i| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        v.* = (t * 2.0 - 1.0) * (0.5 + 0.25 * @as(f32, @floatFromInt(i % 9)));
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m); // production: `matVecQ4_K`
    // The integer path's oracle: the reference integer kernel.
    var want_int: [m]f32 = [_]f32{0} ** m;
    try fixtures.q4_KIntOracle(&data, n, m, &x, &want_int, 0, m);
    // Bound the integer path's error relative to `sum|w*x|` (the q8 activation
    // quantization scale), as the q4_0 grid test does.
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n / 256) |rb| {
            var deq: [256]f32 = undefined;
            try dequant.dequantTo(.q4_k, data[(row * (n / 256) + rb) * 144 ..][0..144], &deq);
            for (0..256) |j| s += @abs(deq[j] * x[rb * 256 + j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.q4_k_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        if (cand.kernel.dot == .int) {
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want_int[i]) <= 1e-3 * scale[i] + 1e-6);
            }
        } else {
            // The f32 body differs from production only by reduction reordering
            // (the lane count and the unrolled accumulators), so the bound is
            // relative to `sum|w*x|`, far below any real decode error.
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
            }
        }
    }
}

test "the Q catalogs are the generated kernel x variant grid after the ISA prune" {
    const testing = std.testing;
    try testing.expectEqual(@as(u16, 0), @intFromEnum(types.MatVecPlan.production));

    // The grid definition is 160 combinations; the effective catalog is the
    // P1-pruned subset (fp at generated widths + int at the native width).
    try testing.expectEqual(@as(usize, 160), types.grid_total);
    try testing.expectEqual(types.fp_variant_count + types.int_variant_count, types.grid_count);
    try testing.expectEqual(types.grid_count, types.q4_0_candidates.len);
    try testing.expectEqual(types.grid_count, types.q4_1_candidates.len);
    try testing.expectEqual(types.grid_count, types.q4_k_candidates.len);
    try testing.expectEqual(types.grid_count, types.q8_0_candidates.len);
    try testing.expectEqual(types.fp_variant_count, types.q6_k_candidates.len);
    try testing.expectEqual(types.fp_variant_count, types.f32_candidates.len);
    try testing.expectEqual(types.fp_variant_count, types.f16_candidates.len);
    try testing.expectEqual(types.fp_variant_count, types.bf16_candidates.len);
    var expected_generated: usize = 0;
    inline for (types.kernel_catalog) |ks| {
        expected_generated += if (ks.dot == .fp) types.fp_variant_count else types.int_variant_count;
    }
    try testing.expectEqual(expected_generated, types.generated_count);

    try fixtures.expectCatalog(.q4_0, &types.q4_0_candidates);
    try fixtures.expectCatalog(.q4_1, &types.q4_1_candidates);
    try fixtures.expectCatalog(.q4_k, &types.q4_k_candidates);
    try fixtures.expectCatalog(.q8_0, &types.q8_0_candidates);
    try fixtures.expectCatalog(.q6_k, &types.q6_k_candidates);
    try fixtures.expectCatalog(.f32, &types.f32_candidates);
    try fixtures.expectCatalog(.f16, &types.f16_candidates);
    try fixtures.expectCatalog(.bf16, &types.bf16_candidates);

    // The families are appended in kernel-catalog order: q4_0, q4_1, q4_k,
    // q8_0, q6_k, f32, f16, bf16.
    try testing.expectEqual(
        @intFromEnum(types.q4_0_candidates[0].plan) + types.grid_count,
        @intFromEnum(types.q4_1_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.q4_1_candidates[0].plan) + types.grid_count,
        @intFromEnum(types.q4_k_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.q4_k_candidates[0].plan) + types.grid_count,
        @intFromEnum(types.q8_0_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.q8_0_candidates[0].plan) + types.grid_count,
        @intFromEnum(types.q6_k_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.q6_k_candidates[0].plan) + types.fp_variant_count,
        @intFromEnum(types.f32_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.f32_candidates[0].plan) + types.fp_variant_count,
        @intFromEnum(types.f16_candidates[0].plan),
    );
    try testing.expectEqual(
        @intFromEnum(types.f16_candidates[0].plan) + types.fp_variant_count,
        @intFromEnum(types.bf16_candidates[0].plan),
    );

    // Every generated plan is unique.
    for (types.candidates, 0..) |a, i| {
        for (types.candidates[i + 1 ..]) |b| {
            try testing.expect(a.plan != b.plan);
        }
    }
}

// The prefix cache (`runtime/13`) shares a `.zpc` across runtimes that differ
// only in `--threads`/`--ram`, on the premise that the K/V does not depend on
// how the rows are split across workers. This asserts it: a row's reduction is
// self-contained (the `parts` split is over output rows, never over K), so the
// full-range and split-range outputs are **bit-identical**. Covers the native
// and `q4_0_scale_split` layouts.
test "matvec is invariant to the row split (parts do not change the K/V)" {
    const testing = std.testing;
    const n: usize = 64;
    const m: usize = 8;
    const dims = [_]u64{ n, m };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const xi: i32 = @as(i32, @intCast(i % 29)) - 14;
        v.* = @as(f32, @floatFromInt(xi)) * 0.0625;
    }

    var q4_0_data: [m * (n / 32) * 18]u8 = undefined;
    fixtures.fillQ4_0Fixture(&q4_0_data, n, m);
    var q4_0_split: [q4_0_data.len]u8 = undefined;
    fixtures.splitQ4_0(&q4_0_data, &q4_0_split, n, m);

    const cases = [_]struct {
        data: []const u8,
        layout: @import("../../../data/tensor_view.zig").TensorLayout,
    }{
        .{ .data = &q4_0_data, .layout = .native_quant },
        .{ .data = &q4_0_split, .layout = .q4_0_scale_split },
    };
    for (cases) |c| {
        const view = TensorView{
            .name = "w",
            .dims = &dims,
            .ggml_type = .q4_0,
            .offset = 0,
            .byte_length = c.data.len,
            .mapped_slice = c.data,
            .layout = c.layout,
        };
        var full: [m]f32 = undefined;
        var split: [m]f32 = [_]f32{0} ** m;
        try single.matVecRows(&view, &x, &full, 0, m);
        const half = m / 2;
        try single.matVecRows(&view, &x, &split, 0, half);
        try single.matVecRows(&view, &x, &split, half, m);
        for (0..m) |i| try testing.expectEqual(full[i], split[i]);
    }
}

test "every generated q8_0 candidate matches production within the q8 bound" {
    const testing = std.testing;
    // 33 blocks (unroll 16 main sweep + tail) and 35 rows (tile 16 + remainder),
    // so the grid's tail takes the tiled and unrolled paths.
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 32) * 34]u8 = undefined;
    fixtures.fillQ8_0Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q8_0,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m); // production: the q8_0 fp body
    // The integer path differs from the f32 path only by the q8 activation
    // quantization, so bound its error relative to `sum|w*x|` (the q4_0/q4_1
    // grid tests use the same scale).
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n / 32) |rb| {
            var deq: [32]f32 = undefined;
            try dequant.dequantTo(.q8_0, data[(row * (n / 32) + rb) * 34 ..][0..34], &deq);
            for (0..32) |j| s += @abs(deq[j] * x[rb * 32 + j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.q8_0_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        if (cand.kernel.dot == .int) {
            for (0..m) |i| {
                try testing.expect(@abs(got[i] - want[i]) <= 0.02 * scale[i] + 1e-6);
            }
        } else {
            // The fp body differs from production only by reduction reordering
            // (the lane count and the unrolled accumulators).
            for (0..m) |i| {
                try testing.expectApproxEqAbs(want[i], got[i], 1e-5);
            }
        }
    }
}

test "every generated q6_k candidate matches production" {
    const testing = std.testing;
    const n: usize = 256 * 17;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * (n / 256) * 210]u8 = undefined;
    fixtures.fillQ6_KFixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q6_k,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    var seed: u32 = 0x2468_1357;
    for (&x, 0..) |*v, i| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        v.* = (t * 2.0 - 1.0) * (0.5 + 0.25 * @as(f32, @floatFromInt(i % 9)));
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m); // production: `Q6_K.production_fp`
    // Bound the reduction-reordering error relative to `sum|w*x|` (cf. the
    // q4_0/q4_1/q4_k grid tests).
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n / 256) |rb| {
            var deq: [256]f32 = undefined;
            try dequant.dequantTo(.q6_k, data[(row * (n / 256) + rb) * 210 ..][0..210], &deq);
            for (0..256) |j| s += @abs(deq[j] * x[rb * 256 + j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.q6_k_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        for (0..m) |i| {
            try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
        }
    }
}

test "every generated f32 candidate matches production" {
    const testing = std.testing;
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * n * 4]u8 = undefined;
    fixtures.fillF32Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m);
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n) |j| {
            const w: f32 = @bitCast(std.mem.readInt(u32, data[(row * n + j) * 4 ..][0..4], .little));
            s += @abs(w * x[j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.f32_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        for (0..m) |i| {
            try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
        }
    }
}

test "every generated f16 candidate matches production" {
    const testing = std.testing;
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * n * 2]u8 = undefined;
    fixtures.fillF16Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f16,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m);
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n) |j| {
            const h: f16 = @bitCast(std.mem.readInt(u16, data[(row * n + j) * 2 ..][0..2], .little));
            s += @abs(@as(f32, h) * x[j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.f16_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        for (0..m) |i| {
            try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
        }
    }
}

test "every generated bf16 candidate matches production" {
    const testing = std.testing;
    const n: usize = 32 * 33;
    const m: usize = 35;
    const dims = [_]u64{ n, m };
    var data: [m * n * 2]u8 = undefined;
    fixtures.fillBF16Fixture(&data, n, m);
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .bf16,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    var x: [n]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const k: i32 = @intCast((i * 5 + 1) % 13);
        v.* = @as(f32, @floatFromInt(k - 6)) * 0.25;
    }
    var want: [m]f32 = [_]f32{0} ** m;
    try single.matVecRows(&view, &x, &want, 0, m);
    var scale: [m]f32 = [_]f32{0} ** m;
    for (0..m) |row| {
        var s: f32 = 0;
        for (0..n) |j| {
            const raw = std.mem.readInt(u16, data[(row * n + j) * 2 ..][0..2], .little);
            const w: f32 = @bitCast(@as(u32, raw) << 16);
            s += @abs(w * x[j]);
        }
        scale[row] = s;
    }

    @setEvalBranchQuota(1_000_000);
    inline for (types.bf16_candidates) |cand| {
        var got: [m]f32 = [_]f32{0} ** m;
        try single.dispatchCandidate(cand, &data, n, m, &x, &got, 0, m);
        for (0..m) |i| {
            try testing.expect(@abs(got[i] - want[i]) <= 1e-5 * scale[i] + 1e-6);
        }
    }
}

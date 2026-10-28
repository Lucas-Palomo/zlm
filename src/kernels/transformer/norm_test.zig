const std = @import("std");
const norm = @import("norm.zig");
const rope_mod = @import("rope.zig");

test "rmsNormNoWeightHeads normalizes each head to unit RMS" {
    var x = [_]f32{ 3, 4, 1, 1, 0, 2, -2, 0 };
    norm.rmsNormNoWeightHeads(&x, 0, 4);
    for ([_][4]f32{ x[0..4].*, x[4..8].* }) |head| {
        var ms: f32 = 0;
        for (head) |v| ms += v * v;
        try std.testing.expectApproxEqAbs(@as(f32, 1), ms / 4, 1e-5);
    }
}

test "qkNormAndRopeHeads matches unfused qkNormHeads and rope" {
    var q_fused = [_]f32{ 3, 4, 1, 2, 0, 0, 0, 0 };
    var q_unfused = q_fused;
    const w = [_]f32{ 1, 1, 1, 1 };
    const cos_r = [_]f32{ 0.8, 0.6 };
    const sin_r = [_]f32{ 0.6, 0.8 };

    norm.qkNormHeads(&q_unfused, &w, 1e-6);
    // RoPE on head 0 and head 1
    rope_mod.ropeProportionalFromTable(q_unfused[0..4], &cos_r, &sin_r, 4, 4);
    rope_mod.ropeProportionalFromTable(q_unfused[4..8], &cos_r, &sin_r, 4, 4);

    norm.qkNormAndRopeHeads(&q_fused, &w, 1e-6, &cos_r, &sin_r, 2, 4, 4);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(q_unfused[i], q_fused[i], 1e-5);
    }
}

test "family run: qk_norm_rope_heads equals qkNormHeads then rope" {
    var got = [_]f32{ 3, 4, 1, 2, 0, 0, 0, 0 };
    var want = got;
    const w = [_]f32{ 1, 1, 1, 1 };
    const cos_r = [_]f32{ 0.8, 0.6 };
    const sin_r = [_]f32{ 0.6, 0.8 };

    norm.qkNormHeads(&want, &w, 1e-6);
    rope_mod.ropeProportionalFromTable(want[0..4], &cos_r, &sin_r, 4, 4);
    rope_mod.ropeProportionalFromTable(want[4..8], &cos_r, &sin_r, 4, 4);

    var args = norm.Args{
        .x = &got,
        .weight = &w,
        .eps = 1e-6,
        .cos_row = &cos_r,
        .sin_row = &sin_r,
        .n_heads = 2,
        .head_dim = 4,
        .n_rot = 4,
    };
    try norm.set.run(@intFromEnum(norm.AtomicId.qk_norm_rope_heads), &args);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
    }
}

test "addRmsNorm matches add then rmsNorm" {
    const a = [_]f32{ 1, -2, 0.5, 3, 0.25, -1.5, 2, 0.75 };
    const b = [_]f32{ 0.5, 1, -0.25, -1, 2, 0.5, -2, 1.25 };
    const weight = [_]f32{ 1, 0.5, 2, -1, 0.75, 1.25, -1.5, 0.25 };
    var got: [a.len]f32 = undefined;
    var tmp: [a.len]f32 = undefined;
    var want: [a.len]f32 = undefined;

    norm.addRmsNorm(&a, &b, &weight, 1e-6, &got);
    for (&tmp, a, b) |*t, av, bv| t.* = av + bv;
    norm.rmsNorm(&tmp, &weight, 1e-6, &want);

    for (0..a.len) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

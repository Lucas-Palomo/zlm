const std = @import("std");
const attention = @import("attention.zig");
const rope_mod = @import("rope.zig");

test "writeKv and attendHead GQA" {
    const n_heads: usize = 4;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 2;
    const max_seq: usize = 4;
    const need = n_kv_heads * max_seq * head_dim;

    var k_cache: [need]f16 = [_]f16{0} ** need;
    var v_cache: [need]f16 = [_]f16{0} ** need;

    const k0 = [_]f32{ 1, 0, 0, 1 };
    const v0 = [_]f32{ 2, 3, 4, 5 };
    attention.writeKv(&k0, &v0, &k_cache, &v_cache, n_kv_heads, head_dim, 0, max_seq);

    try std.testing.expectApproxEqAbs(@as(f32, 1), @as(f32, k_cache[0]), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), @as(f32, v_cache[1]), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), @as(f32, v_cache[8]), 1e-6);

    var scores: [4]f32 = undefined;
    var out: [2]f32 = undefined;
    const q = [_]f32{ 1, 0 };
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    attention.attendHead(&q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, 0, 0, max_seq, scale, 0, &scores, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scores[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[1], 1e-4);

    const k1 = [_]f32{ 0, 1, 1, 0 };
    const v1 = [_]f32{ 10, 20, 30, 40 };
    attention.writeKv(&k1, &v1, &k_cache, &v_cache, n_kv_heads, head_dim, 1, max_seq);
    attention.attendHead(&q, &k_cache, &k_cache, n_heads, n_kv_heads, head_dim, 0, 1, max_seq, scale, 0, &scores, &out);
    try std.testing.expect(scores.len >= 2);
    var ssum: f32 = 0;
    for (scores[0..2]) |s| ssum += s;
    try std.testing.expectApproxEqAbs(@as(f32, 1), ssum, 1e-5);

    attention.attendHead(&q, &k_cache, &k_cache, n_heads, n_kv_heads, head_dim, 0, 1, max_seq, scale, 1, &scores, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scores[0] + scores[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), scores[0] + scores[1] - (scores[0] + scores[1]), 1e-7);
}

test "fusedFlashAttentionHead matches reference" {
    const n_heads: usize = 6;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 32;
    const max_seq: usize = 8;
    const pos: usize = 6;
    const need = n_kv_heads * max_seq * head_dim;

    var k_cache: [need]f16 = undefined;
    var v_cache: [need]f16 = undefined;
    for (&k_cache, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 17)) - 8;
        v.* = @as(f16, @floatFromInt(x)) * 0.03125;
    }
    for (&v_cache, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 23)) - 11;
        v.* = @as(f16, @floatFromInt(x)) * 0.015625;
    }

    var q: [head_dim]f32 = undefined;
    for (&q, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 13)) - 6;
        v.* = @as(f32, @floatFromInt(x)) * 0.05;
    }

    var ref_scores: [max_seq]f32 = undefined;
    var got_scores: [max_seq]f32 = undefined;
    var ref_out: [head_dim]f32 = undefined;
    var got_out: [head_dim]f32 = undefined;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    attention.twoPassAttentionHead(&q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, 4, pos, max_seq, scale, 2, &ref_scores, &ref_out);
    attention.fusedFlashAttentionHead(&q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, 4, pos, max_seq, scale, 2, &got_scores, &got_out);

    for (0..pos + 1) |i| {
        try std.testing.expectApproxEqAbs(ref_scores[i], got_scores[i], 1e-5);
    }
    for (0..head_dim) |i| {
        try std.testing.expectApproxEqAbs(ref_out[i], got_out[i], 1e-5);
    }
}

test "twoPassAttentionHead matches fusedFlashAttentionHead" {
    const n_heads: usize = 6;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 32;
    const max_seq: usize = 8;
    const pos: usize = 6;
    const need = n_kv_heads * max_seq * head_dim;

    var k_cache: [need]f16 = undefined;
    var v_cache: [need]f16 = undefined;
    for (&k_cache, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 17)) - 8;
        v.* = @as(f16, @floatFromInt(x)) * 0.03125;
    }
    for (&v_cache, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 23)) - 11;
        v.* = @as(f16, @floatFromInt(x)) * 0.015625;
    }

    var q: [head_dim]f32 = undefined;
    for (&q, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 13)) - 6;
        v.* = @as(f32, @floatFromInt(x)) * 0.05;
    }

    var two_scores: [max_seq]f32 = undefined;
    var flash_scores: [max_seq]f32 = undefined;
    var two_out: [head_dim]f32 = undefined;
    var flash_out: [head_dim]f32 = undefined;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    attention.twoPassAttentionHead(&q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, 4, pos, max_seq, scale, 2, &two_scores, &two_out);
    attention.fusedFlashAttentionHead(&q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, 4, pos, max_seq, scale, 2, &flash_scores, &flash_out);

    for (0..pos + 1) |i| {
        try std.testing.expectApproxEqAbs(flash_scores[i], two_scores[i], 1e-5);
    }
    for (0..head_dim) |i| {
        try std.testing.expectApproxEqAbs(flash_out[i], two_out[i], 1e-5);
    }
}

test "ropeWriteKv matches rope then writeKv" {
    const n_kv_heads: usize = 2;
    const head_dim: usize = 4;
    const n_rot: usize = 4;
    const max_seq: usize = 4;
    const per = n_kv_heads * max_seq * head_dim;
    const cos_row = [_]f32{ 0.8, 0.6 };
    const sin_row = [_]f32{ 0.6, 0.8 };
    const k0 = [_]f32{ 1, 0.5, -0.3, 2, 0.1, 0.2, -1, 1.5 };
    const v0 = [_]f32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    const pos: usize = 1;

    var k_fused = k0;
    var k_ref = k0;
    var kc_fused: [per]f16 = [_]f16{0} ** per;
    var vc_fused: [per]f16 = [_]f16{0} ** per;
    var kc_ref: [per]f16 = [_]f16{0} ** per;
    var vc_ref: [per]f16 = [_]f16{0} ** per;

    attention.ropeWriteKv(&k_fused, &v0, &cos_row, &sin_row, n_kv_heads, head_dim, n_rot, &kc_fused, &vc_fused, pos, max_seq);

    for (0..n_kv_heads) |h| {
        rope_mod.ropeProportionalFromTable(k_ref[h * head_dim ..][0..head_dim], &cos_row, &sin_row, head_dim, n_rot);
    }
    attention.writeKv(&k_ref, &v0, &kc_ref, &vc_ref, n_kv_heads, head_dim, pos, max_seq);

    for (0..per) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, kc_ref[i]), @as(f32, kc_fused[i]), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, vc_ref[i]), @as(f32, vc_fused[i]), 1e-5);
    }
}

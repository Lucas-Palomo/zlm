const std = @import("std");
const vec = @import("../linear/vector.zig");
const rope_mod = @import("rope.zig");
const simd = @import("../simd.zig");

pub fn writeKv(
    k_proj: []const f32,
    v_proj: []const f32,
    k_cache: []f16,
    v_cache: []f16,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq: usize,
) void {
    const need = n_kv_heads * max_seq * head_dim;
    std.debug.assert(k_cache.len >= need and v_cache.len >= need);
    std.debug.assert(k_proj.len >= n_kv_heads * head_dim);
    std.debug.assert(v_proj.len >= n_kv_heads * head_dim);
    for (0..n_kv_heads) |h| {
        const dst = (h * max_seq + pos) * head_dim;
        const src = h * head_dim;
        for (0..head_dim) |d| {
            k_cache[dst + d] = @floatCast(k_proj[src + d]);
            v_cache[dst + d] = @floatCast(v_proj[src + d]);
        }
    }
}

pub fn ropeWriteKv(
    k: []f32,
    v: []const f32,
    cos_row: []const f32,
    sin_row: []const f32,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    k_cache: []f16,
    v_cache: []f16,
    pos: usize,
    max_seq: usize,
) void {
    std.debug.assert(k.len >= n_kv_heads * head_dim);
    for (0..n_kv_heads) |h| {
        rope_mod.ropeProportionalFromTable(k[h * head_dim ..][0..head_dim], cos_row, sin_row, head_dim, n_rot);
    }
    writeKv(k, v, k_cache, v_cache, n_kv_heads, head_dim, pos, max_seq);
}

pub fn attendHead(
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    fusedFlashAttentionHead(q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, q_head, pos, max_seq, scale, t_start, scores, out);
}

pub fn fusedFlashAttentionHead(
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    fusedFlashAttentionHeadLanes(simd.native.lanes(), q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, q_head, pos, max_seq, scale, t_start, scores, out);
}

pub fn fusedFlashAttentionHeadLanes(
    comptime lanes: usize,
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    std.debug.assert(q.len == head_dim and out.len == head_dim);
    const seq_len = pos + 1;
    std.debug.assert(scores.len >= seq_len);
    std.debug.assert(t_start <= seq_len);
    const group = n_heads / n_kv_heads;
    const kv_head = q_head / group;

    const V = @Vector(lanes, f32);
    const H = @Vector(lanes, f16);
    @memset(out, 0);
    for (0..t_start) |t| scores[t] = -std.math.inf(f32);
    if (t_start >= seq_len) {
        for (0..seq_len) |t| scores[t] = 0;
        return;
    }

    var running_max = -std.math.inf(f32);
    var running_sum: f32 = 0;
    for (t_start..seq_len) |t| {
        const koff = (kv_head * max_seq + t) * head_dim;
        const score = dotScaledLanes(lanes, q, k_cache[koff..][0..head_dim], scale);
        scores[t] = score;

        const new_max = @max(running_max, score);
        const old_scale = if (running_sum == 0) @as(f32, 0) else @exp(running_max - new_max);
        const weight = @exp(score - new_max);
        running_sum = running_sum * old_scale + weight;
        running_max = new_max;

        const voff = (kv_head * max_seq + t) * head_dim;
        var d: usize = 0;
        const old_v: V = @splat(old_scale);
        const wv: V = @splat(weight);
        while (d + lanes <= head_dim) : (d += lanes) {
            const ov: V = @as(*align(1) const V, @ptrCast(&out[d])).*;
            const vv16: H = @as(*align(1) const H, @ptrCast(&v_cache[voff + d])).*;
            const vv: V = @floatCast(vv16);
            @as(*align(1) V, @ptrCast(&out[d])).* = simd.fma(lanes, vv, wv, ov * old_v);
        }
        while (d + 4 <= head_dim) : (d += 4) {
            out[d] = out[d] * old_scale + weight * @as(f32, v_cache[voff + d]);
            out[d + 1] = out[d + 1] * old_scale + weight * @as(f32, v_cache[voff + d + 1]);
            out[d + 2] = out[d + 2] * old_scale + weight * @as(f32, v_cache[voff + d + 2]);
            out[d + 3] = out[d + 3] * old_scale + weight * @as(f32, v_cache[voff + d + 3]);
        }
        while (d < head_dim) : (d += 1) out[d] = out[d] * old_scale + weight * @as(f32, v_cache[voff + d]);
    }

    const inv_sum = 1.0 / running_sum;
    var d: usize = 0;
    const inv_v: V = @splat(inv_sum);
    while (d + lanes <= head_dim) : (d += lanes) {
        const ov: V = @as(*align(1) const V, @ptrCast(&out[d])).*;
        @as(*align(1) V, @ptrCast(&out[d])).* = ov * inv_v;
    }
    while (d < head_dim) : (d += 1) out[d] *= inv_sum;

    for (t_start..seq_len) |t| {
        scores[t] = @exp(scores[t] - running_max) * inv_sum;
    }
    for (0..t_start) |t| scores[t] = 0;
}

fn attendHeadReference(
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    attendHeadReferenceLanes(simd.native.lanes(), q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, q_head, pos, max_seq, scale, t_start, scores, out);
}

fn attendHeadReferenceLanes(
    comptime lanes: usize,
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    std.debug.assert(n_heads % n_kv_heads == 0);
    std.debug.assert(q.len == head_dim and out.len == head_dim);
    const seq_len = pos + 1;
    std.debug.assert(scores.len >= seq_len);
    std.debug.assert(t_start <= seq_len);
    const group = n_heads / n_kv_heads;
    const kv_head = q_head / group;

    const V = @Vector(lanes, f32);
    const H = @Vector(lanes, f16);
    for (t_start..seq_len) |t| {
        const koff = (kv_head * max_seq + t) * head_dim;
        var acc: V = @splat(0);
        var d: usize = 0;
        while (d + lanes <= head_dim) : (d += lanes) {
            const qv: V = @as(*align(1) const V, @ptrCast(&q[d])).*;
            const kv16: H = @as(*align(1) const H, @ptrCast(&k_cache[koff + d])).*;
            acc = simd.fma(lanes, qv, @as(V, @floatCast(kv16)), acc);
        }
        var s: f32 = @reduce(.Add, acc);
        while (d < head_dim) : (d += 1) s = @mulAdd(f32, q[d], @as(f32, k_cache[koff + d]), s);
        scores[t] = s * scale;
    }
    for (0..t_start) |t| scores[t] = -std.math.inf(f32);
    vec.softmaxInPlaceLanes(lanes, scores[0..seq_len]);

    @memset(out, 0);
    for (t_start..seq_len) |t| {
        const w = scores[t];
        const wv: V = @splat(w);
        const voff = (kv_head * max_seq + t) * head_dim;
        var d: usize = 0;
        while (d + lanes <= head_dim) : (d += lanes) {
            const ov: V = @as(*align(1) const V, @ptrCast(&out[d])).*;
            const vv16: H = @as(*align(1) const H, @ptrCast(&v_cache[voff + d])).*;
            const vv: V = @floatCast(vv16);
            const sumv = simd.fma(lanes, wv, vv, ov);
            @as(*align(1) V, @ptrCast(&out[d])).* = sumv;
        }
        while (d < head_dim) : (d += 1) out[d] = @mulAdd(f32, w, @as(f32, v_cache[voff + d]), out[d]);
    }
}

/// Two-pass attention: materialize the scaled scores for every position, apply
/// one softmax, then reduce the value cache. Equivalent to the one-pass online
/// kernel (it reuses the existing `scores` scratch); the Selector may pick it
/// when the roofline favors a compute-bound split over streaming.
pub fn twoPassAttentionHead(
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    twoPassAttentionHeadLanes(simd.native.lanes(), q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, q_head, pos, max_seq, scale, t_start, scores, out);
}

pub fn twoPassAttentionHeadLanes(
    comptime lanes: usize,
    q: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_head: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
    out: []f32,
) void {
    attendHeadReferenceLanes(lanes, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, q_head, pos, max_seq, scale, t_start, scores, out);
}

fn dotScaled(q: []const f32, k: []const f16, scale: f32) f32 {
    return dotScaledLanes(simd.native.lanes(), q, k, scale);
}

fn dotScaledLanes(comptime lanes: usize, q: []const f32, k: []const f16, scale: f32) f32 {
    const V = @Vector(lanes, f32);
    const H = @Vector(lanes, f16);
    var acc: V = @splat(0);
    var d: usize = 0;
    while (d + lanes <= q.len) : (d += lanes) {
        const qv: V = @as(*align(1) const V, @ptrCast(&q[d])).*;
        const kv16: H = @as(*align(1) const H, @ptrCast(&k[d])).*;
        acc = simd.fma(lanes, qv, @as(V, @floatCast(kv16)), acc);
    }
    var s: f32 = @reduce(.Add, acc);
    while (d < q.len) : (d += 1) s = @mulAdd(f32, q[d], @as(f32, k[d]), s);
    return s * scale;
}

const api = @import("../api.zig");

pub const AtomicId = enum(u16) { write_kv, fused_flash_head, rope_write_kv, two_pass_head };

/// Candidates the Selector may never pick (the KV-store atomics; only the head
/// attention variants are selectable).
fn neverApplicable(_: u32, _: u64, _: u64, _: u16) bool {
    return false;
}

/// Both head-attention variants apply to any head shape; the cost ranks them.
fn headApplicable(_: u32, _: u64, _: u64, _: u16) bool {
    return true;
}

/// One-pass online attention: streams K and V once (`8*m*n` bytes) with the
/// online rescale (`6*m*n` flops). `m` is the sequence length, `n` the head
/// width.
fn flashCost(_: u32, m: u64, n: u64, _: u64, _: u16) api.Cost {
    const work = std.math.mul(u64, m, n) catch std.math.maxInt(u64);
    return .{
        .bytes = std.math.mul(u64, work, 8) catch std.math.maxInt(u64),
        .flops = std.math.mul(u64, work, 6) catch std.math.maxInt(u64),
    };
}

/// Two-pass attention: the same K/V stream plus the materialized scores
/// (`2*m` f32 extra bytes) but no online rescale (`4*m*n` flops).
fn twoPassCost(_: u32, m: u64, n: u64, _: u64, _: u16) api.Cost {
    const work = std.math.mul(u64, m, n) catch std.math.maxInt(u64);
    const scores = std.math.mul(u64, m, 8) catch std.math.maxInt(u64);
    return .{
        .bytes = std.math.add(u64, std.math.mul(u64, work, 8) catch std.math.maxInt(u64), scores) catch std.math.maxInt(u64),
        .flops = std.math.mul(u64, work, 4) catch std.math.maxInt(u64),
    };
}

pub const Args = struct {
    q: []const f32 = &.{},
    k: []f32 = &.{},
    v: []const f32 = &.{},
    k_cache: []f16 = &.{},
    v_cache: []f16 = &.{},
    out: []f32 = &.{},
    scores: []f32 = &.{},
    cos_row: []const f32 = &.{},
    sin_row: []const f32 = &.{},
    n_heads: usize = 1,
    n_kv_heads: usize = 1,
    head_dim: usize = 0,
    q_head: usize = 0,
    pos: usize = 0,
    max_seq: usize = 0,
    scale: f32 = 1.0,
    t_start: usize = 0,
    n_rot: usize = 0,
    /// Chosen SIMD width (f32 lanes 4/8/16; `0` = build's native).
    simd_width: u8 = 0,
};

fn run(id: AtomicId, args: *const Args) api.KernelError!void {
    const w = simd.fromCode(args.simd_width);
    switch (id) {
        .write_kv => writeKv(args.k, args.v, args.k_cache, args.v_cache, args.n_kv_heads, args.head_dim, args.pos, args.max_seq),
        .fused_flash_head => switch (w) {
            .s4 => fusedFlashAttentionHeadLanes(4, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
            .s8 => fusedFlashAttentionHeadLanes(8, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
            .s16 => fusedFlashAttentionHeadLanes(16, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
        },
        .rope_write_kv => ropeWriteKv(args.k, args.v, args.cos_row, args.sin_row, args.n_kv_heads, args.head_dim, args.n_rot, args.k_cache, args.v_cache, args.pos, args.max_seq),
        .two_pass_head => switch (w) {
            .s4 => twoPassAttentionHeadLanes(4, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
            .s8 => twoPassAttentionHeadLanes(8, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
            .s16 => twoPassAttentionHeadLanes(16, args.q, args.k_cache, args.v_cache, args.n_heads, args.n_kv_heads, args.head_dim, args.q_head, args.pos, args.max_seq, args.scale, args.t_start, args.scores, args.out),
        },
    }
}

const kernels = [_]api.Kernel{
    .{ .id = "write_kv", .op = .attention, .atomic_id = @intFromEnum(AtomicId.write_kv), .applicable = &neverApplicable },
    .{ .id = "fused_flash_head", .op = .attention, .atomic_id = @intFromEnum(AtomicId.fused_flash_head), .complexity = .quad_flops, .fusable = true, .applicable = &headApplicable, .cost = &flashCost },
    .{ .id = "rope_write_kv", .op = .attention, .atomic_id = @intFromEnum(AtomicId.rope_write_kv), .fusable = true, .applicable = &neverApplicable },
    .{ .id = "two_pass_head", .op = .attention, .atomic_id = @intFromEnum(AtomicId.two_pass_head), .complexity = .quad_flops, .fusable = true, .applicable = &headApplicable, .cost = &twoPassCost },
};

pub const set = api.Family{
    .op = .attention,
    .name = "attention",
    .kernels = &kernels,
    .run = api.erased(AtomicId, Args, run),
};

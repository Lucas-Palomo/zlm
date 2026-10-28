const std = @import("std");
const simd = @import("../simd.zig");

fn sumSqLanes(comptime lanes: usize, x: []const f32) f32 {
    const n = x.len;
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const v: V = @as(*align(1) const V, @ptrCast(&x[i])).*;
        acc = simd.fma(lanes, v, v, acc);
    }
    var s: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) s = @mulAdd(f32, x[i], x[i], s);
    return s / @as(f32, @floatFromInt(n));
}

fn sumSq(x: []const f32) f32 {
    return sumSqLanes(simd.native.lanes(), x);
}

pub fn rmsNorm(x: []const f32, weight: []const f32, eps: f32, out: []f32) void {
    rmsNormLanes(simd.native.lanes(), x, weight, eps, out);
}

pub fn rmsNormLanes(comptime lanes: usize, x: []const f32, weight: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == weight.len and out.len == x.len);
    const inv = 1.0 / @sqrt(sumSqLanes(lanes, x) + eps);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const vx: V = @as(*align(1) const V, @ptrCast(&x[i])).*;
        const vw: V = @as(*align(1) const V, @ptrCast(&weight[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = vx * @as(V, @splat(inv)) * vw;
    }
    while (i < out.len) : (i += 1) out[i] = x[i] * inv * weight[i];
}

pub fn rmsNormInPlace(x: []f32, weight: []const f32, eps: f32) void {
    rmsNormInPlaceLanes(simd.native.lanes(), x, weight, eps);
}

pub fn rmsNormInPlaceLanes(comptime lanes: usize, x: []f32, weight: []const f32, eps: f32) void {
    std.debug.assert(x.len == weight.len);
    const inv = 1.0 / @sqrt(sumSqLanes(lanes, x) + eps);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const vx: V = @as(*align(1) const V, @ptrCast(&x[i])).*;
        const vw: V = @as(*align(1) const V, @ptrCast(&weight[i])).*;
        @as(*align(1) V, @ptrCast(&x[i])).* = vx * @as(V, @splat(inv)) * vw;
    }
    while (i < x.len) : (i += 1) x[i] = x[i] * inv * weight[i];
}

pub fn addRmsNorm(a: []const f32, b: []const f32, weight: []const f32, eps: f32, out: []f32) void {
    addRmsNormLanes(simd.native.lanes(), a, b, weight, eps, out);
}

pub fn addRmsNormLanes(comptime lanes: usize, a: []const f32, b: []const f32, weight: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(a.len == b.len and out.len == a.len and weight.len == a.len);
    const n = out.len;
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        const vb: V = @as(*align(1) const V, @ptrCast(&b[i])).*;
        const s = va + vb;
        @as(*align(1) V, @ptrCast(&out[i])).* = s;
        acc += s * s;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const s = a[i] + b[i];
        out[i] = s;
        sum = @mulAdd(f32, s, s, sum);
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(n)) + eps);
    i = 0;
    while (i + lanes <= n) : (i += lanes) {
        const vo: V = @as(*align(1) const V, @ptrCast(&out[i])).*;
        const vw: V = @as(*align(1) const V, @ptrCast(&weight[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = vo * @as(V, @splat(inv)) * vw;
    }
    while (i < n) : (i += 1) out[i] = out[i] * inv * weight[i];
}

pub fn rmsNormNoWeight(x: []const f32, eps: f32, out: []f32) void {
    rmsNormNoWeightLanes(simd.native.lanes(), x, eps, out);
}

pub fn rmsNormNoWeightLanes(comptime lanes: usize, x: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == out.len);
    const inv = 1.0 / @sqrt(sumSqLanes(lanes, x) + eps);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const vx: V = @as(*align(1) const V, @ptrCast(&x[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = vx * @as(V, @splat(inv));
    }
    while (i < out.len) : (i += 1) out[i] = x[i] * inv;
}

pub fn rmsNormNoWeightHeads(x: []f32, eps: f32, head_dim: usize) void {
    rmsNormNoWeightHeadsLanes(simd.native.lanes(), x, eps, head_dim);
}
pub fn rmsNormNoWeightHeadsLanes(comptime lanes: usize, x: []f32, eps: f32, head_dim: usize) void {
    std.debug.assert(head_dim != 0);
    std.debug.assert(x.len % head_dim == 0);
    var off: usize = 0;
    while (off < x.len) : (off += head_dim) {
        rmsNormNoWeightLanes(lanes, x[off .. off + head_dim], eps, x[off .. off + head_dim]);
    }
}
pub fn qkNormHeads(q: []f32, weight: []const f32, eps: f32) void {
    qkNormHeadsLanes(simd.native.lanes(), q, weight, eps);
}
pub fn qkNormHeadsLanes(comptime lanes: usize, q: []f32, weight: []const f32, eps: f32) void {
    std.debug.assert(q.len % weight.len == 0);
    const head_dim = weight.len;
    var off: usize = 0;
    while (off < q.len) : (off += head_dim) {
        rmsNormInPlaceLanes(lanes, q[off .. off + head_dim], weight, eps);
    }
}

pub fn qkNormAndRopeHeads(
    q: []f32,
    weight: []const f32,
    eps: f32,
    cos_row: []const f32,
    sin_row: []const f32,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
) void {
    const half = head_dim / 2;
    const rope_angles = n_rot / 2;
    std.debug.assert(q.len >= n_heads * head_dim);
    std.debug.assert(weight.len == head_dim);
    std.debug.assert(cos_row.len >= rope_angles and sin_row.len >= rope_angles);

    for (0..n_heads) |h| {
        const head = q[h * head_dim .. (h + 1) * head_dim];
        rmsNormInPlace(head, weight, eps);

        var i: usize = 0;
        while (i + 8 <= rope_angles) : (i += 8) {
            const c: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&cos_row[i])).*;
            const s: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&sin_row[i])).*;
            const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
            const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + half])).*;
            @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = x0 * c - x1 * s;
            @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + half])).* = x0 * s + x1 * c;
        }
        while (i < rope_angles) : (i += 1) {
            const x0 = head[i];
            const x1 = head[i + half];
            const c = cos_row[i];
            const s = sin_row[i];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

// --- KernelSet surface (AOT: one atomic_id per AtomicOp) ---
const api = @import("../api.zig");

pub const AtomicId = enum(u16) {
    rms_norm,
    rms_norm_in_place,
    rms_norm_no_weight,
    rms_norm_no_weight_heads,
    qk_norm_heads,
    qk_norm_rope_heads,
    add_norm,
};

pub const Args = struct {
    x: []f32,
    weight: []const f32 = &.{},
    out: []f32 = &.{},
    eps: f32 = 1e-6,
    cos_row: []const f32 = &.{},
    sin_row: []const f32 = &.{},
    n_heads: usize = 1,
    head_dim: usize = 0,
    n_rot: usize = 0,
    b: []const f32 = &.{},
    /// Chosen SIMD width (f32 lanes 4/8/16; `0` = build's native).
    simd_width: u8 = 0,
};

fn run(id: AtomicId, args: *const Args) api.KernelError!void {
    const w = simd.fromCode(args.simd_width);
    switch (id) {
        .rms_norm => switch (w) {
            .s4 => rmsNormLanes(4, args.x, args.weight, args.eps, args.out),
            .s8 => rmsNormLanes(8, args.x, args.weight, args.eps, args.out),
            .s16 => rmsNormLanes(16, args.x, args.weight, args.eps, args.out),
        },
        .rms_norm_in_place => switch (w) {
            .s4 => rmsNormInPlaceLanes(4, args.x, args.weight, args.eps),
            .s8 => rmsNormInPlaceLanes(8, args.x, args.weight, args.eps),
            .s16 => rmsNormInPlaceLanes(16, args.x, args.weight, args.eps),
        },
        .rms_norm_no_weight => switch (w) {
            .s4 => rmsNormNoWeightLanes(4, args.x, args.eps, args.out),
            .s8 => rmsNormNoWeightLanes(8, args.x, args.eps, args.out),
            .s16 => rmsNormNoWeightLanes(16, args.x, args.eps, args.out),
        },
        .rms_norm_no_weight_heads => switch (w) {
            .s4 => rmsNormNoWeightHeadsLanes(4, args.x, args.eps, args.head_dim),
            .s8 => rmsNormNoWeightHeadsLanes(8, args.x, args.eps, args.head_dim),
            .s16 => rmsNormNoWeightHeadsLanes(16, args.x, args.eps, args.head_dim),
        },
        .qk_norm_heads => switch (w) {
            .s4 => qkNormHeadsLanes(4, args.x, args.weight, args.eps),
            .s8 => qkNormHeadsLanes(8, args.x, args.weight, args.eps),
            .s16 => qkNormHeadsLanes(16, args.x, args.weight, args.eps),
        },
        .qk_norm_rope_heads => qkNormAndRopeHeads(args.x, args.weight, args.eps, args.cos_row, args.sin_row, args.n_heads, args.head_dim, args.n_rot),
        .add_norm => switch (w) {
            .s4 => addRmsNormLanes(4, args.x, args.b, args.weight, args.eps, args.out),
            .s8 => addRmsNormLanes(8, args.x, args.b, args.weight, args.eps, args.out),
            .s16 => addRmsNormLanes(16, args.x, args.b, args.weight, args.eps, args.out),
        },
    }
}

const kernels = [_]api.Kernel{
    .{ .id = "rms_norm", .op = .norm, .atomic_id = @intFromEnum(AtomicId.rms_norm) },
    .{ .id = "rms_norm_in_place", .op = .norm, .atomic_id = @intFromEnum(AtomicId.rms_norm_in_place) },
    .{ .id = "rms_norm_no_weight", .op = .norm, .atomic_id = @intFromEnum(AtomicId.rms_norm_no_weight) },
    .{ .id = "rms_norm_no_weight_heads", .op = .norm, .atomic_id = @intFromEnum(AtomicId.rms_norm_no_weight_heads) },
    .{ .id = "qk_norm_heads", .op = .norm, .atomic_id = @intFromEnum(AtomicId.qk_norm_heads), .fusable = true },
    .{ .id = "qk_norm_rope_heads", .op = .norm, .atomic_id = @intFromEnum(AtomicId.qk_norm_rope_heads), .fusable = true },
    .{ .id = "add_norm", .op = .norm, .atomic_id = @intFromEnum(AtomicId.add_norm), .fusable = true },
};

pub const set = api.Family{
    .op = .norm,
    .name = "norm",
    .kernels = &kernels,
    .run = api.erased(AtomicId, Args, run),
};

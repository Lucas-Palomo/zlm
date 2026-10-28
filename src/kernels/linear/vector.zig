const std = @import("std");
const simd = @import("../simd.zig");

pub fn dot(a: []const f32, b: []const f32) f32 {
    return dotLanes(simd.native.lanes(), a, b);
}

pub fn dotLanes(comptime lanes: usize, a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    const n = a.len;
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        const vb: V = @as(*align(1) const V, @ptrCast(&b[i])).*;
        acc = simd.fma(lanes, va, vb, acc);
    }
    var s: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) s = @mulAdd(f32, a[i], b[i], s);
    return s;
}

pub fn sum(values: []const f32) f32 {
    var acc: f32 = 0;
    for (values) |v| acc += v;
    return acc;
}

pub fn maxReduce(values: []const f32) f32 {
    std.debug.assert(values.len > 0);
    var m = values[0];
    for (values[1..]) |v| m = @max(m, v);
    return m;
}

pub fn add(a: []const f32, b: []const f32, out: []f32) void {
    addLanes(simd.native.lanes(), a, b, out);
}

pub fn addLanes(comptime lanes: usize, a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len and out.len == a.len);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        const vb: V = @as(*align(1) const V, @ptrCast(&b[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = va + vb;
    }
    while (i < out.len) : (i += 1) out[i] = a[i] + b[i];
}

pub fn addAndScale(a: []const f32, b: []const f32, scale: f32, out: []f32) void {
    addAndScaleLanes(simd.native.lanes(), a, b, scale, out);
}

pub fn addAndScaleLanes(comptime lanes: usize, a: []const f32, b: []const f32, scale: f32, out: []f32) void {
    std.debug.assert(a.len == b.len and out.len == a.len);
    const V = @Vector(lanes, f32);
    const s: V = @splat(scale);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        const vb: V = @as(*align(1) const V, @ptrCast(&b[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = (va + vb) * s;
    }
    while (i < out.len) : (i += 1) out[i] = (a[i] + b[i]) * scale;
}

pub fn scaleBy(a: []const f32, s: f32, out: []f32) void {
    scaleByLanes(simd.native.lanes(), a, s, out);
}

pub fn scaleByLanes(comptime lanes: usize, a: []const f32, s: f32, out: []f32) void {
    std.debug.assert(a.len == out.len);
    const V = @Vector(lanes, f32);
    const vs: V = @splat(s);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = va * vs;
    }
    while (i < out.len) : (i += 1) out[i] = a[i] * s;
}

pub fn mul(a: []const f32, b: []const f32, out: []f32) void {
    mulLanes(simd.native.lanes(), a, b, out);
}

pub fn mulLanes(comptime lanes: usize, a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len and out.len == a.len);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const va: V = @as(*align(1) const V, @ptrCast(&a[i])).*;
        const vb: V = @as(*align(1) const V, @ptrCast(&b[i])).*;
        @as(*align(1) V, @ptrCast(&out[i])).* = va * vb;
    }
    while (i < out.len) : (i += 1) out[i] = a[i] * b[i];
}
pub fn softmax(values: []const f32, out: []f32) void {
    softmaxLanes(simd.native.lanes(), values, out);
}

pub fn softmaxLanes(comptime lanes: usize, values: []const f32, out: []f32) void {
    std.debug.assert(values.len == out.len);
    if (values.len == 0) return;
    const V = @Vector(lanes, f32);
    const m = maxReduce(values);
    var total: f32 = 0;
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const v: V = @as(*align(1) const V, @ptrCast(&values[i])).*;
        const e: V = @exp(v - @as(V, @splat(m)));
        @as(*align(1) V, @ptrCast(&out[i])).* = e;
        total += @reduce(.Add, e);
    }
    while (i < out.len) : (i += 1) {
        const e = @exp(values[i] - m);
        out[i] = e;
        total += e;
    }
    const inv = 1.0 / total;
    var j: usize = 0;
    while (j + lanes <= out.len) : (j += lanes) {
        const e: V = @as(*align(1) const V, @ptrCast(&out[j])).*;
        @as(*align(1) V, @ptrCast(&out[j])).* = e * @as(V, @splat(inv));
    }
    while (j < out.len) : (j += 1) out[j] *= inv;
}

pub fn softmaxInPlace(values: []f32) void {
    softmaxInPlaceLanes(simd.native.lanes(), values);
}

pub fn softmaxInPlaceLanes(comptime lanes: usize, values: []f32) void {
    if (values.len == 0) return;
    const V = @Vector(lanes, f32);
    const m = maxReduce(values);
    var total: f32 = 0;
    var i: usize = 0;
    while (i + lanes <= values.len) : (i += lanes) {
        const v: V = @as(*align(1) const V, @ptrCast(&values[i])).*;
        const e: V = @exp(v - @as(V, @splat(m)));
        @as(*align(1) V, @ptrCast(&values[i])).* = e;
        total += @reduce(.Add, e);
    }
    while (i < values.len) : (i += 1) {
        values[i] = @exp(values[i] - m);
        total += values[i];
    }
    const inv = 1.0 / total;
    var j: usize = 0;
    while (j + lanes <= values.len) : (j += lanes) {
        const e: V = @as(*align(1) const V, @ptrCast(&values[j])).*;
        @as(*align(1) V, @ptrCast(&values[j])).* = e * @as(V, @splat(inv));
    }
    while (j < values.len) : (j += 1) values[j] *= inv;
}

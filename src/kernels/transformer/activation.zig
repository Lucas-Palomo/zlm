const std = @import("std");
const simd = @import("../simd.zig");

pub const ActivationKind = enum {
    gelu_tanh,
    silu,

    pub fn apply(self: ActivationKind, gate: f32, up: f32) f32 {
        return switch (self) {
            .gelu_tanh => geluTanh(gate) * up,
            .silu => (gate / (1.0 + @exp(-gate))) * up,
        };
    }
};

pub fn silu(values: []const f32, out: []f32) void {
    std.debug.assert(values.len == out.len);
    for (out, values) |*o, x| {
        o.* = x / (1.0 + @exp(-x));
    }
}
pub fn gatedSilu(gate: []const f32, up: []const f32, out: []f32) void {
    gatedSiluLanes(simd.native.lanes(), gate, up, out);
}

pub fn gatedSiluLanes(comptime lanes: usize, gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len and out.len == gate.len);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const g: V = @as(*align(1) const V, @ptrCast(&gate[i])).*;
        const u: V = @as(*align(1) const V, @ptrCast(&up[i])).*;
        const clamped = @max(@min(g, @as(V, @splat(15.0))), @as(V, @splat(-15.0)));
        const sig = @as(V, @splat(1.0)) / (@as(V, @splat(1.0)) + @exp(-clamped));
        @as(*align(1) V, @ptrCast(&out[i])).* = (g * sig) * u;
    }
    while (i < out.len) : (i += 1) {
        const g = gate[i];
        const u = up[i];
        out[i] = (g / (1.0 + @exp(-g))) * u;
    }
}

pub fn geluTanh(x: f32) f32 {
    return 0.5 * x * (1.0 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

pub fn gatedGelu(gate: []const f32, up: []const f32, out: []f32) void {
    gatedGeluLanes(simd.native.lanes(), gate, up, out);
}

pub fn gatedGeluLanes(comptime lanes: usize, gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len and out.len == gate.len);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const g: V = @as(*align(1) const V, @ptrCast(&gate[i])).*;
        const u: V = @as(*align(1) const V, @ptrCast(&up[i])).*;
        const g3 = g * g * g;
        const inner = @as(V, @splat(0.7978845608028654)) * (g + @as(V, @splat(0.044715)) * g3);
        const clamped = @max(@min(inner, @as(V, @splat(15.0))), @as(V, @splat(-15.0)));
        const e2 = @exp(clamped + clamped);
        const t = (e2 - @as(V, @splat(1.0))) / (e2 + @as(V, @splat(1.0)));
        const gelu = @as(V, @splat(0.5)) * g * (@as(V, @splat(1.0)) + t);
        @as(*align(1) V, @ptrCast(&out[i])).* = gelu * u;
    }
    while (i < out.len) : (i += 1) out[i] = geluTanh(gate[i]) * up[i];
}

/// Vector tanh via `@exp` (Zig has no `@tanh` and `std.math.tanh` is scalar):
/// `(e^{2x} - 1) / (e^{2x} + 1)`. The input is clamped to +/-15 first: for
/// `|x| > 15`, `tanh(x)` is +/-1 to f32 precision, and without the clamp
/// `e^{2x}` overflows to `inf` -> `inf/inf = NaN` (the gemma-4 `gelu` path hits
/// this for large FFN values).
inline fn tanhV(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const V = @Vector(lanes, f32);
    const clamped = @max(@min(x, @as(V, @splat(15.0))), @as(V, @splat(-15.0)));
    const e2 = @exp(clamped + clamped);
    return (e2 - @as(V, @splat(1.0))) / (e2 + @as(V, @splat(1.0)));
}

pub fn softcapInPlace(values: []f32, cap: f32) void {
    const lanes = comptime @intFromEnum(simd.native);
    const V = @Vector(lanes, f32);
    const cv: V = @splat(cap);
    var i: usize = 0;
    while (i + lanes <= values.len) : (i += lanes) {
        const v: V = @as(*align(1) const V, @ptrCast(&values[i])).*;
        @as(*align(1) V, @ptrCast(&values[i])).* = cv * tanhV(lanes, v / cv);
    }
    while (i < values.len) : (i += 1) values[i] = cap * std.math.tanh(values[i] / cap);
}

pub fn geluInPlace(values: []f32) void {
    const lanes = comptime @intFromEnum(simd.native);
    const V = @Vector(lanes, f32);
    var i: usize = 0;
    while (i + lanes <= values.len) : (i += lanes) {
        const v: V = @as(*align(1) const V, @ptrCast(&values[i])).*;
        const inner = @as(V, @splat(0.7978845608028654)) * (v + @as(V, @splat(0.044715)) * v * v * v);
        @as(*align(1) V, @ptrCast(&values[i])).* = @as(V, @splat(0.5)) * v * (@as(V, @splat(1.0)) + tanhV(lanes, inner));
    }
    while (i < values.len) : (i += 1) values[i] = geluTanh(values[i]);
}

const api = @import("../api.zig");

pub const AtomicId = enum(u16) { gated_silu, gated_gelu, gelu };

pub const Args = struct { gate: []const f32 = &.{}, up: []const f32 = &.{}, out: []f32 = &.{}, simd_width: u8 = 0 };

fn run(id: AtomicId, args: *const Args) api.KernelError!void {
    const w = simd.fromCode(args.simd_width);
    switch (id) {
        .gated_silu => switch (w) {
            .s4 => gatedSiluLanes(4, args.gate, args.up, args.out),
            .s8 => gatedSiluLanes(8, args.gate, args.up, args.out),
            .s16 => gatedSiluLanes(16, args.gate, args.up, args.out),
        },
        .gated_gelu => switch (w) {
            .s4 => gatedGeluLanes(4, args.gate, args.up, args.out),
            .s8 => gatedGeluLanes(8, args.gate, args.up, args.out),
            .s16 => gatedGeluLanes(16, args.gate, args.up, args.out),
        },
        .gelu => geluInPlace(args.out),
    }
}

const kernels = [_]api.Kernel{
    .{ .id = "gated_silu", .op = .activation, .atomic_id = @intFromEnum(AtomicId.gated_silu), .fusable = true },
    .{ .id = "gated_gelu", .op = .activation, .atomic_id = @intFromEnum(AtomicId.gated_gelu), .fusable = true },
    .{ .id = "gelu", .op = .activation, .atomic_id = @intFromEnum(AtomicId.gelu) },
};

pub const set = api.Family{
    .op = .activation,
    .name = "activation",
    .kernels = &kernels,
    .run = api.erased(AtomicId, Args, run),
};

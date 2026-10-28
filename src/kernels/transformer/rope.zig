const std = @import("std");
const simd = @import("../simd.zig");

pub fn rope(head: []f32, pos: u32, freq_base: f32) void {
    std.debug.assert(head.len % 2 == 0);
    const half = head.len / 2;
    const dim: f32 = @floatFromInt(head.len);
    const t: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i + 8 <= half) : (i += 8) {
        var fi: @Vector(8, f32) = undefined;
        inline for (0..8) |k| fi[k] = @floatFromInt(i + k);
        const log_base = @log(@as(@Vector(8, f32), @splat(freq_base)));
        const inv = @exp((@as(@Vector(8, f32), @splat(-2.0)) * fi / @as(@Vector(8, f32), @splat(dim))) * log_base);
        const angle = @as(@Vector(8, f32), @splat(t)) * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
        const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + half])).*;
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = simd.fma(8, x1, -s, x0 * c);
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + half])).* = simd.fma(8, x1, c, x0 * s);
    }
    while (i < half) : (i += 1) {
        const fi: f32 = @floatFromInt(i);
        const inv = std.math.pow(f32, freq_base, -2.0 * fi / dim);
        const angle = t * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0 = head[i];
        const x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

pub const RopeTable = struct {
    angles: usize,
    max_pos: usize,
    freq_base: f32,
    dim: f32,
    inv_freq: []f32,
    cos_tab: []f32,
    sin_tab: []f32,
    filled: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        angles: usize,
        max_pos: usize,
        freq_base: f32,
        dim: f32,
    ) !RopeTable {
        std.debug.assert(angles > 0 and max_pos > 0);
        const inv_freq = try allocator.alloc(f32, angles);
        errdefer allocator.free(inv_freq);
        for (inv_freq, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = std.math.pow(f32, freq_base, -2.0 * fi / dim);
        }
        const cos_tab = try allocator.alloc(f32, max_pos * angles);
        errdefer allocator.free(cos_tab);
        const sin_tab = try allocator.alloc(f32, max_pos * angles);
        errdefer allocator.free(sin_tab);
        return .{
            .angles = angles,
            .max_pos = max_pos,
            .freq_base = freq_base,
            .dim = dim,
            .inv_freq = inv_freq,
            .cos_tab = cos_tab,
            .sin_tab = sin_tab,
            .filled = 0,
        };
    }

    pub fn deinit(self: *RopeTable, allocator: std.mem.Allocator) void {
        allocator.free(self.inv_freq);
        allocator.free(self.cos_tab);
        allocator.free(self.sin_tab);
    }

    pub fn ensure(self: *RopeTable, pos: usize) void {
        std.debug.assert(pos < self.max_pos);
        while (self.filled <= pos) {
            const t: f32 = @floatFromInt(self.filled);
            const base = self.filled * self.angles;
            for (0..self.angles) |i| {
                const angle = t * self.inv_freq[i];
                self.cos_tab[base + i] = @cos(angle);
                self.sin_tab[base + i] = @sin(angle);
            }
            self.filled += 1;
        }
    }

    pub fn cosRow(self: *const RopeTable, pos: usize) []const f32 {
        std.debug.assert(pos < self.filled);
        return self.cos_tab[pos * self.angles ..][0..self.angles];
    }

    pub fn sinRow(self: *const RopeTable, pos: usize) []const f32 {
        std.debug.assert(pos < self.filled);
        return self.sin_tab[pos * self.angles ..][0..self.angles];
    }
};

pub fn ropeFromTable(head: []f32, cos_row: []const f32, sin_row: []const f32) void {
    std.debug.assert(head.len % 2 == 0);
    const half = head.len / 2;
    std.debug.assert(cos_row.len >= half and sin_row.len >= half);
    var i: usize = 0;
    while (i + 8 <= half) : (i += 8) {
        const c: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&cos_row[i])).*;
        const s: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&sin_row[i])).*;
        const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
        const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + half])).*;
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = simd.fma(8, x1, -s, x0 * c);
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + half])).* = simd.fma(8, x1, c, x0 * s);
    }
    while (i < half) : (i += 1) {
        const x0 = head[i];
        const x1 = head[i + half];
        const c = cos_row[i];
        const s = sin_row[i];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

pub fn ropeInterleaved(head: []f32, pos: u32, freq_base: f32) void {
    std.debug.assert(head.len % 2 == 0);
    const n = head.len;
    const dim: f32 = @floatFromInt(n);
    const t: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        var fi: @Vector(8, f32) = undefined;
        inline for (0..8) |k| fi[k] = @floatFromInt((i + 2 * k) / 2);
        const log_base = @log(@as(@Vector(8, f32), @splat(freq_base)));
        const inv = @exp((@as(@Vector(8, f32), @splat(-2.0)) * fi / @as(@Vector(8, f32), @splat(dim))) * log_base);
        const angle = @as(@Vector(8, f32), @splat(t)) * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
        const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + 8])).*;
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = simd.fma(8, x1, -s, x0 * c);
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + 8])).* = simd.fma(8, x1, c, x0 * s);
    }
    while (i < n) : (i += 2) {
        const fi: f32 = @floatFromInt(i / 2);
        const inv = std.math.pow(f32, freq_base, -2.0 * fi / dim);
        const angle = t * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0 = head[i];
        const x1 = head[i + 1];
        head[i] = x0 * c - x1 * s;
        head[i + 1] = x0 * s + x1 * c;
    }
}
pub fn ropeProportional(
    head: []f32,
    pos: u32,
    freq_base: f32,
    head_dim: usize,
    rope_dim: usize,
) void {
    std.debug.assert(head.len >= head_dim);
    std.debug.assert(head_dim % 2 == 0 and rope_dim % 2 == 0);
    std.debug.assert(rope_dim <= head_dim);
    const half = head_dim / 2;
    const rope_angles = rope_dim / 2;
    const dim: f32 = @floatFromInt(head_dim);
    const t: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i + 8 <= rope_angles) : (i += 8) {
        var fi: @Vector(8, f32) = undefined;
        inline for (0..8) |k| fi[k] = @floatFromInt(i + k);
        const log_base = @log(@as(@Vector(8, f32), @splat(freq_base)));
        const inv = @exp((@as(@Vector(8, f32), @splat(-2.0)) * fi / @as(@Vector(8, f32), @splat(dim))) * log_base);
        const angle = @as(@Vector(8, f32), @splat(t)) * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
        const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + half])).*;
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = simd.fma(8, x1, -s, x0 * c);
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + half])).* = simd.fma(8, x1, c, x0 * s);
    }
    while (i < rope_angles) : (i += 1) {
        const fi: f32 = @floatFromInt(i);
        const inv = std.math.pow(f32, freq_base, -2.0 * fi / dim);
        const angle = t * inv;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0 = head[i];
        const x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

pub fn ropeProportionalFromTable(
    head: []f32,
    cos_row: []const f32,
    sin_row: []const f32,
    head_dim: usize,
    rope_dim: usize,
) void {
    std.debug.assert(head.len >= head_dim);
    std.debug.assert(head_dim % 2 == 0 and rope_dim % 2 == 0);
    std.debug.assert(rope_dim <= head_dim);
    const half = head_dim / 2;
    const rope_angles = rope_dim / 2;
    std.debug.assert(cos_row.len >= rope_angles and sin_row.len >= rope_angles);
    var i: usize = 0;
    while (i + 8 <= rope_angles) : (i += 8) {
        const c: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&cos_row[i])).*;
        const s: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&sin_row[i])).*;
        const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i])).*;
        const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&head[i + half])).*;
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i])).* = simd.fma(8, x1, -s, x0 * c);
        @as(*align(1) @Vector(8, f32), @ptrCast(&head[i + half])).* = simd.fma(8, x1, c, x0 * s);
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

// --- KernelSet surface (AOT: one atomic_id per AtomicOp) ---
const api = @import("../api.zig");

pub const AtomicId = enum(u16) {
    rope,
    rope_from_table,
    rope_interleaved,
    rope_proportional,
    rope_proportional_from_table,
};

pub const Args = struct {
    head: []f32,
    pos: u32 = 0,
    freq_base: f32 = 1e4,
    cos_row: []const f32 = &.{},
    sin_row: []const f32 = &.{},
    head_dim: usize = 0,
    n_rot: usize = 0,
};

fn run(id: AtomicId, args: *const Args) api.KernelError!void {
    switch (id) {
        .rope => rope(args.head, args.pos, args.freq_base),
        .rope_from_table => ropeFromTable(args.head, args.cos_row, args.sin_row),
        .rope_interleaved => ropeInterleaved(args.head, args.pos, args.freq_base),
        .rope_proportional => ropeProportional(args.head, args.pos, args.freq_base, args.head_dim, args.n_rot),
        .rope_proportional_from_table => ropeProportionalFromTable(args.head, args.cos_row, args.sin_row, args.head_dim, args.n_rot),
    }
}

const kernels = [_]api.Kernel{
    .{ .id = "rope", .op = .rope, .atomic_id = @intFromEnum(AtomicId.rope), .fusable = true },
    .{ .id = "rope_from_table", .op = .rope, .atomic_id = @intFromEnum(AtomicId.rope_from_table), .fusable = true },
    .{ .id = "rope_interleaved", .op = .rope, .atomic_id = @intFromEnum(AtomicId.rope_interleaved), .fusable = true },
    .{ .id = "rope_proportional", .op = .rope, .atomic_id = @intFromEnum(AtomicId.rope_proportional), .fusable = true },
    .{ .id = "rope_proportional_from_table", .op = .rope, .atomic_id = @intFromEnum(AtomicId.rope_proportional_from_table), .fusable = true },
};

pub const set = api.Family{
    .op = .rope,
    .name = "rope",
    .kernels = &kernels,
    .run = api.erased(AtomicId, Args, run),
};

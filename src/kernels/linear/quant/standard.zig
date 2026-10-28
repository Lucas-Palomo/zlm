const std = @import("std");
const types = @import("types.zig");
const KernelError = types.KernelError;
const readF16 = types.readF16;
const QK4_0 = types.QK4_0;
const QK4_1 = types.QK4_1;
const QK8_0 = types.QK8_0;
const Q4_1_BLOCK = types.Q4_1_BLOCK;

pub fn dequantQ8_0(data: []const u8, out: []f32) KernelError!void {
    const n_blocks: usize = out.len / QK8_0;
    const rem: usize = out.len % QK8_0;
    const need: usize = n_blocks * 34 + (if (rem > 0) @as(usize, 34) else 0);
    if (data.len < need) return error.TruncatedData;

    var bi: usize = 0;
    var pos: usize = 0;
    while (bi < n_blocks) : (bi += 1) {
        const block = data[pos .. pos + 34];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const scale: @Vector(16, f32) = @splat(d);
        const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
        const q0: @Vector(16, f32) = @floatFromInt(@as(*align(1) const @Vector(16, i8), @ptrCast(qs)).*);
        const q1: @Vector(16, f32) = @floatFromInt(@as(*align(1) const @Vector(16, i8), @ptrCast(&qs[16])).*);
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK8_0])).* = q0 * scale;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK8_0 + 16])).* = q1 * scale;
        pos += 34;
    }

    if (rem > 0) {
        var tmp: [QK8_0]f32 = undefined;
        const block = data[pos .. pos + 34];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const scale: @Vector(16, f32) = @splat(d);
        const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
        const q0: @Vector(16, f32) = @floatFromInt(@as(*align(1) const @Vector(16, i8), @ptrCast(qs)).*);
        const q1: @Vector(16, f32) = @floatFromInt(@as(*align(1) const @Vector(16, i8), @ptrCast(&qs[16])).*);
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[0])).* = q0 * scale;
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[16])).* = q1 * scale;
        @memcpy(out[n_blocks * QK8_0 ..], tmp[0..rem]);
    }
}

pub fn dequantQ4_0(data: []const u8, out: []f32) KernelError!void {
    const n_blocks: usize = out.len / QK4_0;
    const rem: usize = out.len % QK4_0;
    const need: usize = n_blocks * 18 + (if (rem > 0) @as(usize, 18) else 0);
    if (data.len < need) return error.TruncatedData;

    const lo_mask: @Vector(16, u8) = @splat(0x0F);
    const hi_shift: @Vector(16, u8) = @splat(4);
    const bias: @Vector(16, i16) = @splat(8);

    var bi: usize = 0;
    var pos: usize = 0;
    while (bi < n_blocks) : (bi += 1) {
        const block = data[pos .. pos + 18];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(block[2..18].ptr)).*;
        const lo_i: @Vector(16, i16) = @as(@Vector(16, i16), bytes & lo_mask) - bias;
        const hi_i: @Vector(16, i16) = @as(@Vector(16, i16), bytes >> hi_shift) - bias;
        const scale: @Vector(16, f32) = @splat(d);
        const lo: @Vector(16, f32) = @as(@Vector(16, f32), @floatFromInt(lo_i)) * scale;
        const hi: @Vector(16, f32) = @as(@Vector(16, f32), @floatFromInt(hi_i)) * scale;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK4_0])).* = lo;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK4_0 + 16])).* = hi;
        pos += 18;
    }

    if (rem > 0) {
        var tmp: [QK4_0]f32 = undefined;
        const block = data[pos .. pos + 18];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(block[2..18].ptr)).*;
        const lo_i: @Vector(16, i16) = @as(@Vector(16, i16), bytes & lo_mask) - bias;
        const hi_i: @Vector(16, i16) = @as(@Vector(16, i16), bytes >> hi_shift) - bias;
        const scale: @Vector(16, f32) = @splat(d);
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[0])).* = @as(@Vector(16, f32), @floatFromInt(lo_i)) * scale;
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[16])).* = @as(@Vector(16, f32), @floatFromInt(hi_i)) * scale;
        @memcpy(out[n_blocks * QK4_0 ..], tmp[0..rem]);
    }
}

pub fn dequantQ4_0Block(block: []const u8, out: *[QK4_0]f32) void {
    const d: f32 = @floatCast(readF16(block[0..2]));
    const qs = block[2..18];
    for (0..QK4_0 / 2) |j| {
        const x0: i32 = @as(i32, qs[j] & 0x0F) - 8;
        const x1: i32 = @as(i32, qs[j] >> 4) - 8;
        out[j] = d * @as(f32, @floatFromInt(x0));
        out[QK4_0 / 2 + j] = d * @as(f32, @floatFromInt(x1));
    }
}

pub fn dequantQ4_1(data: []const u8, out: []f32) KernelError!void {
    const n_blocks: usize = out.len / QK4_1;
    const rem: usize = out.len % QK4_1;
    const need: usize = n_blocks * Q4_1_BLOCK + (if (rem > 0) @as(usize, Q4_1_BLOCK) else 0);
    if (data.len < need) return error.TruncatedData;

    const lo_mask: @Vector(16, u8) = @splat(0x0F);
    const hi_shift: @Vector(16, u8) = @splat(4);

    var bi: usize = 0;
    var pos: usize = 0;
    while (bi < n_blocks) : (bi += 1) {
        const block = data[pos .. pos + Q4_1_BLOCK];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const m: f32 = @floatCast(readF16(block[2..4]));
        const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(block[4..20].ptr)).*;
        const lo_i: @Vector(16, u16) = @as(@Vector(16, u16), bytes & lo_mask);
        const hi_i: @Vector(16, u16) = @as(@Vector(16, u16), bytes >> hi_shift);
        const scale: @Vector(16, f32) = @splat(d);
        const minv: @Vector(16, f32) = @splat(m);
        const lo: @Vector(16, f32) = @as(@Vector(16, f32), @floatFromInt(lo_i)) * scale + minv;
        const hi: @Vector(16, f32) = @as(@Vector(16, f32), @floatFromInt(hi_i)) * scale + minv;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK4_1])).* = lo;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[bi * QK4_1 + 16])).* = hi;
        pos += Q4_1_BLOCK;
    }

    if (rem > 0) {
        var tmp: [QK4_1]f32 = undefined;
        const block = data[pos .. pos + Q4_1_BLOCK];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const m: f32 = @floatCast(readF16(block[2..4]));
        const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(block[4..20].ptr)).*;
        const lo_i: @Vector(16, u16) = @as(@Vector(16, u16), bytes & lo_mask);
        const hi_i: @Vector(16, u16) = @as(@Vector(16, u16), bytes >> hi_shift);
        const scale: @Vector(16, f32) = @splat(d);
        const minv: @Vector(16, f32) = @splat(m);
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[0])).* = @as(@Vector(16, f32), @floatFromInt(lo_i)) * scale + minv;
        @as(*align(1) @Vector(16, f32), @ptrCast(&tmp[16])).* = @as(@Vector(16, f32), @floatFromInt(hi_i)) * scale + minv;
        @memcpy(out[n_blocks * QK4_1 ..], tmp[0..rem]);
    }
}

pub fn dequantQ4_1Block(block: []const u8, out: *[QK4_1]f32) void {
    const d: f32 = @floatCast(readF16(block[0..2]));
    const m: f32 = @floatCast(readF16(block[2..4]));
    const qs = block[4..20];
    for (0..QK4_1 / 2) |j| {
        const x0: f32 = @floatFromInt(@as(u8, qs[j] & 0x0F));
        const x1: f32 = @floatFromInt(@as(u8, qs[j] >> 4));
        out[j] = x0 * d + m;
        out[QK4_1 / 2 + j] = x1 * d + m;
    }
}

pub fn dequantEmbeddingQ8_0(data: []const u8, out: []f32) KernelError!void {
    return dequantQ8_0(data, out);
}

pub fn dequantEmbeddingQ4_0(data: []const u8, out: []f32) KernelError!void {
    return dequantQ4_0(data, out);
}

pub fn dequantEmbeddingQ4_1(data: []const u8, out: []f32) KernelError!void {
    return dequantQ4_1(data, out);
}

pub fn encodeQ8_0Block(d: f16, qs: *const [32]i8) [34]u8 {
    var out: [34]u8 = undefined;
    std.mem.writeInt(u16, out[0..2], @bitCast(d), .little);
    for (qs, 2..) |q, i| out[i] = @bitCast(q);
    return out;
}

pub fn encodeQ4_0Block(d: f16, nibbles: *const [16]u8) [18]u8 {
    var out: [18]u8 = undefined;
    std.mem.writeInt(u16, out[0..2], @bitCast(d), .little);
    @memcpy(out[2..18], nibbles);
    return out;
}

pub fn encodeQ4_1Block(d: f16, m: f16, nibbles: *const [16]u8) [20]u8 {
    var out: [20]u8 = undefined;
    std.mem.writeInt(u16, out[0..2], @bitCast(d), .little);
    std.mem.writeInt(u16, out[2..4], @bitCast(m), .little);
    @memcpy(out[4..20], nibbles);
    return out;
}

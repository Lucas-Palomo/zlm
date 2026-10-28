const std = @import("std");
const types = @import("types.zig");
const KernelError = types.KernelError;

pub fn dequantF32(data: []const u8, out: []f32) KernelError!void {
    if (data.len < out.len * 4) return error.TruncatedData;
    for (out, 0..) |*v, i| {
        const bytes = data[i * 4 ..][0..4];
        v.* = @bitCast(std.mem.readInt(u32, bytes, .little));
    }
}

pub fn dequantF16(data: []const u8, out: []f32) KernelError!void {
    if (data.len < out.len * 2) return error.TruncatedData;
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        const raw: @Vector(8, f16) = @as(*align(1) const @Vector(8, f16), @ptrCast(data[i * 2 ..].ptr)).*;
        const vals: @Vector(8, f32) = @floatCast(raw);
        @as(*align(1) @Vector(8, f32), @ptrCast(&out[i])).* = vals;
    }
    while (i < out.len) : (i += 1) {
        const raw = std.mem.readInt(u16, data[i * 2 ..][0..2], .little);
        const h: f16 = @bitCast(raw);
        out[i] = @floatCast(h);
    }
}

pub fn dequantBF16(data: []const u8, out: []f32) KernelError!void {
    if (data.len < out.len * 2) return error.TruncatedData;
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        const raw: @Vector(8, u16) = @as(*align(1) const @Vector(8, u16), @ptrCast(data[i * 2 ..].ptr)).*;
        const widened: @Vector(8, u32) = @as(@Vector(8, u32), raw) << @as(@Vector(8, u32), @splat(16));
        const vals: @Vector(8, f32) = @bitCast(widened);
        @as(*align(1) @Vector(8, f32), @ptrCast(&out[i])).* = vals;
    }
    while (i < out.len) : (i += 1) {
        const raw = std.mem.readInt(u16, data[i * 2 ..][0..2], .little);
        out[i] = @bitCast(@as(u32, raw) << 16);
    }
}

pub fn dequantEmbeddingBF16(data: []const u8, out: []f32) KernelError!void {
    return dequantBF16(data, out);
}

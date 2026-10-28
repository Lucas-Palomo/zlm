const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../../../simd.zig");
const layout = @import("../layout.zig");
const floats = @import("../../quant/floats.zig");
const api = @import("../../../api.zig");

pub const KernelError = types.KernelError;

const Activation = layout.Activation;

/// The raw-float weight family: F32 and F16. Both have no quant block, so the
/// row is the unit; the sweep still iterates a fixed 32-element chunk
/// (`block_elems`), chosen because it is the Q families' granularity and a
/// natural SIMD multiple of every 4/8/16-lane width (2-8 fma per chunk).
/// `fpBlock` folds `w*x` into the `lanes`-wide accumulator; `corr` is unused
/// (there is no min term). Both are `fp`-only: no nibble -> no `int`.
/// f32: 4 bytes/element; `fpBlock` reads a 32-element f32 chunk.
pub const F32 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = block_elems * 4;
    pub const has_min: bool = false;
    pub const production_fp = matVecF32;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const V = @Vector(lanes, f32);
        var i: usize = 0;
        while (i + lanes <= block_elems) : (i += lanes) {
            const w: V = @as(*align(1) const V, @ptrCast(block[i * 4 ..].ptr)).*;
            const xv: V = @as(*align(1) const V, @ptrCast(act.x[rb * block_elems + i ..].ptr)).*;
            acc.* = simd.fma(lanes, w, xv, acc.*);
        }
    }

    /// The L1 block dequant: decode the block's 32 f32 elements, at the same
    /// block granularity the sweep's `fpBlock` uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return floats.dequantF32(block, out);
    }
};

/// f16: 2 bytes/element; `fpBlock` reads a 32-element f16 chunk and widens.
pub const F16 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = block_elems * 2;
    pub const has_min: bool = false;
    pub const production_fp = matVecF16;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const V = @Vector(lanes, f32);
        var i: usize = 0;
        while (i + lanes <= block_elems) : (i += lanes) {
            const raw: @Vector(lanes, f16) = @as(*align(1) const @Vector(lanes, f16), @ptrCast(block[i * 2 ..].ptr)).*;
            const w: V = @floatCast(raw);
            const xv: V = @as(*align(1) const V, @ptrCast(act.x[rb * block_elems + i ..].ptr)).*;
            acc.* = simd.fma(lanes, w, xv, acc.*);
        }
    }

    /// The L1 block dequant: decode the block's 32 f16 elements to f32, at the
    /// same block granularity the sweep's `fpBlock` uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return floats.dequantF16(block, out);
    }
};

pub fn matVecF32(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (data.len < n * m * 4) return error.TruncatedData;
    for (begin..end) |i| {
        const row_bytes = data[i * n * 4 ..][0 .. n * 4];
        var acc: f32 = 0;
        for (0..n) |j| {
            const w: f32 = @bitCast(std.mem.readInt(u32, row_bytes[j * 4 ..][0..4], .little));
            acc += w * x[j];
        }
        out[i] = acc;
    }
}

pub fn matVecF16(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (data.len < n * m * 2) return error.TruncatedData;
    for (begin..end) |i| {
        const row_bytes = data[i * n * 2 ..][0 .. n * 2];
        var acc: f32 = 0;
        for (0..n) |j| {
            const raw = std.mem.readInt(u16, row_bytes[j * 2 ..][0..2], .little);
            const h: f16 = @bitCast(raw);
            const w: f32 = @floatCast(h);
            acc += w * x[j];
        }
        out[i] = acc;
    }
}

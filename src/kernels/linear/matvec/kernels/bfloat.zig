const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../../../simd.zig");
const layout = @import("../layout.zig");
const floats = @import("../../quant/floats.zig");
const api = @import("../../../api.zig");

pub const KernelError = types.KernelError;

const Activation = layout.Activation;

/// The raw-bfloat16 weight family (BF16). It has no quant block, so the row is
/// the unit and the sweep's inner granularity is the same fixed 32-element chunk
/// the F family uses (uniform across families, a natural SIMD multiple of every
/// 4/8/16-lane width). bf16 is the top 16 bits of an f32, so `fpBlock` widens
/// with a 16-bit shift. `fp`-only: no nibble -> no `int`.
/// bf16: 2 bytes/element (`u16 << 16`); `fpBlock` reads a 32-element chunk.
pub const BF16 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = block_elems * 2;
    pub const has_min: bool = false;
    pub const production_fp = matVecBF16;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const V = @Vector(lanes, f32);
        var i: usize = 0;
        while (i + lanes <= block_elems) : (i += lanes) {
            const raw: @Vector(lanes, u16) = @as(*align(1) const @Vector(lanes, u16), @ptrCast(block[i * 2 ..].ptr)).*;
            const w: V = @bitCast(@as(@Vector(lanes, u32), raw) << @as(@Vector(lanes, u32), @splat(16)));
            const xv: V = @as(*align(1) const V, @ptrCast(act.x[rb * block_elems + i ..].ptr)).*;
            acc.* = simd.fma(lanes, w, xv, acc.*);
        }
    }

    /// The L1 block dequant: decode the block's 32 bf16 elements to f32, at the
    /// same block granularity the sweep's `fpBlock` uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return floats.dequantBF16(block, out);
    }
};

pub fn matVecBF16(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (data.len < n * m * 2) return error.TruncatedData;
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        const row_bytes = data[i * n * 2 ..][0 .. n * 2];
        var acc0: V = @splat(0);
        var acc1: V = @splat(0);
        var j: usize = 0;
        while (j + 16 <= n) : (j += 16) {
            const raw0: @Vector(8, u16) = @as(*align(1) const @Vector(8, u16), @ptrCast(row_bytes[j * 2 ..])).*;
            const raw1: @Vector(8, u16) = @as(*align(1) const @Vector(8, u16), @ptrCast(row_bytes[j * 2 + 16 ..])).*;
            const w0: V = @bitCast(@as(@Vector(8, u32), raw0) << @as(@Vector(8, u32), @splat(16)));
            const w1: V = @bitCast(@as(@Vector(8, u32), raw1) << @as(@Vector(8, u32), @splat(16)));
            acc0 += w0 * @as(*align(1) const V, @ptrCast(x[j..])).*;
            acc1 += w1 * @as(*align(1) const V, @ptrCast(x[j + 8 ..])).*;
        }
        var acc: f32 = @reduce(.Add, acc0) + @reduce(.Add, acc1);
        while (j < n) : (j += 1) {
            const raw = std.mem.readInt(u16, row_bytes[j * 2 ..][0..2], .little);
            const w: f32 = @bitCast(@as(u32, raw) << 16);
            acc += w * x[j];
        }
        out[i] = acc;
    }
}

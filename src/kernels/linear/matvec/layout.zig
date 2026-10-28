const std = @import("std");
const types = @import("types.zig");
const dequant = @import("../quant/root.zig");
const simd = @import("../../simd.zig");

pub const KernelError = types.KernelError;
pub const max_activation_cache_blocks = types.max_activation_cache_blocks;

/// The per-call activation context the sweep prepares once. `fp` kernels read
/// `x` (and `xsum`, the per-32-chunk sums, when they have a min term); `int`
/// kernels read the q8-quantized `qx`, its per-32 scale `dx` and integer sum
/// `sums`.
pub const Activation = struct {
    x: []const f32,
    xsum: []const f32 = &.{},
    qx: []const i8 = &.{},
    dx: []const f32 = &.{},
    sums: []const i32 = &.{},
};

/// The unscaled vector partial of a q4_0 block (`sum (nibble-8)*x`, no block
/// scale, no horizontal reduce). The matvec accumulates this across blocks and
/// reduces once per row (see `q8_0BlockVec`).
/// Expand a q4_0 block's 16 packed bytes to 32 signed nibbles (element j =
/// low nibble of byte j, element j+16 = high nibble, both biased by -8).
/// Vectorized (16 bytes at a time), so it is not a scalar loop.
pub fn q4_0Expand(block: []const u8) [32]i8 {
    const qs: *align(1) const [16]u8 = @ptrCast(block[2..18].ptr);
    const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(qs)).*;
    const lo = bytes & @as(@Vector(16, u8), @splat(0x0F));
    const hi = bytes >> @as(@Vector(16, u8), @splat(4));
    const bias: @Vector(16, i8) = @splat(8);
    var q: [32]i8 align(16) = undefined;
    @as(*@Vector(16, i8), @ptrCast(&q[0])).* = @as(@Vector(16, i8), @bitCast(lo)) - bias;
    @as(*@Vector(16, i8), @ptrCast(&q[16])).* = @as(@Vector(16, i8), @bitCast(hi)) - bias;
    return q;
}

pub fn q4_0BlockVec(comptime lanes: usize, block: []const u8, xb: *align(1) const [32]f32) @Vector(lanes, f32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const q = q4_0Expand(block);
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= 32) : (i += lanes) {
        const qv: @Vector(lanes, i8) = @as(*align(1) const @Vector(lanes, i8), @ptrCast(&q[i])).*;
        const xv: V = @as(*align(1) const V, @ptrCast(&xb[i])).*;
        acc = simd.fma(lanes, @as(V, @floatFromInt(qv)), xv, acc);
    }
    return acc;
}

/// The integer partial `sum nibble(j)*qx[j]` of a q4_0 block against an already
/// q8-quantized activation, as a `lanes`-wide i32 vector (no horizontal reduce).
/// `lanes` is the i32 lane count (4/8/16).
///
/// The weight is the **unsigned** nibble (0..15) and `qx` is the signed q8
/// activation. LLVM lowers the widening multiply-and-pairwise-add to
/// `vpmaddwd` (AVX2) / `pmaddwd` (SSE2) — the i16 products fit (15*127*2 =
/// 3810 < 2^15). The caller folds the `-8` weight bias with the per-block
/// activation sum (see the q4_0 integer oracle / the generated `Q4_0_int`
/// sweep).
pub fn q4_0IntBlockVec(comptime lanes: usize, block: []const u8, qx: *align(1) const [32]i8) @Vector(lanes, i32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(block[2..18].ptr)).*;
    const lo = bytes & @as(@Vector(16, u8), @splat(0x0F));
    const hi = bytes >> @as(@Vector(16, u8), @splat(4));
    const q0: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qx[0])).*;
    const q1: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qx[16])).*;

    const wlo: @Vector(16, i16) = @intCast(lo);
    const whi: @Vector(16, i16) = @intCast(hi);
    const qlo: @Vector(16, i16) = @intCast(q0);
    const qhi: @Vector(16, i16) = @intCast(q1);

    const plo: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(wlo)) * @as(@Vector(16, i32), @intCast(qlo));
    const phi: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(whi)) * @as(@Vector(16, i32), @intCast(qhi));
    const p = plo + phi;

    // Fold the 16 products into `lanes` groups of `16/lanes` consecutive terms.
    // The caller reduces the vector, so any partition of the sum is valid; the
    // 8-lane grouping (adjacent pairs) matches the `vpmaddwd` shape.
    const groups = 16 / lanes;
    var acc: @Vector(lanes, i32) = @splat(0);
    inline for (0..groups) |m| {
        const idx = comptime blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |j| a[j] = @intCast(j + m * lanes);
            break :blk a;
        };
        acc += @shuffle(i32, p, undefined, idx);
    }
    return acc;
}

/// The unscaled vector partial of a q8_0 block (`sum q*x`, no block scale, no
/// horizontal reduce). The matvec accumulates this across blocks and reduces
/// **once per row** — the per-block `@reduce` is a scalar `vaddss` chain.
pub fn q8_0BlockVec(comptime lanes: usize, block: []const u8, xb: *align(1) const [32]f32) @Vector(lanes, f32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= 32) : (i += lanes) {
        const q: @Vector(lanes, i8) = @as(*align(1) const @Vector(lanes, i8), @ptrCast(&qs[i])).*;
        const xv: V = @as(*align(1) const V, @ptrCast(&xb[i])).*;
        acc = simd.fma(lanes, @as(V, @floatFromInt(q)), xv, acc);
    }
    return acc;
}

/// The integer partial `sum q*qx` of a q8_0 block against an already q8-quantized
/// activation, as a `lanes`-wide i32 vector (no horizontal reduce). Both the
/// weight `q` and `qx` are signed i8; the widening multiply-and-pairwise-add
/// lowers to `pmaddwd`-class integer SIMD (the i16 products fit: 127*127 =
/// 16129 < 2^15). q8_0 has no nibble bias and no min term, so the caller folds
/// only `d_w * d_x` (no correction).
pub fn q8_0IntBlockVec(comptime lanes: usize, qs: *align(1) const [32]i8, qx: *align(1) const [32]i8) @Vector(lanes, i32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const w: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(qs)).*;
    const x: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(qx)).*;
    const w16: @Vector(32, i16) = @intCast(w);
    const x16: @Vector(32, i16) = @intCast(x);
    const p: @Vector(32, i32) = @as(@Vector(32, i32), @intCast(w16)) * @as(@Vector(32, i32), @intCast(x16));

    const groups = 32 / lanes;
    var acc: @Vector(lanes, i32) = @splat(0);
    inline for (0..groups) |m| {
        const idx = comptime blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |j| a[j] = @intCast(j + m * lanes);
            break :blk a;
        };
        acc += @shuffle(i32, p, undefined, idx);
    }
    return acc;
}

/// The unscaled 8-wide partial `sum q*x` of a q4_1 block (deferred-reduce form).
pub fn q4_1BlockVec(qs: *align(1) const [16]u8, xb: *align(1) const [32]f32) @Vector(8, f32) {
    const bytes0: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(qs)).*;
    const bytes1: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(&qs[8])).*;
    const lo_mask: @Vector(8, u8) = @splat(0x0F);
    const hi_shift: @Vector(8, u8) = @splat(4);

    const lo0: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes0 & lo_mask));
    const hi0: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes0 >> hi_shift));
    const lo1: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes1 & lo_mask));
    const hi1: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes1 >> hi_shift));

    const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(xb)).*;
    const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[8])).*;
    const x2: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[16])).*;
    const x3: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[24])).*;

    return lo0 * x0 + hi0 * x2 + lo1 * x1 + hi1 * x3;
}

/// The lanes-wide unscaled partial `sum q*x` of a q4_1 block: element `j` (0..16)
/// is the low nibble of `qs[j]`, element `16+j` the high nibble (the layout of
/// `dequantQ4_1Block`). The generated q4_1 `fp` sweep uses this; the hand-written
/// 8-wide `q4_1BlockVec` is a fixed-width equivalent.
pub fn q4_1BlockVecLanes(comptime lanes: usize, block: []const u8, xb: *align(1) const [32]f32) @Vector(lanes, f32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const qs: *align(1) const [16]u8 = @ptrCast(block[4..20].ptr);
    const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(qs)).*;
    var q: [32]u8 align(16) = undefined;
    @as(*@Vector(16, u8), @ptrCast(&q[0])).* = bytes & @as(@Vector(16, u8), @splat(0x0F));
    @as(*@Vector(16, u8), @ptrCast(&q[16])).* = bytes >> @as(@Vector(16, u8), @splat(4));
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= 32) : (i += lanes) {
        const qv: @Vector(lanes, u8) = @as(*align(1) const @Vector(lanes, u8), @ptrCast(&q[i])).*;
        const xv: V = @as(*align(1) const V, @ptrCast(&xb[i])).*;
        acc = simd.fma(lanes, @as(V, @floatFromInt(qv)), xv, acc);
    }
    return acc;
}

/// The i32 partial `sum q*qx` of a q4_1 block against a q8 activation (unsigned
/// nibble, no scale). Same `pmaddwd`-class shape as `q4_0IntBlockVec`; the q4_1
/// layout places the low nibbles in `qx[0..16]`, the high in `qx[16..32]`.
pub fn q4_1IntBlockVec(comptime lanes: usize, qs: *align(1) const [16]u8, qx: *align(1) const [32]i8) @Vector(lanes, i32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const bytes: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(qs)).*;
    const lo = bytes & @as(@Vector(16, u8), @splat(0x0F));
    const hi = bytes >> @as(@Vector(16, u8), @splat(4));
    const q0: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qx[0])).*;
    const q1: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qx[16])).*;
    const wlo: @Vector(16, i16) = @intCast(lo);
    const whi: @Vector(16, i16) = @intCast(hi);
    const qlo: @Vector(16, i16) = @intCast(q0);
    const qhi: @Vector(16, i16) = @intCast(q1);
    const plo: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(wlo)) * @as(@Vector(16, i32), @intCast(qlo));
    const phi: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(whi)) * @as(@Vector(16, i32), @intCast(qhi));
    const p = plo + phi;
    const groups = 16 / lanes;
    var acc: @Vector(lanes, i32) = @splat(0);
    inline for (0..groups) |m| {
        const idx = comptime blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |j| a[j] = @intCast(j + m * lanes);
            break :blk a;
        };
        acc += @shuffle(i32, p, undefined, idx);
    }
    return acc;
}

/// The unscaled `lanes`-wide partial `sum q*x` of a q4_K 32-nibble chunk. `hi`
/// selects the high nibbles (elements 32..64 of the 64-element group); `lanes`
/// is the f32 lane count (4/8/16).
pub fn q4KNibVec(comptime lanes: usize, qs: *align(1) const [32]u8, xb: *align(1) const [32]f32, comptime hi: bool) @Vector(lanes, f32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    if (comptime lanes == 8) {
        // The native AVX2 path: four explicit 8-byte group loads and a balanced
        // sum. This is the hot path (the generated unroll sweep), and an
        // array/switch generic form regresses it ~3% (worse scheduling/spills),
        // so it is written out. The narrower/wider paths below are not hot.
        const V = @Vector(8, f32);
        const mask: @Vector(8, u8) = @splat(0x0F);
        const shift: @Vector(8, u8) = @splat(4);
        const b0: @Vector(8, u8) = .{ qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], qs[7] };
        const b1: @Vector(8, u8) = .{ qs[8], qs[9], qs[10], qs[11], qs[12], qs[13], qs[14], qs[15] };
        const b2: @Vector(8, u8) = .{ qs[16], qs[17], qs[18], qs[19], qs[20], qs[21], qs[22], qs[23] };
        const b3: @Vector(8, u8) = .{ qs[24], qs[25], qs[26], qs[27], qs[28], qs[29], qs[30], qs[31] };
        const n0 = if (hi) b0 >> shift else b0 & mask;
        const n1 = if (hi) b1 >> shift else b1 & mask;
        const n2 = if (hi) b2 >> shift else b2 & mask;
        const n3 = if (hi) b3 >> shift else b3 & mask;
        const q0: V = @floatFromInt(n0);
        const q1: V = @floatFromInt(n1);
        const q2: V = @floatFromInt(n2);
        const q3: V = @floatFromInt(n3);
        const x0: V = @as(*align(1) const V, @ptrCast(&xb[0])).*;
        const x1: V = @as(*align(1) const V, @ptrCast(&xb[8])).*;
        const x2: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
        const x3: V = @as(*align(1) const V, @ptrCast(&xb[24])).*;
        return (q0 * x0 + q1 * x1) + (q2 * x2 + q3 * x3);
    }
    const V = @Vector(lanes, f32);
    const mask: @Vector(lanes, u8) = @splat(0x0F);
    const shift: @Vector(lanes, u8) = @splat(4);
    const groups = 32 / lanes;
    var p: [groups]V = undefined;
    inline for (0..groups) |g| {
        const i = g * lanes;
        const bytes: @Vector(lanes, u8) = @as(*align(1) const @Vector(lanes, u8), @ptrCast(&qs[i])).*;
        const nib = if (hi) bytes >> shift else bytes & mask;
        const q: V = @floatFromInt(nib);
        const xv: V = @as(*align(1) const V, @ptrCast(&xb[i])).*;
        p[g] = q * xv;
    }
    return switch (groups) {
        2 => p[0] + p[1],
        8 => ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7])),
        else => unreachable,
    };
}

/// The integer partial `sum nibble*qx` of a q4_K 32-nibble chunk against an
/// already q8-quantized activation, as a `lanes`-wide i32 vector. `hi` selects
/// the high nibbles. The weight is the unsigned nibble (0..15), `qx` the signed
/// q8 activation; LLVM lowers the widening multiply-and-add to `pmaddwd`-class
/// integer SIMD. The caller folds the packed scale and the min term.
pub fn q4KIntNibVec(comptime lanes: usize, qs: *align(1) const [32]u8, qx: *align(1) const [32]i8, comptime hi: bool) @Vector(lanes, i32) {
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    const bytes: @Vector(32, u8) = @as(*align(1) const @Vector(32, u8), @ptrCast(qs)).*;
    const lo = bytes & @as(@Vector(32, u8), @splat(0x0F));
    const hi_b = bytes >> @as(@Vector(32, u8), @splat(4));
    const nib = if (hi) hi_b else lo;
    const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(qx)).*;
    const w16: @Vector(32, i16) = @intCast(nib);
    const q16: @Vector(32, i16) = @intCast(qv);
    const p: @Vector(32, i32) = @as(@Vector(32, i32), @intCast(w16)) * @as(@Vector(32, i32), @intCast(q16));

    const groups = 32 / lanes;
    var acc: @Vector(lanes, i32) = @splat(0);
    inline for (0..groups) |m| {
        const idx = comptime blk: {
            var a: [lanes]i32 = undefined;
            for (0..lanes) |j| a[j] = @intCast(j + m * lanes);
            break :blk a;
        };
        acc += @shuffle(i32, p, undefined, idx);
    }
    return acc;
}

/// The activation-only sum of a 32-element chunk (`sum x`), used by the q4_K
/// min term; the same value for every output row.
pub fn q4KNibXSum32(xb: *align(1) const [32]f32) f32 {
    const V = @Vector(8, f32);
    const x0: V = @as(*align(1) const V, @ptrCast(&xb[0])).*;
    const x1: V = @as(*align(1) const V, @ptrCast(&xb[8])).*;
    const x2: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
    const x3: V = @as(*align(1) const V, @ptrCast(&xb[24])).*;
    return @reduce(.Add, (x0 + x1) + (x2 + x3));
}

/// Accumulate one q4_K super-block into a `lanes`-wide f32 vector partial and a
/// scalar min-term sum, using the precomputed per-32-chunk activation sums. One
/// source of truth for the generic single-row dot (`form/dot.zig`) and the
/// generated f32 body. The
/// q4_K value is `d*sc*q - dmin*mn`, so the vector partial carries `d*sc*sum(q*x)`
/// and the scalar carries `dmin*mn*sum(x)`.
pub fn q4KBlockAccumGen(
    comptime lanes: usize,
    block: []const u8,
    xb: []const f32,
    xsum: []const f32,
    accv: *@Vector(lanes, f32),
    msum: *f32,
) void {
    const d: f32 = @floatCast(readF16(block[0..2]));
    const dmin: f32 = @floatCast(readF16(block[2..4]));
    const scales: *const [12]u8 = block[4..16];
    const qs: *align(1) const [128]u8 = @ptrCast(&block[16]);
    const V = @Vector(lanes, f32);

    var is: usize = 0;
    var qo: usize = 0;
    var yo: usize = 0;
    while (yo < 256) : (yo += 64) {
        var sc1: u8 = undefined;
        var mn1: u8 = undefined;
        var sc2: u8 = undefined;
        var mn2: u8 = undefined;
        dequant.getScaleMinK4(is + 0, scales, &sc1, &mn1);
        dequant.getScaleMinK4(is + 1, scales, &sc2, &mn2);
        const chunk_q: *align(1) const [32]u8 = @ptrCast(&qs[qo]);
        const x_lo: *align(1) const [32]f32 = @ptrCast(&xb[yo]);
        const x_hi: *align(1) const [32]f32 = @ptrCast(&xb[yo + 32]);
        accv.* = simd.fma(lanes, @as(V, @splat(d * @as(f32, sc1))), q4KNibVec(lanes, chunk_q, x_lo, false), accv.*);
        accv.* = simd.fma(lanes, @as(V, @splat(d * @as(f32, sc2))), q4KNibVec(lanes, chunk_q, x_hi, true), accv.*);
        msum.* += (dmin * @as(f32, mn1)) * xsum[yo / 32] + (dmin * @as(f32, mn2)) * xsum[yo / 32 + 1];
        qo += 32;
        is += 2;
    }
}

/// Accumulate one q4_K super-block into a `lanes`-wide f32 vector partial and a
/// scalar min-term correction, using the q8-quantized activation (`qx`, `dx`,
/// `sums`). With `x ~ dx*qx` and the q4_K value `d*sc*q - dmin*mn`:
///   sum(w*x) ~ d*sc*dx*sum(q*qx) - dmin*mn*dx*sum(qx).
/// The first term folds `d*sc*dx` into the FMA, the second accumulates `corr`.
pub fn q4KBlockIntAccum(
    comptime lanes: usize,
    block: []const u8,
    qx: []const i8,
    dx: []const f32,
    sums: []const i32,
    accv: *@Vector(lanes, f32),
    corr: *f32,
) void {
    const d: f32 = @floatCast(readF16(block[0..2]));
    const dmin: f32 = @floatCast(readF16(block[2..4]));
    const scales: *const [12]u8 = block[4..16];
    const qs: *align(1) const [128]u8 = @ptrCast(&block[16]);
    const V = @Vector(lanes, f32);

    var is: usize = 0;
    var qo: usize = 0;
    var yo: usize = 0;
    while (yo < 256) : (yo += 64) {
        var sc1: u8 = undefined;
        var mn1: u8 = undefined;
        var sc2: u8 = undefined;
        var mn2: u8 = undefined;
        dequant.getScaleMinK4(is + 0, scales, &sc1, &mn1);
        dequant.getScaleMinK4(is + 1, scales, &sc2, &mn2);
        const c = yo / 32;
        const chunk_q: *align(1) const [32]u8 = @ptrCast(&qs[qo]);
        const qx_lo: *align(1) const [32]i8 = @ptrCast(&qx[c * 32]);
        const qx_hi: *align(1) const [32]i8 = @ptrCast(&qx[(c + 1) * 32]);
        const iv_lo = q4KIntNibVec(lanes, chunk_q, qx_lo, false);
        const iv_hi = q4KIntNibVec(lanes, chunk_q, qx_hi, true);
        accv.* = simd.fma(lanes, @as(V, @splat(d * @as(f32, sc1) * dx[c])), @as(V, @floatFromInt(iv_lo)), accv.*);
        accv.* = simd.fma(lanes, @as(V, @splat(d * @as(f32, sc2) * dx[c + 1])), @as(V, @floatFromInt(iv_hi)), accv.*);
        corr.* += (dmin * @as(f32, mn1) * dx[c]) * @as(f32, @floatFromInt(sums[c])) +
            (dmin * @as(f32, mn2) * dx[c + 1]) * @as(f32, @floatFromInt(sums[c + 1]));
        qo += 32;
        is += 2;
    }
}

/// The unscaled 16-wide partial of a q6_K chunk (`qf*x`, no scale, no reduce).
/// The block dot accumulates these across chunks and reduces **once per block**
/// (the per-chunk `@reduce` is a scalar `vaddss` chain; cf. `q4_0BlockVec`).
pub fn q6KChunkVec16(
    ql: *align(1) const [16]u8,
    qh: *align(1) const [16]u8,
    xb: *align(1) const [16]f32,
    comptime hi: bool,
    comptime qh_shift: u3,
) @Vector(16, f32) {
    const V = @Vector(16, f32);
    const bytes_q: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(ql)).*;
    const bytes_h: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(qh)).*;
    const nib: @Vector(16, u8) = if (hi)
        bytes_q >> @as(@Vector(16, u8), @splat(4))
    else
        bytes_q & @as(@Vector(16, u8), @splat(0x0F));
    const hb: @Vector(16, u8) = (bytes_h >> @as(@Vector(16, u8), @splat(@as(u8, qh_shift)))) & @as(@Vector(16, u8), @splat(0x03));
    const q_u8 = nib | (hb << @as(@Vector(16, u8), @splat(4)));
    const q_i8: @Vector(16, i8) = @bitCast(q_u8);
    const qf: V = @as(V, @floatFromInt(q_i8)) - @as(V, @splat(32.0));
    const x: V = @as(*align(1) const V, @ptrCast(xb)).*;
    return qf * x;
}

pub fn q6KBlockDot(block: []const u8, xb: []const f32) f32 {
    const QK_K: usize = 256;
    const d: f32 = @floatCast(readF16(block[208..210]));
    const ql = block[0..128];
    const qh = block[128..192];
    const sc = block[192..208];

    var y: usize = 0;
    var ql_off: usize = 0;
    var qh_off: usize = 0;
    var sc_off: usize = 0;
    // Deferred reduction: accumulate the scaled chunk vectors, reduce once.
    const V = @Vector(16, f32);
    var acc: V = @splat(0);
    while (y < QK_K) : (y += 128) {
        inline for ([_]usize{ 0, 16 }) |l_base| {
            const is = l_base / 16;
            const a0: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 0])));
            const a2: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 2])));
            const a4: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 4])));
            const a6: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 6])));
            const ql0: *align(1) const [16]u8 = @ptrCast(&ql[ql_off + l_base]);
            const ql1: *align(1) const [16]u8 = @ptrCast(&ql[ql_off + l_base + 32]);
            const qh0: *align(1) const [16]u8 = @ptrCast(&qh[qh_off + l_base]);
            const xb0: *align(1) const [16]f32 = @ptrCast(&xb[y + l_base]);
            const xb1: *align(1) const [16]f32 = @ptrCast(&xb[y + l_base + 32]);
            const xb2: *align(1) const [16]f32 = @ptrCast(&xb[y + l_base + 64]);
            const xb3: *align(1) const [16]f32 = @ptrCast(&xb[y + l_base + 96]);
            acc = simd.fma(16, @as(V, @splat(d * a0)), q6KChunkVec16(ql0, qh0, xb0, false, 0), acc);
            acc = simd.fma(16, @as(V, @splat(d * a2)), q6KChunkVec16(ql1, qh0, xb1, false, 2), acc);
            acc = simd.fma(16, @as(V, @splat(d * a4)), q6KChunkVec16(ql0, qh0, xb2, true, 4), acc);
            acc = simd.fma(16, @as(V, @splat(d * a6)), q6KChunkVec16(ql1, qh0, xb3, true, 6), acc);
        }
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    return @reduce(.Add, acc);
}

/// Per-32-block q8_0 activation quantization: `qx[j] = round(x[j] / d)` in
/// [-127,127] with `d = max|x| / 127`; `sums[rb]` is the integer `sum(qx)` of the
/// block. The integer q4_0/q4_k matvecs read `qx`/`dx`/`sums` (the q4_0 path
/// pre-multiplies `sums` by 8, the weight bias).
pub fn quantizeActivationQ8Sum(x: []const f32, blocks: usize, qx: []i8, dx: []f32, sums: []i32) void {
    for (0..blocks) |rb| {
        const xb = x[rb * 32 ..][0..32];
        var amax: f32 = 0;
        for (xb) |v| amax = @max(amax, @abs(v));
        const d: f32 = amax / 127.0;
        const inv: f32 = if (d != 0) 1.0 / d else 0;
        var s: i32 = 0;
        for (xb, 0..) |v, j| {
            var q: i32 = @intFromFloat(@round(v * inv));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qx[rb * 32 + j] = @intCast(q);
            s += q;
        }
        dx[rb] = d;
        sums[rb] = s;
    }
}

/// Per-32-block q8_0 activation quantization that the integer q4_0 matvec reads.
/// `dxs[rb]` is pre-multiplied by 8 (the weight bias) so the caller folds the
/// bias with a single multiply per block.
pub fn quantizeActivationQ8(x: []const f32, blocks: usize, qx: []i8, dx: []f32, dxs8: []f32) void {
    var sums: [max_activation_cache_blocks]i32 = undefined;
    quantizeActivationQ8Sum(x, blocks, qx, dx, sums[0..blocks]);
    for (0..blocks) |rb| {
        dxs8[rb] = 8.0 * dx[rb] * @as(f32, @floatFromInt(sums[rb]));
    }
}

pub fn xBlockSum(xb: *align(1) const [32]f32) f32 {
    const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(xb)).*;
    const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[8])).*;
    const x2: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[16])).*;
    const x3: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[24])).*;
    return @reduce(.Add, x0 + x1 + x2 + x3);
}

/// The per-32-chunk activation sums q4_K's min term needs. They depend only on
/// `x`, so a row loop computes them once instead of per output row.
pub fn q4KXSum(x: []const f32, xsum: []f32) void {
    const chunks = x.len / 32;
    for (0..chunks) |c| {
        xsum[c] = q4KNibXSum32(@ptrCast(&x[c * 32]));
    }
}

pub fn readF16(data: []const u8) f16 {
    const raw = std.mem.readInt(u16, data[0..2], .little);
    return @bitCast(raw);
}

pub fn matVecQ4_0ScaleSplit(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (x.len != n or out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const scales_total = m * row_blocks * 2;
    const scales = data[0..scales_total];
    const quants = data[scales_total..need];

    // Deferred reduction (cf. native.zig): one horizontal reduce per row.
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        const s_row = scales[i * row_blocks * 2 ..][0 .. row_blocks * 2];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        var accv: V = @splat(0);
        for (0..row_blocks) |rb| {
            const d: f32 = @floatCast(readF16(s_row[rb * 2 ..][0..2]));
            const qs: *align(1) const [16]u8 = @ptrCast(&q_row[rb * 16]);
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            accv = simd.fma(8, @as(V, @splat(d)), scaleSplitBlockVec(qs, xb), accv);
        }
        out[i] = @reduce(.Add, accv);
    }
}

pub fn matVecQ4_1ScaleSplit(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (x.len != n or out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const meta_total = m * row_blocks * 4;
    const meta = data[0..meta_total];
    const quants = data[meta_total..need];
    if (row_blocks > types.max_activation_cache_blocks) return matVecQ4_1ScaleSplitNoCache(data, n, m, x, out, begin, end);

    var x_sums: [types.max_activation_cache_blocks]f32 = undefined;
    for (0..row_blocks) |rb| {
        const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
        x_sums[rb] = xBlockSum(xb);
    }

    // Deferred reduction; `x_sums` are activation-only.
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        const m_row = meta[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        var accv: V = @splat(0);
        var msum: f32 = 0;
        for (0..row_blocks) |rb| {
            const d: f32 = @floatCast(readF16(m_row[rb * 4 ..][0..2]));
            const mn: f32 = @floatCast(readF16(m_row[rb * 4 + 2 ..][0..2]));
            const qs: *align(1) const [16]u8 = @ptrCast(&q_row[rb * 16]);
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            accv = simd.fma(8, @as(V, @splat(d)), q4_1ScaleSplitBlockVec(qs, xb), accv);
            msum += mn * x_sums[rb];
        }
        out[i] = @reduce(.Add, accv) + msum;
    }
}

fn matVecQ4_1ScaleSplitNoCache(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    const meta_total = m * row_blocks * 4;
    const meta = data[0..meta_total];
    const quants = data[meta_total..need];

    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var accv: V = @splat(0);
        var msum: f32 = 0;
        const m_row = meta[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];

        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const mg = m_row[rb * 4 ..][0..4];
            const qs: *align(1) const [16]u8 = @ptrCast(q_row[rb * 16 ..][0..16].ptr);
            const d: f32 = @floatCast(readF16(mg[0..2]));
            const mn: f32 = @floatCast(readF16(mg[2..4]));
            accv = simd.fma(8, @as(V, @splat(d)), q4_1ScaleSplitBlockVec(qs, xb), accv);
            msum += mn * xBlockSum(xb);
        }
        out[i] = @reduce(.Add, accv) + msum;
    }
}

/// The unscaled 8-wide partial `sum (nibble-8)*x` of a q4_0 scale-split block.
pub fn scaleSplitBlockVec(qs: *align(1) const [16]u8, xb: *align(1) const [32]f32) @Vector(8, f32) {
    const bytes0: @Vector(8, u8) = .{ qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], qs[7] };
    const bytes1: @Vector(8, u8) = .{ qs[8], qs[9], qs[10], qs[11], qs[12], qs[13], qs[14], qs[15] };
    const lo_mask: @Vector(8, u8) = @splat(0x0F);
    const hi_shift: @Vector(8, u8) = @splat(4);

    const lo0: @Vector(8, i8) = @bitCast(bytes0 & lo_mask);
    const hi0: @Vector(8, i8) = @bitCast(bytes0 >> hi_shift);
    const lo1: @Vector(8, i8) = @bitCast(bytes1 & lo_mask);
    const hi1: @Vector(8, i8) = @bitCast(bytes1 >> hi_shift);
    const bias: @Vector(8, i8) = @splat(8);

    const q0: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(lo0 - bias));
    const q1: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(hi0 - bias));
    const q2: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(lo1 - bias));
    const q3: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(hi1 - bias));

    const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(xb)).*;
    const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[8])).*;
    const x2: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[16])).*;
    const x3: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[24])).*;

    return q0 * x0 + q1 * x2 + q2 * x1 + q3 * x3;
}

/// The unscaled 8-wide partial `sum nibble*x` of a q4_1 scale-split block.
pub fn q4_1ScaleSplitBlockVec(qs: *align(1) const [16]u8, xb: *align(1) const [32]f32) @Vector(8, f32) {
    const bytes0: @Vector(8, u8) = .{ qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], qs[7] };
    const bytes1: @Vector(8, u8) = .{ qs[8], qs[9], qs[10], qs[11], qs[12], qs[13], qs[14], qs[15] };
    const lo_mask: @Vector(8, u8) = @splat(0x0F);
    const hi_shift: @Vector(8, u8) = @splat(4);

    const lo0: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes0 & lo_mask));
    const hi0: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes0 >> hi_shift));
    const lo1: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes1 & lo_mask));
    const hi1: @Vector(8, f32) = @as(@Vector(8, f32), @floatFromInt(bytes1 >> hi_shift));

    const x0: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(xb)).*;
    const x1: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[8])).*;
    const x2: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[16])).*;
    const x3: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(&xb[24])).*;

    return lo0 * x0 + hi0 * x2 + lo1 * x1 + hi1 * x3;
}

const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../../../simd.zig");
const layout = @import("../layout.zig");
const single = @import("../form/single.zig");
const standard = @import("../../quant/standard.zig");
const k_quants = @import("../../quant/k_quants.zig");
const api = @import("../../../api.zig");

pub const KernelError = types.KernelError;
pub const MatVecPlan = types.MatVecPlan;
pub const Variant = types.Variant;

const Activation = layout.Activation;
const readF16 = layout.readF16;
const q4_0BlockVec = layout.q4_0BlockVec;
const q4_0IntBlockVec = layout.q4_0IntBlockVec;
const q8_0BlockVec = layout.q8_0BlockVec;
const q8_0IntBlockVec = layout.q8_0IntBlockVec;
const q4_1BlockVecLanes = layout.q4_1BlockVecLanes;
const q4_1IntBlockVec = layout.q4_1IntBlockVec;
const q4KBlockAccumGen = layout.q4KBlockAccumGen;
const q4KBlockIntAccum = layout.q4KBlockIntAccum;
const q6KBlockDot = layout.q6KBlockDot;
const matVecFpGen = single.matVecFpGen;

// --- L1: the weight decode family (a comptime type per family member) ---
//
// Each member exposes `block_elems`, `block_bytes`, `has_min`, `production_fp`
// and the two dot steps the sweep calls: `fpBlock` (fp) and `intBlock` (int).
// The step folds the block's scale(s) and min term into a `lanes`-wide f32
// accumulator and a scalar correction (`out = reduce(acc) + corr`). There is no
// `switch (quant)`: the sweep's `comptime dot` selects the step and
// monomorphization inlines it.

pub const Q4_0 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = 18;
    pub const has_min: bool = false;
    pub const production_fp = matVecQ4_0U4;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const d: f32 = @floatCast(readF16(block[0..2]));
        const xb: *align(1) const [32]f32 = @ptrCast(act.x[rb * 32 ..][0..32].ptr);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(d)), q4_0BlockVec(lanes, block, xb), acc.*);
    }

    pub inline fn intBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        const dw: f32 = @floatCast(readF16(block[0..2]));
        const qx: *align(1) const [32]i8 = @ptrCast(act.qx[rb * 32 ..][0..32].ptr);
        const iv = q4_0IntBlockVec(lanes, block, qx);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(dw * act.dx[rb])), @as(@Vector(lanes, f32), @floatFromInt(iv)), acc.*);
        corr.* -= dw * (8.0 * act.dx[rb] * @as(f32, @floatFromInt(act.sums[rb])));
    }

    /// The L1 block dequant: decode the block's 32 elements to f32, at the same
    /// block granularity the sweep's `fpBlock` uses. Delegates to the codec's
    /// block decode.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return standard.dequantQ4_0(block, out);
    }
};

pub const Q4_1 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = 20;
    pub const has_min: bool = true;
    pub const production_fp = matVecQ4_1U4;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        const d: f32 = @floatCast(readF16(block[0..2]));
        const mn: f32 = @floatCast(readF16(block[2..4]));
        const xb: *align(1) const [32]f32 = @ptrCast(act.x[rb * 32 ..][0..32].ptr);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(d)), q4_1BlockVecLanes(lanes, block, xb), acc.*);
        corr.* += mn * act.xsum[rb];
    }

    pub inline fn intBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        const d: f32 = @floatCast(readF16(block[0..2]));
        const mn: f32 = @floatCast(readF16(block[2..4]));
        const qs: *align(1) const [16]u8 = @ptrCast(block[4..20].ptr);
        const qx: *align(1) const [32]i8 = @ptrCast(act.qx[rb * 32 ..][0..32].ptr);
        const iv = q4_1IntBlockVec(lanes, qs, qx);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(d * act.dx[rb])), @as(@Vector(lanes, f32), @floatFromInt(iv)), acc.*);
        corr.* += mn * act.dx[rb] * @as(f32, @floatFromInt(act.sums[rb]));
    }

    /// The L1 block dequant: decode the block's 32 elements to f32 (`q*d + m`),
    /// at the same block granularity the sweep's `fpBlock` uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return standard.dequantQ4_1(block, out);
    }
};

pub const Q4_K = struct {
    pub const block_elems: usize = 256;
    pub const block_bytes: usize = 144;
    pub const has_min: bool = true;
    pub const production_fp = matVecQ4_K;

    /// The per-super-block f32 partial (the packed scale decode applied), into
    /// `acc`, with the min term folded into `corr`. The 6-bit scale/min decode is
    /// per 32-chunk, so this kernel has no single block scale.
    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        var msum: f32 = 0;
        q4KBlockAccumGen(lanes, block, act.x[rb * 256 ..][0..256], act.xsum[rb * 8 ..][0..8], acc, &msum);
        corr.* -= msum;
    }

    pub inline fn intBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        var c: f32 = 0;
        q4KBlockIntAccum(
            lanes,
            block,
            act.qx[rb * 8 * 32 ..][0..256],
            act.dx[rb * 8 ..][0..8],
            act.sums[rb * 8 ..][0..8],
            acc,
            &c,
        );
        corr.* -= c;
    }

    /// The L1 block dequant: decode the super-block's 256 elements to f32
    /// (`d*sc*q - dmin*mn`), at the same block granularity the sweep's `fpBlock`
    /// uses. Delegates to the codec's block decode.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return k_quants.dequantQ4_K(block, out);
    }
};

/// q8_0: a signed 8-bit block with a single f16 scale (`d * sum(q*x)`), no min
/// term. Both steps read signed i8 weights (`fpBlock` via `q8_0BlockVec`,
/// `intBlock` via `q8_0IntBlockVec`); there is no nibble bias or min correction.
pub const Q8_0 = struct {
    pub const block_elems: usize = 32;
    pub const block_bytes: usize = 34;
    pub const has_min: bool = false;
    pub const production_fp = matVecQ8_0Production;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const d: f32 = @floatCast(readF16(block[0..2]));
        const xb: *align(1) const [32]f32 = @ptrCast(act.x[rb * 32 ..][0..32].ptr);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(d)), q8_0BlockVec(lanes, block, xb), acc.*);
    }

    pub inline fn intBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = corr;
        const d: f32 = @floatCast(readF16(block[0..2]));
        const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
        const qx: *align(1) const [32]i8 = @ptrCast(act.qx[rb * 32 ..][0..32].ptr);
        const iv = q8_0IntBlockVec(lanes, qs, qx);
        acc.* = simd.fma(lanes, @as(@Vector(lanes, f32), @splat(d * act.dx[rb])), @as(@Vector(lanes, f32), @floatFromInt(iv)), acc.*);
    }

    /// The L1 block dequant: decode the block's 32 signed elements to f32
    /// (`d*q`), at the same block granularity the sweep's `fpBlock` uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return standard.dequantQ8_0(block, out);
    }
};

/// q6_k: a signed 6-bit block with a per-16 `i8` scale and a single f16 super
/// scale (`d * sum(sc*q*x)`), no min term. `fpBlock` reuses `q6KBlockDot`, the
/// block-level decode the row-dot helper also uses; there is no integer decode,
/// so q6_k is `fp`-only.
pub const Q6_K = struct {
    pub const block_elems: usize = 256;
    pub const block_bytes: usize = 210;
    pub const has_min: bool = false;
    pub const production_fp = matVecQ6_KProduction;

    pub inline fn fpBlock(comptime lanes: usize, block: []const u8, act: Activation, rb: usize, acc: *@Vector(lanes, f32), corr: *f32) void {
        _ = acc;
        corr.* += q6KBlockDot(block, act.x[rb * 256 ..][0..256]);
    }

    /// The L1 block dequant: decode the super-block's 256 signed 6-bit elements
    /// to f32 (`d*sc*q`), at the same block granularity the sweep's `fpBlock`
    /// uses.
    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return k_quants.dequantQ6_K(block, out);
    }
};

/// q5_k: a 5-bit block with a per-32 packed scale/min and a high-bit plane
/// (`d*sc*q - dmin*mn`). It is **dequant-only** today — used by the
/// `per_layer_token_embd` lookup, not a GEMV — so it carries no `fpBlock`/
/// `intBlock` and is not a matvec `kernel_catalog` member. Its `dequantBlock`
/// reuses the codec, so `q5_k` shares the one L1 dequant.
pub const Q5_K = struct {
    pub const block_elems: usize = 256;
    pub const block_bytes: usize = 176;

    pub inline fn dequantBlock(block: []const u8, out: []f32) api.KernelError!void {
        return k_quants.dequantQ5_K(block, out);
    }
};

/// q4_0 production (`fp`) body: the native-width, tile-1, unroll-1 generated
/// sweep. It is `production_fp` (the too-large-activation fallback), which q4_0
/// never reaches (it has no min term).
pub fn matVecQ4_0U4(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecFpGen(Q4_0, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, data, n, m, x, out, begin, end);
}

/// q8_0 production (`fp`) body: the native-width, tile-1, unroll-1 generated
/// sweep. It is `production_fp` (the too-large-activation fallback), which q8_0
/// never reaches (it has no min term).
fn matVecQ8_0Production(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecFpGen(Q8_0, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, data, n, m, x, out, begin, end);
}

/// q6_k production (`fp`) body: the native-width, tile-1, unroll-1 generated
/// sweep. It is `production_fp`, the `.production` fallback for a q6_k view.
fn matVecQ6_KProduction(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecFpGen(Q6_K, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, data, n, m, x, out, begin, end);
}

/// q8_0 block dot at a compile-time f32 width. One generic source; the build
/// instantiates it for each `simd.Width` and the Tuner picks the fastest.
pub fn q8_0BlockDotLanes(comptime lanes: usize, block: []const u8, xb: *align(1) const [32]f32) f32 {
    const d: f32 = @floatCast(readF16(block[0..2]));
    return d * @reduce(.Add, q8_0BlockVec(lanes, block, xb));
}

/// q4_1 production (`fp`) body: the native-width, tile-1, unroll-1 generated
/// sweep. It is `production_fp`; the sweep precomputes the activation sums the
/// min term needs (`has_min`).
pub fn matVecQ4_1U4(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecFpGen(Q4_1, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, data, n, m, x, out, begin, end);
}

/// q4_K production (`fp`) body: the native-width, tile-1, unroll-1 generated
/// sweep. It is `production_fp`; the sweep precomputes the per-32-chunk
/// activation sums the min term needs (`has_min`).
pub fn matVecQ4_K(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecFpGen(Q4_K, .{ .width = @intCast(simd.native.lanes()), .tile = 1, .unroll = 1 }, data, n, m, x, out, begin, end);
}

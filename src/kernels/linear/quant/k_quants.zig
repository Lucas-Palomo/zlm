const std = @import("std");
const types = @import("types.zig");
const KernelError = types.KernelError;
const readF16 = types.readF16;
const QK_K = types.QK_K;
const Q4_K_BLOCK = types.Q4_K_BLOCK;
const Q5_K_BLOCK = types.Q5_K_BLOCK;
const Q6_K_BLOCK = types.Q6_K_BLOCK;
const simd = @import("../../simd.zig");

/// Dequantize one 16-wide nibble vector: `out[dst..] = f32(vals)*d - m`.
inline fn storeScaled(out: []f32, dst: usize, vals: @Vector(16, u8), d: f32, m: f32) void {
    const V = @Vector(16, f32);
    const f: V = @as(V, @floatFromInt(vals));
    @as(*align(1) V, @ptrCast(&out[dst])).* = simd.fma(16, f, @as(V, @splat(d)), @as(V, @splat(-m)));
}

inline fn storeScaledI(out: []f32, dst: usize, vals: @Vector(16, i32), scale: f32) void {
    const V = @Vector(16, f32);
    const f: V = @as(V, @floatFromInt(vals));
    @as(*align(1) V, @ptrCast(&out[dst])).* = f * @as(V, @splat(scale));
}

pub fn getScaleMinK4(j: usize, scales: *const [12]u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = scales[j] & 63;
        m.* = scales[j + 4] & 63;
    } else {
        d.* = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m.* = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

pub fn dequantQ4_K(data: []const u8, out: []f32) KernelError!void {
    if (out.len % QK_K != 0) return error.ShapeMismatch;
    const n_blocks: usize = out.len / QK_K;
    const need: usize = n_blocks * Q4_K_BLOCK;
    if (data.len < need) return error.TruncatedData;

    for (0..n_blocks) |bi| {
        const block = data[bi * Q4_K_BLOCK ..][0..Q4_K_BLOCK];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const min: f32 = @floatCast(readF16(block[2..4]));
        const scales: *const [12]u8 = block[4..16];
        const q = block[16..144];
        const base = bi * QK_K;

        var is: usize = 0;
        var qo: usize = 0;
        var yo: usize = 0;
        while (yo < QK_K) : (yo += 64) {
            var sc: u8 = undefined;
            var mn: u8 = undefined;
            getScaleMinK4(is + 0, scales, &sc, &mn);
            const d1 = d * @as(f32, sc);
            const m1 = min * @as(f32, mn);
            getScaleMinK4(is + 1, scales, &sc, &mn);
            const d2 = d * @as(f32, sc);
            const m2 = min * @as(f32, mn);
            const b0: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(&q[qo])).*;
            const b1: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(&q[qo + 16])).*;
            const lo_mask: @Vector(16, u8) = @splat(0x0F);
            const hi_shift: @Vector(16, u8) = @splat(4);
            storeScaled(out, base + yo + 0, b0 & lo_mask, d1, m1);
            storeScaled(out, base + yo + 16, b1 & lo_mask, d1, m1);
            storeScaled(out, base + yo + 32, b0 >> hi_shift, d2, m2);
            storeScaled(out, base + yo + 48, b1 >> hi_shift, d2, m2);
            qo += 32;
            is += 2;
        }
    }
}

pub fn dequantQ5_K(data: []const u8, out: []f32) KernelError!void {
    if (out.len % QK_K != 0) return error.ShapeMismatch;
    const n_blocks: usize = out.len / QK_K;
    const need: usize = n_blocks * Q5_K_BLOCK;
    if (data.len < need) return error.TruncatedData;

    for (0..n_blocks) |bi| {
        const block = data[bi * Q5_K_BLOCK ..][0..Q5_K_BLOCK];
        const d: f32 = @floatCast(readF16(block[0..2]));
        const min: f32 = @floatCast(readF16(block[2..4]));
        const scales: *const [12]u8 = block[4..16];
        const qh = block[16..48];
        const q = block[48..176];
        const base = bi * QK_K;

        var is: usize = 0;
        var qo: usize = 0;
        var yo: usize = 0;
        var bit_lo: u8 = 1;
        var bit_hi: u8 = 2;
        while (yo < QK_K) : (yo += 64) {
            var sc: u8 = undefined;
            var mn: u8 = undefined;
            getScaleMinK4(is + 0, scales, &sc, &mn);
            const d1 = d * @as(f32, sc);
            const m1 = min * @as(f32, mn);
            getScaleMinK4(is + 1, scales, &sc, &mn);
            const d2 = d * @as(f32, sc);
            const m2 = min * @as(f32, mn);
            const V16 = @Vector(16, u8);
            const b0: V16 = @as(*align(1) const V16, @ptrCast(&q[qo])).*;
            const b1: V16 = @as(*align(1) const V16, @ptrCast(&q[qo + 16])).*;
            const h0: V16 = @as(*align(1) const V16, @ptrCast(&qh[0])).*;
            const h1: V16 = @as(*align(1) const V16, @ptrCast(&qh[16])).*;
            const lo_mask: V16 = @splat(0x0F);
            const hi_shift: V16 = @splat(4);
            const zero: V16 = @splat(0);
            const sixteen: V16 = @splat(16);
            const hb_lo0 = @select(u8, (h0 & @as(V16, @splat(bit_lo))) != zero, sixteen, zero);
            const hb_lo1 = @select(u8, (h1 & @as(V16, @splat(bit_lo))) != zero, sixteen, zero);
            const hb_hi0 = @select(u8, (h0 & @as(V16, @splat(bit_hi))) != zero, sixteen, zero);
            const hb_hi1 = @select(u8, (h1 & @as(V16, @splat(bit_hi))) != zero, sixteen, zero);
            storeScaled(out, base + yo + 0, (b0 & lo_mask) + hb_lo0, d1, m1);
            storeScaled(out, base + yo + 16, (b1 & lo_mask) + hb_lo1, d1, m1);
            storeScaled(out, base + yo + 32, (b0 >> hi_shift) + hb_hi0, d2, m2);
            storeScaled(out, base + yo + 48, (b1 >> hi_shift) + hb_hi1, d2, m2);
            qo += 32;
            is += 2;
            bit_lo <<= 2;
            bit_hi <<= 2;
        }
    }
}

pub fn dequantQ6_K(data: []const u8, out: []f32) KernelError!void {
    if (out.len % QK_K != 0) return error.ShapeMismatch;
    const n_blocks: usize = out.len / QK_K;
    const need: usize = n_blocks * Q6_K_BLOCK;
    if (data.len < need) return error.TruncatedData;

    for (0..n_blocks) |bi| {
        const block = data[bi * Q6_K_BLOCK ..][0..Q6_K_BLOCK];
        const d: f32 = @floatCast(readF16(block[208..210]));
        const ql = block[0..128];
        const qh = block[128..192];
        const sc = block[192..208];
        const base = bi * QK_K;

        var y: usize = 0;
        var ql_off: usize = 0;
        var qh_off: usize = 0;
        var sc_off: usize = 0;
        while (y < QK_K) : (y += 128) {
            inline for (0..2) |half| {
                const l0 = half * 16;
                const s0: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + half + 0])));
                const s2: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + half + 2])));
                const s4: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + half + 4])));
                const s6: f32 = @floatFromInt(@as(i8, @bitCast(sc[sc_off + half + 6])));
                const V16 = @Vector(16, u8);
                const ql0: V16 = @as(*align(1) const V16, @ptrCast(&ql[ql_off + l0])).*;
                const ql1: V16 = @as(*align(1) const V16, @ptrCast(&ql[ql_off + l0 + 32])).*;
                const qh0: V16 = @as(*align(1) const V16, @ptrCast(&qh[qh_off + l0])).*;
                const m0f: V16 = @splat(0x0F);
                const m03: V16 = @splat(0x03);
                const q1u: V16 = (ql0 & m0f) | ((qh0 & m03) << @as(V16, @splat(4)));
                const q2u: V16 = (ql1 & m0f) | (((qh0 >> @as(V16, @splat(2))) & m03) << @as(V16, @splat(4)));
                const q3u: V16 = (ql0 >> @as(V16, @splat(4))) | (((qh0 >> @as(V16, @splat(4))) & m03) << @as(V16, @splat(4)));
                const q4u: V16 = (ql1 >> @as(V16, @splat(4))) | (((qh0 >> @as(V16, @splat(6))) & m03) << @as(V16, @splat(4)));
                const I32 = @Vector(16, i32);
                const bias32: I32 = @splat(32);
                storeScaledI(out, base + y + l0 + 0, @as(I32, @intCast(q1u)) - bias32, d * s0);
                storeScaledI(out, base + y + l0 + 32, @as(I32, @intCast(q2u)) - bias32, d * s2);
                storeScaledI(out, base + y + l0 + 64, @as(I32, @intCast(q3u)) - bias32, d * s4);
                storeScaledI(out, base + y + l0 + 96, @as(I32, @intCast(q4u)) - bias32, d * s6);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

const std = @import("std");
const types = @import("../types.zig");
const layout = @import("../layout.zig");
const quantized = @import("../kernels/quantized.zig");
const bfloat = @import("../kernels/bfloat.zig");
const simd = @import("../../../simd.zig");
const row_dot = @import("dot.zig");

pub const KernelError = types.KernelError;

pub fn matMulRegisterTiledQ4_0(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    return matVecQ4_0BatchedRows(data, n, m, x, out, M, begin, end);
}

pub fn matMulRegisterTiledQ4_0ScaleSplit(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    return matVecQ4_0ScaleSplitBatchedRows(data, n, m, x, out, M, begin, end);
}

pub fn matMulRegisterTiledQ4_1(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    return matVecQ4_1BatchedRowsTiled(data, n, m, x, out, M, begin, end);
}

pub fn matMulRegisterTiledQ4_1ScaleSplit(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    return matVecQ4_1ScaleSplitBatchedRowsTiled(data, n, m, x, out, M, begin, end);
}

pub fn matMulRegisterTiledQ8_0(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    return matVecQ8_0BatchedRowsTiled(data, n, m, x, out, M, begin, end);
}

pub fn matVecQ4_0BatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var vacc: [32]V = undefined;
        for (0..M) |tok| vacc[tok] = @splat(0);
        const row = data[i * row_bytes ..][0..row_bytes];
        for (0..row_blocks) |rb| {
            const block = row[rb * 18 ..][0..18];
            const d: f32 = @floatCast(readF16(block[0..2]));
            const qs: *align(1) const [16]u8 = @ptrCast(block[2..18].ptr);
            const bytes0: @Vector(8, u8) = .{ qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], qs[7] };
            const bytes1: @Vector(8, u8) = .{ qs[8], qs[9], qs[10], qs[11], qs[12], qs[13], qs[14], qs[15] };
            const lo_mask: @Vector(8, u8) = @splat(0x0F);
            const hi_shift: @Vector(8, u8) = @splat(4);
            const lo0: @Vector(8, i8) = @bitCast(bytes0 & lo_mask);
            const hi0: @Vector(8, i8) = @bitCast(bytes0 >> hi_shift);
            const lo1: @Vector(8, i8) = @bitCast(bytes1 & lo_mask);
            const hi1: @Vector(8, i8) = @bitCast(bytes1 >> hi_shift);
            const bias: @Vector(8, i8) = @splat(8);
            const q0: V = @as(V, @floatFromInt(lo0 - bias));
            const q1: V = @as(V, @floatFromInt(hi0 - bias));
            const q2: V = @as(V, @floatFromInt(lo1 - bias));
            const q3: V = @as(V, @floatFromInt(hi1 - bias));
            const dv: V = @splat(d);
            for (0..M) |tok| {
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
                const x1: V = @as(*align(1) const V, @ptrCast(&xb[8])).*;
                const x2: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
                const x3: V = @as(*align(1) const V, @ptrCast(&xb[24])).*;
                var s = q0 * x0;
                s = simd.fma(8, q1, x2, s);
                s = simd.fma(8, q2, x1, s);
                s = simd.fma(8, q3, x3, s);
                vacc[tok] = simd.fma(8, dv, s, vacc[tok]);
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, vacc[tok]);
        }
    }
}

fn q4_0BlockVecBatched(qs: *align(1) const [16]u8, x: []const f32, n: usize, rb: usize, M: usize) [32]@Vector(8, f32) {
    const V = @Vector(8, f32);
    const bytes0: @Vector(8, u8) = .{ qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], qs[7] };
    const bytes1: @Vector(8, u8) = .{ qs[8], qs[9], qs[10], qs[11], qs[12], qs[13], qs[14], qs[15] };
    const lo_mask: @Vector(8, u8) = @splat(0x0F);
    const hi_shift: @Vector(8, u8) = @splat(4);

    const lo0: @Vector(8, i8) = @bitCast(bytes0 & lo_mask);
    const hi0: @Vector(8, i8) = @bitCast(bytes0 >> hi_shift);
    const lo1: @Vector(8, i8) = @bitCast(bytes1 & lo_mask);
    const hi1: @Vector(8, i8) = @bitCast(bytes1 >> hi_shift);
    const bias: @Vector(8, i8) = @splat(8);

    const q0: V = @as(V, @floatFromInt(lo0 - bias));
    const q1: V = @as(V, @floatFromInt(hi0 - bias));
    const q2: V = @as(V, @floatFromInt(lo1 - bias));
    const q3: V = @as(V, @floatFromInt(hi1 - bias));

    var r: [32]V = undefined;
    for (0..M) |tok| {
        const xb = x[tok * n + rb * 32 ..][0..32];
        const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
        const x1: V = @as(*align(1) const V, @ptrCast(&xb[8])).*;
        const x2: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
        const x3: V = @as(*align(1) const V, @ptrCast(&xb[24])).*;
        r[tok] = q0 * x0 + q1 * x2 + q2 * x1 + q3 * x3;
    }
    return r;
}

pub fn matVecQ4_0ScaleSplitBatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const scales_total = m * row_blocks * 2;
    const scales = data[0..scales_total];
    const quants = data[scales_total..need];

    // Deferred reduction: vector accumulators, one horizontal reduce per token.
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var acc0: [32]V = undefined;
        var acc1: [32]V = undefined;
        var acc2: [32]V = undefined;
        var acc3: [32]V = undefined;
        for (0..M) |tok| {
            acc0[tok] = @splat(0);
            acc1[tok] = @splat(0);
            acc2[tok] = @splat(0);
            acc3[tok] = @splat(0);
        }
        const s_row = scales[i * row_blocks * 2 ..][0 .. row_blocks * 2];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];

        for (0..row_blocks) |rb| {
            const d: f32 = @floatCast(readF16(s_row[rb * 2 ..][0..2]));
            const qs: *align(1) const [16]u8 = @ptrCast(q_row[rb * 16 ..][0..16].ptr);
            const vecs = q4_0BlockVecBatched(qs, x, n, rb, M);
            const dv: V = @splat(d);
            for (0..M) |tok| {
                const scaled = vecs[tok] * dv;
                switch (rb % 4) {
                    0 => acc0[tok] += scaled,
                    1 => acc1[tok] += scaled,
                    2 => acc2[tok] += scaled,
                    else => acc3[tok] += scaled,
                }
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, (acc0[tok] + acc1[tok]) + (acc2[tok] + acc3[tok]));
        }
    }
}

pub fn matVecQ4_1BatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (M >= 4) return matVecQ4_1BatchedRowsTiled(data, n, m, x, out, M, begin, end);
    return matVecQ4_1BatchedRowsScalar(data, n, m, x, out, M, begin, end);
}

fn matVecQ4_1BatchedRowsScalar(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    // Deferred reduction: vector accumulators, one horizontal reduce per token.
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var accv: [32]V = undefined;
        var msum: [32]f32 = undefined;
        for (0..M) |tok| {
            accv[tok] = @splat(0);
            msum[tok] = 0;
        }
        const row = data[i * row_bytes ..][0..row_bytes];
        for (0..row_blocks) |rb| {
            const block = row[rb * 20 ..][0..20];
            const d: f32 = @floatCast(readF16(block[0..2]));
            const mn: f32 = @floatCast(readF16(block[2..4]));
            const qs: *align(1) const [16]u8 = @ptrCast(block[4..20].ptr);
            const dv: V = @splat(d);
            for (0..M) |tok| {
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                accv[tok] = @mulAdd(V, dv, layout.q4_1BlockVec(qs, xb), accv[tok]);
                msum[tok] += mn * layout.xBlockSum(xb);
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, accv[tok]) + msum[tok];
        }
    }
}

fn matVecQ4_1BatchedRowsTiled(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var dacc: [32]V = undefined;
        var sacc: [32]V = undefined;
        for (0..M) |tok| {
            dacc[tok] = @splat(0);
            sacc[tok] = @splat(0);
        }
        const row = data[i * row_bytes ..][0..row_bytes];
        for (0..row_blocks) |rb| {
            const block = row[rb * 20 ..][0..20];
            const d: f32 = @floatCast(readF16(block[0..2]));
            const mn: f32 = @floatCast(readF16(block[2..4]));
            const qs: *align(1) const [16]u8 = @ptrCast(block[4..20].ptr);
            const bytes0: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(qs)).*;
            const bytes1: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(&qs[8])).*;
            const lo_mask: @Vector(8, u8) = @splat(0x0F);
            const hi_shift: @Vector(8, u8) = @splat(4);
            const lo0: V = @as(V, @floatFromInt(bytes0 & lo_mask));
            const hi0: V = @as(V, @floatFromInt(bytes0 >> hi_shift));
            const lo1: V = @as(V, @floatFromInt(bytes1 & lo_mask));
            const hi1: V = @as(V, @floatFromInt(bytes1 >> hi_shift));
            const dv: V = @splat(d);
            const mv: V = @splat(mn);
            for (0..M) |tok| {
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
                const x1: V = @as(*align(1) const V, @ptrCast(&xb[8])).*;
                const x2: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
                const x3: V = @as(*align(1) const V, @ptrCast(&xb[24])).*;
                var dsum = lo0 * x0;
                dsum = simd.fma(8, hi0, x2, dsum);
                dsum = simd.fma(8, lo1, x1, dsum);
                dsum = simd.fma(8, hi1, x3, dsum);
                dacc[tok] = simd.fma(8, dv, dsum, dacc[tok]);
                sacc[tok] = simd.fma(8, mv, (x0 + x1) + (x2 + x3), sacc[tok]);
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, dacc[tok]) + @reduce(.Add, sacc[tok]);
        }
    }
}

pub fn matVecQ4_1ScaleSplitBatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (M >= 4) return matVecQ4_1ScaleSplitBatchedRowsTiled(data, n, m, x, out, M, begin, end);
    return matVecQ4_1ScaleSplitBatchedRowsScalar(data, n, m, x, out, M, begin, end);
}

fn matVecQ4_1ScaleSplitBatchedRowsScalar(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const meta_total = m * row_blocks * 4;
    const meta = data[0..meta_total];
    const quants = data[meta_total..need];

    // Deferred reduction: vector accumulators, one horizontal reduce per token.
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var accv: [32]V = undefined;
        var msum: [32]f32 = undefined;
        for (0..M) |tok| {
            accv[tok] = @splat(0);
            msum[tok] = 0;
        }
        const m_row = meta[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];

        for (0..row_blocks) |rb| {
            const qs: *align(1) const [16]u8 = @ptrCast(q_row[rb * 16 ..][0..16].ptr);
            const m_bytes = m_row[rb * 4 ..][0..4];
            const d: f32 = @floatCast(readF16(m_bytes[0..2]));
            const mn: f32 = @floatCast(readF16(m_bytes[2..4]));
            const dv: V = @splat(d);
            for (0..M) |tok| {
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                accv[tok] = @mulAdd(V, dv, layout.q4_1ScaleSplitBlockVec(qs, xb), accv[tok]);
                msum[tok] += mn * layout.xBlockSum(xb);
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, accv[tok]) + msum[tok];
        }
    }
}

fn matVecQ4_1ScaleSplitBatchedRowsTiled(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const meta_total = m * row_blocks * 4;
    const meta = data[0..meta_total];
    const quants = data[meta_total..need];

    for (begin..end) |i| {
        const m_row = meta[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row = quants[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        const V = @Vector(8, f32);
        var tok: usize = 0;
        while (tok + 4 <= M) : (tok += 4) {
            var acc: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var msum: [4]f32 = .{ 0, 0, 0, 0 };
            for (0..row_blocks) |rb| {
                const qs: *align(1) const [16]u8 = @ptrCast(q_row[rb * 16 ..][0..16].ptr);
                const m_bytes = m_row[rb * 4 ..][0..4];
                const mn: f32 = @floatCast(readF16(m_bytes[2..4]));
                const r = q4_1ScaleSplitBlockVec4(m_bytes, qs, x, n, rb, tok);
                inline for (0..4) |t| {
                    acc[t] += r[t];
                    const xb: *align(1) const [32]f32 = @ptrCast(x[(tok + t) * n + rb * 32 ..][0..32].ptr);
                    msum[t] += mn * layout.xBlockSum(xb);
                }
            }
            inline for (0..4) |t| out[(tok + t) * m + i] = @reduce(.Add, acc[t]) + msum[t];
        }
        while (tok < M) : (tok += 1) {
            var accv: V = @splat(0);
            var msum: f32 = 0;
            for (0..row_blocks) |rb| {
                const qs: *align(1) const [16]u8 = @ptrCast(q_row[rb * 16 ..][0..16].ptr);
                const m_bytes = m_row[rb * 4 ..][0..4];
                const d: f32 = @floatCast(readF16(m_bytes[0..2]));
                const mn: f32 = @floatCast(readF16(m_bytes[2..4]));
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                accv = @mulAdd(V, @as(V, @splat(d)), layout.q4_1ScaleSplitBlockVec(qs, xb), accv);
                msum += mn * layout.xBlockSum(xb);
            }
            out[tok * m + i] = @reduce(.Add, accv) + msum;
        }
    }
}

pub fn matVecQ8_0BatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (M >= 4) return matVecQ8_0BatchedRowsTiled(data, n, m, x, out, M, begin, end);
    return matVecQ8_0BatchedRowsScalar(data, n, m, x, out, M, begin, end);
}

fn matVecQ8_0BatchedRowsScalar(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 34;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    // Deferred reduction: vector accumulators, one horizontal reduce per token.
    const V = @Vector(16, f32);
    for (begin..end) |i| {
        var acc0: [32]V = undefined;
        var acc1: [32]V = undefined;
        var acc2: [32]V = undefined;
        var acc3: [32]V = undefined;
        for (0..M) |tok| {
            acc0[tok] = @splat(0);
            acc1[tok] = @splat(0);
            acc2[tok] = @splat(0);
            acc3[tok] = @splat(0);
        }
        const row = data[i * row_bytes ..][0..row_bytes];
        for (0..row_blocks) |rb| {
            const block = row[rb * 34 ..][0..34];
            const d: f32 = @floatCast(readF16(block[0..2]));
            const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
            const bytes0: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(qs)).*;
            const bytes1: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qs[16])).*;
            const q0: V = @floatFromInt(bytes0);
            const q1: V = @floatFromInt(bytes1);
            const dv: V = @splat(d);

            for (0..M) |tok| {
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
                const x1: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
                const scaled = (q0 * x0 + q1 * x1) * dv;
                switch (rb % 4) {
                    0 => acc0[tok] += scaled,
                    1 => acc1[tok] += scaled,
                    2 => acc2[tok] += scaled,
                    else => acc3[tok] += scaled,
                }
            }
        }
        for (0..M) |tok| {
            out[tok * m + i] = @reduce(.Add, (acc0[tok] + acc1[tok]) + (acc2[tok] + acc3[tok]));
        }
    }
}

fn matVecQ8_0BatchedRowsTiled(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 34;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    // Deferred reduction: vector accumulators, one horizontal reduce per token.
    const V = @Vector(16, f32);
    for (begin..end) |i| {
        const row = data[i * row_bytes ..][0..row_bytes];
        var tok: usize = 0;
        while (tok + 4 <= M) : (tok += 4) {
            var acc: [4]V = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            for (0..row_blocks) |rb| {
                const block = row[rb * 34 ..][0..34];
                const d: f32 = @floatCast(readF16(block[0..2]));
                const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
                const r = q8_0BlockVec4(qs, d, x, n, rb, tok);
                inline for (0..4) |t| acc[t] += r[t];
            }
            inline for (0..4) |t| out[(tok + t) * m + i] = @reduce(.Add, acc[t]);
        }
        while (tok < M) : (tok += 1) {
            var accv: V = @splat(0);
            for (0..row_blocks) |rb| {
                const block = row[rb * 34 ..][0..34];
                const d: f32 = @floatCast(readF16(block[0..2]));
                const qs: *align(1) const [32]i8 = @ptrCast(block[2..34].ptr);
                const xb: *align(1) const [32]f32 = @ptrCast(x[tok * n + rb * 32 ..][0..32].ptr);
                const bytes0: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(qs)).*;
                const bytes1: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qs[16])).*;
                const q0: V = @floatFromInt(bytes0);
                const q1: V = @floatFromInt(bytes1);
                const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
                const x1: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
                accv += (q0 * x0 + q1 * x1) * @as(V, @splat(d));
            }
            out[tok * m + i] = @reduce(.Add, accv);
        }
    }
}

pub fn matVecQ6_KBatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (n % 256 != 0) return error.ShapeMismatch;
    const row_blocks = n / 256;
    const row_bytes = row_blocks * 210;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    for (begin..end) |i| {
        const row = data[i * row_bytes ..][0..row_bytes];
        for (0..M) |tok| {
            out[tok * m + i] = try row_dot.dot(quantized.Q6_K, row, x[tok * n .. (tok + 1) * n], null);
        }
    }
}

pub fn matVecBF16BatchedRows(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    if (data.len < n * m * 2) return error.TruncatedData;
    if (M > 32) return error.ShapeMismatch;
    if (x.len != M * n) return error.ShapeMismatch;
    if (out.len != M * m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    for (0..M) |tok| {
        try bfloat.matVecBF16(data, n, m, x[tok * n .. (tok + 1) * n], out[tok * m .. (tok + 1) * m], begin, end);
    }
}

fn readF16(data: []const u8) f16 {
    const raw = std.mem.readInt(u16, data[0..2], .little);
    return @bitCast(raw);
}

fn q4_1ScaleSplitBlockVec4(m_bytes: []const u8, qs: *align(1) const [16]u8, x: []const f32, n: usize, rb: usize, tok: usize) [4]@Vector(8, f32) {
    const V = @Vector(8, f32);
    const d: f32 = @floatCast(readF16(m_bytes[0..2]));
    const dv: V = @splat(d);
    var r: [4]V = undefined;
    inline for (0..4) |t| {
        const xb: *align(1) const [32]f32 = @ptrCast(x[(tok + t) * n + rb * 32 ..][0..32].ptr);
        r[t] = layout.q4_1ScaleSplitBlockVec(qs, xb) * dv;
    }
    return r;
}

fn q8_0BlockVec4(qs: *align(1) const [32]i8, d: f32, x: []const f32, n: usize, rb: usize, tok: usize) [4]@Vector(16, f32) {
    const V = @Vector(16, f32);
    const bytes0: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(qs)).*;
    const bytes1: @Vector(16, i8) = @as(*align(1) const @Vector(16, i8), @ptrCast(&qs[16])).*;
    const q0: V = @floatFromInt(bytes0);
    const q1: V = @floatFromInt(bytes1);
    const dv: V = @splat(d);
    var r: [4]V = undefined;
    inline for (0..4) |t| {
        const xb: *align(1) const [32]f32 = @ptrCast(x[(tok + t) * n + rb * 32 ..][0..32].ptr);
        const x0: V = @as(*align(1) const V, @ptrCast(xb)).*;
        const x1: V = @as(*align(1) const V, @ptrCast(&xb[16])).*;
        r[t] = (q0 * x0 + q1 * x1) * dv;
    }
    return r;
}

const std = @import("std");
const TensorView = @import("../../../data/tensor_view.zig").TensorView;
const TensorLayout = @import("../../../data/tensor_view.zig").TensorLayout;
const ggml = @import("../../../gguf/format.zig");
const dequant = @import("../quant/root.zig");
const simd = @import("../../simd.zig");
const types = @import("types.zig");
const layout = @import("layout.zig");
const quantized = @import("kernels/quantized.zig");
const batched = @import("form/batched.zig");
const single = @import("form/single.zig");
const root = @import("root.zig");

/// The q4_0 integer oracle: quantize the activation to q8_0 once per call, then
/// dot each block with `vpmaddwd`-style integer SIMD. `sum(w*x) =
/// d_w*d_x*(sum(nibble*qx) - 8*sum(qx))`; the second term is row-independent and
/// precomputed per block. Falls back to the f32 path when the activation is too
/// large to cache.
pub fn q4_0IntOracle(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) types.KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    if (row_blocks > types.max_activation_cache_blocks) {
        return quantized.matVecQ4_0U4(data, n, m, x, out, begin, end);
    }

    var qx_buf: [types.max_activation_cache_blocks * 32]i8 = undefined;
    var dx_buf: [types.max_activation_cache_blocks]f32 = undefined;
    var dxs_buf: [types.max_activation_cache_blocks]f32 = undefined;
    layout.quantizeActivationQ8(x, row_blocks, qx_buf[0..n], dx_buf[0..row_blocks], dxs_buf[0..row_blocks]);

    const V = @Vector(8, f32);
    for (begin..end) |i| {
        const row = data[i * row_bytes ..][0..row_bytes];
        var acc: V = @splat(0);
        var corr: f32 = 0;
        for (0..row_blocks) |rb| {
            const block = row[rb * 18 ..][0..18];
            const dw: f32 = @floatCast(layout.readF16(block[0..2]));
            const iv = layout.q4_0IntBlockVec(8, block, @ptrCast(&qx_buf[rb * 32]));
            acc = simd.fma(8, @as(V, @splat(dw * dx_buf[rb])), @as(V, @floatFromInt(iv)), acc);
            corr -= dw * dxs_buf[rb];
        }
        out[i] = @reduce(.Add, acc) + corr;
    }
}

/// The q4_K integer oracle: quantize the activation to q8_0 once per call, then
/// dot each super-block with integer SIMD, folding the packed scale/min (cf.
/// `q4_0IntOracle`). Falls back to the f32 path when the activation is too large
/// to cache.
pub fn q4_KIntOracle(data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) types.KernelError!void {
    if (n % 256 != 0) return error.ShapeMismatch;
    const row_blocks = n / 256;
    const row_bytes = row_blocks * 144;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;
    const chunks = n / 32;
    if (chunks > types.max_activation_cache_blocks) {
        return quantized.Q4_K.production_fp(data, n, m, x, out, begin, end);
    }

    var qx_buf: [types.max_activation_cache_blocks * 32]i8 = undefined;
    var dx_buf: [types.max_activation_cache_blocks]f32 = undefined;
    var sum_buf: [types.max_activation_cache_blocks]i32 = undefined;
    layout.quantizeActivationQ8Sum(x, chunks, qx_buf[0..n], dx_buf[0..chunks], sum_buf[0..chunks]);

    const V = @Vector(8, f32);
    for (begin..end) |i| {
        const row = data[i * row_bytes ..][0..row_bytes];
        var acc: V = @splat(0);
        var corr: f32 = 0;
        for (0..row_blocks) |rb| {
            layout.q4KBlockIntAccum(
                8,
                row[rb * 144 ..][0..144],
                qx_buf[rb * 8 * 32 ..][0..256],
                dx_buf[rb * 8 ..][0..8],
                sum_buf[rb * 8 ..][0..8],
                &acc,
                &corr,
            );
        }
        out[i] = @reduce(.Add, acc) - corr;
    }
}

pub fn expectMatMulRowsMatchesMatVec(
    dims: []const u64,
    ggml_type: ggml.GGMLType,
    data: []const u8,
    tensor_layout: TensorLayout,
    acts: []const f32,
    comptime M: usize,
    begin: usize,
    end: usize,
) !void {
    const n: usize = @intCast(dims[0]);
    const m: usize = @intCast(dims[1]);
    const view = TensorView{
        .name = "w",
        .dims = dims,
        .ggml_type = ggml_type,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
        .layout = tensor_layout,
    };
    var got: [M * 5]f32 = [_]f32{0} ** (M * 5);
    var want: [M * 5]f32 = [_]f32{0} ** (M * 5);

    try single.matMulRows(&view, acts, &got, begin, end);
    for (0..M) |tok| {
        try single.matVecRows(&view, acts[tok * n ..][0..n], want[tok * m ..][0..m], begin, end);
    }

    for (0..M) |tok| {
        for (begin..end) |row| {
            try std.testing.expectApproxEqAbs(want[tok * m + row], got[tok * m + row], 1e-5);
        }
    }
}

pub fn expectRegisterTiledMatchesMatVec(
    dims: []const u64,
    ggml_type: ggml.GGMLType,
    data: []const u8,
    tensor_layout: TensorLayout,
    acts: []const f32,
    comptime M: usize,
    begin: usize,
    end: usize,
) !void {
    const n: usize = @intCast(dims[0]);
    const m: usize = @intCast(dims[1]);
    const view = TensorView{
        .name = "w",
        .dims = dims,
        .ggml_type = ggml_type,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
        .layout = tensor_layout,
    };
    var got: [M * 5]f32 = [_]f32{0} ** (M * 5);
    var want: [M * 5]f32 = [_]f32{0} ** (M * 5);

    switch (tensor_layout) {
        .q4_0_scale_split => try batched.matMulRegisterTiledQ4_0ScaleSplit(data, n, m, acts, &got, M, begin, end),
        .q4_1_scale_split => try batched.matMulRegisterTiledQ4_1ScaleSplit(data, n, m, acts, &got, M, begin, end),
        .native_quant => switch (ggml_type) {
            .q4_0 => try batched.matMulRegisterTiledQ4_0(data, n, m, acts, &got, M, begin, end),
            .q4_1 => try batched.matMulRegisterTiledQ4_1(data, n, m, acts, &got, M, begin, end),
            .q8_0 => try batched.matMulRegisterTiledQ8_0(data, n, m, acts, &got, M, begin, end),
            else => return error.UnsupportedType,
        },
        else => return error.UnsupportedType,
    }
    for (0..M) |tok| {
        try single.matVecRows(&view, acts[tok * n ..][0..n], want[tok * m ..][0..m], begin, end);
    }

    for (0..M) |tok| {
        for (begin..end) |row| {
            try std.testing.expectApproxEqAbs(want[tok * m + row], got[tok * m + row], 1e-5);
        }
    }
}

pub fn fillQ4_0Fixture(data: []u8, n: usize, m: usize) void {
    const row_blocks = n / 32;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var nib: [16]u8 = undefined;
            for (&nib, 0..) |*b, i| b.* = @truncate((row * 11 + rb * 7 + i * 3) & 0xff);
            const block = dequant.encodeQ4_0Block(@as(f16, 0.125), &nib);
            @memcpy(data[(row * row_blocks + rb) * 18 ..][0..18], &block);
        }
    }
}

pub fn fillQ4_1Fixture(data: []u8, n: usize, m: usize) void {
    const row_blocks = n / 32;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var nib: [16]u8 = undefined;
            for (&nib, 0..) |*b, i| b.* = @truncate((row * 13 + rb * 5 + i * 9) & 0xff);
            const block = dequant.encodeQ4_1Block(@as(f16, 0.0625), @as(f16, -0.25), &nib);
            @memcpy(data[(row * row_blocks + rb) * 20 ..][0..20], &block);
        }
    }
}

pub fn fillQ4_KFixture(data: []u8, n: usize, m: usize) void {
    const row_blocks = n / 256;
    var seed: u32 = 0x1357_2468;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const block = data[(row * row_blocks + rb) * 144 ..][0..144];
            const d: f16 = @floatCast(0.03 + 0.002 * @as(f32, @floatFromInt((row + rb) % 7)));
            const dm: f16 = @floatCast(0.01 + 0.001 * @as(f32, @floatFromInt((row + rb) % 5)));
            std.mem.writeInt(u16, block[0..2], @bitCast(d), .little);
            std.mem.writeInt(u16, block[2..4], @bitCast(dm), .little);
            for (4..144) |i| {
                seed = seed *% 1664525 +% 1013904223;
                block[i] = @truncate(seed >> 13);
            }
        }
    }
}

pub fn fillQ8_0Fixture(data: []u8, n: usize, m: usize) void {
    const row_blocks = n / 32;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var qs: [32]i8 = undefined;
            for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((row * 17 + rb * 3 + i) % 23)) - 11);
            const block = dequant.encodeQ8_0Block(@as(f16, 0.03125), &qs);
            @memcpy(data[(row * row_blocks + rb) * 34 ..][0..34], &block);
        }
    }
}

pub fn fillQ6_KFixture(data: []u8, n: usize, m: usize) void {
    const row_blocks = n / 256;
    var seed: u32 = 0x6b6b_1234;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const block = data[(row * row_blocks + rb) * 210 ..][0..210];
            for (0..192) |i| {
                seed = seed *% 1664525 +% 1013904223;
                block[i] = @truncate(seed >> 13);
            }
            for (0..16) |i| {
                const s: i32 = @as(i32, @intCast((row * 3 + rb + i) % 7)) - 3;
                block[192 + i] = @as(u8, @bitCast(@as(i8, @intCast(s))));
            }
            std.mem.writeInt(u16, block[208..210], @bitCast(@as(f16, 0.03)), .little);
        }
    }
}

pub fn fillF32Fixture(data: []u8, n: usize, m: usize) void {
    var seed: u32 = 0x1357_9bdf;
    for (0..m * n) |idx| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        const v: f32 = (t * 2.0 - 1.0) * 0.5;
        std.mem.writeInt(u32, data[idx * 4 ..][0..4], @bitCast(v), .little);
    }
}

pub fn fillF16Fixture(data: []u8, n: usize, m: usize) void {
    var seed: u32 = 0x2468_ace1;
    for (0..m * n) |idx| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        const v: f16 = @floatCast((t * 2.0 - 1.0) * 0.5);
        std.mem.writeInt(u16, data[idx * 2 ..][0..2], @bitCast(v), .little);
    }
}

pub fn fillBF16Fixture(data: []u8, n: usize, m: usize) void {
    var seed: u32 = 0x6b6b_c0de;
    for (0..m * n) |idx| {
        seed = seed *% 1664525 +% 1013904223;
        const t: f32 = @as(f32, @floatFromInt((seed >> 20) & 0xFF)) / 255.0;
        const v: f32 = (t * 2.0 - 1.0) * 0.5;
        std.mem.writeInt(u16, data[idx * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
    }
}

pub fn splitQ4_0(src: []const u8, dst: []u8, n: usize, m: usize) void {
    const row_blocks = n / 32;
    const meta_total = m * row_blocks * 2;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const block = src[(row * row_blocks + rb) * 18 ..][0..18];
            @memcpy(dst[(row * row_blocks + rb) * 2 ..][0..2], block[0..2]);
            @memcpy(dst[meta_total + (row * row_blocks + rb) * 16 ..][0..16], block[2..18]);
        }
    }
}

pub fn splitQ4_1(src: []const u8, dst: []u8, n: usize, m: usize) void {
    const row_blocks = n / 32;
    const meta_total = m * row_blocks * 4;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const block = src[(row * row_blocks + rb) * 20 ..][0..20];
            @memcpy(dst[(row * row_blocks + rb) * 4 ..][0..4], block[0..4]);
            @memcpy(dst[meta_total + (row * row_blocks + rb) * 16 ..][0..16], block[4..20]);
        }
    }
}

/// Validate one weight family's generated catalog against the kernel x variant
/// definition + P1 prune: the order (the family's kernels in kernel-catalog
/// order), the axes, the plan id (offset by the catalog base), the name, the
/// parse round-trip, the prefetch tag and the quant applicability.
pub fn expectCatalog(comptime tag: types.KernelTag, comptime cands: []const types.Candidate) !void {
    const testing = std.testing;
    const native_width: u8 = @intCast(simd.native.lanes());
    const base: u16 = @intFromEnum(cands[0].plan);
    var expected: usize = 0;
    var saw_p0 = false;
    var saw_p1 = false;
    inline for (types.kernel_catalog) |ks| {
        if (ks.tag != tag) continue;
        const dot_kind = ks.dot;
        const prefix = ks.prefix;
        for (types.grid_widths) |width| {
            if (dot_kind == .fp and width > native_width) continue;
            if (dot_kind == .int and width != native_width) continue;
            for (types.grid_tiles) |tile| {
                for (types.grid_unrolls) |unroll| {
                    if (@as(usize, tile) * @as(usize, unroll) * ((@as(usize, width) + 7) / 8) > types.accumulator_budget) continue;
                    for ([_]u8{ 0, 1 }) |p| {
                        const cand = cands[expected];
                        try testing.expectEqual(tag, cand.kernel.tag);
                        try testing.expectEqual(dot_kind, cand.kernel.dot);
                        try testing.expectEqual(width, cand.variant.width);
                        try testing.expectEqual(tile, cand.variant.tile);
                        try testing.expectEqual(unroll, cand.variant.unroll);
                        try testing.expectEqual(p != 0, cand.variant.prefetch);
                        try testing.expectEqual(base + @as(u16, @intCast(expected)), @intFromEnum(cand.plan));
                        var name_buf: [48]u8 = undefined;
                        const want = try std.fmt.bufPrint(&name_buf, "{s}_w{d}t{d}un{d}_p{d}", .{
                            prefix, width, tile, unroll, p,
                        });
                        try testing.expectEqualStrings(want, types.planLabel(cand.plan));
                        try testing.expectEqual(cand.plan, types.parseMatVecPlan(want));
                        try testing.expectEqual(cand.variant.prefetch, types.isPrefetchPlan(cand.plan));
                        if (cand.variant.prefetch) saw_p1 = true else saw_p0 = true;
                        expected += 1;
                    }
                }
            }
        }
    }
    try testing.expectEqual(expected, cands.len);
    try testing.expect(saw_p0 and saw_p1);

    // The generated applicability accepts the quant type and fits the machine.
    inline for (cands) |cand| {
        const f = root.planApplicable(cand.plan);
        try testing.expect(f(@intFromEnum(root.kernelGgmlType(tag)), 64, 64, native_width));
    }
}

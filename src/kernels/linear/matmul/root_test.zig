const std = @import("std");
const root = @import("root.zig");
const dequant = @import("../quant/root.zig");
const matvec = @import("../matvec/root.zig");
const TensorView = @import("../../../data/tensor_view.zig").TensorView;

const Args = root.Args;
const AtomicId = root.AtomicId;

test "matmul rows and blocked match per-row gemv (f32)" {
    const dims = [_]u64{ 64, 5 };
    const M: usize = 7;
    var w: [64 * 5]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.25;
    const data = std.mem.sliceAsBytes(w[0..]);
    const view = TensorView{ .name = "w", .dims = &dims, .ggml_type = .f32, .offset = 0, .byte_length = data.len, .mapped_slice = data };
    try expectMatmulMatches(&view, 64, 5, M);
}

test "matmul rows and blocked match per-row gemv (q4_0/q4_1/q8_0)" {
    const dims = [_]u64{ 64, 5 };
    const M: usize = 7;

    var q4_0_data: [5 * (64 / 32) * 18]u8 = undefined;
    fillQ4_0(&q4_0_data, 64, 5);
    const v40 = TensorView{ .name = "w", .dims = &dims, .ggml_type = .q4_0, .offset = 0, .byte_length = q4_0_data.len, .mapped_slice = &q4_0_data };
    try expectMatmulMatches(&v40, 64, 5, M);

    var q4_1_data: [5 * (64 / 32) * 20]u8 = undefined;
    fillQ4_1(&q4_1_data, 64, 5);
    const v41 = TensorView{ .name = "w", .dims = &dims, .ggml_type = .q4_1, .offset = 0, .byte_length = q4_1_data.len, .mapped_slice = &q4_1_data };
    try expectMatmulMatches(&v41, 64, 5, M);

    var q8_0_data: [5 * (64 / 32) * 34]u8 = undefined;
    fillQ8_0(&q8_0_data, 64, 5);
    const v80 = TensorView{ .name = "w", .dims = &dims, .ggml_type = .q8_0, .offset = 0, .byte_length = q8_0_data.len, .mapped_slice = &q8_0_data };
    try expectMatmulMatches(&v80, 64, 5, M);
}

fn expectMatmulMatches(view: *const TensorView, comptime K: usize, comptime N: usize, comptime M: usize) !void {
    try std.testing.expectEqual(@as(u64, K), view.dims[0]);
    try std.testing.expectEqual(@as(u64, N), view.dims[1]);
    var acts: [M * K]f32 = undefined;
    for (&acts, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.0625;
    }
    var want: [M * N]f32 = undefined;
    var got_rows: [M * N]f32 = [_]f32{0} ** (M * N);
    var got_blocked: [M * N]f32 = [_]f32{0} ** (M * N);
    var scratch: [K]f32 = undefined;

    for (0..M) |tok| {
        try matvec.matVecRows(view, acts[tok * K ..][0..K], want[tok * N ..][0..N], 0, N);
    }

    var rows_args = Args{ .view = view, .x = &acts, .out = &got_rows, .m = M, .n = N, .k = K, .begin = 0, .end = N };
    try root.set.run(@intFromEnum(AtomicId.rows), &rows_args);
    var blocked_args = Args{ .view = view, .x = &acts, .out = &got_blocked, .m = M, .n = N, .k = K, .begin = 0, .end = N, .scratch = &scratch };
    try root.set.run(@intFromEnum(AtomicId.blocked), &blocked_args);

    for (0..M * N) |i| {
        try std.testing.expectApproxEqAbs(want[i], got_rows[i], 1e-5);
        try std.testing.expectApproxEqAbs(want[i], got_blocked[i], 1e-5);
    }
}

fn fillQ4_0(data: []u8, n: usize, m: usize) void {
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

fn fillQ4_1(data: []u8, n: usize, m: usize) void {
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

fn fillQ8_0(data: []u8, n: usize, m: usize) void {
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

test "matmul rows and blocked fall back for transformed layouts" {
    const dims = [_]u64{ 64, 5 };
    const M: usize = 4;
    var q4_0_data: [5 * (64 / 32) * 18]u8 = undefined;
    fillQ4_0(&q4_0_data, 64, 5);
    var split: [q4_0_data.len]u8 = undefined;
    const row_blocks = 64 / 32;
    const meta_total = 5 * row_blocks * 2;
    for (0..5) |row| {
        for (0..row_blocks) |rb| {
            const block = q4_0_data[(row * row_blocks + rb) * 18 ..][0..18];
            @memcpy(split[(row * row_blocks + rb) * 2 ..][0..2], block[0..2]);
            @memcpy(split[meta_total + (row * row_blocks + rb) * 16 ..][0..16], block[2..18]);
        }
    }
    const view = TensorView{ .name = "w", .dims = &dims, .ggml_type = .q4_0, .offset = 0, .byte_length = split.len, .mapped_slice = &split, .layout = .q4_0_scale_split };
    try expectMatmulMatches(&view, 64, 5, M);
}

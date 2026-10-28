const fixtures = @import("../fixtures.zig");

test "register tiled prefill kernels match repeated matvec" {
    const n: usize = 64;
    const m: usize = 5;
    const M: usize = 7;
    const dims = [_]u64{ n, m };

    var acts: [M * n]f32 = undefined;
    for (&acts, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 19)) - 9;
        v.* = @as(f32, @floatFromInt(x)) * 0.0625;
    }

    var q4_0_data: [m * (n / 32) * 18]u8 = undefined;
    fixtures.fillQ4_0Fixture(&q4_0_data, n, m);
    var q4_1_data: [m * (n / 32) * 20]u8 = undefined;
    fixtures.fillQ4_1Fixture(&q4_1_data, n, m);
    var q8_0_data: [m * (n / 32) * 34]u8 = undefined;
    fixtures.fillQ8_0Fixture(&q8_0_data, n, m);

    try fixtures.expectRegisterTiledMatchesMatVec(&dims, .q4_0, &q4_0_data, .native_quant, &acts, M, 1, 4);
    try fixtures.expectRegisterTiledMatchesMatVec(&dims, .q4_1, &q4_1_data, .native_quant, &acts, M, 1, 4);
    try fixtures.expectRegisterTiledMatchesMatVec(&dims, .q8_0, &q8_0_data, .native_quant, &acts, M, 1, 4);

    var q4_0_split: [q4_0_data.len]u8 = undefined;
    fixtures.splitQ4_0(&q4_0_data, &q4_0_split, n, m);
    try fixtures.expectRegisterTiledMatchesMatVec(&dims, .q4_0, &q4_0_split, .q4_0_scale_split, &acts, M, 1, 4);

    var q4_1_split: [q4_1_data.len]u8 = undefined;
    fixtures.splitQ4_1(&q4_1_data, &q4_1_split, n, m);
    try fixtures.expectRegisterTiledMatchesMatVec(&dims, .q4_1, &q4_1_split, .q4_1_scale_split, &acts, M, 1, 4);
}

test "matMulRows batched GEMV matches repeated matvec per quant type" {
    const n: usize = 64;
    const m: usize = 5;
    const M: usize = 7;
    const dims = [_]u64{ n, m };

    var acts: [M * n]f32 = undefined;
    for (&acts, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 23)) - 11;
        v.* = @as(f32, @floatFromInt(x)) * 0.0625;
    }

    var q4_0_data: [m * (n / 32) * 18]u8 = undefined;
    fixtures.fillQ4_0Fixture(&q4_0_data, n, m);
    var q4_1_data: [m * (n / 32) * 20]u8 = undefined;
    fixtures.fillQ4_1Fixture(&q4_1_data, n, m);
    var q8_0_data: [m * (n / 32) * 34]u8 = undefined;
    fixtures.fillQ8_0Fixture(&q8_0_data, n, m);

    try fixtures.expectMatMulRowsMatchesMatVec(&dims, .q4_0, &q4_0_data, .native_quant, &acts, M, 1, 4);
    try fixtures.expectMatMulRowsMatchesMatVec(&dims, .q4_1, &q4_1_data, .native_quant, &acts, M, 1, 4);
    try fixtures.expectMatMulRowsMatchesMatVec(&dims, .q8_0, &q8_0_data, .native_quant, &acts, M, 1, 4);

    var q4_0_split: [q4_0_data.len]u8 = undefined;
    fixtures.splitQ4_0(&q4_0_data, &q4_0_split, n, m);
    try fixtures.expectMatMulRowsMatchesMatVec(&dims, .q4_0, &q4_0_split, .q4_0_scale_split, &acts, M, 1, 4);

    var q4_1_split: [q4_1_data.len]u8 = undefined;
    fixtures.splitQ4_1(&q4_1_data, &q4_1_split, n, m);
    try fixtures.expectMatMulRowsMatchesMatVec(&dims, .q4_1, &q4_1_split, .q4_1_scale_split, &acts, M, 1, 4);
}

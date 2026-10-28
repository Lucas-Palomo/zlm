const std = @import("std");

/// Per-request sampling controls. `temperature == 0` selects the deterministic
/// greedy argmax (the historical behavior); `top_k == 0` disables top-k and
/// `top_p >= 1` disables nucleus filtering.
pub const Params = struct {
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
};

/// Index of the highest logit. Ties resolve to the lowest index, matching the
/// executor's historical argmax exactly.
pub fn argmax(logits: []const f32) u32 {
    if (logits.len == 0) return 0;
    var best: usize = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = i;
    }
    return @intCast(best);
}

fn lessLogit(logits: []const f32, a: u32, b: u32) bool {
    return logits[a] > logits[b];
}

/// Draw the next token from `logits` under `params`. Greedy when
/// `temperature <= 0`. `scratch` must hold at least `logits.len` u32s and is
/// reused across calls; the result is deterministic for a given `rng`.
pub fn sample(logits: []const f32, params: Params, scratch: []u32, rng: std.Random) u32 {
    if (params.temperature <= 0 or logits.len == 0) return argmax(logits);
    const n = logits.len;
    std.debug.assert(scratch.len >= n);

    for (scratch[0..n], 0..) |*s, i| s.* = @intCast(i);
    std.mem.sort(u32, scratch[0..n], logits, lessLogit);

    const k: usize = if (params.top_k > 0) @min(@as(usize, params.top_k), n) else n;
    const temp: f64 = params.temperature;
    const max_l: f64 = logits[scratch[0]];

    var total: f64 = 0;
    for (scratch[0..k]) |idx| total += @exp((@as(f64, logits[idx]) - max_l) / temp);
    if (total <= 0) return scratch[0];

    var nucleus: usize = k;
    if (params.top_p < 1.0) {
        var cum: f64 = 0;
        for (scratch[0..k], 0..) |idx, j| {
            cum += @exp((@as(f64, logits[idx]) - max_l) / temp) / total;
            if (cum >= params.top_p) {
                nucleus = j + 1;
                break;
            }
        }
    }

    var kept: f64 = 0;
    for (scratch[0..nucleus]) |idx| kept += @exp((@as(f64, logits[idx]) - max_l) / temp);
    const r = rng.float(f64) * kept;
    var cum: f64 = 0;
    for (scratch[0..nucleus]) |idx| {
        cum += @exp((@as(f64, logits[idx]) - max_l) / temp);
        if (r < cum) return idx;
    }
    return scratch[nucleus - 1];
}

test "temperature 0 is greedy argmax, ties to the lowest index" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    const cases = [_][]const f32{
        &[_]f32{ 1, 3, 2 },
        &[_]f32{ -1, -2, -3 },
        &[_]f32{ 0, 0, 0 },
        &[_]f32{ 5, 5, 4 },
    };
    var scratch: [8]u32 = undefined;
    for (cases) |logits| {
        try testing.expectEqual(argmax(logits), sample(logits, .{}, scratch[0..logits.len], rng));
    }
}

test "top_k 1 is greedy even with temperature" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const logits = [_]f32{ 0.2, 2.5, 1.1, -3.0 };
    var scratch: [4]u32 = undefined;
    const got = sample(&logits, .{ .temperature = 1.5, .top_k = 1 }, &scratch, rng);
    try testing.expectEqual(argmax(&logits), got);
}

test "top_p 1 keeps the whole support; top_p 0 is greedy" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    const logits = [_]f32{ 0, 0 };
    var scratch: [2]u32 = undefined;

    var saw_second = false;
    for (0..64) |_| {
        if (sample(&logits, .{ .temperature = 1, .top_p = 1 }, &scratch, rng) == 1) saw_second = true;
    }
    try testing.expect(saw_second);

    for (0..64) |_| {
        try testing.expectEqual(@as(u32, 0), sample(&logits, .{ .temperature = 1, .top_p = 0 }, &scratch, rng));
    }
}

test "seeded sampling is reproducible" {
    const testing = std.testing;
    const logits = [_]f32{ 0.5, 1.5, 1.0, -1.0 };
    var scratch: [4]u32 = undefined;

    var a = std.Random.DefaultPrng.init(42);
    var b = std.Random.DefaultPrng.init(42);
    const ra = a.random();
    const rb = b.random();
    for (0..32) |_| {
        const x = sample(&logits, .{ .temperature = 0.8, .top_k = 3, .top_p = 0.95 }, &scratch, ra);
        const y = sample(&logits, .{ .temperature = 0.8, .top_k = 3, .top_p = 0.95 }, &scratch, rb);
        try testing.expectEqual(x, y);
    }
}

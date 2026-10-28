const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const TensorView = @import("../data/tensor_view.zig").TensorView;
const ggml = @import("../gguf/format.zig");
const matvec = @import("../kernels/linear/matvec/root.zig");
const dequant = @import("../kernels/linear/quant/root.zig");

pub const MatVecPlan = matvec.MatVecPlan;

/// Timed calls per candidate (after one untimed warmup); the result is the
/// per-call average.
pub const bench_iters: usize = 5;

/// The generated candidate list of one weight family (the single source) plus
/// `production` last as the safe fallback.
fn planList(comptime cands: []const matvec.Candidate) [cands.len + 1]MatVecPlan {
    var arr: [cands.len + 1]MatVecPlan = undefined;
    for (cands, 0..) |cand, i| arr[i] = cand.plan;
    arr[cands.len] = .production;
    return arr;
}

const q4_0_candidates_with_production = planList(&matvec.q4_0_candidates);
const q4_1_candidates_with_production = planList(&matvec.q4_1_candidates);
const q4_k_candidates_with_production = planList(&matvec.q4_k_candidates);
const q8_0_candidates_with_production = planList(&matvec.q8_0_candidates);
const q6_k_candidates_with_production = planList(&matvec.q6_k_candidates);

/// Candidate variants for a quant type, `production` last as the safe
/// fallback. Types without interchangeable variants only get `production`.
fn candidatesFor(ggml_type: ggml.GGMLType) []const MatVecPlan {
    return switch (ggml_type) {
        .q4_0 => &q4_0_candidates_with_production,
        .q4_1 => &q4_1_candidates_with_production,
        .q8_0 => &q8_0_candidates_with_production,
        .q6_k => &q6_k_candidates_with_production,
        .q4_k => &q4_k_candidates_with_production,
        else => &[_]MatVecPlan{.production},
    };
}

/// Small deterministic activation values; independent of the shape so two
/// candidates always see the same input.
fn fillX(x: []f32) void {
    for (x, 0..) |*v, i| {
        const k: i32 = @intCast(i % 61);
        v.* = @as(f32, @floatFromInt(k - 30)) / 30.0;
    }
}

/// Monotonic nanoseconds; io-free so the harness keeps the requested API.
fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    if (rc != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A view the harness can benchmark: native layout, 2-D, mapped bytes.
fn validView(view: *const TensorView) bool {
    if (!view.layout.isNative()) return false;
    if (view.dims.len != 2) return false;
    if (view.dims[0] == 0 or view.dims[1] == 0) return false;
    return view.mapped_slice != null;
}

/// Time one `MatVecPlan` on `view`: one warmup call, then `iters` timed calls
/// over the full row range. Returns the mean nanoseconds per call.
pub fn benchMatvec(
    allocator: std.mem.Allocator,
    view: *const TensorView,
    plan: MatVecPlan,
    iters: usize,
) !u64 {
    if (!validView(view)) return error.InvalidView;
    const n: usize = @intCast(view.dims[0]);
    const m: usize = @intCast(view.dims[1]);
    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);
    const out = try allocator.alloc(f32, m);
    defer allocator.free(out);
    fillX(x);

    const count: usize = if (iters == 0) 1 else iters;
    try matvec.matVecRowsWithPlan(view, x, out, 0, m, plan, 0);

    const t0 = nowNs();
    for (0..count) |_| {
        try matvec.matVecRowsWithPlan(view, x, out, 0, m, plan, 0);
    }
    const t1 = nowNs();
    if (t1 <= t0) return 0;
    return (t1 - t0) / count;
}

/// Fastest non-erroring candidate for `view`, or null when the view is not
/// benchmarkable. `production` is always a candidate, so it is the fallback
/// when the explicit variants lose or error out.
pub fn bestMatvecPlan(allocator: std.mem.Allocator, view: *const TensorView) ?MatVecPlan {
    if (!validView(view)) return null;
    var best: MatVecPlan = .production;
    var best_ns: u64 = std.math.maxInt(u64);
    for (candidatesFor(view.ggml_type)) |cand| {
        const ns = benchMatvec(allocator, view, cand, bench_iters) catch continue;
        if (ns < best_ns) {
            best_ns = ns;
            best = cand;
        }
    }
    if (best_ns == std.math.maxInt(u64)) return null;
    return best;
}

/// Cold mmap read cost model: a fixed per-range fault cost plus a per-byte
/// streaming term. `cost(size) = fault_ns + ns_per_byte * size`.
pub const IoCost = struct {
    fault_ns: u64,
    ns_per_byte: f64,
};

/// Drop a byte range from the page cache so the next touch faults it in. A
/// no-op where `posix_fadvise` is unavailable; failures are ignored (the
/// benchmark degrades to whatever the cache gives, and the fit rejects it).
fn fadviseDontNeed(fd: std.posix.fd_t, offset: i64, len: i64) void {
    if (builtin.os.tag != .linux) return;
    _ = std.os.linux.fadvise(fd, offset, len, std.os.linux.POSIX_FADV.DONTNEED);
}

/// Sum a freshly-faulted range to force the reads and time them; 0 on a
/// clock failure.
fn touchNs(data: []const u8) u64 {
    const t0 = nowNs();
    var acc: u64 = 0;
    for (data) |b| acc +%= b;
    std.mem.doNotOptimizeAway(acc);
    const t1 = nowNs();
    if (t1 <= t0) return 0;
    return t1 - t0;
}

/// Time one cold read of `len` bytes from `fd`: a fresh private mapping, drop
/// the cached pages, then fault them in by touching. Null on any error.
fn measureColdRead(fd: std.posix.fd_t, len: usize) ?u64 {
    const map = std.posix.mmap(
        null,
        len,
        std.posix.PROT{ .READ = true },
        std.posix.MAP{ .TYPE = .PRIVATE },
        fd,
        0,
    ) catch return null;
    defer std.posix.munmap(map);
    fadviseDontNeed(fd, 0, @intCast(len));
    const ns = touchNs(map);
    if (ns == 0) return null;
    return ns;
}

/// Fit the cold read cost of `path` from a page-sized (fault-dominated) and a
/// large (throughput-dominated) sample against the existing monotonic clock.
/// Null off Linux, when the file is too small, or when the fit is unusable.
pub fn benchIoCost(allocator: std.mem.Allocator, io: Io, path: []const u8) ?IoCost {
    _ = allocator;
    if (builtin.os.tag != .linux) return null;
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    const size: usize = @intCast(file.length(io) catch return null);

    const page = std.heap.page_size_min;
    if (size < 2 * page) return null;
    const small_len: usize = page;
    const large_len: usize = @min(size, 4096 * page);
    if (large_len <= small_len) return null;

    const small_ns = measureColdRead(file.handle, small_len) orelse return null;
    const large_ns = measureColdRead(file.handle, large_len) orelse return null;

    const dsize = @as(f64, @floatFromInt(large_len - small_len));
    const slope = (@as(f64, @floatFromInt(large_ns)) - @as(f64, @floatFromInt(small_ns))) / dsize;
    if (!std.math.isFinite(slope) or slope < 0) return null;
    const intercept = @as(f64, @floatFromInt(small_ns)) - slope * @as(f64, @floatFromInt(small_len));
    const fault = if (intercept < 0) 0 else intercept;
    return .{ .fault_ns = @intFromFloat(fault), .ns_per_byte = slope };
}

test "benchMatvec times a synthetic q4_0 view and bestMatvecPlan picks one" {
    const testing = std.testing;
    const n: usize = 64;
    const m: usize = 4;
    const row_blocks = n / 32;
    var data: [m * row_blocks * 18]u8 = undefined;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var nib: [16]u8 = undefined;
            for (&nib, 0..) |*b, i| b.* = @truncate((row * 11 + rb * 7 + i * 3) & 0xff);
            const block = dequant.encodeQ4_0Block(@as(f16, 0.125), &nib);
            @memcpy(data[(row * row_blocks + rb) * 18 ..][0..18], &block);
        }
    }
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };

    const chosen = bestMatvecPlan(testing.allocator, &view);
    try testing.expect(chosen != null);

    const ns = try benchMatvec(testing.allocator, &view, chosen.?, bench_iters);
    try testing.expect(ns > 0);

    var unmapped = view;
    unmapped.mapped_slice = null;
    try testing.expect(bestMatvecPlan(testing.allocator, &unmapped) == null);
}

test "benchIoCost yields a plausible cost or null on a synthetic file" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const total = 2 * 1024 * 1024;
    var f = try tmp.dir.createFile(testing.io, "sample.bin", .{});
    var chunk: [4096]u8 = undefined;
    for (&chunk, 0..) |*b, i| b.* = @truncate(i);
    var written: usize = 0;
    while (written < total) : (written += chunk.len) {
        try f.writeStreamingAll(testing.io, &chunk);
    }
    f.close(testing.io);

    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/sample.bin", .{tmp.sub_path[0..]});
    if (benchIoCost(testing.allocator, testing.io, path)) |cost| {
        try testing.expect(cost.ns_per_byte >= 0);
        try testing.expect(std.math.isFinite(cost.ns_per_byte));
    }
    try testing.expect(benchIoCost(testing.allocator, testing.io, ".zig-cache/tmp/does-not-exist") == null);
}

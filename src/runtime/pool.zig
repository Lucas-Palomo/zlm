const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");
const TensorView = @import("../data/tensor_view.zig").TensorView;
const registry = @import("../kernels/registry.zig");
const matvec = @import("../kernels/linear/matvec/root.zig");
const cpu = @import("../utils/cpu.zig");

/// One unit of row-parallel work the pool can execute: a resolved matvec-family
/// dispatch (matvec or lm_head). The runtime supplies the buffers; the plan
/// already supplied `atomic_id` and `parts`.
pub const Job = struct {
    view: *const TensorView,
    x: []const f32,
    out: []f32,
    rows: usize,
    /// First output row this job computes (`0` = the whole tensor). A split CPU
    /// slice sets `rows = row_count` and `row_start = g`, so the workers cover
    /// `[row_start, row_start + rows)` and write `out` at those absolute rows.
    row_start: usize = 0,
    /// Input rows (batch). `> 1` runs a batched GEMV: each worker processes its
    /// output-row slice for every input row.
    batch: usize = 1,
    /// Input width `K` (the batch stride for `x`/`out`).
    k: usize = 0,
    atomic_id: u16,
    plan: u16 = 0,
    /// Chosen SIMD width (f32 lanes 4/8/16; `0` = build's native).
    simd_width: u8 = 0,
    /// Second weight view for a fused two-weight matvec (`gate_up_fused`).
    view_up: ?*const TensorView = null,
    /// Gated activation for `gate_up_fused`.
    act: matvec.ActivationKind = .gelu_tanh,
};

const max_spin_iterations: usize = if (builtin.mode == .Debug) 100 else 3000;

const Sync = struct {
    mutex: Io.Mutex = Io.Mutex.init,
    work: Io.Condition = Io.Condition.init,
    done: Io.Condition = Io.Condition.init,
    shutdown: bool = false,
    generation: u64 = 0,
    finished: usize = 0,
    job: Job = undefined,
    parts: usize = 1,
    results: ?[]?anyerror = null,
};

/// The half-open row range a worker with index `k` owns when `total` rows are
/// split into `parts` parts (ceil-first, covers `0..total`).
fn rangeOf(total: usize, parts: usize, k: usize) struct { begin: usize, end: usize } {
    if (k >= parts) return .{ .begin = total, .end = total };
    var begin: usize = 0;
    var i: usize = 0;
    while (i < k) : (i += 1) {
        const remaining_parts = parts - i;
        const remaining = total - begin;
        const size = (remaining + remaining_parts - 1) / remaining_parts;
        begin += size;
    }
    const remaining_parts = parts - k;
    const remaining = total - begin;
    const size = (remaining + remaining_parts - 1) / remaining_parts;
    return .{ .begin = begin, .end = begin + size };
}

fn runJob(job: Job, begin: usize, end: usize) ?anyerror {
    if (job.batch <= 1) {
        var args = matvec.Args{ .view = job.view, .view_up = job.view_up, .x = job.x, .out = job.out, .begin = begin, .end = end, .plan = job.plan, .simd_width = job.simd_width, .act = job.act };
        registry.run(.matvec, job.atomic_id, &args) catch |e| return e;
        return null;
    }
    if (job.view_up == null) {
        const x = job.x[0 .. job.batch * job.k];
        const out = job.out[0 .. job.batch * job.rows];
        matvec.matMulRows(job.view, x, out, begin, end) catch |e| return e;
        return null;
    }
    var tok: usize = 0;
    while (tok < job.batch) : (tok += 1) {
        var args = matvec.Args{
            .view = job.view,
            .view_up = job.view_up,
            .x = job.x[tok * job.k ..][0..job.k],
            .out = job.out[tok * job.rows ..][0..job.rows],
            .begin = begin,
            .end = end,
            .plan = job.plan,
            .simd_width = job.simd_width,
            .act = job.act,
        };
        registry.run(.matvec, job.atomic_id, &args) catch |e| return e;
    }
    return null;
}

fn helperMain(io: Io, sync: *Sync, index: usize, affinity: []const u16) void {
    if (affinity.len > 0) cpu.pinCurrentThread(affinity[index % affinity.len]);
    sync.mutex.lockUncancelable(io);
    var seen: u64 = 0;
    while (true) {
        while (!sync.shutdown and sync.generation == seen) {
            sync.mutex.unlock(io);
            var spins: usize = 0;
            while (spins < max_spin_iterations) : (spins += 1) {
                if (@atomicLoad(bool, &sync.shutdown, .acquire)) break;
                if (@atomicLoad(u64, &sync.generation, .acquire) != seen) break;
                std.atomic.spinLoopHint();
            }
            sync.mutex.lockUncancelable(io);
            if (sync.shutdown or sync.generation != seen) break;
            sync.work.waitUncancelable(io, &sync.mutex);
        }
        if (sync.shutdown) break;
        seen = sync.generation;
        const job = sync.job;
        const parts = sync.parts;
        sync.mutex.unlock(io);

        const r = rangeOf(job.rows, parts, index + 1);
        const result = runJob(job, job.row_start + r.begin, job.row_start + r.end);

        sync.mutex.lockUncancelable(io);
        sync.results.?[index] = result;
        const finished = sync.finished + 1;
        sync.finished = finished;
        @atomicStore(usize, &sync.finished, finished, .release);
        sync.done.signal(io);
    }
    sync.mutex.unlock(io);
}

/// Persistent worker pool owned by the `Session`. One job slot guarded by a
/// mutex + two conditions; created lazily on the first `parts > 1`.
pub const Pool = struct {
    allocator: std.mem.Allocator,
    io: Io,
    sync: *Sync,
    helpers: []std.Thread,
    spawned: usize,

    pub fn init(allocator: std.mem.Allocator, io: Io, thread_count: usize, affinity: []const u16) !Pool {
        const n: usize = if (thread_count == 0) 1 else thread_count;
        const sync = try allocator.create(Sync);
        errdefer allocator.destroy(sync);
        sync.* = .{};

        const helper_count = n - 1;
        const helpers = try allocator.alloc(std.Thread, helper_count);
        errdefer allocator.free(helpers);
        const results = try allocator.alloc(?anyerror, helper_count);
        errdefer allocator.free(results);
        for (results) |*r| r.* = null;
        sync.results = results;

        var spawned: usize = 0;
        for (0..helper_count) |k| {
            helpers[k] = std.Thread.spawn(.{}, helperMain, .{ io, sync, k, affinity }) catch break;
            spawned += 1;
        }
        return .{ .allocator = allocator, .io = io, .sync = sync, .helpers = helpers, .spawned = spawned };
    }

    pub fn deinit(self: *Pool, io: Io) void {
        self.sync.mutex.lockUncancelable(io);
        self.sync.shutdown = true;
        self.sync.work.broadcast(io);
        self.sync.done.broadcast(io);
        self.sync.mutex.unlock(io);
        for (self.helpers[0..self.spawned]) |t| t.join();
        if (self.helpers.len > 0) self.allocator.free(self.helpers);
        if (self.sync.results) |rs| self.allocator.free(rs);
        self.allocator.destroy(self.sync);
    }

    pub fn run(self: *Pool, job: Job, parts: usize) !void {
        if (self.spawned == 0 or parts <= 1) {
            if (runJob(job, job.row_start, job.row_start + job.rows)) |e| return e;
            return;
        }

        var p = parts;
        if (p > job.rows) p = job.rows;
        if (p > self.spawned + 1) p = self.spawned + 1;
        if (p < 1) p = 1;

        const sync = self.sync;
        sync.mutex.lockUncancelable(self.io);
        sync.job = job;
        sync.parts = p;
        sync.finished = 0;
        @atomicStore(usize, &sync.finished, 0, .release);
        const next_gen = sync.generation + 1;
        sync.generation = next_gen;
        @atomicStore(u64, &sync.generation, next_gen, .release);
        sync.work.broadcast(self.io);
        sync.mutex.unlock(self.io);

        const r0 = rangeOf(job.rows, p, 0);
        const first_error = runJob(job, job.row_start + r0.begin, job.row_start + r0.end);

        sync.mutex.lockUncancelable(self.io);
        while (sync.finished < self.spawned) {
            sync.mutex.unlock(self.io);
            var spins: usize = 0;
            while (spins < max_spin_iterations) : (spins += 1) {
                if (@atomicLoad(usize, &sync.finished, .acquire) >= self.spawned) break;
                std.atomic.spinLoopHint();
            }
            sync.mutex.lockUncancelable(self.io);
            if (sync.finished >= self.spawned) break;
            sync.done.waitUncancelable(self.io, &sync.mutex);
        }
        var first_err: ?anyerror = first_error;
        for (sync.results.?[0..self.spawned]) |r| {
            if (first_err == null) first_err = r;
        }
        sync.mutex.unlock(self.io);
        if (first_err) |e| return e;
    }
};

test "pool splits rows and matches a single-threaded run" {
    const testing = std.testing;
    const io = testing.io;

    const n: usize = 4;
    const m: usize = 4;
    var weights = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    const x = [_]f32{ 1, 2, 3, 4 };
    var expected = [_]f32{0} ** m;
    var got = [_]f32{0} ** m;

    try matvec.matVecRows(&view, &x, &expected, 0, m);

    var pool = try Pool.init(testing.allocator, io, 4, &.{});
    defer pool.deinit(io);
    try testing.expectEqual(@as(usize, 3), pool.spawned);

    try pool.run(.{ .view = &view, .x = &x, .out = &got, .rows = m, .atomic_id = 0 }, 4);
    try testing.expectEqualSlices(f32, &expected, &got);
}

test "pool honors row_start (the CPU slice of a heterogeneous split)" {
    const testing = std.testing;
    const io = testing.io;

    const n: usize = 4;
    const m: usize = 8;
    var weights = [_]f32{0} ** (m * n);
    for (0..m) |r| weights[r * n + (r % n)] = 1;
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    const x = [_]f32{ 1, 2, 3, 4 };
    var want = [_]f32{0} ** m;
    try matvec.matVecRows(&view, &x, &want, 0, m);

    var got = [_]f32{-1} ** m;
    var pool = try Pool.init(testing.allocator, io, 4, &.{});
    defer pool.deinit(io);
    // The CPU slide computes rows `[3, 8)` at their absolute output rows; the
    // leading rows `[0, 3)` (the GPU's) stay untouched.
    try pool.run(.{ .view = &view, .x = &x, .out = &got, .rows = m - 3, .row_start = 3, .atomic_id = 0 }, 4);
    try testing.expectEqualSlices(f32, &[_]f32{ -1, -1, -1 }, got[0..3]);
    try testing.expectEqualSlices(f32, want[3..m], got[3..m]);
}

test "pool batched job matches per-row matvec (q4_0)" {
    const testing = std.testing;
    const io = testing.io;
    const dequant = @import("../kernels/linear/quant/root.zig");

    const n: usize = 64;
    const m: usize = 5;
    const M: usize = 6;
    const dims = [_]u64{ n, m };

    var data: [m * (n / 32) * 18]u8 = undefined;
    const row_blocks = n / 32;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            var nib: [16]u8 = undefined;
            for (&nib, 0..) |*b, i| b.* = @truncate((row * 11 + rb * 7 + i * 3) & 0xff);
            const block = dequant.encodeQ4_0Block(@as(f16, 0.125), &nib);
            @memcpy(data[(row * row_blocks + rb) * 18 ..][0..18], &block);
        }
    }
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = &data,
    };
    var acts: [M * n]f32 = undefined;
    for (&acts, 0..) |*v, i| {
        const x: i32 = @as(i32, @intCast(i % 19)) - 9;
        v.* = @as(f32, @floatFromInt(x)) * 0.0625;
    }
    var expected: [M * m]f32 = [_]f32{0} ** (M * m);
    var got: [M * m]f32 = [_]f32{0} ** (M * m);
    for (0..M) |tok| {
        try matvec.matVecRows(&view, acts[tok * n ..][0..n], expected[tok * m ..][0..m], 0, m);
    }

    var pool = try Pool.init(testing.allocator, io, 4, &.{});
    defer pool.deinit(io);

    try pool.run(.{ .view = &view, .x = &acts, .out = &got, .rows = m, .batch = M, .k = n, .atomic_id = 0 }, 4);
    try testing.expectEqualSlices(f32, &expected, &got);
}

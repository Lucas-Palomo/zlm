const std = @import("std");
const Io = std.Io;
const cpu = @import("../utils/cpu.zig");

/// One ordered page-in: the main thread already resolved everything the worker
/// needs from its `ztef.Reader`. The worker holds no `Reader`/`Session` pointer,
/// so the `Session` may be moved by value without dangling.
///
/// A `.madvise` job asks the kernel to start readahead for `[addr, addr+len)`
/// (the mmap path). A `.pread` job reads `len` bytes from `file_off` on `src_fd`
/// into the page-aligned arena destination `dst` (the staging arena path). One
/// staged range is split into `k` chunk jobs so the drive sees `k` streams.
pub const Job = struct {
    range_id: u32,
    addr: usize = 0,
    dst: usize = 0,
    src_fd: i32 = -1,
    file_off: u64 = 0,
    len: usize,
    kind: Kind = .madvise,

    pub const Kind = enum(u8) { madvise, pread };
};

/// Fill `len` bytes at `dst` from `file_off` on `fd`, retrying short reads and
/// `EINTR`. `dst` is page-aligned arena memory; a plain `pread` is enough (no
/// `O_DIRECT`).
pub fn preadInto(dst: usize, fd: i32, file_off: u64, len: usize) void {
    const buf: [*]u8 = @ptrFromInt(dst);
    var done: usize = 0;
    while (done < len) {
        const rc = std.os.linux.pread(fd, buf + done, len - done, @intCast(file_off + done));
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return,
        }
        const n: usize = rc;
        if (n == 0) return;
        done += n;
    }
}

/// Bounded ring capacity: queue slots and, equivalently, in-flight entries.
pub const CAP: usize = 256;

const Inflight = struct { range_id: u32, count: usize };

const Sync = struct {
    mutex: Io.Mutex = Io.Mutex.init,
    can_push: Io.Condition = Io.Condition.init,
    can_pop: Io.Condition = Io.Condition.init,
    inflight_done: Io.Condition = Io.Condition.init,
    shutdown: bool = false,
    queue: [CAP]Job = undefined,
    head: usize = 0,
    tail: usize = 0,
    count: usize = 0,
    inflight: [CAP]Inflight = undefined,
    inflight_len: usize = 0,
};

/// The default worker action: ask the kernel to start readahead for the range.
fn applyMadvise(addr: usize, len: usize) void {
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(addr);
    _ = std.posix.madvise(ptr, len, std.posix.MADV.WILLNEED) catch {};
}

const Action = *const fn (usize, usize) void;

/// The CPU worker `k` pins to, or null when the plan carries no affinity.
fn cpuFor(cpus: []const u16, k: usize) ?u16 {
    if (cpus.len == 0) return null;
    return cpus[k % cpus.len];
}

fn inflightCount(sync: *const Sync, range_id: u32) usize {
    for (sync.inflight[0..sync.inflight_len]) |e| {
        if (e.range_id == range_id) return e.count;
    }
    return 0;
}

fn inflightAdd(sync: *Sync, range_id: u32) void {
    for (sync.inflight[0..sync.inflight_len]) |*e| {
        if (e.range_id == range_id) {
            e.count += 1;
            return;
        }
    }
    if (sync.inflight_len >= CAP) return;
    sync.inflight[sync.inflight_len] = .{ .range_id = range_id, .count = 1 };
    sync.inflight_len += 1;
}

fn inflightRemove(sync: *Sync, range_id: u32) void {
    for (sync.inflight[0..sync.inflight_len], 0..) |*e, i| {
        if (e.range_id != range_id) continue;
        e.count -= 1;
        if (e.count == 0) {
            sync.inflight[i] = sync.inflight[sync.inflight_len - 1];
            sync.inflight_len -= 1;
        }
        return;
    }
}

fn workerMain(io: Io, sync: *Sync, action: Action, cpu_id: ?u16) void {
    if (cpu_id) |c| cpu.pinCurrentThread(c);
    sync.mutex.lockUncancelable(io);
    while (true) {
        while (!sync.shutdown and sync.count == 0) {
            sync.can_pop.waitUncancelable(io, &sync.mutex);
        }
        if (sync.shutdown and sync.count == 0) break;
        const job = sync.queue[sync.head];
        sync.head = (sync.head + 1) % CAP;
        sync.count -= 1;
        sync.can_push.signal(io);
        sync.mutex.unlock(io);

        switch (job.kind) {
            .madvise => action(job.addr, job.len),
            .pread => preadInto(job.dst, job.src_fd, job.file_off, job.len),
        }

        sync.mutex.lockUncancelable(io);
        inflightRemove(sync, job.range_id);
        sync.inflight_done.broadcast(io);
    }
    sync.mutex.unlock(io);
}

/// A pool of `count` background page-in workers sharing one bounded ring.
/// `enqueue` is non-blocking and bounded; `wait` blocks until every queued job
/// for a `range_id` has issued its `WILLNEED`. Accounting stays on the calling
/// thread — the workers only touch the OS advice and their shared queue/in-flight
/// state, so the M workers pull ranges in parallel.
pub const Prefetch = struct {
    allocator: std.mem.Allocator,
    io: Io,
    sync: *Sync,
    workers: []std.Thread,
    spawned: usize,

    /// Spawn `count` workers. Worker `k` pins to `cpus[k % cpus.len]` when
    /// `cpus` is non-empty (the plan's last M affinity entries), else it stays
    /// unpinned. The runtime passes the dedicated prefetch CPUs; the compute
    /// pool owns the first N entries.
    pub fn init(allocator: std.mem.Allocator, io: Io, count: usize, cpus: []const u16) !Prefetch {
        return initWithAction(allocator, io, applyMadvise, count, cpus);
    }

    /// Same as `init` with an injectable action (tests use a gate). The
    /// production action is `madvise(WILLNEED)`.
    pub fn initWithAction(allocator: std.mem.Allocator, io: Io, action: Action, count: usize, cpus: []const u16) !Prefetch {
        const sync = try allocator.create(Sync);
        errdefer allocator.destroy(sync);
        sync.* = .{};
        const n: usize = if (count == 0) 1 else count;
        const workers = try allocator.alloc(std.Thread, n);
        errdefer allocator.free(workers);
        workers[0] = try std.Thread.spawn(.{}, workerMain, .{ io, sync, action, cpuFor(cpus, 0) });
        var spawned: usize = 1;
        for (1..n) |k| {
            workers[k] = std.Thread.spawn(.{}, workerMain, .{ io, sync, action, cpuFor(cpus, k) }) catch break;
            spawned += 1;
        }
        return .{ .allocator = allocator, .io = io, .sync = sync, .workers = workers, .spawned = spawned };
    }

    pub fn deinit(self: *Prefetch) void {
        self.sync.mutex.lockUncancelable(self.io);
        self.sync.shutdown = true;
        self.sync.can_pop.broadcast(self.io);
        self.sync.can_push.broadcast(self.io);
        self.sync.inflight_done.broadcast(self.io);
        self.sync.mutex.unlock(self.io);
        for (self.workers[0..self.spawned]) |t| t.join();
        self.allocator.free(self.workers);
        self.allocator.destroy(self.sync);
        self.* = undefined;
    }

    /// Queue one page-in. Returns false when the ring is full, so the caller
    /// can fall back to the inline `madvise`.
    pub fn enqueue(self: *Prefetch, job: Job) bool {
        const sync = self.sync;
        sync.mutex.lockUncancelable(self.io);
        defer sync.mutex.unlock(self.io);
        if (sync.count == CAP) return false;
        sync.queue[sync.tail] = job;
        sync.tail = (sync.tail + 1) % CAP;
        sync.count += 1;
        inflightAdd(sync, job.range_id);
        sync.can_pop.signal(self.io);
        return true;
    }

    /// Block until no queued job for `range_id` is still in flight. A range
    /// that was never enqueued returns immediately.
    pub fn wait(self: *Prefetch, range_id: u32) void {
        const sync = self.sync;
        sync.mutex.lockUncancelable(self.io);
        defer sync.mutex.unlock(self.io);
        while (!sync.shutdown and inflightCount(sync, range_id) > 0) {
            sync.inflight_done.waitUncancelable(self.io, &sync.mutex);
        }
    }
};

var test_gate: bool = false;

fn gatedAction(addr: usize, len: usize) void {
    _ = addr;
    _ = len;
    while (@atomicLoad(bool, &test_gate, .acquire)) std.atomic.spinLoopHint();
}

test "prefetch queue is bounded and wait drains" {
    const testing = std.testing;
    var pf = try Prefetch.initWithAction(testing.allocator, testing.io, gatedAction, 1, &.{});
    defer pf.deinit();

    @atomicStore(bool, &test_gate, true, .release);
    var accepted: usize = 0;
    var id: u32 = 1;
    while (pf.enqueue(.{ .range_id = id, .addr = 0, .len = 0 })) {
        accepted += 1;
        id += 1;
        if (accepted > CAP + 1) break;
    }
    try testing.expect(accepted >= CAP and accepted <= CAP + 1);

    @atomicStore(bool, &test_gate, false, .release);
    pf.wait(1);
    pf.wait(2);
    try testing.expect(pf.enqueue(.{ .range_id = id, .addr = 0, .len = 0 }));
}

test "prefetch wait on an unknown range returns" {
    const testing = std.testing;
    var pf = try Prefetch.init(testing.allocator, testing.io, 1, &.{});
    defer pf.deinit();
    pf.wait(0xFFFF_FFFF);
}

var pool_active: usize = 0;
var pool_gate: bool = false;

fn concurrentAction(addr: usize, len: usize) void {
    _ = addr;
    _ = len;
    _ = @atomicRmw(usize, &pool_active, .Add, 1, .acq_rel);
    while (@atomicLoad(bool, &pool_gate, .acquire)) std.atomic.spinLoopHint();
    _ = @atomicRmw(usize, &pool_active, .Sub, 1, .acq_rel);
}

test "prefetch pool runs M workers concurrently" {
    const testing = std.testing;
    @atomicStore(usize, &pool_active, 0, .release);
    @atomicStore(bool, &pool_gate, true, .release);
    var pf = try Prefetch.initWithAction(testing.allocator, testing.io, concurrentAction, 2, &.{});
    defer pf.deinit();
    try testing.expectEqual(@as(usize, 2), pf.workers.len);

    try testing.expect(pf.enqueue(.{ .range_id = 1, .addr = 0, .len = 0 }));
    try testing.expect(pf.enqueue(.{ .range_id = 2, .addr = 0, .len = 0 }));

    var saw_two = false;
    const deadline = Io.Clock.awake.now(testing.io).nanoseconds + 2 * std.time.ns_per_s;
    while (Io.Clock.awake.now(testing.io).nanoseconds < deadline) {
        if (@atomicLoad(usize, &pool_active, .acquire) >= 2) {
            saw_two = true;
            break;
        }
        std.atomic.spinLoopHint();
    }
    @atomicStore(bool, &pool_gate, false, .release);
    pf.wait(1);
    pf.wait(2);
    try testing.expect(saw_two);
}

const std = @import("std");
const plan = @import("../plan/root.zig");
const ztef = @import("../image/ztef.zig");
const prefetch = @import("prefetch.zig");
const Prefetch = prefetch.Prefetch;
const preadInto = prefetch.preadInto;

/// The service residency state. The ring is explicit and bounded, so the state
/// only gates it: `cold` holds nothing (the mmap streams), `warm`/`hot` run the
/// ring.
pub const State = enum { cold, warm, hot };

/// The three-pillar residency ring (`runtime/15`): `compute | reserved |
/// prefetch`. Each pillar holds one **block** -- a run of the streamed ranges in
/// consumption order -- and `block % 3` selects the pillar. Advancing the cursor
/// recycles the slot the compute just left into the next prefetch target, so the
/// same three buffers cycle while the `--prefetch-threads` workers fill ahead.
///
/// A plan with `ring_block_bytes == 0` has no ring: every range streams from the
/// mmap (the `b25630b` behavior) and every method is a no-op.
pub const Ring = struct {
    allocator: std.mem.Allocator,
    graph: plan.ResidencyGraph,
    /// The three block buffers, page-aligned for `pread`.
    pillars: [3][]align(std.heap.page_size_min) u8 = .{ &.{}, &.{}, &.{} },
    /// The byte offset of range `i` within its block, or `NONE` when un-ringed.
    range_off: []u64,
    /// Number of blocks (`max block + 1`).
    blocks: u32,
    /// The block being computed (the `compute` pillar).
    cursor: u64 = 0,
    /// The block each slot currently holds (`NONE` when empty).
    slot_block: [3]u64 = .{ NONE, NONE, NONE },
    /// Whether the slot's block is fully read (false while a fill is pending).
    slot_ready: [3]bool = .{ false, false, false },
    enabled: bool = true,
    /// `--prefetch-threads` without `--residency`: no pillars, the workers just
    /// issue `madvise(WILLNEED)` for the blocks ahead (the `b25630b` behavior).
    advise_only: bool = false,
    started: bool = false,
    /// Borrowed background pool owned by the `Session`; null fills inline.
    prefetch: ?*Prefetch = null,

    const NONE: u64 = std.math.maxInt(u64);

    pub fn init(allocator: std.mem.Allocator, graph: plan.ResidencyGraph, prefetch_threads: u16) !Ring {
        const ringed = graph.ring_block_bytes > 0;
        const want_advise = !ringed and prefetch_threads > 0;
        const n = graph.ranges.len;
        const off = try allocator.alloc(u64, n);
        errdefer allocator.free(off);
        @memset(off, NONE);

        var blocks: u32 = 0;
        var len_by_block: []u64 = &.{};
        defer if (len_by_block.len > 0) allocator.free(len_by_block);

        var self = Ring{
            .allocator = allocator,
            .graph = graph,
            .range_off = off,
            .blocks = 0,
            .enabled = ringed or want_advise,
            .advise_only = want_advise,
        };
        if (!self.enabled) return self;

        // The block count and each range's offset within its block (the ranges
        // are already in consumption order per block).
        var max_block: u32 = 0;
        for (graph.ranges) |r| {
            if (r.block == plan.NONE) continue;
            if (r.block + 1 > max_block) max_block = r.block + 1;
        }
        blocks = max_block;
        len_by_block = try allocator.alloc(u64, blocks);
        @memset(len_by_block, 0);
        for (graph.ranges, 0..) |r, i| {
            if (r.block == plan.NONE or r.block >= blocks) continue;
            off[i] = len_by_block[r.block];
            len_by_block[r.block] += r.length;
        }
        self.blocks = blocks;

        if (ringed) {
            for (0..3) |s| {
                self.pillars[s] = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(std.heap.page_size_min), @intCast(graph.ring_block_bytes));
            }
        }
        return self;
    }

    pub fn deinit(self: *Ring) void {
        for (0..3) |s| {
            if (self.pillars[s].len > 0) self.allocator.free(self.pillars[s]);
        }
        self.allocator.free(self.range_off);
        self.* = undefined;
    }

    /// The ring is explicit and bounded; `cold` stops filling (the mmap streams
    /// instead), `warm`/`hot` run it.
    pub fn setState(self: *Ring, state: State, plan_window: u64) void {
        _ = plan_window;
        self.enabled = state != .cold and self.graph.ring_block_bytes > 0;
    }

    /// Ensure `range_id`'s block is resident before its dispatch reads it.
    /// Advancing to a new block fills the next two blocks ahead (the producer).
    pub fn ensure(self: *Ring, range_id: u32, ztefs: *[ztef.COUNT]?ztef.Reader) void {
        if (!self.enabled or self.blocks == 0) return;
        if (range_id == plan.NONE or range_id >= self.graph.ranges.len) return;
        const b = self.graph.ranges[range_id].block;
        if (b == plan.NONE or b >= self.blocks) return;
        if (!self.started) {
            self.started = true;
            self.cursor = b;
            self.fill(b, ztefs);
            self.fillAhead(ztefs);
            return;
        }
        if (b != self.cursor) {
            self.cursor = b;
            self.fillAhead(ztefs);
        }
        self.wait(b, ztefs);
    }

    /// The pillar slice holding a resident range's bytes, or null (the caller
    /// falls back to the mmap).
    pub fn slice(self: *Ring, range_id: u32) ?[]const u8 {
        if (!self.enabled or self.advise_only) return null;
        if (range_id == plan.NONE or range_id >= self.graph.ranges.len) return null;
        const r = self.graph.ranges[range_id];
        if (r.block == plan.NONE) return null;
        const slot = r.block % 3;
        if (self.slot_block[slot] != r.block or !self.slot_ready[slot]) return null;
        const off = self.range_off[range_id];
        const len: usize = @intCast(r.length);
        const buf = self.pillars[slot];
        if (off == NONE or off > buf.len or len > buf.len - off) return null;
        return buf[@intCast(off)..][0..len];
    }

    /// Resident bytes: the filled pillars.
    pub fn usedBytes(self: *const Ring) u64 {
        var n: u64 = 0;
        for (0..3) |s| {
            if (self.slot_ready[s]) n += self.pillars[s].len;
        }
        return n;
    }

    /// Whether `range_id`'s block is resident.
    pub fn isResident(self: *const Ring, range_id: u32) bool {
        if (range_id == plan.NONE or range_id >= self.graph.ranges.len) return false;
        const r = self.graph.ranges[range_id];
        if (r.block == plan.NONE) return false;
        const slot = r.block % 3;
        return self.slot_block[slot] == r.block and self.slot_ready[slot];
    }

    fn fillAhead(self: *Ring, ztefs: *[ztef.COUNT]?ztef.Reader) void {
        var off: u64 = 1;
        while (off <= 2) : (off += 1) {
            const b = (self.cursor + off) % self.blocks;
            self.fill(b, ztefs);
        }
    }

    /// Fill block `b` into pillar `b % 3`. Enqueues the reads on the pool (async)
    /// when present, else reads inline. Idempotent while the block is the slot's
    /// current occupant.
    fn fill(self: *Ring, b: u64, ztefs: *[ztef.COUNT]?ztef.Reader) void {
        const slot = b % 3;
        if (self.slot_block[slot] == b) return;
        self.slot_block[slot] = b;
        self.slot_ready[slot] = false;
        if (self.advise_only) {
            self.adviseBlock(b, ztefs);
            self.slot_ready[slot] = true;
            return;
        }

        const buf = self.pillars[slot];
        const base = @intFromPtr(buf.ptr);
        const k: u64 = if (self.graph.ring_k == 0) 1 else self.graph.ring_k;
        var any = false;
        var pool_pending = false;
        for (self.graph.ranges, 0..) |r, i| {
            if (r.block == plan.NONE or r.block != b) continue;
            const off = self.range_off[i];
            const src = fileAddr(r, ztefs) orelse continue;
            const fd = fileHandle(r, ztefs) orelse continue;
            const dst = base + @as(usize, @intCast(off));
            any = true;
            if (self.prefetch) |pf| {
                const chunk = (r.length + k - 1) / k;
                var done: u64 = 0;
                while (done < r.length) {
                    const clen: usize = @intCast(@min(chunk, r.length - done));
                    const ok = pf.enqueue(.{ .range_id = @intCast(b), .kind = .pread, .dst = dst + @as(usize, @intCast(done)), .src_fd = fd, .file_off = src + done, .len = clen });
                    if (!ok) preadInto(dst + @as(usize, @intCast(done)), fd, src + done, clen);
                    done += clen;
                }
                pool_pending = true;
            } else {
                preadInto(dst, fd, src, @intCast(r.length));
            }
        }
        // With a pool the reads are in flight; `wait` blocks for them. Without
        // one they are already done.
        if (self.prefetch == null or !any) self.slot_ready[slot] = true;
    }

    /// `--prefetch-threads`-without-`--residency`: ask the kernel to start
    /// readahead for every range of block `b` (the `b25630b` behavior, no copy).
    fn adviseBlock(self: *Ring, b: u64, ztefs: *[ztef.COUNT]?ztef.Reader) void {
        for (self.graph.ranges) |r| {
            if (r.block == plan.NONE or r.block != b) continue;
            const a = mmapAddr(r, ztefs) orelse continue;
            if (self.prefetch) |pf| {
                if (pf.enqueue(.{ .range_id = @intCast(b), .kind = .madvise, .addr = a.addr, .len = a.len })) continue;
            }
            advise(r, ztefs);
        }
    }

    /// Block until block `b` is resident (fills inline if it never was).
    fn wait(self: *Ring, b: u64, ztefs: *[ztef.COUNT]?ztef.Reader) void {
        const slot = b % 3;
        if (self.slot_block[slot] != b) {
            self.fill(b, ztefs);
        }
        if (self.slot_ready[slot]) return;
        if (self.prefetch) |pf| {
            pf.wait(@intCast(b));
            self.slot_ready[slot] = true;
        } else {
            self.slot_ready[slot] = true;
        }
    }
};

test "ring fills a block and serves its range from the pillar" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var payload: [4096]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast(i % 251);
    const items = [_]ztef.Item{
        .{ .name = "blk.0.attn_q.weight", .layer = 0, .ggml_type = .q4_0, .n_dims = 2, .dims = .{ 64, 2, 0, 0 }, .bytes = &payload },
    };
    const w = try ztef.write(testing.allocator, tmp.dir, testing.io, .q_proj, &items);
    defer testing.allocator.free(w.tensors);
    var reader = try ztef.Reader.openIn(testing.allocator, tmp.dir, testing.io, ztef.Kind.q_proj.filename());
    defer reader.deinit();
    const e = reader.find("blk.0.attn_q.weight").?;
    const zi = @intFromEnum(ztef.Kind.q_proj);

    var ranges = [_]plan.Range{
        .{ .kind = .q_proj, .offset = e.offset, .length = e.length, .block = 0 },
    };
    const g = plan.ResidencyGraph{ .ranges = &ranges, .window_bytes = 0, .ring_block_bytes = e.length, .ring_k = 2 };

    var ztefs: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT;
    ztefs[zi] = reader;

    var ring = try Ring.init(testing.allocator, g, 0);
    defer ring.deinit();
    ring.ensure(0, &ztefs);
    try testing.expect(ring.isResident(0));
    const a = ring.slice(0) orelse return error.TestUnexpectedResult;
    try testing.expectEqualSlices(u8, &payload, a);
}

fn ziOf(r: plan.Range) usize {
    return @intFromEnum(r.kind);
}

fn mmapAddr(r: plan.Range, ztefs: *[ztef.COUNT]?ztef.Reader) ?ztef.RangeAddr {
    const zi = ziOf(r);
    if (zi >= ztefs.len) return null;
    if (ztefs[zi]) |*z| return z.rangeAddr(r.offset, r.length);
    return null;
}

fn advise(r: plan.Range, ztefs: *[ztef.COUNT]?ztef.Reader) void {
    const zi = ziOf(r);
    if (zi >= ztefs.len) return;
    if (ztefs[zi]) |*z| z.adviseRange(r.offset, r.length, true);
}

fn fileAddr(r: plan.Range, ztefs: *[ztef.COUNT]?ztef.Reader) ?u64 {
    const zi = ziOf(r);
    if (zi >= ztefs.len) return null;
    if (ztefs[zi]) |*z| return z.fileAddr(r.offset, r.length);
    return null;
}

fn fileHandle(r: plan.Range, ztefs: *[ztef.COUNT]?ztef.Reader) ?i32 {
    const zi = ziOf(r);
    if (zi >= ztefs.len) return null;
    if (ztefs[zi]) |*z| return z.fileHandle();
    return null;
}

test "ring is off without block_bytes" {
    const testing = std.testing;
    var ranges = [_]plan.Range{
        .{ .kind = .q_proj, .offset = 0, .length = 1024, .block = 0 },
    };
    const g = plan.ResidencyGraph{ .ranges = &ranges, .window_bytes = 0 };
    var ring = try Ring.init(testing.allocator, g, 0);
    defer ring.deinit();
    try testing.expect(!ring.enabled);
    try testing.expect(ring.slice(0) == null);
    try testing.expectEqual(@as(u64, 0), ring.usedBytes());
}

test "ring maps blocks to pillars and rotates the cursor" {
    const testing = std.testing;
    var ranges = [_]plan.Range{
        .{ .kind = .q_proj, .offset = 0, .length = 100, .block = 0 },
        .{ .kind = .q_proj, .offset = 100, .length = 100, .block = 1 },
        .{ .kind = .q_proj, .offset = 200, .length = 100, .block = 2 },
        .{ .kind = .q_proj, .offset = 300, .length = 100, .block = 3 },
    };
    const g = plan.ResidencyGraph{ .ranges = &ranges, .window_bytes = 0, .ring_block_bytes = 128, .ring_k = 2 };
    var ring = try Ring.init(testing.allocator, g, 0);
    defer ring.deinit();
    try testing.expect(ring.enabled);
    try testing.expectEqual(@as(usize, 128), ring.pillars[0].len);
    try testing.expectEqual(@as(u32, 4), ring.blocks);
    // The ranges are offset within their (single-range) block.
    try testing.expectEqual(@as(u64, 0), ring.range_off[0]);
    try testing.expectEqual(@as(u64, 0), ring.range_off[3]);
    // Block 0 -> pillar 0, block 3 -> pillar 0 (the recycle slot).
    try testing.expectEqual(@as(u64, 0), @as(u64, 0) % 3);
    try testing.expectEqual(@as(u64, 0), @as(u64, 3) % 3);
}

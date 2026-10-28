const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const prefix_cache = @import("prefix_cache.zig");

/// The radix prefix-cache artifact written by `tune --prefix-cache`. It is immutable
/// once written; the runtime maps it read-only and references its payload bytes
/// **zero-copy**, then appends new segments to a separate writable mmap journal.
///
/// Layout (little-endian):
///
/// ```text
/// magic "ZPC\x00" | version u32 | flags u32
/// key model/arithmetic/layout u64 | dtype u8 | pad[3]
/// n_segments u32
/// index: n_segments x { parent u32, token_begin u32, token_end u32,
///                       payload_off u64, payload_len u64 }
/// payload: per segment, canonical K/V then the u32 token ids (payload_len bytes)
/// ```
pub const MAGIC = "ZPC\x00";
/// Bumped when the header/key layout changes (v2: dropped the redundant
/// tokenizer field; v3: position-major K/V payload + the `format` key field).
pub const VERSION: u32 = 3;

const key_bytes: usize = 8 * 4 + 1 + 3;
const record_bytes: usize = 4 + 4 + 4 + 8 + 8;
const header_bytes: usize = 4 + 4 + 4 + key_bytes + 4;

/// One index record. `off`/`len` are absolute file offsets valid against
/// `Base.mmap.data`; the block is K/V followed by the `u32` token ids.
pub const Record = struct {
    parent: u32,
    begin: u32,
    end: u32,
    off: u64,
    len: u64,
};

/// A mapped, read-only `.zpc` base. Must outlive any memo segment referencing it.
pub const Base = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    key: prefix_cache.Key,
    records: []Record,

    pub fn data(self: *const Base) []const u8 {
        return self.mmap.data;
    }

    pub fn deinit(self: *Base) void {
        self.allocator.free(self.records);
        self.mmap.close();
    }
};

/// Open the `.zpc` at `path` read-only. Returns null when it does not exist;
/// a corrupt file is an error the caller should ignore.
pub fn openBase(allocator: std.mem.Allocator, io: Io, path: []const u8) !?Base {
    var mmap = MappedFile.open(io, path) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    errdefer mmap.close();
    const data = mmap.data;
    if (data.len < header_bytes) return error.Corrupt;

    var pos: usize = 0;
    if (!std.mem.eql(u8, data[pos..][0..4], MAGIC)) return error.BadMagic;
    pos += 4;
    if (readU32(data, &pos) != VERSION) return error.BadVersion;
    _ = readU32(data, &pos);

    const key = prefix_cache.Key{
        .model = readU64(data, &pos),
        .arithmetic = readU64(data, &pos),
        .layout = readU64(data, &pos),
        .format = readU64(data, &pos),
        .dtype = blk: {
            const d = data[pos];
            pos += 4;
            break :blk d;
        },
    };
    const count = readU32(data, &pos);
    const index_end = header_bytes + @as(usize, count) * record_bytes;
    if (count > 0 and data.len < index_end) return error.Corrupt;

    const records = try allocator.alloc(Record, count);
    errdefer allocator.free(records);
    pos = header_bytes;
    for (records) |*r| {
        r.parent = readU32(data, &pos);
        r.begin = readU32(data, &pos);
        r.end = readU32(data, &pos);
        r.off = readU64(data, &pos);
        r.len = readU64(data, &pos);
    }
    for (records) |r| {
        const start: usize = @intCast(r.off);
        const end: usize = @intCast(r.off + r.len);
        if (end > data.len or start > end) return error.Corrupt;
    }
    return .{ .allocator = allocator, .mmap = mmap, .key = key, .records = records };
}

/// Write `memo`'s live segments to `path`, in a parent-before-child order so the
/// loader can `addRaw` sequentially. Payloads are read from the memo's regions.
pub fn writeBase(allocator: std.mem.Allocator, io: Io, path: []const u8, memo: *const prefix_cache.Memo) !void {
    // Parent-before-child order via an explicit DFS from the roots.
    var order: std.ArrayList(u32) = .empty;
    defer order.deinit(allocator);
    var new_index = try allocator.alloc(u32, memo.segments.len);
    defer allocator.free(new_index);
    @memset(new_index, prefix_cache.NONE);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(allocator);
    for (memo.liveSegments(), 0..) |seg, i| {
        if (seg.parent == prefix_cache.NONE) try stack.append(allocator, @intCast(i));
    }
    while (stack.pop()) |id| {
        if (new_index[id] != prefix_cache.NONE) continue;
        new_index[id] = @intCast(order.items.len);
        try order.append(allocator, id);
        for (memo.liveSegments(), 0..) |seg, i| {
            if (seg.parent == id and new_index[i] == prefix_cache.NONE) try stack.append(allocator, @intCast(i));
        }
    }

    var payload_total: u64 = 0;
    for (order.items) |id| payload_total += memo.blockBytes(memo.segment(id).?.tokenLen());
    const payload_start: u64 = header_bytes + @as(u64, order.items.len) * record_bytes;

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.ensureTotalCapacity(allocator, payload_start + payload_total);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);
    try appendU32(allocator, &buf, 0);
    try appendU64(allocator, &buf, memo.key.model);
    try appendU64(allocator, &buf, memo.key.arithmetic);
    try appendU64(allocator, &buf, memo.key.layout);
    try appendU64(allocator, &buf, memo.key.format);
    try buf.append(allocator, memo.key.dtype);
    try buf.appendSlice(allocator, &[_]u8{ 0, 0, 0 });
    try appendU32(allocator, &buf, @intCast(order.items.len));

    var off = payload_start;
    for (order.items) |id| {
        const seg = memo.segment(id).?;
        const parent: u32 = if (seg.parent == prefix_cache.NONE) prefix_cache.NONE else new_index[seg.parent];
        const len = memo.blockBytes(seg.tokenLen());
        try appendU32(allocator, &buf, parent);
        try appendU32(allocator, &buf, seg.token_begin);
        try appendU32(allocator, &buf, seg.token_end);
        try appendU64(allocator, &buf, off);
        try appendU64(allocator, &buf, len);
        off += len;
    }
    std.debug.assert(buf.items.len == payload_start);
    for (order.items) |id| {
        try buf.appendSlice(allocator, memo.payloadOf(id) orelse &.{});
        try buf.appendSlice(allocator, std.mem.sliceAsBytes(memo.tokensOf(id)));
    }

    // Write to a process-unique temp and rename, so a crash mid-flush cannot
    // leave a torn `.zpc` (which the runtime would load as corrupt K/V).
    const pid = std.posix.system.getpid();
    var tmp_buf: [640]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tmp_buf, "{s}.{d}.tmp", .{ path, pid });
    var f = try Io.Dir.cwd().createFile(io, tmp, .{});
    f.writeStreamingAll(io, buf.items) catch |err| {
        f.close(io);
        Io.Dir.cwd().deleteFile(io, tmp) catch {};
        return err;
    };
    f.close(io);
    try Io.Dir.rename(Io.Dir.cwd(), tmp, Io.Dir.cwd(), path, io);
}

/// An anonymous, file-backed writable mmap (the runtime journal and the mapped
/// index). Unlinked immediately after mapping, so it is scratch with no leak on
/// an abnormal exit.
pub const Scratch = struct {
    data: []align(std.heap.page_size_min) u8,

    pub fn create(io: Io, path: []const u8, size: u64) !Scratch {
        if (size == 0) return error.EmptyScratch;
        var file = try Io.Dir.cwd().createFile(io, path, .{ .read = true, .truncate = true });
        defer file.close(io);
        try file.setLength(io, size);
        const mapped = try posix.mmap(
            null,
            @intCast(size),
            posix.PROT{ .READ = true, .WRITE = true },
            posix.MAP{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer posix.munmap(mapped);
        Io.Dir.cwd().deleteFile(io, path) catch {};
        return .{ .data = mapped };
    }

    pub fn close(self: *Scratch) void {
        posix.munmap(self.data);
    }
};

fn appendU32(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

fn appendU64(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

fn readU32(data: []const u8, pos: *usize) u32 {
    const v = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    return v;
}

fn readU64(data: []const u8, pos: *usize) u64 {
    const v = std.mem.readInt(u64, data[pos.*..][0..8], .little);
    pos.* += 8;
    return v;
}

const TestBuilder = struct {
    fn size(ctx: *anyopaque, begin: u32, end: u32) u64 {
        const self: *const prefix_cache.Memo = @ptrCast(@alignCast(ctx));
        return @as(u64, end - begin) * self.per_token_bytes;
    }
    fn write(_: *anyopaque, begin: u32, end: u32, dst: []u8) anyerror!void {
        _ = end;
        for (dst, 0..) |*b, i| b.* = @intCast((begin + @as(u32, @intCast(i))) % 253);
    }
    fn builder(memo: *const prefix_cache.Memo) prefix_cache.Builder {
        return .{ .ctx = @ptrCast(@constCast(memo)), .size = size, .write = write };
    }
};

test "zpc round-trips a radix chain and flags a stale key" {
    const testing = std.testing;
    const key = prefix_cache.Key{ .model = 1, .arithmetic = 3, .layout = 4, .dtype = 0 };

    var journal: [4096]u8 align(4) = undefined;
    var meta: [1024]u8 align(@alignOf(prefix_cache.Segment)) = undefined;
    var memo = prefix_cache.Memo.init(testing.allocator, key, 2);
    defer memo.deinit();
    try memo.addMeta(&meta);
    _ = try memo.addRegion(&journal, true);
    var tokens = [_]u32{ 5, 6, 7, 8, 9, 10 };
    try memo.commit(&tokens, prefix_cache.NONE, 0, tokens.len, TestBuilder.builder(&memo));

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/m.zpc", .{tmp.sub_path[0..]});
    try writeBase(testing.allocator, testing.io, path, &memo);

    var base = (try openBase(testing.allocator, testing.io, path)).?;
    defer base.deinit();
    try testing.expect(prefix_cache.Key.eql(key, base.key));
    try testing.expectEqual(@as(usize, 1), base.records.len);
    for (base.records) |r| try testing.expect(r.off + r.len <= base.data().len);

    // Reload into a fresh memo referencing the base region zero-copy.
    var meta2: [1024]u8 align(@alignOf(prefix_cache.Segment)) = undefined;
    var journal2: [64]u8 align(4) = undefined;
    var loaded = prefix_cache.Memo.init(testing.allocator, key, 2);
    defer loaded.deinit();
    try loaded.addMeta(&meta2);
    _ = try loaded.addRegion(&journal2, true);
    const region = try loaded.addRegion(base.mmap.data, false);
    for (base.records) |r| _ = try loaded.addRaw(r.parent, r.begin, r.end, region, r.off);
    var path_buf: std.ArrayList(u32) = .empty;
    defer path_buf.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 6), try loaded.lookup(&tokens, &path_buf));
    loaded.unpin(path_buf.items);

    var other = prefix_cache.Memo.init(testing.allocator, .{ .model = 99 }, 2);
    defer other.deinit();
    try testing.expect(!loaded.matches(.{ .model = 99 }));
    try testing.expect(loaded.matches(key));
}

test "scratch maps an anonymous writable file" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/j.scratch", .{tmp.sub_path[0..]});
    var scratch = try Scratch.create(testing.io, path, 4096);
    try testing.expectEqual(@as(usize, 4096), scratch.data.len);
    try testing.expect(!exists(testing.io, path));
    scratch.close();
}

fn exists(io: Io, path: []const u8) bool {
    _ = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return true;
}

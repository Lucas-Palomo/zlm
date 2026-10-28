const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const binary_writer = @import("../utils/binary_writer.zig");
const appendU16 = binary_writer.appendU16;
const appendU32 = binary_writer.appendU32;
const appendU64 = binary_writer.appendU64;

/// Runtime alias registry magic and format version.
pub const MAGIC = "ZRUN";
pub const VERSION: u32 = 1;
pub const NAME = "manifest.zrun";

const TMP_NAME = "manifest.zrun.tmp";
const hash_bytes: usize = 8;
const header_bytes: usize = 4 + 4 + 4 + 4;
const max_entries: u32 = 1 << 20;

/// One alias -> shared `(image, fingerprint)` artifact row. `plan`/`tune` are
/// the artifact filenames under `runtimes/`; `image`/`fp` are kept for display
/// and validation.
pub const Entry = struct {
    id: []const u8,
    plan: []const u8,
    tune: []const u8,
    image: []const u8,
    fp: u64,
};

/// Owned view of `manifest.zrun`: entries own their strings.
pub const Registry = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,

    pub fn deinit(self: *Registry) void {
        for (self.entries) |e| freeEntry(self.allocator, e);
        self.allocator.free(self.entries);
    }

    pub fn find(self: *const Registry, id: []const u8) ?*const Entry {
        for (self.entries) |*e| {
            if (std.mem.eql(u8, e.id, id)) return e;
        }
        return null;
    }
};

/// Load `manifest.zrun` from `dir`, or null when it is missing, has a bad
/// magic/version/checksum, or is otherwise corrupt.
pub fn load(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) ?Registry {
    return loadInner(allocator, dir, io) catch null;
}

fn loadInner(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Registry {
    var mmap = try MappedFile.openIn(dir, io, NAME);
    defer mmap.close();
    if (mmap.data.len < header_bytes + hash_bytes) return error.Corrupt;

    const body = mmap.data[0 .. mmap.data.len - hash_bytes];
    var stored: [hash_bytes]u8 = undefined;
    @memcpy(&stored, mmap.data[mmap.data.len - hash_bytes ..]);
    if (std.hash.Wyhash.hash(0, body) != std.mem.readInt(u64, &stored, .little)) {
        return error.BadChecksum;
    }

    var br = BinaryReader.init(body);
    if (!std.mem.eql(u8, try br.readSlice(4), MAGIC)) return error.BadMagic;
    if (try br.readU32() != VERSION) return error.BadVersion;
    _ = try br.readU32();
    const count = try br.readU32();
    if (count > max_entries) return error.Corrupt;

    const entries = try allocator.alloc(Entry, count);
    errdefer allocator.free(entries);
    var read: usize = 0;
    errdefer {
        for (entries[0..read]) |e| freeEntry(allocator, e);
    }
    for (entries) |*e| {
        e.* = try readEntry(allocator, &br);
        read += 1;
    }
    return .{ .allocator = allocator, .entries = entries };
}

/// Serialize `entries` to `manifest.zrun` atomically: write `manifest.zrun.tmp`
/// then rename over the registry.
pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, entries: []const Entry) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);
    try appendU32(allocator, &buf, 0);
    try appendU32(allocator, &buf, @intCast(entries.len));
    for (entries) |e| {
        try appendStr(allocator, &buf, e.id);
        try appendStr(allocator, &buf, e.plan);
        try appendStr(allocator, &buf, e.tune);
        try appendStr(allocator, &buf, e.image);
        try appendU64(allocator, &buf, e.fp);
        try appendU32(allocator, &buf, 0);
    }
    try appendU64(allocator, &buf, std.hash.Wyhash.hash(0, buf.items));

    var f = try dir.createFile(io, TMP_NAME, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
    try Io.Dir.rename(dir, TMP_NAME, dir, NAME, io);
}

/// Insert `entry`, replacing the row with the same `id` when present. The
/// artifacts are not touched; only the registry is rewritten.
pub fn upsert(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, entry: Entry) !void {
    var reg = load(allocator, dir, io) orelse Registry{ .allocator = allocator, .entries = &.{} };
    defer reg.deinit();

    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(allocator);
    var replaced = false;
    for (reg.entries) |e| {
        if (std.mem.eql(u8, e.id, entry.id)) {
            try list.append(allocator, entry);
            replaced = true;
        } else {
            try list.append(allocator, e);
        }
    }
    if (!replaced) try list.append(allocator, entry);
    try write(allocator, dir, io, list.items);
}

/// Drop the row with `id` and rewrite the registry. Returns whether it existed.
pub fn remove(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, id: []const u8) !bool {
    var reg = load(allocator, dir, io) orelse return false;
    defer reg.deinit();

    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(allocator);
    var found = false;
    for (reg.entries) |e| {
        if (std.mem.eql(u8, e.id, id)) {
            found = true;
            continue;
        }
        try list.append(allocator, e);
    }
    if (!found) return false;
    try write(allocator, dir, io, list.items);
    return true;
}

/// Delete `.zplan`/`.zat` files in `dir` that no entry references, and `.zpc`
/// cache files whose **image no longer exists** (`existing_images`). A `.zpc` is
/// keyed by the K/V fingerprint and tied to the **image**, not to a plan or an
/// alias: it stays valid as long as the image is in the store, so `--prune` (a
/// runtime-registry operation) must not reclaim it just because the image has no
/// runtime. (A `.zpc` made stale by an arithmetic change within a live image is
/// not detected here — that needs the plan's K/V fingerprint — and is left to a
/// future compaction.) Returns the number of files removed.
pub fn pruneOrphans(
    allocator: std.mem.Allocator,
    dir: Io.Dir,
    io: Io,
    entries: []const Entry,
    existing_images: []const []const u8,
) !usize {
    var referenced: std.StringHashMapUnmanaged(void) = .empty;
    defer referenced.deinit(allocator);
    var images: std.StringHashMapUnmanaged(void) = .empty;
    defer images.deinit(allocator);
    for (entries) |e| {
        try referenced.put(allocator, e.plan, {});
        try referenced.put(allocator, e.tune, {});
    }
    for (existing_images) |img| try images.put(allocator, img, {});

    var orphans: std.ArrayList([]const u8) = .empty;
    defer {
        for (orphans.items) |n| allocator.free(n);
        orphans.deinit(allocator);
    }

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (isPrefixCache(entry.name)) {
            if (!imageUsed(entry.name, &images)) try orphans.append(allocator, try allocator.dupe(u8, entry.name));
            continue;
        }
        if (!isArtifact(entry.name)) continue;
        if (referenced.contains(entry.name)) continue;
        try orphans.append(allocator, try allocator.dupe(u8, entry.name));
    }

    var removed: usize = 0;
    for (orphans.items) |name| {
        dir.deleteFile(io, name) catch continue;
        removed += 1;
    }
    return removed;
}

fn isArtifact(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".zplan") or std.mem.endsWith(u8, name, ".zat");
}

fn isPrefixCache(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".zpc");
}

/// A `.zpc` is `<image>_<prefix_cache_fp>.zpc`; it is used when some entry's image is
/// the name's prefix (immediately followed by `_`).
fn imageUsed(name: []const u8, images: *const std.StringHashMapUnmanaged(void)) bool {
    var it = images.keyIterator();
    while (it.next()) |img| {
        if (std.mem.startsWith(u8, name, img.*) and name.len > img.len and name[img.len] == '_') return true;
    }
    return false;
}

fn readEntry(allocator: std.mem.Allocator, br: *BinaryReader) !Entry {
    const id = try dupeStr(allocator, br);
    errdefer allocator.free(id);
    const plan = try dupeStr(allocator, br);
    errdefer allocator.free(plan);
    const tune = try dupeStr(allocator, br);
    errdefer allocator.free(tune);
    const image = try dupeStr(allocator, br);
    errdefer allocator.free(image);
    const fp = try br.readU64();
    _ = try br.readU32();
    return .{ .id = id, .plan = plan, .tune = tune, .image = image, .fp = fp };
}

fn freeEntry(allocator: std.mem.Allocator, e: Entry) void {
    allocator.free(e.id);
    allocator.free(e.plan);
    allocator.free(e.tune);
    allocator.free(e.image);
}

fn dupeStr(allocator: std.mem.Allocator, br: *BinaryReader) ![]const u8 {
    return allocator.dupe(u8, try br.readSlice(try br.readU16()));
}

fn appendStr(allocator: std.mem.Allocator, l: *std.ArrayList(u8), s: []const u8) !void {
    const n = @min(s.len, 0xFFFF);
    try appendU16(allocator, l, @intCast(n));
    try l.appendSlice(allocator, s[0..n]);
}

test "registry round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try testing.expect(load(testing.allocator, tmp.dir, testing.io) == null);

    const entries = [_]Entry{
        .{ .id = "rt-a", .plan = "e2b_ab.zplan", .tune = "e2b_ab.zat", .image = "e2b", .fp = 0x1122 },
        .{ .id = "rt-b", .plan = "e2b_ab.zplan", .tune = "e2b_ab.zat", .image = "e2b", .fp = 0x1122 },
    };
    try write(testing.allocator, tmp.dir, testing.io, &entries);

    var reg = load(testing.allocator, tmp.dir, testing.io) orelse return error.LoadFailed;
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 2), reg.entries.len);
    const a = reg.find("rt-a") orelse return error.Missing;
    try testing.expectEqualStrings("e2b_ab.zplan", a.plan);
    try testing.expectEqualStrings("e2b_ab.zat", a.tune);
    try testing.expectEqual(@as(u64, 0x1122), a.fp);
    try testing.expect(reg.find("absent") == null);
}

test "upsert replaces by id and appends a shared alias" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try upsert(testing.allocator, tmp.dir, testing.io, .{
        .id = "rt-a",
        .plan = "e2b_ab.zplan",
        .tune = "e2b_ab.zat",
        .image = "e2b",
        .fp = 1,
    });
    try upsert(testing.allocator, tmp.dir, testing.io, .{
        .id = "rt-b",
        .plan = "e2b_ab.zplan",
        .tune = "e2b_ab.zat",
        .image = "e2b",
        .fp = 1,
    });
    var reg = load(testing.allocator, tmp.dir, testing.io) orelse return error.LoadFailed;
    try testing.expectEqual(@as(usize, 2), reg.entries.len);
    reg.deinit();

    try upsert(testing.allocator, tmp.dir, testing.io, .{
        .id = "rt-a",
        .plan = "e2b_cd.zplan",
        .tune = "e2b_cd.zat",
        .image = "e2b",
        .fp = 2,
    });
    reg = load(testing.allocator, tmp.dir, testing.io) orelse return error.LoadFailed;
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 2), reg.entries.len);
    try testing.expectEqualStrings("e2b_cd.zplan", reg.find("rt-a").?.plan);
    try testing.expectEqualStrings("e2b_ab.zplan", reg.find("rt-b").?.plan);
}

test "remove drops one alias and reports existence" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try upsert(testing.allocator, tmp.dir, testing.io, .{
        .id = "rt-a",
        .plan = "e2b_ab.zplan",
        .tune = "e2b_ab.zat",
        .image = "e2b",
        .fp = 1,
    });
    try testing.expect(try remove(testing.allocator, tmp.dir, testing.io, "rt-a"));
    try testing.expect(!try remove(testing.allocator, tmp.dir, testing.io, "rt-a"));

    var reg = load(testing.allocator, tmp.dir, testing.io) orelse return error.LoadFailed;
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 0), reg.entries.len);
}

test "load rejects corruption" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try upsert(testing.allocator, tmp.dir, testing.io, .{
        .id = "rt-a",
        .plan = "e2b_ab.zplan",
        .tune = "e2b_ab.zat",
        .image = "e2b",
        .fp = 1,
    });

    var file = try tmp.dir.openFile(testing.io, NAME, .{ .mode = .read_write });
    defer file.close(testing.io);
    try file.writePositionalAll(testing.io, &[_]u8{0xFF}, 0);
    try testing.expect(load(testing.allocator, tmp.dir, testing.io) == null);
}

test "pruneOrphans removes only unreferenced artifacts" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const entries = [_]Entry{
        .{ .id = "rt-a", .plan = "e2b_ab.zplan", .tune = "e2b_ab.zat", .image = "e2b", .fp = 1 },
        .{ .id = "rt-b", .plan = "e2b_ab.zplan", .tune = "e2b_ab.zat", .image = "e2b", .fp = 1 },
    };
    try write(testing.allocator, tmp.dir, testing.io, &entries);

    for ([_][]const u8{ "e2b_ab.zplan", "e2b_ab.zat", "e2b_cd.zplan", "e2b_cd.zat", "keep.txt", "e2b_deadbeef.zpc", "ghost_deadbeef.zpc" }) |name| {
        var f = try tmp.dir.createFile(testing.io, name, .{});
        f.close(testing.io);
    }

    const existing = [_][]const u8{"e2b"};
    const removed = try pruneOrphans(testing.allocator, tmp.dir, testing.io, &entries, &existing);
    try testing.expectEqual(@as(usize, 3), removed);
    try testing.expect(testExists(tmp.dir, testing.io, "e2b_ab.zplan"));
    try testing.expect(testExists(tmp.dir, testing.io, "e2b_ab.zat"));
    try testing.expect(!testExists(tmp.dir, testing.io, "e2b_cd.zplan"));
    try testing.expect(!testExists(tmp.dir, testing.io, "e2b_cd.zat"));
    try testing.expect(testExists(tmp.dir, testing.io, "keep.txt"));
    // A `.zpc` of a live image survives; an image-less `.zpc` is reclaimed.
    try testing.expect(testExists(tmp.dir, testing.io, "e2b_deadbeef.zpc"));
    try testing.expect(!testExists(tmp.dir, testing.io, "ghost_deadbeef.zpc"));
}

fn testExists(dir: Io.Dir, io: Io, name: []const u8) bool {
    _ = dir.statFile(io, name, .{}) catch return false;
    return true;
}

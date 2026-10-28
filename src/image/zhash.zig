const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const ztef = @import("ztef.zig");

pub const MAGIC = "ZHAS";
pub const VERSION: u32 = 1;
pub const NAME = "model.zhash";
const SHA_BYTES: usize = 32;

/// Every file in the image, plus its sha256. Covers `zman`, `zmeta` and the
/// `*.ztef` files; the trailing self-sha covers this file.
pub const Kind = enum(u8) {
    q_proj = 0,
    k_proj = 1,
    v_proj = 2,
    o_proj = 3,
    gate_proj = 4,
    up_proj = 5,
    down_proj = 6,
    norms = 7,
    embd = 8,
    zmeta = 9,
    zman = 10,
    zcom = 11,
    // Appended so the values shared with `ztef.Kind` and the earlier
    // `zmeta`/`zman`/`zcom` ids keep their on-disk meaning.
    misc = 12,

    pub fn filename(self: Kind) []const u8 {
        return switch (self) {
            .q_proj => "q_proj.ztef",
            .k_proj => "k_proj.ztef",
            .v_proj => "v_proj.ztef",
            .o_proj => "o_proj.ztef",
            .gate_proj => "gate_proj.ztef",
            .up_proj => "up_proj.ztef",
            .down_proj => "down_proj.ztef",
            .norms => "norms.ztef",
            .embd => "embd.ztef",
            .zmeta => "model.zmeta",
            .zman => "model.zman",
            .zcom => "model.zcom",
            .misc => "misc.ztef",
        };
    }
};

pub fn ofZtef(k: ztef.Kind) Kind {
    return switch (k) {
        .misc => .misc,
        else => @enumFromInt(@intFromEnum(k)),
    };
}

pub const Entry = struct {
    kind: Kind,
    sha256: [32]u8,
};

pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, entries: []const Entry) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);

    try body.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &body, VERSION);
    try appendU32(allocator, &body, @intCast(entries.len));
    for (entries) |e| {
        try body.append(allocator, @intFromEnum(e.kind));
        try body.appendSlice(allocator, &e.sha256);
    }

    var self_sha: [SHA_BYTES]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body.items, &self_sha, .{});

    var file = try dir.createFile(io, NAME, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, body.items);
    try file.writeStreamingAll(io, &self_sha);
}

pub const Reader = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    entries: []Entry,

    pub fn open(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Reader {
        var mmap = try MappedFile.openIn(dir, io, NAME);
        errdefer mmap.close();

        if (mmap.data.len < SHA_BYTES) return error.BadLayout;
        const body = mmap.data[0 .. mmap.data.len - SHA_BYTES];
        const expected = mmap.data[mmap.data.len - SHA_BYTES ..];

        var actual: [SHA_BYTES]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(body, &actual, .{});
        if (!std.mem.eql(u8, &actual, expected)) return error.BadChecksum;

        var br = BinaryReader.init(body);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        const version = try br.readU32();
        if (version != VERSION) return error.BadVersion;
        const count = try br.readU32();
        if (count > 64) return error.BadLayout;

        const entries = try allocator.alloc(Entry, count);
        errdefer allocator.free(entries);
        for (entries) |*e| {
            const kind = @as(Kind, @enumFromInt(try br.readU8()));
            const sha_slice = try br.readSlice(SHA_BYTES);
            var sha: [SHA_BYTES]u8 = undefined;
            @memcpy(&sha, sha_slice);
            e.* = .{ .kind = kind, .sha256 = sha };
        }

        return .{ .allocator = allocator, .mmap = mmap, .entries = entries };
    }

    pub fn deinit(self: *Reader) void {
        self.allocator.free(self.entries);
        self.mmap.close();
    }
};

fn appendU32(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

test "zhash round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var s0: [32]u8 = undefined;
    for (&s0, 0..) |*b, i| b.* = @intCast(i);
    var s1: [32]u8 = undefined;
    for (&s1, 0..) |*b, i| b.* = @intCast(i + 1);

    const entries = [_]Entry{
        .{ .kind = .embd, .sha256 = s0 },
        .{ .kind = .zman, .sha256 = s1 },
    };
    try write(testing.allocator, tmp.dir, testing.io, &entries);

    var r = try Reader.open(testing.allocator, tmp.dir, testing.io);
    defer r.deinit();
    try testing.expectEqual(@as(usize, 2), r.entries.len);
    try testing.expectEqual(Kind.embd, r.entries[0].kind);
    try testing.expectEqualStrings("embd.ztef", r.entries[0].kind.filename());
    try testing.expectEqualStrings("model.zman", r.entries[1].kind.filename());
    try testing.expect(std.mem.eql(u8, &r.entries[1].sha256, &s1));
}

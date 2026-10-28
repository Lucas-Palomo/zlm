const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const binary_writer = @import("../utils/binary_writer.zig");
const appendU16 = binary_writer.appendU16;
const appendU32 = binary_writer.appendU32;
const appendU64 = binary_writer.appendU64;
const ztef = @import("ztef.zig");

pub const MAGIC = "ZFMA";
pub const VERSION: u32 = 1;
pub const NAME = "model.zman";

pub const FileRec = struct {
    kind: ztef.Kind,
    entry_count: u32,
    byte_length: u64,
};

pub const Info = struct {
    content_root: [32]u8,
    gguf_size: u64,
    arch: []const u8,
    tensor_count: u32,
    metadata_kv_count: u64,
    files: []const FileRec,
};

pub const Written = struct { bytes: u64, sha256: [32]u8 };

pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, info: Info) !Written {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);
    try buf.appendSlice(allocator, &info.content_root);
    try appendU64(allocator, &buf, info.gguf_size);
    try appendU16(allocator, &buf, @intCast(info.arch.len));
    try buf.appendSlice(allocator, info.arch);
    try appendU32(allocator, &buf, info.tensor_count);
    try appendU64(allocator, &buf, info.metadata_kv_count);
    try appendU32(allocator, &buf, @intCast(info.files.len));
    for (info.files) |f| {
        try appendU32(allocator, &buf, @intFromEnum(f.kind));
        try appendU32(allocator, &buf, f.entry_count);
        try appendU64(allocator, &buf, f.byte_length);
    }

    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(buf.items, &sha, .{});

    var file = try dir.createFile(io, NAME, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, buf.items);
    return .{ .bytes = buf.items.len, .sha256 = sha };
}

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    content_root: [32]u8,
    gguf_size: u64,
    arch: []const u8,
    tensor_count: u32,
    metadata_kv_count: u64,
    files: []FileRec,

    pub fn open(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Manifest {
        var mmap = try MappedFile.openIn(dir, io, NAME);
        errdefer mmap.close();

        var br = BinaryReader.init(mmap.data);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        const version = try br.readU32();
        if (version != VERSION) return error.BadVersion;

        const sha_slice = try br.readSlice(32);
        var sha: [32]u8 = undefined;
        @memcpy(&sha, sha_slice);
        const gguf_size = try br.readU64();
        const arch = try br.readSlice(try br.readU16());
        const tensor_count = try br.readU32();
        const metadata_kv_count = try br.readU64();
        const file_count = try br.readU32();

        const files = try allocator.alloc(FileRec, file_count);
        errdefer allocator.free(files);
        for (files) |*f| {
            const kind = @as(ztef.Kind, @enumFromInt(try br.readU32()));
            const entry_count = try br.readU32();
            const byte_length = try br.readU64();
            f.* = .{ .kind = kind, .entry_count = entry_count, .byte_length = byte_length };
        }

        return .{
            .allocator = allocator,
            .mmap = mmap,
            .content_root = sha,
            .gguf_size = gguf_size,
            .arch = arch,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
            .files = files,
        };
    }

    pub fn deinit(self: *Manifest) void {
        self.allocator.free(self.files);
        self.mmap.close();
    }
};

test "manifest round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const files = [_]FileRec{
        .{ .kind = .q_proj, .entry_count = 2, .byte_length = 36 },
        .{ .kind = .norms, .entry_count = 1, .byte_length = 18 },
    };
    const arch = "gemma4";
    var sha: [32]u8 = undefined;
    for (&sha, 0..) |*b, i| b.* = @intCast(i);

    const w = try write(testing.allocator, tmp.dir, testing.io, .{
        .content_root = sha,
        .gguf_size = 123,
        .arch = arch,
        .tensor_count = 3,
        .metadata_kv_count = 9,
        .files = &files,
    });
    try testing.expect(w.bytes > 0);

    var m = try Manifest.open(testing.allocator, tmp.dir, testing.io);
    defer m.deinit();
    try testing.expectEqual(@as(u64, 123), m.gguf_size);
    try testing.expectEqualStrings("gemma4", m.arch);
    try testing.expectEqual(@as(u32, 3), m.tensor_count);
    try testing.expectEqual(@as(usize, 2), m.files.len);
    try testing.expectEqual(ztef.Kind.q_proj, m.files[0].kind);
    try testing.expectEqual(@as(u64, 18), m.files[1].byte_length);
}

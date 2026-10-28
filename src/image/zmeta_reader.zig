const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const format = @import("../gguf/format.zig");

pub const MAGIC = "ZMTA";
pub const VERSION: u32 = 1;
pub const NAME = "model.zmeta";

/// Keys the compile appends to `zmeta` for the resolved chat template. The
/// runtime reads them and falls back to the arch template when absent.
pub const CHAT_USER_PREFIX = "zlm.chat.user_prefix";
pub const CHAT_USER_SUFFIX = "zlm.chat.user_suffix";
pub const CHAT_MODEL_PREFIX = "zlm.chat.model_prefix";

pub const Written = struct { bytes: u64, sha256: [32]u8 };

/// Metadata store: reads `model.zmeta` (the GGUF metadata, re-encoded at
/// compile time). Same accessors as `gguf.Gguf` so callers are source-agnostic.
pub const Zmeta = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    metadata: std.StringHashMap(format.MetadataValue),
    keys: [][]const u8,

    pub fn open(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Zmeta {
        var mmap = try MappedFile.openIn(dir, io, NAME);
        errdefer mmap.close();

        var br = BinaryReader.init(mmap.data);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        if (try br.readU32() != VERSION) return error.BadVersion;

        var metadata = std.StringHashMap(format.MetadataValue).init(allocator);
        errdefer metadata.deinit();
        var keys: std.ArrayList([]const u8) = .empty;
        errdefer keys.deinit(allocator);

        const count = try br.readU32();
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const key = try readString(&br);
            const value = try readValue(&br, allocator);
            try metadata.put(key, value);
            try keys.append(allocator, key);
        }

        return .{
            .allocator = allocator,
            .mmap = mmap,
            .metadata = metadata,
            .keys = try keys.toOwnedSlice(allocator),
        };
    }

    pub fn deinit(self: *Zmeta) void {
        self.allocator.free(self.keys);
        self.metadata.deinit();
        self.mmap.close();
    }

    pub fn architecture(self: *const Zmeta) []const u8 {
        return self.getString("general.architecture") orelse "unknown";
    }

    pub fn getString(self: *const Zmeta, key: []const u8) ?[]const u8 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getBool(self: *const Zmeta, key: []const u8) ?bool {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .bool => |b| b,
            else => null,
        };
    }

    pub fn getUint(self: *const Zmeta, key: []const u8) ?u64 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .uint8 => |x| x,
            .uint16 => |x| x,
            .uint32 => |x| x,
            .uint64 => |x| x,
            .int8 => |x| @intCast(@max(0, x)),
            .int16 => |x| @intCast(@max(0, x)),
            .int32 => |x| @intCast(@max(0, x)),
            .int64 => |x| @intCast(@max(0, x)),
            else => null,
        };
    }

    pub fn getFloat(self: *const Zmeta, key: []const u8) ?f32 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .float32 => |x| x,
            .float64 => |x| @floatCast(x),
            .uint32 => |x| @floatFromInt(x),
            .uint64 => |x| @floatFromInt(x),
            .int32 => |x| @floatFromInt(x),
            .int64 => |x| @floatFromInt(x),
            else => null,
        };
    }

    pub fn getArray(self: *const Zmeta, key: []const u8) ?format.Array {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .array => |a| a,
            else => null,
        };
    }
};

fn readValue(br: *BinaryReader, allocator: std.mem.Allocator) !format.MetadataValue {
    const t = format.MetadataValueType.fromInt(try br.readU32()) orelse return error.Malformed;
    return readElem(br, allocator, t);
}

fn readElem(br: *BinaryReader, allocator: std.mem.Allocator, elem: format.MetadataValueType) !format.MetadataValue {
    return switch (elem) {
        .uint8 => .{ .uint8 = try br.readU8() },
        .int8 => .{ .int8 = try br.readI8() },
        .uint16 => .{ .uint16 = try br.readU16() },
        .int16 => .{ .int16 = try br.readI16() },
        .uint32 => .{ .uint32 = try br.readU32() },
        .int32 => .{ .int32 = try br.readI32() },
        .float32 => .{ .float32 = try br.readF32() },
        .bool => .{ .bool = try br.readBool() },
        .string => .{ .string = try readString(br) },
        .uint64 => .{ .uint64 = try br.readU64() },
        .int64 => .{ .int64 = try br.readI64() },
        .float64 => .{ .float64 = try br.readF64() },
        .array => blk: {
            const inner = format.MetadataValueType.fromInt(try br.readU32()) orelse return error.Malformed;
            const len = try br.readU64();
            const items = try allocator.alloc(format.MetadataValue, @intCast(len));
            for (items) |*item| item.* = try readElem(br, allocator, inner);
            break :blk .{ .array = .{ .elem = inner, .items = items } };
        },
    };
}

fn readString(br: *BinaryReader) ![]const u8 {
    return br.readSlice(try br.readU16());
}

test "zmeta reader parses a synthetic blob" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // Build a minimal zmeta by hand (magic|version|count|entries).
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    try buf.appendSlice(testing.allocator, MAGIC);
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, VERSION, .little);
    try buf.appendSlice(testing.allocator, &b4);
    std.mem.writeInt(u32, &b4, 2, .little);
    try buf.appendSlice(testing.allocator, &b4);

    const k1 = "general.architecture";
    var b2: [2]u8 = undefined;
    std.mem.writeInt(u16, &b2, k1.len, .little);
    try buf.appendSlice(testing.allocator, &b2);
    try buf.appendSlice(testing.allocator, k1);
    std.mem.writeInt(u32, &b4, @intFromEnum(format.MetadataValueType.string), .little);
    try buf.appendSlice(testing.allocator, &b4);
    const v1 = "gemma4";
    std.mem.writeInt(u16, &b2, v1.len, .little);
    try buf.appendSlice(testing.allocator, &b2);
    try buf.appendSlice(testing.allocator, v1);

    const k2 = "gemma4.block_count";
    std.mem.writeInt(u16, &b2, k2.len, .little);
    try buf.appendSlice(testing.allocator, &b2);
    try buf.appendSlice(testing.allocator, k2);
    std.mem.writeInt(u32, &b4, @intFromEnum(format.MetadataValueType.uint32), .little);
    try buf.appendSlice(testing.allocator, &b4);
    std.mem.writeInt(u32, &b4, 35, .little);
    try buf.appendSlice(testing.allocator, &b4);

    var f = try tmp.dir.createFile(testing.io, NAME, .{});
    defer f.close(testing.io);
    try f.writeStreamingAll(testing.io, buf.items);

    var z = try Zmeta.open(testing.allocator, tmp.dir, testing.io);
    defer z.deinit();
    try testing.expectEqualStrings("gemma4", z.architecture());
    try testing.expectEqual(@as(?u64, 35), z.getUint("gemma4.block_count"));
    try testing.expectEqual(@as(usize, 2), z.keys.len);
}

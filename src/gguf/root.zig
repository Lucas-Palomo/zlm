const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const format = @import("format.zig");
const reader = @import("reader.zig");

/// Read-only view of a GGUF file: header, metadata and tensor directory over a
/// memory mapping. Strings/names are zero-copy slices into the mapping.
pub const Gguf = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    header: format.Header,
    metadata: std.StringHashMap(format.MetadataValue),
    keys: [][]const u8,
    tensors: []format.TensorInfo,
    data_offset: u64,

    pub fn open(allocator: std.mem.Allocator, io: Io, path: []const u8) !Gguf {
        var mmap = try MappedFile.open(io, path);
        errdefer mmap.close();

        var br = BinaryReader.init(mmap.data);
        const header = try reader.readHeader(&br);

        var metadata = std.StringHashMap(format.MetadataValue).init(allocator);
        errdefer metadata.deinit();
        var keys: std.ArrayList([]const u8) = .empty;
        errdefer keys.deinit(allocator);
        try reader.readMetadata(&br, allocator, header.metadata_kv_count, &metadata, &keys);

        const tensors = try reader.readTensors(&br, allocator, header.tensor_count);
        errdefer allocator.free(tensors);

        const alignment: u64 = blk: {
            if (metadata.get("general.alignment")) |v| {
                switch (v) {
                    .uint32 => |x| break :blk x,
                    .uint64 => |x| break :blk x,
                    else => {},
                }
            }
            break :blk format.DEFAULT_ALIGNMENT;
        };

        const data_offset = format.alignUp(br.pos, alignment);
        if (data_offset > mmap.size) return error.Truncated;

        return .{
            .allocator = allocator,
            .mmap = mmap,
            .header = header,
            .metadata = metadata,
            .keys = try keys.toOwnedSlice(allocator),
            .tensors = tensors,
            .data_offset = data_offset,
        };
    }

    pub fn deinit(self: *Gguf) void {
        self.allocator.free(self.keys);
        self.metadata.deinit();
        self.allocator.free(self.tensors);
        self.mmap.close();
    }

    pub fn architecture(self: *const Gguf) []const u8 {
        return self.getString("general.architecture") orelse "unknown";
    }

    pub fn findTensor(self: *const Gguf, name: []const u8) ?*const format.TensorInfo {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn getString(self: *const Gguf, key: []const u8) ?[]const u8 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getBool(self: *const Gguf, key: []const u8) ?bool {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .bool => |b| b,
            else => null,
        };
    }

    pub fn getArray(self: *const Gguf, key: []const u8) ?format.Array {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .array => |a| a,
            else => null,
        };
    }

    pub fn getUint(self: *const Gguf, key: []const u8) ?u64 {
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

    /// Tensor payload, a slice into the mapping.
    pub fn tensorBytes(self: *const Gguf, t: *const format.TensorInfo) ![]const u8 {
        const len = try t.byteLength();
        const start = self.data_offset + t.offset;
        if (start + len > self.mmap.data.len) return error.Truncated;
        return self.mmap.data[start..][0..len];
    }
};

const std = @import("std");
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const format = @import("format.zig");

pub fn readHeader(br: *BinaryReader) format.Error!format.Header {
    const magic = try br.readSlice(4);
    if (!std.mem.eql(u8, magic, format.MAGIC)) return error.BadMagic;

    const version = try br.readU32();
    if (version < format.VERSION_MIN or version > format.VERSION_MAX) {
        return error.UnsupportedVersion;
    }

    return .{
        .version = version,
        .tensor_count = try br.readU64(),
        .metadata_kv_count = try br.readU64(),
    };
}

pub fn readMetadata(
    br: *BinaryReader,
    allocator: std.mem.Allocator,
    count: u64,
    map: *std.StringHashMap(format.MetadataValue),
    keys: *std.ArrayList([]const u8),
) format.Error!void {
    var i: u64 = 0;
    while (i < count) : (i += 1) {
        const key = try br.readStringSlice();
        const value = try readValue(br, allocator);
        try map.put(key, value);
        try keys.append(allocator, key);
    }
}

fn readValue(br: *BinaryReader, allocator: std.mem.Allocator) format.Error!format.MetadataValue {
    const t = format.MetadataValueType.fromInt(try br.readU32()) orelse return error.Malformed;
    return readElem(br, allocator, t);
}

fn readElem(br: *BinaryReader, allocator: std.mem.Allocator, elem: format.MetadataValueType) format.Error!format.MetadataValue {
    return switch (elem) {
        .uint8 => .{ .uint8 = try br.readU8() },
        .int8 => .{ .int8 = try br.readI8() },
        .uint16 => .{ .uint16 = try br.readU16() },
        .int16 => .{ .int16 = try br.readI16() },
        .uint32 => .{ .uint32 = try br.readU32() },
        .int32 => .{ .int32 = try br.readI32() },
        .float32 => .{ .float32 = try br.readF32() },
        .bool => .{ .bool = try br.readBool() },
        .string => .{ .string = try br.readStringSlice() },
        .uint64 => .{ .uint64 = try br.readU64() },
        .int64 => .{ .int64 = try br.readI64() },
        .float64 => .{ .float64 = try br.readF64() },
        .array => blk: {
            const inner = format.MetadataValueType.fromInt(try br.readU32()) orelse return error.Malformed;
            const len = try br.readU64();
            if (len > 1 << 30) return error.Malformed;
            const out = try allocator.alloc(format.MetadataValue, @intCast(len));
            for (out) |*item| item.* = try readElem(br, allocator, inner);
            break :blk .{ .array = .{ .elem = inner, .items = out } };
        },
    };
}

pub fn readTensors(br: *BinaryReader, allocator: std.mem.Allocator, count: u64) format.Error![]format.TensorInfo {
    if (count > 1 << 20) return error.Malformed;
    const out = try allocator.alloc(format.TensorInfo, @intCast(count));
    for (out) |*t| t.* = try readTensor(br);
    return out;
}

fn readTensor(br: *BinaryReader) format.Error!format.TensorInfo {
    const name = try br.readStringSlice();
    const n_dims = try br.readU32();
    if (n_dims > 4) return error.Malformed;

    var dims = [_]u64{0} ** 4;
    for (0..n_dims) |i| dims[i] = try br.readU64();

    const ggml_type = format.GGMLType.fromInt(try br.readU32()) orelse return error.InvalidTensorType;
    return .{
        .name = name,
        .n_dims = n_dims,
        .dims = dims,
        .ggml_type = ggml_type,
        .offset = try br.readU64(),
    };
}

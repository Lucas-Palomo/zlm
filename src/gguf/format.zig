const std = @import("std");

pub const MAGIC = "GGUF";
pub const VERSION_MIN: u32 = 2;
pub const VERSION_MAX: u32 = 3;
pub const DEFAULT_ALIGNMENT: u32 = 32;

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    InvalidTensorType,
    Malformed,
    UnexpectedEOF,
    OutOfMemory,
};

pub const MetadataValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    pub fn fromInt(v: u32) ?MetadataValueType {
        inline for (@typeInfo(MetadataValueType).@"enum".fields) |f| {
            if (v == f.value) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    q2_0 = 42,

    pub fn fromInt(v: u32) ?GGMLType {
        inline for (@typeInfo(GGMLType).@"enum".fields) |f| {
            if (v == f.value) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const Array = struct {
    elem: MetadataValueType,
    items: []MetadataValue,
};

pub const MetadataValue = union(MetadataValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const Header = struct {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    ggml_type: GGMLType,
    offset: u64,

    pub fn dimsSlice(self: *const TensorInfo) []const u64 {
        return self.dims[0..self.n_dims];
    }

    pub fn byteLength(self: *const TensorInfo) !u64 {
        const n = try elementCount(self.dimsSlice());
        return tensorByteLength(self.ggml_type, n);
    }
};

pub const BlockLayout = struct {
    block_bytes: u64,
    block_elems: u64,
};

pub fn blockLayout(t: GGMLType) ?BlockLayout {
    return switch (t) {
        .f32 => .{ .block_bytes = 4, .block_elems = 1 },
        .f16 => .{ .block_bytes = 2, .block_elems = 1 },
        .bf16 => .{ .block_bytes = 2, .block_elems = 1 },
        .q4_0 => .{ .block_bytes = 18, .block_elems = 32 },
        .q4_1 => .{ .block_bytes = 20, .block_elems = 32 },
        .q5_0 => .{ .block_bytes = 22, .block_elems = 32 },
        .q5_1 => .{ .block_bytes = 24, .block_elems = 32 },
        .q8_0 => .{ .block_bytes = 34, .block_elems = 32 },
        .q8_1 => .{ .block_bytes = 36, .block_elems = 32 },
        .q2_k => .{ .block_bytes = 84, .block_elems = 256 },
        .q3_k => .{ .block_bytes = 110, .block_elems = 256 },
        .q4_k => .{ .block_bytes = 144, .block_elems = 256 },
        .q5_k => .{ .block_bytes = 176, .block_elems = 256 },
        .q6_k => .{ .block_bytes = 210, .block_elems = 256 },
        .q8_k => .{ .block_bytes = 292, .block_elems = 256 },
        .i8 => .{ .block_bytes = 1, .block_elems = 1 },
        .i16 => .{ .block_bytes = 2, .block_elems = 1 },
        .i32 => .{ .block_bytes = 4, .block_elems = 1 },
        .i64 => .{ .block_bytes = 8, .block_elems = 1 },
        .f64 => .{ .block_bytes = 8, .block_elems = 1 },
        else => null,
    };
}

pub fn elementCount(dims: []const u64) !u64 {
    var n: u64 = 1;
    for (dims) |d| {
        if (d == 0) return error.Malformed;
        n = std.math.mul(u64, n, d) catch return error.Malformed;
    }
    return n;
}

pub fn tensorByteLength(t: GGMLType, n_elements: u64) !u64 {
    const layout = blockLayout(t) orelse return error.InvalidTensorType;
    const n_blocks = std.math.divCeil(u64, n_elements, layout.block_elems) catch return error.Malformed;
    return std.math.mul(u64, n_blocks, layout.block_bytes) catch return error.Malformed;
}

pub fn alignUp(value: u64, alignment: u64) u64 {
    std.debug.assert(alignment != 0 and std.math.isPowerOfTwo(alignment));
    return (value + alignment - 1) & ~(alignment - 1);
}

test "tensorByteLength" {
    try std.testing.expectEqual(@as(u64, 8192), try tensorByteLength(.f32, 2048));
    try std.testing.expectEqual(@as(u64, 18), try tensorByteLength(.q4_0, 1));
    try std.testing.expectEqual(@as(u64, 36), try tensorByteLength(.q4_0, 64));
    try std.testing.expectEqual(@as(u64, 144), try tensorByteLength(.q4_k, 256));
    try std.testing.expectError(error.InvalidTensorType, tensorByteLength(.iq4_xs, 256));
}

test "align and element count" {
    try std.testing.expectEqual(@as(u64, 32), alignUp(1, 32));
    try std.testing.expectEqual(@as(u64, 32), alignUp(32, 32));
    try std.testing.expectEqual(@as(u64, 64), alignUp(33, 32));
    try std.testing.expectEqual(@as(u64, 6), try elementCount(&[_]u64{ 2, 3 }));
}

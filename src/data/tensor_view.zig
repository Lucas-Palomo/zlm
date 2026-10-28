const std = @import("std");
const types = @import("../gguf/format.zig");

pub const TensorLayout = enum(u32) {
    native_quant,
    q4_0_scale_split,
    q4_0_row_aligned_64,
    q4_1_scale_split,

    pub fn code(self: TensorLayout) u32 {
        return @intFromEnum(self);
    }

    pub fn fromCode(raw: u32) ?TensorLayout {
        inline for (@typeInfo(TensorLayout).@"enum".fields) |f| {
            if (raw == f.value) return @enumFromInt(f.value);
        }
        return null;
    }

    pub fn label(self: TensorLayout) []const u8 {
        return @tagName(self);
    }

    pub fn isNative(self: TensorLayout) bool {
        return self == .native_quant;
    }

    pub fn writesTransformed(self: TensorLayout) bool {
        return self != .native_quant;
    }
};

pub const TensorView = struct {
    name: []const u8,
    dims: []const u64,
    ggml_type: types.GGMLType,
    offset: u64,
    byte_length: u64,
    mapped_slice: ?[]const u8 = null,
    layout: TensorLayout = .native_quant,
    knee_threads: u8 = 0,

    pub fn fromInfo(info: anytype, byte_length: u64) TensorView {
        return .{
            .name = info.name,
            .dims = info.dims[0..info.n_dims],
            .ggml_type = info.ggml_type,
            .offset = info.offset,
            .byte_length = byte_length,
            .mapped_slice = null,
        };
    }
};

test "TensorLayout code round trip" {
    inline for (@typeInfo(TensorLayout).@"enum".fields) |f| {
        const value: TensorLayout = @enumFromInt(f.value);
        try std.testing.expectEqual(f.value, value.code());
        try std.testing.expectEqual(value, TensorLayout.fromCode(f.value).?);
    }
    try std.testing.expect(TensorLayout.fromCode(99) == null);
    try std.testing.expect(TensorLayout.native_quant.isNative());
    try std.testing.expect(!TensorLayout.q4_0_scale_split.isNative());
    try std.testing.expect(TensorLayout.native_quant.writesTransformed() == false);
    try std.testing.expectEqualStrings("q4_0_scale_split", TensorLayout.q4_0_scale_split.label());
}

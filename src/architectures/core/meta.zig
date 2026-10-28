const std = @import("std");
const Zmeta = @import("../../image/zmeta_reader.zig").Zmeta;

/// Metadata helpers shared by model configs (navigate `zmeta`).
pub const LayerArray = struct {
    scalar: ?u64 = null,
    per_layer: ?[]u64 = null,

    pub fn get(self: *const LayerArray, layer: usize) ?u64 {
        if (self.per_layer) |arr| {
            if (layer >= arr.len) return null;
            return arr[layer];
        }
        return self.scalar;
    }

    pub fn deinit(self: *LayerArray, allocator: std.mem.Allocator) void {
        if (self.per_layer) |arr| {
            allocator.free(arr);
            self.per_layer = null;
        }
    }
};

pub const BoolArray = struct {
    values: ?[]bool = null,

    pub fn get(self: *const BoolArray, layer: usize) bool {
        if (self.values) |arr| {
            if (layer >= arr.len) return false;
            return arr[layer];
        }
        return false;
    }

    pub fn deinit(self: *BoolArray, allocator: std.mem.Allocator) void {
        if (self.values) |arr| {
            allocator.free(arr);
            self.values = null;
        }
    }
};

pub fn readLayerArray(md: *const Zmeta, allocator: std.mem.Allocator, key: []const u8) !LayerArray {
    const arr = md.getArray(key) orelse {
        if (md.getUint(key)) |scalar| return .{ .scalar = scalar };
        return .{};
    };
    const out = try allocator.alloc(u64, arr.items.len);
    for (arr.items, out) |v, *o| {
        o.* = switch (v) {
            .uint8 => |x| x,
            .int8 => |x| @intCast(@max(0, x)),
            .uint16 => |x| x,
            .int16 => |x| @intCast(@max(0, x)),
            .uint32 => |x| x,
            .int32 => |x| @intCast(@max(0, x)),
            .uint64 => |x| x,
            .int64 => |x| @intCast(@max(0, x)),
            else => {
                allocator.free(out);
                return error.UnsupportedArchitecture;
            },
        };
    }
    return .{ .per_layer = out };
}

pub fn readBoolArray(md: *const Zmeta, allocator: std.mem.Allocator, key: []const u8) !BoolArray {
    const arr = md.getArray(key) orelse {
        if (md.getBool(key)) |b| {
            const out = try allocator.alloc(bool, 1);
            out[0] = b;
            return .{ .values = out };
        }
        return .{};
    };
    const out = try allocator.alloc(bool, arr.items.len);
    for (arr.items, out) |v, *o| {
        o.* = switch (v) {
            .bool => |b| b,
            .uint8 => |x| x != 0,
            .int8 => |x| x != 0,
            else => {
                allocator.free(out);
                return error.UnsupportedArchitecture;
            },
        };
    }
    return .{ .values = out };
}

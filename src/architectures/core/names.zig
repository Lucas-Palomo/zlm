const std = @import("std");
const contract = @import("contract.zig");

pub const NameEntry = struct { name: []const u8, role: contract.TensorRole };
pub const LayerEntry = struct { suffix: []const u8, role: contract.TensorRole };

/// Generic name mapper (the "mold"). Given a model's tables, it builds
/// `parse`/`format` for names shaped `<layer_prefix><N>.<suffix>` plus a set of
/// global (non-layered) names.
///
///   const Names = model.names.Tables(globals, layered, "blk.");
///   Names.parse("blk.3.attn_q.weight");
pub fn Tables(comptime globals: []const NameEntry, comptime layered: []const LayerEntry, comptime layer_prefix: []const u8) type {
    return struct {
        pub const globals_list = globals;
        pub const layered_list = layered;
        pub const prefix = layer_prefix;

        pub fn parse(name: []const u8) ?contract.TensorRef {
            for (globals) |t| {
                if (std.mem.eql(u8, name, t.name)) return .{ .role = t.role };
            }
            if (!std.mem.startsWith(u8, name, layer_prefix)) return null;
            const rest = name[layer_prefix.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
            const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
            const suffix = rest[dot + 1 ..];
            for (layered) |t| {
                if (std.mem.eql(u8, suffix, t.suffix)) return .{ .role = t.role, .layer = layer };
            }
            return null;
        }

        pub fn format(buf: []u8, role: contract.TensorRole, layer: u32) ?[]const u8 {
            for (layered) |t| {
                if (t.role == role) return std.fmt.bufPrint(buf, "{s}{}.{s}", .{ layer_prefix, layer, t.suffix }) catch null;
            }
            return null;
        }

        pub fn global(role: contract.TensorRole) ?[]const u8 {
            for (globals) |t| {
                if (t.role == role) return t.name;
            }
            return null;
        }
    };
}

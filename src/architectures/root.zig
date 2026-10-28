const std = @import("std");
const Zmeta = @import("../image/zmeta_reader.zig").Zmeta;
const core = @import("core/root.zig");

pub const Arch = core.Arch;
pub const TensorRef = core.TensorRef;
pub const TensorRole = core.TensorRole;
pub const Config = core.Config;

pub fn find(arch: []const u8) ?Arch {
    return core.find(arch);
}

pub fn isSupported(arch: []const u8) bool {
    return core.isSupported(arch);
}

/// Parse a tensor name with the architecture's rules.
pub fn parse(arch: []const u8, name: []const u8) ?TensorRef {
    const a = core.find(arch) orelse return null;
    return a.parseTensorName(name);
}

pub fn opForRole(arch: []const u8, role: TensorRole) ?@import("../kernels/registry.zig").Op {
    const a = core.find(arch) orelse return null;
    return a.opForRole(role);
}

/// Tensor name for a role in a layer (global roles ignore `layer`).
pub fn tensorName(arch: []const u8, role: TensorRole, layer: u32, buf: []u8) ?[]const u8 {
    const a = core.find(arch) orelse return null;
    if (a.globalName(role)) |n| return n;
    return a.layeredName(buf, role, layer);
}

pub fn loadConfig(arch: []const u8, md: *const Zmeta, allocator: std.mem.Allocator) !Config {
    const a = core.find(arch) orelse return error.UnsupportedArchitecture;
    return a.loadConfig(md, allocator);
}

/// Wrap one user turn with the architecture's single-turn prompt template.
pub fn applyChatTemplate(allocator: std.mem.Allocator, arch: []const u8, text: []const u8) ![]u8 {
    const a = core.find(arch) orelse return error.UnsupportedArchitecture;
    if (a.chat.isEmpty()) return error.UnsupportedArchitecture;
    return a.chat.apply(allocator, text);
}

/// The architecture's declared chat `Template` (turn markers + think
/// decorations), or null when the arch is unknown or declares no template. The
/// runtime uses it to read the think fields the compile cannot store.
pub fn chatTemplate(arch: []const u8) ?@import("core/chat.zig").Template {
    const a = core.find(arch) orelse return null;
    if (a.chat.isEmpty()) return null;
    return a.chat;
}

/// The architecture's native tool-call dialect, or null when it has none.
pub fn toolDialect(arch: []const u8) ?@import("core/tools.zig").Dialect {
    const a = core.find(arch) orelse return null;
    return a.tools;
}

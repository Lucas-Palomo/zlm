const std = @import("std");

/// Resolve a `--soul` value: a name that is an existing file is read as the
/// soul text, anything else is the literal text (including an empty string).
/// The system turn itself is emitted by the architecture's chat template.
pub fn resolve(allocator: std.mem.Allocator, io: std.Io, arg: []const u8) []const u8 {
    if (arg.len > 0) {
        if (allocator.alloc(u8, 64 * 1024) catch null) |buf| {
            if (std.Io.Dir.cwd().readFile(io, arg, buf)) |data| return data else |_| {}
        }
    }
    return allocator.dupe(u8, arg) catch "";
}

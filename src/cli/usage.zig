const std = @import("std");
const Io = std.Io;
const io_utils = @import("../utils/io.zig");
const command = @import("command.zig");
const Command = command.Command;

pub const program = "zlm";

pub fn isHelpToken(a: []const u8) bool {
    return std.mem.eql(u8, a, "help") or
        std.mem.eql(u8, a, "--help") or
        std.mem.eql(u8, a, "-h");
}

pub fn printGeneralHelp(io: Io, commands: []const Command) void {
    io_utils.println(io, "{s} - you decide how your language model runs", .{program});
    io_utils.println(io, "      Anti-OOM by design, RAM/VRAM-budgeted, CPU-first, mmap-first, tuned to your hardware.", .{});
    io_utils.println(io, "", .{});
    io_utils.println(io, "Usage: {s} <command> [options]", .{program});
    io_utils.println(io, "", .{});
    io_utils.println(io, "Commands:", .{});
    for (commands) |c| {
        if (c.hidden) continue;
        io_utils.println(io, "  {s}  {s}", .{ c.name, c.summary });
    }
    io_utils.println(io, "", .{});
    io_utils.println(io, "Run '{s} help <command>' or '{s} <command> --help' for details.", .{ program, program });
}

/// `path` is the space-joined command path, e.g. "image" or "image compile".
pub fn printCommandHelp(io: Io, path: []const u8, cmd: *const Command) void {
    io_utils.println(io, "{s} {s} - {s}", .{ program, path, cmd.summary });
    io_utils.println(io, "", .{});
    io_utils.println(io, "Usage:", .{});
    io_utils.println(io, "  {s}", .{cmd.usage});
    if (cmd.help.len > 0) {
        io_utils.println(io, "", .{});
        io_utils.print(io, "{s}", .{cmd.help});
    }
    if (cmd.subcommands.len > 0) {
        io_utils.println(io, "", .{});
        io_utils.println(io, "Subcommands:", .{});
        for (cmd.subcommands) |sc| {
            io_utils.println(io, "  {s}  {s}", .{ sc.name, sc.summary });
        }
    }
}

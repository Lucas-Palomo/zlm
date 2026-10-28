const std = @import("std");
const Io = std.Io;
const Args = @import("args.zig").Args;

pub const Context = struct {
    allocator: std.mem.Allocator,
    io: Io,
    args: Args,
    commands: []const Command,
};

pub const Command = struct {
    name: []const u8,
    summary: []const u8,
    usage: []const u8,
    help: []const u8 = "",
    run: ?*const fn (*Context) anyerror!void = null,
    subcommands: []const Command = &.{},
    /// Internal command: usable but omitted from the general command list.
    hidden: bool = false,
};

pub fn find(commands: []const Command, name: []const u8) ?*const Command {
    for (commands) |*c| {
        if (std.mem.eql(u8, c.name, name)) return c;
    }
    return null;
}

fn isHelpFlag(token: []const u8) bool {
    return std.mem.eql(u8, token, "--help") or std.mem.eql(u8, token, "-h");
}

pub const Walk = struct {
    command: ?*const Command = null,
    unknown: ?[]const u8 = null,
    help: bool = false,
};

/// Consume command tokens from `args`, descending `subcommands`, until a leaf
/// command, a help request, or an unknown token. The resolved command's
/// space-joined path is written into `path_buf` (length in `path_len`).
/// Remaining `args` are the options/positionals for the resolved command.
pub fn walk(level: []const Command, args: *Args, path_buf: []u8, path_len: *usize) Walk {
    var result = Walk{};
    var current = level;
    while (args.peek()) |token| {
        // "help" requests help only when it is not a registered command at this level.
        const is_help = isHelpFlag(token) or
            (std.mem.eql(u8, token, "help") and find(current, "help") == null);
        if (is_help) {
            result.help = true;
            _ = args.next();
            return result;
        }
        // Leaf reached: the remaining tokens belong to the resolved command.
        if (current.len == 0) return result;

        const found = find(current, token) orelse {
            result.unknown = token;
            return result;
        };
        _ = args.next();

        if (path_len.* > 0 and path_len.* < path_buf.len) {
            path_buf[path_len.*] = ' ';
            path_len.* += 1;
        }
        const copy = @min(token.len, path_buf.len - path_len.*);
        @memcpy(path_buf[path_len.*..][0..copy], token[0..copy]);
        path_len.* += copy;

        result.command = found;
        current = found.subcommands;
    }
    return result;
}

test "find returns matching command" {
    const Unused = struct {
        fn run(_: *Context) anyerror!void {}
    };
    const list = [_]Command{
        .{ .name = "a", .summary = "", .usage = "", .run = Unused.run },
    };
    try std.testing.expect(find(&list, "a") != null);
    try std.testing.expect(find(&list, "b") == null);
}

test "walk resolves subcommand path" {
    const Unused = struct {
        fn run(_: *Context) anyerror!void {}
    };
    const subs = [_]Command{
        .{ .name = "compile", .summary = "", .usage = "", .run = Unused.run },
    };
    const top = [_]Command{
        .{ .name = "image", .summary = "", .usage = "", .subcommands = &subs },
    };
    const items = [_][]const u8{ "image", "compile", "name", "m.gguf" };
    var args = Args.init(&items);
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    const w = walk(&top, &args, &buf, &len);
    try std.testing.expect(w.unknown == null and !w.help);
    try std.testing.expectEqualStrings("compile", w.command.?.name);
    try std.testing.expectEqualStrings("image compile", buf[0..len]);
    try std.testing.expectEqualStrings("name", args.peek().?);
}

const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const usage = @import("../usage.zig");
const Context = command.Context;

pub const help_text =
    \\With no argument, lists every command.
    \\With a command path (e.g. 'image compile'), shows that command's help.
    \\
;

pub fn run(ctx: *Context) anyerror!void {
    var path_buf: [128]u8 = undefined;
    var path_len: usize = 0;
    const w = command.walk(ctx.commands, &ctx.args, &path_buf, &path_len);
    const path = path_buf[0..path_len];

    if (w.unknown) |tok| {
        io_utils.eprintln(ctx.io, "Unknown command: {s}", .{tok});
        return error.UnknownCommand;
    }
    if (w.command) |c| {
        usage.printCommandHelp(ctx.io, path, c);
        return;
    }
    usage.printGeneralHelp(ctx.io, ctx.commands);
}

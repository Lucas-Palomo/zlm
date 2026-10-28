const std = @import("std");
const io_utils = @import("../../../utils/io.zig");
const command = @import("../../command.zig");
const options = @import("../../options.zig");
const store = @import("../../../image/store.zig");
const Context = command.Context;

pub const help_text =
    \\Removes an image and its files.
    \\
;

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &.{}) catch |err| {
        io_utils.eprintln(ctx.io, "remove: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const name = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <name>", .{});
        return error.MissingArgument;
    };
    if (!store.isValidName(name)) {
        io_utils.eprintln(ctx.io, "Invalid image name: {s}", .{name});
        return error.InvalidName;
    }
    if (!store.exists(ctx.io, name)) {
        io_utils.eprintln(ctx.io, "No such image: {s}", .{name});
        return error.NotFound;
    }

    try store.remove(ctx.io, name);
    store.removeImageCaches(ctx.io, name);
    io_utils.println(ctx.io, "removed image '{s}'", .{name});
}

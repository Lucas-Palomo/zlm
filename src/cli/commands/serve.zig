const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const store = @import("../../image/store.zig");
const service = @import("../../runtime/service.zig");
const runtime = @import("../../runtime/root.zig");
const Context = command.Context;

pub const help_text =
    \\Internal: runs the per-plan service for an alias.
    \\
    \\Usage: zlm __serve <runtime-alias> [--state=cold|warm|hot] [--hot=MIN] [--timeout=MIN]
    \\
    \\Opens the alias's session, binds the plan's Unix socket and serves the
    \\line-framed PING/GENERATE/QUIT protocol. Spawned detached by load/run.
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--state", .kind = .value },
    .{ .long = "--hot", .kind = .uint },
    .{ .long = "--timeout", .kind = .uint },
};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "__serve: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const alias = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "__serve: missing <runtime-alias>", .{});
        return error.MissingArgument;
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rt = store.resolveRuntime(arena.allocator(), ctx.io, alias) orelse {
        io_utils.eprintln(ctx.io, "__serve: no such runtime: {s}", .{alias});
        return error.NotFound;
    };

    var plan_buf: [std.fs.max_path_bytes]u8 = undefined;
    const plan_path = rt.planPath(&plan_buf) catch return error.PathTooLong;

    var sock_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = store.serviceSocketPath(&sock_buf, rt.key) catch return error.PathTooLong;
    var pid_buf: [std.fs.max_path_bytes]u8 = undefined;
    const pid_path = store.servicePidPath(&pid_buf, rt.key) catch return error.PathTooLong;

    service.detach();

    var session = runtime.Session.open(ctx.allocator, ctx.io, plan_path, .{
        .image_path = rt.image,
    }) catch |err| {
        io_utils.eprintln(ctx.io, "__serve: cannot open session: {s}", .{@errorName(err)});
        return err;
    };
    defer session.deinit();

    const opts = service.Options{
        .state = if (opt.get("--state")) |s| service.stateFromName(s) else .warm,
        .hot_minutes = opt.uint("--hot", u32),
        .timeout_minutes = opt.uint("--timeout", u32),
    };
    return service.serve(ctx.io, &session, sock_path, pid_path, rt.plan, opts);
}

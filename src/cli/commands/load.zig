const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const Args = @import("../args.zig").Args;
const store = @import("../../image/store.zig");
const service = @import("../../runtime/service.zig");
const Context = command.Context;

pub const help_text =
    \\Loads a tuned runtime as a background service.
    \\
    \\Usage: zlm load <runtime-alias> [--cold | --warm | --hot[=minutes]] [--timeout=minutes]
    \\
    \\Spawns a detached per-plan service that holds the session and serves
    \\generation over a Unix socket. Already-online loads are idempotent.
    \\State defaults to warm; --hot=minutes demotes to warm after idle, and
    \\--timeout=minutes auto-unloads the service after idle (default 30 min;
    \\--timeout=0 holds it until 'zlm unload').
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--cold", .kind = .flag },
    .{ .long = "--warm", .kind = .flag },
    .{ .long = "--hot", .kind = .value },
    .{ .long = "--timeout", .kind = .uint },
};

pub fn run(ctx: *Context) anyerror!void {
    const items = try ctx.allocator.dupe([]const u8, ctx.args.items[ctx.args.index..]);
    defer ctx.allocator.free(items);
    for (items) |*it| {
        if (std.mem.eql(u8, it.*, "--hot")) it.* = "--hot=";
    }
    var args = Args.init(items);
    var opt = options.parse(ctx.allocator, &args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "load: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const alias = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <runtime-alias>", .{});
        return error.MissingArgument;
    };
    if (opt.positional(1) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rt = store.resolveRuntime(arena.allocator(), ctx.io, alias) orelse {
        io_utils.eprintln(ctx.io, "No such runtime: {s}; run 'zlm tune <image> {s} ...'", .{ alias, alias });
        return error.NotFound;
    };
    var plan_buf: [std.fs.max_path_bytes]u8 = undefined;
    const plan_path = rt.planPath(&plan_buf) catch return error.PathTooLong;
    if (!store.pathExists(ctx.io, plan_path)) {
        io_utils.eprintln(ctx.io, "Runtime artifact missing: {s}; re-run 'zlm tune <image> {s} ...'", .{ plan_path, alias });
        return error.NotFound;
    }

    const state_name: []const u8 = if (opt.has("--cold")) "cold" else if (opt.get("--hot") != null) "hot" else "warm";
    const opts = service.Options{
        .state = service.stateFromName(state_name),
        .hot_minutes = opt.uint("--hot", u32),
        .timeout_minutes = opt.uint("--timeout", u32),
    };

    var sock_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = store.serviceSocketPath(&sock_buf, rt.key) catch return error.PathTooLong;
    var log_buf: [std.fs.max_path_bytes]u8 = undefined;
    const log_path = store.serviceLogPath(&log_buf, rt.key) catch return error.PathTooLong;

    const self_fp = service.build_fingerprint;
    const dfp = service.pingFingerprint(ctx.io, sock_path);
    if (dfp != null and (self_fp == 0 or dfp.? == self_fp)) {
        io_utils.println(ctx.io, "loaded '{s}' (already online; state unchanged)", .{alias});
        return;
    }
    // A stale daemon (different build) is stopped so a fresh one loads.
    if (dfp != null) service.stop(ctx.io, rt.key);

    const cold_bytes = service.residentUploadBytes(ctx.allocator, ctx.io, plan_path);
    service.ensureOnline(ctx.allocator, ctx.io, alias, sock_path, log_path, opts, cold_bytes) catch |err| {
        io_utils.eprintln(ctx.io, "load: service failed for '{s}': {s}", .{ alias, @errorName(err) });
        return err;
    };
    io_utils.println(ctx.io, "loaded '{s}' ({s})", .{ alias, state_name });
}

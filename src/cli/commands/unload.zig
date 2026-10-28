const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const store = @import("../../image/store.zig");
const zrun = @import("../../image/zrun.zig");
const service = @import("../../runtime/service.zig");
const Context = command.Context;

pub const summary = "Unload a runtime or prune unused artifacts";
pub const usage = "zlm unload [<runtime-alias>] [--remove] [--prune] [--all]";
pub const help_text =
    \\Unloads a tuned runtime. A bare alias keeps the service online;
    \\--remove/--prune mutate the store and stop services.
    \\
    \\  zlm unload <runtime-alias>          report (service kept)
    \\  zlm unload <runtime-alias> --remove drop the alias; stop the service
    \\                                      when it was the last one for the plan
    \\  zlm unload --prune                  delete orphan .zplan/.zat and stop
    \\                                      orphan services
    \\  zlm unload --all [--remove] [--prune]
    \\
    \\--remove needs <runtime-alias> or --all; --prune is store-wide and
    \\never takes an <runtime-alias>.
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--remove", .kind = .flag },
    .{ .long = "--prune", .kind = .flag },
    .{ .long = "--all", .kind = .flag },
};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "unload: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const alias = opt.positional(0);
    if (opt.positional(1) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }
    const remove = opt.has("--remove");
    const prune = opt.has("--prune");
    const all = opt.has("--all");

    if (remove and alias == null and !all) {
        io_utils.eprintln(ctx.io, "--remove needs <alias> or --all", .{});
        return error.MissingArgument;
    }
    if (prune and alias != null) {
        io_utils.eprintln(ctx.io, "--prune is store-wide; not with <alias>", .{});
        return error.InvalidArgument;
    }
    if (all and alias != null) {
        io_utils.eprintln(ctx.io, "--all takes no <alias>", .{});
        return error.InvalidArgument;
    }

    if (alias) |a| {
        if (!store.aliasExists(ctx.io, a)) {
            io_utils.eprintln(ctx.io, "No such runtime: {s}", .{a});
            return error.NotFound;
        }
        if (remove) {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const rt = store.resolveRuntime(arena.allocator(), ctx.io, a) orelse {
                io_utils.eprintln(ctx.io, "No such runtime: {s}", .{a});
                return error.NotFound;
            };
            _ = try store.removeAlias(ctx.allocator, ctx.io, a);
            if (!store.planHasAliases(arena.allocator(), ctx.io, rt.plan)) {
                service.stop(ctx.io, rt.key);
            }
            io_utils.println(ctx.io, "removed '{s}'", .{a});
            return;
        }
        io_utils.println(ctx.io, "unloaded '{s}' (service kept)", .{a});
        return;
    }

    if (!all and !prune) {
        io_utils.eprintln(ctx.io, "Missing <runtime-alias> or --all/--prune", .{});
        return error.MissingArgument;
    }
    return storeWide(ctx, remove, prune);
}

fn storeWide(ctx: *Context, remove: bool, prune: bool) !void {
    if (remove) {
        stopAllServices(ctx);
        try store.writeRegistry(ctx.allocator, ctx.io, &[_]zrun.Entry{});
        io_utils.println(ctx.io, "removed all runtimes", .{});
    }
    if (prune) {
        const reaped = try service.reapOrphanServices(ctx.allocator, ctx.io);
        const removed = try store.pruneOrphans(ctx.allocator, ctx.io);
        io_utils.println(ctx.io, "reaped {d} orphan service(s), pruned {d} orphan file(s)", .{ reaped, removed });
    }
    if (!remove and !prune) {
        io_utils.println(ctx.io, "unloaded all runtimes", .{});
    }
}

fn stopAllServices(ctx: *Context) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var reg = store.loadRegistry(arena.allocator(), ctx.io) orelse return;
    defer reg.deinit();

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(arena.allocator());
    for (reg.entries) |e| {
        const key = store.planKeyFromPlanName(e.plan);
        if (seen.contains(key)) continue;
        seen.put(arena.allocator(), key, {}) catch return;
        service.stop(ctx.io, key);
    }
}

fn testCtx(items: []const []const u8) Context {
    return .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .args = @import("../args.zig").Args.init(items),
        .commands = &.{},
    };
}

test "unload rejects --remove without alias or --all" {
    const testing = std.testing;
    const args = [_][]const u8{"--remove"};
    var ctx = testCtx(&args);
    try testing.expectError(error.MissingArgument, run(&ctx));
}

test "unload rejects --prune with an alias" {
    const testing = std.testing;
    const args = [_][]const u8{ "rt-a", "--prune" };
    var ctx = testCtx(&args);
    try testing.expectError(error.InvalidArgument, run(&ctx));
}

test "unload rejects --all with an alias" {
    const testing = std.testing;
    const args = [_][]const u8{ "rt-a", "--all" };
    var ctx = testCtx(&args);
    try testing.expectError(error.InvalidArgument, run(&ctx));
}

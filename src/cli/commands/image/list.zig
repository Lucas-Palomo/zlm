const std = @import("std");
const Io = std.Io;
const io_utils = @import("../../../utils/io.zig");
const command = @import("../../command.zig");
const store = @import("../../../image/store.zig");
const zman = @import("../../../image/zman.zig");
const zhash = @import("../../../image/zhash.zig");
const zrun = @import("../../../image/zrun.zig");
const service = @import("../../../runtime/service.zig");
const Context = command.Context;

pub const help_text =
    \\Lists images in the store.
    \\
    \\Columns: NAME, ARCH, IMAGE, SERVICE.
    \\  IMAGE   building (in progress) | ready (valid zhash) | broken
    \\  SERVICE online | offline - a live service for a plan of this image
    \\
;

pub fn run(ctx: *Context) anyerror!void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const reg: ?zrun.Registry = store.loadRegistry(arena.allocator(), ctx.io);

    var dir_buf: [512]u8 = undefined;
    const images = try store.imagesDir(&dir_buf);
    var dir = Io.Dir.cwd().openDir(ctx.io, images, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => {
            io_utils.println(ctx.io, "no images", .{});
            return;
        },
        else => return err,
    };
    defer dir.close(ctx.io);

    io_utils.println(ctx.io, "NAME\tARCH\tIMAGE\tSERVICE", .{});
    var it = dir.iterate();
    while (it.next(ctx.io) catch null) |entry| {
        if (entry.kind != .directory) continue;

        if (std.mem.startsWith(u8, entry.name, ".tmp-")) {
            io_utils.println(ctx.io, "{s}\t-\tbuilding\toffline", .{entry.name[".tmp-".len..]});
            continue;
        }

        const service_state: []const u8 = if (imageOnline(ctx, reg, entry.name)) "online" else "offline";

        var img = store.openImage(ctx.io, entry.name) catch continue;
        defer img.close(ctx.io);

        const valid = blk: {
            var r = zhash.Reader.open(ctx.allocator, img, ctx.io) catch break :blk false;
            r.deinit();
            break :blk true;
        };
        const image_state: []const u8 = if (valid) "ready" else "broken";

        var manifest_opt = zman.Manifest.open(ctx.allocator, img, ctx.io) catch null;
        if (manifest_opt) |*m| {
            io_utils.println(ctx.io, "{s}\t{s}\t{s}\t{s}", .{ entry.name, m.arch, image_state, service_state });
            m.deinit();
        } else {
            io_utils.println(ctx.io, "{s}\t-\t{s}\t{s}", .{ entry.name, image_state, service_state });
        }
    }
}

fn imageOnline(ctx: *Context, reg: ?zrun.Registry, name: []const u8) bool {
    const r = reg orelse return false;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    for (r.entries) |e| {
        if (!std.mem.eql(u8, e.image, name)) continue;
        const key = store.planKeyFromPlanName(e.plan);
        const sock_path = store.serviceSocketPath(&buf, key) catch continue;
        if (service.ping(ctx.io, sock_path)) return true;
    }
    return false;
}

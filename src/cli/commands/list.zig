const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const store = @import("../../image/store.zig");
const Context = command.Context;

pub const summary = "List tuned runtimes";
pub const usage = "zlm list";
pub const help_text =
    \\Lists the tuned runtimes (aliases) in the store.
    \\
    \\Columns: ALIAS, IMAGE, FP (the shared tuned (image, fingerprint)).
    \\
;

pub fn run(ctx: *Context) anyerror!void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const reg = store.loadRegistry(arena.allocator(), ctx.io) orelse {
        io_utils.println(ctx.io, "no runtimes", .{});
        return;
    };
    if (reg.entries.len == 0) {
        io_utils.println(ctx.io, "no runtimes", .{});
        return;
    }

    io_utils.println(ctx.io, "ALIAS\tIMAGE\tFP", .{});
    for (reg.entries) |*e| {
        const fp_hex = std.fmt.bytesToHex(std.mem.asBytes(&e.fp), .lower);
        io_utils.println(ctx.io, "{s}\t{s}\t{s}", .{ e.id, e.image, &fp_hex });
    }
}

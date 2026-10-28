const std = @import("std");
const Io = std.Io;
const io_utils = @import("../../../utils/io.zig");
const command = @import("../../command.zig");
const options = @import("../../options.zig");
const store = @import("../../../image/store.zig");
const zman = @import("../../../image/zman.zig");
const zhash = @import("../../../image/zhash.zig");
const zmeta = @import("../../../image/zmeta_reader.zig");
const Context = command.Context;

pub const help_text =
    \\Shows details about an image (manifest, files, tokenizer).
    \\
    \\Usage: zlm image info <name>
    \\
;

const specs = [_]options.Spec{};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "image info: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const name = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <name>", .{});
        return error.MissingArgument;
    };
    if (!store.isValidName(name)) {
        io_utils.eprintln(ctx.io, "Invalid image name: {s}", .{name});
        return error.InvalidValue;
    }
    if (!store.exists(ctx.io, name)) {
        io_utils.eprintln(ctx.io, "No such image: {s}", .{name});
        return error.NotFound;
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try store.imageDir(&path_buf, name);
    var dir = try store.openImage(ctx.io, name);
    defer dir.close(ctx.io);

    var manifest: ?zman.Manifest = zman.Manifest.open(ctx.allocator, dir, ctx.io) catch null;
    defer {
        if (manifest) |*m| m.deinit();
    }

    io_utils.println(ctx.io, "image:   {s}", .{name});
    io_utils.println(ctx.io, "path:    {s}", .{path});
    if (manifest) |m| {
        io_utils.println(ctx.io, "arch:    {s}", .{m.arch});
        io_utils.println(ctx.io, "gguf:    {d} bytes", .{m.gguf_size});
        io_utils.println(ctx.io, "tensors: {d}", .{m.tensor_count});
        io_utils.println(ctx.io, "metadata: {d} keys", .{m.metadata_kv_count});
    } else {
        io_utils.println(ctx.io, "arch:    -", .{});
    }

    const zhash_ready = blk: {
        var r = zhash.Reader.open(ctx.allocator, dir, ctx.io) catch break :blk false;
        r.deinit();
        break :blk true;
    };
    io_utils.println(ctx.io, "zhash:   {s}", .{if (zhash_ready) "ready" else "broken"});

    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| ctx.allocator.free(n);
        names.deinit(ctx.allocator);
    }
    var it = dir.iterate();
    while (it.next(ctx.io) catch null) |entry| {
        if (entry.kind != .file) continue;
        try names.append(ctx.allocator, try ctx.allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    io_utils.println(ctx.io, "files:", .{});
    for (names.items) |file_name| {
        const stat = dir.statFile(ctx.io, file_name, .{}) catch continue;
        io_utils.println(ctx.io, "  {s:<16} {d} bytes", .{ file_name, stat.size });
    }

    var zm: ?zmeta.Zmeta = zmeta.Zmeta.open(ctx.allocator, dir, ctx.io) catch null;
    if (zm) |*z| {
        defer z.deinit();
        const model = z.getString("tokenizer.ggml.model") orelse "?";
        const tokens = if (z.getArray("tokenizer.ggml.tokens")) |list| list.items.len else 0;
        io_utils.println(ctx.io, "tokenizer: {s}, {d} tokens", .{ model, tokens });
    } else {
        io_utils.println(ctx.io, "tokenizer: (no zmeta)", .{});
    }
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

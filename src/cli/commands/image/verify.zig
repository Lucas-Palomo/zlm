const std = @import("std");
const Io = std.Io;
const io_utils = @import("../../../utils/io.zig");
const command = @import("../../command.zig");
const options = @import("../../options.zig");
const gguf = @import("../../../gguf/root.zig");
const MappedFile = @import("../../../utils/mmap.zig").MappedFile;
const store = @import("../../../image/store.zig");
const ztef = @import("../../../image/ztef.zig");
const zman = @import("../../../image/zman.zig");
const zhash = @import("../../../image/zhash.zig");
const Context = command.Context;

pub const help_text =
    \\Verifies the integrity of an image using its zhash.
    \\
    \\Validates the zhash self-checksum, then every listed file's sha256.
    \\With a GGUF argument, also checks provenance (the recorded content root).
    \\
;

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &.{}) catch |err| {
        io_utils.eprintln(ctx.io, "verify: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const name = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <name>", .{});
        return error.MissingArgument;
    };
    const gguf_path: ?[]const u8 = opt.positional(1);
    if (!store.isValidName(name)) {
        io_utils.eprintln(ctx.io, "Invalid image name: {s}", .{name});
        return error.InvalidValue;
    }
    if (!store.exists(ctx.io, name)) {
        io_utils.eprintln(ctx.io, "No such image: {s}", .{name});
        return error.NotFound;
    }

    var dir = try store.openImage(ctx.io, name);
    defer dir.close(ctx.io);

    var hashes = zhash.Reader.open(ctx.allocator, dir, ctx.io) catch |err| {
        io_utils.eprintln(ctx.io, "FAIL: zhash: {s}", .{@errorName(err)});
        return err;
    };
    defer hashes.deinit();

    var dir_buf: [512]u8 = undefined;
    const image_path = try store.imageDir(&dir_buf, name);

    var checked: usize = 0;
    for (hashes.entries) |e| {
        var file_buf: [512]u8 = undefined;
        const file_path = try std.fmt.bufPrint(&file_buf, "{s}/{s}", .{ image_path, e.kind.filename() });

        var mmap = MappedFile.open(ctx.io, file_path) catch |err| {
            io_utils.eprintln(ctx.io, "FAIL: cannot open {s}: {s}", .{ e.kind.filename(), @errorName(err) });
            return err;
        };
        defer mmap.close();

        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(mmap.data, &sha, .{});
        if (!std.mem.eql(u8, &sha, &e.sha256)) {
            io_utils.eprintln(ctx.io, "FAIL: sha mismatch: {s}", .{e.kind.filename()});
            return error.ShaMismatch;
        }
        checked += 1;
    }

    if (gguf_path) |path| {
        var src = gguf.Gguf.open(ctx.allocator, ctx.io, path) catch |err| {
            io_utils.eprintln(ctx.io, "FAIL: cannot read {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        defer src.deinit();

        var manifest = zman.Manifest.open(ctx.allocator, dir, ctx.io) catch |err| {
            io_utils.eprintln(ctx.io, "FAIL: zman: {s}", .{@errorName(err)});
            return err;
        };
        defer manifest.deinit();

        var all: std.ArrayList(ztef.TensorSha) = .empty;
        defer all.deinit(ctx.allocator);

        for (manifest.files) |f| {
            var zbuf: [512]u8 = undefined;
            const zpath = try std.fmt.bufPrint(&zbuf, "{s}/{s}", .{ image_path, f.kind.filename() });
            var r = ztef.Reader.open(ctx.allocator, ctx.io, zpath) catch |err| {
                io_utils.eprintln(ctx.io, "FAIL: cannot open {s}: {s}", .{ f.kind.filename(), @errorName(err) });
                return err;
            };
            defer r.deinit();
            for (r.entries) |*e| {
                const t = src.findTensor(e.name) orelse {
                    io_utils.eprintln(ctx.io, "FAIL: tensor not in source: {s}", .{e.name});
                    return error.MissingTensor;
                };
                const bytes = src.tensorBytes(t) catch |err| {
                    io_utils.eprintln(ctx.io, "FAIL: cannot read source tensor {s}: {s}", .{ e.name, @errorName(err) });
                    return err;
                };
                var ts: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(bytes, &ts, .{});
                // The entry carries the compile-time sha; it must match the
                // source tensor (a precise per-tensor provenance check).
                if (!std.mem.eql(u8, &ts, &e.sha256)) {
                    io_utils.eprintln(ctx.io, "FAIL: tensor sha mismatch: {s}", .{e.name});
                    return error.TensorShaMismatch;
                }
                const owned = try ctx.allocator.dupe(u8, e.name);
                try all.append(ctx.allocator, .{ .name = owned, .sha256 = ts });
            }
        }

        std.mem.sort(ztef.TensorSha, all.items, {}, ztef.shaLessByName);
        var meta_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(src.mmap.data[0..@intCast(src.data_offset)], &meta_hash, .{});
        var root_h = std.crypto.hash.sha2.Sha256.init(.{});
        root_h.update(&meta_hash);
        for (all.items) |ts| {
            root_h.update(ts.name);
            root_h.update(&ts.sha256);
        }
        var root: [32]u8 = undefined;
        root_h.final(&root);
        if (!std.mem.eql(u8, &root, &manifest.content_root)) {
            io_utils.eprintln(ctx.io, "FAIL: provenance (content root mismatch)", .{});
            return error.ContentMismatch;
        }
    }

    io_utils.println(ctx.io, "ok: {s} ({d} files verified)", .{ name, checked });
}

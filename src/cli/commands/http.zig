const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const config_mod = @import("../../http/config.zig");
const http_server = @import("../../http/server.zig");
const Context = command.Context;

pub const summary = "Serve runtimes over an OpenAI-compatible HTTP API";
pub const usage = "zlm http <serve|config>";
pub const help_text =
    \\zlm http serve <port> --config=<file>   run the HTTP API from a config
    \\zlm http config [--output <file>]        write a bootstrap config
    \\
    \\The config declares the server host and one entry per model:
    \\id (the OpenAI `model`), runtime (the zlm alias), soul, temperature,
    \\top_k, top_p, think, max_tokens, max_think. Models load lazily.
    \\
;

const serve_specs = [_]options.Spec{
    .{ .long = "--config", .kind = .value, .required = true },
};

const config_specs = [_]options.Spec{
    .{ .long = "--output", .kind = .value },
};

const serve_help =
    \\Usage: zlm http serve <port> --config=<file>
    \\
    \\Serves the configured models over GET /v1/models and
    \\POST /v1/chat/completions (JSON or SSE). A request `model` selects the
    \\entry; `system` overrides the entry's soul.
    \\
;

const config_help =
    \\Usage: zlm http config [--output <file>]
    \\
    \\Writes a bootstrap config with one entry per tuned runtime alias (defaults
    \\for the generation controls). Without --output it prints to stdout.
    \\
;

fn serveRun(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &serve_specs) catch |err| {
        io_utils.eprintln(ctx.io, "http serve: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const port_s = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <port>", .{});
        return error.MissingArgument;
    };
    const port = std.fmt.parseInt(u16, port_s, 10) catch {
        io_utils.eprintln(ctx.io, "Invalid port: {s}", .{port_s});
        return error.InvalidValue;
    };
    const cfg_path = opt.get("--config") orelse {
        io_utils.eprintln(ctx.io, "Missing --config <file>", .{});
        return error.MissingArgument;
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const cfg = config_mod.parse(arena.allocator(), ctx.io, cfg_path) catch |err| {
        io_utils.eprintln(ctx.io, "http serve: cannot read config: {s}", .{@errorName(err)});
        return err;
    };
    if (cfg.models.len == 0) {
        io_utils.eprintln(ctx.io, "http serve: the config declares no models", .{});
        return error.InvalidValue;
    }
    io_utils.println(ctx.io, "http: {d} model(s) on http://{s}:{d}/v1 (Ctrl-C to stop)", .{ cfg.models.len, cfg.host, port });
    return http_server.serve(ctx.io, ctx.allocator, cfg, port);
}

fn configRun(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &config_specs) catch |err| {
        io_utils.eprintln(ctx.io, "http config: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const cfg = config_mod.bootstrap(arena.allocator(), ctx.io) catch |err| {
        io_utils.eprintln(ctx.io, "http config: {s}", .{@errorName(err)});
        return err;
    };
    const json = config_mod.write(arena.allocator(), cfg) catch |err| {
        io_utils.eprintln(ctx.io, "http config: {s}", .{@errorName(err)});
        return err;
    };
    if (opt.get("--output")) |path| {
        var f = std.Io.Dir.cwd().createFile(ctx.io, path, .{}) catch |err| {
            io_utils.eprintln(ctx.io, "http config: cannot write {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        defer f.close(ctx.io);
        f.writeStreamingAll(ctx.io, json) catch |err| {
            io_utils.eprintln(ctx.io, "http config: write: {s}", .{@errorName(err)});
            return err;
        };
        io_utils.println(ctx.io, "wrote {s} ({d} models)", .{ path, cfg.models.len });
    } else {
        io_utils.writeAll(ctx.io, io_utils.stdout(), json);
    }
}

pub const subcommands = [_]command.Command{
    .{
        .name = "serve",
        .summary = "Run the OpenAI-compatible HTTP API",
        .usage = "zlm http serve <port> --config=<file>",
        .help = serve_help,
        .run = serveRun,
    },
    .{
        .name = "config",
        .summary = "Write a bootstrap server config",
        .usage = "zlm http config [--output <file>]",
        .help = config_help,
        .run = configRun,
    },
};

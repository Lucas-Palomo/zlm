const std = @import("std");
const Io = std.Io;
const io_utils = @import("../utils/io.zig");
const Args = @import("args.zig").Args;
const command = @import("command.zig");
const usage = @import("usage.zig");
const help_cmd = @import("commands/help.zig");
const image_cmd = @import("commands/image.zig");
const list_cmd = @import("commands/list.zig");
const inspect_cmd = @import("commands/inspect.zig");
const unload_cmd = @import("commands/unload.zig");
const tune_cmd = @import("commands/tune.zig");
const load_cmd = @import("commands/load.zig");
const run_cmd = @import("commands/run.zig");
const chat_cmd = @import("commands/chat.zig");
const http_cmd = @import("commands/http.zig");
const serve_cmd = @import("commands/serve.zig");

pub const Command = command.Command;
pub const Context = command.Context;

pub const registry: []const Command = &.{
    .{
        .name = "help",
        .summary = "Show help for zlm or a command",
        .usage = "zlm help [command]",
        .help = help_cmd.help_text,
        .run = help_cmd.run,
    },
    .{
        .name = "image",
        .summary = image_cmd.summary,
        .usage = image_cmd.usage,
        .help = image_cmd.help_text,
        .subcommands = &image_cmd.subcommands,
    },
    .{
        .name = "list",
        .summary = list_cmd.summary,
        .usage = list_cmd.usage,
        .help = list_cmd.help_text,
        .run = list_cmd.run,
    },
    .{
        .name = "inspect",
        .summary = inspect_cmd.summary,
        .usage = inspect_cmd.usage,
        .help = inspect_cmd.help_text,
        .run = inspect_cmd.run,
    },
    .{
        .name = "unload",
        .summary = unload_cmd.summary,
        .usage = unload_cmd.usage,
        .help = unload_cmd.help_text,
        .run = unload_cmd.run,
    },
    .{
        .name = "tune",
        .summary = "Tune an image for this machine into a runtime plan",
        .usage = "zlm tune <image> <runtime-alias> --threads N --ram <mb|gb> [--residency <mb|gb> --prefetch-threads M] [--gpu --vram <mb|gb>]",
        .help = tune_cmd.help_text,
        .run = tune_cmd.run,
    },
    .{
        .name = "load",
        .summary = "Load a tuned runtime",
        .usage = "zlm load <runtime-alias>",
        .help = load_cmd.help_text,
        .run = load_cmd.run,
    },
    .{
        .name = "run",
        .summary = "Run a tuned runtime",
        .usage = "zlm run <runtime-alias> --prompt <text> --soul <text|file.txt> [--session <uuid>] [--max-tokens N] [--max-think N] [--temperature F] [--top-k N] [--top-p F] [--metrics[=full]] [--verbose]",
        .help = run_cmd.help_text,
        .run = run_cmd.run,
    },
    .{
        .name = "chat",
        .summary = "Start an interactive chat session",
        .usage = "zlm chat <runtime-alias> [<uuid>] --soul <text|file.txt> [--max-tokens N] [--max-think N] [--temperature F] [--top-k N] [--top-p F]",
        .help = chat_cmd.help_text,
        .run = chat_cmd.run,
    },
    .{
        .name = "http",
        .summary = http_cmd.summary,
        .usage = http_cmd.usage,
        .help = http_cmd.help_text,
        .subcommands = &http_cmd.subcommands,
    },
    .{
        .name = "__serve",
        .summary = "Internal: run the per-plan service",
        .usage = "zlm __serve <runtime-alias>",
        .help = serve_cmd.help_text,
        .run = serve_cmd.run,
        .hidden = true,
    },
};

pub fn run(allocator: std.mem.Allocator, io: Io, argv: []const []const u8) u8 {
    var args = Args.init(argv);
    var path_buf: [128]u8 = undefined;
    var path_len: usize = 0;
    const w = command.walk(registry, &args, &path_buf, &path_len);
    const path = path_buf[0..path_len];

    if (w.unknown) |tok| {
        io_utils.eprintln(io, "Unknown {s}: {s}", .{ if (w.command == null) "command" else "subcommand", tok });
        if (w.command) |c| usage.printCommandHelp(io, path, c) else usage.printGeneralHelp(io, registry);
        return 1;
    }

    if (w.help) {
        if (w.command) |c| usage.printCommandHelp(io, path, c) else usage.printGeneralHelp(io, registry);
        return 0;
    }

    const c = w.command orelse {
        usage.printGeneralHelp(io, registry);
        return 1;
    };

    const run_fn = c.run orelse {
        io_utils.eprintln(io, "Missing subcommand for '{s} {s}'", .{ usage.program, path });
        usage.printCommandHelp(io, path, c);
        return 1;
    };

    var ctx = Context{
        .allocator = allocator,
        .io = io,
        .args = args,
        .commands = registry,
    };
    run_fn(&ctx) catch {
        return 1;
    };
    return 0;
}

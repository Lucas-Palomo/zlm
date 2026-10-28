const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const store = @import("../../image/store.zig");
const service = @import("../../runtime/service.zig");
const soul_mod = @import("soul.zig");
const repl = @import("chat/repl.zig");
const sessions = @import("../../runtime/sessions.zig");
const Context = command.Context;

pub const help_text =
    \\Starts an interactive chat session against a tuned runtime. Each line is a
    \\user turn; the reply streams to stdout. Slash commands (not sent to the
    \\model): /help, /exit, /clear, /soul, /enable|/disable think|show-think,
    \\/verbose, /metrics, /temperature, /top-k, /top-p, /max-tokens, /max-think,
    \\/rename, /sessions.
    \\
    \\Usage: zlm chat <runtime-alias> [<uuid>] --soul <text|file.txt>
    \\       [--max-tokens N] [--max-think N]
    \\       [--temperature F] [--top-k N] [--top-p F]
    \\       [--think] [--show-think] [--verbose[=min|info|all]]
    \\       [--cold | --warm | --hot[=minutes]] [--timeout=minutes]
    \\
    \\--soul is required: the system turn that sets the model's persona (a
    \\literal string, "" for none, or a path to an existing file).
    \\The conversation is persisted under $ZLM_HOME/sessions/<uuid>.json; an
    \\explicit <uuid> resumes it, none starts a new one. /sessions lists them.
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--soul", .kind = .value, .required = true },
    .{ .long = "--max-tokens", .kind = .uint },
    .{ .long = "--max-think", .kind = .uint },
    .{ .long = "--temperature", .kind = .float },
    .{ .long = "--top-k", .kind = .uint },
    .{ .long = "--top-p", .kind = .float },
    .{ .long = "--think", .kind = .flag },
    .{ .long = "--show-think", .kind = .flag },
    .{ .long = "--verbose", .kind = .value },
    .{ .long = "--metrics", .kind = .value },
    .{ .long = "--cold", .kind = .flag },
    .{ .long = "--warm", .kind = .flag },
    .{ .long = "--hot", .kind = .value },
    .{ .long = "--timeout", .kind = .uint },
};

fn verboseLevel(v: ?[]const u8) u8 {
    const s = v orelse return 0;
    if (std.mem.eql(u8, s, "info")) return 2;
    if (std.mem.eql(u8, s, "all")) return 3;
    return 1;
}

fn metricsLevel(v: ?[]const u8) u8 {
    const s = v orelse return 0;
    if (std.mem.eql(u8, s, "off")) return 0;
    if (std.mem.eql(u8, s, "all")) return 2;
    return 1;
}

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "chat: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const alias = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <runtime-alias>", .{});
        return error.MissingArgument;
    };
    if (opt.positional(2) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }
    // An explicit uuid resumes that session; without one, a new session starts
    // (and is persisted, discoverable via /sessions).
    var id_buf: [16]u8 = undefined;
    const session_id = opt.positional(1) orelse sessions.newId(ctx.io, &id_buf);
    const soul_arg = opt.get("--soul") orelse {
        io_utils.eprintln(ctx.io, "Missing --soul <text|file.txt>", .{});
        return error.MissingArgument;
    };

    var cfg = repl.Config{
        .max_tokens = opt.uint("--max-tokens", usize) orelse 256,
        .max_think = opt.uint("--max-think", usize) orelse 256,
        .temperature = opt.float("--temperature", f32) orelse 0,
        .top_k = opt.uint("--top-k", u32) orelse 0,
        .top_p = opt.float("--top-p", f32) orelse 1,
        .think = opt.has("--think"),
        .show_think = opt.has("--show-think"),
        .verbose = verboseLevel(opt.get("--verbose")),
        .metrics = metricsLevel(opt.get("--metrics")),
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    cfg.soul = soul_mod.resolve(arena.allocator(), ctx.io, soul_arg);

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
    var sock_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = store.serviceSocketPath(&sock_buf, rt.key) catch return error.PathTooLong;

    // Bring the service up (or reuse it), like `run`.
    const self_fp = service.build_fingerprint;
    const dfp = service.pingFingerprint(ctx.io, sock_path);
    const stale = dfp != null and self_fp != 0 and dfp.? != self_fp;
    if (dfp == null or stale) {
        if (stale) {
            service.stop(ctx.io, rt.key);
            io_utils.println(ctx.io, "restarting '{s}' (stale daemon)...", .{alias});
        } else {
            io_utils.println(ctx.io, "loading '{s}'...", .{alias});
        }
        var log_buf: [std.fs.max_path_bytes]u8 = undefined;
        const log_path = store.serviceLogPath(&log_buf, rt.key) catch return error.PathTooLong;
        const state_name: []const u8 = if (opt.has("--cold")) "cold" else if (opt.get("--hot") != null) "hot" else "warm";
        const opts = service.Options{
            .state = service.stateFromName(state_name),
            .hot_minutes = opt.uint("--hot", u32),
            .timeout_minutes = opt.uint("--timeout", u32),
        };
        const cold_bytes = service.residentUploadBytes(ctx.allocator, ctx.io, plan_path);
        service.ensureOnline(ctx.allocator, ctx.io, alias, sock_path, log_path, opts, cold_bytes) catch |err| {
            io_utils.eprintln(ctx.io, "chat: cannot load '{s}': {s}", .{ alias, @errorName(err) });
            return err;
        };
    }

    var client = service.ChatClient.connect(ctx.io, sock_path) catch |err| {
        io_utils.eprintln(ctx.io, "chat: connect: {s}", .{@errorName(err)});
        return err;
    };
    defer client.close();

    try client.session(ctx.allocator, session_id, "", alias);
    cfg.session_id = session_id;
    cfg.model = alias;
    try repl.run(ctx, &client, &cfg);
}

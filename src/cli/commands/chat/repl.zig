const std = @import("std");
const io_utils = @import("../../../utils/io.zig");
const service = @import("../../../runtime/service.zig");
const session_mod = @import("../../../runtime/session.zig");
const command = @import("../../command.zig");
const soul_mod = @import("../soul.zig");
const sessions = @import("../../../runtime/sessions.zig");

/// The interactive session state the slash commands mutate.
pub const Config = struct {
    max_tokens: usize = 256,
    max_think: usize = 256,
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    think: bool = false,
    show_think: bool = false,
    /// 0 off / 1 min / 2 info / 3 all (the `run --verbose` levels).
    verbose: u8 = 0,
    /// 0 off / 1 min / 2 all (the `run --metrics` levels).
    metrics: u8 = 0,
    soul: []const u8 = "",
    /// The persistent session this REPL is bound to.
    session_id: []const u8 = "",
    session_name: []const u8 = "",
    model: []const u8 = "",
};

const StdoutSink = struct {
    io: std.Io,

    fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *StdoutSink = @ptrCast(@alignCast(ctx));
        io_utils.writeAll(self.io, io_utils.stdout(), bytes);
    }
};

const help_text =
    \\/help                 show this help
    \\/exit                 leave the chat (also Ctrl-D)
    \\/clear                forget the conversation
    \\/soul <text|file.txt> replace the soul for later turns
    \\/enable <think|show-think>
    \\/disable <think|show-think>
    \\/verbose <off|min|info|all>
    \\/metrics <off|min|all>
    \\/temperature <F>      sampling temperature (0 = greedy)
    \\/top-k <N>            top-k (0 = off)
    \\/top-p <F>            nucleus
    \\/max-tokens <N>       answer budget
    \\/max-think <N>        reasoning budget
    \\/rename <name>        rename the current session
    \\/sessions             list sessions and switch to one
    \\
;

pub fn printHelp(io: std.Io) void {
    io_utils.println(io, "{s}", .{help_text});
}

fn parseLevel(name: []const u8, levels: []const []const u8) ?u8 {
    for (levels, 0..) |level, i| {
        if (std.mem.eql(u8, name, level)) return @intCast(i);
    }
    return null;
}

fn usage(io: std.Io, text: []const u8) bool {
    io_utils.eprintln(io, "usage: {s}", .{text});
    return false;
}

fn bad(io: std.Io, what: []const u8, value: []const u8) bool {
    io_utils.eprintln(io, "invalid {s}: {s}", .{ what, value });
    return false;
}

/// Handle a `/command`. Returns true when the REPL must exit.
fn slash(ctx: *const command.Context, client: *service.ChatClient, cfg: *Config, arena: std.mem.Allocator, text: []const u8) bool {
    const io = ctx.io;
    var it = std.mem.tokenizeAny(u8, text, " \t");
    const name = it.next() orelse return false;
    if (std.mem.eql(u8, name, "/exit")) return true;
    if (std.mem.eql(u8, name, "/help")) {
        printHelp(io);
        return false;
    }
    if (std.mem.eql(u8, name, "/clear")) {
        client.reset() catch |err| io_utils.eprintln(io, "clear: {s}", .{@errorName(err)});
        io_utils.println(io, "conversation cleared", .{});
        return false;
    }
    if (std.mem.eql(u8, name, "/enable") or std.mem.eql(u8, name, "/disable")) {
        const on = std.mem.eql(u8, name, "/enable");
        const feature = it.next() orelse {
            io_utils.eprintln(io, "usage: {s} <think|show-think>", .{name});
            return false;
        };
        if (std.mem.eql(u8, feature, "think")) {
            cfg.think = on;
            io_utils.println(io, "think {s}", .{if (on) "on" else "off"});
        } else if (std.mem.eql(u8, feature, "show-think")) {
            cfg.show_think = on;
            io_utils.println(io, "show-think {s}", .{if (on) "on" else "off"});
        } else {
            io_utils.eprintln(io, "unknown feature: {s} (think|show-think)", .{feature});
        }
        return false;
    }
    if (std.mem.eql(u8, name, "/verbose")) {
        const v = it.next() orelse "min";
        cfg.verbose = parseLevel(v, &.{ "off", "min", "info", "all" }) orelse {
            io_utils.eprintln(io, "usage: /verbose <off|min|info|all>", .{});
            return false;
        };
        io_utils.println(io, "verbose {s}", .{v});
        return false;
    }
    if (std.mem.eql(u8, name, "/metrics")) {
        const v = it.next() orelse "min";
        cfg.metrics = parseLevel(v, &.{ "off", "min", "all" }) orelse {
            io_utils.eprintln(io, "usage: /metrics <off|min|all>", .{});
            return false;
        };
        io_utils.println(io, "metrics {s}", .{v});
        return false;
    }
    if (std.mem.eql(u8, name, "/temperature")) {
        const v = it.next() orelse return usage(io, "/temperature <F>");
        cfg.temperature = std.fmt.parseFloat(f32, v) catch return bad(io, "temperature", v);
        io_utils.println(io, "temperature {d}", .{cfg.temperature});
        return false;
    }
    if (std.mem.eql(u8, name, "/top-p")) {
        const v = it.next() orelse return usage(io, "/top-p <F>");
        cfg.top_p = std.fmt.parseFloat(f32, v) catch return bad(io, "top-p", v);
        io_utils.println(io, "top-p {d}", .{cfg.top_p});
        return false;
    }
    if (std.mem.eql(u8, name, "/top-k")) {
        const v = it.next() orelse return usage(io, "/top-k <N>");
        cfg.top_k = std.fmt.parseInt(u32, v, 10) catch return bad(io, "top-k", v);
        io_utils.println(io, "top-k {d}", .{cfg.top_k});
        return false;
    }
    if (std.mem.eql(u8, name, "/max-tokens")) {
        const v = it.next() orelse return usage(io, "/max-tokens <N>");
        cfg.max_tokens = std.fmt.parseInt(usize, v, 10) catch return bad(io, "max-tokens", v);
        io_utils.println(io, "max-tokens {d}", .{cfg.max_tokens});
        return false;
    }
    if (std.mem.eql(u8, name, "/max-think")) {
        const v = it.next() orelse return usage(io, "/max-think <N>");
        cfg.max_think = std.fmt.parseInt(usize, v, 10) catch return bad(io, "max-think", v);
        io_utils.println(io, "max-think {d}", .{cfg.max_think});
        return false;
    }
    if (std.mem.eql(u8, name, "/rename")) {
        const rest = std.mem.trimStart(u8, text["/rename".len..], " \t");
        if (rest.len == 0) return usage(io, "/rename <name>");
        sessions.rename(ctx.allocator, ctx.io, cfg.session_id, rest) catch |err| {
            io_utils.eprintln(io, "rename: {s}", .{@errorName(err)});
            return false;
        };
        cfg.session_name = arena.dupe(u8, rest) catch cfg.session_name;
        // Refresh the service's name so later saves keep it.
        client.session(ctx.allocator, cfg.session_id, cfg.session_name, cfg.model) catch {};
        io_utils.println(io, "session renamed to '{s}'", .{cfg.session_name});
        return false;
    }
    if (std.mem.eql(u8, name, "/sessions")) {
        const all = sessions.list(arena, io) catch |err| {
            io_utils.eprintln(io, "sessions: {s}", .{@errorName(err)});
            return false;
        };
        if (all.len == 0) {
            io_utils.println(io, "(no sessions)", .{});
            return false;
        }
        for (all, 0..) |rec, i| {
            const label: []const u8 = if (rec.name.len > 0) rec.name else "(unnamed)";
            const mark: []const u8 = if (std.mem.eql(u8, rec.id, cfg.session_id)) "*" else " ";
            io_utils.println(io, "{s} {d}. {s}  {s}  {d} turns", .{ mark, i + 1, rec.id, label, rec.turns.len });
        }
        const pick = it.next() orelse return false;
        const idx = std.fmt.parseInt(usize, pick, 10) catch return bad(io, "session", pick);
        if (idx == 0 or idx > all.len) return bad(io, "session", pick);
        const target = all[idx - 1];
        client.session(ctx.allocator, target.id, target.name, cfg.model) catch |err| {
            io_utils.eprintln(io, "switch: {s}", .{@errorName(err)});
            return false;
        };
        cfg.session_id = arena.dupe(u8, target.id) catch cfg.session_id;
        cfg.session_name = arena.dupe(u8, target.name) catch cfg.session_name;
        io_utils.println(io, "switched to {s} ({d} turns)", .{ cfg.session_id, target.turns.len });
        return false;
    }
    if (std.mem.eql(u8, name, "/soul")) {
        // Everything after the command name is the soul (a string or a file).
        const rest = std.mem.trimStart(u8, text["/soul".len..], " \t");
        cfg.soul = soul_mod.resolve(arena, io, rest);
        io_utils.println(io, "soul set ({d} bytes)", .{cfg.soul.len});
        return false;
    }
    io_utils.eprintln(io, "unknown command: {s} (/help)", .{name});
    return false;
}

fn printMetrics(io: std.Io, m: service.Metrics, level: u8) void {
    const decode_s = m.decode_ms / 1000.0;
    const tps: f64 = if (decode_s > 0) @as(f64, @floatFromInt(m.decode_tokens)) / decode_s else 0;
    io_utils.println(io, "[prefill {d:.0} ms | decode {d:.0} ms | {d} tok | {d:.1} tok/s]", .{
        m.prefill_ms, m.decode_ms, m.decode_tokens, tps,
    });
    if (level >= 2) {
        io_utils.println(io, "[prompt {d} tok | cache {d} | rss {d} kB]", .{
            m.prefill_tokens, m.cache_matched, m.rss_kb,
        });
    }
}

/// The text-to-text REPL: one user line per turn, the reply streamed to stdout.
pub fn run(ctx: *const command.Context, client: *service.ChatClient, cfg: *Config) !void {
    const io = ctx.io;
    io_utils.println(io, "chat — /help for commands, /exit or Ctrl-D to quit", .{});

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    // `/soul` values outlive the per-turn arena reset.
    var soul_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer soul_arena.deinit();
    var rbuf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &rbuf);

    while (true) {
        io_utils.print(io, "\n> ", .{});
        const maybe = stdin_reader.interface.takeDelimiter('\n') catch null;
        const line = maybe orelse break;
        const text = std.mem.trim(u8, line, " \t\r");
        if (text.len == 0) continue;
        if (text[0] == '/') {
            if (slash(ctx, client, cfg, soul_arena.allocator(), text)) break;
            continue;
        }
        var sink_ctx = StdoutSink{ .io = io };
        const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = StdoutSink.write };
        const m = client.send(
            arena.allocator(),
            text,
            cfg.max_tokens,
            cfg.max_think,
            cfg.temperature,
            cfg.top_k,
            cfg.top_p,
            cfg.think,
            cfg.show_think,
            cfg.verbose,
            cfg.soul,
            sink,
        ) catch |err| {
            io_utils.eprintln(io, "\nchat: {s}", .{@errorName(err)});
            continue;
        };
        io_utils.println(io, "", .{});
        if (cfg.metrics > 0) printMetrics(io, m, cfg.metrics);
        _ = arena.reset(.retain_capacity);
    }
}

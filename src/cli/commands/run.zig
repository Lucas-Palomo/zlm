const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const Args = @import("../args.zig").Args;
const store = @import("../../image/store.zig");
const service = @import("../../runtime/service.zig");
const session_mod = @import("../../runtime/session.zig");
const soul_mod = @import("soul.zig");
const Context = command.Context;

/// Writes streamed generation chunks to stdout as they arrive.
const StdoutSink = struct {
    io: std.Io,

    fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *StdoutSink = @ptrCast(@alignCast(ctx));
        io_utils.writeAll(self.io, io_utils.stdout(), bytes);
    }
};

pub const help_text =
    \\Runs a tuned runtime.
    \\
    \\Usage: zlm run <runtime-alias> --prompt <text> --soul <text|file.txt>
    \\       [--session <uuid>] [--max-tokens N] [--max-think N]
    \\       [--temperature F] [--top-k N] [--top-p F] [--no-stream]
    \\       [--metrics[=full]] [--verbose[=min|info|all]]
    \\       [--cold | --warm | --hot[=minutes]] [--timeout=minutes]
    \\
    \\Sends the prompt to the alias's background service and prints the
    \\generated text. When the service is offline it is loaded first
    \\(printing a notice); `load` only preloads. State/TTL flags apply only
    \\to the auto-load (defaults: warm, 30 min idle timeout; --timeout=0 holds).
    \\
    \\--soul is required: the system turn that sets the model's persona. A
    \\value naming an existing file is read as the soul text; "" is no soul.
    \\
    \\--metrics prints TTFT/prefill/decode timings and peak RSS; `=full`
    \\adds the per-phase trace. --verbose[=min|info|all] traces to stderr
    \\(bare = min): min = the phase timings only, info = min plus the per-layer
    \\decode time and the per-token logits, all = info plus prompt ids and the raw
    \\think block. Diagnostic only; stdout is unchanged.
    \\
    \\Sampling defaults to greedy: --temperature 0 (unset) is argmax.
    \\--top-k 0 (unset) disables top-k; --top-p 1 (unset) disables nucleus.
    \\A non-zero temperature samples with a request-seeded RNG (reproducible).
    \\
    \\--think enables the model's reasoning mode (default off; ignored by
    \\models without a think template). --max-think N caps the reasoning tokens
    \\in a budget separate from --max-tokens (the answer keeps its full budget;
    \\default 256 + 256).
    \\
    \\Streaming is on by default (tokens print as they are generated);
    \\--no-stream waits for the whole decode. The final text is identical.
    \\
    \\--show-think includes the model's reasoning block in the output; without
    \\it the reasoning is hidden (batch and stream).
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--prompt", .short = 'p', .kind = .value },
    .{ .long = "--soul", .kind = .value, .required = true },
    .{ .long = "--session", .kind = .value },
    .{ .long = "--max-tokens", .kind = .uint },
    .{ .long = "--max-think", .kind = .uint },
    .{ .long = "--temperature", .kind = .float },
    .{ .long = "--top-k", .kind = .uint },
    .{ .long = "--top-p", .kind = .float },
    .{ .long = "--metrics", .kind = .value },
    .{ .long = "--profile", .kind = .value },
    .{ .long = "--think", .kind = .flag },
    .{ .long = "--show-think", .kind = .flag },
    .{ .long = "--stream", .kind = .flag },
    .{ .long = "--no-stream", .kind = .flag },
    .{ .long = "--verbose", .kind = .value },
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
        if (std.mem.eql(u8, it.*, "--metrics")) it.* = "--metrics=";
        if (std.mem.eql(u8, it.*, "--profile")) it.* = "--profile=all";
        if (std.mem.eql(u8, it.*, "--verbose")) it.* = "--verbose=min";
    }
    var args = Args.init(items);
    var opt = options.parse(ctx.allocator, &args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "run: {s}", .{@errorName(err)});
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
    const prompt = opt.get("--prompt");
    // `--soul` is required: the system turn that gives the model its persona.
    // A value naming an existing file is read; anything else (including "") is
    // the literal text.
    const soul_arg = opt.get("--soul") orelse {
        io_utils.eprintln(ctx.io, "Missing --soul <text|file.txt>", .{});
        return error.MissingArgument;
    };
    const max_tokens: usize = opt.uint("--max-tokens", usize) orelse 256;
    const max_think: usize = opt.uint("--max-think", usize) orelse 256;
    const temperature: f32 = opt.float("--temperature", f32) orelse 0;
    const top_k: u32 = opt.uint("--top-k", u32) orelse 0;
    const top_p: f32 = opt.float("--top-p", f32) orelse 1;
    const think = opt.has("--think");
    const show_think = opt.has("--show-think");
    const stream = !opt.has("--no-stream");
    const metrics_val = opt.get("--metrics");
    const want_metrics = metrics_val != null;
    const metrics_full = if (metrics_val) |v| std.mem.eql(u8, v, "full") else false;
    const profile_val = opt.get("--profile");
    const want_profile = profile_val != null;
    // `--verbose` (bare = `min`): 0 off / 1 min / 2 info / 3 all; unknown -> min.
    const verbose: u8 = if (opt.get("--verbose")) |v|
        (if (std.mem.eql(u8, v, "info")) @as(u8, 2) else if (std.mem.eql(u8, v, "all")) 3 else 1)
    else
        0;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const soul = soul_mod.resolve(arena.allocator(), ctx.io, soul_arg);
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

    const load_start_ns = std.Io.Clock.awake.now(ctx.io).nanoseconds;
    // A stale daemon (online but from a different build / protocol) is restarted
    // rather than reused, so it can never serve a request it cannot parse.
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
            io_utils.eprintln(ctx.io, "run: cannot load '{s}': {s}", .{ alias, @errorName(err) });
            return err;
        };
    }
    const load_ns = std.Io.Clock.awake.now(ctx.io).nanoseconds - load_start_ns;
    const load_ms = @as(f64, @floatFromInt(load_ns)) / 1_000_000.0;

    var reply_metrics: ?service.Metrics = null;
    if (opt.get("--session")) |sid| {
        // A run inside a persistent session: the service loads the saved turns,
        // appends this prompt + the reply and saves it back.
        var client = service.ChatClient.connect(ctx.io, sock_path) catch |err| {
            io_utils.eprintln(ctx.io, "run: connect: {s}", .{@errorName(err)});
            return err;
        };
        defer client.close();
        client.session(ctx.allocator, sid, "", alias) catch |err| {
            io_utils.eprintln(ctx.io, "run: session: {s}", .{@errorName(err)});
            return err;
        };
        io_utils.println(ctx.io, "runtime: {s}", .{alias});
        io_utils.println(ctx.io, "", .{});
        var sink_ctx = StdoutSink{ .io = ctx.io };
        const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = StdoutSink.write };
        const m = client.send(ctx.allocator, prompt orelse "", max_tokens, max_think, temperature, top_k, top_p, think, show_think, verbose, soul, sink) catch |err| {
            io_utils.eprintln(ctx.io, "generation failed: {s}", .{@errorName(err)});
            return err;
        };
        io_utils.println(ctx.io, "", .{});
        reply_metrics = m;
    } else if (stream) {
        io_utils.println(ctx.io, "runtime: {s}", .{alias});
        io_utils.println(ctx.io, "", .{});
        var sink_ctx = StdoutSink{ .io = ctx.io };
        const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = StdoutSink.write };
        reply_metrics = service.generateStream(ctx.allocator, ctx.io, sock_path, prompt orelse "", max_tokens, max_think, temperature, top_k, top_p, think, show_think, verbose, soul, sink) catch |err| {
            io_utils.eprintln(ctx.io, "generation failed: {s}", .{@errorName(err)});
            return err;
        };
        io_utils.println(ctx.io, "", .{});
    } else {
        const reply = service.generate(ctx.allocator, ctx.io, sock_path, prompt orelse "", max_tokens, max_think, temperature, top_k, top_p, think, show_think, verbose, soul) catch |err| {
            io_utils.eprintln(ctx.io, "generation failed: {s}", .{@errorName(err)});
            return err;
        };
        defer ctx.allocator.free(reply.text);
        io_utils.println(ctx.io, "runtime: {s}", .{alias});
        io_utils.println(ctx.io, "", .{});
        io_utils.println(ctx.io, "{s}", .{reply.text});
        reply_metrics = reply.metrics;
    }

    if (verbose > 0) printVerbose(ctx.io, reply_metrics, load_ms);
    if (want_metrics) printMetrics(ctx.io, reply_metrics, metrics_full, load_ms);
    if (want_profile) printProfile(ctx.io, reply_metrics, alias, load_ms);
}

/// `run --profile`: a per-AtomicOp breakdown of the decode (diagnostic only).
fn printProfile(io: std.Io, metrics: ?service.Metrics, alias: []const u8, load_ms: f64) void {
    const m = metrics orelse {
        io_utils.println(io, "profile: unavailable", .{});
        return;
    };
    const p = m.profile;
    const total = p.opNs() + p.residency_ns;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        }
    }.f;
    const tok = m.decode_tokens;
    // Complete generation wall-clock; prefill + decode fallback for an old daemon.
    const gen = if (m.gen_ms > 0) m.gen_ms else m.prefill_ms + m.decode_ms;
    io_utils.println(io, "", .{});
    io_utils.println(io, "ZLM Execution Profile", .{});
    io_utils.println(io, "------------------------------------------", .{});
    io_utils.println(io, "Runtime:  {s}   Profile: all", .{alias});
    io_utils.println(io, "", .{});
    io_utils.println(io, "PHASE", .{});
    io_utils.println(io, "  load       {d:.2} ms", .{load_ms});
    io_utils.println(io, "  generation {d:.2} ms  ({d:.1} tok/s)", .{ gen, rate(tok, gen) });
    io_utils.println(io, "  prefill    {d:.2} ms", .{m.prefill_ms});
    io_utils.println(io, "  decode     {d:.2} ms", .{m.decode_ms});
    io_utils.println(io, "", .{});
    io_utils.println(io, "DECODE BREAKDOWN (per token)", .{});
    io_utils.println(io, "  MatVec             {d:.2} ms", .{ms(p.matvec_ns) / pt(tok)});
    io_utils.println(io, "  MatMul             {d:.2} ms", .{ms(p.matmul_ns) / pt(tok)});
    io_utils.println(io, "  Attention / KV     {d:.2} ms", .{ms(p.attention_ns) / pt(tok)});
    io_utils.println(io, "  Norm               {d:.2} ms", .{ms(p.norm_ns) / pt(tok)});
    io_utils.println(io, "  Activation         {d:.2} ms", .{ms(p.activation_ns) / pt(tok)});
    io_utils.println(io, "  Elementwise        {d:.2} ms", .{ms(p.elementwise_ns) / pt(tok)});
    io_utils.println(io, "  Rope               {d:.2} ms", .{ms(p.rope_ns) / pt(tok)});
    io_utils.println(io, "  Dequant            {d:.2} ms", .{ms(p.dequant_ns) / pt(tok)});
    io_utils.println(io, "  Residency          {d:.2} ms", .{ms(p.residency_ns) / pt(tok)});
    io_utils.println(io, "", .{});
    io_utils.println(io, "HOTSPOTS", .{});
    printHotspot(io, "matvec", p.matvec_ns, total);
    printHotspot(io, "attention", p.attention_ns, total);
    printHotspot(io, "norm", p.norm_ns, total);
    printHotspot(io, "activation", p.activation_ns, total);
    printHotspot(io, "elementwise", p.elementwise_ns, total);
    printHotspot(io, "residency", p.residency_ns, total);
    io_utils.println(io, "", .{});
    io_utils.println(io, "RUNTIME METRICS", .{});
    const mb = @as(f64, @floatFromInt(p.weight_bytes)) / (1024.0 * 1024.0) / pt(tok);
    const gbs = if (p.matvec_ns > 0) @as(f64, @floatFromInt(p.weight_bytes)) / (@as(f64, @floatFromInt(p.matvec_ns)) / 1e9) / 1e9 else 0;
    io_utils.println(io, "  Weight bytes/token: {d:.1} MiB", .{mb});
    io_utils.println(io, "  Matvec bandwidth:   {d:.1} GB/s", .{gbs});
    io_utils.println(io, "  Dispatches/token:  {d}", .{if (tok > 0) p.dispatches / tok else 0});
    io_utils.println(io, "  Avg requested threads: {d:.2}  (serial dispatches {d}%)", .{
        if (p.dispatches > 0) @as(f64, @floatFromInt(p.parts_sum)) / @as(f64, @floatFromInt(p.dispatches)) else 0,
        if (p.dispatches > 0) 100.0 * @as(f64, @floatFromInt(p.serial_dispatches)) / @as(f64, @floatFromInt(p.dispatches)) else 0,
    });
    const serial_pct = if (p.opNs() + p.residency_ns > 0)
        100.0 * @as(f64, @floatFromInt(p.serial_ns)) / @as(f64, @floatFromInt(p.opNs() + p.residency_ns))
    else
        0;
    io_utils.println(io, "  Serial-dispatch time: {d:.1}%", .{serial_pct});
    io_utils.println(io, "  Matvec: {d} dispatches, avg parts {d:.2}", .{
        if (tok > 0) p.matvec_count / tok else p.matvec_count,
        if (p.matvec_count > 0) @as(f64, @floatFromInt(p.matvec_parts_sum)) / @as(f64, @floatFromInt(p.matvec_count)) else 0,
    });
    io_utils.println(io, "  Decode latency:    {d:.2} ms/token", .{if (tok > 0) m.decode_ms / @as(f64, @floatFromInt(tok)) else 0});
    io_utils.println(io, "  Generation:        {d:.1} tok/s", .{rate(tok, gen)});
}

fn pt(tokens: usize) f64 {
    return @floatFromInt(@max(tokens, 1));
}

fn printHotspot(io: std.Io, name: []const u8, ns: u64, total: u64) void {
    const pct = if (total > 0) @as(f64, @floatFromInt(ns)) * 100.0 / @as(f64, @floatFromInt(total)) else 0;
    io_utils.println(io, "  {s:<12} {d:.1}%", .{ name, pct });
}

fn rate(tokens: usize, ms: f64) f64 {
    if (ms <= 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / (ms / 1000.0);
}

/// `run --verbose` phase line (stderr). The per-layer and per-token facts
/// arrive as `TRACE` frames from the daemon; `load` is only known here.
fn printVerbose(io: std.Io, metrics: ?service.Metrics, load_ms: f64) void {
    const m = metrics orelse {
        io_utils.eprintln(io, "phase load={d:.3} prefill=0 decode=0 generation=0", .{load_ms});
        return;
    };
    const gen = if (m.gen_ms > 0) m.gen_ms else m.prefill_ms + m.decode_ms;
    io_utils.eprintln(io, "phase load={d:.3} prefill={d:.3} decode={d:.3} generation={d:.3}", .{
        load_ms, m.prefill_ms, m.decode_ms, gen,
    });
}

fn printMetrics(io: std.Io, metrics: ?service.Metrics, full: bool, load_ms: f64) void {
    const m = metrics orelse {
        io_utils.println(io, "", .{});
        io_utils.println(io, "metrics: unavailable", .{});
        return;
    };
    var b0: [32]u8 = undefined;
    var b1: [32]u8 = undefined;
    var b2: [32]u8 = undefined;
    var b3: [32]u8 = undefined;
    // The complete generation wall-clock (tokenize + prefill + decode +
    // detokenize); fall back to prefill + decode for an older daemon.
    const gen = if (m.gen_ms > 0) m.gen_ms else m.prefill_ms + m.decode_ms;
    io_utils.println(io, "", .{});
    io_utils.println(io, "metrics:", .{});
    io_utils.println(io, "  ttft:       {s}", .{io_utils.formatDuration(&b0, m.prefill_ms)});
    io_utils.println(io, "  generation: {s}  {d:.1} tok/s  ({d} tok)", .{
        io_utils.formatDuration(&b3, gen), rate(m.decode_tokens, gen), m.decode_tokens,
    });
    io_utils.println(io, "  prefill:  {s}  {d:.1} tok/s  ({d} tok)", .{
        io_utils.formatDuration(&b1, m.prefill_ms), rate(m.prefill_tokens, m.prefill_ms), m.prefill_tokens,
    });
    io_utils.println(io, "  decode:   {s}  {d:.1} tok/s  ({d} tok)", .{
        io_utils.formatDuration(&b2, m.decode_ms), rate(m.decode_tokens, m.decode_ms), m.decode_tokens,
    });
    if (m.cache_matched > 0) {
        io_utils.println(io, "  cache:    {d} tok reused (prefill skipped)", .{m.cache_matched});
    }
    io_utils.println(io, "  peak rss: {d} kB", .{m.rss_kb});
    if (full) {
        io_utils.println(io, "  total:    {s}", .{io_utils.formatDuration(&b0, m.prefill_ms + m.decode_ms)});
        io_utils.println(io, "  phases:   load={s} prefill={s} decode={s}", .{
            io_utils.formatDuration(&b0, load_ms),
            io_utils.formatDuration(&b1, m.prefill_ms),
            io_utils.formatDuration(&b2, m.decode_ms),
        });
    }
}

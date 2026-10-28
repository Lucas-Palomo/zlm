const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const store = @import("../image/store.zig");
const zrun = @import("../image/zrun.zig");
const plan_mod = @import("../plan/root.zig");
const residency = @import("residency.zig");
const session_mod = @import("session.zig");
const sessions_mod = @import("sessions.zig");
const io_utils = @import("../utils/io.zig");

/// Line-framed, single-client protocol. Requests: `PING`,
/// `GENERATE\t<max_tokens>\t<temperature>\t<top_k>\t<top_p>\t<think>\t<stream>\t<show_think>\t<verbose>\t<prompt>`,
/// `QUIT`. Replies: `PONG`,
/// `OK\t<byte_len>\t<prefill_ms>\t<decode_ms>\t<decode_tokens>\t<rss_kb>\t<prefill_tokens>\n<bytes>`
/// (the older `OK\t<byte_len>` is still understood by the client), `ERR\t<msg>`.
///
/// When `verbose > 0`, the daemon emits `TRACE\t<line>\n` diagnostic frames
/// before the final `OK` header; the client prints each to stderr. They never
/// feed a decision and never change the stdout text.
///
/// A `stream = 1` request replies with a sequence of `DATA\t<chunk_len>\n` frames
/// (each followed by the chunk bytes), then the same `OK\t<total_len>\t<metrics>`
/// header with **no** body. A `stream = 0` request keeps the batch reply above.
/// `TRACE` frames may interleave with `DATA` frames.
///
/// The prompt is the variable-length tail of `GENERATE`, placed after the eight
/// fixed fields so it may contain any byte except the `\n` frame
/// delimiter (tabs included). The previous `prompt\t<max_tokens>\t...` and the
/// older greedy `prompt\t<max_tokens>` shapes are still accepted on a
/// best-effort basis (see `parseGenerate`); they default think and stream off.
pub const ping_request = "PING";
pub const pong_reply = "PONG";
pub const quit_request = "QUIT";
pub const generate_prefix = "GENERATE\t";
/// `chat` request: same field shape as `GENERATE`, but the trailing text is the
/// new user turn. The service keeps the connection's conversation, formats it
/// with the runtime's chat template and appends the assistant reply.
pub const chat_prefix = "CHAT\t";
/// Clear the connection's chat conversation (the REPL `/clear`).
pub const chat_reset_request = "CHAT_RESET";
/// Seed the connection's conversation from a saved session: the payload is the
/// turns joined by 0x1e, each `u`/`a` + the escaped text.
pub const chat_seed_prefix = "CHAT_SEED\t";
/// Bind the connection to a persistent session: `SESSION\t<uuid>\t<name>\t<model>`.
/// The service loads the session's turns (if any) and saves them after each
/// `CHAT`, so the conversation survives across invocations.
pub const session_prefix = "SESSION\t";
pub const trace_prefix = "TRACE\t";

const max_line = 64 * 1024;
/// Base cold-start deadline (ms): process spawn plus `Session.open` up to the
/// point the socket is bound. A CPU plan never needs more than this.
const spawn_timeout_ms: usize = 15000;
/// Conservative effective host<->device staging rate for the daemon's
/// VRAM-resident upload, GB/s. Numerically GB/s == bytes/ns, so
/// `bytes / (bw * 1e6)` is a duration in ms. Deliberately far below a PCIe
/// link: the daemon maps the (possibly cold) image, faults each resident
/// tensor's pages in from disk, and copies them tensor-by-tensor into
/// host-visible VRAM, so the first cold run is bound by the slowest of the
/// disk read and the coherent copy — not the raw bus. Overestimating only
/// extends the maximum wait (a live socket returns immediately), so err low.
const cold_upload_gbps: f64 = 0.05;
const poll_ms: usize = 25;
const idle_poll_ms: i32 = 250;
/// Idle polls between registry orphan checks (250 ms each -> ~5 s).
const orphan_check_polls: u32 = 20;

/// Default idle minutes before a service auto-unloads. Without this a daemon
/// started without `--timeout` would hold its model in RAM forever. `--timeout=0`
/// opts out (hold until an explicit `unload`).
pub const default_timeout_minutes: u32 = 30;

/// Residency state and idle TTLs the service applies at startup. `hot_minutes`
/// demotes `hot -> warm` after that many idle minutes (null holds); `timeout_minutes`
/// auto-unloads the service after that many idle minutes (null = the default
/// `default_timeout_minutes`, 0 = hold until an explicit `unload`).
pub const Options = struct {
    state: residency.State = .warm,
    hot_minutes: ?u32 = null,
    timeout_minutes: ?u32 = null,
};

/// Resolve the service idle timeout: an explicit `--timeout=N` wins (`0` = no
/// timeout), otherwise the default.
fn resolveTimeoutNs(opts_timeout: ?u32) ?i96 {
    const m = opts_timeout orelse default_timeout_minutes;
    if (m == 0) return null;
    return minutesToNs(m);
}

/// Map a `--state=` name to a residency state; unknown names fall back to warm.
pub fn stateFromName(name: []const u8) residency.State {
    if (std.mem.eql(u8, name, "cold")) return .cold;
    if (std.mem.eql(u8, name, "hot")) return .hot;
    return .warm;
}

fn stateName(state: residency.State) []const u8 {
    return switch (state) {
        .cold => "cold",
        .warm => "warm",
        .hot => "hot",
    };
}

const IdleAction = enum { none, demote, quit };

/// The TTL decision for an idle duration. Auto-unload outranks demotion.
fn idleAction(idle_ns: i96, hot_ttl_ns: ?i96, timeout_ns: ?i96, hot_active: bool) IdleAction {
    if (timeout_ns) |t| {
        if (idle_ns >= t) return .quit;
    }
    if (hot_active) {
        if (hot_ttl_ns) |t| {
            if (idle_ns >= t) return .demote;
        }
    }
    return .none;
}

fn minutesToNs(minutes: u32) i96 {
    return @as(i96, minutes) * std.time.ns_per_min;
}

/// Connect to the plan's socket and round-trip a `PING`. Online iff it answers.
pub fn ping(io: Io, sock_path: []const u8) bool {
    const ua = Io.net.UnixAddress.init(sock_path) catch return false;
    var stream = ua.connect(io) catch return false;
    defer stream.close(io);
    var wbuf: [64]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    w.interface.writeAll(ping_request ++ "\n") catch return false;
    w.interface.flush() catch return false;
    var rbuf: [64]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    const maybe = r.interface.takeDelimiter('\n') catch return false;
    const line = maybe orelse return false;
    return std.mem.startsWith(u8, line, pong_reply);
}

/// The build fingerprint injected at compile time by `build.zig`: a hash of the
/// sources plus the build config. It changes on any code change, so the service
/// handshake detects a daemon left over from a previous build without a manual
/// version to bump. `0` when the build could not compute it (the handshake is
/// then skipped and an online daemon is reused).
pub const build_fingerprint: u64 = @import("build_options").build_fingerprint;

/// Like `ping`, but returns the daemon's build fingerprint (0 for a daemon that
/// replies a bare `PONG`, e.g. an older build), or null when offline.
pub fn pingFingerprint(io: Io, sock_path: []const u8) ?u64 {
    const ua = Io.net.UnixAddress.init(sock_path) catch return null;
    var stream = ua.connect(io) catch return null;
    defer stream.close(io);
    var wbuf: [64]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    w.interface.writeAll(ping_request ++ "\n") catch return null;
    w.interface.flush() catch return null;
    var rbuf: [64]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    const maybe = r.interface.takeDelimiter('\n') catch return null;
    const line = maybe orelse return null;
    if (!std.mem.startsWith(u8, line, pong_reply)) return null;
    if (line.len == pong_reply.len) return 0;
    if (line[pong_reply.len] != '\t') return null;
    return std.fmt.parseInt(u64, line[pong_reply.len + 1 ..], 16) catch 0;
}

/// Connect to the plan's socket and send `QUIT`. Returns whether the service
/// was reachable.
pub fn quit(io: Io, sock_path: []const u8) bool {
    const ua = Io.net.UnixAddress.init(sock_path) catch return false;
    var stream = ua.connect(io) catch return false;
    defer stream.close(io);
    var wbuf: [64]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    w.interface.writeAll(quit_request ++ "\n") catch return false;
    w.interface.flush() catch return false;
    return true;
}

/// Timing fields the daemon appends to a `GENERATE` reply. Diagnostic only.
pub const Metrics = struct {
    prefill_ms: f64,
    decode_ms: f64,
    decode_tokens: usize,
    rss_kb: u64,
    prefill_tokens: usize,
    /// Prompt tokens served from the KV cache (0 when disabled/miss). Diagnostic.
    cache_matched: usize = 0,
    /// Per-AtomicOp decode accounting. Diagnostic only (`run --profile`).
    profile: session_mod.OpProfile = .{},
    /// Wall-clock of the whole `GENERATE` in the daemon (tokenize + prefill +
    /// decode + detokenize), the complete "generation". 0 when the daemon is
    /// older and omits it (the client then falls back to prefill + decode).
    gen_ms: f64 = 0,
};

pub const Reply = struct {
    text: []u8,
    metrics: ?Metrics = null,
};

pub const Header = struct {
    len: usize,
    metrics: ?Metrics = null,
};

/// Parse the reply header. `OK\t<len>` (no timing) yields `metrics = null`;
/// the extended `OK\t<len>\t<prefill_ms>\t<decode_ms>\t<tokens>\t<rss_kb>\t<prefill_tokens>\t<cache_matched>`
/// carries them, followed by the per-op profile and a trailing `<gen_ms>` (the
/// complete generation wall-clock). Trailing fields are optional for older
/// daemons (the client falls back to `prefill + decode` when `gen_ms` is 0).
fn nextU64(it: anytype) u64 {
    const v = it.next() orelse return 0;
    return std.fmt.parseInt(u64, v, 10) catch 0;
}

pub fn parseHeader(line: []const u8) !Header {
    if (!std.mem.startsWith(u8, line, "OK\t")) return error.ProtocolError;
    var it = std.mem.splitScalar(u8, line[3..], '\t');
    const len_str = it.next() orelse return error.ProtocolError;
    const len = try std.fmt.parseInt(usize, len_str, 10);

    var metrics: ?Metrics = null;
    if (it.next()) |pf| {
        const df = it.next() orelse return .{ .len = len };
        const tk = it.next() orelse return .{ .len = len };
        metrics = .{
            .prefill_ms = std.fmt.parseFloat(f64, pf) catch 0,
            .decode_ms = std.fmt.parseFloat(f64, df) catch 0,
            .decode_tokens = std.fmt.parseInt(usize, tk, 10) catch 0,
            .rss_kb = if (it.next()) |rs| (std.fmt.parseInt(u64, rs, 10) catch 0) else 0,
            .prefill_tokens = if (it.next()) |ps| (std.fmt.parseInt(usize, ps, 10) catch 0) else 0,
            .cache_matched = if (it.next()) |mm| (std.fmt.parseInt(usize, mm, 10) catch 0) else 0,
            .profile = .{
                .matvec_ns = nextU64(&it),
                .matmul_ns = nextU64(&it),
                .attention_ns = nextU64(&it),
                .norm_ns = nextU64(&it),
                .activation_ns = nextU64(&it),
                .elementwise_ns = nextU64(&it),
                .rope_ns = nextU64(&it),
                .dequant_ns = nextU64(&it),
                .residency_ns = nextU64(&it),
                .dispatches = nextU64(&it),
                .weight_bytes = nextU64(&it),
                .parts_sum = nextU64(&it),
                .serial_dispatches = nextU64(&it),
                .serial_ns = nextU64(&it),
                .matvec_count = nextU64(&it),
                .matvec_parts_sum = nextU64(&it),
            },
            .gen_ms = if (it.next()) |g| (std.fmt.parseFloat(f64, g) catch 0) else 0,
        };
    }
    return .{ .len = len, .metrics = metrics };
}

/// Send `GENERATE` and return the generated text (allocated from `allocator`)
/// plus the daemon's timing when present. The fixed numeric fields precede the
/// prompt, which is the trailing field, so the prompt may contain any
/// non-newline byte (tabs included).
pub fn generate(
    allocator: std.mem.Allocator,
    io: Io,
    sock_path: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    max_think: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    think: bool,
    show_think: bool,
    verbose: u8,
    soul: []const u8,
) !Reply {
    const ua = try Io.net.UnixAddress.init(sock_path);
    var stream = try ua.connect(io);
    defer stream.close(io);
    const soul_esc = try escapeField(allocator, soul);
    defer allocator.free(soul_esc);
    var wbuf: [4096]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    try w.interface.print(generate_prefix ++ "{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{s}\n", .{ max_tokens, temperature, top_k, top_p, @intFromBool(think), 0, @intFromBool(show_think), verbose, max_think, soul_esc, prompt });
    try w.interface.flush();
    var rbuf: [4096]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    while (true) {
        const maybe = try r.interface.takeDelimiter('\n');
        const line = maybe orelse return error.ProtocolError;
        if (std.mem.startsWith(u8, line, "ERR\t")) return error.ServiceError;
        if (std.mem.startsWith(u8, line, trace_prefix)) {
            io_utils.eprintln(io, "{s}", .{line[trace_prefix.len..]});
            continue;
        }
        const header = try parseHeader(line);
        const out = try allocator.alloc(u8, header.len);
        errdefer allocator.free(out);
        try r.interface.readSliceAll(out);
        return .{ .text = out, .metrics = header.metrics };
    }
}

/// Send a `stream = 1` `GENERATE` and feed each `DATA` chunk to `sink` as it
/// arrives. Returns the final `OK` header's metrics. Every emitted byte is
/// written through `sink`; the metrics' `decode_tokens`/timings are the same as
/// the batch path.
pub fn generateStream(
    allocator: std.mem.Allocator,
    io: Io,
    sock_path: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    max_think: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    think: bool,
    show_think: bool,
    verbose: u8,
    soul: []const u8,
    sink: session_mod.Sink,
) !Metrics {
    const ua = try Io.net.UnixAddress.init(sock_path);
    var stream = try ua.connect(io);
    defer stream.close(io);
    const soul_esc = try escapeField(allocator, soul);
    defer allocator.free(soul_esc);
    var wbuf: [4096]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    try w.interface.print(generate_prefix ++ "{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{s}\n", .{ max_tokens, temperature, top_k, top_p, @intFromBool(think), 1, @intFromBool(show_think), verbose, max_think, soul_esc, prompt });
    try w.interface.flush();
    var rbuf: [4096]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var chunk_buf: [4096]u8 = undefined;
    while (true) {
        const maybe = try r.interface.takeDelimiter('\n');
        const line = maybe orelse return error.ProtocolError;
        if (std.mem.startsWith(u8, line, "ERR\t")) return error.ServiceError;
        if (std.mem.startsWith(u8, line, trace_prefix)) {
            io_utils.eprintln(io, "{s}", .{line[trace_prefix.len..]});
            continue;
        }
        if (std.mem.startsWith(u8, line, "DATA\t")) {
            var remaining = std.fmt.parseInt(usize, line["DATA\t".len..], 10) catch return error.ProtocolError;
            while (remaining > 0) {
                const take = @min(remaining, chunk_buf.len);
                try r.interface.readSliceAll(chunk_buf[0..take]);
                try sink.write(sink.ctx, chunk_buf[0..take]);
                remaining -= take;
            }
            continue;
        }
        const header = try parseHeader(line);
        return header.metrics orelse error.ProtocolError;
    }
}

/// A long-lived `chat` connection: the service keeps the conversation for the
/// connection's lifetime, so the REPL sends one user turn per `send`.
pub const ChatClient = struct {
    io: Io,
    stream: Io.net.Stream,

    pub fn connect(io: Io, sock_path: []const u8) !ChatClient {
        const ua = try Io.net.UnixAddress.init(sock_path);
        return .{ .io = io, .stream = try ua.connect(io) };
    }

    pub fn close(self: *ChatClient) void {
        self.stream.close(self.io);
    }

    /// Clear the service's conversation for this connection (the REPL `/clear`).
    pub fn reset(self: *ChatClient) !void {
        var wbuf: [64]u8 = undefined;
        var w = self.stream.writer(self.io, &wbuf);
        try w.interface.print("{s}\n", .{chat_reset_request});
        try w.interface.flush();
        var rbuf: [64]u8 = undefined;
        var r = self.stream.reader(self.io, &rbuf);
        _ = try r.interface.takeDelimiter('\n');
    }

    /// Bind the connection to a persistent session and load its history.
    pub fn session(self: *ChatClient, allocator: std.mem.Allocator, id: []const u8, name: []const u8, model: []const u8) !void {
        const id_esc = try escapeField(allocator, id);
        defer allocator.free(id_esc);
        const name_esc = try escapeField(allocator, name);
        defer allocator.free(name_esc);
        const model_esc = try escapeField(allocator, model);
        defer allocator.free(model_esc);
        var wbuf: [4096]u8 = undefined;
        var w = self.stream.writer(self.io, &wbuf);
        try w.interface.print(session_prefix ++ "{s}\t{s}\t{s}\n", .{ id_esc, name_esc, model_esc });
        try w.interface.flush();
        var rbuf: [64]u8 = undefined;
        var r = self.stream.reader(self.io, &rbuf);
        _ = try r.interface.takeDelimiter('\n');
    }

    /// Replace the connection's conversation with `turns` (a saved session).
    pub fn seed(self: *ChatClient, allocator: std.mem.Allocator, turns: []const sessions_mod.Turn) !void {
        var wbuf: [4096]u8 = undefined;
        var w = self.stream.writer(self.io, &wbuf);
        try w.interface.writeAll(chat_seed_prefix);
        for (turns, 0..) |t, i| {
            if (i > 0) try w.interface.writeAll("\x1e");
            try w.interface.writeAll(if (std.mem.eql(u8, t.role, "user")) "u" else "a");
            const esc = try escapeField(allocator, t.text);
            defer allocator.free(esc);
            try w.interface.writeAll(esc);
        }
        try w.interface.writeAll("\n");
        try w.interface.flush();
        var rbuf: [64]u8 = undefined;
        var r = self.stream.reader(self.io, &rbuf);
        _ = try r.interface.takeDelimiter('\n');
    }

    /// Send one user turn and stream the reply through `sink`; returns the
    /// turn's metrics (the caller prints them per its `/metrics` level).
    pub fn send(
        self: *ChatClient,
        allocator: std.mem.Allocator,
        user_text: []const u8,
        max_tokens: usize,
        max_think: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        think: bool,
        show_think: bool,
        verbose: u8,
        soul: []const u8,
        sink: session_mod.Sink,
    ) !Metrics {
        const soul_esc = try escapeField(allocator, soul);
        defer allocator.free(soul_esc);
        var wbuf: [4096]u8 = undefined;
        var w = self.stream.writer(self.io, &wbuf);
        try w.interface.print(chat_prefix ++ "{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{s}\n", .{ max_tokens, temperature, top_k, top_p, @intFromBool(think), 1, @intFromBool(show_think), verbose, max_think, soul_esc, user_text });
        try w.interface.flush();

        var rbuf: [4096]u8 = undefined;
        var r = self.stream.reader(self.io, &rbuf);
        var chunk_buf: [4096]u8 = undefined;
        while (true) {
            const maybe = try r.interface.takeDelimiter('\n');
            const line = maybe orelse return error.ProtocolError;
            if (std.mem.startsWith(u8, line, "ERR\t")) return error.ServiceError;
            if (std.mem.startsWith(u8, line, trace_prefix)) {
                io_utils.eprintln(self.io, "{s}", .{line[trace_prefix.len..]});
                continue;
            }
            if (std.mem.startsWith(u8, line, "DATA\t")) {
                var remaining = std.fmt.parseInt(usize, line["DATA\t".len..], 10) catch return error.ProtocolError;
                while (remaining > 0) {
                    const take = @min(remaining, chunk_buf.len);
                    try r.interface.readSliceAll(chunk_buf[0..take]);
                    try sink.write(sink.ctx, chunk_buf[0..take]);
                    remaining -= take;
                }
                continue;
            }
            const header = try parseHeader(line);
            return header.metrics orelse error.ProtocolError;
        }
    }
};

/// Peak resident set size of the calling process in kB (Linux `VmHWM` from
/// `/proc/self/status`); 0 when unavailable.
fn peakRssKb(io: Io) u64 {
    if (builtin.os.tag != .linux) return 0;
    var buf: [4096]u8 = undefined;
    const data = Io.Dir.cwd().readFile(io, "/proc/self/status", &buf) catch return 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "VmHWM:")) continue;
        var fields = std.mem.tokenizeAny(u8, line["VmHWM:".len..], " \t");
        const value = fields.next() orelse return 0;
        return std.fmt.parseInt(u64, value, 10) catch 0;
    }
    return 0;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

/// Spawn a detached `zlm __serve <alias>` with stdout/stderr appended to
/// `log_path`. The child calls `setsid` on startup to leave the caller's
/// session.
pub fn spawn(
    allocator: std.mem.Allocator,
    io: Io,
    alias: []const u8,
    log_path: []const u8,
    opts: Options,
) !void {
    const exe = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(exe);
    var log = try Io.Dir.cwd().createFile(io, log_path, .{});
    defer log.close(io);

    var state_buf: [32]u8 = undefined;
    var hot_buf: [32]u8 = undefined;
    var timeout_buf: [32]u8 = undefined;
    var argv_buf: [8][]const u8 = undefined;
    var n: usize = 0;
    argv_buf[n] = exe;
    n += 1;
    argv_buf[n] = "__serve";
    n += 1;
    argv_buf[n] = alias;
    n += 1;
    argv_buf[n] = try std.fmt.bufPrint(&state_buf, "--state={s}", .{stateName(opts.state)});
    n += 1;
    if (opts.hot_minutes) |m| {
        argv_buf[n] = try std.fmt.bufPrint(&hot_buf, "--hot={d}", .{m});
        n += 1;
    }
    if (opts.timeout_minutes) |m| {
        argv_buf[n] = try std.fmt.bufPrint(&timeout_buf, "--timeout={d}", .{m});
        n += 1;
    }

    const child = try std.process.spawn(io, .{
        .argv = argv_buf[0..n],
        .stdin = .ignore,
        .stdout = .{ .file = log },
        .stderr = .{ .file = log },
    });
    _ = child;
}

/// Cold-start deadline for a plan whose daemon uploads `resident_bytes` into
/// VRAM: the base deadline plus the estimated upload time. `resident_bytes == 0`
/// (a CPU plan, or a GPU plan with no resident set) keeps the historical fixed
/// deadline, so the CPU path is unchanged.
fn spawnDeadlineMs(resident_bytes: u64) usize {
    if (resident_bytes == 0) return spawn_timeout_ms;
    const upload_ms = @as(f64, @floatFromInt(resident_bytes)) / (cold_upload_gbps * 1e6);
    return spawn_timeout_ms + @as(usize, @intFromFloat(upload_ms));
}

/// Resident VRAM bytes the daemon uploads on a cold start, read from the plan
/// artifact so the client can size the startup deadline. `0` when the plan is a
/// CPU plan or cannot be read (then the base deadline applies).
pub fn residentUploadBytes(allocator: std.mem.Allocator, io: Io, plan_path: []const u8) u64 {
    var r = plan_mod.Reader.openFile(allocator, io, plan_path) catch return 0;
    defer r.deinit();
    if (!r.plan.config.gpu) return 0;
    return r.plan.vram_resident_bytes;
}

/// Ensure a service is online: reuse an existing one, else spawn and wait.
/// `cold_bytes` is the plan's VRAM-resident size (0 for CPU); it only extends the
/// wait deadline, never the CPU path.
pub fn ensureOnline(
    allocator: std.mem.Allocator,
    io: Io,
    alias: []const u8,
    sock_path: []const u8,
    log_path: []const u8,
    opts: Options,
    cold_bytes: u64,
) !void {
    if (ping(io, sock_path)) return;
    try spawn(allocator, io, alias, log_path, opts);
    if (!waitOnline(io, sock_path, cold_bytes)) return error.ServiceTimeout;
}

/// Poll `ping` until the plan's socket answers or the (resident-scaled) deadline
/// elapses.
pub fn waitOnline(io: Io, sock_path: []const u8, cold_bytes: u64) bool {
    const deadline = spawnDeadlineMs(cold_bytes);
    var waited: usize = 0;
    while (waited < deadline) : (waited += poll_ms) {
        if (ping(io, sock_path)) return true;
        std.Io.sleep(io, std.Io.Duration.fromMilliseconds(poll_ms), .real) catch {};
    }
    return ping(io, sock_path);
}

/// Stop the plan's service: `QUIT` (fallback `SIGTERM` by pid) and remove the
/// socket/pid state. A no-op when no service is present.
pub fn stop(io: Io, key: []const u8) void {
    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const sp = store.serviceSocketPath(&sbuf, key) catch return;
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pp = store.servicePidPath(&pbuf, key) catch return;

    if (!quit(io, sp)) {
        if (readPid(io, pp)) |pid| std.posix.kill(pid, std.posix.SIG.TERM) catch {};
    }
    std.Io.sleep(io, std.Io.Duration.fromMilliseconds(poll_ms), .real) catch {};
    Io.Dir.cwd().deleteFile(io, sp) catch {};
    Io.Dir.cwd().deleteFile(io, pp) catch {};
}

/// Reap services whose plan is referenced by no registry alias: `SIGTERM` a
/// live `__serve` daemon and remove its socket/pid/log state. Returns the
/// number of services signalled. Aliased plans are never touched.
pub fn reapOrphanServices(allocator: std.mem.Allocator, io: Io) !usize {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const reg = store.loadRegistry(arena.allocator(), io);
    const entries: []const zrun.Entry = if (reg) |r| r.entries else &[_]zrun.Entry{};

    var dbuf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = store.runtimesDir(&dbuf) catch return 0;
    var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer dir.close(io);

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var kit = seen.keyIterator();
        while (kit.next()) |k| allocator.free(k.*);
        seen.deinit(allocator);
    }

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        const key = serviceKeyFromStateName(entry.name) orelse continue;
        if (key.len == 0) continue;
        if (seen.contains(key)) continue;
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const plan = std.fmt.bufPrint(&pbuf, "{s}.zplan", .{key}) catch continue;
        if (store.planInEntries(entries, plan)) continue;
        const owned = try allocator.dupe(u8, key);
        try seen.put(allocator, owned, {});
    }

    var reaped: usize = 0;
    var kit = seen.keyIterator();
    while (kit.next()) |k| {
        if (reapOne(io, k.*)) reaped += 1;
    }
    return reaped;
}

/// The plan key in a service state filename (`<key>.socket|.pid|.log`), or
/// null when the name is not service state.
fn serviceKeyFromStateName(name: []const u8) ?[]const u8 {
    for ([_][]const u8{ ".socket", ".pid", ".log" }) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return name[0 .. name.len - ext.len];
    }
    return null;
}

/// Best-effort: whether `pid` is a live zlm `__serve` process (Linux reads
/// `/proc/<pid>/cmdline`; other platforms fall back to the liveness probe).
fn isOurServeProcess(io: Io, pid: std.posix.pid_t) bool {
    std.posix.kill(pid, @enumFromInt(0)) catch return false;
    if (builtin.os.tag != .linux) return true;
    var pbuf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "/proc/{d}/cmdline", .{pid}) catch return false;
    var buf: [1024]u8 = undefined;
    const data = Io.Dir.cwd().readFile(io, path, &buf) catch return false;
    return std.mem.indexOf(u8, data, "__serve") != null;
}

/// `SIGTERM` the key's live service when it is ours, then remove its
/// socket/pid/log state. Returns whether a process was signalled.
fn reapOne(io: Io, key: []const u8) bool {
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pp = store.servicePidPath(&pbuf, key) catch return false;
    var signalled = false;
    if (readPid(io, pp)) |pid| {
        if (isOurServeProcess(io, pid)) {
            std.posix.kill(pid, std.posix.SIG.TERM) catch {};
            signalled = true;
        }
    }
    Io.Dir.cwd().deleteFile(io, pp) catch {};
    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (store.serviceSocketPath(&sbuf, key)) |sp| Io.Dir.cwd().deleteFile(io, sp) catch {} else |_| {}
    var lbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (store.serviceLogPath(&lbuf, key)) |lp| Io.Dir.cwd().deleteFile(io, lp) catch {} else |_| {}
    return signalled;
}

/// Leave the caller's session/process group (POSIX). The daemon is reparented
/// to init and no longer holds the controlling terminal.
pub fn detach() void {
    if (builtin.os.tag != .windows) {
        _ = std.posix.system.setsid();
    }
}

fn readPid(io: Io, path: []const u8) ?std.posix.pid_t {
    var buf: [64]u8 = undefined;
    const data = Io.Dir.cwd().readFile(io, path, &buf) catch return null;
    const trimmed = std.mem.trim(u8, data, " \t\r\n");
    return std.fmt.parseInt(std.posix.pid_t, trimmed, 10) catch null;
}

/// True when the daemon should exit because it no longer owns its state: the
/// socket path is missing or its inode differs, or the pid file is missing or
/// names another process.
fn shouldSelfExit(
    io: Io,
    sock_path: []const u8,
    sock_ino: std.posix.ino_t,
    pid_path: []const u8,
    self_pid: std.posix.pid_t,
) bool {
    const st = Io.Dir.cwd().statFile(io, sock_path, .{}) catch return true;
    if (st.inode != sock_ino) return true;
    const pid = readPid(io, pid_path) orelse return true;
    return pid != self_pid;
}

/// Whether the registry still references `plan_name`. A missing/unreadable
/// registry is treated as "still referenced" (conservative: never self-exit on
/// a transient read error).
fn planStillReferenced(io: Io, plan_name: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var reg = store.loadRegistry(arena.allocator(), io) orelse return true;
    defer reg.deinit();
    return store.planInEntries(reg.entries, plan_name);
}

var g_sock: [512]u8 = undefined;
var g_pid: [512]u8 = undefined;
var g_sock_len: usize = 0;
var g_pid_len: usize = 0;

fn cleanupPaths() void {
    if (g_sock_len > 0) _ = std.posix.system.unlink(@as([*:0]const u8, @ptrCast(&g_sock)));
    if (g_pid_len > 0) _ = std.posix.system.unlink(@as([*:0]const u8, @ptrCast(&g_pid)));
    g_sock_len = 0;
    g_pid_len = 0;
}

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    cleanupPaths();
    std.process.exit(0);
}

fn installSignals() void {
    if (builtin.is_test) return;
    if (builtin.os.tag != .windows) {
        const act = std.posix.Sigaction{
            .handler = .{ .handler = onSignal },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.TERM, &act, null);
        std.posix.sigaction(std.posix.SIG.INT, &act, null);
    }
}

fn writePid(io: Io, path: []const u8) !void {
    var buf: [32]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{d}\n", .{std.posix.system.getpid()});
    var f = try Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, text);
}

fn chmodSocket(io: Io, path: []const u8) void {
    if (builtin.os.tag == .windows) return;
    const perms: Io.File.Permissions = @enumFromInt(0o600);
    Io.Dir.cwd().setFilePermissions(io, path, perms, .{}) catch {};
}

/// Persist the connection's chat conversation to its session file.
fn saveChat(io: Io, allocator: std.mem.Allocator, id: []const u8, name: []const u8, model: []const u8, turns: []const session_mod.Session.Turn) void {
    if (id.len == 0) return;
    var list: std.ArrayList(sessions_mod.Turn) = .empty;
    defer list.deinit(allocator);
    for (turns) |t| {
        const role: []const u8 = switch (t.role) {
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
        list.append(allocator, .{ .role = role, .text = t.text }) catch return;
    }
    const now: u64 = @intCast(@divTrunc(Io.Clock.awake.now(io).nanoseconds, std.time.ns_per_s));
    sessions_mod.save(allocator, io, .{
        .id = id,
        .name = name,
        .model = model,
        .updated = now,
        .turns = list.items,
    }) catch {};
}

fn writeErr(w: anytype, msg: []const u8) !void {
    try w.interface.print("ERR\t{s}\n", .{msg});
    try w.interface.flush();
}

/// Index of the `n`th tab counted from the end of `s`, or null. Used by the
/// legacy `GENERATE` parsers to split the prompt from the fixed trailing fields.
fn tabFromEnd(s: []const u8, n: usize) ?usize {
    var count: usize = 0;
    var i: usize = s.len;
    while (i > 0) : (i -= 1) {
        if (s[i - 1] == '\t') {
            count += 1;
            if (count == n) return i - 1;
        }
    }
    return null;
}

/// Escape a free-form field (the `--soul` text) onto one protocol line: the
/// backslash and the whitespace that would break the line/tab framing.
fn escapeField(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (s) |c| switch (c) {
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        else => try out.append(allocator, c),
    };
    return out.toOwnedSlice(allocator);
}

/// Inverse of `escapeField`.
fn unescapeField(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len) {
            i += 1;
            const c: u8 = switch (s[i]) {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                else => blk: {
                    try out.append(allocator, '\\');
                    break :blk s[i];
                },
            };
            try out.append(allocator, c);
        } else try out.append(allocator, s[i]);
    }
    return out.toOwnedSlice(allocator);
}

const GenerateRequest = struct {
    prompt: []const u8,
    /// The `--soul` (system) text, already unescaped; empty = none.
    soul: []const u8 = "",
    max_tokens: usize,
    /// Separate budget for the reasoning (`think`) tokens; the answer keeps its
    /// full `max_tokens`. The legacy request shapes set the default 256.
    max_think: usize = 256,
    sampling: session_mod.Sampling,
    think: bool = false,
    stream: bool = false,
    show_think: bool = false,
    verbose: u8 = 0,
};

/// Parse the body of a `GENERATE` line. The canonical shape is
/// `max_tokens\ttemperature\ttop_k\ttop_p\tthink\tstream\tshow_think\tverbose\tprompt`, where the
/// prompt is the trailing field and may contain tabs. For compatibility the
/// previous `prompt\tmax_tokens\t...` and the older greedy `prompt\tmax_tokens`
/// are also accepted (both default think and stream off). The canonical and
/// previous shapes are mirror images, so a request whose leading fields all
/// parse as fixed numerics is read as the canonical shape; a legacy prompt that
/// looks exactly like the fixed fields is therefore ambiguous. Null on a
/// malformed line.
fn parseGenerate(allocator: std.mem.Allocator, rest: []const u8) ?GenerateRequest {
    return parseGenerateCanonical(allocator, rest) orelse
        parseGenerateExtended(rest) orelse
        parseGenerateGreedy(rest);
}

/// The canonical `max_tokens\ttemperature\ttop_k\ttop_p\tthink\tstream\tshow_think\tverbose\tmax_think\tsoul\tprompt`
/// shape. The nine leading fields are tab-free (the soul is escaped, see
/// `escapeField`), so the prompt is the raw remaining tail and is preserved
/// byte-for-byte (leading and trailing tabs included).
fn parseGenerateCanonical(allocator: std.mem.Allocator, rest: []const u8) ?GenerateRequest {
    var fields: [9][]const u8 = undefined;
    var start: usize = 0;
    for (&fields) |*field| {
        const rel = std.mem.indexOfScalar(u8, rest[start..], '\t') orelse return null;
        field.* = rest[start .. start + rel];
        start += rel + 1;
    }
    const max_tokens = std.fmt.parseInt(usize, fields[0], 10) catch return null;
    var sampling = session_mod.Sampling{};
    sampling.temperature = std.fmt.parseFloat(f32, fields[1]) catch return null;
    sampling.top_k = std.fmt.parseInt(u32, fields[2], 10) catch return null;
    sampling.top_p = std.fmt.parseFloat(f32, fields[3]) catch return null;
    const think_flag = std.fmt.parseInt(u8, fields[4], 10) catch return null;
    const stream_flag = std.fmt.parseInt(u8, fields[5], 10) catch return null;
    const show_think_flag = std.fmt.parseInt(u8, fields[6], 10) catch return null;
    const verbose = std.fmt.parseInt(u8, fields[7], 10) catch return null;
    const max_think = std.fmt.parseInt(usize, fields[8], 10) catch return null;
    // The soul field follows the nine numerics; the prompt is everything after.
    const soul_tab = std.mem.indexOfScalar(u8, rest[start..], '\t') orelse return null;
    const soul_raw = rest[start .. start + soul_tab];
    const soul = unescapeField(allocator, soul_raw) catch return null;
    return .{
        .prompt = rest[start + soul_tab + 1 ..],
        .soul = soul,
        .max_tokens = max_tokens,
        .max_think = max_think,
        .sampling = sampling,
        .think = think_flag != 0,
        .stream = stream_flag != 0,
        .show_think = show_think_flag != 0,
        .verbose = verbose,
    };
}

/// The previous `prompt\tmax_tokens\ttemperature\ttop_k\ttop_p` shape.
fn parseGenerateExtended(rest: []const u8) ?GenerateRequest {
    const tab = tabFromEnd(rest, 4) orelse return null;
    var it = std.mem.splitScalar(u8, rest[tab + 1 ..], '\t');
    const max_tokens = std.fmt.parseInt(usize, it.next() orelse "", 10) catch return null;
    var sampling = session_mod.Sampling{};
    sampling.temperature = std.fmt.parseFloat(f32, it.next() orelse "") catch 0;
    sampling.top_k = std.fmt.parseInt(u32, it.next() orelse "", 10) catch 0;
    sampling.top_p = std.fmt.parseFloat(f32, it.next() orelse "") catch 1;
    return .{ .prompt = rest[0..tab], .max_tokens = max_tokens, .sampling = sampling };
}

/// The original greedy `prompt\tmax_tokens` shape (prompt may contain tabs).
fn parseGenerateGreedy(rest: []const u8) ?GenerateRequest {
    const tab = tabFromEnd(rest, 1) orelse return null;
    const max_tokens = std.fmt.parseInt(usize, rest[tab + 1 ..], 10) catch return null;
    return .{ .prompt = rest[0..tab], .max_tokens = max_tokens, .sampling = .{} };
}

/// Streams decoded chunks to the client as `DATA\t<len>\n<bytes>` frames,
/// counting the bytes for the final `OK` header.
const StreamSink = struct {
    interface: *std.Io.Writer,
    total: *usize,

    fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *StreamSink = @ptrCast(@alignCast(ctx));
        try self.interface.print("DATA\t{d}\n", .{bytes.len});
        try self.interface.writeAll(bytes);
        try self.interface.flush();
        self.total.* += bytes.len;
    }
};

/// Write the `OK\t<byte_len>\t<metrics...>` reply header (no body). Shared by
/// the batch and streaming paths; the stream path passes the byte count it
/// emitted instead of the text length.
fn writeOkHeader(io: Io, w: anytype, byte_len: usize, stats: session_mod.GenerateStats, gen_ns: i96) !void {
    const pr = stats.profile;
    try w.interface.print("OK\t{d}\t{d:.3}\t{d:.3}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d:.3}\n", .{
        byte_len,
        nsToMs(stats.prefill_ns),
        nsToMs(stats.decode_ns),
        stats.decode_tokens,
        peakRssKb(io),
        stats.prefill_tokens,
        stats.cache_matched,
        pr.matvec_ns,
        pr.matmul_ns,
        pr.attention_ns,
        pr.norm_ns,
        pr.activation_ns,
        pr.elementwise_ns,
        pr.rope_ns,
        pr.dequant_ns,
        pr.residency_ns,
        pr.dispatches,
        pr.weight_bytes,
        pr.parts_sum,
        pr.serial_dispatches,
        pr.serial_ns,
        pr.matvec_count,
        pr.matvec_parts_sum,
        nsToMs(@intCast(gen_ns)),
    });
}

/// Write a byte string with tab/newline/backslash escapes so a decoded piece
/// stays on one `TRACE` line. Diagnostic only.
fn writeEscaped(w: *std.Io.Writer, s: []const u8) !void {
    var buf: [2]u8 = undefined;
    for (s) |c| {
        const out: []const u8 = switch (c) {
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            '\\' => "\\\\",
            else => blk: {
                buf[0] = c;
                break :blk buf[0..1];
            },
        };
        try w.writeAll(out);
    }
}

/// Emit the `TRACE\t<line>\n` diagnostic frames for `level` from the runtime's
/// captured `Verbose` trace. Never feeds a decision; the caller prints them to
/// stderr. Emitted before the final `OK` header (after any `DATA` frames).
fn writeTrace(w: anytype, runtime: anytype, level: u8) !void {
    const tr = runtime.verboseTrace();
    // `min` (1) prints only the phase line (client-side); the per-layer detail
    // starts at `info` (2).
    if (level >= 2) {
        for (tr.layer_ns, 0..) |ns, i| {
            if (ns == 0) continue;
            try w.interface.print("{s}layer {d}: {d:.3} ms\n", .{ trace_prefix, i, nsToMs(ns) });
        }
    }
    if (level >= 2) {
        for (tr.tokens, 0..) |tok, i| {
            try w.interface.print("{s}tok {d} id={d} logit={d:.3} piece=", .{ trace_prefix, i, tok.id, tok.logit });
            try writeEscaped(&w.interface, tok.piece);
            try w.interface.writeAll("\n");
        }
    }
    if (level >= 3) {
        if (tr.prompt_ids.len > 0) {
            try w.interface.writeAll(trace_prefix ++ "prompt ids:");
            for (tr.prompt_ids) |id| try w.interface.print(" {d}", .{id});
            try w.interface.writeAll("\n");
        }
        if (tr.think.len > 0) {
            var lines = std.mem.splitScalar(u8, tr.think, '\n');
            while (lines.next()) |ln| {
                try w.interface.print("{s}think: {s}\n", .{ trace_prefix, ln });
            }
        }
    }
}

fn handleClient(io: Io, runtime: anytype, stream: *Io.net.Stream, last_ns: *i96, build_fp: u64) !bool {
    var rbuf: [max_line]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var wbuf: [4096]u8 = undefined;
    var w = stream.writer(io, &wbuf);

    // Per-connection chat conversation (the REPL keeps one connection open).
    var chat_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer chat_arena.deinit();
    var chat_turns: std.ArrayList(session_mod.Session.Turn) = .empty;
    var chat_id: []const u8 = "";
    var chat_name: []const u8 = "";
    var chat_model: []const u8 = "";

    while (true) {
        const maybe = r.interface.takeDelimiter('\n') catch |err| return err;
        const line = maybe orelse return false;
        last_ns.* = Io.Clock.awake.now(io).nanoseconds;
        if (std.mem.eql(u8, line, ping_request)) {
            try w.interface.print("{s}\t{x}\n", .{ pong_reply, build_fp });
            try w.interface.flush();
        } else if (std.mem.eql(u8, line, quit_request)) {
            return true;
        } else if (std.mem.eql(u8, line, chat_reset_request)) {
            chat_turns.clearRetainingCapacity();
            try w.interface.writeAll("OK\t0\n");
            try w.interface.flush();
        } else if (std.mem.startsWith(u8, line, session_prefix)) {
            const ca = chat_arena.allocator();
            var sit = std.mem.splitScalar(u8, line[session_prefix.len..], '\t');
            const id = sit.next() orelse "";
            const name = sit.next() orelse "";
            const model = sit.next() orelse "";
            chat_id = ca.dupe(u8, id) catch "";
            chat_name = ca.dupe(u8, name) catch "";
            chat_model = ca.dupe(u8, model) catch "";
            chat_turns.clearRetainingCapacity();
            if (sessions_mod.load(ca, io, chat_id)) |rec| {
                for (rec.turns) |t| {
                    const role: @TypeOf(chat_turns.items[0].role) = if (std.mem.eql(u8, t.role, "user")) .user else .assistant;
                    chat_turns.append(ca, .{ .role = role, .text = t.text }) catch continue;
                }
                if (rec.name.len > 0) chat_name = ca.dupe(u8, rec.name) catch chat_name;
            }
            try w.interface.writeAll("OK\t0\n");
            try w.interface.flush();
        } else if (std.mem.startsWith(u8, line, chat_seed_prefix)) {
            const ca = chat_arena.allocator();
            // A seed replaces the conversation (a saved session).
            chat_turns.clearRetainingCapacity();
            var sit = std.mem.splitScalar(u8, line[chat_seed_prefix.len..], 0x1e);
            while (sit.next()) |part| {
                if (part.len < 1) continue;
                const role: @TypeOf(chat_turns.items[0].role) = if (part[0] == 'u') .user else .assistant;
                const text = unescapeField(ca, part[1..]) catch continue;
                chat_turns.append(ca, .{ .role = role, .text = text }) catch continue;
            }
            try w.interface.writeAll("OK\t0\n");
            try w.interface.flush();
        } else if (std.mem.startsWith(u8, line, chat_prefix)) {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const req = parseGenerate(arena.allocator(), line[chat_prefix.len..]) orelse {
                try writeErr(&w, "bad chat request");
                continue;
            };
            // The conversation owns its turn texts for the connection's life.
            const ca = chat_arena.allocator();
            const user_text = ca.dupe(u8, req.prompt) catch {
                try writeErr(&w, "out of memory");
                continue;
            };
            chat_turns.append(ca, .{ .role = .user, .text = user_text }) catch {
                try writeErr(&w, "out of memory");
                continue;
            };
            const seq = runtime.maxSeq();
            const reserve = req.max_tokens + req.max_think;
            const budget: usize = if (seq == 0) 0 else if (seq > reserve) seq - reserve else 1;
            const formatted = runtime.formatConversation(arena.allocator(), chat_turns.items, req.think, req.soul, budget) catch {
                try writeErr(&w, "format failed");
                continue;
            };
            const gen_t0 = Io.Clock.awake.now(io).nanoseconds;
            var reply_text: []const u8 = "";
            if (req.stream) {
                var total: usize = 0;
                var sink_ctx = StreamSink{ .interface = &w.interface, .total = &total };
                const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = StreamSink.write };
                const text = runtime.generate(arena.allocator(), formatted, req.max_tokens, req.max_think, req.sampling, req.think, req.show_think, req.verbose, "", true, sink) catch {
                    try writeErr(&w, "generation failed");
                    continue;
                };
                reply_text = text;
                try chat_turns.append(ca, .{ .role = .assistant, .text = ca.dupe(u8, reply_text) catch "" });
                saveChat(io, ca, chat_id, chat_name, chat_model, chat_turns.items);
                const gen_ns = Io.Clock.awake.now(io).nanoseconds - gen_t0;
                if (req.verbose > 0) try writeTrace(&w, runtime, req.verbose);
                try writeOkHeader(io, &w, total, runtime.generateStats(), gen_ns);
                try w.interface.flush();
            } else {
                const text = runtime.generate(arena.allocator(), formatted, req.max_tokens, req.max_think, req.sampling, req.think, req.show_think, req.verbose, "", true, null) catch {
                    try writeErr(&w, "generation failed");
                    continue;
                };
                reply_text = text;
                try chat_turns.append(ca, .{ .role = .assistant, .text = ca.dupe(u8, reply_text) catch "" });
                saveChat(io, ca, chat_id, chat_name, chat_model, chat_turns.items);
                const gen_ns = Io.Clock.awake.now(io).nanoseconds - gen_t0;
                if (req.verbose > 0) try writeTrace(&w, runtime, req.verbose);
                try writeOkHeader(io, &w, text.len, runtime.generateStats(), gen_ns);
                try w.interface.writeAll(text);
                try w.interface.flush();
            }
        } else if (std.mem.startsWith(u8, line, generate_prefix)) {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const req = parseGenerate(arena.allocator(), line[generate_prefix.len..]) orelse {
                try writeErr(&w, "bad generate request");
                continue;
            };
            // Wall-clock of the whole request: tokenize + prefill + decode +
            // detokenize. This is the complete "generation" time the client
            // reports (prefill + decode miss tokenization and sampling).
            const gen_t0 = Io.Clock.awake.now(io).nanoseconds;
            if (req.stream) {
                var total: usize = 0;
                var sink_ctx = StreamSink{ .interface = &w.interface, .total = &total };
                const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = StreamSink.write };
                _ = runtime.generate(arena.allocator(), req.prompt, req.max_tokens, req.max_think, req.sampling, req.think, req.show_think, req.verbose, req.soul, false, sink) catch {
                    try writeErr(&w, "generation failed");
                    continue;
                };
                const gen_ns = Io.Clock.awake.now(io).nanoseconds - gen_t0;
                if (req.verbose > 0) try writeTrace(&w, runtime, req.verbose);
                try writeOkHeader(io, &w, total, runtime.generateStats(), gen_ns);
                try w.interface.flush();
            } else {
                const text = runtime.generate(arena.allocator(), req.prompt, req.max_tokens, req.max_think, req.sampling, req.think, req.show_think, req.verbose, req.soul, false, null) catch {
                    try writeErr(&w, "generation failed");
                    continue;
                };
                const gen_ns = Io.Clock.awake.now(io).nanoseconds - gen_t0;
                if (req.verbose > 0) try writeTrace(&w, runtime, req.verbose);
                try writeOkHeader(io, &w, text.len, runtime.generateStats(), gen_ns);
                try w.interface.writeAll(text);
                try w.interface.flush();
            }
        } else {
            try writeErr(&w, "unknown request");
        }
    }
}

/// Bind the plan socket, record the pid, and serve requests until `QUIT` or a
/// termination signal. The socket/pid are removed on exit.
pub fn serve(
    io: Io,
    runtime: anytype,
    sock_path: []const u8,
    pid_path: []const u8,
    /// The plan artifact filename (`<image>_<fp>.zplan`) this service owns. When
    /// non-empty the daemon periodically checks that the registry still
    /// references it and self-exits when orphaned; empty (tests) skips the check.
    plan_name: []const u8,
    opts: Options,
) !void {
    if (sock_path.len >= g_sock.len or pid_path.len >= g_pid.len) return error.NameTooLong;
    runtime.setResidencyState(opts.state);
    // Persist the prefix cache (compact base + journal into the `.zpc`) on exit,
    // so the daemon's appends survive a restart.
    defer runtime.writeCacheFile() catch {};

    var hot_active = opts.state == .hot;
    const hot_ttl: ?i96 = if (hot_active) blk: {
        const m = opts.hot_minutes orelse break :blk null;
        break :blk minutesToNs(m);
    } else null;
    const timeout_ns: ?i96 = resolveTimeoutNs(opts.timeout_minutes);
    var last_ns = Io.Clock.awake.now(io).nanoseconds;

    Io.Dir.cwd().deleteFile(io, sock_path) catch {};
    const ua = try Io.net.UnixAddress.init(sock_path);
    var listener = try ua.listen(io, .{});
    defer listener.deinit(io);
    chmodSocket(io, sock_path);
    writePid(io, pid_path) catch {};
    const self_pid = std.posix.system.getpid();
    const self_ino = (Io.Dir.cwd().statFile(io, sock_path, .{}) catch return error.SocketStat).inode;
    // The daemon's own build fingerprint, reported in `PING` so a client from a
    // different build restarts it instead of reusing a daemon whose code it may
    // not match.
    const build_fp = build_fingerprint;

    @memcpy(g_sock[0..sock_path.len], sock_path);
    g_sock[sock_path.len] = 0;
    g_sock_len = sock_path.len;
    @memcpy(g_pid[0..pid_path.len], pid_path);
    g_pid[pid_path.len] = 0;
    g_pid_len = pid_path.len;
    installSignals();
    defer cleanupPaths();

    var orphan_ticks: u32 = 0;
    while (true) {
        var fds = [_]std.posix.pollfd{.{
            .fd = listener.socket.handle,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const ready = std.posix.poll(&fds, idle_poll_ms) catch 0;
        if (ready == 0) {
            if (shouldSelfExit(io, sock_path, self_ino, pid_path, self_pid)) return;
            // Orphan check: the daemon's owner is the registry, not a process
            // (it is detached). When no alias references its plan anymore it has
            // been superseded (e.g. re-tuned) and must release the model. Polled
            // less often than the state machine since it reads the manifest.
            if (plan_name.len > 0) {
                orphan_ticks += 1;
                if (orphan_ticks >= orphan_check_polls) {
                    orphan_ticks = 0;
                    if (!planStillReferenced(io, plan_name)) return;
                }
            }
            // Flush a dirty prefix cache on idle (the io is live here, unlike a
            // shutdown).
            if (runtime.cacheDirty()) runtime.writeCacheFile() catch {};
            const now_ns = Io.Clock.awake.now(io).nanoseconds;
            switch (idleAction(now_ns - last_ns, hot_ttl, timeout_ns, hot_active)) {
                .quit => return,
                .demote => {
                    runtime.setResidencyState(.warm);
                    hot_active = false;
                },
                .none => {},
            }
            continue;
        }

        var stream = listener.accept(io) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => continue,
        };
        const quit_now = handleClient(io, runtime, &stream, &last_ns, build_fp) catch false;
        stream.close(io);
        if (quit_now) return;
    }
}

const FakeRuntime = struct {
    fn generate(_: *FakeRuntime, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize, max_think: usize, sampling: session_mod.Sampling, think: bool, show_think: bool, verbose: u8, soul: []const u8, raw: bool, sink: ?session_mod.Sink) ![]u8 {
        _ = max_think;
        _ = soul;
        _ = raw;
        _ = sampling;
        _ = show_think;
        _ = verbose;
        const text = try std.fmt.allocPrint(allocator, "echo:{s}:{d}:{d}", .{ prompt, max_tokens, @intFromBool(think) });
        if (sink) |s| try s.write(s.ctx, text);
        return text;
    }

    fn maxSeq(_: *FakeRuntime) u32 {
        return 0;
    }

    fn formatConversation(_: *FakeRuntime, allocator: std.mem.Allocator, turns: []const session_mod.Session.Turn, think: bool, soul: []const u8, budget: usize) ![]u8 {
        _ = think;
        _ = soul;
        _ = budget;
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);
        for (turns) |turn| {
            try buf.appendSlice(allocator, turn.text);
            try buf.append(allocator, '|');
        }
        return buf.toOwnedSlice(allocator);
    }

    fn generateStats(_: *FakeRuntime) session_mod.GenerateStats {
        return .{
            .prefill_ns = 1_500_000,
            .decode_ns = 2_500_000,
            .prefill_tokens = 2,
            .decode_tokens = 3,
        };
    }

    fn verboseTrace(_: *FakeRuntime) session_mod.Verbose {
        return .{};
    }

    fn setResidencyState(_: *FakeRuntime, _: residency.State) void {}

    fn writeCacheFile(_: *FakeRuntime) !void {}

    fn cacheDirty(_: *FakeRuntime) bool {
        return false;
    }
};

/// Test sink that accumulates streamed chunks.
const CollectSink = struct {
    allocator: std.mem.Allocator,
    list: *std.ArrayList(u8),

    fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *CollectSink = @ptrCast(@alignCast(ctx));
        try self.list.appendSlice(self.allocator, bytes);
    }
};

fn serveThread(sock_path: []const u8, pid_path: []const u8, fake: *FakeRuntime) void {
    serve(std.testing.io, fake, sock_path, pid_path, "", .{}) catch {};
}

fn tmpSocketPath(buf: []u8, tmp: *std.testing.TmpDir, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path[0..], name });
}

fn waitReady(io: Io, sock_path: []const u8) bool {
    var tries: usize = 0;
    while (tries < 200) : (tries += 1) {
        if (ping(io, sock_path)) return true;
        std.Io.sleep(io, std.Io.Duration.fromMilliseconds(5), .real) catch {};
    }
    return false;
}

test "protocol round-trip, liveness, and cleanup on QUIT" {
    const testing = std.testing;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = try tmpSocketPath(&sbuf, &tmp, "svc.socket");
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pid_path = try tmpSocketPath(&pbuf, &tmp, "svc.pid");

    try testing.expect(!ping(io, sock_path));

    var fake = FakeRuntime{};
    const thread = try std.Thread.spawn(.{}, serveThread, .{ sock_path, pid_path, &fake });
    try testing.expect(waitReady(io, sock_path));
    try testing.expect(ping(io, sock_path));
    try testing.expect(store.pathExists(io, sock_path));
    try testing.expect(store.pathExists(io, pid_path));

    const reply = try generate(testing.allocator, io, sock_path, "hi", 2, 256, 0, 0, 1, false, false, 0, "");
    defer testing.allocator.free(reply.text);
    try testing.expectEqualStrings("echo:hi:2:0", reply.text);

    const tabbed = try generate(testing.allocator, io, sock_path, "a\tb\tc", 5, 256, 0.7, 40, 0.9, true, false, 0, "");
    defer testing.allocator.free(tabbed.text);
    try testing.expectEqualStrings("echo:a\tb\tc:5:1", tabbed.text);
    const m = reply.metrics orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 3), m.decode_tokens);
    try testing.expectEqual(@as(usize, 2), m.prefill_tokens);
    try testing.expect(m.prefill_ms > 0);
    try testing.expect(m.decode_ms > 0);

    try testing.expect(quit(io, sock_path));
    thread.join();

    try testing.expect(!ping(io, sock_path));
    try testing.expect(!store.pathExists(io, sock_path));
    try testing.expect(!store.pathExists(io, pid_path));
}

test "streaming generate round-trip emits DATA frames then the metrics OK" {
    const testing = std.testing;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = try tmpSocketPath(&sbuf, &tmp, "stream.socket");
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pid_path = try tmpSocketPath(&pbuf, &tmp, "stream.pid");

    var fake = FakeRuntime{};
    const thread = try std.Thread.spawn(.{}, serveThread, .{ sock_path, pid_path, &fake });
    try testing.expect(waitReady(io, sock_path));

    var collected: std.ArrayList(u8) = .empty;
    defer collected.deinit(testing.allocator);
    var sink_ctx = CollectSink{ .allocator = testing.allocator, .list = &collected };
    const sink = session_mod.Sink{ .ctx = @ptrCast(&sink_ctx), .write = CollectSink.write };

    const m = try generateStream(testing.allocator, io, sock_path, "hi", 2, 256, 0, 0, 1, false, false, 0, "", sink);
    try testing.expectEqualStrings("echo:hi:2:0", collected.items);
    try testing.expectEqual(@as(usize, 3), m.decode_tokens);

    try testing.expect(quit(io, sock_path));
    thread.join();
}

test "cold-start deadline scales with resident bytes, base for CPU" {
    const testing = std.testing;
    // A CPU plan (or no resident set) keeps the historical deadline.
    try testing.expectEqual(spawn_timeout_ms, spawnDeadlineMs(0));
    // A resident set extends it by the estimated upload time.
    const resident: u64 = 1935 << 20;
    const scaled = spawnDeadlineMs(resident);
    try testing.expect(scaled > spawn_timeout_ms);
    try testing.expectEqual(spawn_timeout_ms + @as(usize, @intFromFloat(@as(f64, @floatFromInt(resident)) / (cold_upload_gbps * 1e6))), scaled);
}

test "idleAction demotes hot then quits on timeout" {
    try std.testing.expectEqual(IdleAction.none, idleAction(30, 60, null, true));
    try std.testing.expectEqual(IdleAction.demote, idleAction(60, 60, null, true));
    try std.testing.expectEqual(IdleAction.none, idleAction(60, 60, null, false));
    try std.testing.expectEqual(IdleAction.quit, idleAction(60, 60, 60, true));
    try std.testing.expectEqual(IdleAction.quit, idleAction(120, null, 60, false));
}

test "resolveTimeoutNs: default when unset, 0 opts out" {
    const testing = std.testing;
    try testing.expectEqual(minutesToNs(default_timeout_minutes), resolveTimeoutNs(null).?);
    try testing.expectEqual(minutesToNs(5), resolveTimeoutNs(5).?);
    try testing.expect(resolveTimeoutNs(0) == null);
}

test "minutesToNs and stateFromName" {
    try std.testing.expectEqual(@as(i96, 2 * std.time.ns_per_min), minutesToNs(2));
    try std.testing.expectEqual(residency.State.cold, stateFromName("cold"));
    try std.testing.expectEqual(residency.State.hot, stateFromName("hot"));
    try std.testing.expectEqual(residency.State.warm, stateFromName("nonsense"));
}

test "unknown request is rejected" {
    const testing = std.testing;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = try tmpSocketPath(&sbuf, &tmp, "svc.socket");
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pid_path = try tmpSocketPath(&pbuf, &tmp, "svc.pid");

    var fake = FakeRuntime{};
    const thread = try std.Thread.spawn(.{}, serveThread, .{ sock_path, pid_path, &fake });
    try testing.expect(waitReady(io, sock_path));

    var stream = try (try Io.net.UnixAddress.init(sock_path)).connect(io);
    var wbuf: [64]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    try w.interface.writeAll("NOPE\n");
    try w.interface.flush();
    var rbuf: [64]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    const maybe = try r.interface.takeDelimiter('\n');
    const line = maybe orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.startsWith(u8, line, "ERR\t"));
    stream.close(io);

    try testing.expect(quit(io, sock_path));
    thread.join();
}

test "shouldSelfExit detects missing or foreign socket/pid" {
    const testing = std.testing;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const sock_path = try tmpSocketPath(&sbuf, &tmp, "self.socket");
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pid_path = try tmpSocketPath(&pbuf, &tmp, "self.pid");

    try tmp.dir.writeFile(io, .{ .sub_path = "self.socket", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "self.pid", .data = "4242\n" });
    const ino = (try tmp.dir.statFile(io, "self.socket", .{})).inode;

    try testing.expect(!shouldSelfExit(io, sock_path, ino, pid_path, 4242));
    try testing.expect(shouldSelfExit(io, sock_path, ino + 1, pid_path, 4242));
    try testing.expect(shouldSelfExit(io, sock_path, ino, pid_path, 1234));

    try tmp.dir.deleteFile(io, "self.pid");
    try testing.expect(shouldSelfExit(io, sock_path, ino, pid_path, 4242));

    try tmp.dir.writeFile(io, .{ .sub_path = "self.pid", .data = "4242\n" });
    try tmp.dir.deleteFile(io, "self.socket");
    try testing.expect(shouldSelfExit(io, sock_path, ino, pid_path, 4242));
}

test "serviceKeyFromStateName strips state suffixes" {
    try std.testing.expectEqualStrings("img_ab", serviceKeyFromStateName("img_ab.socket").?);
    try std.testing.expectEqualStrings("img_ab", serviceKeyFromStateName("img_ab.pid").?);
    try std.testing.expectEqualStrings("img_ab", serviceKeyFromStateName("img_ab.log").?);
    try std.testing.expect(serviceKeyFromStateName("img_ab.zplan") == null);
    try std.testing.expect(serviceKeyFromStateName("manifest.zrun") == null);
}

test "parseHeader handles the old and extended replies" {
    const testing = std.testing;

    const old = try parseHeader("OK\t12");
    try testing.expectEqual(@as(usize, 12), old.len);
    try testing.expect(old.metrics == null);

    const ext = try parseHeader("OK\t12\t1.500\t2.500\t3\t4096\t7\t5");
    try testing.expectEqual(@as(usize, 12), ext.len);
    const m = ext.metrics orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(f64, 1.5), m.prefill_ms);
    try testing.expectEqual(@as(f64, 2.5), m.decode_ms);
    try testing.expectEqual(@as(usize, 3), m.decode_tokens);
    try testing.expectEqual(@as(u64, 4096), m.rss_kb);
    try testing.expectEqual(@as(usize, 7), m.prefill_tokens);
    try testing.expectEqual(@as(usize, 5), m.cache_matched);

    const partial = try parseHeader("OK\t9\t0.500\t0.250\t1");
    try testing.expectEqual(@as(usize, 9), partial.len);
    const pm = partial.metrics orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 0), pm.rss_kb);
    try testing.expectEqual(@as(usize, 0), pm.cache_matched);

    try testing.expectError(error.ProtocolError, parseHeader("ERR\tnope"));
}

test "parseGenerate reads the canonical, previous, and legacy shapes" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const canonical = parseGenerate(a, "8\t0.7\t40\t0.9\t1\t0\t0\t2\t5\t\ta\tb\tc") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("a\tb\tc", canonical.prompt);
    try testing.expectEqual(@as(usize, 8), canonical.max_tokens);
    try testing.expectEqual(@as(f32, 0.7), canonical.sampling.temperature);
    try testing.expectEqual(@as(u32, 40), canonical.sampling.top_k);
    try testing.expectEqual(@as(f32, 0.9), canonical.sampling.top_p);
    try testing.expect(canonical.think);
    try testing.expect(!canonical.stream);
    try testing.expect(!canonical.show_think);
    try testing.expectEqual(@as(u8, 2), canonical.verbose);
    try testing.expectEqual(@as(usize, 5), canonical.max_think);
    try testing.expectEqualStrings("", canonical.soul);

    const soul = parseGenerate(a, "4\t0\t0\t1\t0\t0\t0\t0\t7\thello\\nworld\tx") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("hello\nworld", soul.soul);
    try testing.expectEqualStrings("x", soul.prompt);

    const canonical_stream = parseGenerate(a, "8\t0.7\t40\t0.9\t0\t1\t1\t3\t5\t\ta\tb\tc") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("a\tb\tc", canonical_stream.prompt);
    try testing.expect(!canonical_stream.think);
    try testing.expect(canonical_stream.stream);
    try testing.expect(canonical_stream.show_think);
    try testing.expectEqual(@as(u8, 3), canonical_stream.verbose);

    const empty = parseGenerate(a, "4\t0\t0\t1\t0\t0\t0\t0\t0\t\t") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("", empty.prompt);
    try testing.expectEqual(@as(usize, 4), empty.max_tokens);
    try testing.expectEqual(@as(f32, 1), empty.sampling.top_p);
    try testing.expectEqual(@as(u8, 0), empty.verbose);

    const leading_tab = parseGenerate(a, "4\t0\t0\t1\t0\t0\t0\t0\t0\t\t\tx") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("\tx", leading_tab.prompt);

    const ext = parseGenerate(a, "a\tb\t8\t0.7\t40\t0.9") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("a\tb", ext.prompt);
    try testing.expectEqual(@as(usize, 8), ext.max_tokens);
    try testing.expectEqual(@as(f32, 0.7), ext.sampling.temperature);
    try testing.expectEqual(@as(u32, 40), ext.sampling.top_k);
    try testing.expectEqual(@as(f32, 0.9), ext.sampling.top_p);
    try testing.expect(!ext.think);

    const legacy = parseGenerate(a, "hello world\t16") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("hello world", legacy.prompt);
    try testing.expectEqual(@as(usize, 16), legacy.max_tokens);
    try testing.expectEqual(@as(f32, 0), legacy.sampling.temperature);
    try testing.expectEqual(@as(u32, 0), legacy.sampling.top_k);
    try testing.expectEqual(@as(f32, 1), legacy.sampling.top_p);
    try testing.expect(!legacy.think);

    const tabbed_legacy = parseGenerate(a, "a\tb\tc\t16") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("a\tb\tc", tabbed_legacy.prompt);
    try testing.expectEqual(@as(usize, 16), tabbed_legacy.max_tokens);

    try testing.expect(parseGenerate(a, "nope") == null);
    try testing.expect(parseGenerate(a, "x\tbad") == null);
}

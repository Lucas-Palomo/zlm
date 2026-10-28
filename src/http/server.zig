//! The HTTP edge: an OpenAI-compatible server driven by a declarative config
//! (`src/http/config.zig`). Models load lazily on the first request that names
//! them; `GET /v1/models` lists the configured ids; `POST /v1/chat/completions`
//! maps the OpenAI `messages` to the runtime's multi-turn chat template.

const std = @import("std");
const session_mod = @import("../runtime/session.zig");
const store = @import("../image/store.zig");
const config = @import("config.zig");
const sessions = @import("../runtime/sessions.zig");
const tools_mod = @import("../architectures/core/tools.zig");
const Session = session_mod.Session;

const http_ok = "HTTP/1.1 200 OK\r\n";
const http_bad = "HTTP/1.1 400 Bad Request\r\n";
const max_body = 16 * 1024 * 1024;

const Context = struct {
    allocator: std.mem.Allocator,
    cfg: config.Config,
    /// Metadata (resolved alias, image name) lives as long as the server.
    meta: std.heap.ArenaAllocator,
    sessions: std.StringHashMapUnmanaged(*Session) = .empty,

    fn find(self: *Context, requested: []const u8) ?config.Model {
        for (self.cfg.models) |m| if (std.mem.eql(u8, m.id, requested)) return m;
        for (self.cfg.models) |m| if (std.mem.eql(u8, m.id, self.cfg.default)) return m;
        return null;
    }

    fn session(self: *Context, io: std.Io, m: config.Model) !*Session {
        if (self.sessions.get(m.id)) |s| return s;
        const rt = store.resolveRuntime(self.meta.allocator(), io, m.runtime) orelse return error.UnknownRuntime;
        var plan_buf: [std.fs.max_path_bytes]u8 = undefined;
        const plan_path = try rt.planPath(&plan_buf);
        const image = try self.meta.allocator().dupe(u8, rt.image);
        const path = try self.meta.allocator().dupe(u8, plan_path);
        const s = try self.allocator.create(Session);
        errdefer self.allocator.destroy(s);
        s.* = try Session.open(self.allocator, io, path, .{ .image_path = image });
        try self.sessions.put(self.allocator, m.id, s);
        return s;
    }
};

pub fn serve(io: std.Io, allocator: std.mem.Allocator, cfg: config.Config, port: u16) !void {
    var ctx = Context{
        .allocator = allocator,
        .cfg = cfg,
        .meta = std.heap.ArenaAllocator.init(allocator),
    };
    defer {
        var it = ctx.sessions.valueIterator();
        while (it.next()) |s| {
            s.*.deinit();
            allocator.destroy(s.*);
        }
        ctx.sessions.deinit(allocator);
        ctx.meta.deinit();
    }

    const addr = try std.Io.net.IpAddress.parse(cfg.host, port);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    while (true) {
        var stream = server.accept(io) catch continue;
        handle(io, &ctx, &stream) catch |err| {
            std.log.debug("http: {s}", .{@errorName(err)});
        };
        stream.close(io);
    }
}

const Request = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
};

fn readRequest(io: std.Io, allocator: std.mem.Allocator, stream: *std.Io.net.Stream) !Request {
    var rbuf: [8192]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    const request_line = (try r.interface.takeDelimiter('\n')) orelse return error.EmptyRequest;
    var it = std.mem.tokenizeAny(u8, request_line, " \r\t");
    const method = it.next() orelse return error.BadRequest;
    const path = it.next() orelse return error.BadRequest;

    var content_length: usize = 0;
    while (true) {
        const raw = (try r.interface.takeDelimiter('\n')) orelse return error.BadRequest;
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0) break;
        if (std.ascii.startsWithIgnoreCase(line, "content-length:")) {
            content_length = std.fmt.parseInt(usize, std.mem.trim(u8, line["content-length:".len..], " \t"), 10) catch 0;
            if (content_length > max_body) return error.RequestTooLarge;
        }
    }
    const body = try allocator.alloc(u8, content_length);
    errdefer allocator.free(body);
    if (content_length > 0) try r.interface.readSliceAll(body);
    return .{ .method = method, .path = path, .body = body };
}

fn handle(io: std.Io, ctx: *Context, stream: *std.Io.net.Stream) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const req = readRequest(io, a, stream) catch |err| {
        if (err == error.RequestTooLarge) {
            var ebuf: [8192]u8 = undefined;
            var ew = stream.writer(io, &ebuf);
            try ew.interface.writeAll("HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"request body too large\"}}");
            try ew.interface.flush();
        }
        return err;
    };

    var wbuf: [8192]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/models")) {
        try w.interface.print(http_ok ++ "Content-Type: application/json\r\nConnection: close\r\n\r\n", .{});
        try w.interface.writeAll("{\"object\":\"list\",\"data\":[");
        for (ctx.cfg.models, 0..) |m, i| {
            if (i > 0) try w.interface.writeAll(",");
            try w.interface.print("{{\"id\":\"{s}\",\"object\":\"model\",\"owned_by\":\"zlm\"}}", .{m.id});
        }
        try w.interface.writeAll("]}");
        try w.interface.flush();
        return;
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/sessions")) {
        const all = sessions.list(a, io) catch &.{};
        try w.interface.print(http_ok ++ "Content-Type: application/json\r\nConnection: close\r\n\r\n", .{});
        try w.interface.writeAll("{\"object\":\"list\",\"data\":[");
        for (all, 0..) |rec, i| {
            if (i > 0) try w.interface.writeAll(",");
            try w.interface.writeAll("{\"id\":");
            try writeJsonString(&w.interface, rec.id);
            try w.interface.writeAll(",\"name\":");
            try writeJsonString(&w.interface, rec.name);
            try w.interface.writeAll(",\"model\":");
            try writeJsonString(&w.interface, rec.model);
            try w.interface.print(",\"updated\":{d},\"turns\":{d}}}", .{ rec.updated, rec.turns.len });
        }
        try w.interface.writeAll("]}");
        try w.interface.flush();
        return;
    }
    if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/chat/completions")) {
        try completions(io, a, ctx, &w, req.body);
        try w.interface.flush();
        return;
    }
    try w.interface.writeAll("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"not found\"}}");
    try w.interface.flush();
}

fn completions(io: std.Io, allocator: std.mem.Allocator, ctx: *Context, w: anytype, body: []const u8) !void {
    const root = std.json.parseFromSliceLeaky(std.json.Value, allocator, body, .{}) catch {
        try writeJsonError(w, 400, "invalid JSON");
        return;
    };
    const obj = switch (root) {
        .object => |o| o,
        else => {
            try writeJsonError(w, 400, "expected a JSON object");
            return;
        },
    };

    const requested = blk: {
        if (obj.get("model")) |mv| switch (mv) {
            .string => |s| break :blk s,
            else => {},
        };
        break :blk "";
    };
    const model = ctx.find(requested) orelse {
        try writeJsonError(w, 400, "unknown model");
        return;
    };
    const session = ctx.session(io, model) catch |err| {
        try writeJsonError(w, 500, @errorName(err));
        return;
    };

    const messages = obj.get("messages") orelse {
        try writeJsonError(w, 400, "missing messages");
        return;
    };
    const arr = switch (messages) {
        .array => |a| a,
        else => {
            try writeJsonError(w, 400, "messages must be an array");
            return;
        },
    };

    var turns: std.ArrayList(Session.Turn) = .empty;
    var soul: []const u8 = model.soul;
    for (arr.items) |item| {
        const m = switch (item) {
            .object => |o| o,
            else => continue,
        };
        const role_s = jsonStringField(m, "role") orelse continue;
        const text = try contentText(allocator, m);
        if (std.mem.eql(u8, role_s, "system")) {
            if (text) |t| soul = t;
        } else if (std.mem.eql(u8, role_s, "user")) {
            if (text) |t| try turns.append(allocator, .{ .role = .user, .text = t });
        } else if (std.mem.eql(u8, role_s, "assistant")) {
            const calls = try parseMessageToolCalls(allocator, m.get("tool_calls"));
            if (text == null and calls.len == 0) continue;
            try turns.append(allocator, .{
                .role = .assistant,
                .text = text orelse "",
                .calls = calls,
            });
        } else if (std.mem.eql(u8, role_s, "tool")) {
            try turns.append(allocator, .{
                .role = .tool,
                .text = text orelse "",
                .tool_call_id = jsonStringField(m, "tool_call_id") orelse "",
                .name = jsonStringField(m, "name") orelse "",
            });
        }
    }
    if (turns.items.len == 0) {
        try writeJsonError(w, 400, "no user message");
        return;
    }
    const tool_list = try parseTools(allocator, obj.get("tools"));

    const temperature = jsonFloat(obj.get("temperature")) orelse model.temperature;
    const top_p = jsonFloat(obj.get("top_p")) orelse model.top_p;
    const top_k_raw = jsonUsize(obj.get("top_k")) orelse model.top_k;
    const top_k = std.math.cast(u32, top_k_raw) orelse {
        try writeJsonError(w, 400, "top_k out of range");
        return;
    };
    const max_tokens = jsonUsize(obj.get("max_tokens")) orelse
        jsonUsize(obj.get("max_completion_tokens")) orelse model.max_tokens;
    if (!std.math.isFinite(temperature) or !std.math.isFinite(top_p)) {
        try writeJsonError(w, 400, "sampling parameters must be finite");
        return;
    }
    const think = if (obj.get("think")) |_| jsonBool(obj.get("think")) else model.think;
    const stream_flag = jsonBool(obj.get("stream"));
    const session_id = blk: {
        if (obj.get("session")) |sv| switch (sv) {
            .string => |s| break :blk s,
            else => {},
        };
        break :blk "";
    };

    const seq = session.maxSeq();
    const reserve = max_tokens +| model.max_think;
    const budget: usize = if (seq == 0) 0 else if (seq > reserve) seq - reserve else 1;
    const formatted = session.formatConversationWithTools(allocator, turns.items, think, soul, tool_list, budget) catch |err| {
        try writeJsonError(w, 500, @errorName(err));
        return;
    };

    var sampling = session_mod.Sampling{};
    sampling.temperature = temperature;
    sampling.top_k = top_k;
    sampling.top_p = top_p;

    if (stream_flag) {
        try w.interface.print(http_ok ++ "Content-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", .{});
        if (tool_list.len == 0) {
            var sink = SseSink{ .interface = &w.interface };
            const reply_text = session.generate(allocator, formatted, max_tokens, model.max_think, sampling, think, false, 0, "", true, session_mod.Sink{ .ctx = @ptrCast(&sink), .write = SseSink.write }) catch |err| {
                try writeSseError(&w.interface, @errorName(err));
                return;
            };
            try writeSseFinish(&w.interface, "stop");
            try w.interface.writeAll("data: [DONE]\n\n");
            if (session_id.len > 0) saveSession(io, allocator, session_id, model.id, turns.items, reply_text);
            return;
        }
        // With tools the reply must be inspected before it is sent: the raw
        // `<|tool_call>` block is not user-visible content.
        const reply_text = session.generate(allocator, formatted, max_tokens, model.max_think, sampling, think, false, 0, "", true, null) catch |err| {
            try writeSseError(&w.interface, @errorName(err));
            return;
        };
        const parsed = session.extractToolCalls(allocator, reply_text) catch |err| {
            try writeSseError(&w.interface, @errorName(err));
            return;
        };
        const calls = try assignToolIds(allocator, parsed.calls);
        // A call is the actionable reply; pre-call text from small models is
        // often a hallucinated tool result, so it is not forwarded.
        if (calls.len == 0 and parsed.content.len > 0) try writeSseContent(&w.interface, parsed.content);
        for (calls, 0..) |call, i| try writeSseToolCall(&w.interface, i, call);
        try writeSseFinish(&w.interface, if (calls.len > 0) "tool_calls" else "stop");
        try w.interface.writeAll("data: [DONE]\n\n");
        if (session_id.len > 0) saveSession(io, allocator, session_id, model.id, turns.items, parsed.content);
        return;
    }

    const reply_text = session.generate(allocator, formatted, max_tokens, model.max_think, sampling, think, false, 0, "", true, null) catch |err| {
        try writeJsonError(w, 500, @errorName(err));
        return;
    };
    const parsed = session.extractToolCalls(allocator, reply_text) catch |err| {
        try writeJsonError(w, 500, @errorName(err));
        return;
    };
    const calls = try assignToolIds(allocator, parsed.calls);

    try w.interface.print(http_ok ++ "Content-Type: application/json\r\nConnection: close\r\n\r\n", .{});
    try w.interface.writeAll("{\"id\":\"chatcmpl-zlm\",\"object\":\"chat.completion\",\"model\":");
    try writeJsonString(&w.interface, model.id);
    try w.interface.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (calls.len == 0 and parsed.content.len > 0) {
        try writeJsonString(&w.interface, parsed.content);
    } else {
        try w.interface.writeAll("null");
    }
    if (calls.len > 0) {
        try w.interface.writeAll(",\"tool_calls\":[");
        for (calls, 0..) |call, i| {
            if (i > 0) try w.interface.writeAll(",");
            try writeToolCallJson(&w.interface, call);
        }
        try w.interface.writeAll("]");
    }
    try w.interface.writeAll("},\"finish_reason\":");
    try writeJsonString(&w.interface, if (calls.len > 0) "tool_calls" else "stop");
    const done = session.generateStats().decode_tokens;
    try w.interface.writeAll("}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":");
    try w.interface.print("{d}", .{done});
    try w.interface.writeAll(",\"total_tokens\":");
    try w.interface.print("{d}", .{done});
    try w.interface.writeAll("}}");
    if (session_id.len > 0) saveSession(io, allocator, session_id, model.id, turns.items, parsed.content);
}

fn jsonStringField(m: std.json.ObjectMap, name: []const u8) ?[]const u8 {
    const v = m.get(name) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn contentText(allocator: std.mem.Allocator, m: std.json.ObjectMap) !?[]const u8 {
    const v = m.get("content") orelse return null;
    return switch (v) {
        .string => |s| s,
        .array => |parts| try joinParts(allocator, parts),
        else => null,
    };
}

fn parseTools(allocator: std.mem.Allocator, value: ?std.json.Value) ![]const tools_mod.Tool {
    const v = value orelse return &.{};
    if (v != .array) return &.{};
    var list: std.ArrayList(tools_mod.Tool) = .empty;
    errdefer list.deinit(allocator);
    for (v.array.items) |item| {
        const o = switch (item) {
            .object => |o| o,
            else => continue,
        };
        const fv = o.get("function") orelse continue;
        if (fv != .object) continue;
        const f = fv.object;
        const name = jsonStringField(f, "name") orelse continue;
        try list.append(allocator, .{
            .name = name,
            .description = jsonStringField(f, "description") orelse "",
            .parameters = f.get("parameters"),
        });
    }
    return list.toOwnedSlice(allocator);
}

fn parseMessageToolCalls(allocator: std.mem.Allocator, value: ?std.json.Value) ![]const tools_mod.Call {
    const v = value orelse return &.{};
    if (v != .array) return &.{};
    var list: std.ArrayList(tools_mod.Call) = .empty;
    errdefer list.deinit(allocator);
    for (v.array.items) |item| {
        const o = switch (item) {
            .object => |o| o,
            else => continue,
        };
        const fv = o.get("function") orelse continue;
        if (fv != .object) continue;
        const f = fv.object;
        const name = jsonStringField(f, "name") orelse continue;
        const args: []const u8 = blk: {
            if (f.get("arguments")) |av| switch (av) {
                .string => |s| if (s.len > 0) break :blk s,
                .object, .array => break :blk try std.json.Stringify.valueAlloc(allocator, av, .{}),
                else => {},
            };
            break :blk "{}";
        };
        try list.append(allocator, .{
            .id = jsonStringField(o, "id") orelse "",
            .name = name,
            .arguments = args,
        });
    }
    return list.toOwnedSlice(allocator);
}

fn assignToolIds(allocator: std.mem.Allocator, calls: []tools_mod.Call) ![]const tools_mod.Call {
    for (calls, 0..) |*call, i| {
        if (call.id.len == 0) call.id = try std.fmt.allocPrint(allocator, "call_{d}", .{i + 1});
    }
    return calls;
}

fn writeToolCallJson(w: *std.Io.Writer, call: tools_mod.Call) !void {
    try w.writeAll("{\"id\":");
    try writeJsonString(w, call.id);
    try w.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
    try writeJsonString(w, call.name);
    try w.writeAll(",\"arguments\":");
    try writeJsonString(w, call.arguments);
    try w.writeAll("}}");
}

/// Persist a conversation (the request's turns + the assistant reply) under
/// `id`, so external tools can inspect/list it later.
fn saveSession(io: std.Io, allocator: std.mem.Allocator, id: []const u8, model: []const u8, turns: []const Session.Turn, reply: []const u8) void {
    var list: std.ArrayList(sessions.Turn) = .empty;
    defer list.deinit(allocator);
    for (turns) |t| {
        const role: []const u8 = switch (t.role) {
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
        list.append(allocator, .{ .role = role, .text = t.text }) catch return;
    }
    list.append(allocator, .{ .role = "assistant", .text = reply }) catch return;
    const now: u64 = @intCast(@divTrunc(std.Io.Clock.awake.now(io).nanoseconds, std.time.ns_per_s));
    sessions.save(allocator, io, .{ .id = id, .model = model, .updated = now, .turns = list.items }) catch {};
}

fn joinParts(allocator: std.mem.Allocator, parts: std.json.Array) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    for (parts.items) |part| {
        const o = switch (part) {
            .object => |o| o,
            else => continue,
        };
        if (o.get("text")) |tv| switch (tv) {
            .string => |s| try buf.appendSlice(allocator, s),
            else => {},
        };
    }
    return buf.toOwnedSlice(allocator);
}

fn jsonFloat(v: ?std.json.Value) ?f32 {
    const x = v orelse return null;
    return switch (x) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f32, s) catch null,
        else => null,
    };
}

fn jsonUsize(v: ?std.json.Value) ?usize {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| blk: {
            const max_f: f64 = @floatFromInt(std.math.maxInt(usize));
            if (!std.math.isFinite(f) or f < 0 or f >= max_f) break :blk null;
            break :blk @intFromFloat(f);
        },
        else => null,
    };
}

fn jsonBool(v: ?std.json.Value) bool {
    const x = v orelse return false;
    return switch (x) {
        .bool => |b| b,
        else => false,
    };
}

fn writeJsonError(w: anytype, status: u16, msg: []const u8) !void {
    const head = switch (status) {
        400 => http_bad,
        500 => "HTTP/1.1 500 Internal Server Error\r\n",
        else => http_bad,
    };
    try w.interface.print("{s}Content-Type: application/json\r\nConnection: close\r\n\r\n", .{head});
    try w.interface.writeAll("{\"error\":{\"message\":\"");
    try writeJsonChars(&w.interface, msg);
    try w.interface.writeAll("\"}}");
}

fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeAll("\"");
    try writeJsonChars(w, s);
    try w.writeAll("\"");
}

fn writeJsonChars(w: *std.Io.Writer, s: []const u8) !void {
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => {
            if (c < 0x20) {
                try w.print("\\u{x:0>4}", .{c});
            } else {
                try w.writeAll(&[_]u8{c});
            }
        },
    };
}

/// The terminal SSE frame. OpenAI clients (notably the AI SDK used by OpenCode)
/// reject a stream that ends without an explicit `finish_reason`.
pub fn writeSseFinish(w: *std.Io.Writer, reason: []const u8) !void {
    try w.writeAll("data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    try writeJsonString(w, reason);
    try w.writeAll("}]}\n\n");
}

fn writeSseContent(w: *std.Io.Writer, bytes: []const u8) !void {
    try w.writeAll("data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
    try writeJsonString(w, bytes);
    try w.writeAll("}}]}\n\n");
}

fn writeSseToolCall(w: *std.Io.Writer, index: usize, call: tools_mod.Call) !void {
    try w.writeAll("data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    try w.print("{d}", .{index});
    try w.writeAll(",\"id\":");
    try writeJsonString(w, call.id);
    try w.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
    try writeJsonString(w, call.name);
    try w.writeAll(",\"arguments\":");
    try writeJsonString(w, call.arguments);
    try w.writeAll("}}]}}]}\n\n");
}

fn writeSseError(w: *std.Io.Writer, msg: []const u8) !void {
    try w.writeAll("data: {\"error\":{\"message\":\"");
    try writeJsonChars(w, msg);
    try w.writeAll("\"}}\n\n");
}

const SseSink = struct {
    interface: *std.Io.Writer,

    fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *SseSink = @ptrCast(@alignCast(ctx));
        try writeSseContent(self.interface, bytes);
        try self.interface.flush();
    }
};

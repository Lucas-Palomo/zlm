//! Persistent chat sessions: one JSON record per conversation, keyed by a UUID,
//! under `$ZLM_HOME/sessions/<uuid>.json`. The CLI loads a session's turns
//! (seeding the service's conversation) and saves them after each turn, so a
//! conversation survives across `run`/`chat` invocations and service restarts.

const std = @import("std");
const store = @import("../image/store.zig");

pub const Turn = struct {
    role: []const u8,
    text: []const u8,
};

pub const Record = struct {
    id: []const u8,
    name: []const u8 = "",
    model: []const u8 = "",
    /// Unix seconds of the last update.
    updated: u64 = 0,
    turns: []Turn = &.{},
};

/// Session ids become file names (`sessions/<id>.json`); reject path
/// separators and traversal instead of letting a client escape the directory.
pub fn isValidId(id: []const u8) bool {
    if (id.len == 0 or id.len > 128) return false;
    if (std.mem.eql(u8, id, ".") or std.mem.eql(u8, id, "..")) return false;
    for (id) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_') return false;
    }
    return true;
}

/// Load `sessions/<id>.json` (owned by `allocator`), or null when absent.
pub fn load(allocator: std.mem.Allocator, io: std.Io, id: []const u8) ?Record {
    if (!isValidId(id)) return null;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = store.sessionPath(&buf, id) catch return null;
    var data_buf: [1 << 20]u8 = undefined;
    const data = std.Io.Dir.cwd().readFile(io, path, &data_buf) catch return null;
    return std.json.parseFromSliceLeaky(Record, allocator, data, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch null;
}

/// Write `rec` to `sessions/<id>.json`, creating the directory if needed.
pub fn save(allocator: std.mem.Allocator, io: std.Io, rec: Record) !void {
    if (!isValidId(rec.id)) return error.InvalidId;
    try store.ensureSessionsDir(io);
    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{\"id\":");
    try jsonString(w, rec.id);
    try w.writeAll(",\"name\":");
    try jsonString(w, rec.name);
    try w.writeAll(",\"model\":");
    try jsonString(w, rec.model);
    try w.print(",\"updated\":{d},\"turns\":[", .{rec.updated});
    for (rec.turns, 0..) |t, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"role\":");
        try jsonString(w, t.role);
        try w.writeAll(",\"text\":");
        try jsonString(w, t.text);
        try w.writeAll("}");
    }
    try w.writeAll("]}");
    const json = try aw.toOwnedSlice();
    defer allocator.free(json);

    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try store.sessionPath(&pbuf, rec.id);
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, json);
}

/// All sessions (owned by `allocator`), newest first.
pub fn list(allocator: std.mem.Allocator, io: std.Io) ![]Record {
    var dir = store.openSessions(io) catch return &.{};
    defer dir.close(io);
    var out: std.ArrayList(Record) = .empty;
    errdefer out.deinit(allocator);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        const id = entry.name[0 .. entry.name.len - ".json".len];
        if (load(allocator, io, id)) |rec| try out.append(allocator, rec);
    }
    const items = try out.toOwnedSlice(allocator);
    std.mem.sort(Record, items, {}, newerFirst);
    return items;
}

fn newerFirst(_: void, a: Record, b: Record) bool {
    return a.updated > b.updated;
}

/// Set the display name of `id`.
pub fn rename(allocator: std.mem.Allocator, io: std.Io, id: []const u8, name: []const u8) !void {
    var rec = load(allocator, io, id) orelse return error.NotFound;
    rec.name = name;
    try save(allocator, io, rec);
}

/// A short random session id (hex, 16 chars).
pub fn newId(io: std.Io, buf: *[16]u8) []const u8 {
    var raw: [8]u8 = undefined;
    std.Io.random(io, &raw);
    return std.fmt.bufPrint(buf, "{x}", .{std.mem.readInt(u64, &raw, .little)}) catch buf[0..0];
}

fn jsonString(w: anytype, s: []const u8) !void {
    try w.writeAll("\"");
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => if (c < 0x20) {
            try w.print("\\u{x:0>4}", .{c});
        } else {
            try w.writeAll(&[_]u8{c});
        },
    };
    try w.writeAll("\"");
}

test "session ids cannot escape the sessions directory" {
    const testing = std.testing;
    try testing.expect(isValidId("abc-123"));
    try testing.expect(isValidId("0123456789abcdef"));
    try testing.expect(!isValidId(""));
    try testing.expect(!isValidId(".."));
    try testing.expect(!isValidId("../escape"));
    try testing.expect(!isValidId("a/b"));
    try testing.expect(!isValidId("a\\b"));
}

test "session record round-trips through the JSON shape" {
    const testing = std.testing;
    var aw = std.Io.Writer.Allocating.init(testing.allocator);
    defer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{\"id\":");
    try jsonString(w, "abc");
    try w.writeAll(",\"name\":");
    try jsonString(w, "default");
    try w.writeAll(",\"model\":");
    try jsonString(w, "m");
    try w.print(",\"updated\":{d},\"turns\":[", .{7});
    try w.writeAll("{\"role\":");
    try jsonString(w, "user");
    try w.writeAll(",\"text\":");
    try jsonString(w, "hi \"there\"\n\x01");
    try w.writeAll("}]}");
    const json = try aw.toOwnedSlice();
    defer testing.allocator.free(json);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rec = try std.json.parseFromSliceLeaky(Record, arena.allocator(), json, .{});
    try testing.expectEqualStrings("abc", rec.id);
    try testing.expectEqualStrings("default", rec.name);
    try testing.expectEqual(@as(u64, 7), rec.updated);
    try testing.expectEqual(@as(usize, 1), rec.turns.len);
    try testing.expectEqualStrings("user", rec.turns[0].role);
    try testing.expectEqualStrings("hi \"there\"\n\x01", rec.turns[0].text);
}

//! The `zlm http` server config: a declarative list of models (runtime aliases)
//! with their per-model generation defaults and persona. The HTTP edge loads a
//! model lazily on the first request that names it.

const std = @import("std");
const store = @import("../image/store.zig");

pub const Config = struct {
    version: u32 = 1,
    host: []const u8 = "127.0.0.1",
    /// The model id used when a request omits `model` (or names an unknown one).
    default: []const u8 = "",
    models: []const Model = &.{},
};

pub const Model = struct {
    /// The OpenAI `model` id the client sends (`/v1/models` lists these).
    id: []const u8,
    /// The zlm runtime alias (a tuned plan).
    runtime: []const u8,
    /// The persona (system turn) for this model.
    soul: []const u8 = "",
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    think: bool = false,
    max_tokens: usize = 256,
    max_think: usize = 256,
};

pub fn parse(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Config {
    var buf: [1 << 20]u8 = undefined;
    const data = try std.Io.Dir.cwd().readFile(io, path, &buf);
    return std.json.parseFromSliceLeaky(Config, allocator, data, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

/// Build an empty config from the registry's tuned aliases (the `config`
/// bootstrap): one model per alias, defaults for everything else.
pub fn bootstrap(allocator: std.mem.Allocator, io: std.Io) !Config {
    var reg = store.loadRegistry(allocator, io) orelse return .{ .models = &.{} };
    defer reg.deinit();
    var models = try allocator.alloc(Model, reg.entries.len);
    for (reg.entries, 0..) |e, i| {
        models[i] = .{ .id = try allocator.dupe(u8, e.id), .runtime = try allocator.dupe(u8, e.id) };
    }
    return .{
        .default = if (models.len > 0) models[0].id else "",
        .models = models,
    };
}

/// Serialize the config as pretty JSON (the `config --output` payload).
pub fn write(allocator: std.mem.Allocator, cfg: Config) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("{{\n  \"version\": {d},\n  \"host\": ", .{cfg.version});
    try jsonString(w, cfg.host);
    try w.writeAll(",\n  \"default\": ");
    try jsonString(w, cfg.default);
    try w.writeAll(",\n  \"models\": [\n");
    for (cfg.models, 0..) |m, i| {
        try w.writeAll("    {\n      \"id\": ");
        try jsonString(w, m.id);
        try w.writeAll(",\n      \"runtime\": ");
        try jsonString(w, m.runtime);
        try w.writeAll(",\n      \"soul\": ");
        try jsonString(w, m.soul);
        try w.print(",\n      \"temperature\": {d},\n      \"top_k\": {d},\n      \"top_p\": {d},\n      \"think\": {s},\n      \"max_tokens\": {d},\n      \"max_think\": {d}\n    }}", .{
            m.temperature, m.top_k, m.top_p, if (m.think) "true" else "false", m.max_tokens, m.max_think,
        });
        if (i + 1 < cfg.models.len) try w.writeAll(",");
        try w.writeAll("\n");
    }
    try w.writeAll("  ]\n}\n");
    return aw.toOwnedSlice();
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

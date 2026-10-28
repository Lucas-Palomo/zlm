const std = @import("std");
const Io = std.Io;
const gguf = @import("../gguf/root.zig");
const format = @import("../gguf/format.zig");
const zmeta_reader = @import("zmeta_reader.zig");

pub const MAGIC = "ZMTA";
pub const VERSION: u32 = 1;
pub const NAME = "model.zmeta";

pub const Written = struct { bytes: u64, sha256: [32]u8 };

/// The GGUF metadata, re-encoded in zlm's own format. After compile the source
/// GGUF is gone, so this is the metadata store (arch, hyperparams, tokenizer).
pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, model: *const gguf.Gguf) !Written {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);

    const markers = markersFromModel(model);
    var count: u32 = 0;
    for (model.keys) |key| {
        if (model.metadata.get(key) == null) continue;
        count += 1;
    }
    if (markers.user_prefix != null) count += 1;
    if (markers.user_suffix != null) count += 1;
    if (markers.model_prefix != null) count += 1;
    try appendU32(allocator, &buf, count);
    for (model.keys) |key| {
        const value = model.metadata.get(key) orelse continue;
        try appendString(allocator, &buf, key);
        try appendValue(allocator, &buf, value);
    }
    if (markers.user_prefix) |s| {
        try appendString(allocator, &buf, zmeta_reader.CHAT_USER_PREFIX);
        try appendValue(allocator, &buf, .{ .string = s });
    }
    if (markers.user_suffix) |s| {
        try appendString(allocator, &buf, zmeta_reader.CHAT_USER_SUFFIX);
        try appendValue(allocator, &buf, .{ .string = s });
    }
    if (markers.model_prefix) |s| {
        try appendString(allocator, &buf, zmeta_reader.CHAT_MODEL_PREFIX);
        try appendValue(allocator, &buf, .{ .string = s });
    }

    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(buf.items, &sha, .{});

    var file = try dir.createFile(io, NAME, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, buf.items);
    return .{ .bytes = buf.items.len, .sha256 = sha };
}

/// The three markers that wrap one user turn, as resolved from the model.
/// A null field means the format was not recognized; the runtime then uses the
/// architecture's literal fallback.
pub const ChatMarkers = struct {
    user_prefix: ?[]const u8 = null,
    user_suffix: ?[]const u8 = null,
    model_prefix: ?[]const u8 = null,
};

/// Detect the chat markers from the vocab special-token strings. Deterministic
/// and independent of the (complex) `tokenizer.chat_template` Jinja source:
/// `<|turn>`/`<turn|>` win over the classic `<start_of_turn>`/`<end_of_turn>`;
/// any other vocabulary yields all-null.
pub fn deriveChatMarkers(tokens: []const format.MetadataValue) ChatMarkers {
    var has_turn_open = false;
    var has_turn_close = false;
    var has_start = false;
    var has_end = false;
    var has_im_start = false;
    var has_im_end = false;
    for (tokens) |item| {
        const s = switch (item) {
            .string => |x| x,
            else => continue,
        };
        if (std.mem.eql(u8, s, "<|turn>")) has_turn_open = true;
        if (std.mem.eql(u8, s, "<turn|>")) has_turn_close = true;
        if (std.mem.eql(u8, s, "<start_of_turn>")) has_start = true;
        if (std.mem.eql(u8, s, "<end_of_turn>")) has_end = true;
        if (std.mem.eql(u8, s, "<|im_start|>")) has_im_start = true;
        if (std.mem.eql(u8, s, "<|im_end|>")) has_im_end = true;
    }
    if (has_turn_open and has_turn_close) return .{
        .user_prefix = "<|turn>user\n",
        .user_suffix = "<turn|>\n",
        .model_prefix = "<|turn>model\n",
    };
    if (has_im_start and has_im_end) return .{
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "<|im_end|>\n",
        .model_prefix = "<|im_start|>assistant\n",
    };
    if (has_start and has_end) return .{
        .user_prefix = "<start_of_turn>user\n",
        .user_suffix = "<end_of_turn>\n",
        .model_prefix = "<start_of_turn>model\n",
    };
    return .{};
}

fn markersFromModel(model: *const gguf.Gguf) ChatMarkers {
    const arr = model.getArray("tokenizer.ggml.tokens") orelse return .{};
    return deriveChatMarkers(arr.items);
}

fn appendValue(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: format.MetadataValue) !void {
    try appendU32(allocator, l, @intFromEnum(v));
    switch (v) {
        .uint8 => |x| try l.append(allocator, x),
        .int8 => |x| try l.append(allocator, @bitCast(x)),
        .uint16 => |x| try appendU16(allocator, l, x),
        .int16 => |x| try appendU16(allocator, l, @bitCast(x)),
        .uint32 => |x| try appendU32(allocator, l, x),
        .int32 => |x| try appendU32(allocator, l, @bitCast(x)),
        .float32 => |x| try appendU32(allocator, l, @bitCast(x)),
        .bool => |x| try l.append(allocator, if (x) 1 else 0),
        .string => |s| try appendString(allocator, l, s),
        .uint64 => |x| try appendU64(allocator, l, x),
        .int64 => |x| try appendU64(allocator, l, @bitCast(x)),
        .float64 => |x| try appendU64(allocator, l, @bitCast(x)),
        .array => |a| {
            try appendU32(allocator, l, @intFromEnum(a.elem));
            try appendU64(allocator, l, a.items.len);
            for (a.items) |item| try appendElem(allocator, l, item);
        },
    }
}

/// Encode just the payload of a value (no type tag) — used for array items.
fn appendElem(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: format.MetadataValue) !void {
    switch (v) {
        .uint8 => |x| try l.append(allocator, x),
        .int8 => |x| try l.append(allocator, @bitCast(x)),
        .uint16 => |x| try appendU16(allocator, l, x),
        .int16 => |x| try appendU16(allocator, l, @bitCast(x)),
        .uint32 => |x| try appendU32(allocator, l, x),
        .int32 => |x| try appendU32(allocator, l, @bitCast(x)),
        .float32 => |x| try appendU32(allocator, l, @bitCast(x)),
        .bool => |x| try l.append(allocator, if (x) 1 else 0),
        .string => |s| try appendString(allocator, l, s),
        .uint64 => |x| try appendU64(allocator, l, x),
        .int64 => |x| try appendU64(allocator, l, @bitCast(x)),
        .float64 => |x| try appendU64(allocator, l, @bitCast(x)),
        .array => return error.NestedArrayUnsupported,
    }
}

fn appendString(allocator: std.mem.Allocator, l: *std.ArrayList(u8), s: []const u8) !void {
    const n = @min(s.len, 0xFFFF);
    try appendU16(allocator, l, @intCast(n));
    try l.appendSlice(allocator, s[0..n]);
}

fn appendU16(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

fn appendU32(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

fn appendU64(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

test "deriveChatMarkers detects the <|turn> format" {
    const testing = std.testing;
    const tokens = [_]format.MetadataValue{
        .{ .string = "<|turn>" },
        .{ .string = "<turn|>" },
        .{ .string = "hello" },
    };
    const m = deriveChatMarkers(&tokens);
    try testing.expectEqualStrings("<|turn>user\n", m.user_prefix.?);
    try testing.expectEqualStrings("<turn|>\n", m.user_suffix.?);
    try testing.expectEqualStrings("<|turn>model\n", m.model_prefix.?);
}

test "deriveChatMarkers detects the <start_of_turn> format" {
    const testing = std.testing;
    const tokens = [_]format.MetadataValue{
        .{ .string = "<start_of_turn>" },
        .{ .string = "<end_of_turn>" },
    };
    const m = deriveChatMarkers(&tokens);
    try testing.expectEqualStrings("<start_of_turn>user\n", m.user_prefix.?);
    try testing.expectEqualStrings("<end_of_turn>\n", m.user_suffix.?);
    try testing.expectEqualStrings("<start_of_turn>model\n", m.model_prefix.?);
}

test "deriveChatMarkers omits an unknown vocabulary" {
    const testing = std.testing;
    const tokens = [_]format.MetadataValue{
        .{ .string = "hello" },
        .{ .string = "<|im_start|>" },
    };
    const m = deriveChatMarkers(&tokens);
    try testing.expect(m.user_prefix == null);
    try testing.expect(m.user_suffix == null);
    try testing.expect(m.model_prefix == null);
}

test "zmeta round-trips the derived chat markers" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = std.StringHashMap(format.MetadataValue).init(testing.allocator);
    defer md.deinit();
    var keys = [_][]const u8{"general.architecture"};
    try md.put("general.architecture", .{ .string = "gemma4" });
    var vocab = [_]format.MetadataValue{
        .{ .string = "<|turn>" },
        .{ .string = "<turn|>" },
    };
    try md.put("tokenizer.ggml.tokens", .{ .array = .{ .elem = .string, .items = &vocab } });

    var model = gguf.Gguf{
        .allocator = testing.allocator,
        .mmap = undefined,
        .header = .{ .version = 3, .tensor_count = 0, .metadata_kv_count = 1 },
        .metadata = md,
        .keys = &keys,
        .tensors = &.{},
        .data_offset = 0,
    };
    _ = try write(testing.allocator, tmp.dir, testing.io, &model);

    var z = try zmeta_reader.Zmeta.open(testing.allocator, tmp.dir, testing.io);
    defer z.deinit();
    try testing.expectEqualStrings("gemma4", z.architecture());
    try testing.expectEqualStrings("<|turn>user\n", z.getString(zmeta_reader.CHAT_USER_PREFIX).?);
    try testing.expectEqualStrings("<turn|>\n", z.getString(zmeta_reader.CHAT_USER_SUFFIX).?);
    try testing.expectEqualStrings("<|turn>model\n", z.getString(zmeta_reader.CHAT_MODEL_PREFIX).?);
}

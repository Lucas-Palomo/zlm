//! Gemma 4's native tool-call dialect:
//!
//! ```text
//! <|tool>declaration:name{description:<|"|>…<|"|>,parameters:{…}}<tool|>
//! <|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>
//! <|tool_response>response:name{value:<|"|>…<|"|>}<tool_response|>
//! ```
//!
//! The runtime moves structured calls through `architectures/core/tools.zig`;
//! this file owns the Gemma literal syntax (declaration only, no forward logic).

const std = @import("std");
const tools = @import("../../core/tools.zig");

pub const dialect = tools.Dialect{
    .renderDeclarations = renderDeclarations,
    .renderCall = renderCall,
    .renderResult = renderResult,
    .parseCalls = parseCalls,
};

const str_open = "<|\"|>";
const str_close = "<|\"|>";

pub fn renderDeclarations(allocator: std.mem.Allocator, list: []const tools.Tool) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    for (list) |tool| {
        try buf.appendSlice(allocator, "<|tool>declaration:");
        try buf.appendSlice(allocator, tool.name);
        try buf.appendSlice(allocator, "{description:");
        try buf.appendSlice(allocator, str_open);
        try buf.appendSlice(allocator, tool.description);
        try buf.appendSlice(allocator, str_close);
        if (tool.parameters) |params| {
            if (params == .object and params.object.count() > 0) {
                try buf.appendSlice(allocator, ",parameters:");
                try renderSchema(&buf, allocator, params);
            }
        }
        try buf.appendSlice(allocator, "}<tool|>");
    }
    return buf.toOwnedSlice(allocator);
}

pub fn renderCall(allocator: std.mem.Allocator, call: tools.Call) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "<|tool_call>call:");
    try buf.appendSlice(allocator, call.name);
    try buf.append(allocator, '{');
    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, parse_arena.allocator(), call.arguments, .{}) catch null;
    if (parsed) |args| {
        if (args == .object) try renderObject(&buf, allocator, args.object);
    }
    try buf.append(allocator, '}');
    try buf.appendSlice(allocator, "<tool_call|>");
    return buf.toOwnedSlice(allocator);
}

pub fn renderResult(allocator: std.mem.Allocator, name: []const u8, content: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "<|tool_response>response:");
    try buf.appendSlice(allocator, name);
    try buf.appendSlice(allocator, "{value:");
    try buf.appendSlice(allocator, str_open);
    try buf.appendSlice(allocator, content);
    try buf.appendSlice(allocator, str_close);
    try buf.appendSlice(allocator, "}<tool_response|>");
    return buf.toOwnedSlice(allocator);
}

/// Split `text` into its content and the `<|tool_call>…<tool_call|>` blocks it
/// carries. The blocks are removed from the content.
pub fn parseCalls(allocator: std.mem.Allocator, text: []const u8) !tools.Parsed {
    var content: std.ArrayList(u8) = .empty;
    errdefer content.deinit(allocator);
    var calls: std.ArrayList(tools.Call) = .empty;
    errdefer calls.deinit(allocator);

    var rest = text;
    while (true) {
        const open = std.mem.indexOf(u8, rest, "<|tool_call>") orelse {
            try content.appendSlice(allocator, rest);
            break;
        };
        try content.appendSlice(allocator, rest[0..open]);
        const after = rest[open + "<|tool_call>".len ..];
        const close = std.mem.indexOf(u8, after, "<tool_call|>") orelse {
            try content.appendSlice(allocator, rest[open..]);
            break;
        };
        const body = after[0..close];
        rest = after[close + "<tool_call|>".len ..];
        if (try parseCallBlock(allocator, body)) |call| {
            try calls.append(allocator, call);
        } else {
            try content.appendSlice(allocator, "<|tool_call>");
            try content.appendSlice(allocator, body);
            try content.appendSlice(allocator, "<tool_call|>");
        }
    }

    const trimmed = std.mem.trim(u8, content.items, " \t\r\n");
    const content_dup = try allocator.dupe(u8, trimmed);
    allocator.free(content.items);
    return .{ .content = content_dup, .calls = try calls.toOwnedSlice(allocator) };
}

fn parseCallBlock(allocator: std.mem.Allocator, raw: []const u8) !?tools.Call {
    var body = std.mem.trim(u8, raw, " \t\r\n");
    if (!std.mem.startsWith(u8, body, "call:")) return null;
    body = body["call:".len..];
    const brace = std.mem.indexOfScalar(u8, body, '{') orelse return null;
    const name = std.mem.trim(u8, body[0..brace], " \t\r\n");
    if (name.len == 0) return null;
    const end = matchingBrace(body, brace) orelse return null;
    const args = body[brace .. end + 1];

    const name_dup = try allocator.dupe(u8, name);
    var json: std.ArrayList(u8) = .empty;
    errdefer json.deinit(allocator);
    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    if (std.json.parseFromSliceLeaky(std.json.Value, parse_arena.allocator(), args, .{})) |value| {
        try writeJsonValue(&json, allocator, value);
    } else |_| {
        try json.append(allocator, '{');
        try parseDslObject(&json, allocator, args[1 .. args.len - 1]);
        try json.append(allocator, '}');
    }
    return .{ .name = name_dup, .arguments = try json.toOwnedSlice(allocator) };
}

/// Index of the `}` matching the `{` at `open`, skipping `<|"|>` strings.
fn matchingBrace(s: []const u8, open: usize) ?usize {
    var depth: usize = 0;
    var i = open;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], str_open)) {
            const next = std.mem.indexOfPos(u8, s, i + str_open.len, str_close) orelse return null;
            i = next + str_close.len;
            continue;
        }
        switch (s[i]) {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if (depth == 0) return i;
            },
            else => {},
        }
        i += 1;
    }
    return null;
}

fn renderSchema(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, params: std.json.Value) !void {
    const obj = params.object;
    try buf.append(allocator, '{');
    var comma = false;
    if (obj.get("properties")) |props| {
        if (props == .object and props.object.count() > 0) {
            try buf.appendSlice(allocator, "properties:{");
            try renderProperties(buf, allocator, props.object);
            try buf.appendSlice(allocator, "}");
            comma = true;
        }
    }
    if (obj.get("required")) |required| {
        if (required == .array and required.array.items.len > 0) {
            if (comma) try buf.append(allocator, ',');
            try renderStringArray(buf, allocator, required);
            comma = true;
        }
    }
    if (obj.get("type")) |t| {
        if (comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "type:");
        try renderType(buf, allocator, t);
    }
    try buf.append(allocator, '}');
}

fn renderProperties(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, props: std.json.ObjectMap) anyerror!void {
    const keys = try sortedKeys(allocator, props);
    defer allocator.free(keys);
    for (keys, 0..) |key, i| {
        if (i > 0) try buf.append(allocator, ',');
        const value = props.get(key).?;
        try buf.appendSlice(allocator, key);
        try buf.appendSlice(allocator, ":{");
        var comma = false;
        if (value == .object) {
            const obj = value.object;
            if (obj.get("description")) |d| {
                if (d == .string) {
                    try buf.appendSlice(allocator, "description:");
                    try renderString(buf, allocator, d.string);
                    comma = true;
                }
            }
            if (obj.get("enum")) |e| {
                if (e == .array) {
                    if (comma) try buf.append(allocator, ',');
                    try buf.appendSlice(allocator, "enum:");
                    try renderValue(buf, allocator, e);
                    comma = true;
                }
            }
            if (obj.get("items")) |items| {
                if (items == .object) {
                    if (comma) try buf.append(allocator, ',');
                    try renderItems(buf, allocator, items.object);
                    comma = true;
                }
            }
            if (obj.get("nullable")) |n| {
                if (n == .bool and n.bool) {
                    if (comma) try buf.append(allocator, ',');
                    try buf.appendSlice(allocator, "nullable:true");
                    comma = true;
                }
            }
            if (obj.get("type")) |t| {
                if (t == .string and std.ascii.eqlIgnoreCase(t.string, "object")) {
                    if (obj.get("properties")) |nested| {
                        if (nested == .object and nested.object.count() > 0) {
                            if (comma) try buf.append(allocator, ',');
                            try buf.appendSlice(allocator, "properties:{");
                            try renderProperties(buf, allocator, nested.object);
                            try buf.appendSlice(allocator, "}");
                            comma = true;
                        }
                    }
                    if (obj.get("required")) |req| {
                        if (req == .array and req.array.items.len > 0) {
                            if (comma) try buf.append(allocator, ',');
                            try renderStringArray(buf, allocator, req);
                            comma = true;
                        }
                    }
                }
            }
            if (obj.get("type")) |t| {
                if (comma) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "type:");
                try renderType(buf, allocator, t);
            }
        }
        try buf.append(allocator, '}');
    }
}

fn renderItems(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, items: std.json.ObjectMap) anyerror!void {
    try buf.appendSlice(allocator, "items:{");
    var comma = false;
    if (items.get("properties")) |props| {
        if (props == .object and props.object.count() > 0) {
            try buf.appendSlice(allocator, "properties:{");
            try renderProperties(buf, allocator, props.object);
            try buf.appendSlice(allocator, "}");
            comma = true;
        }
    }
    if (items.get("required")) |req| {
        if (req == .array and req.array.items.len > 0) {
            if (comma) try buf.append(allocator, ',');
            try renderStringArray(buf, allocator, req);
            comma = true;
        }
    }
    if (items.get("type")) |t| {
        if (comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "type:");
        try renderType(buf, allocator, t);
    }
    try buf.append(allocator, '}');
}

fn renderType(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, t: std.json.Value) !void {
    switch (t) {
        .string => |s| try renderTypeName(buf, allocator, s),
        .array => |a| {
            try buf.append(allocator, '[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try buf.append(allocator, ',');
                if (item == .string) try renderTypeName(buf, allocator, item.string) else try renderValue(buf, allocator, item);
            }
            try buf.append(allocator, ']');
        },
        else => try renderValue(buf, allocator, t),
    }
}

fn renderTypeName(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.appendSlice(allocator, str_open);
    for (s) |c| try buf.append(allocator, std.ascii.toUpper(c));
    try buf.appendSlice(allocator, str_close);
}

fn renderStringArray(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, value: std.json.Value) !void {
    try buf.appendSlice(allocator, "required:[");
    for (value.array.items, 0..) |item, i| {
        if (i > 0) try buf.append(allocator, ',');
        if (item == .string) try renderString(buf, allocator, item.string) else try renderValue(buf, allocator, item);
    }
    try buf.append(allocator, ']');
}

fn renderObject(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, obj: std.json.ObjectMap) anyerror!void {
    const keys = try sortedKeys(allocator, obj);
    defer allocator.free(keys);
    for (keys, 0..) |key, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, key);
        try buf.append(allocator, ':');
        try renderValue(buf, allocator, obj.get(key).?);
    }
}

fn renderValue(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, value: std.json.Value) anyerror!void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| try appendInt(buf, allocator, i, null),
        .float => |f| try appendInt(buf, allocator, 0, f),
        .number_string => |s| try buf.appendSlice(allocator, s),
        .string => |s| try renderString(buf, allocator, s),
        .array => |a| {
            try buf.append(allocator, '[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try buf.append(allocator, ',');
                try renderValue(buf, allocator, item);
            }
            try buf.append(allocator, ']');
        },
        .object => |o| {
            try buf.append(allocator, '{');
            try renderObject(buf, allocator, o);
            try buf.append(allocator, '}');
        },
    }
}

fn renderString(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.appendSlice(allocator, str_open);
    try buf.appendSlice(allocator, s);
    try buf.appendSlice(allocator, str_close);
}

fn sortedKeys(allocator: std.mem.Allocator, obj: std.json.ObjectMap) ![][]const u8 {
    const keys = try allocator.alloc([]const u8, obj.count());
    @memcpy(keys, obj.keys());
    std.mem.sort([]const u8, keys, {}, lessThanStr);
    return keys;
}

fn lessThanStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Parse the Gemma argument DSL (`key:<|"|>value<|"|>,n:1`) into a JSON object
/// body. Unknown bare values become JSON strings.
fn parseDslObject(out: *std.ArrayList(u8), allocator: std.mem.Allocator, src: []const u8) !void {
    var p = DslParser{ .src = src, .allocator = allocator, .out = out };
    try p.pairs();
}

const DslParser = struct {
    src: []const u8,
    i: usize = 0,
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    first: bool = true,

    fn pairs(self: *DslParser) !void {
        self.skip();
        while (self.i < self.src.len) {
            try self.key();
            try self.out.append(self.allocator, ':');
            self.skip();
            if (self.i < self.src.len and self.src[self.i] == ':') self.i += 1;
            try self.value();
            self.skip();
            if (self.i < self.src.len and self.src[self.i] == ',') self.i += 1;
            self.skip();
        }
    }

    fn key(self: *DslParser) !void {
        const start = self.i;
        while (self.i < self.src.len and self.src[self.i] != ':') self.i += 1;
        const raw = std.mem.trim(u8, self.src[start..self.i], " \t\r\n\"");
        if (!self.first) try self.out.append(self.allocator, ',');
        self.first = false;
        try writeJsonString(self.out, self.allocator, raw);
    }

    fn value(self: *DslParser) anyerror!void {
        self.skip();
        if (self.i >= self.src.len) {
            try self.out.appendSlice(self.allocator, "null");
            return;
        }
        if (std.mem.startsWith(u8, self.src[self.i..], str_open)) {
            self.i += str_open.len;
            const end = std.mem.indexOfPos(u8, self.src, self.i, str_close) orelse self.src.len;
            try writeJsonString(self.out, self.allocator, self.src[self.i..end]);
            self.i = @min(end + str_close.len, self.src.len);
            return;
        }
        switch (self.src[self.i]) {
            '{' => {
                self.i += 1;
                try self.out.append(self.allocator, '{');
                try self.objectBody();
                try self.out.append(self.allocator, '}');
            },
            '[' => {
                self.i += 1;
                try self.out.append(self.allocator, '[');
                var first = true;
                while (true) {
                    self.skip();
                    if (self.i >= self.src.len or self.src[self.i] == ']') {
                        if (self.i < self.src.len) self.i += 1;
                        break;
                    }
                    if (!first) try self.out.append(self.allocator, ',');
                    first = false;
                    try self.value();
                    self.skip();
                    if (self.i < self.src.len and self.src[self.i] == ',') self.i += 1;
                }
                try self.out.append(self.allocator, ']');
            },
            '"' => {
                self.i += 1;
                const start = self.i;
                while (self.i < self.src.len and self.src[self.i] != '"') self.i += 1;
                try writeJsonString(self.out, self.allocator, self.src[start..self.i]);
                if (self.i < self.src.len) self.i += 1;
            },
            else => {
                const start = self.i;
                while (self.i < self.src.len and self.src[self.i] != ',' and self.src[self.i] != '}' and self.src[self.i] != ']') self.i += 1;
                const token = std.mem.trim(u8, self.src[start..self.i], " \t\r\n");
                if (std.mem.eql(u8, token, "true") or std.mem.eql(u8, token, "false") or std.mem.eql(u8, token, "null")) {
                    try self.out.appendSlice(self.allocator, token);
                } else if (isNumber(token)) {
                    try self.out.appendSlice(self.allocator, token);
                } else {
                    try writeJsonString(self.out, self.allocator, token);
                }
            },
        }
    }

    fn objectBody(self: *DslParser) anyerror!void {
        var first = true;
        while (true) {
            self.skip();
            if (self.i >= self.src.len or self.src[self.i] == '}') {
                if (self.i < self.src.len) self.i += 1;
                break;
            }
            if (!first) try self.out.append(self.allocator, ',');
            first = false;
            const start = self.i;
            while (self.i < self.src.len and self.src[self.i] != ':') self.i += 1;
            const raw = std.mem.trim(u8, self.src[start..self.i], " \t\r\n\"");
            if (self.i < self.src.len) self.i += 1;
            try writeJsonString(self.out, self.allocator, raw);
            try self.out.append(self.allocator, ':');
            try self.value();
            self.skip();
            if (self.i < self.src.len and self.src[self.i] == ',') self.i += 1;
        }
    }

    fn skip(self: *DslParser) void {
        while (self.i < self.src.len and (self.src[self.i] == ' ' or self.src[self.i] == '\t' or self.src[self.i] == '\r' or self.src[self.i] == '\n')) self.i += 1;
    }
};

fn writeJsonValue(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: std.json.Value) !void {
    switch (value) {
        .null => try out.appendSlice(allocator, "null"),
        .bool => |b| try out.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| try appendInt(out, allocator, i, null),
        .float => |f| try appendInt(out, allocator, 0, f),
        .number_string => |s| try out.appendSlice(allocator, s),
        .string => |s| try writeJsonString(out, allocator, s),
        .array => |a| {
            try out.append(allocator, '[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try out.append(allocator, ',');
                try writeJsonValue(out, allocator, item);
            }
            try out.append(allocator, ']');
        },
        .object => |o| {
            const keys = try sortedKeys(allocator, o);
            defer allocator.free(keys);
            try out.append(allocator, '{');
            for (keys, 0..) |key, i| {
                if (i > 0) try out.append(allocator, ',');
                try writeJsonString(out, allocator, key);
                try out.append(allocator, ':');
                try writeJsonValue(out, allocator, o.get(key).?);
            }
            try out.append(allocator, '}');
        },
    }
}

fn appendInt(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, i: i64, f: ?f64) !void {
    var tmp: [64]u8 = undefined;
    const text = if (f) |v| try std.fmt.bufPrint(&tmp, "{d}", .{v}) else try std.fmt.bufPrint(&tmp, "{d}", .{i});
    try buf.appendSlice(allocator, text);
}

fn isNumber(s: []const u8) bool {
    if (s.len == 0) return false;
    var i: usize = 0;
    if (s[i] == '-' or s[i] == '+') i += 1;
    var digits: usize = 0;
    while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) digits += 1;
    if (i < s.len and s[i] == '.') {
        i += 1;
        while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) digits += 1;
    }
    if (digits == 0) return false;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '-' or s[i] == '+')) i += 1;
        const exp = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == exp) return false;
    }
    return i == s.len;
}

fn writeJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        else => {
            if (c < 0x20) {
                var esc: [6]u8 = undefined;
                const text = try std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c});
                try out.appendSlice(allocator, text);
            } else {
                try out.append(allocator, c);
            }
        },
    };
    try out.append(allocator, '"');
}

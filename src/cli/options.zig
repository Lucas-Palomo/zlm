const std = @import("std");
const Args = @import("args.zig").Args;

/// How an option's value is interpreted.
pub const Kind = enum { flag, value, uint, float, size };

/// One option a command accepts.
pub const Spec = struct {
    long: []const u8,
    short: ?u8 = null,
    kind: Kind = .flag,
    required: bool = false,
};

pub const Error = error{
    OutOfMemory,
    UnknownOption,
    MissingValue,
    InvalidValue,
    MissingRequired,
};

/// Result of parsing: flags seen, option values, and positional arguments.
pub const Parsed = struct {
    allocator: std.mem.Allocator,
    flags: std.StringHashMapUnmanaged(void) = .empty,
    values: std.StringHashMapUnmanaged([]const u8) = .empty,
    positionals: std.ArrayList([]const u8) = .empty,

    pub fn deinit(self: *Parsed) void {
        self.flags.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.positionals.deinit(self.allocator);
    }

    pub fn has(self: *const Parsed, long: []const u8) bool {
        return self.flags.contains(long);
    }

    pub fn get(self: *const Parsed, long: []const u8) ?[]const u8 {
        return self.values.get(long);
    }

    pub fn uint(self: *const Parsed, long: []const u8, comptime T: type) ?T {
        const v = self.values.get(long) orelse return null;
        return std.fmt.parseInt(T, v, 10) catch null;
    }

    pub fn float(self: *const Parsed, long: []const u8, comptime T: type) ?T {
        const v = self.values.get(long) orelse return null;
        return std.fmt.parseFloat(T, v) catch null;
    }

    /// A size in `mb`/`gb` -> bytes, or a bare number as bytes.
    pub fn size(self: *const Parsed, long: []const u8) ?u64 {
        const v = self.values.get(long) orelse return null;
        return parseSize(v);
    }

    pub fn positional(self: *const Parsed, i: usize) ?[]const u8 {
        if (i >= self.positionals.items.len) return null;
        return self.positionals.items[i];
    }
};

pub fn parse(allocator: std.mem.Allocator, args: *Args, specs: []const Spec) Error!Parsed {
    var out = Parsed{ .allocator = allocator };
    errdefer out.deinit();

    while (args.next()) |a| {
        if (std.mem.startsWith(u8, a, "--")) {
            try consume(allocator, &out, specs, a, args);
        } else if (a.len >= 2 and a[0] == '-') {
            try consumeShort(allocator, &out, specs, a, args);
        } else {
            try out.positionals.append(allocator, a);
        }
    }

    for (specs) |s| {
        if (!s.required) continue;
        const present = if (s.kind == .flag) out.flags.contains(s.long) else out.values.contains(s.long);
        if (!present) return Error.MissingRequired;
    }
    return out;
}

fn consume(
    allocator: std.mem.Allocator,
    out: *Parsed,
    specs: []const Spec,
    a: []const u8,
    args: *Args,
) Error!void {
    const eq = std.mem.indexOfScalar(u8, a, '=');
    const name = if (eq) |i| a[0..i] else a;
    const inline_val = if (eq) |i| a[i + 1 ..] else null;

    const spec = findLong(specs, name) orelse return Error.UnknownOption;
    try apply(allocator, out, spec, inline_val, args);
}

fn consumeShort(
    allocator: std.mem.Allocator,
    out: *Parsed,
    specs: []const Spec,
    a: []const u8,
    args: *Args,
) Error!void {
    // Support `-x`, `-x=v`, `-xV`? Keep it to `-x` / `-x=v` / `-x v`.
    const flag = a[1];
    const spec = findShort(specs, flag) orelse return Error.UnknownOption;
    const inline_val: ?[]const u8 = if (a.len > 2 and a[2] == '=') a[3..] else null;
    try apply(allocator, out, spec, inline_val, args);
}

fn apply(
    allocator: std.mem.Allocator,
    out: *Parsed,
    spec: Spec,
    inline_val: ?[]const u8,
    args: *Args,
) Error!void {
    if (spec.kind == .flag) {
        if (inline_val != null) return Error.UnknownOption;
        try out.flags.put(allocator, spec.long, {});
        return;
    }
    const value = inline_val orelse (args.next() orelse return Error.MissingValue);
    switch (spec.kind) {
        .value => {},
        .uint => _ = std.fmt.parseInt(u64, value, 10) catch return Error.InvalidValue,
        .float => _ = std.fmt.parseFloat(f64, value) catch return Error.InvalidValue,
        .size => _ = parseSize(value) orelse return Error.InvalidValue,
        .flag => unreachable,
    }
    try out.values.put(allocator, spec.long, value);
}

fn findLong(specs: []const Spec, name: []const u8) ?Spec {
    for (specs) |s| {
        if (std.mem.eql(u8, s.long, name)) return s;
    }
    return null;
}

fn findShort(specs: []const Spec, flag: u8) ?Spec {
    for (specs) |s| {
        if (s.short) |c| if (c == flag) return s;
    }
    return null;
}

fn parseSize(s: []const u8) ?u64 {
    if (std.mem.endsWith(u8, s, "mb")) {
        const n = std.fmt.parseInt(u64, s[0 .. s.len - 2], 10) catch return null;
        return n * 1024 * 1024;
    }
    if (std.mem.endsWith(u8, s, "gb")) {
        const n = std.fmt.parseInt(u64, s[0 .. s.len - 2], 10) catch return null;
        return n * 1024 * 1024 * 1024;
    }
    return std.fmt.parseInt(u64, s, 10) catch null;
}

test "parse accepts =, space, and short forms" {
    const testing = std.testing;
    const specs = [_]Spec{
        .{ .long = "--threads", .short = 't', .kind = .uint, .required = true },
        .{ .long = "--ram", .kind = .size, .required = true },
        .{ .long = "--gpu", .kind = .flag },
    };
    const items = [_][]const u8{ "--threads=4", "--ram", "2gb", "--gpu", "img", "alias" };
    var args = Args.init(&items);
    var opt = try parse(testing.allocator, &args, &specs);
    defer opt.deinit();
    try testing.expectEqual(@as(?u16, 4), opt.uint("--threads", u16));
    try testing.expectEqual(@as(?u64, 2 * 1024 * 1024 * 1024), opt.size("--ram"));
    try testing.expect(opt.has("--gpu"));
    try testing.expectEqualStrings("img", opt.positional(0).?);
    try testing.expectEqualStrings("alias", opt.positional(1).?);
}

test "parse reports unknown, missing, and required" {
    const testing = std.testing;
    const specs = [_]Spec{
        .{ .long = "--ram", .kind = .size, .required = true },
    };

    var a1 = Args.init(&[_][]const u8{"--nope"});
    try testing.expectError(Error.UnknownOption, parse(testing.allocator, &a1, &specs));

    var a2 = Args.init(&[_][]const u8{"--ram"});
    try testing.expectError(Error.MissingValue, parse(testing.allocator, &a2, &specs));

    var a3 = Args.init(&[_][]const u8{"--ram=bad"});
    try testing.expectError(Error.InvalidValue, parse(testing.allocator, &a3, &specs));

    var a4 = Args.init(&[_][]const u8{});
    try testing.expectError(Error.MissingRequired, parse(testing.allocator, &a4, &specs));
}

test "parse short with equals" {
    const testing = std.testing;
    const specs = [_]Spec{
        .{ .long = "--threads", .short = 't', .kind = .uint, .required = true },
    };
    var args = Args.init(&[_][]const u8{"-t=8"});
    var opt = try parse(testing.allocator, &args, &specs);
    defer opt.deinit();
    try testing.expectEqual(@as(?u16, 8), opt.uint("--threads", u16));
}

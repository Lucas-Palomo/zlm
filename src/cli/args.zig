const std = @import("std");

/// Cursor over a command's argv (everything after the command name).
pub const Args = struct {
    items: []const []const u8,
    index: usize = 0,

    pub fn init(items: []const []const u8) Args {
        return .{ .items = items };
    }

    pub fn isEmpty(self: *const Args) bool {
        return self.index >= self.items.len;
    }

    pub fn peek(self: *const Args) ?[]const u8 {
        if (self.index >= self.items.len) return null;
        return self.items[self.index];
    }

    pub fn next(self: *Args) ?[]const u8 {
        if (self.index >= self.items.len) return null;
        const a = self.items[self.index];
        self.index += 1;
        return a;
    }

    /// Resolve `current` against `name`: `name`, `name=value`, or `name value`.
    pub fn value(self: *Args, current: []const u8, name: []const u8) ?[]const u8 {
        if (current.len > name.len and std.mem.startsWith(u8, current, name) and current[name.len] == '=') {
            return current[name.len + 1 ..];
        }
        if (std.mem.eql(u8, current, name)) return self.next();
        return null;
    }

    pub fn is(current: []const u8, name: []const u8) bool {
        return std.mem.eql(u8, current, name);
    }

    pub fn rest(self: *Args) []const []const u8 {
        const out = self.items[self.index..];
        self.index = self.items.len;
        return out;
    }
};

test "value forms" {
    const items = [_][]const u8{ "--name", "x", "--ram=2gb", "--gpu" };
    var args = Args.init(&items);
    const a0 = args.next().?;
    try std.testing.expectEqualStrings("x", args.value(a0, "--name").?);
    const a1 = args.next().?;
    try std.testing.expectEqualStrings("2gb", args.value(a1, "--ram").?);
    const a2 = args.next().?;
    try std.testing.expect(args.value(a2, "--gpu") == null);
    try std.testing.expect(Args.is(a2, "--gpu"));
}

test "rest consumes positionals" {
    const items = [_][]const u8{ "model.gguf", "extra" };
    var args = Args.init(&items);
    const first = args.next().?;
    try std.testing.expectEqualStrings("model.gguf", first);
    try std.testing.expectEqual(@as(usize, 1), args.rest().len);
    try std.testing.expect(args.isEmpty());
}

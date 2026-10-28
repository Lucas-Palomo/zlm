const std = @import("std");
const Io = std.Io;
const File = Io.File;

pub fn stdout() File {
    return File.stdout();
}

pub fn stderr() File {
    return File.stderr();
}

pub fn writeAll(io: Io, file: File, bytes: []const u8) void {
    file.writeStreamingAll(io, bytes) catch {};
}

pub fn print(io: Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, fmt, args) catch return;
    writeAll(io, stdout(), result);
}

pub fn eprint(io: Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, fmt, args) catch return;
    writeAll(io, stderr(), result);
}

pub fn println(io: Io, comptime fmt: []const u8, args: anytype) void {
    print(io, fmt ++ "\n", args);
}

pub fn eprintln(io: Io, comptime fmt: []const u8, args: anytype) void {
    eprint(io, fmt ++ "\n", args);
}

/// Human-readable duration from milliseconds, written into `buf`: minutes +
/// seconds for long spans (`1m 5.9s`), seconds for >= 1 s (`8.09 s`),
/// milliseconds otherwise (`227.0 ms`). Avoids raw `8085.555 ms`.
pub fn formatDuration(buf: []u8, ms: f64) []const u8 {
    if (!(ms > 0)) return std.fmt.bufPrint(buf, "0 ms", .{}) catch "0 ms";
    if (ms >= 60_000.0) {
        const total_s = ms / 1000.0;
        const minutes: u64 = @intFromFloat(total_s / 60.0);
        const seconds = total_s - @as(f64, @floatFromInt(minutes)) * 60.0;
        return std.fmt.bufPrint(buf, "{d}m {d:.1}s", .{ minutes, seconds }) catch "?";
    }
    if (ms >= 1000.0) return std.fmt.bufPrint(buf, "{d:.2} s", .{ms / 1000.0}) catch "?";
    return std.fmt.bufPrint(buf, "{d:.1} ms", .{ms}) catch "?";
}

test "formatDuration picks a human unit" {
    const testing = std.testing;
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("227.0 ms", formatDuration(&buf, 227.0));
    try testing.expectEqualStrings("8.09 s", formatDuration(&buf, 8085.555));
    try testing.expectEqualStrings("1m 5.9s", formatDuration(&buf, 65904.0));
    try testing.expectEqualStrings("0 ms", formatDuration(&buf, 0));
}

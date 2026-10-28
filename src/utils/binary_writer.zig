const std = @import("std");

/// Append a little-endian `u16` to `l`.
pub fn appendU16(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

/// Append a little-endian `u32` to `l`.
pub fn appendU32(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

/// Append a little-endian `u64` to `l`.
pub fn appendU64(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try l.appendSlice(allocator, &b);
}

test "append little-endian integers" {
    const testing = std.testing;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    try appendU16(testing.allocator, &buf, 0x0201);
    try appendU32(testing.allocator, &buf, 0x06050403);
    try appendU64(testing.allocator, &buf, 0x0E0D0C0B0A090807);
    try testing.expectEqualSlices(
        u8,
        &[_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E },
        buf.items,
    );
}

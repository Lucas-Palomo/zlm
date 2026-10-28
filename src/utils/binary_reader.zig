const std = @import("std");

/// Little-endian cursor over a byte slice. Reads are bounds-checked.
pub const BinaryReader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn init(data: []const u8) BinaryReader {
        return .{ .data = data };
    }

    pub fn remaining(self: *const BinaryReader) usize {
        return self.data.len - self.pos;
    }

    pub fn skip(self: *BinaryReader, n: usize) !void {
        if (self.pos + n > self.data.len) return error.UnexpectedEOF;
        self.pos += n;
    }

    /// Zero-copy slice into the underlying data.
    pub fn readSlice(self: *BinaryReader, len: usize) ![]const u8 {
        if (self.pos + len > self.data.len) return error.UnexpectedEOF;
        const s = self.data[self.pos..][0..len];
        self.pos += len;
        return s;
    }

    fn readInt(self: *BinaryReader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (self.pos + n > self.data.len) return error.UnexpectedEOF;
        const arr: *const [n]u8 = @ptrCast(self.data[self.pos..][0..n].ptr);
        self.pos += n;
        return std.mem.readInt(T, arr, .little);
    }

    pub fn readU8(self: *BinaryReader) !u8 {
        return (try self.readSlice(1))[0];
    }

    pub fn readI8(self: *BinaryReader) !i8 {
        return @bitCast(try self.readU8());
    }

    pub fn readU16(self: *BinaryReader) !u16 {
        return self.readInt(u16);
    }

    pub fn readI16(self: *BinaryReader) !i16 {
        return @bitCast(try self.readU16());
    }

    pub fn readU32(self: *BinaryReader) !u32 {
        return self.readInt(u32);
    }

    pub fn readI32(self: *BinaryReader) !i32 {
        return @bitCast(try self.readU32());
    }

    pub fn readU64(self: *BinaryReader) !u64 {
        return self.readInt(u64);
    }

    pub fn readI64(self: *BinaryReader) !i64 {
        return @bitCast(try self.readU64());
    }

    pub fn readF32(self: *BinaryReader) !f32 {
        return @bitCast(try self.readU32());
    }

    pub fn readF64(self: *BinaryReader) !f64 {
        return @bitCast(try self.readU64());
    }

    pub fn readBool(self: *BinaryReader) !bool {
        return (try self.readU8()) != 0;
    }

    /// GGUF string: u64 length followed by that many bytes (zero-copy).
    pub fn readStringSlice(self: *BinaryReader) ![]const u8 {
        const len = try self.readU64();
        return self.readSlice(@intCast(len));
    }
};

test "binary reader basics" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    var r = BinaryReader.init(&data);
    try std.testing.expectEqual(@as(u8, 0x01), try r.readU8());
    try std.testing.expectEqual(@as(u16, 0x0302), try r.readU16());
    try std.testing.expectEqual(@as(u32, 0x07060504), try r.readU32());
}

test "binary reader zero-copy string" {
    const data = [_]u8{ 0x05, 0, 0, 0, 0, 0, 0, 0, 'h', 'e', 'l', 'l', 'o' };
    var r = BinaryReader.init(&data);
    const s = try r.readStringSlice();
    try std.testing.expectEqualStrings("hello", s);
    try std.testing.expect(@intFromPtr(s.ptr) == @intFromPtr(&data[8]));
}

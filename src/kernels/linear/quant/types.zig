const std = @import("std");
const api = @import("../../api.zig");

pub const KernelError = api.KernelError;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;
pub const Q6_K_BLOCK: usize = 210;
pub const Q5_K_BLOCK: usize = 176;
pub const Q4_1_BLOCK: usize = 20;
pub const Q4_K_BLOCK: usize = 144;

pub fn elementCount(dims: []const u64) usize {
    var n: usize = 1;
    for (dims) |d| n *= @intCast(d);
    return n;
}

pub fn readF16(data: []const u8) f16 {
    return @bitCast(std.mem.readInt(u16, data[0..2], .little));
}

pub fn q6KRowBytes(n: usize) usize {
    std.debug.assert(n % QK_K == 0);
    return (n / QK_K) * Q6_K_BLOCK;
}

pub fn q4_0RowBytes(n: usize) usize {
    std.debug.assert(n % QK4_0 == 0);
    return (n / QK4_0) * 18;
}

pub fn q8_0RowBytes(n: usize) usize {
    std.debug.assert(n % QK8_0 == 0);
    return (n / QK8_0) * 34;
}

pub fn q4_1RowBytes(n: usize) usize {
    std.debug.assert(n % QK4_1 == 0);
    return (n / QK4_1) * Q4_1_BLOCK;
}

pub fn q4KRowBytes(n: usize) usize {
    std.debug.assert(n % QK_K == 0);
    return (n / QK_K) * Q4_K_BLOCK;
}

pub fn q5KRowBytes(n: usize) usize {
    std.debug.assert(n % QK_K == 0);
    return (n / QK_K) * Q5_K_BLOCK;
}

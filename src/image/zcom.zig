const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;

pub const MAGIC = "ZCOM";
pub const VERSION: u32 = 1;
pub const NAME = "model.zcom";
const ALIGN: usize = 64;
const HEADER_BYTES: usize = 4 + 4 + 4;

/// What a region holds. The runtime keys on this to find the derived bytes it
/// needs without re-deriving them from the weights.
pub const Kind = enum(u8) {
    /// Norm weights already in f32 (`rms_norm` consumes f32).
    norm_f32 = 0,
    /// RoPE angles: a contiguous cos table, then the sin table.
    rope_cos_sin = 1,
    /// Embedding / PLE row weights dequantized to f32.
    embedding_f32 = 2,
    /// RoPE inverse frequencies for one config: `n_rot/2` f32 values, where
    /// `inv[i] = base^(-2*i/head_dim)`. Small and sufficient: the runtime
    /// rebuilds `cos(pos*inv[i])`/`sin(pos*inv[i])` per position. Storing full
    /// cos/sin tables would be ~268 MB for E2B (131072 positions), so this is
    /// the encoding `zcom` carries. Keyed by `role = 0`, `layer = config id`.
    rope_inv_freq = 3,
};

/// One derived region. `role` + `layer` + `kind` is the stable key the runtime
/// resolves; `offset`/`length` point into the payload.
pub const Region = struct {
    kind: Kind,
    role: u16,
    layer: u32,
    offset: u64,
    length: u64,
};

pub const Written = struct { bytes: u64, sha256: [32]u8 };

pub const Source = struct {
    kind: Kind,
    role: u16,
    layer: u32,
    bytes: []const u8,
};

pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, sources: []const Source) !Written {
    var table: std.ArrayList(u8) = .empty;
    defer table.deinit(allocator);

    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);

    try appendU32(allocator, &table, @intCast(sources.len));
    for (sources) |s| {
        try table.append(allocator, @intFromEnum(s.kind));
        try appendU16(allocator, &table, s.role);
        try appendU32(allocator, &table, s.layer);
        try appendU64(allocator, &table, payload.items.len);
        try appendU64(allocator, &table, s.bytes.len);
        try payload.appendSlice(allocator, s.bytes);
        const pad = alignUp(payload.items.len) - payload.items.len;
        try payload.appendNTimes(allocator, 0, pad);
    }
    // Pad the table so the payload starts 64-byte aligned (region offsets are
    // 64-aligned, so every region lands on a 64-byte boundary).
    const table_pad = alignUp(HEADER_BYTES + table.items.len) - (HEADER_BYTES + table.items.len);
    try table.appendNTimes(allocator, 0, table_pad);

    var header: [HEADER_BYTES]u8 = undefined;
    @memcpy(header[0..4], MAGIC);
    std.mem.writeInt(u32, header[4..8], VERSION, .little);
    std.mem.writeInt(u32, header[8..12], @intCast(table.items.len), .little);

    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var file = try dir.createFile(io, NAME, .{});
    defer file.close(io);
    h.update(&header);
    try file.writeStreamingAll(io, &header);
    h.update(table.items);
    try file.writeStreamingAll(io, table.items);
    h.update(payload.items);
    try file.writeStreamingAll(io, payload.items);
    var sha: [32]u8 = undefined;
    h.final(&sha);

    return .{ .bytes = HEADER_BYTES + table.items.len + payload.items.len, .sha256 = sha };
}

/// Build the `rope_inv_freq` payload for one config: `n_rot/2` f32 values,
/// `inv[i] = base^(-2*i/head_dim)`. Matches `RopeTable.init` and
/// `ropeProportional`'s scalar tail bit-for-bit (same `std.math.pow(f32)`), so
/// the runtime reproduces identical cos/sin rows.
pub fn ropeInvFreq(
    allocator: std.mem.Allocator,
    freq_base: f32,
    head_dim: u32,
    n_rot: u32,
) ![]f32 {
    const angles = n_rot / 2;
    const dim: f32 = @floatFromInt(head_dim);
    const inv = try allocator.alloc(f32, angles);
    for (inv, 0..) |*v, i| {
        const fi: f32 = @floatFromInt(i);
        v.* = std.math.pow(f32, freq_base, -2.0 * fi / dim);
    }
    return inv;
}

pub const Reader = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    regions: []Region,
    payload_start: usize,

    pub fn open(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Reader {
        var mmap = try MappedFile.openIn(dir, io, NAME);
        errdefer mmap.close();

        var br = BinaryReader.init(mmap.data);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        if (try br.readU32() != VERSION) return error.BadVersion;
        const table_bytes = try br.readU32();
        const count = try br.readU32();

        const regions = try allocator.alloc(Region, count);
        errdefer allocator.free(regions);
        for (regions) |*r| {
            const kind: Kind = @enumFromInt(try br.readU8());
            const role = try br.readU16();
            const layer = try br.readU32();
            const offset = try br.readU64();
            const length = try br.readU64();
            r.* = .{ .kind = kind, .role = role, .layer = layer, .offset = offset, .length = length };
        }

        return .{
            .allocator = allocator,
            .mmap = mmap,
            .regions = regions,
            .payload_start = HEADER_BYTES + table_bytes,
        };
    }

    pub fn deinit(self: *Reader) void {
        self.allocator.free(self.regions);
        self.mmap.close();
    }

    /// Resolve a region by `(kind, role, layer)`. For `rope_inv_freq` regions
    /// `role` is always 0 and `layer` is a *rope config id* (the index of the
    /// first layer that used the distinct `(freq_base, head_dim, n_rot)`), not a
    /// model layer. The config id is assigned deterministically at compile.
    pub fn find(self: *const Reader, kind: Kind, role: u16, layer: u32) ?*const Region {
        for (self.regions) |*r| {
            if (r.kind == kind and r.role == role and r.layer == layer) return r;
        }
        return null;
    }

    /// Zero-copy slice of a region's derived bytes.
    pub fn slice(self: *const Reader, r: *const Region) []const u8 {
        const start = self.payload_start + @as(usize, @intCast(r.offset));
        return self.mmap.data[start .. start + @as(usize, @intCast(r.length))];
    }
};

fn alignUp(n: usize) usize {
    return (n + ALIGN - 1) & ~(ALIGN - 1);
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

test "zcom round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const norm = [_]f32{ 1.0, 2.0, 3.5 };
    const rope = [_]f32{ 0.0, 1.0, 0.0, 1.0 };
    const sources = [_]Source{
        .{ .kind = .norm_f32, .role = 7, .layer = 0, .bytes = std.mem.sliceAsBytes(&norm) },
        .{ .kind = .rope_cos_sin, .role = 0, .layer = 0, .bytes = std.mem.sliceAsBytes(&rope) },
    };
    const w = try write(testing.allocator, tmp.dir, testing.io, &sources);
    try testing.expect(w.bytes > 0);

    var r = try Reader.open(testing.allocator, tmp.dir, testing.io);
    defer r.deinit();
    try testing.expectEqual(@as(usize, 2), r.regions.len);

    const nr = r.find(.norm_f32, 7, 0).?;
    const got = r.slice(nr);
    try testing.expectEqual(@as(usize, norm.len * 4), got.len);
    const got_f32 = std.mem.bytesAsSlice(f32, got);
    try testing.expectEqual(norm[2], got_f32[2]);

    try testing.expect(r.find(.norm_f32, 9, 0) == null);
}

test "zcom rope_inv_freq round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const inv = try ropeInvFreq(testing.allocator, 10000.0, 8, 8);
    defer testing.allocator.free(inv);
    const sources = [_]Source{
        .{ .kind = .rope_inv_freq, .role = 0, .layer = 2, .bytes = std.mem.sliceAsBytes(inv) },
    };
    _ = try write(testing.allocator, tmp.dir, testing.io, &sources);

    var r = try Reader.open(testing.allocator, tmp.dir, testing.io);
    defer r.deinit();

    const reg = r.find(.rope_inv_freq, 0, 2).?;
    const got = std.mem.bytesAsSlice(f32, r.slice(reg));
    try testing.expectEqual(inv.len, got.len);
    for (inv, got) |a, b| try testing.expectEqual(a, b);
    try testing.expect(r.find(.rope_inv_freq, 0, 1) == null);
}

test "zcom rope_inv_freq matches ropeProportional" {
    const testing = std.testing;
    const rope = @import("../kernels/transformer/rope.zig");

    const base: f32 = 10000.0;
    const head_dim: u32 = 8;
    const n_rot: u32 = 8;
    const inv = try ropeInvFreq(testing.allocator, base, head_dim, n_rot);
    defer testing.allocator.free(inv);

    var t = try rope.RopeTable.init(testing.allocator, n_rot / 2, 4, base, @floatFromInt(head_dim));
    defer t.deinit(testing.allocator);
    t.ensure(3);
    for (inv, 0..) |v, i| try testing.expectEqual(v, t.inv_freq[i]);

    var head = [_]f32{ 1, 1, 1, 1, 0, 0, 0, 0 };
    rope.ropeProportional(&head, 3, base, head_dim, n_rot);
    for (0..n_rot / 2) |i| {
        const angle = @as(f32, @floatFromInt(3)) * inv[i];
        try testing.expectApproxEqAbs(@cos(angle), head[i], 1e-5);
        try testing.expectApproxEqAbs(@sin(angle), head[head_dim / 2 + i], 1e-5);
    }
}

test "zcom payload regions are 64B aligned" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const a = [_]u8{1} ** 5;
    const b = [_]u8{2} ** 5;
    const sources = [_]Source{
        .{ .kind = .norm_f32, .role = 1, .layer = 0, .bytes = &a },
        .{ .kind = .embedding_f32, .role = 2, .layer = 0, .bytes = &b },
    };
    _ = try write(testing.allocator, tmp.dir, testing.io, &sources);

    var r = try Reader.open(testing.allocator, tmp.dir, testing.io);
    defer r.deinit();
    try testing.expectEqual(@as(u64, 0), r.regions[0].offset);
    try testing.expectEqual(@as(u64, 64), r.regions[1].offset);
}

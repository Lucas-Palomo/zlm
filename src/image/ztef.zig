const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const binary_writer = @import("../utils/binary_writer.zig");
const appendU16 = binary_writer.appendU16;
const appendU32 = binary_writer.appendU32;
const appendU64 = binary_writer.appendU64;
const format = @import("../gguf/format.zig");

pub const MAGIC = "ZTEF";
/// v2 adds the per-tensor `sha256` to the entry table (content identity for the
/// tuner's shared-weight dedup and for provenance diagnostics).
pub const VERSION: u32 = 2;
pub const GLOBAL_LAYER: u32 = 0xFFFF_FFFF;

pub const Kind = enum(u32) {
    q_proj = 0,
    k_proj = 1,
    v_proj = 2,
    o_proj = 3,
    gate_proj = 4,
    up_proj = 5,
    down_proj = 6,
    norms = 7,
    embd = 8,
    /// Auxiliary tensors with no role bucket (e.g. `rope_freqs.weight`), plus
    /// the seven projection kinds: `compile` merges those into this file,
    /// layer-major in decode consumption order. Kept so no tensor is silently
    /// dropped from the image.
    misc = 9,

    pub fn filename(self: Kind) []const u8 {
        return switch (self) {
            .q_proj => "q_proj.ztef",
            .k_proj => "k_proj.ztef",
            .v_proj => "v_proj.ztef",
            .o_proj => "o_proj.ztef",
            .gate_proj => "gate_proj.ztef",
            .up_proj => "up_proj.ztef",
            .down_proj => "down_proj.ztef",
            .norms => "norms.ztef",
            .embd => "embd.ztef",
            .misc => "misc.ztef",
        };
    }
};

pub const COUNT: usize = 10;

pub const Bucket = struct { kind: Kind, layer: u32 };

pub fn bucketOfName(name: []const u8) ?Bucket {
    if (std.mem.startsWith(u8, name, "blk.")) {
        const rest = name["blk.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
        const tail = rest[dot + 1 ..];

        if (std.mem.startsWith(u8, tail, "attn_q")) return .{ .kind = .q_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "attn_k")) return .{ .kind = .k_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "attn_v")) return .{ .kind = .v_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "attn_output")) return .{ .kind = .o_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "ffn_gate") or std.mem.startsWith(u8, tail, "inp_gate")) return .{ .kind = .gate_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "ffn_up")) return .{ .kind = .up_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "ffn_down")) return .{ .kind = .down_proj, .layer = layer };
        if (std.mem.startsWith(u8, tail, "proj")) return .{ .kind = .o_proj, .layer = layer };
        if (std.mem.endsWith(u8, tail, "norm.weight") or std.mem.eql(u8, tail, "layer_output_scale.weight")) {
            return .{ .kind = .norms, .layer = layer };
        }
        return null;
    }

    if (std.mem.eql(u8, name, "token_embd.weight")) return .{ .kind = .embd, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "output.weight")) return .{ .kind = .embd, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "per_layer_token_embd.weight")) return .{ .kind = .embd, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "per_layer_model_proj.weight")) return .{ .kind = .o_proj, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "output_norm.weight")) return .{ .kind = .norms, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "per_layer_proj_norm.weight")) return .{ .kind = .norms, .layer = GLOBAL_LAYER };
    if (std.mem.eql(u8, name, "rope_freqs.weight")) return .{ .kind = .misc, .layer = GLOBAL_LAYER };
    return null;
}

pub const Item = struct {
    name: []const u8,
    layer: u32,
    ggml_type: format.GGMLType,
    n_dims: u32,
    dims: [4]u64,
    bytes: []const u8,
};

pub const Entry = struct {
    name: []const u8,
    layer: u32,
    ggml_type: format.GGMLType,
    n_dims: u32,
    dims: [4]u64,
    offset: u64,
    length: u64,
    /// Content identity of this tensor's bytes. Persisted at compile so the
    /// tuner can detect byte-identical tensors without re-reading the payload.
    sha256: [32]u8,
};

const HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 4 + 8 + 8;

pub const TensorSha = struct { name: []const u8, sha256: [32]u8 };
pub const Written = struct { bytes: u64, sha256: [32]u8, tensors: []TensorSha };

/// A page-aligned absolute byte range inside a `Reader`'s mapping.
pub const RangeAddr = struct { addr: usize, len: usize };

pub fn shaLessByName(_: void, a: TensorSha, b: TensorSha) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, kind: Kind, items: []const Item) !Written {
    var table: std.ArrayList(u8) = .empty;
    defer table.deinit(allocator);

    var tensor_shas: std.ArrayList(TensorSha) = .empty;
    errdefer tensor_shas.deinit(allocator);
    var payload_bytes: u64 = 0;
    for (items) |it| {
        var th = std.crypto.hash.sha2.Sha256.init(.{});
        th.update(it.bytes);
        var ts: [32]u8 = undefined;
        th.final(&ts);
        try tensor_shas.append(allocator, .{ .name = it.name, .sha256 = ts });

        try appendU16(allocator, &table, @intCast(it.name.len));
        try table.appendSlice(allocator, it.name);
        try appendU32(allocator, &table, it.layer);
        try appendU32(allocator, &table, @intFromEnum(it.ggml_type));
        try appendU32(allocator, &table, it.n_dims);
        for (it.dims) |d| try appendU64(allocator, &table, d);
        try appendU64(allocator, &table, payload_bytes);
        try appendU64(allocator, &table, it.bytes.len);
        try table.appendSlice(allocator, &ts);
        payload_bytes += it.bytes.len;
    }

    var header: [HEADER_BYTES]u8 = undefined;
    @memcpy(header[0..4], MAGIC);
    std.mem.writeInt(u32, header[4..8], VERSION, .little);
    std.mem.writeInt(u32, header[8..12], @intFromEnum(kind), .little);
    std.mem.writeInt(u32, header[12..16], @intCast(items.len), .little);
    std.mem.writeInt(u32, header[16..20], 0, .little);
    std.mem.writeInt(u64, header[20..28], table.items.len, .little);
    std.mem.writeInt(u64, header[28..36], payload_bytes, .little);

    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var f = try dir.createFile(io, kind.filename(), .{});
    defer f.close(io);
    h.update(&header);
    try f.writeStreamingAll(io, &header);
    h.update(table.items);
    try f.writeStreamingAll(io, table.items);
    for (items) |it| {
        h.update(it.bytes);
        try f.writeStreamingAll(io, it.bytes);
    }
    var sha: [32]u8 = undefined;
    h.final(&sha);

    return .{
        .bytes = HEADER_BYTES + table.items.len + payload_bytes,
        .sha256 = sha,
        .tensors = try tensor_shas.toOwnedSlice(allocator),
    };
}

pub const Reader = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    kind: Kind,
    payload_start: usize,
    entries: []Entry,

    pub fn open(allocator: std.mem.Allocator, io: Io, path: []const u8) !Reader {
        return openIn(allocator, Io.Dir.cwd(), io, path);
    }

    pub fn openIn(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, sub_path: []const u8) !Reader {
        var mmap = try MappedFile.openIn(dir, io, sub_path);
        errdefer mmap.close();

        var br = BinaryReader.init(mmap.data);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        const version = try br.readU32();
        if (version != VERSION) return error.BadVersion;
        const kind = @as(Kind, @enumFromInt(try br.readU32()));
        const count = try br.readU32();
        _ = try br.readU32();
        const entry_bytes = try br.readU64();
        _ = try br.readU64();

        const entries = try allocator.alloc(Entry, count);
        errdefer allocator.free(entries);
        for (entries) |*e| {
            const name_len = try br.readU16();
            const name = try br.readSlice(name_len);
            const layer = try br.readU32();
            const ggml_type = format.GGMLType.fromInt(try br.readU32()) orelse return error.InvalidTensorType;
            const n_dims = try br.readU32();
            var dims = [_]u64{0} ** 4;
            for (0..4) |i| dims[i] = try br.readU64();
            const offset = try br.readU64();
            const length = try br.readU64();
            const sha_slice = try br.readSlice(32);
            var sha: [32]u8 = undefined;
            @memcpy(&sha, sha_slice);
            e.* = .{ .name = name, .layer = layer, .ggml_type = ggml_type, .n_dims = n_dims, .dims = dims, .offset = offset, .length = length, .sha256 = sha };
        }

        const payload_start = HEADER_BYTES + @as(usize, @intCast(entry_bytes));
        return .{ .allocator = allocator, .mmap = mmap, .kind = kind, .payload_start = payload_start, .entries = entries };
    }

    pub fn deinit(self: *Reader) void {
        self.allocator.free(self.entries);
        self.mmap.close();
    }

    pub fn find(self: *const Reader, name: []const u8) ?*const Entry {
        for (self.entries) |*e| {
            if (std.mem.eql(u8, e.name, name)) return e;
        }
        return null;
    }

    pub fn slice(self: *const Reader, e: *const Entry) []const u8 {
        const start = self.payload_start + @as(usize, @intCast(e.offset));
        return self.mmap.data[start..][0..@intCast(e.length)];
    }

    /// Resolve a byte range of the mapped payload to its page-aligned absolute
    /// `{ addr, len }`, clamped to the mapping, or null when the range is empty
    /// or out of bounds. `offset` is into the file payload, as recorded in the
    /// plan's residency ranges.
    pub fn rangeAddr(self: *const Reader, offset: u64, length: u64) ?RangeAddr {
        if (length == 0) return null;
        const data = self.mmap.data;
        const off: usize = @intCast(offset);
        const len: usize = @intCast(length);
        if (off > data.len or len > data.len - off) return null;
        const start = self.payload_start + off;
        if (start > data.len or len > data.len - start) return null;
        const end = start + len;
        const base = @intFromPtr(data.ptr);
        const page = std.heap.page_size_min;
        const a_start = std.mem.alignBackward(usize, base + start, page);
        const a_end = std.mem.alignForward(usize, base + end, page);
        if (a_start >= a_end) return null;
        return .{ .addr = a_start, .len = a_end - a_start };
    }

    /// The file descriptor backing the mapping, for the arena's `pread`
    /// workers (a read from `fileAddr`).
    pub fn fileHandle(self: *const Reader) std.posix.fd_t {
        return self.mmap.handle;
    }

    /// The file byte offset of a payload range: the `pread` source for the
    /// staging arena. `offset` is into the file payload, as recorded in the
    /// plan's residency ranges; null when the range is empty or out of bounds.
    pub fn fileAddr(self: *const Reader, offset: u64, length: u64) ?u64 {
        if (length == 0) return null;
        const start = @as(u64, self.payload_start) + offset;
        if (start < @as(u64, self.payload_start)) return null;
        if (start > self.mmap.size or length > self.mmap.size - start) return null;
        return start;
    }

    /// Page a byte range of the mapped payload in (`will_need`) or out. `offset`
    /// is into the file payload, as recorded in the plan's residency ranges.
    pub fn adviseRange(self: *const Reader, offset: u64, length: u64, will_need: bool) void {
        const fr = self.rangeAddr(offset, length) orelse return;
        const advice: u32 = if (will_need) std.posix.MADV.WILLNEED else std.posix.MADV.DONTNEED;
        const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(fr.addr);
        _ = std.posix.madvise(ptr, fr.len, advice) catch {};
    }
};

test "ztef round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const a = [_]u8{ 1, 2, 3, 4 };
    const b = [_]u8{ 5, 6 };
    const items = [_]Item{
        .{ .name = "blk.0.attn_q.weight", .layer = 0, .ggml_type = .q4_0, .n_dims = 2, .dims = .{ 64, 2, 0, 0 }, .bytes = &a },
        .{ .name = "blk.1.attn_q.weight", .layer = 1, .ggml_type = .q4_0, .n_dims = 2, .dims = .{ 64, 2, 0, 0 }, .bytes = &b },
    };
    const w = try write(testing.allocator, tmp.dir, testing.io, .q_proj, &items);
    defer testing.allocator.free(w.tensors);
    try testing.expect(w.bytes > 0);
    try testing.expectEqual(@as(usize, 2), w.tensors.len);

    var r = try Reader.openIn(testing.allocator, tmp.dir, testing.io, Kind.q_proj.filename());
    defer r.deinit();
    try testing.expectEqual(@as(usize, 2), r.entries.len);

    const e0 = r.find("blk.0.attn_q.weight").?;
    try testing.expectEqual(@as(u32, 0), e0.layer);
    try testing.expectEqualSlices(u8, &a, r.slice(e0));
    const e1 = r.find("blk.1.attn_q.weight").?;
    try testing.expectEqual(@as(u32, 1), e1.layer);
    try testing.expectEqualSlices(u8, &b, r.slice(e1));

    // The per-tensor content identity is persisted and matches the payload.
    var expect: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&a, &expect, .{});
    try testing.expectEqualSlices(u8, &expect, &e0.sha256);
    try testing.expectEqualSlices(u8, &w.tensors[0].sha256, &e0.sha256);
    try testing.expect(!std.mem.eql(u8, &e0.sha256, &e1.sha256));
}

test "bucket of name" {
    const b = bucketOfName("blk.3.attn_q.weight").?;
    try std.testing.expectEqual(Kind.q_proj, b.kind);
    try std.testing.expectEqual(@as(u32, 3), b.layer);
    try std.testing.expectEqual(Kind.norms, bucketOfName("blk.0.attn_norm.weight").?.kind);
    try std.testing.expectEqual(Kind.embd, bucketOfName("token_embd.weight").?.kind);
    try std.testing.expectEqual(GLOBAL_LAYER, bucketOfName("output.weight").?.layer);
    // Auxiliary roots are kept (not dropped): the previously-unmapped
    // `rope_freqs.weight` lands in `misc`.
    try std.testing.expectEqual(Kind.misc, bucketOfName("rope_freqs.weight").?.kind);
    try std.testing.expectEqual(GLOBAL_LAYER, bucketOfName("rope_freqs.weight").?.layer);
    try std.testing.expect(bucketOfName("blk.0.unknown.weight") == null);
}

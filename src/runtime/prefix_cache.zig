const std = @import("std");

/// Absent reference sentinel for a segment's parent (and an absent journal).
pub const NONE: u32 = 0xFFFF_FFFF;
/// Absent region index.
pub const NO_REGION: u16 = 0xFFFF;

/// The identity a cached K/V depends on. A mismatch invalidates the whole tree.
/// It is the **projection of the config onto the K/V structure**, so runtimes of
/// the same image whose K/V plan matches share one artifact.
/// - `model` = the **weights** content root; the K/V is `f(weights, arithmetic,
///   tokens)`, so different weights must never share. (No separate tokenizer
///   field: the tokenizer is already part of the content root.)
/// - `arithmetic` includes the ISA, the **device** and every dispatch's kernel
///   variant; it deliberately excludes `--threads`/`--ram`/`--vram` (row-split
///   matvec and per-head attention do not change the K/V).
/// - `layout` is the canonical K/V shape, **without** `max_seq` and **without**
///   the plan's buffer slot (the payload is canonical), so the cache is shared
///   across context lengths and buffer numbering.
pub const Key = struct {
    model: u64 = 0,
    arithmetic: u64 = 0,
    layout: u64 = 0,
    /// Prompt-format identity: the effective chat template (turn markers +
    /// think decorations). A recompile that changes the prompt shape (e.g. new
    /// `zlm.chat.*` markers) with an unchanged `model` still invalidates the
    /// cache, so a stale `.zpc` is never reused.
    format: u64 = 0,
    dtype: u8 = 0,

    pub fn eql(a: Key, b: Key) bool {
        return a.model == b.model and
            a.arithmetic == b.arithmetic and
            a.layout == b.layout and
            a.format == b.format and
            a.dtype == b.dtype;
    }

    /// Stable fingerprint of the whole key, used to name the `.zpc` artifact so
    /// runtimes with the same K/V-relevant identity share one file.
    pub fn fingerprint(self: Key) u64 {
        var h = std.hash.Wyhash.init(0x4B56_43_4143_4800);
        h.update(std.mem.asBytes(&self.model));
        h.update(std.mem.asBytes(&self.arithmetic));
        h.update(std.mem.asBytes(&self.layout));
        h.update(std.mem.asBytes(&self.format));
        h.update(&[_]u8{self.dtype});
        return h.final();
    }
};

/// One node of the causal prefix radix tree. Its edge label is the token range
/// `[token_begin, token_end)`; its payload is the canonical K/V of that range
/// (stored at `kv_off` in `region`) immediately followed by the `u32` token ids.
/// The K/V never lives on the heap: `kv_off` points into a mapped region.
pub const Segment = struct {
    parent: u32 = NONE,
    token_begin: u32 = 0,
    token_end: u32 = 0,
    kv_off: u64 = 0,
    region: u16 = NO_REGION,
    /// Direct children, used by eviction to spare interior nodes.
    children: u32 = 0,
    /// Live readers; a pinned segment is never evicted.
    refcount: u32 = 0,
    last_used: u64 = 0,
    alive: bool = true,

    pub fn tokenLen(self: Segment) u32 {
        return self.token_end - self.token_begin;
    }
};

/// Counters reported through the existing metrics surface. Diagnostic only.
pub const Stats = struct {
    lookups: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    hit_tokens: u64 = 0,
    evictions: u64 = 0,
};

/// A mapped region K/V payloads may reference. `writable` is the runtime
/// journal (appended segments); read-only regions back the immutable `.zpc`.
pub const Region = struct {
    bytes: []u8,
    writable: bool,
};

/// Callback that materializes the canonical K/V bytes for `[begin, end)`.
/// `size` reserves the journal space; `write` fills it in place.
pub const Builder = struct {
    ctx: *anyopaque,
    size: *const fn (ctx: *anyopaque, begin: u32, end: u32) u64,
    write: *const fn (ctx: *anyopaque, begin: u32, end: u32, dst: []u8) anyerror!void,
};

/// Free block in the writable journal region (first-fit, coalesced).
const Block = struct { off: u64, len: u64 };

/// Persisted counters at the front of the meta region. The segment/free-list
/// arrays mapped right after it persist in place, so re-mapping a **persistent**
/// meta restores the whole tree when `magic` matches; otherwise the meta is
/// initialized fresh.
const State = extern struct {
    magic: u64,
    seg_count: u32,
    free_count: u32,
    block_count: u32,
    pad: u32 = 0,
    bytes: u64,
    clock: u64,
};
const state_magic: u64 = 0x5A50_4343_5452_4545; // "ZPCCTREE"

/// Bytes of index storage one segment needs (record + free slot + free block),
/// plus the fixed `State` header.
pub fn metaBytesPerSegment() u64 {
    return @sizeOf(Segment) + @sizeOf(u32) + @sizeOf(Block) + 8;
}

/// Fixed size of the persisted meta header.
pub fn metaHeaderBytes() u64 {
    return @sizeOf(State);
}

/// An mmap-backed, byte-budgeted prefix **radix tree**. Nothing is heap
/// allocated: the K/V payloads and the token ids live in mapped regions, and the
/// segment index is itself a mapped region (`addMeta`).
pub const Memo = struct {
    allocator: std.mem.Allocator,
    key: Key,
    /// Canonical K/V bytes per token across every KV buffer (f16).
    per_token_bytes: u64,
    /// Soft byte ceiling of the journal region.
    budget: u64 = 0,
    segments: []Segment = &.{},
    free_slots: []u32 = &.{},
    free_blocks: []Block = &.{},
    seg_count: u32 = 0,
    free_count: u32 = 0,
    block_count: u32 = 0,
    regions: std.ArrayList(Region) = .empty,
    /// Index of the single writable (journal) region, or `NO_REGION`.
    journal: u16 = NO_REGION,
    bytes: u64 = 0,
    clock: u64 = 0,
    /// Persisted counters header inside the mapped meta region (null until
    /// `addMeta`), and whether `addMeta` restored a previous tree.
    state: ?*State = null,
    restored: bool = false,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator, key: Key, per_token_bytes: u64) Memo {
        return .{ .allocator = allocator, .key = key, .per_token_bytes = per_token_bytes };
    }

    pub fn deinit(self: *Memo) void {
        // Region bytes (including the index) are owned by the caller (mmaps).
        self.regions.deinit(self.allocator);
    }

    /// Register a mapped K/V region. `writable` marks the journal (only one)
    /// whose capacity becomes the budget.
    pub fn addRegion(self: *Memo, bytes: []u8, writable: bool) !u16 {
        if (writable and self.journal != NO_REGION) return error.DuplicateJournal;
        const idx: u16 = @intCast(self.regions.items.len);
        try self.regions.append(self.allocator, .{ .bytes = bytes, .writable = writable });
        if (writable) {
            self.journal = idx;
            self.budget = bytes.len;
            // A restored meta already carries the journal's free list; only a
            // fresh meta is seeded with the whole region free.
            if (!self.restored) {
                self.block_count = 0;
                if (bytes.len > 0) {
                    self.free_blocks[0] = .{ .off = 0, .len = bytes.len };
                    self.block_count = 1;
                }
            }
            self.sync();
        }
        return idx;
    }

    /// Carve the `State` header, the segment index and the free lists out of a
    /// mapped metadata region. Must be called **before** `addRegion(writable)`.
    /// `bytes` must be aligned. When the header's magic matches, the previous
    /// tree (counters + the mapped arrays) is **restored** instead of reset.
    pub fn addMeta(self: *Memo, bytes: []u8) !void {
        const st_sz = @sizeOf(State);
        if (bytes.len <= st_sz) return error.MetaTooSmall;
        self.state = @ptrCast(@alignCast(bytes.ptr));
        const body = bytes[st_sz..];
        const seg_sz = @sizeOf(Segment);
        const slot_sz = @sizeOf(u32);
        const blk_sz = @sizeOf(Block);
        const cap: usize = body.len / (seg_sz + slot_sz + blk_sz + 8);
        if (cap == 0) return error.MetaTooSmall;
        const slot_off = cap * seg_sz;
        const blk_off = std.mem.alignForward(usize, slot_off + cap * slot_sz, @alignOf(Block));
        if (blk_off + cap * blk_sz > body.len) return error.MetaTooSmall;
        self.segments = @alignCast(std.mem.bytesAsSlice(Segment, body[0 .. cap * seg_sz]));
        self.free_slots = @alignCast(std.mem.bytesAsSlice(u32, body[slot_off .. slot_off + cap * slot_sz]));
        self.free_blocks = @alignCast(std.mem.bytesAsSlice(Block, body[blk_off .. blk_off + cap * blk_sz]));
        const st = self.state.?;
        if (st.magic == state_magic) {
            self.seg_count = st.seg_count;
            self.free_count = st.free_count;
            self.block_count = st.block_count;
            self.bytes = st.bytes;
            self.clock = st.clock;
            self.restored = true;
        } else {
            self.seg_count = 0;
            self.free_count = 0;
            self.block_count = 0;
            self.bytes = 0;
            self.clock = 0;
            self.restored = false;
        }
    }

    /// Persist the mutation counters into the mapped header. Cheap; a persistent
    /// meta uses it to survive a restart.
    pub fn sync(self: *Memo) void {
        const st = self.state orelse return;
        st.magic = state_magic;
        st.seg_count = self.seg_count;
        st.free_count = self.free_count;
        st.block_count = self.block_count;
        st.bytes = self.bytes;
        st.clock = self.clock;
    }

    pub fn regionBytes(self: *const Memo, idx: u16) []const u8 {
        if (idx == NO_REGION or idx >= self.regions.items.len) return &.{};
        return self.regions.items[idx].bytes;
    }

    pub fn matches(self: *const Memo, key: Key) bool {
        return Key.eql(self.key, key);
    }

    pub fn liveSegments(self: *const Memo) []const Segment {
        return self.segments[0..self.seg_count];
    }

    pub fn segment(self: *const Memo, id: u32) ?*const Segment {
        if (id == NONE or id >= self.seg_count) return null;
        const seg = &self.segments[id];
        return if (seg.alive) seg else null;
    }

    /// Total bytes one segment's block needs: canonical K/V plus its `u32` ids.
    pub fn blockBytes(self: *const Memo, tok_len: u32) u64 {
        return @as(u64, tok_len) * (self.per_token_bytes + @sizeOf(u32));
    }

    /// The canonical K/V bytes of a segment (a slice into a mapped region).
    pub fn payloadOf(self: *const Memo, id: u32) ?[]const u8 {
        const seg = self.segment(id) orelse return null;
        const len = self.blockBytes(seg.tokenLen()) - @as(u64, seg.tokenLen()) * @sizeOf(u32);
        const bytes = self.regionBytes(seg.region);
        if (seg.kv_off + len > bytes.len) return null;
        return bytes[@intCast(seg.kv_off)..@intCast(seg.kv_off + len)];
    }

    /// The token ids of a segment (stored right after its K/V).
    pub fn tokensOf(self: *const Memo, id: u32) []const u32 {
        const seg = self.segment(id) orelse return &.{};
        const tok_len = seg.tokenLen();
        if (tok_len == 0) return &.{};
        const bytes = self.regionBytes(seg.region);
        const kv_len = @as(u64, tok_len) * self.per_token_bytes;
        const start: usize = @intCast(seg.kv_off + kv_len);
        const end: usize = @intCast(seg.kv_off + kv_len + @as(u64, tok_len) * @sizeOf(u32));
        if (end > bytes.len) return &.{};
        return @alignCast(std.mem.bytesAsSlice(u32, bytes[start..end]));
    }

    pub fn totalBytes(self: *const Memo) u64 {
        return self.bytes;
    }

    /// Longest causal path of segments whose edge labels match `tokens` from
    /// position 0 (the **maximal common prefix**). A node may be matched
    /// partially; a divergent continuation is a child that starts at the
    /// divergence offset (no split, no copy). Matched segments are pinned
    /// (refcount++) until `unpin`. Returns the number of matched tokens.
    pub fn lookup(self: *Memo, tokens: []const u32, path: *std.ArrayList(u32)) !u32 {
        self.clock +%= 1;
        self.stats.lookups += 1;
        var pos: u32 = 0;
        var parent: u32 = NONE;
        while (pos < tokens.len) {
            const id = self.findChild(parent, pos, tokens[pos]) orelse break;
            const seg = &self.segments[id];
            const seg_tokens = self.tokensOf(id);
            const seg_len: u32 = @intCast(seg_tokens.len);
            const max = @min(seg_len, @as(u32, @intCast(tokens.len)) - pos);
            var d: u32 = 0;
            while (d < max and tokens[pos + d] == seg_tokens[d]) d += 1;
            if (d == 0) break;
            seg.last_used = self.clock;
            seg.refcount += 1;
            try path.append(self.allocator, id);
            pos += d;
            parent = id;
            // If `d < seg_len` the prompt diverged inside this node; the loop
            // looks for a child starting at `pos` (a previous divergence).
        }
        if (pos > 0) {
            self.stats.hits += 1;
            self.stats.hit_tokens += pos;
        } else {
            self.stats.misses += 1;
        }
        return pos;
    }

    /// Drop the pins taken by `lookup` for `path`.
    pub fn unpin(self: *Memo, path: []const u32) void {
        for (path) |id| {
            const seg = self.segment(id) orelse continue;
            if (seg.refcount > 0) self.segments[id].refcount -= 1;
        }
    }

    /// Commit the K/V of `tokens[from..to)` as a child of `parent`. The lookup
    /// already split at `from`, so `parent` has no child matching `tokens[from]`.
    pub fn commit(
        self: *Memo,
        tokens: []const u32,
        parent: u32,
        from: u32,
        to: u32,
        builder: Builder,
    ) !void {
        if (to <= from or to > tokens.len) return;
        if (self.journal == NO_REGION) return error.NoJournal;
        self.clock +%= 1;
        const tok_len = to - from;
        const kv_len = builder.size(builder.ctx, from, to);
        const block_len = kv_len + @as(u64, tok_len) * @sizeOf(u32);
        const block = try self.reserve(block_len);
        errdefer self.journalFree(block.off, block.len);
        const jbytes = self.regions.items[self.journal].bytes;
        const kv_dst = jbytes[@intCast(block.off)..@intCast(block.off + kv_len)];
        try builder.write(builder.ctx, from, to, kv_dst);
        const tok_dst = jbytes[@intCast(block.off + kv_len)..@intCast(block.off + block_len)];
        @memcpy(tok_dst, std.mem.sliceAsBytes(tokens[from..to]));
        _ = try self.addRaw(parent, from, to, self.journal, block.off);
    }

    /// Insert a segment referencing an existing region block (used by `commit`
    /// and by the `.zpc` loader). The block is `K/V` then `u32` token ids.
    pub fn addRaw(self: *Memo, parent: u32, begin: u32, end: u32, region: u16, kv_off: u64) !u32 {
        const id = try self.allocSlot();
        self.segments[id] = .{
            .parent = parent,
            .token_begin = begin,
            .token_end = end,
            .kv_off = kv_off,
            .region = region,
            .children = 0,
            .refcount = 0,
            .last_used = self.clock,
            .alive = true,
        };
        self.bytes += self.blockBytes(end - begin);
        if (parent != NONE and parent < self.seg_count and self.segments[parent].alive) {
            self.segments[parent].children += 1;
        }
        self.sync();
        return id;
    }

    /// Re-run the byte-budget pass (e.g. after loading an artifact).
    pub fn trim(self: *Memo) void {
        while (self.bytes > self.budget) {
            if (!self.evictOneLeaf()) break;
        }
    }

    fn reserve(self: *Memo, size: u64) !Block {
        if (size == 0) return .{ .off = 0, .len = 0 };
        while (self.bytes + size > self.budget) {
            if (!self.evictOneLeaf()) return error.OutOfSpace;
        }
        if (self.journalAlloc(size)) |block| return block else |_| {}
        self.compactJournal();
        return self.journalAlloc(size);
    }

    fn journalAlloc(self: *Memo, size: u64) !Block {
        // 4-byte alignment keeps the `u32` token ids aligned; K/V bytes are
        // copied as bytes so they need none.
        var i: usize = 0;
        while (i < self.block_count) : (i += 1) {
            const b = self.free_blocks[i];
            const off = std.mem.alignForward(u64, b.off, @alignOf(u32));
            const pad = off - b.off;
            if (b.len < pad + size) continue;
            const consumed = pad + size;
            if (b.len == consumed) {
                self.removeBlock(i);
            } else {
                self.free_blocks[i] = .{ .off = off + size, .len = b.len - consumed };
            }
            return .{ .off = off, .len = size };
        }
        return error.OutOfSpace;
    }

    fn journalFree(self: *Memo, off: u64, len: u64) void {
        if (len == 0 or self.block_count >= self.free_blocks.len) return;
        var i: usize = 0;
        while (i < self.block_count and self.free_blocks[i].off < off) : (i += 1) {}
        var j: usize = self.block_count;
        while (j > i) : (j -= 1) self.free_blocks[j] = self.free_blocks[j - 1];
        self.free_blocks[i] = .{ .off = off, .len = len };
        self.block_count += 1;
        if (i > 0) {
            const prev = &self.free_blocks[i - 1];
            if (prev.off + prev.len == self.free_blocks[i].off) {
                prev.len += self.free_blocks[i].len;
                self.removeBlock(i);
                i -= 1;
            }
        }
        if (i + 1 < self.block_count) {
            const cur = &self.free_blocks[i];
            if (cur.off + cur.len == self.free_blocks[i + 1].off) {
                cur.len += self.free_blocks[i + 1].len;
                self.removeBlock(i + 1);
            }
        }
    }

    fn removeBlock(self: *Memo, i: usize) void {
        var j = i;
        while (j + 1 < self.block_count) : (j += 1) self.free_blocks[j] = self.free_blocks[j + 1];
        self.block_count -= 1;
    }

    /// Pack live journal segments to the front, updating their offsets.
    fn compactJournal(self: *Memo) void {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.allocator);
        for (self.segments[0..self.seg_count], 0..) |seg, i| {
            if (seg.alive and seg.region == self.journal and seg.tokenLen() > 0) {
                ids.append(self.allocator, @intCast(i)) catch return;
            }
        }
        std.mem.sort(u32, ids.items, self, lessByOff);
        const jbytes = self.regions.items[self.journal].bytes;
        var write: u64 = 0;
        for (ids.items) |id| {
            const seg = &self.segments[id];
            const len = self.blockBytes(seg.tokenLen());
            if (seg.kv_off != write) {
                std.mem.copyForwards(u8, jbytes[@intCast(write)..@intCast(write + len)], jbytes[@intCast(seg.kv_off)..@intCast(seg.kv_off + len)]);
            }
            seg.kv_off = write;
            write += len;
        }
        self.block_count = 0;
        if (write < jbytes.len) {
            self.free_blocks[0] = .{ .off = write, .len = jbytes.len - write };
            self.block_count = 1;
        }
    }

    fn lessByOff(self: *Memo, a: u32, b: u32) bool {
        return self.segments[a].kv_off < self.segments[b].kv_off;
    }

    fn allocSlot(self: *Memo) !u32 {
        if (self.free_count > 0) {
            self.free_count -= 1;
            return self.free_slots[self.free_count];
        }
        if (self.seg_count >= self.segments.len) return error.MemoFull;
        const id = self.seg_count;
        self.seg_count += 1;
        return id;
    }

    fn findChild(self: *Memo, parent: u32, begin: u32, first_tok: u32) ?u32 {
        var i: usize = 0;
        while (i < self.seg_count) : (i += 1) {
            const seg = &self.segments[i];
            if (!seg.alive or seg.parent != parent or seg.token_begin != begin) continue;
            const t = self.tokensOf(@intCast(i));
            if (t.len == 0 or t[0] != first_tok) continue;
            return @intCast(i);
        }
        return null;
    }

    fn evictOneLeaf(self: *Memo) bool {
        var victim: ?u32 = null;
        var oldest: u64 = std.math.maxInt(u64);
        for (self.segments[0..self.seg_count], 0..) |seg, i| {
            if (!seg.alive or seg.refcount != 0 or seg.children != 0) continue;
            if (seg.last_used <= oldest) {
                oldest = seg.last_used;
                victim = @intCast(i);
            }
        }
        const id = victim orelse return false;
        self.removeSegment(id);
        self.stats.evictions += 1;
        return true;
    }

    fn removeSegment(self: *Memo, id: u32) void {
        const seg = &self.segments[id];
        if (!seg.alive) return;
        const len = self.blockBytes(seg.tokenLen());
        if (seg.region == self.journal) self.journalFree(seg.kv_off, len);
        const parent = seg.parent;
        self.bytes -= @min(self.bytes, len);
        seg.alive = false;
        seg.token_end = seg.token_begin;
        if (parent != NONE and parent < self.seg_count and self.segments[parent].alive) {
            if (self.segments[parent].children > 0) self.segments[parent].children -= 1;
        }
        if (self.free_count < self.free_slots.len) {
            self.free_slots[self.free_count] = id;
            self.free_count += 1;
        }
        self.sync();
    }
};

const TestBuilder = struct {
    fn size(ctx: *anyopaque, begin: u32, end: u32) u64 {
        const self: *const Memo = @ptrCast(@alignCast(ctx));
        return @as(u64, end - begin) * self.per_token_bytes;
    }
    fn write(_: *anyopaque, begin: u32, end: u32, dst: []u8) anyerror!void {
        _ = end;
        for (dst, 0..) |*b, i| b.* = @intCast((begin + @as(u32, @intCast(i))) % 251);
    }
    fn builder(memo: *const Memo) Builder {
        return .{ .ctx = @ptrCast(@constCast(memo)), .size = size, .write = write };
    }
};

fn testMemo(key: Key, per_tok: u64, journal: []u8, meta: []u8) Memo {
    var memo = Memo.init(std.testing.allocator, key, per_tok);
    memo.addMeta(meta) catch unreachable;
    _ = memo.addRegion(journal, true) catch unreachable;
    return memo;
}

test "lookup matches a committed chain and returns its payload" {
    const testing = std.testing;
    var journal: [4096]u8 align(4) = undefined;
    var meta: [1024]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{ .model = 7 }, 4, &journal, &meta);
    defer memo.deinit();

    var tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try memo.commit(&tokens, NONE, 0, tokens.len, TestBuilder.builder(&memo));

    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);
    const matched = try memo.lookup(&tokens, &path);
    try testing.expectEqual(@as(u32, 6), matched);
    // A pure chain is a single radix node.
    try testing.expectEqual(@as(usize, 1), path.items.len);
    try testing.expectEqual(@as(usize, 6), memo.tokensOf(path.items[0]).len);
    try testing.expectEqualSlices(u32, &tokens, memo.tokensOf(path.items[0]));
    memo.unpin(path.items);
}

test "radix matches the maximal common prefix and branches on divergence" {
    const testing = std.testing;
    var journal: [4096]u8 align(4) = undefined;
    var meta: [1024]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();

    var a = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try memo.commit(&a, NONE, 0, a.len, TestBuilder.builder(&memo));

    // Diverges at position 3 (4 vs 9): matches only [0,3), no node split.
    var b = [_]u32{ 1, 2, 3, 9, 9, 9, 9 };
    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);
    const matched = try memo.lookup(&b, &path);
    try testing.expectEqual(@as(u32, 3), matched);
    memo.unpin(path.items);

    // The divergent suffix becomes a child starting at offset 3.
    try memo.commit(&b, path.items[0], 3, b.len, TestBuilder.builder(&memo));

    var p2: std.ArrayList(u32) = .empty;
    defer p2.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 7), try memo.lookup(&b, &p2));
    memo.unpin(p2.items);
    var p3: std.ArrayList(u32) = .empty;
    defer p3.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 6), try memo.lookup(&a, &p3));
    memo.unpin(p3.items);
}

test "payloads and the index are slices into the mapped regions" {
    const testing = std.testing;
    var journal: [1024]u8 align(4) = undefined;
    var meta: [512]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();
    var tokens = [_]u32{ 1, 2, 3, 4 };
    try memo.commit(&tokens, NONE, 0, 4, TestBuilder.builder(&memo));

    const payload = memo.payloadOf(0).?;
    try testing.expect(@intFromPtr(payload.ptr) >= @intFromPtr(&journal) and
        @intFromPtr(payload.ptr) < @intFromPtr(&journal) + journal.len);
    const segs = memo.liveSegments();
    try testing.expect(@intFromPtr(segs.ptr) >= @intFromPtr(&meta) and
        @intFromPtr(segs.ptr) < @intFromPtr(&meta) + meta.len);
    // Token ids alias the same block, after the K/V.
    const toks = memo.tokensOf(0);
    try testing.expectEqualSlices(u32, &tokens, toks);
}

test "commit reuses an already-present prefix (no duplicate node)" {
    const testing = std.testing;
    var journal: [4096]u8 align(4) = undefined;
    var meta: [1024]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();
    var tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try memo.commit(&tokens, NONE, 0, 6, TestBuilder.builder(&memo));
    const before = memo.seg_count;
    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);
    const matched = try memo.lookup(&tokens, &path);
    try testing.expectEqual(@as(u32, 6), matched);
    memo.unpin(path.items);
    try memo.commit(&tokens, path.items[path.items.len - 1], 6, 6, TestBuilder.builder(&memo));
    try testing.expectEqual(before, memo.seg_count);
}

test "LRU evicts unpinned leaves" {
    const testing = std.testing;
    // budget = 2 tokens * (2 kv + 4 tok) = 12 bytes -> one node only.
    var journal: [30]u8 align(4) = undefined;
    var meta: [512]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();
    var a = [_]u32{ 1, 2, 3 };
    try memo.commit(&a, NONE, 0, 3, TestBuilder.builder(&memo));
    var b = [_]u32{ 9, 8, 7 };
    try memo.commit(&b, NONE, 0, 3, TestBuilder.builder(&memo));
    try testing.expect(memo.totalBytes() <= journal.len);
    try testing.expect(memo.stats.evictions > 0);
}

test "index exhaustion is reported, not corrupted" {
    const testing = std.testing;
    var journal: [65536]u8 align(4) = undefined;
    var meta: [256]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();
    var tok = [_]u32{ 1, 2 };
    var i: u32 = 0;
    while (i < 1000) : (i += 1) {
        tok[0] = i;
        _ = memo.commit(&tok, NONE, 0, 2, TestBuilder.builder(&memo)) catch |err| {
            try testing.expectEqual(error.MemoFull, err);
            return;
        };
    }
    return error.TestUnexpectedResult;
}

// The coding workload: a large stable head (system prompt + repo) with a
// volatile tail. A middle edit must keep the head (and the unedited prefix) and
// invalidate only from the edit point on.
test "coding: a middle edit keeps the stable prefix, invalidates the tail" {
    const testing = std.testing;
    var journal: [1 << 16]u8 align(4) = undefined;
    var meta: [8192]u8 align(@alignOf(Segment)) = undefined;
    var memo = testMemo(.{}, 4, &journal, &meta);
    defer memo.deinit();

    // A stable head [0,48) committed on its own (the system prompt), then the
    // first tail [48,64).
    var a: [64]u32 = undefined;
    for (&a, 0..) |*v, i| v.* = @intCast(i + 1);
    const head_len: u32 = 48;
    try memo.commit(&a, NONE, 0, head_len, TestBuilder.builder(&memo));
    var p0: std.ArrayList(u32) = .empty;
    defer p0.deinit(testing.allocator);
    const head_slice: []const u32 = a[0..head_len];
    _ = try memo.lookup(head_slice, &p0);
    try memo.commit(&a, p0.items[p0.items.len - 1], head_len, a.len, TestBuilder.builder(&memo));
    memo.unpin(p0.items);

    // The user edits the middle (position 56): the head + the unedited prefix
    // [0,56) still match; only [56,64) is new.
    const edit: u32 = 56;
    var b: [64]u32 = a;
    b[edit] = 999;
    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);
    try testing.expectEqual(edit, try memo.lookup(&b, &path));
    memo.unpin(path.items);

    // Committing the new tail reuses the head (no duplicate of the stable part).
    var p1: std.ArrayList(u32) = .empty;
    defer p1.deinit(testing.allocator);
    _ = try memo.lookup(&b, &p1);
    try memo.commit(&b, p1.items[p1.items.len - 1], edit, b.len, TestBuilder.builder(&memo));
    memo.unpin(p1.items);

    var p2: std.ArrayList(u32) = .empty;
    defer p2.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 64), try memo.lookup(&b, &p2));
    memo.unpin(p2.items);
    // The original prompt is untouched.
    var p3: std.ArrayList(u32) = .empty;
    defer p3.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 64), try memo.lookup(&a, &p3));
    memo.unpin(p3.items);
}

test "Key: the prompt-format field participates in eql and fingerprint" {
    const testing = std.testing;
    const a = Key{ .model = 1, .arithmetic = 2, .layout = 3, .format = 0, .dtype = 0 };
    const b = Key{ .model = 1, .arithmetic = 2, .layout = 3, .format = 1, .dtype = 0 };
    try testing.expect(Key.eql(a, a));
    try testing.expect(!Key.eql(a, b));
    try testing.expect(a.fingerprint() != b.fingerprint());
}

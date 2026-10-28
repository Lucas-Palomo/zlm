const std = @import("std");

/// A bounded row-LRU for embedding tables. The tuner reserves a byte budget
/// (the plan's `row_cache_bytes`); the runtime caches dequantized rows, one per
/// `(tensor, token id)`, and evicts the least recently used row when the budget
/// is exceeded. The huge table stays mmap-backed: a miss reads and dequantizes
/// the single row on demand (see `executor.embedRow`), and the cache only holds
/// the hottest rows. Determinism is untouched — the cache never changes the
/// value a lookup returns, only whether the row is re-read.
pub const RowCache = struct {
    allocator: std.mem.Allocator,
    budget: u64,
    entries: std.ArrayList(Entry) = .empty,
    used: u64 = 0,
    tick: u64 = 0,

    const Entry = struct {
        tensor: u32,
        row: u32,
        values: []f32,
        last: u64,
    };

    pub fn init(allocator: std.mem.Allocator, budget: u64) RowCache {
        return .{ .allocator = allocator, .budget = budget };
    }

    pub fn deinit(self: *RowCache) void {
        for (self.entries.items) |e| self.allocator.free(e.values);
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Copy the cached row for `(tensor, row)` into `out` and mark it most
    /// recently used. Returns false on a miss or a width mismatch.
    pub fn get(self: *RowCache, tensor: u32, row: u32, out: []f32) bool {
        for (self.entries.items) |*e| {
            if (e.tensor == tensor and e.row == row) {
                if (e.values.len != out.len) return false;
                @memcpy(out, e.values);
                self.tick += 1;
                e.last = self.tick;
                return true;
            }
        }
        return false;
    }

    /// Cache a copy of `values` for `(tensor, row)`, evicting least-recently
    /// used rows until it fits. No-op when the row alone exceeds the whole
    /// budget (it could never be resident). Existing entries are left as-is.
    pub fn put(self: *RowCache, tensor: u32, row: u32, values: []const f32) void {
        const bytes = values.len * @sizeOf(f32);
        if (self.budget == 0 or bytes > self.budget) return;
        if (self.find(tensor, row) != null) return;
        while (self.used + bytes > self.budget and self.entries.items.len > 0) {
            self.evictOldest();
        }
        if (self.used + bytes > self.budget) return;
        const copy = self.allocator.dupe(f32, values) catch return;
        self.tick += 1;
        self.entries.append(self.allocator, .{
            .tensor = tensor,
            .row = row,
            .values = copy,
            .last = self.tick,
        }) catch {
            self.allocator.free(copy);
            return;
        };
        self.used += bytes;
    }

    fn find(self: *const RowCache, tensor: u32, row: u32) ?usize {
        for (self.entries.items, 0..) |e, i| {
            if (e.tensor == tensor and e.row == row) return i;
        }
        return null;
    }

    fn evictOldest(self: *RowCache) void {
        if (self.entries.items.len == 0) return;
        var victim: usize = 0;
        var oldest: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |e, i| {
            if (e.last < oldest) {
                oldest = e.last;
                victim = i;
            }
        }
        const e = self.entries.swapRemove(victim);
        self.used -|= e.values.len * @sizeOf(f32);
        self.allocator.free(e.values);
    }

    pub fn len(self: *const RowCache) usize {
        return self.entries.items.len;
    }

    pub fn usedBytes(self: *const RowCache) u64 {
        return self.used;
    }
};

test "row cache get returns exactly what was put" {
    const testing = std.testing;
    var cache = RowCache.init(testing.allocator, 1024);
    defer cache.deinit();

    const row = [_]f32{ 1.5, -2.0, 0.0, 3.25 };
    cache.put(7, 2, &row);

    var out: [4]f32 = undefined;
    try testing.expect(cache.get(7, 2, &out));
    try testing.expectEqualSlices(f32, &row, &out);

    try testing.expect(!cache.get(7, 3, &out));
    try testing.expect(!cache.get(8, 2, &out));
}

test "row cache evicts least recently used and stays within budget" {
    const testing = std.testing;
    const row_bytes = 4 * @sizeOf(f32);
    var cache = RowCache.init(testing.allocator, 2 * row_bytes);
    defer cache.deinit();

    const a = [_]f32{ 1, 1, 1, 1 };
    const b = [_]f32{ 2, 2, 2, 2 };
    const c = [_]f32{ 3, 3, 3, 3 };

    cache.put(0, 0, &a);
    cache.put(0, 1, &b);
    try testing.expectEqual(@as(usize, 2), cache.len());
    try testing.expectEqual(2 * row_bytes, cache.usedBytes());

    cache.put(0, 2, &c);
    try testing.expectEqual(@as(usize, 2), cache.len());
    try testing.expectEqual(2 * row_bytes, cache.usedBytes());
    try testing.expect(cache.usedBytes() <= cache.budget);

    var out: [4]f32 = undefined;
    try testing.expect(!cache.get(0, 0, &out));
    try testing.expect(cache.get(0, 1, &out));
    try testing.expectEqualSlices(f32, &b, &out);
    try testing.expect(cache.get(0, 2, &out));
    try testing.expectEqualSlices(f32, &c, &out);
}

test "row cache refreshes LRU order on a hit" {
    const testing = std.testing;
    const row_bytes = 4 * @sizeOf(f32);
    var cache = RowCache.init(testing.allocator, 2 * row_bytes);
    defer cache.deinit();

    const a = [_]f32{ 1, 1, 1, 1 };
    const b = [_]f32{ 2, 2, 2, 2 };
    const c = [_]f32{ 3, 3, 3, 3 };

    cache.put(0, 0, &a);
    cache.put(0, 1, &b);
    var out: [4]f32 = undefined;
    try testing.expect(cache.get(0, 0, &out));
    cache.put(0, 2, &c);

    try testing.expect(cache.get(0, 0, &out));
    try testing.expect(!cache.get(0, 1, &out));
    try testing.expect(cache.get(0, 2, &out));
}

test "row cache never stores a row larger than the budget" {
    const testing = std.testing;
    var cache = RowCache.init(testing.allocator, 8);
    defer cache.deinit();

    const row = [_]f32{ 1, 2, 3, 4 };
    cache.put(0, 0, &row);
    try testing.expectEqual(@as(usize, 0), cache.len());
    try testing.expectEqual(@as(u64, 0), cache.usedBytes());
}

test "row cache with zero budget caches nothing" {
    const testing = std.testing;
    var cache = RowCache.init(testing.allocator, 0);
    defer cache.deinit();

    const row = [_]f32{ 1, 2 };
    cache.put(0, 0, &row);
    var out: [2]f32 = undefined;
    try testing.expect(!cache.get(0, 0, &out));
    try testing.expectEqual(@as(usize, 0), cache.len());
}

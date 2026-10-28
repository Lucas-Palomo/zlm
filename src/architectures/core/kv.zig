const std = @import("std");

/// An architecture's KV-cache declaration. It describes *which* layers store a
/// cache and their shapes; it allocates nothing. The tuner resolves it into
/// concrete buffers in the plan; the runtime owns the live cache.
pub const KvLayout = struct {
    /// One entry per transformer layer (index = layer).
    layers: []const Layer,
    /// Bytes per element (f16 = 2): the KV is stored half-precision so the
    /// page cache can hold more streamed weights.
    elem_bytes: u32 = @sizeOf(f16),

    pub const Layer = struct {
        /// KV heads for this layer (GQA).
        kv_heads: u32,
        /// Per-head dimension.
        head_dim: u32,
        /// Sliding-window size (0 = full context).
        sliding_window: u32 = 0,
        /// If false, this layer stores no cache (reuses another layer's).
        stores: bool = true,
        /// Layer whose cache this layer reads (self when it stores).
        source: u32,
    };

    pub fn count(self: KvLayout) u32 {
        return @intCast(self.layers.len);
    }

    /// Byte size of one layer's cache for a given context length:
    /// two tensors (k and v), `kv_heads * head_dim` elements each, per position.
    pub fn layerBytes(self: KvLayout, layer: u32, context_length: u32) u64 {
        if (layer >= self.layers.len) return 0;
        const l = self.layers[layer];
        const per_pos: u64 = @as(u64, l.kv_heads) * l.head_dim;
        return 2 * per_pos * context_length * self.elem_bytes;
    }
};

test "KvLayout counts and sizes" {
    const testing = std.testing;
    const layers = [_]KvLayout.Layer{
        .{ .kv_heads = 4, .head_dim = 256, .source = 0 },
        .{ .kv_heads = 2, .head_dim = 256, .source = 1 },
    };
    const kv = KvLayout{ .layers = &layers };
    try testing.expectEqual(@as(u32, 2), kv.count());
    try testing.expectEqual(@as(u64, 2 * 4 * 256 * 8 * 2), kv.layerBytes(0, 8));
    try testing.expectEqual(@as(u64, 2 * 2 * 256 * 8 * 2), kv.layerBytes(1, 8));
}

const std = @import("std");
const Zmeta = @import("../../../image/zmeta_reader.zig").Zmeta;
const meta = @import("../../core/meta.zig");
const LayerArray = meta.LayerArray;
const BoolArray = meta.BoolArray;

pub const GemmaConfig = struct {
    allocator: std.mem.Allocator,
    architecture: []const u8,
    block_count: u32,
    embedding_length: u32,
    context_length: u32,
    head_count: u32,
    head_count_kv: LayerArray,
    key_length: u32,
    value_length: u32,
    key_length_swa: u32,
    value_length_swa: u32,
    sliding_window: u32,
    sliding_window_pattern: BoolArray,
    layer_norm_rms_epsilon: f32,
    rope_freq_base: f32,
    rope_freq_base_swa: f32,
    rope_dimension_count: u32,
    rope_dimension_count_swa: u32,
    feed_forward_length: LayerArray,
    final_logit_softcapping: ?f32,
    shared_kv_layers: ?u32,
    embedding_length_per_layer_input: u32,

    pub fn deinit(self: *GemmaConfig) void {
        self.head_count_kv.deinit(self.allocator);
        self.sliding_window_pattern.deinit(self.allocator);
        self.feed_forward_length.deinit(self.allocator);
    }

    pub fn headsKv(self: *const GemmaConfig, layer: usize) ?u64 {
        return self.head_count_kv.get(layer);
    }

    pub fn isSliding(self: *const GemmaConfig, layer: usize) bool {
        return self.sliding_window_pattern.get(layer);
    }

    pub fn ffnLength(self: *const GemmaConfig, layer: usize) ?u64 {
        return self.feed_forward_length.get(layer);
    }

    pub fn firstSharedLayer(self: *const GemmaConfig) usize {
        const shared = self.shared_kv_layers orelse 0;
        if (shared >= self.block_count) return 0;
        return self.block_count - shared;
    }

    pub fn kvSourceLayer(self: *const GemmaConfig, layer: usize) usize {
        const first = self.firstSharedLayer();
        if (layer < first) return layer;
        const sliding = self.isSliding(layer);
        var j = first;
        while (j > 0) {
            j -= 1;
            if (self.isSliding(j) == sliding) return j;
        }
        return 0;
    }

    pub fn headDim(self: *const GemmaConfig, layer: usize) u32 {
        if (self.isSliding(layer)) return self.key_length_swa;
        return self.key_length;
    }

    pub fn ropeDim(self: *const GemmaConfig, layer: usize) u32 {
        if (self.isSliding(layer)) return self.rope_dimension_count_swa;
        return self.rope_dimension_count;
    }

    pub fn ropeRotDim(self: *const GemmaConfig, layer: usize) u32 {
        const hd = self.headDim(layer);
        if (self.isSliding(layer)) return hd;
        return hd / 4;
    }

    pub fn ropeBase(self: *const GemmaConfig, layer: usize) f32 {
        if (self.isSliding(layer)) return self.rope_freq_base_swa;
        return self.rope_freq_base;
    }

    /// One distinct RoPE configuration: the key `zcom` regions are deduped on.
    pub const RopeConfig = struct { freq_base: f32, head_dim: u32, n_rot: u32 };

    /// Sentinel for a layer with no RoPE config (e.g. zero dims).
    pub const no_rope_config: u32 = 0xFFFF_FFFF;

    pub fn ropeConfig(self: *const GemmaConfig, layer: usize) RopeConfig {
        return .{
            .freq_base = self.ropeBase(layer),
            .head_dim = self.headDim(layer),
            .n_rot = self.ropeRotDim(layer),
        };
    }

    fn isFirstRopeConfig(self: *const GemmaConfig, layer: usize) bool {
        const c = self.ropeConfig(layer);
        if (c.head_dim == 0 or c.n_rot == 0) return false;
        var j: usize = 0;
        while (j < layer) : (j += 1) {
            const p = self.ropeConfig(j);
            if (p.head_dim != 0 and p.n_rot != 0 and
                p.freq_base == c.freq_base and p.head_dim == c.head_dim and p.n_rot == c.n_rot) return false;
        }
        return true;
    }

    /// The `zcom` rope config id for a layer: the index (order of first
    /// appearance over layers) of its distinct `(freq_base, head_dim, n_rot)`.
    /// The image compiler uses the same assignment, so the runtime can resolve
    /// the serialized `inv_freq` region by this id.
    pub fn ropeConfigId(self: *const GemmaConfig, layer: usize) u32 {
        const target = self.ropeConfig(layer);
        if (target.head_dim == 0 or target.n_rot == 0) return no_rope_config;
        var id: u32 = 0;
        var j: usize = 0;
        while (j < layer) : (j += 1) {
            const c = self.ropeConfig(j);
            if (c.head_dim == 0 or c.n_rot == 0) continue;
            if (c.freq_base == target.freq_base and c.head_dim == target.head_dim and c.n_rot == target.n_rot) return id;
            if (self.isFirstRopeConfig(j)) id += 1;
        }
        return id;
    }

    pub fn fromMetadata(md: *const Zmeta, allocator: std.mem.Allocator) error{ UnsupportedArchitecture, OutOfMemory }!GemmaConfig {
        const arch = md.getString("general.architecture") orelse return error.UnsupportedArchitecture;
        if (!std.mem.eql(u8, arch, "gemma4")) return error.UnsupportedArchitecture;

        const block_count = md.getUint("gemma4.block_count") orelse return error.UnsupportedArchitecture;
        const embedding_length = md.getUint("gemma4.embedding_length") orelse return error.UnsupportedArchitecture;
        const context_length = md.getUint("gemma4.context_length") orelse return error.UnsupportedArchitecture;
        const head_count = md.getUint("gemma4.attention.head_count") orelse return error.UnsupportedArchitecture;
        const key_length = md.getUint("gemma4.attention.key_length") orelse return error.UnsupportedArchitecture;
        const value_length = md.getUint("gemma4.attention.value_length") orelse return error.UnsupportedArchitecture;
        const key_length_swa = md.getUint("gemma4.attention.key_length_swa") orelse key_length;
        const value_length_swa = md.getUint("gemma4.attention.value_length_swa") orelse value_length;
        const sliding_window = md.getUint("gemma4.attention.sliding_window") orelse 0;
        const layer_norm_rms_epsilon = md.getFloat("gemma4.attention.layer_norm_rms_epsilon") orelse 1e-6;
        const freq_base = md.getFloat("gemma4.rope.freq_base") orelse @as(f32, @floatFromInt(md.getUint("gemma4.rope.freq_base") orelse 10000));
        const freq_base_swa = md.getFloat("gemma4.rope.freq_base_swa") orelse freq_base;

        const rope_dimension_count = md.getUint("gemma4.rope.dimension_count") orelse 0;
        const rope_dimension_count_swa = md.getUint("gemma4.rope.dimension_count_swa") orelse rope_dimension_count;

        const softcap = md.getFloat("gemma4.final_logit_softcapping");
        const shared_kv = md.getUint("gemma4.attention.shared_kv_layers");
        const plen: u32 = @intCast(md.getUint("gemma4.embedding_length_per_layer_input") orelse 0);

        var head_count_kv = try meta.readLayerArray(md, allocator, "gemma4.attention.head_count_kv");
        errdefer head_count_kv.deinit(allocator);
        var sliding_window_pattern = try meta.readBoolArray(md, allocator, "gemma4.attention.sliding_window_pattern");
        errdefer sliding_window_pattern.deinit(allocator);
        var feed_forward_length = try meta.readLayerArray(md, allocator, "gemma4.feed_forward_length");
        errdefer feed_forward_length.deinit(allocator);

        return .{
            .allocator = allocator,
            .architecture = arch,
            .block_count = @intCast(block_count),
            .embedding_length = @intCast(embedding_length),
            .context_length = @intCast(context_length),
            .head_count = @intCast(head_count),
            .head_count_kv = head_count_kv,
            .key_length = @intCast(key_length),
            .value_length = @intCast(value_length),
            .key_length_swa = @intCast(key_length_swa),
            .value_length_swa = @intCast(value_length_swa),
            .sliding_window = @intCast(sliding_window),
            .sliding_window_pattern = sliding_window_pattern,
            .layer_norm_rms_epsilon = layer_norm_rms_epsilon,
            .rope_freq_base = freq_base,
            .rope_freq_base_swa = freq_base_swa,
            .rope_dimension_count = @intCast(rope_dimension_count),
            .rope_dimension_count_swa = @intCast(rope_dimension_count_swa),
            .feed_forward_length = feed_forward_length,
            .final_logit_softcapping = softcap,
            .shared_kv_layers = if (shared_kv) |v| @intCast(v) else null,
            .embedding_length_per_layer_input = plen,
        };
    }
};

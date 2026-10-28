const std = @import("std");
const Zmeta = @import("../../../image/zmeta_reader.zig").Zmeta;

/// Qwen3 dense config, read from the image metadata (`qwen3.*`). Uniform across
/// layers: no per-layer arrays (unlike gemma4). Declaration only.
pub const Qwen3Config = struct {
    allocator: std.mem.Allocator,
    architecture: []const u8,
    block_count: u32,
    embedding_length: u32,
    context_length: u32,
    head_count: u32,
    head_count_kv: u32,
    key_length: u32,
    value_length: u32,
    feed_forward_length: u32,
    layer_norm_rms_epsilon: f32,
    rope_freq_base: f32,

    pub fn deinit(_: *Qwen3Config) void {}

    pub fn headDim(self: *const Qwen3Config) u32 {
        return self.key_length;
    }

    pub fn fromMetadata(md: *const Zmeta, allocator: std.mem.Allocator) error{UnsupportedArchitecture}!Qwen3Config {
        const arch = md.getString("general.architecture") orelse return error.UnsupportedArchitecture;
        if (!std.mem.eql(u8, arch, "qwen3")) return error.UnsupportedArchitecture;

        const block_count = md.getUint("qwen3.block_count") orelse return error.UnsupportedArchitecture;
        const embedding_length = md.getUint("qwen3.embedding_length") orelse return error.UnsupportedArchitecture;
        const context_length = md.getUint("qwen3.context_length") orelse return error.UnsupportedArchitecture;
        const feed_forward_length = md.getUint("qwen3.feed_forward_length") orelse return error.UnsupportedArchitecture;
        const head_count = md.getUint("qwen3.attention.head_count") orelse return error.UnsupportedArchitecture;
        const head_count_kv = md.getUint("qwen3.attention.head_count_kv") orelse head_count;
        const key_length = md.getUint("qwen3.attention.key_length") orelse (embedding_length / head_count);
        const value_length = md.getUint("qwen3.attention.value_length") orelse key_length;
        const eps = md.getFloat("qwen3.attention.layer_norm_rms_epsilon") orelse 1e-6;
        const rope_base = md.getFloat("qwen3.rope.freq_base") orelse
            @as(f32, @floatFromInt(md.getUint("qwen3.rope.freq_base") orelse 1_000_000));

        return .{
            .allocator = allocator,
            .architecture = arch,
            .block_count = @intCast(block_count),
            .embedding_length = @intCast(embedding_length),
            .context_length = @intCast(context_length),
            .head_count = @intCast(head_count),
            .head_count_kv = @intCast(head_count_kv),
            .key_length = @intCast(key_length),
            .value_length = @intCast(value_length),
            .feed_forward_length = @intCast(feed_forward_length),
            .layer_norm_rms_epsilon = eps,
            .rope_freq_base = rope_base,
        };
    }
};

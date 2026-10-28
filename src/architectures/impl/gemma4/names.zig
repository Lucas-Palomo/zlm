const std = @import("std");
const model = @import("../../core/contract.zig");
const names = @import("../../core/names.zig");

pub const TensorRole = model.TensorRole;
pub const ParsedTensorName = model.TensorRef;

const globals = [_]names.NameEntry{
    .{ .name = "token_embd.weight", .role = .token_embd },
    .{ .name = "output_norm.weight", .role = .output_norm },
    .{ .name = "output.weight", .role = .output },
    .{ .name = "per_layer_token_embd.weight", .role = .per_layer_token_embd },
    .{ .name = "per_layer_model_proj.weight", .role = .per_layer_model_proj },
    .{ .name = "per_layer_proj_norm.weight", .role = .per_layer_proj_norm },
    .{ .name = "rope_freqs.weight", .role = .rope_freqs },
};

const layered = [_]names.LayerEntry{
    .{ .suffix = "attn_q.weight", .role = .attn_q },
    .{ .suffix = "attn_k.weight", .role = .attn_k },
    .{ .suffix = "attn_v.weight", .role = .attn_v },
    .{ .suffix = "attn_output.weight", .role = .attn_output },
    .{ .suffix = "attn_q_norm.weight", .role = .attn_q_norm },
    .{ .suffix = "attn_k_norm.weight", .role = .attn_k_norm },
    .{ .suffix = "attn_norm.weight", .role = .attn_norm },
    .{ .suffix = "ffn_gate.weight", .role = .ffn_gate },
    .{ .suffix = "ffn_up.weight", .role = .ffn_up },
    .{ .suffix = "ffn_down.weight", .role = .ffn_down },
    .{ .suffix = "ffn_norm.weight", .role = .ffn_norm },
    .{ .suffix = "post_attention_norm.weight", .role = .post_attention_norm },
    .{ .suffix = "post_ffw_norm.weight", .role = .post_ffw_norm },
    .{ .suffix = "layer_output_scale.weight", .role = .layer_output_scale },
    .{ .suffix = "inp_gate.weight", .role = .inp_gate },
    .{ .suffix = "proj.weight", .role = .proj },
    .{ .suffix = "post_norm.weight", .role = .post_norm },
};

pub const Names = names.Tables(&globals, &layered, "blk.");

pub const parseTensorName = Names.parse;
pub const formatLayeredName = Names.format;
pub const globalName = Names.global;

test "parse and format layer names" {
    const std_t = std.testing;
    const p = parseTensorName("blk.12.attn_q.weight").?;
    try std_t.expectEqual(TensorRole.attn_q, p.role);
    try std_t.expectEqual(@as(?u32, 12), p.layer);
    try std_t.expectEqual(TensorRole.token_embd, parseTensorName("token_embd.weight").?.role);
    try std_t.expect(parseTensorName("blk.0.unknown.weight") == null);
    var b: [64]u8 = undefined;
    try std_t.expectEqualStrings("blk.7.ffn_down.weight", formatLayeredName(&b, .ffn_down, 7).?);
}

const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const model = @import("../../core/contract.zig");
const graph = @import("../../core/graph.zig");
const kv = @import("../../core/kv.zig");
const TensorRole = model.TensorRole;
const TensorRef = model.TensorRef;
const kernels = @import("../../../kernels/registry.zig");

pub const GemmaConfig = config.GemmaConfig;

fn matches(a: []const u8) bool {
    return std.mem.eql(u8, a, "gemma4");
}

fn parseTensorName(name: []const u8) ?TensorRef {
    return names.parseTensorName(name);
}

fn globalName(role: TensorRole) ?[]const u8 {
    return names.globalName(role);
}

fn layeredName(buf: []u8, role: TensorRole, layer: u32) ?[]const u8 {
    return names.formatLayeredName(buf, role, layer);
}

fn opForRole(role: TensorRole) ?kernels.Op {
    return switch (role) {
        .attn_q, .attn_k, .attn_v, .attn_output, .ffn_gate, .ffn_up, .ffn_down, .per_layer_model_proj, .inp_gate, .proj => .matvec,
        .attn_norm, .attn_q_norm, .attn_k_norm, .post_attention_norm, .ffn_norm, .post_ffw_norm, .post_norm, .output_norm, .per_layer_proj_norm, .layer_output_scale => .norm,
        .token_embd, .output, .per_layer_token_embd => .dequant,
        else => null,
    };
}

/// Gemma4's global (non-sliding) layers in the larger variants (12B/31B) omit
/// the `attn_v` tensor and share K as V (`k_eq_v`); the same `attn_k`
/// projection fills both the `k` and `v` slots. E2B/E4B have `attn_v` on every
/// layer, so the primary lookup succeeds and this fallback is never consulted.
fn roleFallback(role: TensorRole, layer: u32) ?TensorRole {
    _ = layer;
    return switch (role) {
        .attn_v => .attn_k,
        else => null,
    };
}

fn loadConfig(md: *const @import("../../../image/zmeta_reader.zig").Zmeta, allocator: std.mem.Allocator) anyerror!@import("../../core/root.zig").Config {
    const cfg = try GemmaConfig.fromMetadata(md, allocator);
    return .{ .gemma4 = cfg };
}

fn kvLayout(config_union: *const @import("../../core/root.zig").Config, arena: std.mem.Allocator) anyerror!kv.KvLayout {
    const cfg = switch (config_union.*) {
        .gemma4 => |*c| c,
        else => unreachable,
    };
    const layers = try arena.alloc(kv.KvLayout.Layer, cfg.block_count);
    for (layers, 0..) |*l, i| {
        const stores = i < cfg.firstSharedLayer();
        const source: u32 = if (stores) @intCast(i) else @intCast(cfg.kvSourceLayer(i));
        l.* = .{
            .kv_heads = @intCast(cfg.headsKv(i) orelse cfg.head_count),
            .head_dim = cfg.headDim(i),
            .sliding_window = if (cfg.isSliding(i)) cfg.sliding_window else 0,
            .stores = stores,
            .source = source,
        };
    }
    return .{ .layers = layers };
}

fn layerDims(config_union: *const @import("../../core/root.zig").Config, layer: u32) model.LayerDims {
    const cfg = switch (config_union.*) {
        .gemma4 => |*c| c,
        else => unreachable,
    };
    const hd = cfg.headDim(layer);
    const kv_heads: u32 = @intCast(cfg.headsKv(layer) orelse cfg.head_count);
    const ffn: u32 = @intCast(cfg.ffnLength(layer) orelse cfg.embedding_length);
    const ple = cfg.embedding_length_per_layer_input;
    return .{
        .hidden = cfg.embedding_length,
        .q = cfg.head_count * hd,
        .kv = kv_heads * hd,
        .ffn = ffn,
        .vocab = 0,
        .ple = ple,
        .ple_all = ple * cfg.block_count,
    };
}

fn resolveParams(config_union: *const @import("../../core/root.zig").Config, instr: graph.Instr, layer: u32) graph.Params {
    const cfg = switch (config_union.*) {
        .gemma4 => |*c| c,
        else => unreachable,
    };
    const plen: u32 = cfg.embedding_length_per_layer_input;
    return switch (instr.op) {
        .norm => blk: {
            const role = instr.role orelse break :blk .{ .norm = .{ .eps = cfg.layer_norm_rms_epsilon } };
            break :blk switch (role) {
                .attn_q_norm, .attn_k_norm => .{ .norm = .{
                    .eps = cfg.layer_norm_rms_epsilon,
                    .head_dim = cfg.headDim(layer),
                    .kind = graph.Params.norm_heads,
                } },
                .layer_output_scale => .{ .norm = .{
                    .eps = cfg.layer_norm_rms_epsilon,
                    .kind = graph.Params.norm_scale,
                } },
                .per_layer_proj_norm => .{ .norm = .{
                    .eps = cfg.layer_norm_rms_epsilon,
                    .head_dim = plen,
                    .kind = graph.Params.norm_heads,
                } },
                else => .{ .norm = .{ .eps = cfg.layer_norm_rms_epsilon } },
            };
        },
        .v_norm => .{ .norm = .{
            .eps = cfg.layer_norm_rms_epsilon,
            .head_dim = cfg.headDim(layer),
            .kind = graph.Params.norm_heads_no_weight,
        } },
        .rope => blk: {
            const is_q = if (instr.in) |slot| slot == .q else true;
            break :blk .{ .rope = .{
                .pos = 0,
                .n_rot = cfg.ropeRotDim(layer),
                .freq_base = cfg.ropeBase(layer),
                .head_dim = cfg.headDim(layer),
                .n_heads = if (is_q) cfg.head_count else @intCast(cfg.headsKv(layer) orelse cfg.head_count),
            } };
        },
        .attention, .write_kv => blk: {
            const hd = cfg.headDim(layer);
            break :blk .{ .attention = .{
                .pos = 0,
                .scale = 1.0,
                .sliding = cfg.isSliding(layer),
                .n_heads = cfg.head_count,
                .n_kv_heads = @intCast(cfg.headsKv(layer) orelse cfg.head_count),
                .head_dim = hd,
                .window = if (cfg.isSliding(layer)) cfg.sliding_window else 0,
            } };
        },
        .embd_lookup => blk: {
            const scale = if (instr.role == .per_layer_token_embd)
                @sqrt(@as(f32, @floatFromInt(plen)))
            else
                @sqrt(@as(f32, @floatFromInt(cfg.embedding_length)));
            break :blk .{ .embed = .{ .scale = scale } };
        },
        .activation => .{ .activation = .{ .kind = graph.Params.activation_gelu } },
        .gate_up => .{ .activation = .{ .kind = graph.Params.activation_gelu } },
        .mlp => .{ .activation = .{ .kind = graph.Params.activation_gelu } },
        .softcap => .{ .softcap = .{ .cap = cfg.final_logit_softcapping orelse 0 } },
        .add => .{ .add = .{ .scale = 1.0 } },
        .gelu => .none,
        .mul => .{ .mul = .{ .off = layer * plen } },
        .scale => .{ .scale = .{ .s = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.embedding_length))) } },
        .combine => .{ .combine = .{ .s = 1.0 / @sqrt(@as(f32, 2.0)) } },
        else => .none,
    };
}

pub const arch = model.Arch{
    .name = "gemma4",
    .matches = matches,
    .parseTensorName = parseTensorName,
    .opForRole = opForRole,
    .globalName = globalName,
    .layeredName = layeredName,
    .roleFallback = roleFallback,
    .loadConfig = loadConfig,
    .block = @import("block.zig").block,
    .kvLayout = kvLayout,
    .resolveParams = resolveParams,
    .layerDims = layerDims,
    .chat = @import("chat.zig").template,
    .tools = @import("tools.zig").dialect,
};

test "roleFallback reuses attn_k only for attn_v" {
    const testing = std.testing;
    try testing.expectEqual(@as(?TensorRole, .attn_k), roleFallback(.attn_v, 5));
    try testing.expectEqual(@as(?TensorRole, null), roleFallback(.attn_q, 5));
    try testing.expectEqual(@as(?TensorRole, null), roleFallback(.ffn_gate, 0));
    try testing.expectEqual(@as(?TensorRole, null), roleFallback(.attn_output, 11));
}

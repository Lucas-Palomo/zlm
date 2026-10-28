const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const model = @import("../../core/contract.zig");
const graph = @import("../../core/graph.zig");
const kv = @import("../../core/kv.zig");
const kernels = @import("../../../kernels/registry.zig");
const TensorRole = model.TensorRole;
const TensorRef = model.TensorRef;

pub const Qwen3Config = config.Qwen3Config;

fn matches(a: []const u8) bool {
    return std.mem.eql(u8, a, "qwen3");
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
        .attn_q, .attn_k, .attn_v, .attn_output, .ffn_gate, .ffn_up, .ffn_down => .matvec,
        .attn_norm, .attn_q_norm, .attn_k_norm, .ffn_norm, .output_norm => .norm,
        .token_embd, .output => .dequant,
        else => null,
    };
}

fn loadConfig(md: *const @import("../../../image/zmeta_reader.zig").Zmeta, allocator: std.mem.Allocator) anyerror!@import("../../core/root.zig").Config {
    const cfg = try Qwen3Config.fromMetadata(md, allocator);
    return .{ .qwen3 = cfg };
}

fn kvLayout(config_union: *const @import("../../core/root.zig").Config, arena: std.mem.Allocator) anyerror!kv.KvLayout {
    const cfg = switch (config_union.*) {
        .qwen3 => |*c| c,
        else => unreachable,
    };
    const layers = try arena.alloc(kv.KvLayout.Layer, cfg.block_count);
    for (layers, 0..) |*l, i| {
        l.* = .{
            .kv_heads = cfg.head_count_kv,
            .head_dim = cfg.headDim(),
            .sliding_window = 0,
            .stores = true,
            .source = @intCast(i),
        };
    }
    return .{ .layers = layers };
}

fn layerDims(config_union: *const @import("../../core/root.zig").Config, layer: u32) model.LayerDims {
    _ = layer;
    const cfg = switch (config_union.*) {
        .qwen3 => |*c| c,
        else => unreachable,
    };
    const hd = cfg.headDim();
    return .{
        .hidden = cfg.embedding_length,
        .q = cfg.head_count * hd,
        .kv = cfg.head_count_kv * hd,
        .ffn = cfg.feed_forward_length,
        .vocab = 0,
        .ple = 0,
        .ple_all = 0,
    };
}

fn resolveParams(config_union: *const @import("../../core/root.zig").Config, instr: graph.Instr, layer: u32) graph.Params {
    _ = layer;
    const cfg = switch (config_union.*) {
        .qwen3 => |*c| c,
        else => unreachable,
    };
    const hd = cfg.headDim();
    return switch (instr.op) {
        .norm => blk: {
            const role = instr.role orelse break :blk .{ .norm = .{ .eps = cfg.layer_norm_rms_epsilon } };
            break :blk switch (role) {
                .attn_q_norm, .attn_k_norm => .{ .norm = .{
                    .eps = cfg.layer_norm_rms_epsilon,
                    .head_dim = hd,
                    .kind = graph.Params.norm_heads,
                } },
                else => .{ .norm = .{ .eps = cfg.layer_norm_rms_epsilon } },
            };
        },
        .rope => blk: {
            const is_q = if (instr.in) |slot| slot == .q else true;
            break :blk .{ .rope = .{
                .pos = 0,
                .n_rot = hd,
                .freq_base = cfg.rope_freq_base,
                .head_dim = hd,
                .n_heads = if (is_q) cfg.head_count else cfg.head_count_kv,
            } };
        },
        .attention, .write_kv => .{ .attention = .{
            .pos = 0,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
            .sliding = false,
            .n_heads = cfg.head_count,
            .n_kv_heads = cfg.head_count_kv,
            .head_dim = hd,
            .window = 0,
        } },
        .activation => .{ .activation = .{ .kind = graph.Params.activation_silu } },
        .gate_up => .{ .activation = .{ .kind = graph.Params.activation_silu } },
        .mlp => .{ .activation = .{ .kind = graph.Params.activation_silu } },
        .add => .{ .add = .{ .scale = 1.0 } },
        .embd_lookup => .none,
        else => .none,
    };
}

pub const arch = model.Arch{
    .name = "qwen3",
    .matches = matches,
    .parseTensorName = parseTensorName,
    .opForRole = opForRole,
    .globalName = globalName,
    .layeredName = layeredName,
    .loadConfig = loadConfig,
    .block = @import("block.zig").block,
    .kvLayout = kvLayout,
    .resolveParams = resolveParams,
    .layerDims = layerDims,
    .chat = @import("chat.zig").template,
};

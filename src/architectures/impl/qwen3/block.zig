const g = @import("../../core/graph.zig");

/// qwen3's forward, declared as a semantic instruction sequence (decode, single
/// token). No forward code. Params are placeholders here; the tuner resolves
/// them via `arch.resolveParams`.
///
/// Head: token_embd lookup (no embedding scale).
///
/// Per layer (uniform):
///   attn_norm -> q/k/v -> q_norm/k_norm (per-head) -> rope -> write_kv ->
///   attention -> attn_output -> add(hidden) ->
///   ffn_norm -> gate/up -> silu -> down -> add(hidden)
///
/// Tail: output_norm -> lm_head (tied to token_embd when `output.weight` is
/// absent).
pub const head = [_]g.Instr{
    .{ .op = .embd_lookup, .role = .token_embd, .out = .hidden },
};

pub const layer = [_]g.Instr{
    // attention block
    .{ .op = .norm, .role = .attn_norm, .in = .hidden, .out = .normed },
    .{ .op = .matvec, .role = .attn_q, .in = .normed, .out = .q },
    .{ .op = .matvec, .role = .attn_k, .in = .normed, .out = .k, .store_only = true },
    .{ .op = .matvec, .role = .attn_v, .in = .normed, .out = .v, .store_only = true },
    .{ .op = .norm, .role = .attn_q_norm, .in = .q, .out = .q },
    .{ .op = .norm, .role = .attn_k_norm, .in = .k, .out = .k, .store_only = true },
    .{ .op = .rope, .in = .q, .out = .q },
    .{ .op = .rope, .in = .k, .out = .k, .store_only = true },
    .{ .op = .write_kv, .in = .k, .in2 = .v, .store_only = true },
    .{ .op = .attention, .in = .q, .out = .attn_out },
    .{ .op = .matvec, .role = .attn_output, .in = .attn_out, .out = .normed },
    .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden },
    // MLP block
    .{ .op = .norm, .role = .ffn_norm, .in = .hidden, .out = .normed },
    .{ .op = .matvec, .role = .ffn_gate, .in = .normed, .out = .ffn_a },
    .{ .op = .matvec, .role = .ffn_up, .in = .normed, .out = .ffn_b },
    .{ .op = .activation, .in = .ffn_a, .in2 = .ffn_b, .out = .ffn_mid },
    .{ .op = .matvec, .role = .ffn_down, .in = .ffn_mid, .out = .normed },
    .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden },
};

pub const tail = [_]g.Instr{
    .{ .op = .norm, .role = .output_norm, .in = .hidden, .out = .normed },
    .{ .op = .lm_head, .role = .output, .in = .normed, .out = .logits },
};

pub const block = g.Block{
    .slots = &.{ .hidden, .normed, .q, .k, .v, .attn_out, .ffn_a, .ffn_b, .ffn_mid, .logits },
    .head = &head,
    .layer = &layer,
    .tail = &tail,
};

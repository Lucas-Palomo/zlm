const g = @import("../../core/graph.zig");
const contract = @import("../../core/contract.zig");
const Op = g.Op;
const Slot = g.Slot;

/// gemma4's forward, declared as a semantic instruction sequence (decode,
/// single token). No forward code: the tuner compiles this into Dispatch[].
/// Params are placeholders here; the tuner resolves them from the config via
/// `arch.resolveParams`.
///
/// Head: token_embd lookup; when PLE is present, per_layer_token_embd lookup,
/// per_layer_model_proj, scale, per_layer_proj_norm and combine set up
/// `ple_combined`.
///
/// Per layer:
///   attn_norm -> q/k/v -> qk_norm -> rope -> write_kv -> attention ->
///   attn_output -> post_attention_norm -> add(hidden) -> ffn_norm ->
///   gate/up -> activation -> down -> post_ffw_norm -> add(hidden) ->
///   [PLE: inp_gate -> gelu -> mul(ple_combined) -> proj -> post_norm ->
///   add(hidden)] -> layer_output_scale
pub const head = [_]g.Instr{
    .{ .op = .embd_lookup, .role = .token_embd, .out = .hidden },
    .{ .op = .embd_lookup, .role = .per_layer_token_embd, .out = .ple_token, .gate = .ple },
    .{ .op = .matvec, .role = .per_layer_model_proj, .in = .hidden, .out = .ple_projected, .gate = .ple },
    .{ .op = .scale, .in = .ple_projected, .out = .ple_projected, .gate = .ple },
    .{ .op = .norm, .role = .per_layer_proj_norm, .in = .ple_projected, .out = .ple_projected, .gate = .ple },
    .{ .op = .combine, .in = .ple_projected, .in2 = .ple_token, .out = .ple_combined, .gate = .ple },
};

pub const layer = [_]g.Instr{
    // attention block
    .{ .op = .norm, .role = .attn_norm, .in = .hidden, .out = .normed },
    .{ .op = .matvec, .role = .attn_q, .in = .normed, .out = .q },
    .{ .op = .matvec, .role = .attn_k, .in = .normed, .out = .k, .store_only = true },
    .{ .op = .matvec, .role = .attn_v, .in = .normed, .out = .v, .store_only = true },
    .{ .op = .norm, .role = .attn_q_norm, .in = .q, .out = .q },
    .{ .op = .norm, .role = .attn_k_norm, .in = .k, .out = .k, .store_only = true },
    .{ .op = .v_norm, .in = .v, .out = .v, .store_only = true },
    .{ .op = .rope, .in = .q, .out = .q },
    .{ .op = .rope, .in = .k, .out = .k, .store_only = true },
    .{ .op = .write_kv, .in = .k, .in2 = .v, .store_only = true },
    .{ .op = .attention, .in = .q, .out = .attn_out },
    .{ .op = .matvec, .role = .attn_output, .in = .attn_out, .out = .normed },
    .{ .op = .norm, .role = .post_attention_norm, .in = .normed, .out = .normed },
    // residual: hidden = hidden + post_attention_norm(attn_out)
    .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden },
    // MLP block
    .{ .op = .norm, .role = .ffn_norm, .in = .hidden, .out = .normed },
    .{ .op = .matvec, .role = .ffn_gate, .in = .normed, .out = .ffn_a },
    .{ .op = .matvec, .role = .ffn_up, .in = .normed, .out = .ffn_b },
    .{ .op = .activation, .in = .ffn_a, .in2 = .ffn_b, .out = .ffn_mid },
    .{ .op = .matvec, .role = .ffn_down, .in = .ffn_mid, .out = .normed },
    .{ .op = .norm, .role = .post_ffw_norm, .in = .normed, .out = .normed },
    // residual: hidden = hidden + ffn_out
    .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden },
    .{ .op = .matvec, .role = .inp_gate, .in = .hidden, .out = .ple_a, .gate = .ple },
    .{ .op = .gelu, .in = .ple_a, .out = .ple_a, .gate = .ple },
    .{ .op = .mul, .in = .ple_a, .in2 = .ple_combined, .out = .ple_a, .gate = .ple },
    .{ .op = .matvec, .role = .proj, .in = .ple_a, .out = .normed, .gate = .ple },
    .{ .op = .norm, .role = .post_norm, .in = .normed, .out = .normed, .gate = .ple },
    .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden, .gate = .ple },
    // per-layer output scale
    .{ .op = .norm, .role = .layer_output_scale, .in = .hidden, .out = .hidden },
};

pub const tail = [_]g.Instr{
    .{ .op = .norm, .role = .output_norm, .in = .hidden, .out = .normed },
    .{ .op = .lm_head, .role = .output, .in = .normed, .out = .logits },
    .{ .op = .softcap, .in = .logits, .out = .logits },
};

pub const block = g.Block{
    .slots = &.{ .hidden, .normed, .q, .k, .v, .attn_out, .ffn_a, .ffn_b, .ffn_mid, .logits, .ple_token, .ple_projected, .ple_combined, .ple_a },
    .head = &head,
    .layer = &layer,
    .tail = &tail,
};

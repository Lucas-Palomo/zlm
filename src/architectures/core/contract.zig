const std = @import("std");
const Zmeta = @import("../../image/zmeta_reader.zig").Zmeta;
const kernels = @import("../../kernels/registry.zig");
pub const ZmetaType = Zmeta;

/// The contract every architecture implements (declaration only).
///
/// An `Arch` is a comptime bundle: it declares the mapping from tensor
/// names/roles to kernel `Op`s and how to load its `Config`. It has no forward
/// logic — the forward *is* the plan.
pub const Arch = struct {
    /// Stable id used in metadata/`zman` (e.g. "gemma4").
    name: []const u8,
    /// Does this metadata belong to this architecture?
    matches: *const fn (arch: []const u8) bool,
    /// Parse a tensor name into a role (+ layer). Shared `TensorRole` space.
    parseTensorName: *const fn (name: []const u8) ?TensorRef,
    /// Which kernel op consumes a role (bootstrap mapping).
    opForRole: *const fn (role: TensorRole) ?kernels.Op,
    /// Tensor name for a global role, or null if the role is not global.
    globalName: *const fn (role: TensorRole) ?[]const u8,
    /// Tensor name for a layered role in a given layer, or null.
    layeredName: *const fn (buf: []u8, role: TensorRole, layer: u32) ?[]const u8,
    /// Alternate role to resolve when a layer has no tensor for `role` (e.g.
    /// gemma4's k_eq_v layers reuse `attn_k` as `attn_v`). Consulted by the
    /// tuner only after the primary lookup returns no tensor; null means no
    /// fallback. Leave unset for architectures whose roles are all present.
    roleFallback: ?*const fn (role: TensorRole, layer: u32) ?TensorRole = null,
    /// Load the architecture config from the image metadata. Returns the shared
    /// `model.Config` union (defined in `model/root.zig` to avoid a cycle).
    loadConfig: *const fn (md: *const Zmeta, allocator: std.mem.Allocator) anyerror!@import("root.zig").Config,
    /// The model's forward, declared as a semantic instruction sequence.
    block: @import("graph.zig").Block,
    /// Resolve the KV-cache declaration from the config (built into `arena`).
    kvLayout: *const fn (cfg: *const @import("root.zig").Config, arena: std.mem.Allocator) anyerror!@import("kv.zig").KvLayout,
    /// Resolve an instruction's params (eps, rope base/dim, attention scale,
    /// softcap) from the config, for a given op/role/layer.
    resolveParams: *const fn (cfg: *const @import("root.zig").Config, instr: @import("graph.zig").Instr, layer: u32) @import("graph.zig").Params,
    /// Resolve a layer's activation dims (element counts in f32) so the tuner
    /// can set each dispatch's operand slice lengths. `vocab` is global and may
    /// be 0 when the config does not carry it (the caller fills it in).
    layerDims: *const fn (cfg: *const @import("root.zig").Config, layer: u32) LayerDims,
    /// The architecture's single-turn chat prompt shape. The runtime fills in
    /// the user text; the edge never builds a prompt.
    chat: @import("chat.zig").Template = .{},
    /// The architecture's native tool-call dialect, when it declares one. The
    /// runtime and the edge move structured calls; the dialect owns the literal
    /// prompt syntax (declaration only).
    tools: ?@import("tools.zig").Dialect = null,
};

/// Per-layer activation dimensions (element counts in f32) shared across
/// architectures; the tuner maps `graph.Slot` to one of these.
pub const LayerDims = struct {
    hidden: u32 = 0,
    q: u32 = 0,
    kv: u32 = 0,
    ffn: u32 = 0,
    vocab: u32 = 0,
    /// Per-layer embedding width (`embedding_length_per_layer_input`); 0 when
    /// the architecture has no PLE.
    ple: u32 = 0,
    /// `block_count * ple`; the width of the global PLE buffers.
    ple_all: u32 = 0,
};

/// Generic parsed tensor name. Every model maps into this space so the tuner/
/// scheduler can reason without knowing the model.
pub const TensorRef = struct {
    role: TensorRole,
    layer: ?u32 = null,
};

/// The shared role vocabulary. A model only uses a subset.
pub const TensorRole = enum {
    token_embd,
    output_norm,
    output,
    attn_q,
    attn_k,
    attn_v,
    attn_output,
    attn_q_norm,
    attn_k_norm,
    attn_norm,
    ffn_gate,
    ffn_up,
    ffn_down,
    ffn_norm,
    post_attention_norm,
    post_ffw_norm,
    layer_output_scale,
    per_layer_token_embd,
    per_layer_model_proj,
    per_layer_proj_norm,
    rope_freqs,
    inp_gate,
    proj,
    post_norm,
    attn_q_bias,
    attn_k_bias,
    attn_v_bias,
};

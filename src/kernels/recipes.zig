const api = @import("api.zig");
const matvec = @import("linear/matvec/root.zig");
const quant = @import("linear/quant/root.zig");
const norm = @import("transformer/norm.zig");
const attention = @import("transformer/attention.zig");

/// Execution form of a recipe.
pub const Form = enum { fused, seq };

/// A named composition the Composer may emit. `op` is the owning family and
/// `atomic_id` its atomic that implements the composed form; `ops` are the
/// model ops the recipe replaces.
pub const Recipe = struct {
    name: []const u8,
    form: Form,
    op: api.Op,
    atomic_id: u16,
    ops: []const api.Op,
    bytes_saved: u32 = 0,
    tier: api.Tier = .cpu,
};

/// Declared recipes (data only; wired by the family once implemented).
/// `bytes_saved` is a coarse count of eliminated round-trip passes, not a
/// shape-specific byte figure.
pub const catalog = [_]Recipe{
    .{
        .name = "gate_up_fused",
        .form = .fused,
        .op = .matvec,
        .atomic_id = @intFromEnum(matvec.AtomicId.gate_up_fused),
        .ops = &[_]api.Op{ .matvec, .matvec, .activation },
        .bytes_saved = 2,
    },
    .{
        .name = "mlp_fused",
        .form = .fused,
        .op = .matvec,
        .atomic_id = @intFromEnum(matvec.AtomicId.mlp_fused),
        .ops = &[_]api.Op{ .matvec, .matvec, .activation, .matvec },
        .bytes_saved = 3,
    },
    .{
        .name = "qk_norm_rope",
        .form = .fused,
        .op = .norm,
        .atomic_id = @intFromEnum(norm.AtomicId.qk_norm_rope_heads),
        .ops = &[_]api.Op{ .norm, .rope },
        .bytes_saved = 1,
    },
    .{
        .name = "norm_proj",
        .form = .fused,
        .op = .matvec,
        .atomic_id = @intFromEnum(matvec.AtomicId.norm_proj),
        .ops = &[_]api.Op{ .norm, .matvec },
        .bytes_saved = 2,
    },
    .{
        .name = "rope_write_kv",
        .form = .fused,
        .op = .attention,
        .atomic_id = @intFromEnum(attention.AtomicId.rope_write_kv),
        .ops = &[_]api.Op{ .rope, .attention },
        .bytes_saved = 1,
    },
    .{
        .name = "add_norm",
        .form = .fused,
        .op = .norm,
        .atomic_id = @intFromEnum(norm.AtomicId.add_norm),
        .ops = &[_]api.Op{ .elementwise, .norm },
        .bytes_saved = 2,
    },
    .{
        .name = "dequant_scale",
        .form = .fused,
        .op = .dequant,
        .atomic_id = @intFromEnum(quant.AtomicId.dequant_scale),
        .ops = &[_]api.Op{ .dequant, .elementwise },
        .bytes_saved = 1,
    },
    .{
        .name = "matvec_bias",
        .form = .fused,
        .op = .matvec,
        .atomic_id = @intFromEnum(matvec.AtomicId.matvec_bias),
        .ops = &[_]api.Op{ .matvec, .elementwise },
        .bytes_saved = 1,
    },
};

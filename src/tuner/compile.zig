const std = @import("std");
const plan = @import("../plan/root.zig");
const graph = @import("../architectures/core/graph.zig");
const contract = @import("../architectures/core/contract.zig");
const kernels = @import("../kernels/registry.zig");

const Binding = plan.Binding;
const Dispatch = plan.Dispatch;
const Buffer = plan.Buffer;
const Op = graph.Op;
const Slot = graph.Slot;

/// Resolve a weight tensor name to its index in the plan's tensor table.
pub const Lookup = *const fn (ctx: *anyopaque, name: []const u8) ?u32;

/// Called before resolving a tensor for `layer`, so a name-based lookup can
/// map the block's layer-0 declaration to the concrete tensor.
pub const SetLayer = *const fn (ctx: *anyopaque, layer: u32) void;

pub const LayerLayout = struct {
    count: u32,
    kv: []const u32,

    pub fn layerOf(self: LayerLayout, layer: u32) u32 {
        return if (layer < self.kv.len) layer else 0;
    }

    pub fn kvOf(self: LayerLayout, layer: u32) u32 {
        return self.kv[self.layerOf(layer)];
    }
};

/// Map a semantic op to a kernel `AtomicId` number for this plan's codec.
pub const AtomicFn = *const fn (op: Op) u16;

/// Resolve a layer's activation dims (from the arch config, or a test fake).
pub const LayerDimsFn = *const fn (ctx: ?*anyopaque, layer: u32) contract.LayerDims;

/// Resolve instruction params from the config (identity when no config).
pub const ConfigRef = @import("../architectures/core/root.zig").Config;

pub const Options = struct {
    /// The architecture whose `globalName`/`layeredName` resolve tensor names.
    arch: contract.Arch,
    /// Number of layers to emit (the model's `block_count`).
    layer_count: u32,
    /// Threads / parallelism recorded on each dispatch (`parts`).
    threads: u16 = 0,
    /// Activation slot byte sizes (index = `@intFromEnum(Slot)`); emitted as
    /// `plan.Buffer[]`. Empty means "no buffers".
    slot_bytes: []const u64 = &.{},
    /// Extra buffers appended after the slots (e.g. KV caches); their plan
    /// buffer index is `slot_bytes.len + i`.
    extra_buffers: []const u64 = &.{},
    /// Resolve a weight tensor name to its index in the plan's tensor table.
    lookup: Lookup = noLookup,
    lookup_ctx: ?*anyopaque = null,
    /// Optional hook called with the current layer before each name lookup.
    set_layer: ?SetLayer = null,
    /// Layer/KV mapping; when set, tensor + KV indices are resolved per layer.
    layer_layout: ?*const LayerLayout = null,
    /// Map a `graph.Op` to a kernel `AtomicId` number.
    atomic: AtomicFn = defaultAtomic,
    /// When set, instruction params are resolved from this config via the
    /// arch's `resolveParams`; otherwise the block's placeholder params are kept.
    config: ?*const ConfigRef = null,
    /// layer -> KV buffer index (or NONE); set on write_kv/attention dispatches.
    kv_index: []const u32 = &.{},
    /// Positions the KV cache holds; copied onto every dispatch (`max_seq`).
    max_seq: u32 = 0,
    /// layer -> `zcom` rope config id (or NONE); set on rope dispatches.
    rope_config: []const u32 = &.{},
    /// layer -> whether the layer stores a KV cache. `store_only` instructions
    /// are dropped for layers that do not (shared KV): the runtime only sees
    /// the minimum the layer needs.
    kv_stores: []const bool = &.{},
    /// Per-layer activation dims used to set each dispatch's `in_len`/`out_len`.
    /// When unset, the arch's `layerDims` is used (with `config`); when no
    /// config exists either, lengths stay 0 ("whole buffer").
    layer_dims: ?LayerDimsFn = null,
    layer_dims_ctx: ?*anyopaque = null,
    /// Emit the batched prefill forward instead of the decode one: projections
    /// become `.matmul` and dispatches are tagged `phase = prefill`. The block
    /// passed in is expected to be the unfused declaration (the Composer's
    /// gate/up fusion is decode-only).
    prefill: bool = false,
};

pub const Result = struct {
    dispatch: []Dispatch,
    buffers: []Buffer,
};

/// Compile an architecture's declared forward (`Block`) into an ordered
/// `Dispatch[]` + the `Buffer[]` the runtime must allocate. Pure: it reads the
/// block and the inventory and writes into the allocator; decides nothing.
pub fn compileBlock(allocator: std.mem.Allocator, block: graph.Block, opts: Options) !Result {
    var dispatch: std.ArrayList(Dispatch) = .empty;
    errdefer dispatch.deinit(allocator);

    try emit(allocator, &dispatch, block.head, 0, opts);
    var layer: u32 = 0;
    while (layer < opts.layer_count) : (layer += 1) {
        try emit(allocator, &dispatch, block.layer, layer, opts);
    }
    try emit(allocator, &dispatch, block.tail, 0, opts);

    const total = opts.slot_bytes.len + opts.extra_buffers.len;
    const buffers = try allocator.alloc(Buffer, total);
    for (opts.slot_bytes, 0..) |bytes, i| {
        buffers[i] = .{ .slot = @intCast(i), .bytes = bytes };
    }
    for (opts.extra_buffers, opts.slot_bytes.len..) |bytes, i| {
        buffers[i] = .{ .slot = @intCast(i), .bytes = bytes };
    }

    return .{ .dispatch = try dispatch.toOwnedSlice(allocator), .buffers = buffers };
}

fn emit(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(Dispatch),
    instrs: []const graph.Instr,
    layer: u32,
    opts: Options,
) !void {
    for (instrs) |instr| {
        const dims = resolveLayerDims(layer, opts);
        switch (instr.gate) {
            .always => {},
            .ple => if (dims.ple == 0) continue,
            .no_ple => if (dims.ple != 0) continue,
        }
        if (instr.store_only and layer < opts.kv_stores.len and !opts.kv_stores[layer]) continue;
        const params = if (opts.config) |c| opts.arch.resolveParams(c, instr, layer) else instr.params;
        var d = Dispatch{
            .op = dispatchOp(instr.op, opts.prefill),
            .atomic_id = opts.atomic(instr.op),
            .phase = if (opts.prefill) 1 else 0,
            .parts = partsFor(instr, dims, opts.threads),
            .layer = layer,
            .sem_op = params.code(),
            .params = packParams(params),
            .max_seq = opts.max_seq,
            .in_len = instrLen(dims, instr.in, instr.op),
            .out_len = instrLen(dims, instr.out, instr.op),
        };
        switch (params) {
            .activation => |a| {
                if (instr.op != .gate_up and instr.op != .mlp) d.atomic_id = activationAtomic(a.kind);
            },
            .norm => |n| {
                if (instr.fuse_rope and n.kind == graph.Params.norm_heads) {
                    d.atomic_id = normFusedAtomic();
                    if (layer < opts.rope_config.len) d.rope_cfg = opts.rope_config[layer];
                } else {
                    d.atomic_id = normAtomic(n.kind);
                }
            },
            else => {},
        }
        if (instr.op == .rope and layer < opts.rope_config.len) {
            d.rope_cfg = opts.rope_config[layer];
        }
        if (opts.set_layer) |hook| {
            if (opts.lookup_ctx) |lc| hook(lc, layer);
        }
        if (instr.role) |role| {
            d.tensor = resolveTensor(role, layer, opts);
            // Tied embeddings: models without a separate `output.weight` reuse
            // the token embedding matrix for the lm_head.
            if (d.tensor == plan.NONE and instr.op == .lm_head) {
                d.tensor = resolveTensor(.token_embd, layer, opts);
            }
        }
        if (instr.role2) |role2| d.tensor2 = resolveTensor(role2, layer, opts);
        if (instr.role3) |role3| d.tensor3 = resolveTensor(role3, layer, opts);
        if (instr.in) |slot| d.in = bindBuffer(slot);
        if (instr.in2) |slot| d.in.ref2 = @intFromEnum(slot);
        if (instr.out) |slot| d.out = bindBuffer(slot);
        if (instr.op == .write_kv or instr.op == .attention) {
            if (opts.layer_layout) |l| {
                d.kv = l.kvOf(layer);
            } else if (layer < opts.kv_index.len) {
                d.kv = opts.kv_index[layer];
            }
        }
        try list.append(allocator, d);
    }
}

/// First-cut grain: minimum MACs per part before coordination pays off.
const grain_work: u64 = 1_000_000;

/// Scheduler's parallelism decision for one instruction: `parts = clamp(units /
/// grain, 1, budget)`, and `1` for op classes the runtime cannot split yet.
fn partsFor(instr: graph.Instr, dims: contract.LayerDims, budget: u16) u16 {
    if (budget <= 1) return 1;
    if (!kernels.rowParallel(kernelOp(instr.op))) return 1;
    const out_slot = instr.out orelse return 1;
    const in_slot = instr.in orelse return 1;
    const rows: u64 = slotLen(dims, out_slot);
    const cols: u64 = slotLen(dims, in_slot);
    if (rows == 0 or cols == 0) return 1;
    var parts = (rows * cols) / grain_work;
    if (parts < 1) parts = 1;
    if (parts > budget) parts = budget;
    if (parts > rows) parts = rows;
    return @intCast(parts);
}

fn resolveLayerDims(layer: u32, opts: Options) contract.LayerDims {
    if (opts.layer_dims) |f| return f(opts.layer_dims_ctx, layer);
    if (opts.config) |c| return opts.arch.layerDims(c, layer);
    return .{};
}

/// Element count (f32) of a slot for a layer: hidden/q/kv/ffn/vocab.
fn slotLen(dims: contract.LayerDims, slot: Slot) u32 {
    return switch (slot) {
        .hidden, .normed, .emb_row => dims.hidden,
        .q, .attn_out => dims.q,
        .k, .v => dims.kv,
        .ffn_a, .ffn_b, .ffn_mid => dims.ffn,
        .logits => dims.vocab,
        .ple_a => dims.ple,
        .ple_token, .ple_projected, .ple_combined => dims.ple_all,
    };
}

/// Slice length for an operand, or 0 (whole buffer) for globals: KV stores keep
/// the buffer as-is. Embedding lookups carry the per-row width so the batched
/// prefill and the M-row buffers slice a single row (decode reads row 0).
fn instrLen(dims: contract.LayerDims, slot: ?Slot, dop: Op) u32 {
    if (dop == .write_kv) return 0;
    return slotLen(dims, slot orelse return 0);
}

fn bindBuffer(slot: Slot) Binding {
    return .{ .kind = .buffer, .ref = @intFromEnum(slot) };
}

/// Pack `graph.Params` into the plan's fixed `[8]u32` blob (floats bit-cast).
/// The executor unpacks by `sem_op`; the tuner is the only packer.
pub fn packParams(p: graph.Params) [8]u32 {
    const bits = struct {
        fn f(x: f32) u32 {
            return @bitCast(x);
        }
    };
    const zero: [8]u32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    return switch (p) {
        .none => zero,
        .norm => |n| .{ bits.f(n.eps), n.head_dim, n.kind, 0, 0, 0, 0, 0 },
        .rope => |r| .{ r.pos, r.n_rot, bits.f(r.freq_base), r.head_dim, r.n_heads, 0, 0, 0 },
        .attention => |a| .{ a.pos, bits.f(a.scale), @intFromBool(a.sliding), a.n_heads, a.n_kv_heads, a.head_dim, a.t_start, a.window },
        .activation => |a| .{ a.kind, 0, 0, 0, 0, 0, 0, 0 },
        .softcap => |s| .{ bits.f(s.cap), 0, 0, 0, 0, 0, 0, 0 },
        .add => |a| .{ bits.f(a.scale), 0, 0, 0, 0, 0, 0, 0 },
        .embed => |e| .{ bits.f(e.scale), 0, 0, 0, 0, 0, 0, 0 },
        .mul => |m| .{ m.off, 0, 0, 0, 0, 0, 0, 0 },
        .scale => |s| .{ bits.f(s.s), 0, 0, 0, 0, 0, 0, 0 },
        .combine => |c| .{ bits.f(c.s), 0, 0, 0, 0, 0, 0, 0 },
    };
}

/// Map a recorded activation kind to the family's `AtomicId`.
fn activationAtomic(kind: u32) u16 {
    const ac = @import("../kernels/transformer/activation.zig");
    return switch (kind) {
        graph.Params.activation_gelu => @intFromEnum(ac.AtomicId.gated_gelu),
        else => @intFromEnum(ac.AtomicId.gated_silu),
    };
}

/// Map a recorded norm kind to the norm family's `AtomicId`.
fn normAtomic(kind: u32) u16 {
    const nm = @import("../kernels/transformer/norm.zig");
    return switch (kind) {
        graph.Params.norm_rms => @intFromEnum(nm.AtomicId.rms_norm),
        graph.Params.norm_rms_no_weight => @intFromEnum(nm.AtomicId.rms_norm_no_weight),
        graph.Params.norm_heads => @intFromEnum(nm.AtomicId.qk_norm_heads),
        graph.Params.norm_heads_no_weight => @intFromEnum(nm.AtomicId.rms_norm_no_weight_heads),
        else => @intFromEnum(nm.AtomicId.rms_norm),
    };
}

/// The fused `norm` + `rope` recipe's `AtomicId` (the Composer's `qk_norm_rope`
/// for a per-head norm).
fn normFusedAtomic() u16 {
    const nm = @import("../kernels/transformer/norm.zig");
    return @intFromEnum(nm.AtomicId.qk_norm_rope_heads);
}

fn resolveTensor(role: contract.TensorRole, layer: u32, opts: Options) u32 {
    const primary = lookupRole(role, layer, opts);
    if (primary != plan.NONE) return primary;
    const fallback = opts.arch.roleFallback orelse return plan.NONE;
    const alt = fallback(role, layer) orelse return plan.NONE;
    return lookupRole(alt, layer, opts);
}

fn lookupRole(role: contract.TensorRole, layer: u32, opts: Options) u32 {
    const ctx = opts.lookup_ctx orelse return plan.NONE;
    if (opts.arch.globalName(role)) |name| {
        return opts.lookup(ctx, name) orelse plan.NONE;
    }
    var buf: [512]u8 = undefined;
    const name = opts.arch.layeredName(&buf, role, layer) orelse return plan.NONE;
    return opts.lookup(ctx, name) orelse plan.NONE;
}

fn noLookup(_: *anyopaque, _: []const u8) ?u32 {
    return null;
}

/// Bootstrap atomic-id map: the family's first AtomicOp for each op, matching
/// the codecs in `kernels/`. The real Selector refines this with layout/params.
fn defaultAtomic(op: Op) u16 {
    const q = @import("../kernels/linear/quant/root.zig");
    const mv = @import("../kernels/linear/matvec/root.zig");
    const nm = @import("../kernels/transformer/norm.zig");
    const rp = @import("../kernels/transformer/rope.zig");
    const at = @import("../kernels/transformer/attention.zig");
    const ac = @import("../kernels/transformer/activation.zig");
    return switch (op) {
        .embd_lookup, .dequant => @intFromEnum(q.AtomicId.dequant),
        .matvec => @intFromEnum(mv.AtomicId.mat_vec),
        .gate_up => @intFromEnum(mv.AtomicId.gate_up_fused),
        .mlp => @intFromEnum(mv.AtomicId.mlp_fused),
        .lm_head => @intFromEnum(mv.AtomicId.lm_head),
        .norm => @intFromEnum(nm.AtomicId.rms_norm),
        .v_norm => @intFromEnum(nm.AtomicId.rms_norm_no_weight_heads),
        .rope => @intFromEnum(rp.AtomicId.rope),
        .write_kv => @intFromEnum(at.AtomicId.write_kv),
        .attention => @intFromEnum(at.AtomicId.fused_flash_head),
        .activation => @intFromEnum(ac.AtomicId.gated_silu),
        .add => @intFromEnum(@import("../kernels/transformer/elementwise.zig").AtomicId.add),
        .softcap => @intFromEnum(@import("../kernels/transformer/elementwise.zig").AtomicId.softcap),
        .gelu => @intFromEnum(ac.AtomicId.gelu),
        .mul => @intFromEnum(@import("../kernels/transformer/elementwise.zig").AtomicId.mul),
        .scale => @intFromEnum(@import("../kernels/transformer/elementwise.zig").AtomicId.scale),
        .combine => @intFromEnum(@import("../kernels/transformer/elementwise.zig").AtomicId.combine),
    };
}

/// Kernel op for a dispatch: the prefill phase lowers every plain projection
/// (`.matvec`) to the GEMM family; decode keeps the bootstrap mapping.
fn dispatchOp(op: Op, prefill: bool) @import("../kernels/registry.zig").Op {
    if (prefill and op == .matvec) return .matmul;
    return kernelOp(op);
}

/// Kernel op for a semantic op (bootstrap mapping).
pub fn kernelOp(op: Op) @import("../kernels/registry.zig").Op {
    return switch (op) {
        .embd_lookup, .dequant => .dequant,
        .matvec, .lm_head => .matvec,
        .gate_up => .matvec,
        .mlp => .matvec,
        .norm => .norm,
        .v_norm => .norm,
        .rope => .rope,
        .write_kv, .attention => .attention,
        .activation => .activation,
        .add => .elementwise,
        .softcap => .activation,
        .gelu => .activation,
        .mul, .scale, .combine => .elementwise,
    };
}

test "compileBlock: head + 2 layers + tail, ordered and bound" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const head = [_]Instr{
        .{ .op = .embd_lookup, .role = .token_embd, .out = .hidden },
    };
    const layer = [_]Instr{
        .{ .op = .norm, .role = .attn_norm, .in = .hidden, .out = .normed },
        .{ .op = .matvec, .role = .attn_q, .in = .normed, .out = .q },
    };
    const tail = [_]Instr{
        .{ .op = .lm_head, .role = .output, .in = .normed, .out = .logits },
    };
    const block = g.Block{ .slots = &.{ .hidden, .normed, .q, .logits }, .head = &head, .layer = &layer, .tail = &tail };

    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 2,
        .threads = 8,
        .slot_bytes = &.{ 16, 16, 16, 32 },
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    // 1 head + 2*2 layer + 1 tail = 6 dispatches, in forward order.
    try testing.expectEqual(@as(usize, 6), result.dispatch.len);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.dequant, result.dispatch[0].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.norm, result.dispatch[1].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[2].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.norm, result.dispatch[3].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[5].op);

    // in/out become buffer bindings (slot refs), tensor is NONE without lookup.
    try testing.expectEqual(plan.Binding.kindOf(.buffer), result.dispatch[1].in.kind);
    try testing.expectEqual(@intFromEnum(Slot.hidden), result.dispatch[1].in.ref);
    try testing.expectEqual(@intFromEnum(Slot.normed), result.dispatch[1].out.ref);
    try testing.expectEqual(plan.NONE, result.dispatch[0].tensor);

    // parts + buffers come from Options / slot_bytes.
    try testing.expectEqual(@as(u16, 1), result.dispatch[1].parts);
    try testing.expectEqual(@as(usize, 4), result.buffers.len);
    try testing.expectEqual(@as(u64, 32), result.buffers[3].bytes);
}

test "compileBlock: store_only instructions dropped for shared-KV layers" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .matvec, .role = .attn_q, .in = .normed, .out = .q },
        .{ .op = .write_kv, .in = .k, .in2 = .v, .store_only = true },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .normed, .q, .k, .v }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const stores = [_]bool{ true, false, true };
    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 3,
        .kv_stores = &stores,
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    // Layer 0 + 2 keep write_kv; layer 1 (shared) drops it.
    try testing.expectEqual(@as(usize, 5), result.dispatch.len);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[0].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.attention, result.dispatch[1].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[2].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[3].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.attention, result.dispatch[4].op);
}

fn fakeLayerDims(_: ?*anyopaque, layer: u32) contract.LayerDims {
    if (layer == 0) return .{ .hidden = 16, .q = 8, .kv = 4, .ffn = 32, .vocab = 64 };
    return .{ .hidden = 16, .q = 4, .kv = 2, .ffn = 64, .vocab = 64 };
}

test "compileBlock: per-layer dims drive in_len/out_len" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .matvec, .role = .attn_q, .in = .normed, .out = .q },
        .{ .op = .activation, .in = .ffn_a, .in2 = .ffn_b, .out = .ffn_mid },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .hidden, .normed, .q, .ffn_a, .ffn_b, .ffn_mid }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 2,
        .layer_dims = fakeLayerDims,
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 4), result.dispatch.len);
    // Layer 0: q = 8, ffn = 32.
    try testing.expectEqual(@as(u32, 16), result.dispatch[0].in_len);
    try testing.expectEqual(@as(u32, 8), result.dispatch[0].out_len);
    try testing.expectEqual(@as(u32, 32), result.dispatch[1].in_len);
    try testing.expectEqual(@as(u32, 32), result.dispatch[1].out_len);
    // Layer 1: q = 4, ffn = 64 (distinct from layer 0).
    try testing.expectEqual(@as(u32, 16), result.dispatch[2].in_len);
    try testing.expectEqual(@as(u32, 4), result.dispatch[2].out_len);
    try testing.expectEqual(@as(u32, 64), result.dispatch[3].in_len);
    try testing.expectEqual(@as(u32, 64), result.dispatch[3].out_len);
}

test "compileBlock: v_norm and norm_heads select head atomics" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;
    const nm = @import("../kernels/transformer/norm.zig");

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .v_norm, .in = .v, .out = .v, .store_only = true },
        .{ .op = .norm, .role = .attn_q_norm, .in = .q, .out = .q, .params = .{ .norm = .{ .eps = 1e-6, .head_dim = 4, .kind = g.Params.norm_heads } } },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .q, .v }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 1,
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 2), result.dispatch.len);
    try testing.expectEqual(@as(u16, @intFromEnum(nm.AtomicId.rms_norm_no_weight_heads)), result.dispatch[0].atomic_id);
    try testing.expectEqual(@as(u16, @intFromEnum(nm.AtomicId.qk_norm_heads)), result.dispatch[1].atomic_id);
}

test "compileBlock: fuse_rope norm emits qk_norm_rope_heads with the layer rope cfg" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;
    const nm = @import("../kernels/transformer/norm.zig");

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .norm, .role = .attn_q_norm, .in = .q, .out = .q, .fuse_rope = true, .params = .{ .norm = .{ .eps = 1e-6, .head_dim = 4, .kind = g.Params.norm_heads } } },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{.q}, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const rope_config = [_]u32{ 7, 9 };
    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 2,
        .rope_config = &rope_config,
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 2), result.dispatch.len);
    try testing.expectEqual(@as(u16, @intFromEnum(nm.AtomicId.qk_norm_rope_heads)), result.dispatch[0].atomic_id);
    try testing.expectEqual(@as(u32, 7), result.dispatch[0].rope_cfg);
    try testing.expectEqual(@as(u16, @intFromEnum(nm.AtomicId.qk_norm_rope_heads)), result.dispatch[1].atomic_id);
    try testing.expectEqual(@as(u32, 9), result.dispatch[1].rope_cfg);
}

const GateUpLut = struct {
    fn lookup(_: *anyopaque, name: []const u8) ?u32 {
        if (std.mem.endsWith(u8, name, ".ffn_gate.weight")) return 0;
        if (std.mem.endsWith(u8, name, ".ffn_up.weight")) return 1;
        return null;
    }
};

test "compileBlock: gate_up emits gate_up_fused with both tensors bound" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;
    const mv = @import("../kernels/linear/matvec/root.zig");

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .gate_up, .role = .ffn_gate, .role2 = .ffn_up, .in = .normed, .out = .ffn_mid, .params = .{ .activation = .{ .kind = g.Params.activation_gelu } } },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .hidden, .normed, .ffn_mid }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    var dummy: u8 = 0;
    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 1,
        .lookup = GateUpLut.lookup,
        .lookup_ctx = @ptrCast(&dummy),
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 1), result.dispatch.len);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, result.dispatch[0].op);
    try testing.expectEqual(@as(u16, @intFromEnum(mv.AtomicId.gate_up_fused)), result.dispatch[0].atomic_id);
    try testing.expectEqual(@as(u32, 0), result.dispatch[0].tensor);
    try testing.expectEqual(@as(u32, 1), result.dispatch[0].tensor2);
    try testing.expectEqual(g.Params.activation_gelu, result.dispatch[0].params[0]);
}

fn noPleDims(_: ?*anyopaque, _: u32) contract.LayerDims {
    return .{ .hidden = 16, .ffn = 32, .ple = 0, .ple_all = 0 };
}

fn withPleDims(_: ?*anyopaque, _: u32) contract.LayerDims {
    return .{ .hidden = 16, .ffn = 32, .ple = 16, .ple_all = 16 };
}

test "compileBlock: PLE instructions gated on dims.ple" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .matvec, .in = .hidden, .out = .ffn_a },
        .{ .op = .add, .in = .hidden, .in2 = .ffn_a, .out = .hidden },
        .{ .op = .matvec, .role = .inp_gate, .in = .hidden, .out = .ple_a, .gate = .ple },
        .{ .op = .gelu, .in = .ple_a, .out = .ple_a, .gate = .ple },
        .{ .op = .mul, .in = .ple_a, .in2 = .ple_combined, .out = .ple_a, .gate = .ple },
        .{ .op = .matvec, .role = .proj, .in = .ple_a, .out = .normed, .gate = .ple },
        .{ .op = .norm, .role = .post_norm, .in = .normed, .out = .normed, .gate = .ple },
        .{ .op = .add, .in = .hidden, .in2 = .normed, .out = .hidden, .gate = .ple },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .hidden, .normed, .ffn_a, .ple_a, .ple_combined }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const arch = @import("../architectures/impl/gemma4/root.zig").arch;

    const off = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 1,
        .layer_dims = noPleDims,
    });
    defer testing.allocator.free(off.dispatch);
    defer testing.allocator.free(off.buffers);
    try testing.expectEqual(@as(usize, 2), off.dispatch.len);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.matvec, off.dispatch[0].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.elementwise, off.dispatch[1].op);

    const on = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 1,
        .layer_dims = withPleDims,
    });
    defer testing.allocator.free(on.dispatch);
    defer testing.allocator.free(on.buffers);
    try testing.expectEqual(@as(usize, 8), on.dispatch.len);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.activation, on.dispatch[3].op);
    try testing.expectEqual(@import("../kernels/registry.zig").Op.elementwise, on.dispatch[4].op);
    try testing.expectEqual(@as(u32, 16), on.dispatch[3].in_len);
}

fn parallelLayerDims(_: ?*anyopaque, layer: u32) contract.LayerDims {
    if (layer == 0) return .{ .hidden = 1024, .ffn = 2048 };
    return .{ .hidden = 16, .ffn = 32 };
}

const KvFallbackLut = struct {
    fn lookup(_: *anyopaque, name: []const u8) ?u32 {
        if (std.mem.eql(u8, name, "blk.0.attn_v.weight")) return 9;
        if (std.mem.endsWith(u8, name, ".attn_k.weight")) return 5;
        return null;
    }
};

test "compileBlock: absent attn_v resolves to attn_k (k_eq_v)" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .matvec, .role = .attn_k, .in = .normed, .out = .k, .store_only = true },
        .{ .op = .matvec, .role = .attn_v, .in = .normed, .out = .v, .store_only = true },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .normed, .k, .v }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    var dummy: u8 = 0;
    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 2,
        .lookup = KvFallbackLut.lookup,
        .lookup_ctx = @ptrCast(&dummy),
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 4), result.dispatch.len);
    // Layer 0 has attn_v: k -> k tensor, v -> its own tensor.
    try testing.expectEqual(@as(u32, 5), result.dispatch[0].tensor);
    try testing.expectEqual(@as(u32, 9), result.dispatch[1].tensor);
    // Layer 1 has no attn_v: the v projection reuses the attn_k tensor.
    try testing.expectEqual(@as(u32, 5), result.dispatch[2].tensor);
    try testing.expectEqual(@as(u32, 5), result.dispatch[3].tensor);
}

test "compileBlock: scheduler gives wide row-parallel matvec >1 parts" {
    const testing = std.testing;
    const g = graph;
    const Instr = g.Instr;

    const no_head = [_]Instr{};
    const layer = [_]Instr{
        .{ .op = .matvec, .in = .normed, .out = .ffn_mid },
        .{ .op = .norm, .in = .ffn_mid, .out = .ffn_mid },
    };
    const no_tail = [_]Instr{};
    const block = g.Block{ .slots = &.{ .hidden, .normed, .ffn_mid }, .head = &no_head, .layer = &layer, .tail = &no_tail };

    const arch = @import("../architectures/impl/gemma4/root.zig").arch;
    const result = try compileBlock(testing.allocator, block, .{
        .arch = arch,
        .layer_count = 2,
        .threads = 4,
        .layer_dims = parallelLayerDims,
    });
    defer testing.allocator.free(result.dispatch);
    defer testing.allocator.free(result.buffers);

    try testing.expectEqual(@as(usize, 4), result.dispatch.len);
    try testing.expect(result.dispatch[0].parts > 1);
    try testing.expect(result.dispatch[0].parts <= 4);
    try testing.expectEqual(@as(u16, 1), result.dispatch[1].parts);
    try testing.expectEqual(@as(u16, 1), result.dispatch[2].parts);
    try testing.expectEqual(@as(u16, 1), result.dispatch[3].parts);
}

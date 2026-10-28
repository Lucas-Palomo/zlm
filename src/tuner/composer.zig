const std = @import("std");
const graph = @import("../architectures/core/graph.zig");

/// Result of a Composer pass: the transformed `Block` plus the fusion tally
/// the `tune` summary reports. `dropped` counts the instructions removed by
/// fusion (one `rope` per fused pair).
pub const Result = struct {
    block: graph.Block,
    fused: u32 = 0,
    dropped: u32 = 0,
};

/// Compose the declared forward into a fused form, deterministically.
///
/// First recipe: fuse a `norm` with a later `rope` on the same activation slot
/// (q or k) into one instruction carrying `fuse_rope = true`; the `rope` is
/// dropped. The pair is fused when the next access to that slot is the rope and
/// nothing else touches the slot in between, so the two passes become one in
/// place. The cost model scores the fused form strictly lower (one dispatch
/// latency term and one activation pass removed), so the decision is a
/// deterministic "always fuse on match" with no timing input.
///
/// Second recipe: the FFN triple `ffn_gate` matvec -> `ffn_up` matvec ->
/// activation collapses into one `.gate_up` instruction (gate weight in
/// `role`, up weight in `role2`, the shared activation in `in`, the gated
/// result in `out`); the `up` matvec and the activation are dropped. The
/// activation kind is resolved by the architecture at compile time.
///
/// The Composer does **not** fuse the `ffn_down` into a `mlp`: whether the
/// whole-MLP fusion pays depends on the device (a discrete GPU's single-
/// workgroup fused shader loses to the multi-workgroup pair), so the tuner
/// makes that choice per instruction (see `tune.zig`'s fusion pass). Keeping it
/// here would force the fused form on every device.
///
/// The returned slices are allocated from `allocator`; the caller owns them.
pub fn compose(allocator: std.mem.Allocator, block: graph.Block) !Result {
    var result = Result{ .block = block };
    result.block.head = try composeSection(allocator, block.head, &result);
    result.block.layer = try composeSection(allocator, block.layer, &result);
    result.block.tail = try composeSection(allocator, block.tail, &result);
    return result;
}

fn composeSection(
    allocator: std.mem.Allocator,
    instrs: []const graph.Instr,
    result: *Result,
) ![]graph.Instr {
    const fuse = try allocator.alloc(bool, instrs.len);
    defer allocator.free(fuse);
    const drop = try allocator.alloc(bool, instrs.len);
    defer allocator.free(drop);
    const gate = try allocator.alloc(bool, instrs.len);
    defer allocator.free(gate);
    @memset(fuse, false);
    @memset(drop, false);
    @memset(gate, false);

    for (instrs, 0..) |instr, i| {
        if (instr.op != .norm or instr.fuse_rope) continue;
        const slot = instr.in orelse continue;
        var j = i + 1;
        while (j < instrs.len) : (j += 1) {
            const cand = instrs[j];
            if (!drop[j] and cand.op == .rope and cand.in == slot and cand.out == slot) {
                fuse[i] = true;
                drop[j] = true;
                result.fused += 1;
                result.dropped += 1;
                break;
            }
            if (touches(cand, slot)) break;
        }
    }

    for (instrs, 0..) |instr, i| {
        if (instr.op != .matvec or drop[i]) continue;
        if ((instr.role orelse continue) != .ffn_gate) continue;
        const in_slot = instr.in orelse continue;
        const a = instr.out orelse continue;
        const j = i + 1;
        if (j >= instrs.len or drop[j]) continue;
        const up = instrs[j];
        if (up.op != .matvec or up.in != in_slot or up.role != .ffn_up) continue;
        const b = up.out orelse continue;
        const k = j + 1;
        if (k >= instrs.len or drop[k]) continue;
        const act = instrs[k];
        if (act.op != .activation or act.in != a or act.in2 != b) continue;
        if (act.out == null) continue;
        gate[i] = true;
        drop[j] = true;
        drop[k] = true;
        result.fused += 1;
        result.dropped += 2;
    }

    var out: std.ArrayList(graph.Instr) = .empty;
    errdefer out.deinit(allocator);
    for (instrs, 0..) |instr, i| {
        if (drop[i]) continue;
        var it = instr;
        if (fuse[i]) it.fuse_rope = true;
        if (gate[i]) {
            it.op = .gate_up;
            it.role2 = .ffn_up;
            it.out = instrs[i + 2].out;
            it.in2 = null;
        }
        try out.append(allocator, it);
    }
    return out.toOwnedSlice(allocator);
}

fn touches(instr: graph.Instr, slot: graph.Slot) bool {
    return instr.in == slot or instr.in2 == slot or instr.out == slot;
}

test "compose fuses the q and k norm+rope pairs in the gemma block" {
    const testing = std.testing;
    const gemma = @import("../architectures/impl/gemma4/block.zig");
    const contract = @import("../architectures/core/contract.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const result = try compose(arena.allocator(), gemma.block);
    try testing.expectEqual(@as(u32, 3), result.fused);
    try testing.expectEqual(@as(u32, 4), result.dropped);
    try testing.expectEqual(gemma.block.layer.len - 4, result.block.layer.len);

    var fused_norms: usize = 0;
    var gate_ups: usize = 0;
    for (result.block.layer) |instr| {
        try testing.expect(instr.op != .rope);
        if (instr.fuse_rope) fused_norms += 1;
        if (instr.op == .gate_up) {
            gate_ups += 1;
            try testing.expectEqual(contract.TensorRole.ffn_gate, instr.role.?);
            try testing.expectEqual(contract.TensorRole.ffn_up, instr.role2.?);
            try testing.expectEqual(graph.Slot.normed, instr.in.?);
            try testing.expectEqual(graph.Slot.ffn_mid, instr.out.?);
        }
    }
    try testing.expectEqual(@as(usize, 2), fused_norms);
    try testing.expectEqual(@as(usize, 1), gate_ups);
    try testing.expectEqual(gemma.block.head.len, result.block.head.len);
    try testing.expectEqual(gemma.block.tail.len, result.block.tail.len);
}

test "compose leaves a lone norm and a later unrelated rope untouched" {
    const testing = std.testing;
    const Instr = graph.Instr;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const layer = [_]Instr{
        .{ .op = .norm, .in = .q, .out = .q },
        .{ .op = .attention, .in = .q, .out = .attn_out },
        .{ .op = .rope, .in = .q, .out = .q },
    };
    const block = graph.Block{
        .slots = &.{ .q, .attn_out },
        .head = &.{},
        .layer = &layer,
        .tail = &.{},
    };
    const result = try compose(arena.allocator(), block);
    try testing.expectEqual(@as(u32, 0), result.fused);
    try testing.expectEqual(layer.len, result.block.layer.len);
    try testing.expect(!result.block.layer[0].fuse_rope);
}

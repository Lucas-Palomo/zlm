const std = @import("std");
const registry = @import("registry.zig");
const quant = @import("linear/quant/root.zig");
const activation = @import("transformer/activation.zig");
const norm = @import("transformer/norm.zig");
const rope = @import("transformer/rope.zig");
const attention = @import("transformer/attention.zig");
const elementwise = @import("transformer/elementwise.zig");
const matvec_mod = @import("linear/matvec/root.zig");
const matmul_mod = @import("linear/matmul/root.zig");

const sets = registry.sets;
const find = registry.find;
const run = registry.run;
const kernels = registry.kernels;
const kernelsFor = registry.kernelsFor;
const recipes = registry.recipes;
const Kernel = registry.Kernel;
const Op = registry.Op;

test "registry: every op has a family" {
    const testing = std.testing;
    for (sets) |s| {
        try testing.expect(find(s.op) != null);
    }
    try testing.expect(find(.dequant) != null);
    try testing.expect(find(.matvec) != null);
    try testing.expect(find(.matmul) != null);
    try testing.expect(find(.norm) != null);
    try testing.expect(find(.rope) != null);
    try testing.expect(find(.attention) != null);
    try testing.expect(find(.activation) != null);
}

test "registry: run dispatches a dequant" {
    const ggml = @import("../gguf/format.zig");
    var qs: [32]i8 = undefined;
    for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i)) - 16);
    const block = quant.encodeQ8_0Block(0.5, &qs);
    var out: [32]f32 = undefined;
    var args = quant.Args{
        .ggml_type = @intFromEnum(ggml.GGMLType.q8_0),
        .data = &block,
        .out = &out,
    };
    try run(.dequant, @intFromEnum(quant.AtomicId.dequant), &args);
    for (&qs, 0..) |q, i| {
        try std.testing.expectApproxEqAbs(0.5 * @as(f32, @floatFromInt(q)), out[i], 1e-6);
    }
}

fn expectValidKernels(comptime Id: type, ks: []const Kernel) !void {
    try std.testing.expect(ks.len > 0);
    for (ks) |k| {
        try std.testing.expect(std.enums.fromInt(Id, k.atomic_id) != null);
    }
}

test "registry: every family exposes valid candidate kernels" {
    try expectValidKernels(quant.AtomicId, quant.set.kernels);
    try expectValidKernels(activation.AtomicId, activation.set.kernels);
    try expectValidKernels(norm.AtomicId, norm.set.kernels);
    try expectValidKernels(rope.AtomicId, rope.set.kernels);
    try expectValidKernels(attention.AtomicId, attention.set.kernels);
    try expectValidKernels(elementwise.AtomicId, elementwise.set.kernels);
    try expectValidKernels(matvec_mod.AtomicId, matvec_mod.set.kernels);
    try expectValidKernels(matmul_mod.AtomicId, matmul_mod.set.kernels);
}

test "registry: matmul exposes every MatMulPlan variant" {
    const testing = std.testing;
    for (std.enums.values(matmul_mod.MatMulPlan)) |plan| {
        var found = false;
        for (matmul_mod.set.kernels) |k| {
            if (k.op == .matmul and k.variant == @intFromEnum(plan)) found = true;
        }
        try testing.expect(found);
    }
}

test "registry: matvec exposes every MatVecPlan variant" {
    const testing = std.testing;
    for (std.enums.values(matvec_mod.MatVecPlan)) |plan| {
        var found = false;
        for (matvec_mod.set.kernels) |k| {
            if (k.op == .matvec and
                k.atomic_id == @intFromEnum(matvec_mod.AtomicId.mat_vec) and
                k.variant == @intFromEnum(plan))
            {
                found = true;
            }
        }
        try testing.expect(found);
    }
}

test "registry: kernels() aggregates every family's catalog" {
    const testing = std.testing;
    var total: usize = 0;
    for (sets) |s| {
        total += s.kernels.len;
        try testing.expect(s.kernels.len > 0);
    }
    try testing.expectEqual(total, kernels().len);
    for (sets) |s| {
        try testing.expectEqualSlices(Kernel, s.kernels, kernelsFor(s.op));
    }
}

fn recipeHasKernel(op: Op, atomic_id: u16) bool {
    for (kernelsFor(op)) |k| {
        if (k.atomic_id == atomic_id) return true;
    }
    return false;
}

test "registry: recipes() enumerates valid recipes" {
    const testing = std.testing;
    const rs = recipes();
    try testing.expect(rs.len > 0);
    for (rs) |r| {
        try testing.expect(recipeHasKernel(r.op, r.atomic_id));
    }
    const expected = [_][]const u8{
        "gate_up_fused",
        "qk_norm_rope",
        "norm_proj",
        "rope_write_kv",
        "add_norm",
        "dequant_scale",
        "matvec_bias",
    };
    for (expected) |name| {
        var found = false;
        for (rs) |r| {
            if (std.mem.eql(u8, r.name, name)) found = true;
        }
        try testing.expect(found);
    }
}

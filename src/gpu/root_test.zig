const std = @import("std");
const device = @import("root.zig");
const contract = @import("backend/root.zig");
const ggml = @import("../gguf/format.zig");
const TensorView = @import("../data/tensor_view.zig").TensorView;
const matvec = @import("../kernels/linear/matvec/root.zig");

const Device = device.Device;
const Quant = device.Quant;
const GpuVariant = device.GpuVariant;
const GateUpAct = device.GateUpAct;
const Resident = device.Resident;
const ResidentOp = device.ResidentOp;

// ---- parity oracle: GPU vs the v2 CPU matvec kernel ----

fn ggmlType(quant: Quant) ggml.GGMLType {
    return switch (quant.base()) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q4_k => .q4_k,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => unreachable,
    };
}

/// Relative-tolerant comparison: the GPU sums each row sequentially while the
/// CPU kernel uses SIMD lanes, so the results differ only by float ordering.
fn expectClose(cpu: []const f32, gpu: []const f32) !void {
    try std.testing.expectEqual(cpu.len, gpu.len);
    for (cpu, gpu) |c, g| {
        const tol = 1e-4 + 1e-4 * @abs(c);
        try std.testing.expect(@abs(c - g) <= tol);
    }
}

/// Like `expectClose` but with an absolute floor large enough for the long
/// `k = 8192` axis sums: the true value can cancel near zero while the
/// reordering error stays ~1e-3, so the tight relative term alone is too strict.
fn expectCloseAxis(cpu: []const f32, gpu: []const f32) !void {
    try std.testing.expectEqual(cpu.len, gpu.len);
    for (cpu, gpu) |c, g| {
        const tol = 5e-3 + 1e-4 * @abs(c);
        try std.testing.expect(@abs(c - g) <= tol);
    }
}

fn expectMatVecParity(dev: *Device, quant: Quant) !void {
    const rows: usize = 64;
    const k: usize = 256;

    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();
    var wbuf: [rows * k * 4]u8 = undefined;
    const w = wbuf[0 .. rows * row_bytes];
    contract.fillWeights(quant, w, k, rows);

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;
    }

    var gpu_out: [rows]f32 = undefined;
    const ran = dev.matVec(quant, w, rows, k, &x, &gpu_out);
    try std.testing.expect(ran);

    const dims = [_]u64{ k, rows };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = ggmlType(quant),
        .offset = 0,
        .byte_length = w.len,
        .mapped_slice = w,
    };
    var cpu_out: [rows]f32 = undefined;
    try matvec.matVecRows(&view, &x, &cpu_out, 0, rows);

    try expectClose(&cpu_out, &gpu_out);

    // The resident path uploads the weight once and reads the same buffer.
    var res_out: [rows]f32 = undefined;
    const res = dev.uploadResident(w) orelse return error.SkipZigTest;
    defer dev.freeResident(res);
    try std.testing.expect(dev.matVecResident(res, quant, rows, k, &x, &res_out));
    try expectClose(&cpu_out, &res_out);

    // The device-owned activation path (shared burst `x`, async handle) drains
    // the same value.
    var burst_out: [rows]f32 = undefined;
    const hb = dev.matVecResidentBurstAsync(res, quant, rows, k, &x, true) orelse return error.SkipZigTest;
    try std.testing.expect(dev.waitResident(hb, &burst_out, rows));
    try expectClose(&cpu_out, &burst_out);
}

/// Parity for the UNROLL / WORKGROUP_SIZE axes at a larger `k`, so the unroll
/// lanes (k-blocks spaced by `local_size_x`) and the 32/128-lane reductions are
/// actually exercised against the v2 CPU kernel. Returns false when the device
/// pruned the variant's pipeline (a workgroup size it rejected).
fn expectAxisParity(dev: *Device, quant: Quant) !bool {
    const rows: usize = 64;
    const k: usize = 8192; // nb = 256 blocks: exercises unroll lanes 0..3
    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();

    const w = try std.testing.allocator.alloc(u8, rows * row_bytes + 64);
    defer std.testing.allocator.free(w);
    contract.fillWeights(quant, w[0 .. rows * row_bytes], k, rows);

    const x = try std.testing.allocator.alloc(f32, k);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    const gpu_out = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(gpu_out);
    if (!dev.matVec(quant, w, rows, k, x, gpu_out)) return false;

    const dims = [_]u64{ k, rows };
    const view = TensorView{ .name = "w", .dims = &dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = w.len, .mapped_slice = w };
    const cpu_out = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(cpu_out);
    try matvec.matVecRows(&view, x, cpu_out, 0, rows);

    try expectCloseAxis(cpu_out, gpu_out);
    return true;
}

/// Batched-GEMM parity: the `m`-row prompt block computes the same values as
/// the v2 CPU GEMM (`matvec.matMulRows`, the fallback `matVecBatch` uses). Only
/// the quants with a batched GEMM shader are exercised (no `q6_k`).
fn expectMatMulParity(dev: *Device, quant: Quant) !void {
    if (quant.base() == .q6_k) return;
    const rows: usize = 64;
    const k: usize = 256;
    const m: usize = 4;

    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();
    var wbuf: [rows * k * 4]u8 = undefined;
    const w = wbuf[0 .. rows * row_bytes];
    contract.fillWeights(quant, w, k, rows);

    var x: [m * k]f32 = undefined;
    for (&x, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;
    }

    var gpu_out: [m * rows]f32 = undefined;
    const ran = dev.matMulBatch(quant, w, rows, k, m, &x, &gpu_out);
    try std.testing.expect(ran);

    const dims = [_]u64{ k, rows };
    const view = TensorView{
        .name = "w",
        .dims = &dims,
        .ggml_type = ggmlType(quant),
        .offset = 0,
        .byte_length = w.len,
        .mapped_slice = w,
    };
    var cpu_out: [m * rows]f32 = undefined;
    try matvec.matMulRows(&view, &x, &cpu_out, 0, rows);

    try expectClose(&cpu_out, &gpu_out);

    // The resident path uploads the weight once and reads the same buffer.
    var res_out: [m * rows]f32 = undefined;
    const res = dev.uploadResident(w) orelse return error.SkipZigTest;
    defer dev.freeResident(res);
    try std.testing.expect(dev.matMulResident(res, quant, rows, k, m, &x, &res_out));
    try expectClose(&cpu_out, &res_out);
}

/// Fused `gate_up` parity: the GPU computes
/// `act(dot(gate_i, x), dot(up_i, x))` for a two-weight GEMV (q4_0, q4_1 or
/// q8_0); compare against the v2 CPU `fusedGateUpRows` (the `gate_up_fused`
/// atomic) expressed as two matvecs plus the gated activation. Returns false
/// when the device pruned the variant's pipeline (or has no fused shader).
fn expectGateUpParity(dev: *Device, quant: Quant, act: GateUpAct, act_kind: matvec.ActivationKind, variant: GpuVariant) !bool {
    const rows: usize = 64;
    const k: usize = 256;
    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();

    var gbuf: [rows * k * 4]u8 = undefined;
    var ubuf: [rows * k * 4]u8 = undefined;
    const gw = gbuf[0 .. rows * row_bytes];
    const uw = ubuf[0 .. rows * row_bytes];
    contract.fillWeights(quant, gw, k, rows);
    contract.fillWeights(quant, uw, k, rows);

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    const resg = dev.uploadResident(gw) orelse return error.SkipZigTest;
    defer dev.freeResident(resg);
    const resu = dev.uploadResident(uw) orelse return error.SkipZigTest;
    defer dev.freeResident(resu);

    var gpu_out: [rows]f32 = undefined;
    const h = dev.matVecGateUpResidentBurstAsync(resg, resu, quant, rows, k, &x, act, variant, true) orelse return false;
    try std.testing.expect(dev.waitResident(h, &gpu_out, rows));

    const dims = [_]u64{ k, rows };
    const vg = TensorView{ .name = "gate", .dims = &dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = gw.len, .mapped_slice = gw };
    const vu = TensorView{ .name = "up", .dims = &dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = uw.len, .mapped_slice = uw };
    var g_cpu: [rows]f32 = undefined;
    var u_cpu: [rows]f32 = undefined;
    try matvec.matVecRows(&vg, &x, &g_cpu, 0, rows);
    try matvec.matVecRows(&vu, &x, &u_cpu, 0, rows);
    var cpu_out: [rows]f32 = undefined;
    for (0..rows) |i| cpu_out[i] = act_kind.apply(g_cpu[i], u_cpu[i]);

    try expectClose(&cpu_out, &gpu_out);
    return true;
}

/// Fused MLP parity: the GPU computes the whole SwiGLU MLP (`act(gate·x, up·x) ·
/// down`) over three resident weights; compare against the v2 CPU oracle
/// (`gate_up` expressed as two matvecs plus the gated activation, then the
/// `down` matvec). `pair` selects the discrete two-kernel form (the fused
/// `gate_up` then the `down`, `ffn_mid` in VRAM) over the single-workgroup fused
/// shader (the `ffn_mid` in shared memory). Returns false when the device pruned
/// the variant's pipeline.
fn expectMlpParity(dev: *Device, quant: Quant, act: GateUpAct, act_kind: matvec.ActivationKind, variant: GpuVariant, pair: bool) !bool {
    const rows: usize = 64; // output width (the `down` rows)
    const ffn: usize = 64; // gate/up rows (the `down` input width)
    const k: usize = 256;
    const gate_row = (k / quant.blockElems()) * quant.blockBytes();
    const down_row = (ffn / quant.blockElems()) * quant.blockBytes();

    var gbuf: [ffn * k * 4]u8 = undefined;
    var ubuf: [ffn * k * 4]u8 = undefined;
    var dbuf: [rows * ffn * 4]u8 = undefined;
    const gw = gbuf[0 .. ffn * gate_row];
    const uw = ubuf[0 .. ffn * gate_row];
    const dw = dbuf[0 .. rows * down_row];
    contract.fillWeights(quant, gw, k, ffn);
    contract.fillWeights(quant, uw, k, ffn);
    contract.fillWeights(quant, dw, ffn, rows);

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    const resg = dev.uploadResident(gw) orelse return error.SkipZigTest;
    defer dev.freeResident(resg);
    const resu = dev.uploadResident(uw) orelse return error.SkipZigTest;
    defer dev.freeResident(resu);
    const resd = dev.uploadResident(dw) orelse return error.SkipZigTest;
    defer dev.freeResident(resd);

    var gpu_out: [rows]f32 = undefined;
    const h = if (pair)
        dev.matVecMlpPairResidentBurstAsync(resg, resu, resd, quant, quant, rows, k, ffn, &x, act, variant, true) orelse return false
    else
        dev.matVecMlpResidentBurstAsync(resg, resu, resd, quant, rows, k, ffn, &x, act, variant, true) orelse return false;
    try std.testing.expect(dev.waitResident(h, &gpu_out, rows));

    const g_dims = [_]u64{ k, ffn };
    const d_dims = [_]u64{ ffn, rows };
    const vg = TensorView{ .name = "gate", .dims = &g_dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = gw.len, .mapped_slice = gw };
    const vu = TensorView{ .name = "up", .dims = &g_dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = uw.len, .mapped_slice = uw };
    const vd = TensorView{ .name = "down", .dims = &d_dims, .ggml_type = ggmlType(quant), .offset = 0, .byte_length = dw.len, .mapped_slice = dw };
    var g_cpu: [ffn]f32 = undefined;
    var u_cpu: [ffn]f32 = undefined;
    var m_cpu: [ffn]f32 = undefined;
    try matvec.matVecRows(&vg, &x, &g_cpu, 0, ffn);
    try matvec.matVecRows(&vu, &x, &u_cpu, 0, ffn);
    for (0..ffn) |i| m_cpu[i] = act_kind.apply(g_cpu[i], u_cpu[i]);
    var cpu_out: [rows]f32 = undefined;
    try matvec.matVecRows(&vd, &m_cpu, &cpu_out, 0, rows);

    try expectClose(&cpu_out, &gpu_out);
    return true;
}

/// Mixed-quant two-kernel MLP parity: the `gate`/`up` carry `gu` and the `down`
/// `dn` (the two kernels are independent), compared against the v2 CPU oracle.
fn expectMlpPairMixedParity(dev: *Device, gu: Quant, dn: Quant, act: GateUpAct, act_kind: matvec.ActivationKind, variant: GpuVariant) !bool {
    const rows: usize = 64;
    const ffn: usize = 64;
    const k: usize = 256;
    const gate_row = (k / gu.blockElems()) * gu.blockBytes();
    const down_row = (ffn / dn.blockElems()) * dn.blockBytes();

    var gbuf: [ffn * k * 4]u8 = undefined;
    var ubuf: [ffn * k * 4]u8 = undefined;
    var dbuf: [rows * ffn * 4]u8 = undefined;
    const gw = gbuf[0 .. ffn * gate_row];
    const uw = ubuf[0 .. ffn * gate_row];
    const dw = dbuf[0 .. rows * down_row];
    contract.fillWeights(gu, gw, k, ffn);
    contract.fillWeights(gu, uw, k, ffn);
    contract.fillWeights(dn, dw, ffn, rows);

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    const resg = dev.uploadResident(gw) orelse return error.SkipZigTest;
    defer dev.freeResident(resg);
    const resu = dev.uploadResident(uw) orelse return error.SkipZigTest;
    defer dev.freeResident(resu);
    const resd = dev.uploadResident(dw) orelse return error.SkipZigTest;
    defer dev.freeResident(resd);

    var gpu_out: [rows]f32 = undefined;
    const h = dev.matVecMlpPairResidentBurstAsync(resg, resu, resd, gu, dn, rows, k, ffn, &x, act, variant, true) orelse return false;
    try std.testing.expect(dev.waitResident(h, &gpu_out, rows));

    const g_dims = [_]u64{ k, ffn };
    const d_dims = [_]u64{ ffn, rows };
    const vg = TensorView{ .name = "gate", .dims = &g_dims, .ggml_type = ggmlType(gu), .offset = 0, .byte_length = gw.len, .mapped_slice = gw };
    const vu = TensorView{ .name = "up", .dims = &g_dims, .ggml_type = ggmlType(gu), .offset = 0, .byte_length = uw.len, .mapped_slice = uw };
    const vd = TensorView{ .name = "down", .dims = &d_dims, .ggml_type = ggmlType(dn), .offset = 0, .byte_length = dw.len, .mapped_slice = dw };
    var g_cpu: [ffn]f32 = undefined;
    var u_cpu: [ffn]f32 = undefined;
    var m_cpu: [ffn]f32 = undefined;
    try matvec.matVecRows(&vg, &x, &g_cpu, 0, ffn);
    try matvec.matVecRows(&vu, &x, &u_cpu, 0, ffn);
    for (0..ffn) |i| m_cpu[i] = act_kind.apply(g_cpu[i], u_cpu[i]);
    var cpu_out: [rows]f32 = undefined;
    try matvec.matVecRows(&vd, &m_cpu, &cpu_out, 0, rows);

    try expectClose(&cpu_out, &gpu_out);
    return true;
}

test "device without a GPU is inert and never panics" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    try std.testing.expect(dev.available() == (dev.backend == .vulkan));
    if (!dev.available()) {
        const ran = dev.matVec(.q4_0, &.{}, 0, 0, &.{}, &.{});
        try std.testing.expect(!ran);
    }
}

test "gpu matvec parity with the v2 cpu kernel (q4_0, q4_1, q8_0, q6_k, q4_k, f32, f16, bf16)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    try expectMatVecParity(&dev, .q4_0);
    try expectMatVecParity(&dev, .q4_1);
    try expectMatVecParity(&dev, .q8_0);
    try expectMatVecParity(&dev, .q6_k);
    try expectMatVecParity(&dev, .q4_k);
    try expectMatVecParity(&dev, .f32);
    try expectMatVecParity(&dev, .f16);
    try expectMatVecParity(&dev, .bf16);

    dev.sync();
}

// The spec-constant variant grid, uniform across quants: every `(quant,
// variant)` is the same scaffold specialized on the axes (rows/unroll/
// reduction/workgroup) and must match the v2 CPU kernel. A variant the device
// pruned (a rejected workgroup size, the subgroup path on a 1.0 device) is
// skipped. `k = 8192` exercises the unroll lanes and the 32/128-lane reductions.
test "gpu matvec variant grid parity for every (quant, variant)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const bases = [_]Quant{ .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 };
    const variants = [_]GpuVariant{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 };
    for (bases) |b| {
        for (variants) |v| {
            if (!try expectAxisParity(&dev, b.variant(v))) continue;
        }
    }

    dev.sync();
}

// The subgroup/wave reduction variant matches the shared-memory `plain` variant
// and the CPU kernel, on the q4_0 base (the original `gpu/09` oracle).
// Skipped when the device lacks the subgroup pipeline (pruned at init).
test "gpu subgroup reduction variant is parity-correct (q4_0_sub)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const rows: usize = 64;
    const k: usize = 256;
    const row_bytes = (k / 32) * 18;
    var wbuf: [rows * row_bytes]u8 = undefined;
    const w = wbuf[0 .. rows * row_bytes];
    contract.fillWeights(.q4_0, w, k, rows);

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    var out: [rows]f32 = undefined;
    if (!dev.matVec(.q4_0_sub, w, rows, k, &x, &out)) return error.SkipZigTest;

    try expectMatVecParity(&dev, .q4_0_sub);

    var plain_out: [rows]f32 = undefined;
    try std.testing.expect(dev.matVec(.q4_0, w, rows, k, &x, &plain_out));
    try expectClose(&plain_out, &out);

    dev.sync();
}

test "gpu batched-matmul parity with the v2 cpu kernel (q4_0, q4_1, q8_0, q4_k)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    try expectMatMulParity(&dev, .q4_0);
    try expectMatMulParity(&dev, .q4_1);
    try expectMatMulParity(&dev, .q8_0);
    try expectMatMulParity(&dev, .q4_k);

    dev.sync();
}

test "gpu fused gate_up parity with the v2 cpu kernel (q4_0, q4_1, q8_0, gelu/silu)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const variants = [_]GpuVariant{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 };
    for ([_]Quant{ .q4_0, .q4_1, .q8_0 }) |q| {
        for (variants) |v| {
            if (!try expectGateUpParity(&dev, q, .gelu, .gelu_tanh, v)) continue;
        }
    }
    // The silu activation on the base variant of each quant.
    try std.testing.expect(try expectGateUpParity(&dev, .q4_0, .silu, .silu, .plain));
    try std.testing.expect(try expectGateUpParity(&dev, .q4_1, .silu, .silu, .plain));
    try std.testing.expect(try expectGateUpParity(&dev, .q8_0, .silu, .silu, .plain));

    dev.sync();
}

test "gpu fused mlp parity with the v2 cpu kernel (q4_0, q4_1, q8_0, gelu/silu)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const variants = [_]GpuVariant{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 };
    for ([_]Quant{ .q4_0, .q4_1, .q8_0 }) |q| {
        for (variants) |v| {
            if (!try expectMlpParity(&dev, q, .gelu, .gelu_tanh, v, false)) continue;
        }
    }
    // The silu activation on the base variant of each quant.
    try std.testing.expect(try expectMlpParity(&dev, .q4_0, .silu, .silu, .plain, false));
    try std.testing.expect(try expectMlpParity(&dev, .q4_1, .silu, .silu, .plain, false));
    try std.testing.expect(try expectMlpParity(&dev, .q8_0, .silu, .silu, .plain, false));

    dev.sync();
}

// The discrete two-kernel form: the fused `gate_up` writes the `ffn_mid` into a
// device-local buffer and the `down` GEMV reads it back, both multi-workgroup.
// Parity must match the single-workgroup fused shader and the CPU oracle.
test "gpu two-kernel mlp parity with the v2 cpu kernel (q4_0, q4_1, q8_0)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const variants = [_]GpuVariant{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 };
    for ([_]Quant{ .q4_0, .q4_1, .q8_0 }) |q| {
        for (variants) |v| {
            if (!try expectMlpParity(&dev, q, .gelu, .gelu_tanh, v, true)) continue;
        }
    }
    try std.testing.expect(try expectMlpParity(&dev, .q4_0, .silu, .silu, .plain, true));
    try std.testing.expect(try expectMlpParity(&dev, .q4_1, .silu, .silu, .plain, true));
    try std.testing.expect(try expectMlpParity(&dev, .q8_0, .silu, .silu, .plain, true));

    dev.sync();
}

// The two kernels are independent, so the `down` may carry a different quant
// than the `gate`/`up` (a discrete-only capability).
test "gpu two-kernel mlp mixed-quant parity (gate/up q4_0, down q4_1/q8_0)" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    try std.testing.expect(try expectMlpPairMixedParity(&dev, .q4_0, .q4_1, .gelu, .gelu_tanh, .plain));
    try std.testing.expect(try expectMlpPairMixedParity(&dev, .q4_0, .q8_0, .gelu, .gelu_tanh, .plain));
    try std.testing.expect(try expectMlpPairMixedParity(&dev, .q8_0, .q4_0, .silu, .silu, .plain));

    dev.sync();
}

/// Build `rows` of a structurally valid `q4_0` weight with a seed-dependent
/// payload, so a batch of three weights produces three distinct results.
fn makeDistinctQ4(buf: []u8, k: usize, rows: usize, seed: u8) void {
    contract.fillWeights(.q4_0, buf, k, rows);
    const row_bytes = (k / 32) * 18;
    for (0..rows) |r| {
        for (0..(k / 32)) |b| {
            const off = r * row_bytes + b * 18;
            buf[off + 4 + (seed % 12)] ^= seed;
        }
    }
}

// The async seam: a batch of resident GEMVs sharing one input (the q/k/v
// shape) must equal the per-op CPU kernel, and the single async handle must
// drain the same value. This exercises the command-buffer ring and fences.
test "gpu resident batch and async handles match the cpu kernel" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const rows: usize = 64;
    const k: usize = 256;
    const row_bytes = (k / 32) * 18;

    const seeds = [_]u8{ 0x11, 0x27, 0x3d };
    var wbufs: [3][rows * row_bytes]u8 = undefined;
    var views: [3]TensorView = undefined;
    var dims = [_]u64{ k, rows };
    for (&wbufs, &views, seeds) |*wb, *view, seed| {
        const w = wb[0 .. rows * row_bytes];
        makeDistinctQ4(w, k, rows, seed);
        view.* = .{
            .name = "w",
            .dims = &dims,
            .ggml_type = .q4_0,
            .offset = 0,
            .byte_length = w.len,
            .mapped_slice = w,
        };
    }

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    var res: [3]*Resident = undefined;
    for (&wbufs, &res) |*wb, *r| {
        r.* = dev.uploadResident(wb[0 .. rows * row_bytes]) orelse return error.SkipZigTest;
    }
    defer for (res) |r| dev.freeResident(r);

    var gpu_out: [3][rows]f32 = undefined;
    const ops = [_]ResidentOp{
        .{ .res = res[0], .quant = .q4_0, .rows = rows, .out = &gpu_out[0] },
        .{ .res = res[1], .quant = .q4_0, .rows = rows, .out = &gpu_out[1] },
        .{ .res = res[2], .quant = .q4_0, .rows = rows, .out = &gpu_out[2] },
    };
    try std.testing.expect(dev.matVecResidentBatch(k, &x, &ops));

    for (&views, gpu_out) |*view, gpu| {
        var cpu: [rows]f32 = undefined;
        try matvec.matVecRows(view, &x, &cpu, 0, rows);
        try expectClose(&cpu, gpu[0..rows]);
    }

    // A single async handle drains the same value as the synchronous path.
    const h = dev.matVecResidentAsync(res[1], .q4_0, rows, k, &x) orelse return error.SkipZigTest;
    var async_out: [rows]f32 = undefined;
    try std.testing.expect(dev.waitResident(h, &async_out, rows));
    try expectClose(&gpu_out[1], &async_out);

    dev.sync();
}

// The device-owned activation path: a burst of resident GEMVs shares one
// device buffer for `x` (only the first member uploads it) and each drains its
// own result. The outputs must match the per-op CPU kernel and the batch path.
test "gpu burst async reuses one device activation" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;

    const rows: usize = 64;
    const k: usize = 256;
    const row_bytes = (k / 32) * 18;

    const seeds = [_]u8{ 0x11, 0x27, 0x3d };
    var wbufs: [3][rows * row_bytes]u8 = undefined;
    var views: [3]TensorView = undefined;
    var dims = [_]u64{ k, rows };
    for (&wbufs, &views, seeds) |*wb, *view, seed| {
        const w = wb[0 .. rows * row_bytes];
        makeDistinctQ4(w, k, rows, seed);
        view.* = .{
            .name = "w",
            .dims = &dims,
            .ggml_type = .q4_0,
            .offset = 0,
            .byte_length = w.len,
            .mapped_slice = w,
        };
    }

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    var res: [3]*Resident = undefined;
    for (&wbufs, &res) |*wb, *r| {
        r.* = dev.uploadResident(wb[0 .. rows * row_bytes]) orelse return error.SkipZigTest;
    }
    defer for (res) |r| dev.freeResident(r);

    var gpu_out: [3][rows]f32 = undefined;
    const h0 = dev.matVecResidentBurstAsync(res[0], .q4_0, rows, k, &x, true) orelse return error.SkipZigTest;
    const h1 = dev.matVecResidentBurstAsync(res[1], .q4_0, rows, k, &x, false) orelse return error.SkipZigTest;
    const h2 = dev.matVecResidentBurstAsync(res[2], .q4_0, rows, k, &x, false) orelse return error.SkipZigTest;
    try std.testing.expect(dev.waitResident(h2, &gpu_out[2], rows));
    try std.testing.expect(dev.waitResident(h1, &gpu_out[1], rows));
    try std.testing.expect(dev.waitResident(h0, &gpu_out[0], rows));

    for (&views, gpu_out) |*view, gpu| {
        var cpu: [rows]f32 = undefined;
        try matvec.matVecRows(view, &x, &cpu, 0, rows);
        try expectClose(&cpu, gpu[0..rows]);
    }

    dev.sync();
}

// ---- microbatch GEMM benchmark (`runtime/17` P0) ----

fn benchNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    if (std.posix.system.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn benchMatMulUs(dev: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, m: usize, x: []const f32, out: []f32, iters: usize) f64 {
    _ = dev.matMulResident(res, quant, rows, k, m, x, out);
    const t0 = benchNowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) _ = dev.matMulResident(res, quant, rows, k, m, x, out);
    const ns = benchNowNs() - t0;
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
}

fn benchMatVecUs(dev: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, out: []f32, iters: usize) f64 {
    const ok = dev.matVecResident(res, quant, rows, k, x, out);
    if (!ok) return -1.0;
    const t0 = benchNowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) _ = dev.matVecResident(res, quant, rows, k, x, out);
    const ns = benchNowNs() - t0;
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
}

// Measures the microbatch GEMM (and the GEMV it replaces) for the real 12B
// shapes across the verify batch sizes, printing latency and effective
// bandwidth so the bottleneck can be classified (P0). Diagnostic output only.
test "gpu microbatch gemm benchmark matrix" {
    var dev = Device.init(std.testing.allocator, undefined);
    defer dev.deinit();
    if (!dev.available()) return error.SkipZigTest;
    const a = std.testing.allocator;

    const Shape = struct { name: []const u8, quant: Quant, rows: usize, k: usize };
    const shapes = [_]Shape{
        .{ .name = "q4_0 4096x4096", .quant = .q4_0, .rows = 4096, .k = 4096 },
        .{ .name = "q4_k 4096x4096", .quant = .q4_k, .rows = 4096, .k = 4096 },
        .{ .name = "q4_k lm_head 262144x3840", .quant = .q4_k, .rows = 262144, .k = 3840 },
    };
    const ms = [_]usize{ 1, 2, 3, 4, 5, 8, 9 };

    std.debug.print("\n[bench] microbatch GEMM (resident) — latency us / effective GB/s\n", .{});
    for (shapes) |sh| {
        const row_bytes = (sh.k / sh.quant.blockElems()) * sh.quant.blockBytes();
        const bytes: usize = sh.rows * row_bytes;
        const w = a.alloc(u8, bytes) catch return error.SkipZigTest;
        defer a.free(w);
        contract.fillWeights(sh.quant, w, sh.k, sh.rows);
        const res = dev.uploadResident(w) orelse return error.SkipZigTest;
        defer dev.freeResident(res);

        const x = a.alloc(f32, 9 * sh.k) catch return error.SkipZigTest;
        defer a.free(x);
        for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7 + 3) % 17)) * 0.1 - 0.8;
        const out = a.alloc(f32, 9 * sh.rows) catch return error.SkipZigTest;
        defer a.free(out);

        const iters: usize = if (sh.rows > 100000) 4 else 30;
        const gemv_us = benchMatVecUs(&dev, res, sh.quant, sh.rows, sh.k, x[0..sh.k], out[0..sh.rows], iters);
        const gemv_gbps = @as(f64, @floatFromInt(bytes)) / (gemv_us * 1000.0);

        std.debug.print("[bench] {s}: GEMV {d:.1}us {d:.1}GB/s\n", .{ sh.name, gemv_us, gemv_gbps });
        for (ms) |m| {
            const us = benchMatMulUs(&dev, res, sh.quant, sh.rows, sh.k, m, x[0 .. m * sh.k], out[0 .. m * sh.rows], iters);
            const gbps = @as(f64, @floatFromInt(bytes)) / (us * 1000.0);
            const per_row = us / @as(f64, @floatFromInt(m));
            std.debug.print("[bench] {s}: M={d} GEMM {d:.1}us {d:.1}GB/s  per-row {d:.1}us  (M*GEMV {d:.1}us)\n", .{
                sh.name, m, us, gbps, per_row, gemv_us * @as(f64, @floatFromInt(m)),
            });
        }
    }
    dev.sync();
}

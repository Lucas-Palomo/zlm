const std = @import("std");
const fused = @import("fused.zig");
const single = @import("single.zig");
const activation = @import("../../../transformer/activation.zig");
const norm = @import("../../../transformer/norm.zig");
const TensorView = @import("../../../../data/tensor_view.zig").TensorView;

test "gate_up_fused matches gate and up matvec plus gated activation" {
    const n: usize = 4;
    const m: usize = 3;
    var gate_w = [_]f32{ 1, 2, 3, 4, -1, 0.5, 2, -2, 0, 1, 0, 1 };
    var up_w = [_]f32{ 0.5, -1, 2, 0, 1, 3, -1, 1, 2, 0, -2, 1 };
    const gate_data = std.mem.sliceAsBytes(gate_w[0..]);
    const up_data = std.mem.sliceAsBytes(up_w[0..]);
    const dims = [_]u64{ n, m };
    const vg = TensorView{ .name = "gate", .dims = &dims, .ggml_type = .f32, .offset = 0, .byte_length = gate_data.len, .mapped_slice = gate_data };
    const vu = TensorView{ .name = "up", .dims = &dims, .ggml_type = .f32, .offset = 0, .byte_length = up_data.len, .mapped_slice = up_data };
    const x = [_]f32{ 1, -1, 0.5, 2 };
    var got = [_]f32{0} ** m;
    var g = [_]f32{0} ** m;
    var u = [_]f32{0} ** m;
    var want = [_]f32{0} ** m;

    try fused.fusedGateUpRows(&vg, &vu, &x, &got, 0, m, .gelu_tanh);

    try single.matVecRows(&vg, &x, &g, 0, m);
    try single.matVecRows(&vu, &x, &u, 0, m);
    activation.gatedGelu(&g, &u, &want);
    for (0..m) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

test "mlp_fused matches gate_up plus down matvec" {
    const n: usize = 4;
    const ffn: usize = 3;
    const hidden: usize = 2;
    var gate_w = [_]f32{ 1, 2, 3, 4, -1, 0.5, 2, -2, 0, 1, 0, 1 };
    var up_w = [_]f32{ 0.5, -1, 2, 0, 1, 3, -1, 1, 2, 0, -2, 1 };
    var down_w = [_]f32{ 1, 0.5, -1, -2, 1, 0.5 };
    const gate_data = std.mem.sliceAsBytes(gate_w[0..]);
    const up_data = std.mem.sliceAsBytes(up_w[0..]);
    const down_data = std.mem.sliceAsBytes(down_w[0..]);
    const g_dims = [_]u64{ n, ffn };
    const d_dims = [_]u64{ ffn, hidden };
    const vg = TensorView{ .name = "gate", .dims = &g_dims, .ggml_type = .f32, .offset = 0, .byte_length = gate_data.len, .mapped_slice = gate_data };
    const vu = TensorView{ .name = "up", .dims = &g_dims, .ggml_type = .f32, .offset = 0, .byte_length = up_data.len, .mapped_slice = up_data };
    const vd = TensorView{ .name = "down", .dims = &d_dims, .ggml_type = .f32, .offset = 0, .byte_length = down_data.len, .mapped_slice = down_data };
    const x = [_]f32{ 1, -1, 0.5, 2 };
    var got = [_]f32{0} ** hidden;
    var mid = [_]f32{0} ** ffn;
    var g = [_]f32{0} ** ffn;
    var u = [_]f32{0} ** ffn;
    var m = [_]f32{0} ** ffn;
    var want = [_]f32{0} ** hidden;

    try fused.fusedMlpRows(&vg, &vu, &vd, &x, &got, &mid, 0, hidden, .gelu_tanh);

    try single.matVecRows(&vg, &x, &g, 0, ffn);
    try single.matVecRows(&vu, &x, &u, 0, ffn);
    activation.gatedGelu(&g, &u, &m);
    try single.matVecRows(&vd, &m, &want, 0, hidden);
    for (0..hidden) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

test "norm_proj matches rmsNorm then matvec" {
    const n: usize = 4;
    const m: usize = 3;
    var w = [_]f32{ 1, 2, 3, 4, -1, 0.5, 2, -2, 0, 1, 0, 1 };
    const data = std.mem.sliceAsBytes(w[0..]);
    const dims = [_]u64{ n, m };
    const view = TensorView{ .name = "w", .dims = &dims, .ggml_type = .f32, .offset = 0, .byte_length = data.len, .mapped_slice = data };
    const x = [_]f32{ 1, -1, 0.5, 2 };
    const weight = [_]f32{ 1, 0.5, 2, -1 };
    var got = [_]f32{0} ** m;
    var want = [_]f32{0} ** m;
    var normed = [_]f32{0} ** n;
    var scratch = [_]f32{0} ** n;

    try fused.normThenMatVecRows(&view, &x, &weight, 1e-6, &got, 0, m, &scratch);

    norm.rmsNorm(&x, &weight, 1e-6, &normed);
    try single.matVecRows(&view, &normed, &want, 0, m);
    for (0..m) |i| try std.testing.expectApproxEqAbs(want[i], got[i], 1e-5);
}

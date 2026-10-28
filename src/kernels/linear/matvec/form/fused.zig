const std = @import("std");
const types = @import("../types.zig");
const layout = @import("../layout.zig");
const norm = @import("../../../transformer/norm.zig");
const simd = @import("../../../simd.zig");
const TensorView = @import("../../../../data/tensor_view.zig").TensorView;

pub const KernelError = types.KernelError;
pub const ActivationKind = types.ActivationKind;

pub fn normThenMatVecRows(
    view: *const TensorView,
    x: []const f32,
    weight: []const f32,
    eps: f32,
    out: []f32,
    begin: usize,
    end: usize,
    scratch: []f32,
) KernelError!void {
    if (x.len != weight.len) return error.ShapeMismatch;
    if (scratch.len < x.len) return error.ShapeMismatch;
    norm.rmsNorm(x, weight, eps, scratch[0..x.len]);
    return matVecRowsWithPlan(view, scratch[0..x.len], out, begin, end, .production);
}

pub fn fusedGateUpRows(
    v_gate: *const TensorView,
    v_up: *const TensorView,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    const data_g = v_gate.mapped_slice orelse return error.NoMappedData;
    const data_u = v_up.mapped_slice orelse return error.NoMappedData;
    if (v_gate.dims.len != 2 or v_up.dims.len != 2) return error.ShapeMismatch;
    const n_g: usize = @intCast(v_gate.dims[0]);
    const m_g: usize = @intCast(v_gate.dims[1]);
    const n_u: usize = @intCast(v_up.dims[0]);
    const m_u: usize = @intCast(v_up.dims[1]);
    if (n_g != n_u or m_g != m_u) return error.ShapeMismatch;
    if (x.len != n_g) return error.ShapeMismatch;
    if (mid_out.len != m_g) return error.ShapeMismatch;
    if (begin > end or end > m_g) return error.InvalidRange;

    if (v_gate.layout == .q4_0_scale_split and v_up.layout == .q4_0_scale_split and
        v_gate.ggml_type == .q4_0 and v_up.ggml_type == .q4_0)
    {
        return fusedGateUpQ4_0ScaleSplit(data_g, data_u, n_g, m_g, x, mid_out, begin, end, act);
    }

    if (v_gate.layout == .q4_1_scale_split and v_up.layout == .q4_1_scale_split and
        v_gate.ggml_type == .q4_1 and v_up.ggml_type == .q4_1)
    {
        return fusedGateUpQ4_1ScaleSplit(data_g, data_u, n_g, m_g, x, mid_out, begin, end, act);
    }

    if (v_gate.layout.isNative() and v_up.layout.isNative() and
        v_gate.ggml_type == .q4_0 and v_up.ggml_type == .q4_0)
    {
        return fusedGateUpQ4_0Native(data_g, data_u, n_g, m_g, x, mid_out, begin, end, act);
    }

    if (v_gate.layout.isNative() and v_up.layout.isNative() and
        v_gate.ggml_type == .q4_1 and v_up.ggml_type == .q4_1)
    {
        return fusedGateUpQ4_1Native(data_g, data_u, n_g, m_g, x, mid_out, begin, end, act);
    }

    var stack_g: [16384]f32 = undefined;
    var stack_u: [16384]f32 = undefined;
    const g_buf = if (m_g <= stack_g.len) stack_g[0..m_g] else try std.heap.page_allocator.alloc(f32, m_g);
    defer if (m_g > stack_g.len) std.heap.page_allocator.free(g_buf);
    const u_buf = if (m_g <= stack_u.len) stack_u[0..m_g] else try std.heap.page_allocator.alloc(f32, m_g);
    defer if (m_g > stack_u.len) std.heap.page_allocator.free(u_buf);

    try matVecRowsWithPlan(v_gate, x, g_buf, begin, end, .production);
    try matVecRowsWithPlan(v_up, x, u_buf, begin, end, .production);

    for (begin..end) |i| {
        mid_out[i] = act.apply(g_buf[i], u_buf[i]);
    }
}

pub fn fusedGateUpQ4_0Native(
    data_g: []const u8,
    data_u: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data_g.len < need or data_u.len < need) return error.TruncatedData;
    if (x.len != n) return error.ShapeMismatch;
    if (mid_out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;

    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        const row_g = data_g[i * row_bytes ..][0..row_bytes];
        const row_u = data_u[i * row_bytes ..][0..row_bytes];

        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const bg = row_g[rb * 18 ..][0..18];
            const bu = row_u[rb * 18 ..][0..18];
            const dg: f32 = @floatCast(readF16(bg[0..2]));
            const du: f32 = @floatCast(readF16(bu[0..2]));
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.q4_0BlockVec(8, bg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.q4_0BlockVec(8, bu, xb), u_acc);
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc), @reduce(.Add, u_acc));
    }
}

pub fn fusedGateUpQ4_1Native(
    data_g: []const u8,
    data_u: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data_g.len < need or data_u.len < need) return error.TruncatedData;
    if (x.len != n) return error.ShapeMismatch;
    if (mid_out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    if (row_blocks > types.max_activation_cache_blocks) {
        return fusedGateUpQ4_1NativeNoCache(data_g, data_u, n, m, x, mid_out, begin, end, act);
    }

    var x_sums: [types.max_activation_cache_blocks]f32 = undefined;
    for (0..row_blocks) |rb| {
        const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
        x_sums[rb] = layout.xBlockSum(xb);
    }

    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        var g_msum: f32 = 0;
        var u_msum: f32 = 0;
        const row_g = data_g[i * row_bytes ..][0..row_bytes];
        const row_u = data_u[i * row_bytes ..][0..row_bytes];

        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const x_sum = x_sums[rb];
            const bg = row_g[rb * 20 ..][0..20];
            const bu = row_u[rb * 20 ..][0..20];
            const qsg: *align(1) const [16]u8 = @ptrCast(bg[4..20].ptr);
            const qsu: *align(1) const [16]u8 = @ptrCast(bu[4..20].ptr);
            const dg: f32 = @floatCast(readF16(bg[0..2]));
            const du: f32 = @floatCast(readF16(bu[0..2]));
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.q4_1BlockVec(qsg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.q4_1BlockVec(qsu, xb), u_acc);
            g_msum += @as(f32, @floatCast(readF16(bg[2..4]))) * x_sum;
            u_msum += @as(f32, @floatCast(readF16(bu[2..4]))) * x_sum;
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc) + g_msum, @reduce(.Add, u_acc) + u_msum);
    }
}

fn fusedGateUpQ4_1NativeNoCache(
    data_g: []const u8,
    data_u: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data_g.len < need or data_u.len < need) return error.TruncatedData;
    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        var g_msum: f32 = 0;
        var u_msum: f32 = 0;
        const row_g = data_g[i * row_bytes ..][0..row_bytes];
        const row_u = data_u[i * row_bytes ..][0..row_bytes];
        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const x_sum = layout.xBlockSum(xb);
            const bg = row_g[rb * 20 ..][0..20];
            const bu = row_u[rb * 20 ..][0..20];
            const qsg: *align(1) const [16]u8 = @ptrCast(bg[4..20].ptr);
            const qsu: *align(1) const [16]u8 = @ptrCast(bu[4..20].ptr);
            const dg: f32 = @floatCast(readF16(bg[0..2]));
            const du: f32 = @floatCast(readF16(bu[0..2]));
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.q4_1BlockVec(qsg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.q4_1BlockVec(qsu, xb), u_acc);
            g_msum += @as(f32, @floatCast(readF16(bg[2..4]))) * x_sum;
            u_msum += @as(f32, @floatCast(readF16(bu[2..4]))) * x_sum;
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc) + g_msum, @reduce(.Add, u_acc) + u_msum);
    }
}

pub fn fusedGateUpQ4_0ScaleSplit(
    data_g: []const u8,
    data_u: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 18;
    const need = m * row_bytes;
    if (data_g.len < need or data_u.len < need) return error.TruncatedData;
    if (x.len != n) return error.ShapeMismatch;
    if (mid_out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const scales_total = m * row_blocks * 2;
    const scales_g = data_g[0..scales_total];
    const quants_g = data_g[scales_total..need];
    const scales_u = data_u[0..scales_total];
    const quants_u = data_u[scales_total..need];

    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        const s_row_g = scales_g[i * row_blocks * 2 ..][0 .. row_blocks * 2];
        const q_row_g = quants_g[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        const s_row_u = scales_u[i * row_blocks * 2 ..][0 .. row_blocks * 2];
        const q_row_u = quants_u[i * row_blocks * 16 ..][0 .. row_blocks * 16];

        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const dg: f32 = @floatCast(readF16(s_row_g[rb * 2 ..][0..2]));
            const du: f32 = @floatCast(readF16(s_row_u[rb * 2 ..][0..2]));
            const qsg: *align(1) const [16]u8 = @ptrCast(&q_row_g[rb * 16]);
            const qsu: *align(1) const [16]u8 = @ptrCast(&q_row_u[rb * 16]);
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.scaleSplitBlockVec(qsg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.scaleSplitBlockVec(qsu, xb), u_acc);
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc), @reduce(.Add, u_acc));
    }
}

pub fn fusedGateUpQ4_1ScaleSplit(
    data_g: []const u8,
    data_u: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (n % 32 != 0) return error.ShapeMismatch;
    const row_blocks = n / 32;
    const row_bytes = row_blocks * 20;
    const need = m * row_bytes;
    if (data_g.len < need or data_u.len < need) return error.TruncatedData;
    if (x.len != n) return error.ShapeMismatch;
    if (mid_out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    const meta_total = m * row_blocks * 4;
    const meta_g = data_g[0..meta_total];
    const quants_g = data_g[meta_total..need];
    const meta_u = data_u[0..meta_total];
    const quants_u = data_u[meta_total..need];

    if (row_blocks > types.max_activation_cache_blocks) {
        return fusedGateUpQ4_1ScaleSplitNoCache(meta_g, quants_g, meta_u, quants_u, row_blocks, m, x, mid_out, begin, end, act);
    }

    var x_sums: [types.max_activation_cache_blocks]f32 = undefined;
    for (0..row_blocks) |rb| {
        const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
        x_sums[rb] = layout.xBlockSum(xb);
    }

    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        var g_msum: f32 = 0;
        var u_msum: f32 = 0;
        const m_row_g = meta_g[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row_g = quants_g[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        const m_row_u = meta_u[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row_u = quants_u[i * row_blocks * 16 ..][0 .. row_blocks * 16];

        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const x_sum = x_sums[rb];
            const mg = m_row_g[rb * 4 ..][0..4];
            const mu = m_row_u[rb * 4 ..][0..4];
            const qsg: *align(1) const [16]u8 = @ptrCast(q_row_g[rb * 16 ..][0..16].ptr);
            const qsu: *align(1) const [16]u8 = @ptrCast(q_row_u[rb * 16 ..][0..16].ptr);
            const dg: f32 = @floatCast(readF16(mg[0..2]));
            const du: f32 = @floatCast(readF16(mu[0..2]));
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.q4_1ScaleSplitBlockVec(qsg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.q4_1ScaleSplitBlockVec(qsu, xb), u_acc);
            g_msum += @as(f32, @floatCast(readF16(mg[2..4]))) * x_sum;
            u_msum += @as(f32, @floatCast(readF16(mu[2..4]))) * x_sum;
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc) + g_msum, @reduce(.Add, u_acc) + u_msum);
    }
}

fn fusedGateUpQ4_1ScaleSplitNoCache(
    meta_g: []const u8,
    quants_g: []const u8,
    meta_u: []const u8,
    quants_u: []const u8,
    row_blocks: usize,
    m: usize,
    x: []const f32,
    mid_out: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    _ = m;
    // Deferred reduction: one horizontal reduce per row (gate and up).
    const V = @Vector(8, f32);
    for (begin..end) |i| {
        var g_acc: V = @splat(0);
        var u_acc: V = @splat(0);
        var g_msum: f32 = 0;
        var u_msum: f32 = 0;
        const m_row_g = meta_g[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row_g = quants_g[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        const m_row_u = meta_u[i * row_blocks * 4 ..][0 .. row_blocks * 4];
        const q_row_u = quants_u[i * row_blocks * 16 ..][0 .. row_blocks * 16];
        for (0..row_blocks) |rb| {
            const xb: *align(1) const [32]f32 = @ptrCast(x[rb * 32 ..][0..32].ptr);
            const mg = m_row_g[rb * 4 ..][0..4];
            const mu = m_row_u[rb * 4 ..][0..4];
            const qsg: *align(1) const [16]u8 = @ptrCast(q_row_g[rb * 16 ..][0..16].ptr);
            const qsu: *align(1) const [16]u8 = @ptrCast(q_row_u[rb * 16 ..][0..16].ptr);
            const dg: f32 = @floatCast(readF16(mg[0..2]));
            const du: f32 = @floatCast(readF16(mu[0..2]));
            g_acc = simd.fma(8, @as(V, @splat(dg)), layout.q4_1ScaleSplitBlockVec(qsg, xb), g_acc);
            u_acc = simd.fma(8, @as(V, @splat(du)), layout.q4_1ScaleSplitBlockVec(qsu, xb), u_acc);
            g_msum += @as(f32, @floatCast(readF16(mg[2..4]))) * layout.xBlockSum(xb);
            u_msum += @as(f32, @floatCast(readF16(mu[2..4]))) * layout.xBlockSum(xb);
        }
        mid_out[i] = act.apply(@reduce(.Add, g_acc) + g_msum, @reduce(.Add, u_acc) + u_msum);
    }
}

/// The whole SwiGLU MLP in one pass: `m_j = act(gate_j·x, up_j·x)` for every
/// ffn row `j` (the gated activation), then `out_r = down_r·m` over the computed
/// `m` in `mid`. The `ffn_mid` never leaves the kernel — it lives only in the
/// caller's `mid` scratch, so no dispatch split and no extra activation pass.
pub fn fusedMlpRows(
    v_gate: *const TensorView,
    v_up: *const TensorView,
    v_down: *const TensorView,
    x: []const f32,
    out: []f32,
    mid: []f32,
    begin: usize,
    end: usize,
    act: ActivationKind,
) KernelError!void {
    if (v_gate.dims.len != 2 or v_up.dims.len != 2 or v_down.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(v_gate.dims[0]);
    const ffn: usize = @intCast(v_gate.dims[1]);
    const n_down: usize = @intCast(v_down.dims[0]);
    const hidden: usize = @intCast(v_down.dims[1]);
    if (n_down != ffn) return error.ShapeMismatch;
    if (x.len != n) return error.ShapeMismatch;
    if (mid.len < ffn) return error.ShapeMismatch;
    if (out.len < hidden) return error.ShapeMismatch;
    if (begin > end or end > hidden) return error.InvalidRange;

    const m = mid[0..ffn];
    try fusedGateUpRows(v_gate, v_up, x, m, 0, ffn, act);
    return matVecRowsWithPlan(v_down, m, out, begin, end, .production);
}

fn matVecRowsWithPlan(view: *const TensorView, x: []const f32, out: []f32, begin: usize, end: usize, requested_plan: types.MatVecPlan) KernelError!void {
    const single = @import("single.zig");
    return single.matVecRowsWithPlan(view, x, out, begin, end, requested_plan, 0);
}

fn readF16(data: []const u8) f16 {
    const raw = std.mem.readInt(u16, data[0..2], .little);
    return @bitCast(raw);
}

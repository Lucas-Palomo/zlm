const std = @import("std");
const types = @import("../types.zig");
const TensorView = @import("../../../../data/tensor_view.zig").TensorView;
const dequant = @import("../../quant/root.zig");
const dot = @import("../../vector.zig").dot;
const simd = @import("../../../simd.zig");
const layout = @import("../layout.zig");
const quantized = @import("../kernels/quantized.zig");
const float = @import("../kernels/float.zig");
const bfloat = @import("../kernels/bfloat.zig");
const batched = @import("batched.zig");
const row_dot = @import("dot.zig");

const KernelError = types.KernelError;
const MatVecPlan = types.MatVecPlan;
const Variant = types.Variant;
const Dot = types.Dot;
const Activation = layout.Activation;
const max_activation_cache_blocks = types.max_activation_cache_blocks;
const quantizeActivationQ8Sum = layout.quantizeActivationQ8Sum;
const q4KXSum = layout.q4KXSum;

/// The L3 sweep — the single generic body every kernel x variant instantiation
/// shares. It prepares the activation context once, then sweeps `tile` output
/// rows together and `unroll` blocks per iteration with independent
/// accumulators (both `inline for`-expanded), optionally prefetching the next
/// blocks. `dot_kind` selects the L2 step (`K.fpBlock`/`K.intBlock`), resolved
/// at comptime — no runtime switch.
pub fn sweep(
    comptime K: type,
    comptime V: Variant,
    comptime dot_kind: Dot,
    data: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    begin: usize,
    end: usize,
) KernelError!void {
    @setEvalBranchQuota(100_000);
    const lanes: usize = comptime if (V.width == 0) simd.native.lanes() else V.width;
    comptime std.debug.assert(lanes == 4 or lanes == 8 or lanes == 16);
    comptime std.debug.assert(V.tile >= 1);
    comptime std.debug.assert(V.unroll >= 1);
    if (n % K.block_elems != 0) return error.ShapeMismatch;
    const row_blocks = n / K.block_elems;
    const row_bytes = row_blocks * K.block_bytes;
    const need = m * row_bytes;
    if (data.len < need) return error.TruncatedData;

    const T: usize = V.tile;
    const U: usize = V.unroll;
    const Vec = @Vector(lanes, f32);

    // The activation context: quantize to q8 once for `int`, or precompute the
    // per-32-chunk sums the min-term `fp` kernels need. A too-large activation
    // falls back to the fp body (the q8/xsum caches are bounded).
    const chunks = n / 32;
    var act = Activation{ .x = x };
    var qx_buf: [max_activation_cache_blocks * 32]i8 = undefined;
    var dx_buf: [max_activation_cache_blocks]f32 = undefined;
    var sum_buf: [max_activation_cache_blocks]i32 = undefined;
    var xsum_buf: [max_activation_cache_blocks]f32 = undefined;
    if (comptime dot_kind == .int) {
        if (chunks > max_activation_cache_blocks) {
            return matVecFpGen(K, V, data, n, m, x, out, begin, end);
        }
        quantizeActivationQ8Sum(x, chunks, qx_buf[0..n], dx_buf[0..chunks], sum_buf[0..chunks]);
        act.qx = qx_buf[0..n];
        act.dx = dx_buf[0..chunks];
        act.sums = sum_buf[0..chunks];
    } else if (comptime K.has_min) {
        if (chunks > max_activation_cache_blocks) {
            return K.production_fp(data, n, m, x, out, begin, end);
        }
        q4KXSum(x, xsum_buf[0..chunks]);
        act.xsum = xsum_buf[0..chunks];
    }

    var i = begin;
    while (i + T <= end) : (i += T) {
        // A flat `[T*U]` register array keeps the codegen identical to the
        // hand-written per-row unrolled body when `T == 1`.
        var acc: [T * U]Vec = undefined;
        var corr: [T]f32 = undefined;
        inline for (0..T * U) |k| acc[k] = @splat(0);
        inline for (0..T) |r| corr[r] = 0;
        var rb: usize = 0;
        while (rb + U <= row_blocks) : (rb += U) {
            inline for (0..T) |r| {
                const row = data[(i + r) * row_bytes ..][0..row_bytes];
                if (comptime V.prefetch) {
                    if (rb + U < row_blocks) @prefetch(&row[(rb + U) * K.block_bytes], .{});
                }
                inline for (0..U) |u| {
                    const block = row[(rb + u) * K.block_bytes ..][0..K.block_bytes];
                    if (comptime dot_kind == .fp) {
                        K.fpBlock(lanes, block, act, rb + u, &acc[r * U + u], &corr[r]);
                    } else {
                        K.intBlock(lanes, block, act, rb + u, &acc[r * U + u], &corr[r]);
                    }
                }
            }
        }
        while (rb < row_blocks) : (rb += 1) {
            inline for (0..T) |r| {
                const row = data[(i + r) * row_bytes ..][0..row_bytes];
                const block = row[rb * K.block_bytes ..][0..K.block_bytes];
                if (comptime dot_kind == .fp) {
                    K.fpBlock(lanes, block, act, rb, &acc[r * U], &corr[r]);
                } else {
                    K.intBlock(lanes, block, act, rb, &acc[r * U], &corr[r]);
                }
            }
        }
        inline for (0..T) |r| {
            var sum: Vec = acc[r * U];
            inline for (1..U) |u| sum += acc[r * U + u];
            out[i + r] = @reduce(.Add, sum) + corr[r];
        }
    }
    // The tile remainder (only tiled instantiations have one) is swept one row
    // at a time with the same kernel.
    while (i < end) : (i += 1) {
        var acc: Vec = @splat(0);
        var corr: f32 = 0;
        const row = data[i * row_bytes ..][0..row_bytes];
        var rb: usize = 0;
        while (rb < row_blocks) : (rb += 1) {
            const block = row[rb * K.block_bytes ..][0..K.block_bytes];
            if (comptime dot_kind == .fp) {
                K.fpBlock(lanes, block, act, rb, &acc, &corr);
            } else {
                K.intBlock(lanes, block, act, rb, &acc, &corr);
            }
        }
        out[i] = @reduce(.Add, acc) + corr;
    }
}

/// The f32 specialization of the sweep for kernel `K`.
pub fn matVecFpGen(comptime K: type, comptime V: Variant, data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return sweep(K, V, .fp, data, n, m, x, out, begin, end);
}

/// The integer specialization of the sweep for kernel `K`.
pub fn matVecIntGen(comptime K: type, comptime V: Variant, data: []const u8, n: usize, m: usize, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return sweep(K, V, .int, data, n, m, x, out, begin, end);
}

pub fn matVec(view: *const TensorView, x: []const f32, out: []f32) KernelError!void {
    if (view.dims.len != 2) return error.ShapeMismatch;
    const m: usize = @intCast(view.dims[1]);
    return matVecRows(view, x, out, 0, m);
}

pub fn matVecRows(view: *const TensorView, x: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    return matVecRowsWithPlan(view, x, out, begin, end, .production, 0);
}

pub fn matVecWithBiasRows(view: *const TensorView, x: []const f32, bias: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    try matVecRows(view, x, out, begin, end);
    if (bias.len != out.len) return error.ShapeMismatch;
    addBiasRange(out, bias, begin, end);
}

pub fn matVecRowsWithPlan(view: *const TensorView, x: []const f32, out: []f32, begin: usize, end: usize, requested_plan: MatVecPlan, simd_width: u8) KernelError!void {
    const data = view.mapped_slice orelse return error.NoMappedData;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const m: usize = @intCast(view.dims[1]);
    if (x.len != n or out.len != m) return error.ShapeMismatch;
    if (begin > end or end > m) return error.InvalidRange;
    _ = simd_width;

    switch (view.layout) {
        .q4_0_scale_split => return layout.matVecQ4_0ScaleSplit(data, n, m, x, out, begin, end),
        .q4_1_scale_split => return layout.matVecQ4_1ScaleSplit(data, n, m, x, out, begin, end),
        .native_quant => {},
        else => return error.UnsupportedLayout,
    }

    // The generated kernel x variant catalog: the plan id selects one candidate,
    // and the generic sweep instantiates its kernel type (L1) and dot (L2). The
    // catalog is the single source; no per-variant tag or body.
    @setEvalBranchQuota(1_000_000);
    inline for (types.candidates) |cand| {
        if (requested_plan == cand.plan) {
            return dispatchCandidate(cand, data, n, m, x, out, begin, end);
        }
    }

    switch (view.ggml_type) {
        .q4_0 => return quantized.Q4_0.production_fp(data, n, m, x, out, begin, end),
        .q4_1 => return quantized.Q4_1.production_fp(data, n, m, x, out, begin, end),
        .q8_0 => return quantized.Q8_0.production_fp(data, n, m, x, out, begin, end),
        .q6_k => return quantized.Q6_K.production_fp(data, n, m, x, out, begin, end),
        .q4_k => return quantized.Q4_K.production_fp(data, n, m, x, out, begin, end),
        .f32 => return float.matVecF32(data, n, m, x, out, begin, end),
        .f16 => return float.matVecF16(data, n, m, x, out, begin, end),
        .bf16 => return bfloat.matVecBF16(data, n, m, x, out, begin, end),
        else => return error.UnsupportedType,
    }
}

pub fn matMul(view: *const TensorView, activations: []const f32, out: []f32) KernelError!void {
    if (view.dims.len != 2) return error.ShapeMismatch;
    const out_features: usize = @intCast(view.dims[1]);
    return matMulRows(view, activations, out, 0, out_features);
}

pub fn matMulRows(view: *const TensorView, activations: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const out_features: usize = @intCast(view.dims[1]);
    if (n == 0) return error.ShapeMismatch;
    if (activations.len % n != 0) return error.ShapeMismatch;
    const M = activations.len / n;
    if (out.len != M * out_features) return error.ShapeMismatch;
    if (begin > end or end > out_features) return error.InvalidRange;
    if (M == 0) return;
    if (M == 1) return matVecRows(view, activations, out, begin, end);

    var tok: usize = 0;
    while (tok < M) {
        const sub_M = @min(M - tok, 32);
        const sub_act = activations[tok * n .. (tok + sub_M) * n];
        const sub_out = out[tok * out_features .. (tok + sub_M) * out_features];
        try matMulSubBatch(view, n, out_features, sub_act, sub_out, sub_M, begin, end);
        tok += sub_M;
    }
}

pub fn matMulWithBiasRows(view: *const TensorView, activations: []const f32, bias: []const f32, out: []f32, begin: usize, end: usize) KernelError!void {
    try matMulRows(view, activations, out, begin, end);
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const out_features: usize = @intCast(view.dims[1]);
    if (n == 0 or activations.len % n != 0) return error.ShapeMismatch;
    const M = activations.len / n;
    if (bias.len != out_features or out.len != M * out_features) return error.ShapeMismatch;
    for (0..M) |tok| {
        addBiasRange(out[tok * out_features ..][0..out_features], bias, begin, end);
    }
}

fn addBiasRange(out: []f32, bias: []const f32, begin: usize, end: usize) void {
    var i = begin;
    while (i + 16 <= end) : (i += 16) {
        const o: @Vector(16, f32) = @as(*align(1) const @Vector(16, f32), @ptrCast(&out[i])).*;
        const b: @Vector(16, f32) = @as(*align(1) const @Vector(16, f32), @ptrCast(&bias[i])).*;
        @as(*align(1) @Vector(16, f32), @ptrCast(&out[i])).* = o + b;
    }
    while (i < end) : (i += 1) out[i] += bias[i];
}

fn matMulSubBatch(view: *const TensorView, n: usize, out_features: usize, activations: []const f32, out: []f32, M: usize, begin: usize, end: usize) KernelError!void {
    const data = view.mapped_slice orelse return error.NoMappedData;
    if (view.layout == .q4_0_scale_split and view.ggml_type == .q4_0) {
        if (M >= 4) return batched.matMulRegisterTiledQ4_0ScaleSplit(data, n, out_features, activations, out, M, begin, end);
        return batched.matVecQ4_0ScaleSplitBatchedRows(data, n, out_features, activations, out, M, begin, end);
    }
    if (view.layout == .q4_1_scale_split and view.ggml_type == .q4_1) {
        if (M >= 4) return batched.matMulRegisterTiledQ4_1ScaleSplit(data, n, out_features, activations, out, M, begin, end);
        return batched.matVecQ4_1ScaleSplitBatchedRows(data, n, out_features, activations, out, M, begin, end);
    }
    if (view.layout.isNative()) {
        switch (view.ggml_type) {
            .q4_0 => {
                const row_bytes = dequant.q4_0RowBytes(n);
                if (data.len < out_features * row_bytes) return error.TruncatedData;
                if (M >= 4) return batched.matMulRegisterTiledQ4_0(data, n, out_features, activations, out, M, begin, end);
                return batched.matVecQ4_0BatchedRows(data, n, out_features, activations, out, M, begin, end);
            },
            .q4_1 => {
                if (M >= 4) return batched.matMulRegisterTiledQ4_1(data, n, out_features, activations, out, M, begin, end);
                return batched.matVecQ4_1BatchedRows(data, n, out_features, activations, out, M, begin, end);
            },
            .q8_0 => {
                if (M >= 4) return batched.matMulRegisterTiledQ8_0(data, n, out_features, activations, out, M, begin, end);
                return batched.matVecQ8_0BatchedRows(data, n, out_features, activations, out, M, begin, end);
            },
            .q6_k => return batched.matVecQ6_KBatchedRows(data, n, out_features, activations, out, M, begin, end),
            .bf16 => return batched.matVecBF16BatchedRows(data, n, out_features, activations, out, M, begin, end),
            else => {},
        }
    }
    for (0..M) |i| {
        try matVecRows(view, activations[i * n ..][0..n], out[i * out_features ..][0..out_features], begin, end);
    }
}

pub fn lmHeadRows(view: *const TensorView, hidden: []const f32, logits: []f32, begin: usize, end: usize, scratch: []f32) KernelError!void {
    if (!view.layout.isNative()) return error.UnsupportedLayout;
    const data = view.mapped_slice orelse return error.NoMappedData;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const rows: usize = @intCast(view.dims[1]);
    if (hidden.len != n) return error.ShapeMismatch;
    if (logits.len != rows) return error.ShapeMismatch;
    if (begin > end or end > rows) return error.InvalidRange;
    if (view.ggml_type == .q6_k) {
        const row_bytes = dequant.q6KRowBytes(n);
        if (data.len < rows * row_bytes) return error.TruncatedData;
        var i = begin;
        while (i < end) : (i += 1) {
            const row = data[i * row_bytes ..][0..row_bytes];
            logits[i] = try row_dot.dot(quantized.Q6_K, row, hidden, null);
        }
        return;
    }
    if (view.ggml_type == .q4_k) {
        const row_bytes = dequant.q4KRowBytes(n);
        if (data.len < rows * row_bytes) return error.TruncatedData;
        const chunks = n / 32;
        var xsum_buf: [1024]f32 = undefined;
        if (chunks <= xsum_buf.len) {
            layout.q4KXSum(hidden, xsum_buf[0..chunks]);
            var i = begin;
            while (i < end) : (i += 1) {
                const row = data[i * row_bytes ..][0..row_bytes];
                logits[i] = try row_dot.dot(quantized.Q4_K, row, hidden, xsum_buf[0..chunks]);
            }
            return;
        }
        var i = begin;
        while (i < end) : (i += 1) {
            const row = data[i * row_bytes ..][0..row_bytes];
            logits[i] = try row_dot.dot(quantized.Q4_K, row, hidden, null);
        }
        return;
    }
    if (view.ggml_type == .q8_0) {
        const row_bytes = dequant.q8_0RowBytes(n);
        if (data.len < rows * row_bytes) return error.TruncatedData;
        var i = begin;
        while (i < end) : (i += 1) {
            const row = data[i * row_bytes ..][0..row_bytes];
            logits[i] = try row_dot.dot(quantized.Q8_0, row, hidden, null);
        }
        return;
    }
    if (view.ggml_type == .q4_1) {
        return quantized.Q4_1.production_fp(data, n, rows, hidden, logits, begin, end);
    }
    const row_bytes: usize = switch (view.ggml_type) {
        .f32 => n * 4,
        .f16 => n * 2,
        else => return error.UnsupportedType,
    };
    if (data.len < rows * row_bytes) return error.TruncatedData;
    if (scratch.len < n) return error.ShapeMismatch;
    var i = begin;
    while (i < end) : (i += 1) {
        const row = data[i * row_bytes ..][0..row_bytes];
        try dequant.dequantTo(view.ggml_type, row, scratch[0..n]);
        logits[i] = dot(scratch[0..n], hidden);
    }
}

/// The L1 weight type of a catalog kernel tag.
fn kernelType(comptime tag: types.KernelTag) type {
    return switch (tag) {
        .q4_0 => quantized.Q4_0,
        .q4_1 => quantized.Q4_1,
        .q4_k => quantized.Q4_K,
        .q8_0 => quantized.Q8_0,
        .q6_k => quantized.Q6_K,
        .f32 => float.F32,
        .f16 => float.F16,
        .bf16 => bfloat.BF16,
    };
}

/// Instantiate one generated candidate: its kernel type (L1) and dot (L2) select
/// the sweep specialization. The dispatch is a comptime switch over the catalog,
/// so the body itself has no `switch (quant)` and no runtime dot branch.
pub fn dispatchCandidate(
    comptime cand: types.Candidate,
    data: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    begin: usize,
    end: usize,
) KernelError!void {
    const K = comptime kernelType(cand.kernel.tag);
    if (comptime cand.kernel.dot == .int) {
        return matVecIntGen(K, cand.variant, data, n, m, x, out, begin, end);
    }
    return matVecFpGen(K, cand.variant, data, n, m, x, out, begin, end);
}

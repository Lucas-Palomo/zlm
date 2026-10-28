const std = @import("std");
const ggml = @import("../gguf/format.zig");
const api = @import("../kernels/api.zig");
const registry = @import("../kernels/registry.zig");
const matvec = @import("../kernels/linear/matvec/root.zig");
const matmul = @import("../kernels/linear/matmul/root.zig");
const caps_mod = @import("caps.zig");
const calibrate = @import("calibrate.zig");
const cost = @import("cost.zig");

const Caps = caps_mod.Caps;

/// Output-row count at or above which the matmul family's tiled variant is
/// preferred. Re-exported from the matvec module that owns the constant.
pub const tile_min_rows: u64 = matvec.tile_min_rows;

/// The context the generic Selector ranks candidates against. `m`/`n`/`k` are
/// the family's shape dims (see each family's `applicable`/`cost`), `rows` the
/// dimension parallelism may split, and `working_set` the cache-aware bytes
/// (`0` = use the candidate's own bytes).
pub const SelectCtx = struct {
    caps: Caps,
    /// Raw ggml type code; `0` for untyped ops (e.g. attention).
    type_code: u32,
    m: u64,
    n: u64,
    k: u64,
    rows: u64,
    threads: u16,
    tier: cost.Tier = .ram,
    working_set: u64 = 0,
    /// P2: whether the q4_0 `prefetch = 1` variants are eligible. The plan sets
    /// it from `cost.prefetchWorthwhile` (caps + roofline); the pruned variants
    /// are never ranked, so the benchmark/plan never sees a prefetch that cannot
    /// win. `true` by default (the non-prefetch axes are unaffected).
    allow_prefetch: bool = true,
};

/// The Selector's generic result: the chosen candidate and its parallelism.
pub const Selection = struct {
    atomic_id: u16,
    variant: u16,
    parts: u16,
};

/// Family-agnostic Selector. Enumerate `registry.kernelsFor(op)`, keep the
/// `cpu` candidates whose `applicable` accepts the type/shape, and pick the one
/// minimizing `cost.opCost` at one part (the roofline is monotone in `parts`, so
/// the ranking is part-independent). The candidate's cost is divided by the
/// device-calibrated per-variant efficiency, so interchangeable variants rank by
/// their measured throughput. `parts` is then the smallest value in
/// `1..=threads` (and `<= rows`) with a strict `opCost` improvement. No family
/// code is consulted.
pub fn select(op: api.Op, ctx: SelectCtx) Selection {
    const lanes = ctx.caps.simd_lanes;
    var best: ?api.Kernel = null;
    var best_cost: f64 = std.math.inf(f64);
    for (registry.kernelsFor(op)) |c| {
        if (c.tier != .cpu) continue;
        // P2: the pruned q4_0 `prefetch = 1` variants are never ranked.
        if (op == .matvec and !ctx.allow_prefetch) {
            if (std.enums.fromInt(matvec.MatVecPlan, c.variant)) |p| {
                if (matvec.isPrefetchPlan(p)) continue;
            }
        }
        if (c.applicable) |f| {
            if (!f(ctx.type_code, ctx.m, ctx.n, lanes)) continue;
        }
        const cc = candidateCost(c, ctx, lanes);
        const ws = if (ctx.working_set != 0) ctx.working_set else cc.bytes;
        const base = cost.opCost(op, cc.bytes, cc.flops, 1, ctx.tier, ws, ctx.caps);
        const rank = base / variantEff(op, c.variant, ctx.caps);
        if (rank < best_cost) {
            best_cost = rank;
            best = c;
        }
    }
    const chosen = best orelse return .{ .atomic_id = 0, .variant = 0, .parts = 1 };
    return .{
        .atomic_id = chosen.atomic_id,
        .variant = chosen.variant,
        .parts = selectParts(op, ctx, chosen, lanes),
    };
}

/// Calibrated efficiency of `variant` in `(0, 1]` for `op`; `1.0` for ops or
/// variants with no calibrated entry. A variant that reaches a smaller fraction
/// of the best measured throughput is charged proportionally more time.
fn variantEff(op: api.Op, variant: u16, caps: Caps) f64 {
    const table: *const [calibrate.max_variants]f64 = switch (op) {
        .matvec => &caps.variant_eff.matvec,
        .matmul => &caps.variant_eff.matmul,
        else => return 1.0,
    };
    if (variant >= table.len) return 1.0;
    const e = table[variant];
    if (!(e > 0) or e > 1.0) return 1.0;
    return e;
}

fn candidateCost(c: api.Kernel, ctx: SelectCtx, lanes: u16) api.Cost {
    if (c.cost) |f| return f(ctx.type_code, ctx.m, ctx.n, ctx.k, lanes);
    return .{};
}

/// Smallest `parts` in `1..=threads` (and `<= rows`) that minimizes `opCost` for
/// the chosen candidate. The roofline flattens once the op stops being
/// compute-limited: requiring a strict improvement keeps the first minimizer, so
/// a memory-bound op saturates early instead of oversubscribing.
fn selectParts(op: api.Op, ctx: SelectCtx, c: api.Kernel, lanes: u16) u16 {
    if (ctx.threads <= 1 or ctx.rows == 0) return 1;
    const rows: u64 = @min(ctx.rows, std.math.maxInt(u16));
    var max_parts: u16 = @intCast(rows);
    if (max_parts > ctx.threads) max_parts = ctx.threads;

    const cc = candidateCost(c, ctx, lanes);
    const ws = if (ctx.working_set != 0) ctx.working_set else cc.bytes;

    var best: u16 = 1;
    var best_cost: f64 = std.math.inf(f64);
    var p: u16 = 1;
    while (p <= max_parts) : (p += 1) {
        const cst = cost.opCost(op, cc.bytes, cc.flops, p, ctx.tier, ws, ctx.caps);
        if (cst < best_cost) {
            best_cost = cst;
            best = p;
        }
    }
    return best;
}

/// Deterministic matvec decision: the `MatVecPlan` variant and the degree of
/// row parallelism. Thin wrapper over the generic Selector; the variant rule is
/// the matvec catalog's `applicable` metadata.
pub const MatvecSelection = struct {
    plan: matvec.MatVecPlan,
    parts: u16,
};

pub fn selectMatvec(
    caps: Caps,
    ggml_type: ggml.GGMLType,
    n: u64,
    m: u64,
    threads: u16,
) MatvecSelection {
    // P2: prune the `prefetch = 1` variants unless the op streams from RAM and
    // is memory-bound (caps + roofline). The weight stream is the working set.
    const elems = std.math.mul(u64, n, m) catch 0;
    const bytes = ggml.tensorByteLength(ggml_type, elems) catch 0;
    const flops = std.math.mul(u64, elems, 2) catch std.math.maxInt(u64);
    const sel = select(.matvec, .{
        .caps = caps,
        .type_code = @intFromEnum(ggml_type),
        .m = m,
        .n = n,
        .k = 0,
        .rows = m,
        .threads = threads,
        .allow_prefetch = cost.prefetchWorthwhile(caps, bytes, flops, bytes),
    });
    return .{
        .plan = std.enums.fromInt(matvec.MatVecPlan, sel.variant) orelse .production,
        .parts = sel.parts,
    };
}

/// Deterministic GEMM decision: the `MatMulPlan` variant and the degree of row
/// parallelism. Thin wrapper over the generic Selector; the variant rule is the
/// matmul catalog's `applicable` metadata.
pub const MatmulSelection = struct {
    plan: matmul.MatMulPlan,
    parts: u16,
};

pub fn selectMatmul(
    caps: Caps,
    ggml_type: ggml.GGMLType,
    k: u64,
    n: u64,
    m_rows: u64,
    threads: u16,
) MatmulSelection {
    const sel = select(.matmul, .{
        .caps = caps,
        .type_code = @intFromEnum(ggml_type),
        .m = m_rows,
        .n = n,
        .k = k,
        .rows = if (m_rows == 0) 0 else n,
        .threads = threads,
    });
    return .{
        .plan = std.enums.fromInt(matmul.MatMulPlan, sel.atomic_id) orelse .rows,
        .parts = sel.parts,
    };
}

/// Execution phase of a projection. `decode` is the zero value so an unphased
/// (legacy) dispatch keeps the historical matvec-only plan.
pub const Phase = enum(u8) { decode = 0, prefill = 1 };

/// The Selector's phase-aware projection choice.
pub const Projection = union(enum) {
    matvec: MatvecSelection,
    matmul: MatmulSelection,
};

/// Map one semantic projection op to its kernel by phase: decode (M=1) stays
/// GEMV (memory-bound); prefill (M>1) becomes GEMM (compute-bound). A `decode`
/// phase or a single row always yields `matvec`, so the decode plan is
/// unchanged.
pub fn selectProjection(
    caps: Caps,
    phase: Phase,
    ggml_type: ggml.GGMLType,
    k: u64,
    n: u64,
    m_rows: u64,
    threads: u16,
) Projection {
    if (phase == .prefill and m_rows > 1) {
        return .{ .matmul = selectMatmul(caps, ggml_type, k, n, m_rows, threads) };
    }
    return .{ .matvec = selectMatvec(caps, ggml_type, k, n, threads) };
}

const test_caps = Caps{
    .physical_cores = 8,
    .logical_cpus = 8,
    .simd_lanes = 8,
    .l1_bytes = 32 * 1024,
    .l2_bytes = 256 * 1024,
    .l3_bytes = 8 * 1024 * 1024,
    .dram_bw_gbps = 25.0,
    .base_freq_hz = 3_000_000_000,
};

test "selectMatvec: narrow decode prefers the unrolled variant" {
    const testing = std.testing;
    const sel = selectMatvec(test_caps, .q4_0, 2048, 256, 4);
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0, sel.plan);
}

test "selectMatvec: uncalibrated wide output keeps the catalog order (u4)" {
    const testing = std.testing;
    const wide = selectMatvec(test_caps, .q4_0, 2048, 14336, 8);
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0, wide.plan);
}

test "selectMatvec: q4_0 grid variants are shape- and ISA-free" {
    const testing = std.testing;
    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    // Every grid candidate loses to the w4 tile4 candidate; the lane count must
    // not change the winner (the width is explicit in the variant now).
    for (std.enums.values(matvec.MatVecPlan)) |p| {
        caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    }
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t4un1_p0)] = 1.0;

    caps.simd_lanes = 4;
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t4un1_p0, selectMatvec(caps, .q4_0, 2048, 14336, 8).plan);

    caps.simd_lanes = 16;
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t4un1_p0, selectMatvec(caps, .q4_0, 2048, 14336, 8).plan);
}

test "selectMatvec: the calibrated efficiency ranks the variants" {
    const testing = std.testing;
    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    for (std.enums.values(matvec.MatVecPlan)) |p| {
        caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    }
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0)] = 0.6;
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t2un1_p0)] = 1.0;
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t4un1_p0)] = 0.6;

    const narrow = selectMatvec(caps, .q4_0, 2048, 256, 4);
    const wide = selectMatvec(caps, .q4_0, 2048, 14336, 8);
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t2un1_p0, narrow.plan);
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t2un1_p0, wide.plan);
    try testing.expectEqual(narrow.plan, selectMatvec(caps, .q4_0, 2048, 256, 4).plan);
}

test "selectMatvec: generated q4_0 candidates are selectable" {
    const testing = std.testing;
    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    caps.simd_lanes = 8;
    // Every variant loses to the generated unroll-4 candidate.
    for (std.enums.values(matvec.MatVecPlan)) |p| {
        caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    }
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t1un4_p0)] = 1.0;
    try testing.expectEqual(matvec.MatVecPlan.Q4_0_fp_w4t1un4_p0, selectMatvec(caps, .q4_0, 2048, 14336, 8).plan);
}

test "selectMatvec: P2 prunes the prefetch variants when not streaming" {
    const testing = std.testing;
    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    // Every candidate ties; only the prefetch variant has a calibrated edge.
    for (std.enums.values(matvec.MatVecPlan)) |p| {
        caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    }
    const p1 = matvec.MatVecPlan.Q4_0_fp_w4t1un1_p1;
    const p0 = matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0;
    caps.variant_eff.matvec[@intFromEnum(p1)] = 1.0;

    // A shape whose weight stream fits L3: prefetch is pruned, so p0 wins.
    caps.l3_bytes = 64 * 1024 * 1024;
    try testing.expectEqual(p0, selectMatvec(caps, .q4_0, 2048, 256, 4).plan);

    // A shape whose weight stream exceeds L3 and is memory-bound: p1 is eligible
    // and wins on its calibrated edge.
    caps.l3_bytes = 8 * 1024 * 1024;
    try testing.expectEqual(p1, selectMatvec(caps, .q4_0, 2048, 14336, 8).plan);
}

test "selectMatvec: q8_0 generated variants rank by calibrated efficiency" {
    const testing = std.testing;
    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    caps.simd_lanes = 16;
    // Every variant loses to the chosen generated candidate.
    for (std.enums.values(matvec.MatVecPlan)) |p| {
        caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    }
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q8_0_fp_w4t1un1_p0)] = 0.6;
    caps.variant_eff.matvec[@intFromEnum(matvec.MatVecPlan.Q8_0_fp_w4t1un2_p0)] = 1.0;

    const sel = selectMatvec(caps, .q8_0, 2048, 14336, 8);
    try testing.expectEqual(matvec.MatVecPlan.Q8_0_fp_w4t1un2_p0, sel.plan);
    try testing.expectEqual(matvec.MatVecPlan.Q8_0_fp_w4t1un2_p0, selectMatvec(caps, .q8_0, 2048, 256, 8).plan);
}

test "selectMatvec: unlisted types keep production" {
    const testing = std.testing;
    try testing.expectEqual(matvec.MatVecPlan.production, selectMatvec(test_caps, .q5_0, 2048, 14336, 8).plan);
}

test "selectMatvec: uncalibrated q4_k keeps the catalog order" {
    const testing = std.testing;
    // The q4_k grid subsumes the production kernel (which is not applicable to
    // q4_k), so with no calibrated edge the first grid point wins.
    try testing.expectEqual(matvec.MatVecPlan.Q4_K_fp_w4t1un1_p0, selectMatvec(test_caps, .q4_k, 2048, 14336, 8).plan);
}

test "selectMatvec: q4_k grid variants rank by calibrated efficiency" {
    const testing = std.testing;
    const first = matvec.MatVecPlan.Q4_K_fp_w4t1un1_p0;
    const unrolled = matvec.MatVecPlan.Q4_K_fp_w4t1un8_p0;

    @setEvalBranchQuota(1_000_000);
    var caps = test_caps;
    for (std.enums.values(matvec.MatVecPlan)) |p| caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    caps.variant_eff.matvec[@intFromEnum(unrolled)] = 1.0;
    try testing.expectEqual(unrolled, selectMatvec(caps, .q4_k, 2048, 14336, 8).plan);

    for (std.enums.values(matvec.MatVecPlan)) |p| caps.variant_eff.matvec[@intFromEnum(p)] = 0.5;
    caps.variant_eff.matvec[@intFromEnum(first)] = 1.0;
    try testing.expectEqual(first, selectMatvec(caps, .q4_k, 2048, 14336, 8).plan);
}

test "selectMatvec: parts is within [1, threads] and deterministic" {
    const testing = std.testing;
    var threads: u16 = 1;
    while (threads <= 8) : (threads += 1) {
        const a = selectMatvec(test_caps, .q4_0, 2048, 14336, threads);
        const b = selectMatvec(test_caps, .q4_0, 2048, 14336, threads);
        try testing.expectEqual(a.plan, b.plan);
        try testing.expectEqual(a.parts, b.parts);
        try testing.expect(a.parts >= 1);
        try testing.expect(a.parts <= threads);
    }
}

test "selectMatvec: compute-limited parts reach the physical-core count" {
    const testing = std.testing;
    var caps = test_caps;
    caps.dram_bw_gbps = 1_000_000.0;
    const sel = selectMatvec(caps, .q4_0, 2048, 14336, 16);
    try testing.expectEqual(@as(u16, 8), sel.parts);
}

test "selectMatvec: single physical core saturates at one part" {
    const testing = std.testing;
    var caps = test_caps;
    caps.physical_cores = 1;
    caps.dram_bw_gbps = 1_000_000.0;
    const sel = selectMatvec(caps, .q4_0, 2048, 14336, 8);
    try testing.expectEqual(@as(u16, 1), sel.parts);
}

test "selectMatvec: the same calibrated profile yields the same plan" {
    const testing = std.testing;
    const profile = calibrate.Coeffs{
        .dram_bw_gbps = 31.0,
        .disk_bw_gbps = 3.0,
        .dispatch_latency_ns = 41.0,
    };

    var caps_a = test_caps;
    caps_a.applyCoeffs(profile);
    var caps_b = test_caps;
    caps_b.applyCoeffs(profile);

    const a = selectMatvec(caps_a, .q4_0, 2048, 14336, 8);
    const b = selectMatvec(caps_b, .q4_0, 2048, 14336, 8);
    try testing.expectEqual(a.plan, b.plan);
    try testing.expectEqual(a.parts, b.parts);
}

test "selectMatmul: narrow output keeps the per-row variant" {
    const testing = std.testing;
    const sel = selectMatmul(test_caps, .q4_0, 2048, 4, 8, 8);
    try testing.expectEqual(matmul.MatMulPlan.rows, sel.plan);
}

test "selectMatmul: wide batched output prefers the blocked variant" {
    const testing = std.testing;
    const sel = selectMatmul(test_caps, .q4_0, 2048, 14336, 8, 8);
    try testing.expectEqual(matmul.MatMulPlan.blocked, sel.plan);

    var narrow_isa = test_caps;
    narrow_isa.simd_lanes = 4;
    try testing.expectEqual(matmul.MatMulPlan.rows, selectMatmul(narrow_isa, .q4_0, 2048, 14336, 8, 8).plan);
    try testing.expectEqual(matmul.MatMulPlan.rows, selectMatmul(test_caps, .q4_0, 2048, 14336, 1, 8).plan);
    try testing.expectEqual(matmul.MatMulPlan.rows, selectMatmul(test_caps, .q5_0, 2048, 14336, 8, 8).plan);
}

test "selectMatmul: parts is within [1, threads] and deterministic" {
    const testing = std.testing;
    var threads: u16 = 1;
    while (threads <= 8) : (threads += 1) {
        const a = selectMatmul(test_caps, .q4_0, 2048, 14336, 8, threads);
        const b = selectMatmul(test_caps, .q4_0, 2048, 14336, 8, threads);
        try testing.expectEqual(a.plan, b.plan);
        try testing.expectEqual(a.parts, b.parts);
        try testing.expect(a.parts >= 1);
        try testing.expect(a.parts <= threads);
    }
}

test "selectProjection: decode and M=1 map to matvec, prefill M>1 to matmul" {
    const testing = std.testing;

    const decode = selectProjection(test_caps, .decode, .q4_0, 2048, 14336, 1, 8);
    try testing.expect(decode == .matvec);
    try testing.expectEqual(selectMatvec(test_caps, .q4_0, 2048, 14336, 8).plan, decode.matvec.plan);

    const prefill_one = selectProjection(test_caps, .prefill, .q4_0, 2048, 14336, 1, 8);
    try testing.expect(prefill_one == .matvec);

    const prefill = selectProjection(test_caps, .prefill, .q4_0, 2048, 14336, 8, 8);
    try testing.expect(prefill == .matmul);
    try testing.expectEqual(matmul.MatMulPlan.blocked, prefill.matmul.plan);

    const again = selectProjection(test_caps, .prefill, .q4_0, 2048, 14336, 8, 8);
    try testing.expectEqual(prefill.matmul.plan, again.matmul.plan);
    try testing.expectEqual(prefill.matmul.parts, again.matmul.parts);
}

test "select: matvec filters candidates by applicable and never picks recipes" {
    const testing = std.testing;

    const narrow = select(.matvec, .{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q4_0),
        .m = 256,
        .n = 2048,
        .k = 0,
        .rows = 256,
        .threads = 4,
    });
    try testing.expectEqual(@intFromEnum(matvec.AtomicId.mat_vec), narrow.atomic_id);
    try testing.expectEqual(@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0), narrow.variant);

    const wide = select(.matvec, .{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q4_0),
        .m = 14336,
        .n = 2048,
        .k = 0,
        .rows = 14336,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(matvec.MatVecPlan.Q4_0_fp_w4t1un1_p0), wide.variant);

    const other = select(.matvec, .{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q5_0),
        .m = 14336,
        .n = 2048,
        .k = 0,
        .rows = 14336,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(matvec.MatVecPlan.production), other.variant);
}

test "select: argmin by cost prefers the blocked GEMM only when it applies" {
    const testing = std.testing;

    const wide = select(.matmul, .{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q4_0),
        .m = 8,
        .n = 14336,
        .k = 2048,
        .rows = 14336,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(matmul.AtomicId.blocked), wide.atomic_id);

    const narrow = select(.matmul, .{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q4_0),
        .m = 8,
        .n = 4,
        .k = 2048,
        .rows = 4,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(matmul.AtomicId.rows), narrow.atomic_id);
}

test "select: attention ranks the head variants by cost" {
    const testing = std.testing;
    const at = @import("../kernels/transformer/attention.zig");

    const long = select(.attention, .{
        .caps = test_caps,
        .type_code = 0,
        .m = 8192,
        .n = 512,
        .k = 0,
        .rows = 0,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(at.AtomicId.fused_flash_head), long.atomic_id);

    var compute_bound = test_caps;
    compute_bound.dram_bw_gbps = 1e12;
    compute_bound.dispatch_latency_ns = 0.0001;
    const split = select(.attention, .{
        .caps = compute_bound,
        .type_code = 0,
        .m = 100000,
        .n = 512,
        .k = 0,
        .rows = 0,
        .threads = 8,
    });
    try testing.expectEqual(@intFromEnum(at.AtomicId.two_pass_head), split.atomic_id);
}

test "select: is a pure function of the declaration and caps" {
    const testing = std.testing;
    const ctx = SelectCtx{
        .caps = test_caps,
        .type_code = @intFromEnum(ggml.GGMLType.q4_0),
        .m = 14336,
        .n = 2048,
        .k = 0,
        .rows = 14336,
        .threads = 8,
    };
    const a = select(.matvec, ctx);
    const b = select(.matvec, ctx);
    try testing.expectEqual(a.atomic_id, b.atomic_id);
    try testing.expectEqual(a.variant, b.variant);
    try testing.expectEqual(a.parts, b.parts);
}

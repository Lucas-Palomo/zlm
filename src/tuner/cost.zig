const std = @import("std");
const api = @import("../kernels/api.zig");
const caps_mod = @import("caps.zig");
const calibrate = @import("calibrate.zig");
const Caps = caps_mod.Caps;

/// Memory tiers from slowest to fastest. `disk` is the streaming fallback for
/// weights that are not resident; caches sit above DRAM.
pub const Tier = enum { disk, ram, l3, l2, l1 };

/// Cache bandwidth multipliers relative to DRAM: L3 4x, L2 8x, L1 16x.
pub const l3_mult: f64 = 4.0;
pub const l2_mult: f64 = 8.0;
pub const l1_mult: f64 = 16.0;

/// Serial fraction of the Amdahl-like compute parallel-efficiency curve.
pub const serial_fraction: f64 = 0.05;

/// Objective weights: generation dominates TTFT (w_gen > w_ttft).
pub const w_ttft: f64 = 0.3;
pub const w_gen: f64 = 0.7;

/// Bandwidth of a tier in bytes/ns (numerically equal to GB/s). DRAM and disk
/// use the (possibly calibrated) capabilities; caches scale DRAM.
pub fn tierBw(caps: Caps, tier: Tier) f64 {
    const dram = if (caps.dram_bw_gbps > 0) caps.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
    const disk = if (caps.disk_bw_gbps > 0) caps.disk_bw_gbps else calibrate.fallback_disk_bw_gbps;
    return switch (tier) {
        .disk => disk,
        .ram => dram,
        .l3 => dram * l3_mult,
        .l2 => dram * l2_mult,
        .l1 => dram * l1_mult,
    };
}

/// The highest-bandwidth cache tier the working set fits, else DRAM. Used to
/// decide how a candidate's bytes are actually moved.
pub fn fitTier(working_set: u64, caps: Caps) Tier {
    if (working_set <= caps.l1_bytes) return .l1;
    if (working_set <= caps.l2_bytes) return .l2;
    if (working_set <= caps.l3_bytes) return .l3;
    return .ram;
}

/// Cache-aware effective bytes: when the working set fits a cache, the logical
/// bytes are discounted by that tier's bandwidth advantage over DRAM, so
/// candidates that move the same logical bytes are differentiated. Returns a
/// DRAM-equivalent byte count (<= logical_bytes); divide by the streaming
/// tier's bandwidth to get time.
pub fn bytesEffective(logical_bytes: u64, working_set: u64, caps: Caps) f64 {
    const logical: f64 = @floatFromInt(logical_bytes);
    const ram = tierBw(caps, .ram);
    const fit = tierBw(caps, fitTier(working_set, caps));
    if (fit <= 0) return logical;
    return logical * ram / fit;
}

/// Step 1 — logical cost of one op: the hardware-independent work, produced by
/// the AST static analysis (per declaration, then per candidate). `bytes` is the
/// quantized weight/activation traffic, `flops` the arithmetic.
pub const LogicalCost = struct {
    bytes: u64,
    flops: u64,
    complexity: api.Complexity = .linear,
};

/// Whether the roofline is bandwidth- or flops-dominated (Step 2's label).
pub const Regime = enum { memory, compute };

/// Step 2 — Roofline: the machine's ceiling for a logical cost, plus the regime.
/// `bytes_ns` uses the cache-aware effective bytes at the streaming tier;
/// `flops_ns` is the ideal (peak, no parallelism) compute time; `dispatch_ns` is
/// the fixed per-dispatch latency. `eff_bytes` and `stream_tier` are retained so
/// Step 3 can re-price the bytes term at the measured thread scaling.
pub const Roofline = struct {
    bytes_ns: f64,
    flops_ns: f64,
    dispatch_ns: f64,
    regime: Regime,
    eff_bytes: f64 = 0,
    stream_tier: Tier = .ram,
};

pub fn roofline(lc: LogicalCost, tier: Tier, working_set: u64, caps: Caps) Roofline {
    const bandwidth = tierBw(caps, tier);
    const be = bytesEffective(lc.bytes, working_set, caps);
    const bytes_ns = if (bandwidth > 0) be / bandwidth else 0;

    const peak_per_ns = caps.peakFlops() / 1e9;
    const flops_ns = if (peak_per_ns > 0)
        @as(f64, @floatFromInt(lc.flops)) / peak_per_ns
    else
        0;

    const dispatch = if (caps.dispatch_latency_ns > 0)
        caps.dispatch_latency_ns
    else
        calibrate.fallback_dispatch_latency_ns;

    return .{
        .bytes_ns = bytes_ns,
        .flops_ns = flops_ns,
        .dispatch_ns = dispatch,
        .regime = if (bytes_ns >= flops_ns) .memory else .compute,
        .eff_bytes = be,
        .stream_tier = tier,
    };
}

/// Step 3 — scaling: turn the Roofline ceiling into the effective time with
/// `parts` threads. The compute term is divided by the parallel efficiency
/// (Amdahl-like). A memory-regime op that streams from RAM is re-priced with the
/// op-aware measured bandwidth curve `bw_eff(n) = min(ceiling, per_thread * n)`
/// (`bwAtOp`): compute-bound ops use the matvec's own per-thread figure, so the
/// bytes term keeps falling until the aggregate ceiling instead of saturating at
/// the streaming knee. Disk streaming keeps the disk-bandwidth term.
pub fn scaling(r: Roofline, op: api.Op, parts: u16, caps: Caps) f64 {
    const pe = parEff(op, parts, caps);
    const flops_ns = if (pe > 0) r.flops_ns / pe else r.flops_ns;

    var bytes_ns = r.bytes_ns;
    if (r.regime == .memory and r.stream_tier == .ram) {
        const bw = caps.bwAtOp(op, parts);
        if (bw > 0) bytes_ns = r.eff_bytes / bw;
    }

    return @max(bytes_ns, flops_ns) + r.dispatch_ns;
}

/// Ops whose roofline is flops-dominated rather than bandwidth-dominated.
fn isComputeBound(op: api.Op) bool {
    return switch (op) {
        .matvec, .matmul, .attention => true,
        else => false,
    };
}

/// Bounded, monotone parallel efficiency in (0, 1] for the compute term.
/// Compute-bound ops follow an Amdahl-like curve that reaches 1 at the physical
/// core count and stays there. Non-compute ops return 1: their memory term is
/// governed by the measured bandwidth curve in `scaling`, not an Amdahl guess.
pub fn parEff(op: api.Op, parts: u16, caps: Caps) f64 {
    if (!isComputeBound(op)) return 1.0;
    const n: f64 = @floatFromInt(@max(parts, 1));
    const p: f64 = @floatFromInt(@max(caps.physical_cores, 1));
    const active = @min(n, p);
    if (active <= 0) return 0;
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) * p / active);
}

/// Analytical roofline cost (ns) of one dispatch:
///   max(bytes_effective / bw(tier), flops / (peak_flops * par_eff))
///   + dispatch_latency
/// `tier` is the streaming tier (`.ram` for RAM-resident work, `.disk` for
/// non-resident weights); `working_set` drives the cache-aware bytes.
pub fn opCost(
    op: api.Op,
    bytes: u64,
    flops: u64,
    parts: u16,
    tier: Tier,
    working_set: u64,
    caps: Caps,
) f64 {
    const r = roofline(.{ .bytes = bytes, .flops = flops }, tier, working_set, caps);
    return scaling(r, op, parts, caps);
}

/// P2 — the prefetch prune heuristic. A software prefetch (`prefetch = 1`) pays
/// only when the op **streams from RAM** — the working set is past the last-level
/// cache, so the hardware prefetcher cannot cover the sequential access from
/// cache — **and is memory-bound** (the roofline regime is memory, not flops).
/// Otherwise `prefetch = 1` cannot win and the plan prunes it *before* the
/// benchmark. Device- and shape-derived only; the tuner records the outcome in
/// the plan and the runtime re-decides nothing.
pub fn prefetchWorthwhile(caps: Caps, bytes: u64, flops: u64, working_set: u64) bool {
    const ws = if (working_set != 0) working_set else bytes;
    if (caps.l3_bytes > 0 and ws <= caps.l3_bytes) return false;
    const r = roofline(.{ .bytes = bytes, .flops = flops }, .ram, ws, caps);
    return r.regime == .memory;
}

/// How many dispatch steps ahead to start an async page-in so the disk read
/// finishes before the consumer: the time to read a whole streamed weight range
/// from disk divided by the per-step compute time. Unlike a cache-line model,
/// this sizes the lookahead for the real prefetch unit (a weight range), so a
/// large range on a slow disk looks far ahead. `read_bytes` is the
/// representative streamed range; `per_step_ns` is the compute cost of one
/// dispatch step. Clamped to `[1, max]`; a non-positive step time or an empty
/// range returns `1`. Derived only from `caps` and the two inputs.
pub fn lookaheadForRead(caps: Caps, read_bytes: u64, per_step_ns: f64, max: u32) u32 {
    const lo: u32 = 1;
    const hi: u32 = @max(lo, max);
    if (!(per_step_ns > 0) or read_bytes == 0) return lo;
    const disk = tierBw(caps, .disk);
    if (disk <= 0) return lo;
    const read_ns = @as(f64, @floatFromInt(read_bytes)) / disk;
    const raw = read_ns / per_step_ns;
    if (!std.math.isFinite(raw) or raw < @as(f64, @floatFromInt(lo))) return lo;
    if (raw >= @as(f64, @floatFromInt(hi))) return hi;
    return @intFromFloat(raw);
}

/// The compute cost (ns) of one dispatch step that consumes a `bytes` weight:
/// a memory-bound `matvec` of that range at the RAM tier with `parts` threads.
/// The prefetch lookahead divides a range's disk read time by this.
pub fn matvecStepNs(caps: Caps, bytes: u64, parts: u16) f64 {
    return opCost(.matvec, bytes, 0, parts, .ram, bytes, caps);
}

/// One op's cost inputs for the plan objective.
pub const OpCostInput = struct {
    op: api.Op,
    bytes: u64,
    flops: u64,
    parts: u16,
    tier: Tier,
    working_set: u64,
};

/// Minimal plan input: the TTFT and generation op lists plus the streaming and
/// startup terms paid at disk bandwidth.
pub const PlanInput = struct {
    ttft_ops: []const OpCostInput = &.{},
    gen_ops: []const OpCostInput = &.{},
    streamed_live_bytes: u64 = 0,
    startup_bytes: u64 = 0,
};

/// Plan objective skeleton: `w_ttft * cost_ttft + w_gen * cost_gen`, with the
/// startup streamed in during TTFT and the non-resident live bytes streamed
/// during generation. Feasibility (resident <= RAM, parts + residency workers
/// <= threads) is enforced by the caller (Stage 3).
pub fn planScore(plan: PlanInput, caps: Caps) f64 {
    var ttft: f64 = 0;
    for (plan.ttft_ops) |o| {
        ttft += opCost(o.op, o.bytes, o.flops, o.parts, o.tier, o.working_set, caps);
    }
    var gen: f64 = 0;
    for (plan.gen_ops) |o| {
        gen += opCost(o.op, o.bytes, o.flops, o.parts, o.tier, o.working_set, caps);
    }

    const disk = tierBw(caps, .disk);
    ttft += @as(f64, @floatFromInt(plan.startup_bytes)) / disk;
    gen += @as(f64, @floatFromInt(plan.streamed_live_bytes)) / disk;

    return w_ttft * ttft + w_gen * gen;
}

const test_caps = Caps{
    .physical_cores = 8,
    .logical_cpus = 8,
    .simd_lanes = 8,
    .l1_bytes = 32 * 1024,
    .l2_bytes = 256 * 1024,
    .l3_bytes = 8 * 1024 * 1024,
    .dram_bw_gbps = 25.0,
    .dram_bw_1t_gbps = 10.0,
    .base_freq_hz = 3_000_000_000,
};

test "tierBw orders cache above ram above disk" {
    const testing = std.testing;
    try testing.expect(tierBw(test_caps, .l1) > tierBw(test_caps, .l2));
    try testing.expect(tierBw(test_caps, .l2) > tierBw(test_caps, .l3));
    try testing.expect(tierBw(test_caps, .l3) > tierBw(test_caps, .ram));
    try testing.expect(tierBw(test_caps, .ram) > tierBw(test_caps, .disk));
}

test "bytesEffective discounts a working set that fits L2" {
    const testing = std.testing;
    const logical: u64 = 1_000_000;
    const l2_ws: u64 = 64 * 1024;
    const dram_ws: u64 = 64 * 1024 * 1024;
    const be_l2 = bytesEffective(logical, l2_ws, test_caps);
    const be_dram = bytesEffective(logical, dram_ws, test_caps);
    try testing.expect(be_l2 < be_dram);
    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(logical)), be_dram, 1e-6);
}

test "roofline labels the regime and scaling divides the compute term" {
    const testing = std.testing;
    const ws: u64 = 64 * 1024 * 1024;
    const mem = roofline(.{ .bytes = 1_000_000, .flops = 1 }, .ram, ws, test_caps);
    try testing.expectEqual(Regime.memory, mem.regime);
    const comp = roofline(.{ .bytes = 1024, .flops = 1_000_000_000 }, .ram, ws, test_caps);
    try testing.expectEqual(Regime.compute, comp.regime);

    const s1 = scaling(comp, .matmul, 1, test_caps);
    const s8 = scaling(comp, .matmul, 8, test_caps);
    try testing.expect(s1 > s8);
    try testing.expectApproxEqAbs(scaling(comp, .matmul, 8, test_caps), scaling(comp, .matmul, 16, test_caps), 1e-9);
}

test "bwAt grows with parts and saturates at the aggregate ceiling" {
    const testing = std.testing;
    try testing.expectApproxEqAbs(@as(f64, 10.0), test_caps.bwAt(1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), test_caps.bwAt(2), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 25.0), test_caps.bwAt(4), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 25.0), test_caps.bwAt(8), 1e-9);
    try testing.expect(test_caps.bwAt(3) >= test_caps.bwAt(2));
    try testing.expect(test_caps.bwAt(8) <= test_caps.dram_bw_gbps);
    try testing.expectApproxEqAbs(test_caps.bwAt(0), test_caps.bwAt(1), 1e-9);
}

test "memory-bound scaling falls with parts then flattens at the ceiling" {
    const testing = std.testing;
    const ws: u64 = 64 * 1024 * 1024;
    const lc = LogicalCost{ .bytes = 1_000_000_000, .flops = 0 };
    const r = roofline(lc, .ram, ws, test_caps);
    try testing.expectEqual(Regime.memory, r.regime);

    const s1 = scaling(r, .elementwise, 1, test_caps);
    const s2 = scaling(r, .elementwise, 2, test_caps);
    const s4 = scaling(r, .elementwise, 4, test_caps);
    const s8 = scaling(r, .elementwise, 8, test_caps);
    try testing.expect(s1 > s2);
    try testing.expect(s2 > s4);
    try testing.expectApproxEqAbs(s4, s8, 1e-9);
}

test "memory scaling is deterministic" {
    const testing = std.testing;
    const ws: u64 = 64 * 1024 * 1024;
    const r = roofline(.{ .bytes = 3_000_000, .flops = 0 }, .ram, ws, test_caps);
    try testing.expectEqual(scaling(r, .elementwise, 5, test_caps), scaling(r, .elementwise, 5, test_caps));
    try testing.expectEqual(
        opCost(.elementwise, 3_000_000, 0, 5, .ram, ws, test_caps),
        opCost(.elementwise, 3_000_000, 0, 5, .ram, ws, test_caps),
    );
}

test "op-aware memory term saturates later for a compute-bound matvec" {
    const testing = std.testing;
    var caps = test_caps;
    caps.dram_bw_gbps = 60.0;
    caps.matvec_bw_1t_gbps = 4.0;
    const ws: u64 = 64 * 1024 * 1024;
    const r = roofline(.{ .bytes = 1_000_000_000, .flops = 0 }, .ram, ws, caps);
    try testing.expectEqual(Regime.memory, r.regime);

    const s1 = scaling(r, .matvec, 1, caps);
    const s2 = scaling(r, .matvec, 2, caps);
    const s4 = scaling(r, .matvec, 4, caps);
    const s8 = scaling(r, .matvec, 8, caps);
    try testing.expect(s1 > s2);
    try testing.expect(s2 > s4);
    try testing.expect(s4 > s8);

    const stream = scaling(r, .elementwise, 8, caps);
    try testing.expect(stream < s8);
    try testing.expectEqual(scaling(r, .matvec, 8, caps), scaling(r, .matvec, 8, caps));
}

test "opCost decreases with parts then flattens" {
    const testing = std.testing;
    const c1 = opCost(.matvec, 1024, 1_000_000_000, 1, .ram, 64 * 1024 * 1024, test_caps);
    const c8 = opCost(.matvec, 1024, 1_000_000_000, 8, .ram, 64 * 1024 * 1024, test_caps);
    const c16 = opCost(.matvec, 1024, 1_000_000_000, 16, .ram, 64 * 1024 * 1024, test_caps);
    try testing.expect(c1 > c8);
    try testing.expectApproxEqAbs(c8, c16, 1e-9);
}

test "memory-bound op with more bytes costs more" {
    const testing = std.testing;
    const small = opCost(.elementwise, 1_000_000, 0, 1, .ram, 64 * 1024 * 1024, test_caps);
    const large = opCost(.elementwise, 2_000_000, 0, 1, .ram, 64 * 1024 * 1024, test_caps);
    try testing.expect(large > small);
}

test "compute-bound op with more flops costs more" {
    const testing = std.testing;
    const small = opCost(.matvec, 1024, 1_000_000_000, 8, .ram, 64 * 1024 * 1024, test_caps);
    const large = opCost(.matvec, 1024, 2_000_000_000, 8, .ram, 64 * 1024 * 1024, test_caps);
    try testing.expect(large > small);
}

test "matmul opCost grows with flops and falls with parts (compute-bound)" {
    const testing = std.testing;
    const ws: u64 = 64 * 1024 * 1024;
    const small = opCost(.matmul, 4096, 1_000_000_000, 8, .ram, ws, test_caps);
    const large = opCost(.matmul, 4096, 4_000_000_000, 8, .ram, ws, test_caps);
    try testing.expect(large > small);

    const p1 = opCost(.matmul, 4096, 4_000_000_000, 1, .ram, ws, test_caps);
    const p4 = opCost(.matmul, 4096, 4_000_000_000, 4, .ram, ws, test_caps);
    const p8 = opCost(.matmul, 4096, 4_000_000_000, 8, .ram, ws, test_caps);
    const p16 = opCost(.matmul, 4096, 4_000_000_000, 16, .ram, ws, test_caps);
    try testing.expect(p1 > p4);
    try testing.expect(p4 > p8);
    try testing.expectApproxEqAbs(p8, p16, 1e-9);
}

test "planScore weights generation more than TTFT" {
    const testing = std.testing;
    const op = OpCostInput{
        .op = .elementwise,
        .bytes = 1_000_000,
        .flops = 0,
        .parts = 1,
        .tier = .ram,
        .working_set = 64 * 1024 * 1024,
    };
    const ttft_ops = [_]OpCostInput{op};
    const gen_ops = [_]OpCostInput{op};
    const ttft_only = planScore(.{ .ttft_ops = &ttft_ops }, test_caps);
    const gen_only = planScore(.{ .gen_ops = &gen_ops }, test_caps);
    try testing.expect(w_gen > w_ttft);
    try testing.expect(gen_only > ttft_only);
}

test "lookaheadForRead sizes by the range read time and clamps" {
    const testing = std.testing;
    var caps = test_caps;
    caps.disk_bw_gbps = 2.0;
    const step: f64 = 1e6;

    // A 4 MB range at 2 GB/s = ~2 ms -> 2 steps; a bigger range looks further.
    const small = lookaheadForRead(caps, 4 << 20, step, 64);
    try testing.expectEqual(@as(u32, 2), small);
    try testing.expect(lookaheadForRead(caps, 40 << 20, step, 64) > small);

    // A slower disk looks further ahead for the same range.
    var slow = caps;
    slow.disk_bw_gbps = 0.5;
    try testing.expect(lookaheadForRead(slow, 4 << 20, step, 64) > small);

    // A faster step (more threads) looks less far ahead.
    try testing.expect(lookaheadForRead(caps, 4 << 20, 4 * step, 64) < small);

    // Clamps to [1, max] and handles degenerate inputs.
    try testing.expectEqual(@as(u32, 1), lookaheadForRead(caps, 1, 1e12, 64));
    try testing.expectEqual(@as(u32, 64), lookaheadForRead(slow, 1 << 40, 1.0, 64));
    try testing.expectEqual(@as(u32, 1), lookaheadForRead(caps, 4 << 20, 0, 64));
    try testing.expectEqual(@as(u32, 1), lookaheadForRead(caps, 0, step, 64));
    try testing.expectEqual(@as(u32, 1), lookaheadForRead(caps, 4 << 20, step, 0));
    try testing.expectEqual(small, lookaheadForRead(caps, 4 << 20, step, 64));
}

test "matvecStepNs is a memory-bound matvec of the range" {
    const testing = std.testing;
    var caps = test_caps;
    caps.dram_bw_gbps = 20.0;
    caps.dram_bw_1t_gbps = 10.0;
    caps.matvec_bw_1t_gbps = 10.0;
    const ns = matvecStepNs(caps, 40 << 20, 8);
    try testing.expect(ns > 0);
    // The step is faster with more threads (until the aggregate ceiling).
    try testing.expect(matvecStepNs(caps, 40 << 20, 8) < matvecStepNs(caps, 40 << 20, 1));
}

test "prefetchWorthwhile: only for a RAM-streaming, memory-bound op" {
    const testing = std.testing;
    var caps = test_caps;
    caps.l3_bytes = 8 * 1024 * 1024;
    // A large, memory-bound stream (past L3): prefetch pays.
    try testing.expect(prefetchWorthwhile(caps, 64 * 1024 * 1024, 1, 64 * 1024 * 1024));
    // Fits the last-level cache: the hardware prefetcher covers it -> pruned.
    try testing.expect(!prefetchWorthwhile(caps, 1 * 1024 * 1024, 1, 1 * 1024 * 1024));
    // Compute-bound: prefetch cannot win -> pruned.
    try testing.expect(!prefetchWorthwhile(caps, 1024, 1_000_000_000, 64 * 1024 * 1024));
    try testing.expectEqual(
        prefetchWorthwhile(caps, 64 * 1024 * 1024, 1, 64 * 1024 * 1024),
        prefetchWorthwhile(caps, 64 * 1024 * 1024, 1, 64 * 1024 * 1024),
    );
}

test "applyCoeffs overrides dram, disk and dispatch" {
    const testing = std.testing;
    var caps = test_caps;
    const ram_before = tierBw(caps, .ram);
    const disk_before = tierBw(caps, .disk);

    caps.applyCoeffs(.{
        .dram_bw_gbps = ram_before * 2.0,
        .matvec_bw_1t_gbps = 5.0,
        .disk_bw_gbps = disk_before * 3.0,
        .dispatch_latency_ns = 7.0,
    });

    try testing.expectApproxEqAbs(ram_before * 2.0, tierBw(caps, .ram), 1e-9);
    try testing.expectApproxEqAbs(disk_before * 3.0, tierBw(caps, .disk), 1e-9);
    try testing.expect(tierBw(caps, .ram) > ram_before);
    try testing.expect(tierBw(caps, .disk) > disk_before);
    try testing.expectApproxEqAbs(@as(f64, 5.0), caps.matvec_bw_1t_gbps, 1e-9);

    const low_dispatch = test_caps;
    var fast_dispatch = test_caps;
    fast_dispatch.dispatch_latency_ns = 0.5;
    const a = opCost(.elementwise, 1024, 0, 1, .ram, 64 * 1024 * 1024, low_dispatch);
    const b = opCost(.elementwise, 1024, 0, 1, .ram, 64 * 1024 * 1024, fast_dispatch);
    try testing.expect(b < a);
}

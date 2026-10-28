const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const cpu = @import("../utils/cpu.zig");
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const binary_writer = @import("../utils/binary_writer.zig");
const appendU16 = binary_writer.appendU16;
const appendU32 = binary_writer.appendU32;
const appendU64 = binary_writer.appendU64;
const caps_mod = @import("caps.zig");
const Caps = caps_mod.Caps;
const device = @import("../gpu/root.zig");
const GpuVariant = device.GpuVariant;
const matvec_native = @import("../kernels/linear/matvec/kernels/quantized.zig");
const matvec_float = @import("../kernels/linear/matvec/kernels/float.zig");
const matvec_bfloat = @import("../kernels/linear/matvec/kernels/bfloat.zig");
const matvec_single = @import("../kernels/linear/matvec/form/single.zig");
const matvec = @import("../kernels/linear/matvec/root.zig");
const simd = @import("../kernels/simd.zig");
const MatVecPlan = @import("../kernels/linear/matvec/types.zig").MatVecPlan;
const quant = @import("../kernels/linear/quant/root.zig");
const TensorView = @import("../data/tensor_view.zig").TensorView;

/// Device calibration profile magic and format version. Supersedes the
/// per-image `ZAT1` measurement cache; older files are ignored by the magic
/// check.
pub const MAGIC = "ZAT2";
pub const VERSION: u32 = 27;

/// Slots in a family's per-variant efficiency table (>= the variant count).
/// The fixed size keeps the `.zat` layout independent of the build's ISA: an
/// AVX-512 build generates more plans than an AVX2 build. The comptime check
/// below fails if the catalog outgrows the table, instead of silently
/// truncating a target-dependent index.
pub const max_variants: usize = 1024;

comptime {
    if (matvec.types.plan_count > max_variants) {
        @compileError("calibrate.max_variants is smaller than the MatVecPlan catalog; bump it and the .zat VERSION");
    }
}

/// Device-calibrated per-variant efficiency in `(0, 1]`: the fraction of the
/// best measured variant's throughput a variant reaches. `1.0` is the best
/// variant (or an uncalibrated entry). Indexed by the family's variant id.
pub const VariantEfficiencies = struct {
    matvec: [max_variants]f64 = [_]f64{1.0} ** max_variants,
    matmul: [max_variants]f64 = [_]f64{1.0} ** max_variants,
};

/// Conservative fallbacks used when a native measurement fails.
pub const fallback_dram_bw_gbps: f64 = 25.0;
pub const fallback_disk_bw_gbps: f64 = 2.0;
pub const fallback_dispatch_latency_ns: f64 = 50.0;
/// Fallback `madvise(WILLNEED)` cost model: `min(cap, base + per_page*pages)`.
pub const fallback_madvise_ns_base: f64 = 500.0;
pub const fallback_madvise_ns_per_page: f64 = 50.0;
pub const fallback_madvise_ns_cap: f64 = 100_000.0;

/// Measured co-processor (GPU) throughput, filled at `tune --gpu` time by the
/// device probe. Rates are effective end-to-end bytes/ns (== GB/s) over the raw
/// weight bytes of a representative op, so `length_bytes / rate` is a duration.
/// Every field defaults to `0` = "unmeasured" (a CPU-only tune records nothing
/// here); the offload policy then falls back to its conservative constants.
pub const GpuCoeffs = struct {
    /// Resident decode GEMV: the weight lives in VRAM, so the rate is the
    /// device's own processing over its VRAM read.
    gemv_gbps: f64 = 0,
    /// Streamed decode GEMV: includes the host->device weight transfer and the
    /// dispatch sync, i.e. the real per-token cost of a non-resident offload.
    gemv_streamed_gbps: f64 = 0,
    /// Streamed batched GEMM at the probe block: the weight transfer is
    /// amortized over the prompt rows, so this is the effective weight rate the
    /// prefill offload actually sees.
    matmul_gbps: f64 = 0,
    /// The heterogeneous **split's efficiency**: the measured split rate as a
    /// fraction of the GPU-only rate. `1.0` = the ideal (the CPU and the GPU
    /// bands add); `< 1` = the two memories contend, so a split *loses* to the
    /// GPU alone and the oracle keeps the op whole on the GPU. `0` =
    /// unmeasured (treated as `1.0`, the ideal).
    split_efficiency: f64 = 0,
    /// Chosen GEMV shader variant per base quant, measured by the probe. The
    /// tuner writes the winner onto every dispatch of that quant (`plain` =
    /// unmeasured / base shader). The runtime re-decides nothing.
    q4_0_variant: GpuVariant = .plain,
    q4_1_variant: GpuVariant = .plain,
    q8_0_variant: GpuVariant = .plain,
    q6_k_variant: GpuVariant = .plain,
    q4_k_variant: GpuVariant = .plain,
    f32_variant: GpuVariant = .plain,
    f16_variant: GpuVariant = .plain,
    bf16_variant: GpuVariant = .plain,
    /// Chosen variant of the fused `gate_up` shader, per quant (q4_0/q4_1/q8_0).
    gate_up_variant: GpuVariant = .plain,
    gate_up_q4_1_variant: GpuVariant = .plain,
    gate_up_q8_0_variant: GpuVariant = .plain,
    /// Measured resident GEMV rates of the unrolled variants (GB/s); kept for
    /// diagnostics/inspection. The decision lives in the `*_variant` fields.
    q4_0_u4_gbps: f64 = 0,
    /// Measured resident GEMV rate of the subgroup/wave reduction variant.
    q4_0_sub_gbps: f64 = 0,
    /// Measured resident GEMV rates of the inner k-block unroll variants.
    q4_0_un2_gbps: f64 = 0,
    q4_0_un4_gbps: f64 = 0,
    /// Measured resident GEMV rates of the workgroup-size variants.
    q4_0_w32_gbps: f64 = 0,
    q4_0_w128_gbps: f64 = 0,
    q4_k_gbps: f64 = 0,
    q4_k_u4_gbps: f64 = 0,
    /// Measured resident fused `gate_up` rates (GB/s over both weights' bytes):
    /// the plain shader and the unrolled variant. The choice lives in
    /// `gate_up_variant`.
    gate_up_gbps: f64 = 0,
    gate_up_u4_gbps: f64 = 0,
};

/// Measured cost-model coefficients for one device. All values are min-of-N
/// estimates (least contaminated by drift), never means. `dram_bw_gbps` is the
/// aggregate ceiling (best pass over a small thread sweep); `dram_bw_1t_gbps`
/// is the measured single-thread streaming bandwidth; `matvec_bw_1t_gbps` is
/// the matvec's own single-thread achieved bandwidth (compute-bound, below the
/// streaming figure) used for the compute-bound ops' memory term.
pub const Coeffs = struct {
    dram_bw_gbps: f64 = fallback_dram_bw_gbps,
    dram_bw_1t_gbps: f64 = fallback_dram_bw_gbps,
    matvec_bw_1t_gbps: f64 = fallback_dram_bw_gbps,
    disk_bw_gbps: f64 = fallback_disk_bw_gbps,
    dispatch_latency_ns: f64 = fallback_dispatch_latency_ns,
    /// `madvise(WILLNEED)` cost model: `min(cap, base + per_page*pages)`.
    madvise_ns_base: f64 = fallback_madvise_ns_base,
    madvise_ns_per_page: f64 = fallback_madvise_ns_per_page,
    madvise_ns_cap: f64 = fallback_madvise_ns_cap,
    /// Fastest measured SIMD width (f32 lanes: 4/8/16) for the compute kernels;
    /// `0` = unmeasured (the runtime falls back to the build's native width).
    simd_width: u8 = 0,
    variant_eff: VariantEfficiencies = .{},
    /// Measured co-processor throughput (all `0` when unmeasured).
    gpu: GpuCoeffs = .{},
};

/// The per-device calibration artifact: capabilities + refined coefficients.
pub const Profile = struct {
    caps: Caps,
    coeffs: Coeffs,
};

/// Timed samples per coefficient; the minimum (least contaminated) is kept.
const samples: usize = 5;
/// DRAM stream buffer; larger than any cache so the pass is DRAM-bound, and
/// large enough that each sweep thread's slice amortizes spawn overhead. Debug
/// builds use a smaller/faster probe: tests only need positivity, and the
/// production `tune` runs in a release build.
const dram_bytes: usize = if (builtin.mode == .Debug) 32 * 1024 * 1024 else 256 * 1024 * 1024;
/// DRAM passes per point. More than the other coefficients because the memory
/// term drives `parts`, so a stable min matters for plan reproducibility.
const dram_samples: usize = if (builtin.mode == .Debug) 3 else 31;
/// Cap on the disk sample so a huge image does not dominate calibration.
const disk_bytes_max: usize = 128 * 1024 * 1024;
/// Repetitions per size when probing `madvise(WILLNEED)` cost.
const madvise_samples: usize = if (builtin.mode == .Debug) 8 else 50;
/// Serial dependent calls used to estimate per-dispatch latency.
const dispatch_iters: usize = 200_000;
/// Largest thread count the DRAM scaling sweep will exercise.
const max_sweep_threads: usize = 16;

/// Representative matvec probe for the per-variant efficiency: `n` input
/// features and `m` output rows, sized so the quantized weight stream exceeds
/// any cache (DRAM-bound). Debug builds use a small, fast probe; tests only
/// need an ordering.
const variant_probe_n: usize = if (builtin.mode == .Debug) 1024 else 4096;
const variant_probe_m: usize = if (builtin.mode == .Debug) 512 else 16384;
const variant_samples: usize = if (builtin.mode == .Debug) 3 else 5;

/// Synthetic q4_0 weight stream for the matvec per-thread bandwidth probe:
/// `n` input features and `m` output rows, sized so the weight stream exceeds
/// any cache (DRAM-scale). Debug builds use a small, fast probe; tests only
/// need positivity.
const matvec_probe_n: usize = if (builtin.mode == .Debug) 1024 else 4096;
const matvec_probe_m: usize = if (builtin.mode == .Debug) 512 else 32768;
const matvec_samples: usize = if (builtin.mode == .Debug) 3 else 5;

const hash_bytes: usize = 8;
const header_bytes: usize = 4 + 4;

fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    if (rc != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sane(value: ?f64, fallback: f64) f64 {
    if (value) |v| {
        if (std.math.isFinite(v) and v > 0) return v;
    }
    return fallback;
}

/// One worker's slice of the aggregate DRAM stream. `acc` keeps the read from
/// being optimized away.
const StreamJob = struct {
    data: []const u8,
    cpu: ?u16 = null,
    acc: u64 = 0,

    fn run(self: *StreamJob) void {
        if (self.cpu) |c| cpu.pinCurrentThread(c);
        var acc: u64 = 0;
        for (self.data) |b| acc +%= b;
        self.acc = acc;
    }
};

const DramCurve = struct {
    per_thread: f64,
    aggregate: f64,
};

/// Aggregate DRAM bandwidth when `n` threads stream disjoint slices of `buf`
/// concurrently, keeping the fastest of `dram_samples` passes. The whole buffer
/// is read once per pass, so `buf.len / elapsed` is the aggregate throughput.
fn measureDramStreamN(
    allocator: std.mem.Allocator,
    buf: []const u8,
    n: usize,
    cpus: []const u16,
) ?f64 {
    if (n < 2) return null;
    const jobs = allocator.alloc(StreamJob, n) catch return null;
    defer allocator.free(jobs);
    const threads = allocator.alloc(std.Thread, n) catch return null;
    defer allocator.free(threads);

    var best_ns: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < dram_samples) : (i += 1) {
        var spawned: usize = 0;
        var failed = false;
        const t0 = nowNs();
        while (spawned < n) : (spawned += 1) {
            const lo = buf.len * spawned / n;
            const hi = buf.len * (spawned + 1) / n;
            jobs[spawned] = .{
                .data = buf[lo..hi],
                .cpu = if (spawned < cpus.len) cpus[spawned] else null,
            };
            threads[spawned] = std.Thread.spawn(.{}, StreamJob.run, .{&jobs[spawned]}) catch {
                failed = true;
                break;
            };
        }
        var joined: usize = 0;
        while (joined < spawned) : (joined += 1) threads[joined].join();
        const t1 = nowNs();
        var sink: u64 = 0;
        for (jobs[0..spawned]) |j| sink +%= j.acc;
        std.mem.doNotOptimizeAway(sink);
        if (failed) break;
        if (t1 > t0 and t1 - t0 < best_ns) best_ns = t1 - t0;
    }
    if (best_ns == std.math.maxInt(u64) or best_ns == 0) return null;
    const bytes: f64 = @floatFromInt(buf.len);
    return bytes / @as(f64, @floatFromInt(best_ns));
}

/// Per-thread (1 thread) and aggregate DRAM bandwidth from one shared buffer:
/// the single-thread pass is the per-thread figure, and the best pass over a
/// small power-of-two thread sweep (up to `max_threads`) is the ceiling. The
/// aggregate is never below the per-thread measurement.
fn measureDram(allocator: std.mem.Allocator, max_threads: usize, cpus: []const u16) ?DramCurve {
    const buf = allocator.alloc(u8, dram_bytes) catch return null;
    defer allocator.free(buf);
    @memset(buf, 0x5A);

    var per_thread: f64 = 0;
    var i: usize = 0;
    while (i < dram_samples) : (i += 1) {
        const t0 = nowNs();
        var acc: u64 = 0;
        for (buf) |b| acc +%= b;
        std.mem.doNotOptimizeAway(acc);
        const t1 = nowNs();
        if (t1 > t0) {
            const bytes: f64 = @floatFromInt(buf.len);
            const bps = bytes / @as(f64, @floatFromInt(t1 - t0));
            if (bps > per_thread) per_thread = bps;
        }
    }
    if (!(per_thread > 0)) return null;

    var aggregate = per_thread;
    const cap = @min(max_threads, @max(@as(usize, 2), buf.len / (64 * 1024)));
    var n: usize = 2;
    while (n <= cap) : (n *= 2) {
        if (measureDramStreamN(allocator, buf, n, cpus)) |v| {
            if (v > aggregate) aggregate = v;
        }
    }
    if (cap > 2 and (cap & (cap - 1)) != 0) {
        if (measureDramStreamN(allocator, buf, cap, cpus)) |v| {
            if (v > aggregate) aggregate = v;
        }
    }
    return .{ .per_thread = per_thread, .aggregate = aggregate };
}

/// Thread count the DRAM sweep may use: physical cores, bounded by the logical
/// count (and the allowed cgroup/affinity set) and the sweep cap.
fn measureThreadBudget(allocator: std.mem.Allocator) usize {
    const logical: usize = @intCast(std.Thread.getCpuCount() catch 1);
    const phys: usize = @intCast(cpu.physicalCoreCount() orelse @as(u32, @intCast(@min(logical, std.math.maxInt(u32)))));
    var cap: usize = @max(@as(usize, 1), @min(phys, logical));
    if (cpu.affinityPlan(allocator, @intCast(max_sweep_threads)) catch null) |plan| {
        defer allocator.free(plan);
        if (plan.len > 0 and plan.len < cap) cap = plan.len;
    }
    return @min(@max(cap, 1), max_sweep_threads);
}

/// Drop the sample's pages then force a cold read, keeping the fastest pass.
fn measureDiskBw(io: Io, sample_path: ?[]const u8) ?f64 {
    if (builtin.os.tag != .linux) return null;
    const path = sample_path orelse return null;
    const page: usize = 4096;
    const probe = Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    const size: usize = @intCast(probe.length(io) catch {
        probe.close(io);
        return null;
    });
    probe.close(io);
    if (size < 2 * page) return null;
    const len = @min(size, disk_bytes_max) / page * page;
    if (len == 0) return null;
    const buf = std.heap.page_allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(page), len) catch return null;
    defer std.heap.page_allocator.free(buf);

    var best_ns: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        // `O_DIRECT` bypasses the page cache: an `mmap` read (even after
        // `madvise(DONTNEED)`) can still be served from the RAM and inflate the
        // rate several-fold, so the disk coefficient must be a real cold read.
        const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .DIRECT = true }, 0) catch return null;
        var got: usize = 0;
        const t0 = nowNs();
        while (got < len) {
            const n = std.posix.read(fd, buf[got..]) catch break;
            if (n == 0) break;
            got += n;
        }
        const t1 = nowNs();
        _ = std.os.linux.close(fd);
        if (got < len) continue;
        if (t1 > t0 and t1 - t0 < best_ns) best_ns = t1 - t0;
    }
    if (best_ns == std.math.maxInt(u64) or best_ns == 0) return null;
    const bytes: f64 = @floatFromInt(len);
    return bytes / @as(f64, @floatFromInt(best_ns));
}

/// `madvise(WILLNEED)` cost model: `min(cap, base + per_page*pages)`.
const MadviseCost = struct { base_ns: f64, per_page_ns: f64, cap_ns: f64 };

/// Sweep `madvise(WILLNEED)` over a few page-aligned sizes of the sample and
/// fit `base + per_page*pages` capped at the largest measured cost. The
/// prefetch worker decision needs this: the syscall walks the range's pages, so
/// its cost grows with the range (i.e. with the model size). Null off Linux or
/// without a sample.
fn measureMadviseCost(io: Io, sample_path: ?[]const u8) ?MadviseCost {
    if (builtin.os.tag != .linux) return null;
    const path = sample_path orelse return null;
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    const size: usize = @intCast(file.length(io) catch return null);
    const page = std.heap.page_size_min;
    if (size < 2 * page) return null;
    const map_len = @min(size, 64 * 1024 * 1024);
    const map = std.posix.mmap(
        null,
        map_len,
        std.posix.PROT{ .READ = true },
        std.posix.MAP{ .TYPE = .PRIVATE },
        file.handle,
        0,
    ) catch return null;
    defer std.posix.munmap(map);
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrCast(map.ptr);

    const wants = [_]usize{ 4 * 1024, 64 * 1024, 1 << 20, 16 << 20, 64 << 20 };
    var base: ?f64 = null;
    var per_page: f64 = 0;
    var cap: f64 = 0;
    for (wants) |want| {
        const len = std.mem.alignForward(usize, @min(want, map_len), page);
        if (len == 0 or len > map_len) continue;
        var best: u64 = std.math.maxInt(u64);
        var i: usize = 0;
        while (i < madvise_samples) : (i += 1) {
            _ = std.os.linux.fadvise(file.handle, 0, @intCast(len), std.os.linux.POSIX_FADV.DONTNEED);
            const t0 = nowNs();
            const advice: u32 = std.posix.MADV.WILLNEED;
            _ = std.posix.madvise(ptr, len, advice) catch {};
            const t1 = nowNs();
            if (t1 > t0 and t1 - t0 < best) best = t1 - t0;
        }
        if (best == std.math.maxInt(u64)) continue;
        const ns: f64 = @floatFromInt(best);
        const pages: f64 = @floatFromInt(len / page);
        const b = base orelse blk: {
            base = ns;
            break :blk ns;
        };
        per_page = @max(per_page, (ns - b) / @max(pages, 1));
        cap = @max(cap, ns);
    }
    const b = base orelse return null;
    if (cap <= 0 or b <= 0) return null;
    return .{ .base_ns = b, .per_page_ns = per_page, .cap_ns = cap };
}

/// A tiny serial-dependent op; its min per-call time estimates dispatch/launch
/// overhead.
noinline fn tinyOp(x: u64) u64 {
    var v = x *% 0x9E37_79B9_7F4A_7C15;
    v ^= v >> 29;
    v *%= 0xBF58_476D_1CE4_E5B9;
    v ^= v >> 32;
    return v;
}

fn measureDispatchLatency() ?f64 {
    var acc: u64 = 0x1234_5678;
    var best_ns: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        const t0 = nowNs();
        var j: usize = 0;
        while (j < dispatch_iters) : (j += 1) acc = tinyOp(acc);
        std.mem.doNotOptimizeAway(acc);
        const t1 = nowNs();
        if (t1 > t0) {
            const per = (t1 - t0) / dispatch_iters;
            if (per < best_ns) best_ns = per;
        }
    }
    if (best_ns == std.math.maxInt(u64) or best_ns == 0) return null;
    return @floatFromInt(best_ns);
}

const ProbeKind = enum { q4_0, q8_0, q4_1, q4_k, q6_k, f32, f16, bf16 };

fn probeBlockElems(kind: ProbeKind) usize {
    return switch (kind) {
        .q4_0, .q8_0, .q4_1, .f32, .f16, .bf16 => 32,
        .q4_k, .q6_k => 256,
    };
}

fn probeBlockBytes(kind: ProbeKind) usize {
    return switch (kind) {
        .q4_0 => 18,
        .q8_0 => 34,
        .q4_1 => 20,
        .q4_k => 144,
        .q6_k => 210,
        .f32 => 128,
        .f16, .bf16 => 64,
    };
}

fn probeRowBytes(kind: ProbeKind, n: usize) usize {
    return (n / probeBlockElems(kind)) * probeBlockBytes(kind);
}

/// Fill `data` with deterministic synthetic blocks of `kind`.
fn fillProbe(data: []u8, kind: ProbeKind, n: usize, m: usize) void {
    const row_blocks = n / probeBlockElems(kind);
    const block_bytes = probeBlockBytes(kind);
    const row_bytes = row_blocks * block_bytes;
    for (0..m) |row| {
        for (0..row_blocks) |rb| {
            const off = row * row_bytes + rb * block_bytes;
            switch (kind) {
                .q4_0 => {
                    var nib: [16]u8 = undefined;
                    for (&nib, 0..) |*b, i| b.* = @truncate((row * 11 + rb * 7 + i * 3) & 0xff);
                    const block = quant.encodeQ4_0Block(@as(f16, 0.125), &nib);
                    @memcpy(data[off..][0..18], &block);
                },
                .q8_0 => {
                    var qs: [32]i8 = undefined;
                    for (&qs, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast((row * 17 + rb * 3 + i) % 23)) - 11);
                    const block = quant.encodeQ8_0Block(@as(f16, 0.03125), &qs);
                    @memcpy(data[off..][0..34], &block);
                },
                .q4_1 => {
                    var nib: [16]u8 = undefined;
                    for (&nib, 0..) |*b, i| b.* = @truncate((row * 13 + rb * 5 + i * 9) & 0xff);
                    const block = quant.encodeQ4_1Block(@as(f16, 0.0625), @as(f16, -0.25), &nib);
                    @memcpy(data[off..][0..20], &block);
                },
                .q4_k => {
                    for (0..144) |i| data[off + i] = @truncate((row * 29 + rb * 17 + i * 11) & 0xff);
                    // Overwrite `d`/`dmin` with finite f16 so no block is NaN/Inf.
                    std.mem.writeInt(u16, data[off..][0..2], @bitCast(@as(f16, 0.03)), .little);
                    std.mem.writeInt(u16, data[off + 2 ..][0..2], @bitCast(@as(f16, 0.01)), .little);
                },
                .q6_k => {
                    for (0..192) |i| data[off + i] = @truncate((row * 41 + rb * 23 + i * 13) & 0xff);
                    for (0..16) |i| {
                        const s: i32 = @as(i32, @intCast((row * 3 + rb + i) % 7)) - 3;
                        data[off + 192 + i] = @as(u8, @bitCast(@as(i8, @intCast(s))));
                    }
                    std.mem.writeInt(u16, data[off + 208 ..][0..2], @bitCast(@as(f16, 0.03)), .little);
                },
                .f32 => {
                    for (0..32) |j| {
                        const v: f32 = @as(f32, @floatFromInt((row * 29 + rb * 17 + j * 7) % 251)) / 251.0 - 0.5;
                        std.mem.writeInt(u32, data[off + j * 4 ..][0..4], @bitCast(v), .little);
                    }
                },
                .f16 => {
                    for (0..32) |j| {
                        const v: f16 = @floatCast(@as(f32, @floatFromInt((row * 29 + rb * 17 + j * 7) % 251)) / 251.0 - 0.5);
                        std.mem.writeInt(u16, data[off + j * 2 ..][0..2], @bitCast(v), .little);
                    }
                },
                .bf16 => {
                    for (0..32) |j| {
                        const v: f32 = @as(f32, @floatFromInt((row * 29 + rb * 17 + j * 7) % 251)) / 251.0 - 0.5;
                        std.mem.writeInt(u16, data[off + j * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
                    }
                },
            }
        }
    }
}

/// The generated candidates of one weight family (the single source the
/// efficiency probe times). q8_0 is a grid family too, so its catalog is
/// measured the same way as q4_0/q4_1/q4_k.
fn gridCandidates(comptime tag: matvec.types.KernelTag) []const matvec.Candidate {
    return switch (tag) {
        .q4_0 => &matvec.q4_0_candidates,
        .q4_1 => &matvec.q4_1_candidates,
        .q4_k => &matvec.q4_k_candidates,
        .q8_0 => &matvec.q8_0_candidates,
        .q6_k => &matvec.q6_k_candidates,
        .f32 => &matvec.f32_candidates,
        .f16 => &matvec.f16_candidates,
        .bf16 => &matvec.bf16_candidates,
    };
}

/// Run one generated catalog variant: its kernel type (L1) and dot (L2) select
/// the sweep specialization.
fn runGridGen(
    comptime tag: matvec.types.KernelTag,
    comptime dot: matvec.types.Dot,
    data: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    comptime v: matvec.Variant,
) !void {
    const K = switch (tag) {
        .q4_0 => matvec_native.Q4_0,
        .q4_1 => matvec_native.Q4_1,
        .q4_k => matvec_native.Q4_K,
        .q8_0 => matvec_native.Q8_0,
        .q6_k => matvec_native.Q6_K,
        .f32 => matvec_float.F32,
        .f16 => matvec_float.F16,
        .bf16 => matvec_bfloat.BF16,
    };
    if (dot == .int) return matvec_single.matVecIntGen(K, v, data, n, m, x, out, 0, m);
    return matvec_single.matVecFpGen(K, v, data, n, m, x, out, 0, m);
}

/// Minimum time (ns) of `variant_samples` runs of one generated catalog variant.
fn measureGridGenNs(
    comptime tag: matvec.types.KernelTag,
    comptime dot: matvec.types.Dot,
    data: []const u8,
    n: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    comptime v: matvec.Variant,
) ?f64 {
    var best: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < variant_samples) : (i += 1) {
        const t0 = nowNs();
        runGridGen(tag, dot, data, n, m, x, out, v) catch return null;
        std.mem.doNotOptimizeAway(out[0]);
        const t1 = nowNs();
        if (t1 > t0 and t1 - t0 < best) best = t1 - t0;
    }
    if (best == std.math.maxInt(u64) or best == 0) return null;
    return @floatFromInt(best);
}

/// Time the generated candidates of one weight family on a shared probe and
/// record `best_time / this_time` in `(0, 1]` per plan. The catalog is the
/// single source, so a new kernel/variant is measured automatically.
fn measureGridVariants(
    comptime tag: matvec.types.KernelTag,
    allocator: std.mem.Allocator,
    n: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    eff: *VariantEfficiencies,
) void {
    const cands = comptime gridCandidates(tag);
    @setEvalBranchQuota(1_000_000);
    const kind: ProbeKind = switch (tag) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q4_k => .q4_k,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
    };
    const bytes = std.math.mul(usize, m, probeRowBytes(kind, n)) catch return;
    const data = allocator.alloc(u8, bytes + 64) catch return;
    defer allocator.free(data);
    fillProbe(data[0..bytes], kind, n, m);

    var times: [cands.len]f64 = [_]f64{0} ** cands.len;
    var best: f64 = std.math.inf(f64);
    inline for (cands, 0..) |cand, i| {
        if (measureGridGenNs(tag, cand.kernel.dot, data, n, m, x, out, cand.variant)) |t| {
            times[i] = t;
            if (t < best) best = t;
        }
    }
    if (!(best > 0) or !std.math.isFinite(best)) return;
    inline for (cands, 0..) |cand, i| {
        if (times[i] > 0) {
            const e = best / times[i];
            if (e > 0) eff.matvec[@intFromEnum(cand.plan)] = @min(e, 1.0);
        }
    }
}

/// Measure the per-variant efficiency table for the interchangeable matvec
/// variants (the generated catalog of every grid family). Every failure leaves
/// the affected entry at its `1.0` default.
fn measureVariantEffs(allocator: std.mem.Allocator) VariantEfficiencies {
    var eff = VariantEfficiencies{};
    const n = variant_probe_n;
    const m = variant_probe_m;

    const x = allocator.alloc(f32, n) catch return eff;
    defer allocator.free(x);
    for (x, 0..) |*v, i| {
        const q: i32 = @intCast((i * 7 + 3) % 17);
        v.* = @as(f32, @floatFromInt(q - 8)) * 0.125;
    }

    const out = allocator.alloc(f32, m) catch return eff;
    defer allocator.free(out);

    measureGridVariants(.q4_0, allocator, n, m, x, out, &eff);
    measureGridVariants(.q4_1, allocator, n, m, x, out, &eff);
    measureGridVariants(.q4_k, allocator, n, m, x, out, &eff);
    measureGridVariants(.q8_0, allocator, n, m, x, out, &eff);
    measureGridVariants(.q6_k, allocator, n, m, x, out, &eff);
    measureGridVariants(.f32, allocator, n, m, x, out, &eff);
    measureGridVariants(.f16, allocator, n, m, x, out, &eff);
    measureGridVariants(.bf16, allocator, n, m, x, out, &eff);
    return eff;
}

/// Measure the q8_0 matvec at each compile-time width and return the fastest
/// lane count (4/8/16), or 0 when every width failed. The width is a machine
/// property: on an AVX2 host a 512-bit vector is split (2 ymm) and loses, so the
/// measurement, not a constant, picks. q8_0 is a grid family; the probe
/// instantiates the generated sweep at each width.
fn measureSimdWidth(allocator: std.mem.Allocator) u8 {
    const n = variant_probe_n;
    const m = variant_probe_m;
    const bytes = std.math.mul(usize, m, probeRowBytes(.q8_0, n)) catch return 0;
    const data = allocator.alloc(u8, bytes + 64) catch return 0;
    defer allocator.free(data);
    fillProbe(data[0..bytes], .q8_0, n, m);

    const x = allocator.alloc(f32, n) catch return 0;
    defer allocator.free(x);
    for (x, 0..) |*v, i| {
        const q: i32 = @intCast((i * 7 + 3) % 17);
        v.* = @as(f32, @floatFromInt(q - 8)) * 0.125;
    }
    const out = allocator.alloc(f32, m) catch return 0;
    defer allocator.free(out);

    var best: u8 = 0;
    var best_ns: f64 = std.math.inf(f64);
    inline for (simd.all) |w| {
        var b: u64 = std.math.maxInt(u64);
        var it: usize = 0;
        while (it < variant_samples) : (it += 1) {
            const t0 = nowNs();
            matvec_single.matVecFpGen(matvec_native.Q8_0, .{ .width = @intFromEnum(w), .tile = 1, .unroll = 1 }, data, n, m, x, out, 0, m) catch break;
            std.mem.doNotOptimizeAway(out[0]);
            const t1 = nowNs();
            if (t1 > t0 and t1 - t0 < b) b = t1 - t0;
        }
        if (b != std.math.maxInt(u64) and b > 0) {
            const t: f64 = @floatFromInt(b);
            if (t < best_ns) {
                best_ns = t;
                best = @intFromEnum(w);
            }
        }
    }
    return best;
}

/// Measure the matvec's single-thread achieved bandwidth (GB/s): a synthetic
/// q4_0 `[n, m]` weight stream processed by one full single-row matvec pass,
/// min-of-N timed. The matvec is compute-bound per thread, so this is well
/// below the streaming per-thread bandwidth; the compute-bound ops' memory
/// term uses it. Returns null on any failure.
fn measureMatvecBw(allocator: std.mem.Allocator) ?f64 {
    const n = matvec_probe_n;
    const m = matvec_probe_m;
    const bytes = std.math.mul(usize, m, probeRowBytes(.q4_0, n)) catch return null;
    const data = allocator.alloc(u8, bytes + 64) catch return null;
    defer allocator.free(data);
    fillProbe(data[0..bytes], .q4_0, n, m);

    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "matvec_bw_probe",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = bytes,
        .mapped_slice = data[0..bytes],
    };

    const x = allocator.alloc(f32, n) catch return null;
    defer allocator.free(x);
    for (x, 0..) |*v, i| {
        const q: i32 = @intCast((i * 7 + 3) % 17);
        v.* = @as(f32, @floatFromInt(q - 8)) * 0.125;
    }

    const out = allocator.alloc(f32, m) catch return null;
    defer allocator.free(out);

    var best: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < matvec_samples) : (i += 1) {
        const t0 = nowNs();
        matvec.matVecRowsWithPlan(&view, x, out, 0, m, .production, 0) catch return null;
        std.mem.doNotOptimizeAway(out[0]);
        const t1 = nowNs();
        if (t1 > t0 and t1 - t0 < best) best = t1 - t0;
    }
    if (best == std.math.maxInt(u64) or best == 0) return null;
    const b: f64 = @floatFromInt(bytes);
    return b / @as(f64, @floatFromInt(best));
}

/// The heterogeneous split probe: run the GPU GEMV alone, then the same op split
/// across the GPU (rows `[0, m/2)`) and `nthreads` CPU workers (rows
/// `[m/2, m)`) **concurrently**, and report the split rate over the GPU rate.
/// `< 1` = the two memories contend, so a split loses to the whole-GPU op;
/// `> 1` = the bands add. `null` when the device is unavailable.
pub fn measureSplitEfficiency(allocator: std.mem.Allocator, dev_in: device.Device, nthreads: usize) ?f64 {
    var dev = dev_in;
    const n = matvec_probe_n;
    const m = matvec_probe_m;
    const bytes = std.math.mul(usize, m, probeRowBytes(.q4_0, n)) catch return null;
    const data = allocator.alloc(u8, bytes) catch return null;
    defer allocator.free(data);
    fillProbe(data, .q4_0, n, m);
    const dims = [_]u64{ n, m };
    const view = TensorView{
        .name = "split_probe",
        .dims = &dims,
        .ggml_type = .q4_0,
        .offset = 0,
        .byte_length = bytes,
        .mapped_slice = data,
    };
    const x = allocator.alloc(f32, n) catch return null;
    defer allocator.free(x);
    for (x, 0..) |*v, i| {
        const q: i32 = @intCast((i * 7 + 3) % 17);
        v.* = @as(f32, @floatFromInt(q - 8)) * 0.125;
    }
    const out = allocator.alloc(f32, m) catch return null;
    defer allocator.free(out);

    const res = dev.uploadResident(data) orelse return null;
    defer dev.freeResident(res);

    // Sustained warm-up so the device clock is ramped (the decode's steady
    // state); a cold probe would flatter the split.
    for (0..32) |_| _ = dev.matVecResident(res, .q4_0, m, n, x, out);
    dev.sync();

    const g: usize = m / 2;
    var gpu_best: u64 = std.math.maxInt(u64);
    var split_best: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < matvec_samples) : (i += 1) {
        var t0 = nowNs();
        _ = dev.matVecResident(res, .q4_0, m, n, x, out);
        dev.sync();
        var t1 = nowNs();
        if (t1 > t0 and t1 - t0 < gpu_best) gpu_best = t1 - t0;

        t0 = nowNs();
        const h = dev.matVecResidentAsync(res, .q4_0, g, n, x) orelse return null;
        cpuRows(&view, x, out, g, m, nthreads);
        _ = dev.waitResident(h, out[0..g], g);
        t1 = nowNs();
        if (t1 > t0 and t1 - t0 < split_best) split_best = t1 - t0;
    }
    if (gpu_best == std.math.maxInt(u64) or split_best == std.math.maxInt(u64) or gpu_best == 0 or split_best == 0) return null;
    const r_gpu = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gpu_best));
    const r_split = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(split_best));
    return r_split / r_gpu;
}

const SplitJob = struct { view: *const TensorView, x: []const f32, out: []f32, begin: usize, end: usize };
fn splitWorker(job: SplitJob) void {
    matvec.matVecRows(job.view, job.x, job.out, job.begin, job.end) catch {};
}
/// Run the matvec over rows `[begin, end)` on up to `nthreads` CPU threads.
fn cpuRows(view: *const TensorView, x: []const f32, out: []f32, begin: usize, end: usize, nthreads: usize) void {
    var threads: [64]std.Thread = undefined;
    var nt: usize = 0;
    var b = begin;
    var t: usize = 0;
    const n = @max(nthreads, 1);
    while (t < n and b < end) : (t += 1) {
        const rem = end - b;
        const left = n - t;
        var chunk = rem / left;
        if (rem % left != 0) chunk += 1;
        if (chunk == 0) break;
        threads[nt] = std.Thread.spawn(.{}, splitWorker, .{SplitJob{ .view = view, .x = x, .out = out, .begin = b, .end = b + chunk }}) catch break;
        nt += 1;
        b += chunk;
    }
    for (threads[0..nt]) |th| th.join();
}

/// Measure the device coefficients. Any failed sample falls back to the
/// matching conservative constant, so the result is always usable (never an
/// error). DRAM workers are pinned to distinct CPUs by `measureDram`; the
/// calling thread is left unpinned so it does not restrict the spawned threads.
pub fn measure(allocator: std.mem.Allocator, io: Io, sample_path: ?[]const u8) Coeffs {
    const max_threads = measureThreadBudget(allocator);
    const sweep_cpus = cpu.affinityPlan(allocator, @intCast(max_threads)) catch null;
    defer if (sweep_cpus) |c| allocator.free(c);

    const cpus: []const u16 = sweep_cpus orelse &.{};
    const curve = measureDram(allocator, max_threads, cpus);
    // When the sweep fails, fall back to a per-thread figure that still scales
    // to the aggregate at the core count, so the parts search does not stall on
    // a flat bandwidth curve.
    const per_thread_fallback = fallback_dram_bw_gbps / @as(f64, @floatFromInt(@max(max_threads, 1)));
    const per_thread = sane(if (curve) |c| c.per_thread else null, per_thread_fallback);
    const aggregate = sane(if (curve) |c| c.aggregate else null, per_thread);
    const madvise = measureMadviseCost(io, sample_path);

    return .{
        .dram_bw_gbps = @max(aggregate, per_thread),
        .dram_bw_1t_gbps = per_thread,
        .matvec_bw_1t_gbps = sane(measureMatvecBw(allocator), per_thread),
        .disk_bw_gbps = sane(measureDiskBw(io, sample_path), fallback_disk_bw_gbps),
        .dispatch_latency_ns = sane(measureDispatchLatency(), fallback_dispatch_latency_ns),
        .madvise_ns_base = if (madvise) |m| m.base_ns else fallback_madvise_ns_base,
        .madvise_ns_per_page = if (madvise) |m| m.per_page_ns else fallback_madvise_ns_per_page,
        .madvise_ns_cap = if (madvise) |m| m.cap_ns else fallback_madvise_ns_cap,
        .simd_width = measureSimdWidth(allocator),
        .variant_eff = measureVariantEffs(allocator),
    };
}

fn zatName(buf: []u8, digest: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.zat", .{digest});
}

fn appendBool(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: bool) !void {
    try appendU32(allocator, l, if (v) 1 else 0);
}

fn appendF64(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: f64) !void {
    try appendU64(allocator, l, @bitCast(v));
}

fn appendVariantEff(allocator: std.mem.Allocator, l: *std.ArrayList(u8), v: VariantEfficiencies) !void {
    for (v.matvec) |e| try appendF64(allocator, l, e);
    for (v.matmul) |e| try appendF64(allocator, l, e);
}

fn readVariantEff(br: *BinaryReader) !VariantEfficiencies {
    var v = VariantEfficiencies{};
    for (&v.matvec) |*e| e.* = try br.readF64();
    for (&v.matmul) |*e| e.* = try br.readF64();
    return v;
}

fn appendCaps(allocator: std.mem.Allocator, l: *std.ArrayList(u8), caps: Caps) !void {
    try appendU16(allocator, l, caps.simd_lanes);
    try appendBool(allocator, l, caps.avx512);
    try appendBool(allocator, l, caps.avx2);
    try appendBool(allocator, l, caps.neon);
    try appendBool(allocator, l, caps.vnni);
    try appendBool(allocator, l, caps.bf16);
    try appendU32(allocator, l, caps.physical_cores);
    try appendU32(allocator, l, caps.logical_cpus);
    try appendBool(allocator, l, caps.smt);
    try appendU32(allocator, l, caps.numa_nodes);
    try appendU64(allocator, l, caps.l1_bytes);
    try appendU64(allocator, l, caps.l2_bytes);
    try appendU64(allocator, l, caps.l3_bytes);
    try appendU16(allocator, l, caps.cache_line);
    try appendU64(allocator, l, caps.base_freq_hz);
    try appendF64(allocator, l, caps.dram_bw_gbps);
    try appendF64(allocator, l, caps.dram_bw_1t_gbps);
    try appendF64(allocator, l, caps.matvec_bw_1t_gbps);
    try appendF64(allocator, l, caps.madvise_ns_base);
    try appendF64(allocator, l, caps.madvise_ns_per_page);
    try appendF64(allocator, l, caps.madvise_ns_cap);
    try appendU32(allocator, l, caps.cgroup_cpus);
    try appendU32(allocator, l, caps.available_cpus);
}

fn readCaps(br: *BinaryReader) !Caps {
    var caps = Caps{};
    caps.simd_lanes = try br.readU16();
    caps.avx512 = (try br.readU32()) != 0;
    caps.avx2 = (try br.readU32()) != 0;
    caps.neon = (try br.readU32()) != 0;
    caps.vnni = (try br.readU32()) != 0;
    caps.bf16 = (try br.readU32()) != 0;
    caps.physical_cores = try br.readU32();
    caps.logical_cpus = try br.readU32();
    caps.smt = (try br.readU32()) != 0;
    caps.numa_nodes = try br.readU32();
    caps.l1_bytes = try br.readU64();
    caps.l2_bytes = try br.readU64();
    caps.l3_bytes = try br.readU64();
    caps.cache_line = try br.readU16();
    caps.base_freq_hz = try br.readU64();
    caps.dram_bw_gbps = try br.readF64();
    caps.dram_bw_1t_gbps = try br.readF64();
    caps.matvec_bw_1t_gbps = try br.readF64();
    caps.madvise_ns_base = try br.readF64();
    caps.madvise_ns_per_page = try br.readF64();
    caps.madvise_ns_cap = try br.readF64();
    caps.cgroup_cpus = try br.readU32();
    caps.available_cpus = try br.readU32();
    return caps;
}

/// Serialize the device profile to `<digest>.zat` inside `dir`.
pub fn save(
    allocator: std.mem.Allocator,
    dir: Io.Dir,
    io: Io,
    digest: []const u8,
    profile: Profile,
) !void {
    var name_buf: [256]u8 = undefined;
    const name = try zatName(&name_buf, digest);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);
    try appendCaps(allocator, &buf, profile.caps);
    try appendF64(allocator, &buf, profile.coeffs.dram_bw_gbps);
    try appendF64(allocator, &buf, profile.coeffs.dram_bw_1t_gbps);
    try appendF64(allocator, &buf, profile.coeffs.matvec_bw_1t_gbps);
    try appendF64(allocator, &buf, profile.coeffs.disk_bw_gbps);
    try appendF64(allocator, &buf, profile.coeffs.dispatch_latency_ns);
    try appendF64(allocator, &buf, profile.coeffs.madvise_ns_base);
    try appendF64(allocator, &buf, profile.coeffs.madvise_ns_per_page);
    try appendF64(allocator, &buf, profile.coeffs.madvise_ns_cap);
    try buf.append(allocator, profile.coeffs.simd_width);
    try appendVariantEff(allocator, &buf, profile.coeffs.variant_eff);
    try appendF64(allocator, &buf, profile.coeffs.gpu.gemv_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.gemv_streamed_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.matmul_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.split_efficiency);
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.q4_0_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.q4_1_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.q8_0_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.q6_k_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.q4_k_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.f32_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.f16_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.bf16_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.gate_up_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.gate_up_q4_1_variant));
    try buf.append(allocator, @intFromEnum(profile.coeffs.gpu.gate_up_q8_0_variant));
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_u4_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_sub_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_un2_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_un4_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_w32_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_0_w128_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_k_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.q4_k_u4_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.gate_up_gbps);
    try appendF64(allocator, &buf, profile.coeffs.gpu.gate_up_u4_gbps);
    try appendU64(allocator, &buf, std.hash.Wyhash.hash(0, buf.items));

    var f = try dir.createFile(io, name, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
}

/// Load `<digest>.zat`, or null when it is missing, corrupt, or an older
/// (incompatible) format.
pub fn load(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, digest: []const u8) ?Profile {
    return loadInner(allocator, dir, io, digest) catch null;
}

fn loadInner(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, digest: []const u8) !Profile {
    _ = allocator;
    var name_buf: [256]u8 = undefined;
    const name = try zatName(&name_buf, digest);

    var mmap = try MappedFile.openIn(dir, io, name);
    defer mmap.close();
    if (mmap.data.len < header_bytes + hash_bytes + 1) return error.Corrupt;

    const body = mmap.data[0 .. mmap.data.len - hash_bytes];
    var stored: [hash_bytes]u8 = undefined;
    @memcpy(&stored, mmap.data[mmap.data.len - hash_bytes ..]);
    if (std.hash.Wyhash.hash(0, body) != std.mem.readInt(u64, &stored, .little)) {
        return error.BadHash;
    }

    var br = BinaryReader.init(body);
    if (!std.mem.eql(u8, try br.readSlice(4), MAGIC)) return error.BadMagic;
    if (try br.readU32() != VERSION) return error.BadVersion;

    const caps = try readCaps(&br);
    const coeffs = Coeffs{
        .dram_bw_gbps = try br.readF64(),
        .dram_bw_1t_gbps = try br.readF64(),
        .matvec_bw_1t_gbps = try br.readF64(),
        .disk_bw_gbps = try br.readF64(),
        .dispatch_latency_ns = try br.readF64(),
        .madvise_ns_base = try br.readF64(),
        .madvise_ns_per_page = try br.readF64(),
        .madvise_ns_cap = try br.readF64(),
        .simd_width = try br.readU8(),
        .variant_eff = try readVariantEff(&br),
        .gpu = .{
            .gemv_gbps = try br.readF64(),
            .gemv_streamed_gbps = try br.readF64(),
            .matmul_gbps = try br.readF64(),
            .split_efficiency = try br.readF64(),
            .q4_0_variant = @enumFromInt(try br.readU8()),
            .q4_1_variant = @enumFromInt(try br.readU8()),
            .q8_0_variant = @enumFromInt(try br.readU8()),
            .q6_k_variant = @enumFromInt(try br.readU8()),
            .q4_k_variant = @enumFromInt(try br.readU8()),
            .f32_variant = @enumFromInt(try br.readU8()),
            .f16_variant = @enumFromInt(try br.readU8()),
            .bf16_variant = @enumFromInt(try br.readU8()),
            .gate_up_variant = @enumFromInt(try br.readU8()),
            .gate_up_q4_1_variant = @enumFromInt(try br.readU8()),
            .gate_up_q8_0_variant = @enumFromInt(try br.readU8()),
            .q4_0_u4_gbps = try br.readF64(),
            .q4_0_sub_gbps = try br.readF64(),
            .q4_0_un2_gbps = try br.readF64(),
            .q4_0_un4_gbps = try br.readF64(),
            .q4_0_w32_gbps = try br.readF64(),
            .q4_0_w128_gbps = try br.readF64(),
            .q4_k_gbps = try br.readF64(),
            .q4_k_u4_gbps = try br.readF64(),
            .gate_up_gbps = try br.readF64(),
            .gate_up_u4_gbps = try br.readF64(),
        },
    };
    return .{ .caps = caps, .coeffs = coeffs };
}

test "measure yields positive coefficients and falls back on a bad sample" {
    const testing = std.testing;
    const c = measure(testing.allocator, testing.io, null);
    try testing.expect(c.dram_bw_gbps > 0);
    try testing.expect(c.dram_bw_1t_gbps > 0);
    try testing.expect(c.dram_bw_gbps >= c.dram_bw_1t_gbps);
    try testing.expect(c.matvec_bw_1t_gbps > 0);
    try testing.expect(c.disk_bw_gbps > 0);
    try testing.expect(c.dispatch_latency_ns > 0);
    try testing.expect(c.madvise_ns_base > 0);
    try testing.expect(c.madvise_ns_per_page > 0);
    try testing.expect(c.madvise_ns_cap >= c.madvise_ns_base);

    for (c.variant_eff.matvec) |e| try testing.expect(e > 0 and e <= 1.0);
    for (c.variant_eff.matmul) |e| try testing.expect(e > 0 and e <= 1.0);

    const missing = measure(testing.allocator, testing.io, ".zig-cache/tmp/does-not-exist.ztef");
    try testing.expectEqual(fallback_disk_bw_gbps, missing.disk_bw_gbps);
}

test "profile save/load round-trip and corruption rejection" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const digest = "0123456789abcdef";
    try testing.expect(load(testing.allocator, tmp.dir, testing.io, digest) == null);

    const caps = Caps{
        .simd_lanes = 16,
        .avx512 = true,
        .physical_cores = 8,
        .logical_cpus = 16,
        .smt = true,
        .l1_bytes = 48 * 1024,
        .l2_bytes = 2 * 1024 * 1024,
        .l3_bytes = 32 * 1024 * 1024,
        .cache_line = 64,
        .base_freq_hz = 4_200_000_000,
        .dram_bw_gbps = 30.0,
        .dram_bw_1t_gbps = 9.5,
        .matvec_bw_1t_gbps = 3.75,
        .disk_bw_gbps = 3.5,
        .dispatch_latency_ns = 42.0,
        .madvise_ns_base = 410.0,
        .madvise_ns_per_page = 44.0,
        .madvise_ns_cap = 92_000.0,
        .available_cpus = 16,
    };
    var veps = VariantEfficiencies{};
    veps.matvec[@intFromEnum(MatVecPlan.Q4_0_fp_w4t1un1_p0)] = 0.875;
    veps.matvec[@intFromEnum(MatVecPlan.Q4_0_fp_w4t1un2_p0)] = 0.625;
    veps.matmul[1] = 0.5;
    const coeffs = Coeffs{
        .dram_bw_gbps = 30.0,
        .dram_bw_1t_gbps = 9.5,
        .matvec_bw_1t_gbps = 3.75,
        .disk_bw_gbps = 3.5,
        .dispatch_latency_ns = 42.0,
        .madvise_ns_base = 410.0,
        .madvise_ns_per_page = 44.0,
        .madvise_ns_cap = 92_000.0,
        .variant_eff = veps,
        .gpu = .{ .gemv_gbps = 2.5, .gemv_streamed_gbps = 1.25, .matmul_gbps = 14.0 },
    };
    try save(testing.allocator, tmp.dir, testing.io, digest, .{ .caps = caps, .coeffs = coeffs });

    const got = load(testing.allocator, tmp.dir, testing.io, digest) orelse return error.LoadFailed;
    try testing.expectEqual(caps.simd_lanes, got.caps.simd_lanes);
    try testing.expectEqual(caps.avx512, got.caps.avx512);
    try testing.expectEqual(caps.physical_cores, got.caps.physical_cores);
    try testing.expectEqual(caps.logical_cpus, got.caps.logical_cpus);
    try testing.expectEqual(caps.smt, got.caps.smt);
    try testing.expectEqual(caps.l1_bytes, got.caps.l1_bytes);
    try testing.expectEqual(caps.l2_bytes, got.caps.l2_bytes);
    try testing.expectEqual(caps.l3_bytes, got.caps.l3_bytes);
    try testing.expectEqual(caps.base_freq_hz, got.caps.base_freq_hz);
    try testing.expectEqual(caps.available_cpus, got.caps.available_cpus);
    try testing.expectEqual(caps.dram_bw_1t_gbps, got.caps.dram_bw_1t_gbps);
    try testing.expectEqual(caps.matvec_bw_1t_gbps, got.caps.matvec_bw_1t_gbps);
    try testing.expectEqual(coeffs.dram_bw_gbps, got.coeffs.dram_bw_gbps);
    try testing.expectEqual(coeffs.dram_bw_1t_gbps, got.coeffs.dram_bw_1t_gbps);
    try testing.expectEqual(coeffs.matvec_bw_1t_gbps, got.coeffs.matvec_bw_1t_gbps);
    try testing.expectEqual(coeffs.disk_bw_gbps, got.coeffs.disk_bw_gbps);
    try testing.expectEqual(coeffs.dispatch_latency_ns, got.coeffs.dispatch_latency_ns);
    try testing.expectEqual(coeffs.madvise_ns_base, got.coeffs.madvise_ns_base);
    try testing.expectEqual(coeffs.madvise_ns_per_page, got.coeffs.madvise_ns_per_page);
    try testing.expectEqual(coeffs.madvise_ns_cap, got.coeffs.madvise_ns_cap);
    try testing.expectEqualSlices(f64, &coeffs.variant_eff.matvec, &got.coeffs.variant_eff.matvec);
    try testing.expectEqualSlices(f64, &coeffs.variant_eff.matmul, &got.coeffs.variant_eff.matmul);
    try testing.expectEqual(coeffs.gpu.gemv_gbps, got.coeffs.gpu.gemv_gbps);
    try testing.expectEqual(coeffs.gpu.gemv_streamed_gbps, got.coeffs.gpu.gemv_streamed_gbps);
    try testing.expectEqual(coeffs.gpu.matmul_gbps, got.coeffs.gpu.matmul_gbps);

    var f = try tmp.dir.openFile(testing.io, "0123456789abcdef.zat", .{ .mode = .read_write });
    defer f.close(testing.io);
    try f.writePositionalAll(testing.io, &[_]u8{0xFF}, 0);
    try testing.expect(load(testing.allocator, tmp.dir, testing.io, digest) == null);
}

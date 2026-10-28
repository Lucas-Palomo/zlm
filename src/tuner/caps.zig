const std = @import("std");
const builtin = @import("builtin");
const cpu = @import("../utils/cpu.zig");
const api = @import("../kernels/api.zig");
const calibrate = @import("calibrate.zig");

/// Conservative DRAM bandwidth (GB/s) used when SMBIOS/calibration is not
/// available. Calibration refines this coefficient per device.
pub const default_dram_bw_gbps: f64 = calibrate.fallback_dram_bw_gbps;

/// FMA ports assumed by the peak model when the microarchitecture is unknown.
pub const fma_ports: f64 = 2.0;

/// Deterministic, best-effort hardware capabilities: ISA/SIMD, topology, cache,
/// frequency, memory and cgroup budget. Every field has a conservative default
/// so `detect` never fails and works on any Linux/macOS.
pub const Caps = struct {
    simd_lanes: u16 = 1,
    avx512: bool = false,
    avx2: bool = false,
    neon: bool = false,
    vnni: bool = false,
    bf16: bool = false,

    physical_cores: u32 = 1,
    logical_cpus: u32 = 1,
    smt: bool = false,
    numa_nodes: u32 = 1,

    l1_bytes: u64 = 32 * 1024,
    l2_bytes: u64 = 256 * 1024,
    l3_bytes: u64 = 8 * 1024 * 1024,
    cache_line: u16 = 64,

    base_freq_hz: u64 = 3_000_000_000,

    dram_bw_gbps: f64 = default_dram_bw_gbps,
    /// Measured single-thread streaming bandwidth (GB/s). The scaling curve is
    /// `bw(n) = min(dram_bw_gbps, dram_bw_1t_gbps * n)`; it falls back to the
    /// aggregate so an uncalibrated machine does not scale the memory term.
    dram_bw_1t_gbps: f64 = default_dram_bw_gbps,
    /// Measured matvec single-thread achieved bandwidth (GB/s). The matvec is
    /// compute-bound per thread, so this is below the streaming figure; the
    /// compute-bound ops' memory term uses it (see `bwAtOp`). Falls back to the
    /// aggregate so an uncalibrated machine does not scale the memory term.
    matvec_bw_1t_gbps: f64 = default_dram_bw_gbps,
    /// Measured disk streaming bandwidth (GB/s); conservative default until
    /// calibration refines it.
    disk_bw_gbps: f64 = calibrate.fallback_disk_bw_gbps,
    /// Measured per-dispatch latency (ns); conservative default until
    /// calibration refines it.
    dispatch_latency_ns: f64 = calibrate.fallback_dispatch_latency_ns,
    /// Measured `madvise(WILLNEED)` cost model: `min(cap, base + per_page*pages)`.
    /// Used to decide whether a residency prefetch is worth a dedicated worker.
    madvise_ns_base: f64 = calibrate.fallback_madvise_ns_base,
    madvise_ns_per_page: f64 = calibrate.fallback_madvise_ns_per_page,
    madvise_ns_cap: f64 = calibrate.fallback_madvise_ns_cap,
    /// Fastest measured SIMD width (f32 lanes: 4/8/16) for the compute kernels;
    /// `0` = unmeasured. Derived from `Coeffs.simd_width` by `applyCoeffs`.
    simd_width: u8 = 0,
    /// Measured per-variant efficiency in `(0, 1]` for the interchangeable
    /// matvec/matmul variants; all `1.0` until calibration refines it.
    variant_eff: calibrate.VariantEfficiencies = .{},

    /// Measured co-processor throughput (all-zero until a `tune --gpu` probe
    /// records it; the offload policy then falls back to its constants).
    gpu: calibrate.GpuCoeffs = .{},

    /// cgroup CPU quota in cores (0 when unset/unlimited/unreadable).
    cgroup_cpus: u32 = 0,
    /// Effective CPUs available to the process (affinity + cgroup quota).
    available_cpus: u32 = 1,

    /// Peak f32 FMA throughput (FLOP/s): cores * SIMD lanes * FMA ports * 2
    /// (mul+add) * clock. Calibration (Stage 2) can refine `fma_ports`/clock.
    pub fn peakFlops(self: Caps) f64 {
        const cores: f64 = @floatFromInt(@max(self.physical_cores, 1));
        const lanes: f64 = @floatFromInt(@max(self.simd_lanes, 1));
        const freq: f64 = @floatFromInt(self.base_freq_hz);
        return cores * lanes * fma_ports * 2.0 * freq;
    }

    /// Override the bandwidth/latency coefficients with native measurements.
    /// Calibration shifts the cost model; it never picks an algorithm.
    pub fn applyCoeffs(self: *Caps, coeffs: calibrate.Coeffs) void {
        self.dram_bw_gbps = coeffs.dram_bw_gbps;
        self.dram_bw_1t_gbps = coeffs.dram_bw_1t_gbps;
        self.matvec_bw_1t_gbps = coeffs.matvec_bw_1t_gbps;
        self.disk_bw_gbps = coeffs.disk_bw_gbps;
        self.dispatch_latency_ns = coeffs.dispatch_latency_ns;
        self.madvise_ns_base = coeffs.madvise_ns_base;
        self.madvise_ns_per_page = coeffs.madvise_ns_per_page;
        self.madvise_ns_cap = coeffs.madvise_ns_cap;
        self.simd_width = coeffs.simd_width;
        self.variant_eff = coeffs.variant_eff;
        self.gpu = coeffs.gpu;
    }

    /// Estimated cost (ns) of issuing one `madvise(WILLNEED)` over `bytes`: the
    /// syscall base plus a per-page walk, capped. Grows with the range size.
    pub fn prefetchIssueNs(self: Caps, bytes: u64) f64 {
        const page: f64 = @floatFromInt(@max(std.heap.page_size_min, 1));
        const pages = @as(f64, @floatFromInt(bytes)) / page;
        const est = self.madvise_ns_base + self.madvise_ns_per_page * pages;
        return @min(self.madvise_ns_cap, est);
    }

    /// Effective DRAM bandwidth (GB/s) with `parts` threads, from the measured
    /// scaling curve: `min(ceiling, per_thread * parts)`. `parts` is clamped to
    /// at least 1. When the per-thread figure is unavailable it falls back to
    /// the aggregate, so the curve is flat (no scaling).
    pub fn bwAt(self: Caps, parts: u16) f64 {
        const n: f64 = @floatFromInt(@max(parts, 1));
        const ceiling = if (self.dram_bw_gbps > 0) self.dram_bw_gbps else default_dram_bw_gbps;
        const per_thread = if (self.dram_bw_1t_gbps > 0) self.dram_bw_1t_gbps else ceiling;
        return @min(ceiling, per_thread * n);
    }

    /// Effective DRAM bandwidth (GB/s) with `parts` threads, op-aware: the
    /// compute-bound ops (`matvec`, `matmul`, `attention`) scale with the
    /// measured matvec per-thread figure (they are compute-bound per thread),
    /// the rest with the streaming per-thread figure. The aggregate ceiling is
    /// unchanged. `parts` is clamped to at least 1; a missing figure falls back
    /// to the ceiling so an uncalibrated machine does not scale the memory term.
    pub fn bwAtOp(self: Caps, op: api.Op, parts: u16) f64 {
        const n: f64 = @floatFromInt(@max(parts, 1));
        const ceiling = if (self.dram_bw_gbps > 0) self.dram_bw_gbps else default_dram_bw_gbps;
        const per_thread = switch (op) {
            .matvec, .matmul, .attention => if (self.matvec_bw_1t_gbps > 0) self.matvec_bw_1t_gbps else ceiling,
            else => if (self.dram_bw_1t_gbps > 0) self.dram_bw_1t_gbps else ceiling,
        };
        return @min(ceiling, per_thread * n);
    }
};

/// Assemble a `Caps` for the current machine. Best-effort: missing sources fall
/// back to the struct defaults; the function never returns an error.
pub fn detect(allocator: std.mem.Allocator) Caps {
    var caps = Caps{};

    const isa = detectIsa();
    caps.simd_lanes = isa.simd_lanes;
    caps.avx512 = isa.avx512;
    caps.avx2 = isa.avx2;
    caps.neon = isa.neon;
    caps.vnni = isa.vnni;
    caps.bf16 = isa.bf16;

    const logical = std.Thread.getCpuCount() catch 1;
    caps.logical_cpus = clampU32(logical);
    const phys = cpu.physicalCoreCount() orelse caps.logical_cpus;
    caps.physical_cores = @min(phys, caps.logical_cpus);
    if (caps.physical_cores == 0) caps.physical_cores = 1;
    caps.smt = caps.logical_cpus > caps.physical_cores;

    caps.numa_nodes = numaNodeCount();
    detectCache(&caps);
    if (detectBaseFreqHz()) |hz| caps.base_freq_hz = hz;

    if (cpu.quotaCpus()) |q| caps.cgroup_cpus = clampU32(q);
    caps.available_cpus = availableCpus(allocator, caps.logical_cpus);

    return caps;
}

fn detectIsa() struct { simd_lanes: u16, avx512: bool, avx2: bool, neon: bool, vnni: bool, bf16: bool } {
    var avx512 = false;
    var avx2 = false;
    var avx = false;
    var sse = false;
    var neon = false;
    var vnni = false;
    var bf16 = false;

    const f = builtin.cpu.features;
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => {
            avx512 = std.Target.x86.featureSetHas(f, .avx512f);
            avx2 = std.Target.x86.featureSetHas(f, .avx2);
            avx = std.Target.x86.featureSetHas(f, .avx);
            sse = std.Target.x86.featureSetHas(f, .sse);
            vnni = std.Target.x86.featureSetHas(f, .avx512vnni) or
                std.Target.x86.featureSetHas(f, .avxvnni);
            bf16 = std.Target.x86.featureSetHas(f, .avx512bf16);
        },
        .aarch64 => {
            neon = std.Target.aarch64.featureSetHas(f, .neon);
            bf16 = std.Target.aarch64.featureSetHas(f, .bf16);
        },
        else => {},
    }

    const lanes: u16 = if (avx512)
        16
    else if (avx2 or avx)
        8
    else if (neon or sse)
        4
    else
        1;
    return .{ .simd_lanes = lanes, .avx512 = avx512, .avx2 = avx2, .neon = neon, .vnni = vnni, .bf16 = bf16 };
}

/// Linux-only cgroup-aware available CPU count via the shared affinity plan;
/// falls back to `logical` off Linux or on any failure.
fn availableCpus(allocator: std.mem.Allocator, logical: u32) u32 {
    if (logical == 0) return 1;
    const requested: u16 = @intCast(@min(logical, std.math.maxInt(u16)));
    const plan = cpu.affinityPlan(allocator, requested) catch return logical;
    defer allocator.free(plan);
    if (plan.len == 0) return logical;
    return @intCast(@min(plan.len, logical));
}

fn clampU32(n: usize) u32 {
    if (n == 0) return 1;
    if (n > std.math.maxInt(u32)) return std.math.maxInt(u32);
    return @intCast(n);
}

/// Count CPUs in a sysfs list such as "0-3,8-11". Returns 0 when unparseable.
fn countList(data: []const u8) u32 {
    var total: u64 = 0;
    var it = std.mem.tokenizeAny(u8, data, ", \t\r\n");
    while (it.next()) |tok| {
        var lo: u64 = 0;
        var hi: u64 = 0;
        if (std.mem.indexOfScalar(u8, tok, '-')) |dash| {
            lo = std.fmt.parseInt(u64, tok[0..dash], 10) catch continue;
            hi = std.fmt.parseInt(u64, tok[dash + 1 ..], 10) catch continue;
        } else {
            lo = std.fmt.parseInt(u64, tok, 10) catch continue;
            hi = lo;
        }
        if (hi < lo) continue;
        total += hi - lo + 1;
    }
    if (total == 0) return 0;
    if (total > std.math.maxInt(u32)) return std.math.maxInt(u32);
    return @intCast(total);
}

fn numaNodeCount() u32 {
    if (builtin.os.tag == .linux) {
        var buf: [256]u8 = undefined;
        if (cpu.readFileSmall("/sys/devices/system/node/online", &buf)) |data| {
            const n = countList(data);
            if (n > 0) return n;
        }
    }
    return 1;
}

/// Parse a sysfs cache size such as "32K", "1024K", "8M" into bytes.
fn parseCacheSize(s: []const u8) ?u64 {
    const trimmed = std.mem.trim(u8, s, " \t\r\n");
    if (trimmed.len == 0) return null;
    const last = trimmed[trimmed.len - 1];
    const mult: u64 = switch (last) {
        'K', 'k' => 1024,
        'M', 'm' => 1024 * 1024,
        'G', 'g' => 1024 * 1024 * 1024,
        else => 1,
    };
    const digits = if (mult == 1) trimmed else trimmed[0 .. trimmed.len - 1];
    const value = std.fmt.parseInt(u64, digits, 10) catch return null;
    return value * mult;
}

fn readCacheField(index: u32, field: []const u8, buf: []u8) ?[]const u8 {
    var path_buf: [160]u8 = undefined;
    const path = std.fmt.bufPrint(
        &path_buf,
        "/sys/devices/system/cpu/cpu0/cache/index{d}/{s}",
        .{ index, field },
    ) catch return null;
    return cpu.readFileSmall(path, buf);
}

/// Fill cache sizes/line from Linux sysfs, level by level, keeping the struct
/// defaults for any level or line that is missing.
fn detectCache(caps: *Caps) void {
    if (builtin.os.tag != .linux) return;
    var index: u32 = 0;
    while (index < 16) : (index += 1) {
        var level_buf: [16]u8 = undefined;
        const level_data = readCacheField(index, "level", &level_buf) orelse continue;
        const level = std.fmt.parseInt(u8, std.mem.trim(u8, level_data, " \t\r\n"), 10) catch continue;

        var size_buf: [32]u8 = undefined;
        const size_data = readCacheField(index, "size", &size_buf) orelse continue;
        const size = parseCacheSize(size_data) orelse continue;

        var type_buf: [32]u8 = undefined;
        const type_data = readCacheField(index, "type", &type_buf);
        const is_instr = if (type_data) |t|
            std.mem.startsWith(u8, std.mem.trim(u8, t, " \t\r\n"), "Instruction")
        else
            false;

        switch (level) {
            1 => {
                if (!is_instr) caps.l1_bytes = size;
            },
            2 => caps.l2_bytes = size,
            3 => caps.l3_bytes = size,
            else => {},
        }

        var line_buf: [16]u8 = undefined;
        if (readCacheField(index, "coherency_line_size", &line_buf)) |line_data| {
            if (std.fmt.parseInt(u16, std.mem.trim(u8, line_data, " \t\r\n"), 10)) |line| {
                if (line > 0) caps.cache_line = line;
            } else |_| {}
        }
    }
}

/// Linux base/boost clock in Hz (sysfs reports kHz). Null off Linux/unknown.
fn detectBaseFreqHz() ?u64 {
    if (builtin.os.tag != .linux) return null;
    const paths = [_][]const u8{
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
    };
    for (paths) |path| {
        var buf: [32]u8 = undefined;
        const data = cpu.readFileSmall(path, &buf) orelse continue;
        const khz = std.fmt.parseInt(u64, std.mem.trim(u8, data, " \t\r\n"), 10) catch continue;
        if (khz > 0) return khz * 1000;
    }
    return null;
}

test "detect returns sane non-zero values with fallbacks" {
    const testing = std.testing;
    const caps = detect(testing.allocator);
    try testing.expect(caps.simd_lanes == 1 or caps.simd_lanes == 4 or
        caps.simd_lanes == 8 or caps.simd_lanes == 16);
    try testing.expect(caps.logical_cpus >= 1);
    try testing.expect(caps.physical_cores >= 1);
    try testing.expect(caps.physical_cores <= caps.logical_cpus);
    try testing.expect(caps.numa_nodes >= 1);
    try testing.expect(caps.l1_bytes > 0);
    try testing.expect(caps.l2_bytes > 0);
    try testing.expect(caps.l3_bytes > 0);
    try testing.expect(caps.cache_line > 0);
    try testing.expect(caps.base_freq_hz > 0);
    try testing.expect(caps.dram_bw_gbps > 0);
    try testing.expect(caps.available_cpus >= 1);
    try testing.expect(caps.available_cpus <= caps.logical_cpus);
    try testing.expect(caps.peakFlops() > 0);
}

test "physical_cores <= logical_cpus and smt is consistent" {
    const testing = std.testing;
    const caps = detect(testing.allocator);
    try testing.expectEqual(caps.logical_cpus > caps.physical_cores, caps.smt);
}

test "peakFlops scales with cores, lanes and clock" {
    const testing = std.testing;
    const a = Caps{ .physical_cores = 4, .simd_lanes = 8, .base_freq_hz = 2_000_000_000 };
    const b = Caps{ .physical_cores = 8, .simd_lanes = 8, .base_freq_hz = 2_000_000_000 };
    try testing.expect(b.peakFlops() > a.peakFlops());
    try testing.expect(a.peakFlops() > 0);
}

test "bwAtOp uses the matvec figure only for compute-bound ops" {
    const testing = std.testing;
    const caps = Caps{
        .dram_bw_gbps = 32.0,
        .dram_bw_1t_gbps = 10.0,
        .matvec_bw_1t_gbps = 4.0,
    };
    try testing.expectApproxEqAbs(@as(f64, 4.0), caps.bwAtOp(.matvec, 1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 16.0), caps.bwAtOp(.matvec, 4), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 32.0), caps.bwAtOp(.matvec, 8), 1e-9);
    try testing.expectApproxEqAbs(caps.bwAtOp(.matvec, 8), caps.bwAtOp(.matmul, 8), 1e-9);
    try testing.expectApproxEqAbs(caps.bwAtOp(.matvec, 8), caps.bwAtOp(.attention, 8), 1e-9);

    try testing.expectApproxEqAbs(@as(f64, 10.0), caps.bwAtOp(.elementwise, 1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 32.0), caps.bwAtOp(.elementwise, 4), 1e-9);
    try testing.expect(caps.bwAtOp(.elementwise, 2) > caps.bwAtOp(.matvec, 2));
}

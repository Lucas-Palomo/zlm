const std = @import("std");
const ggml = @import("../../gguf/format.zig");
const grid = @import("../kernels/grid.zig");

/// The GPU backend contract.
///
/// A backend is a module under `gpu/backend/<name>/` exposing a `Device` type
/// whose `pub` methods implement the ops listed in `required_ops`, over the
/// shared types below. `gpu/root.zig` selects one backend at build time (a
/// comptime choice, no runtime plugin) and re-exports its `Device`. A new
/// backend is a new directory implementing this contract; the gpu-facing API,
/// the kernels grid and the runtime stay untouched.
/// Quantized weight layouts the GPU GEMV supports. These mirror the raw ggml
/// block layouts (see `kernels/linear/quant`); the gpu layer deliberately does
/// not import the CPU dequant code, so the backend stays a raw-layout consumer.
pub const Quant = enum {
    q4_0,
    q4_1,
    q8_0,
    q6_k,
    q4_k,
    f32,
    f16,
    bf16,
    /// Per-quant spec-constant shader variants (the same axes for every quant:
    /// `u4` = 8-row tile, `sub` = subgroup/wave reduction, `un2`/`un4` = inner
    /// k-block unroll, `w32`/`w128` = workgroup size). Identical layout and byte
    /// size to the base quant; only the specialized pipeline differs.
    q4_0_u4,
    q4_0_sub,
    q4_0_un2,
    q4_0_un4,
    q4_0_w32,
    q4_0_w128,
    q4_1_u4,
    q4_1_sub,
    q4_1_un2,
    q4_1_un4,
    q4_1_w32,
    q4_1_w128,
    q8_0_u4,
    q8_0_sub,
    q8_0_un2,
    q8_0_un4,
    q8_0_w32,
    q8_0_w128,
    q6_k_u4,
    q6_k_sub,
    q6_k_un2,
    q6_k_un4,
    q6_k_w32,
    q6_k_w128,
    q4_k_u4,
    q4_k_sub,
    q4_k_un2,
    q4_k_un4,
    q4_k_w32,
    q4_k_w128,
    f32_u4,
    f32_sub,
    f32_un2,
    f32_un4,
    f32_w32,
    f32_w128,
    f16_u4,
    f16_sub,
    f16_un2,
    f16_un4,
    f16_w32,
    f16_w128,
    bf16_u4,
    bf16_sub,
    bf16_un2,
    bf16_un4,
    bf16_w32,
    bf16_w128,

    /// The base quant a variant decodes the same as (itself when already base).
    pub fn base(self: Quant) Quant {
        return switch (self) {
            .q4_0_u4, .q4_0_sub, .q4_0_un2, .q4_0_un4, .q4_0_w32, .q4_0_w128 => .q4_0,
            .q4_1_u4, .q4_1_sub, .q4_1_un2, .q4_1_un4, .q4_1_w32, .q4_1_w128 => .q4_1,
            .q8_0_u4, .q8_0_sub, .q8_0_un2, .q8_0_un4, .q8_0_w32, .q8_0_w128 => .q8_0,
            .q6_k_u4, .q6_k_sub, .q6_k_un2, .q6_k_un4, .q6_k_w32, .q6_k_w128 => .q6_k,
            .q4_k_u4, .q4_k_sub, .q4_k_un2, .q4_k_un4, .q4_k_w32, .q4_k_w128 => .q4_k,
            .f32_u4, .f32_sub, .f32_un2, .f32_un4, .f32_w32, .f32_w128 => .f32,
            .f16_u4, .f16_sub, .f16_un2, .f16_un4, .f16_w32, .f16_w128 => .f16,
            .bf16_u4, .bf16_sub, .bf16_un2, .bf16_un4, .bf16_w32, .bf16_w128 => .bf16,
            else => self,
        };
    }

    /// The ggml type this quant (or variant) decodes as; the spec-constant axes
    /// are keyed by it.
    pub fn ggmlType(self: Quant) ggml.GGMLType {
        return switch (self.base()) {
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

    /// Select the shader for this base quant from a recorded `GpuVariant`: a
    /// variant the base quant does not ship falls back to the plain shader.
    pub fn variant(self: Quant, v: grid.GpuVariant) Quant {
        return switch (self.base()) {
            .q4_0 => switch (v) {
                .u4 => .q4_0_u4,
                .sub => .q4_0_sub,
                .un2 => .q4_0_un2,
                .un4 => .q4_0_un4,
                .w32 => .q4_0_w32,
                .w128 => .q4_0_w128,
                else => .q4_0,
            },
            .q4_1 => switch (v) {
                .u4 => .q4_1_u4,
                .sub => .q4_1_sub,
                .un2 => .q4_1_un2,
                .un4 => .q4_1_un4,
                .w32 => .q4_1_w32,
                .w128 => .q4_1_w128,
                else => .q4_1,
            },
            .q8_0 => switch (v) {
                .u4 => .q8_0_u4,
                .sub => .q8_0_sub,
                .un2 => .q8_0_un2,
                .un4 => .q8_0_un4,
                .w32 => .q8_0_w32,
                .w128 => .q8_0_w128,
                else => .q8_0,
            },
            .q6_k => switch (v) {
                .u4 => .q6_k_u4,
                .sub => .q6_k_sub,
                .un2 => .q6_k_un2,
                .un4 => .q6_k_un4,
                .w32 => .q6_k_w32,
                .w128 => .q6_k_w128,
                else => .q6_k,
            },
            .q4_k => switch (v) {
                .u4 => .q4_k_u4,
                .sub => .q4_k_sub,
                .un2 => .q4_k_un2,
                .un4 => .q4_k_un4,
                .w32 => .q4_k_w32,
                .w128 => .q4_k_w128,
                else => .q4_k,
            },
            .f32 => switch (v) {
                .u4 => .f32_u4,
                .sub => .f32_sub,
                .un2 => .f32_un2,
                .un4 => .f32_un4,
                .w32 => .f32_w32,
                .w128 => .f32_w128,
                else => .f32,
            },
            .f16 => switch (v) {
                .u4 => .f16_u4,
                .sub => .f16_sub,
                .un2 => .f16_un2,
                .un4 => .f16_un4,
                .w32 => .f16_w32,
                .w128 => .f16_w128,
                else => .f16,
            },
            .bf16 => switch (v) {
                .u4 => .bf16_u4,
                .sub => .bf16_sub,
                .un2 => .bf16_un2,
                .un4 => .bf16_un4,
                .w32 => .bf16_w32,
                .w128 => .bf16_w128,
                else => .bf16,
            },
            else => self,
        };
    }

    /// Rows a GEMV workgroup owns for this quant+variant; must match the
    /// shader's `ROWS_PER_GROUP` (`specConsts`'s `u4` tile is the only 8).
    pub fn rowsPerGroup(self: Quant) usize {
        return switch (self) {
            .q4_0_u4, .q4_1_u4, .q8_0_u4, .q6_k_u4, .q4_k_u4, .f32_u4, .f16_u4, .bf16_u4 => 8,
            else => 4,
        };
    }

    pub fn blockElems(self: Quant) usize {
        return switch (self.base()) {
            .q4_0, .q4_1, .q8_0 => 32,
            .q6_k, .q4_k => 256,
            // Unquantized types carry one element per "block"; the row width is
            // passed as `num_blocks_per_row`.
            .f32, .f16, .bf16 => 1,
            else => unreachable,
        };
    }

    pub fn blockBytes(self: Quant) usize {
        return switch (self.base()) {
            .q4_0 => 18,
            .q4_1 => 20,
            .q8_0 => 34,
            .q6_k => 210,
            .q4_k => 144,
            .f32 => 4,
            .f16, .bf16 => 2,
            else => unreachable,
        };
    }
};

/// Gated activation a fused `gate_up` GEMV applies, recorded by the arch. The
/// numeric values match the shader's `apply_act` selector.
pub const GateUpAct = enum(u32) { silu = 0, gelu = 1 };

/// Number of in-flight GPU dispatches the resident ring supports. A batch of
/// consecutive resident GEMVs (e.g. the q/k/v projections sharing one input)
/// is submitted back-to-back, one slot each, and drained with a single wait.
pub const RingSize = 8;

/// A handle to an asynchronously submitted resident dispatch: the backend's ring
/// slot. Opaque to the caller, which only drains it with `waitResident`.
pub const Handle = struct { slot: u8 };

/// A persistent GPU weight buffer uploaded once and reused across every GEMV
/// that reads it, instead of re-staging the weight per dispatch. `bytes` is the
/// raw quantized payload length; `device` is the backend-owned device storage
/// (Vulkan buffers, Metal buffers, ...), which the creating backend owns and
/// interprets.
pub const Resident = struct {
    bytes: u64 = 0,
    device: ?*anyopaque = null,
};

/// One resident GEMV in a batch: a persistent weight, its quant, the output row
/// count and the caller's result buffer. Every item in a batch shares the same
/// input activation `x`.
pub const ResidentOp = struct {
    res: *Resident,
    quant: Quant,
    rows: usize,
    out: []f32,
};

/// A measured co-processor throughput probe: effective end-to-end bytes/ns
/// (== GB/s) over the raw weight bytes of a representative op. Unmeasured
/// fields stay `0`, so the caller keeps its conservative fallback. The
/// per-variant resident rates feed the tuner's shader-variant choice.
pub const Probe = struct {
    gemv_gbps: f64 = 0,
    gemv_streamed_gbps: f64 = 0,
    matmul_gbps: f64 = 0,
    /// Chosen GEMV shader variant per base quant, from the probe's measured
    /// resident rates (`plain` when the variant did not beat it). The tuner
    /// writes the winner onto every dispatch of that quant/op.
    q4_0_variant: grid.GpuVariant = .plain,
    q4_1_variant: grid.GpuVariant = .plain,
    q8_0_variant: grid.GpuVariant = .plain,
    q6_k_variant: grid.GpuVariant = .plain,
    q4_k_variant: grid.GpuVariant = .plain,
    f32_variant: grid.GpuVariant = .plain,
    f16_variant: grid.GpuVariant = .plain,
    bf16_variant: grid.GpuVariant = .plain,
    /// Chosen variant of the fused `gate_up` shader, per quant (q4_0/q4_1/q8_0).
    gate_up_variant: grid.GpuVariant = .plain,
    gate_up_q4_1_variant: grid.GpuVariant = .plain,
    gate_up_q8_0_variant: grid.GpuVariant = .plain,
    /// Resident GEMV rates of the unrolled variants (GB/s), for diagnostics.
    q4_0_u4_gbps: f64 = 0,
    /// Resident GEMV rate of the subgroup/wave reduction variant (GB/s).
    q4_0_sub_gbps: f64 = 0,
    /// Resident GEMV rates of the inner k-block unroll variants (GB/s).
    q4_0_un2_gbps: f64 = 0,
    q4_0_un4_gbps: f64 = 0,
    /// Resident GEMV rates of the workgroup-size variants (GB/s).
    q4_0_w32_gbps: f64 = 0,
    q4_0_w128_gbps: f64 = 0,
    q4_k_gbps: f64 = 0,
    q4_k_u4_gbps: f64 = 0,
    /// Resident fused `gate_up` rates (GB/s over both weights' bytes): the
    /// plain shader and the unrolled variant.
    gate_up_gbps: f64 = 0,
    gate_up_u4_gbps: f64 = 0,
};

/// The ops a backend's `Device` must provide (all over the shared types). This
/// is the contract's single definition; `check` asserts it at comptime.
pub const required_ops = .{
    "init",                            "deinit",
    "available",                       "backendName",
    "deviceName",                      "isUma",
    "matVec",                          "matMulBatch",
    "matMulResident",                  "matVecResident",
    "matVecResidentAsync",             "matVecResidentBurstAsync",
    "matVecGateUpResidentBurstAsync",  "matVecMlpResidentBurstAsync",
    "matVecMlpPairResidentBurstAsync", "waitResident",
    "matVecResidentBatch",             "uploadResident",
    "freeResident",                    "sync",
    "probeGpu",
};

/// Comptime assertion that `B` satisfies the backend contract: a `Device` type
/// carrying every required op. Called by `gpu/root.zig` on the selected backend.
pub fn check(comptime B: type) void {
    comptime {
        if (!@hasDecl(B, "Device")) @compileError(@typeName(B) ++ ": backend must declare a `Device` type");
        for (required_ops) |name| {
            if (!@hasDecl(B.Device, name)) {
                @compileError(@typeName(B) ++ ".Device is missing `" ++ name ++ "`");
            }
        }
    }
}

// ---- the generic probe ----
//
// Built on the contract ops, so every backend measures the same representative
// ops (a streamed and a resident q4_0 GEMV, a streamed batched q4_0 GEMM, the
// catalogued shader variants) and reports the same `Probe`. `D` is the
// backend's `Device` type; `self` is a live device.

/// Timed samples per probe; the minimum (least contaminated by drift) is kept.
/// Deliberately a cold-ish single-dispatch measurement (no sustained warm-up):
/// a tight back-to-back loop keeps the GPU at boost clocks that interleaved
/// decode does not reach, so the warm figure overvalues offload (measured:
/// 12B decode got ~2x slower with the warm value).
const probe_samples: usize = 5;
/// Upper bounds so the probe stays cheap on any model shape.
const probe_max_rows: usize = 12 * 1024;
const probe_max_k: usize = 4 * 1024;

fn probeNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    if (rc != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Measure representative co-processor throughput at `tune --gpu` time: a
/// streamed and a resident q4_0 GEMV and a streamed batched q4_0 GEMM (`m`
/// rows), each min-of-N. Bounded by `probe_max_rows`/`probe_max_k`; every
/// failure leaves the affected field `0`. The streamed figures include the
/// host<->device transfer, so no separate staging-rate constant is needed.
/// The shape comes from the model (FFN + block), so the profile is
/// deterministic given the recorded result.
pub fn probeGpu(comptime D: type, self: *D, allocator: std.mem.Allocator, rows_in: usize, k_in: usize, m_in: usize) Probe {
    var p = Probe{};
    if (!self.available()) return p;
    const rows: usize = @min(rows_in, probe_max_rows);
    var k: usize = @min(k_in, probe_max_k);
    k -= k % 32;
    const m: usize = @max(m_in, 2);
    if (rows == 0 or k == 0) return p;

    const bytes = rows * (k / 32) * 18; // q4_0: 18 bytes per 32 elements
    const w = allocator.alloc(u8, bytes) catch return p;
    defer allocator.free(w);
    fillWeights(.q4_0, w, k, rows);

    const x = allocator.alloc(f32, k) catch return p;
    defer allocator.free(x);
    for (x, 0..) |*v, i| {
        const q: i32 = @intCast((i * 7 + 3) % 17);
        v.* = @as(f32, @floatFromInt(q - 8)) * 0.125;
    }
    const out = allocator.alloc(f32, rows) catch return p;
    defer allocator.free(out);
    const xm = allocator.alloc(f32, m * k) catch return p;
    defer allocator.free(xm);
    for (0..m) |j| {
        for (x, 0..) |v, i| xm[j * k + i] = v * @as(f32, @floatFromInt(j + 1));
    }
    const outm = allocator.alloc(f32, m * rows) catch return p;
    defer allocator.free(outm);

    // The GEMV shader variants the grid declares for this shape. The probe
    // measures what the grid lists; it does not name variants.
    const q4_0_variants = catalogGpuVariants(@intFromEnum(grid.AtomicId.mat_vec), .q4_0, rows, k);
    const gate_up_variants = catalogGpuVariants(@intFromEnum(grid.AtomicId.gate_up_fused), .q4_0, rows, k);

    // Warm up shader modules and persistent buffers before timing.
    _ = self.matVec(.q4_0, w, rows, k, x, out);
    _ = self.matMulBatch(.q4_0, w, rows, k, m, xm, outm);
    for (q4_0_variants.slice()) |v| _ = self.matVec(Quant.q4_0.variant(v), w, rows, k, x, out);
    if (self.uploadResident(w)) |res| {
        for (q4_0_variants.slice()) |v| _ = self.matVecResident(res, Quant.q4_0.variant(v), rows, k, x, out);
        _ = self.matMulResident(res, .q4_0, rows, k, m, xm, outm);
        self.freeResident(res);
    }
    self.sync();

    p.gemv_streamed_gbps = probeRate(D, self, .q4_0, .streamed, w, rows, k, m, x, out, xm, outm);
    p.matmul_gbps = probeRate(D, self, .q4_0, .gemm, w, rows, k, m, x, out, xm, outm);
    // Measure the resident q4_0 variants **interleaved** so the RX 580's slow
    // memory-clock ramp cannot flatter whichever happens to be measured later
    // (the sequential plain/u4/sub order would). `plain` is one of them, so the
    // base `gemv_gbps` comes from the same fair pass.
    const q4_0_rates = probeResidentVariants(D, self, .q4_0, w, rows, k, x, out, q4_0_variants.slice());
    p.gemv_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.plain)];
    p.q4_0_u4_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.u4)];
    p.q4_0_sub_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.sub)];
    p.q4_0_un2_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.un2)];
    p.q4_0_un4_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.un4)];
    p.q4_0_w32_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.w32)];
    p.q4_0_w128_gbps = q4_0_rates[@intFromEnum(grid.GpuVariant.w128)];
    p.q4_0_variant = fastestVariant(q4_0_variants.slice(), &q4_0_rates);

    // Fused `gate_up` (two weights, one dispatch): the GPU's heaviest decode
    // op. Warm every catalogued variant, then measure them interleaved and
    // record the faster.
    const gu_act: GateUpAct = .gelu;
    if (self.uploadResident(w)) |rg| {
        defer self.freeResident(rg);
        if (self.uploadResident(w)) |ru| {
            defer self.freeResident(ru);
            for (gate_up_variants.slice()) |v| {
                if (self.matVecGateUpResidentBurstAsync(rg, ru, .q4_0, rows, k, x, gu_act, v, true)) |h| _ = self.waitResident(h, out, rows);
            }
            self.sync();
            const rates = probeGateUpRates(D, self, rg, ru, .q4_0, rows, k, x, out, gu_act, gate_up_variants.slice());
            p.gate_up_gbps = rates[@intFromEnum(grid.GpuVariant.plain)];
            p.gate_up_u4_gbps = rates[@intFromEnum(grid.GpuVariant.u4)];
            p.gate_up_variant = fastestVariant(gate_up_variants.slice(), &rates);
        }
    }

    // The q8_0 fused `gate_up` is the same spec-constant shader over the q8_0
    // block; measure its variants too (a distinct weight layout).
    const gate_up_q8_0_variants = catalogGpuVariants(@intFromEnum(grid.AtomicId.gate_up_fused), .q8_0, rows, k);
    const q8_bytes = rows * (k / 32) * 34;
    if (allocator.alloc(u8, q8_bytes)) |w8| {
        defer allocator.free(w8);
        fillWeights(.q8_0, w8, k, rows);
        if (self.uploadResident(w8)) |rg| {
            defer self.freeResident(rg);
            if (self.uploadResident(w8)) |ru| {
                defer self.freeResident(ru);
                for (gate_up_q8_0_variants.slice()) |v| {
                    if (self.matVecGateUpResidentBurstAsync(rg, ru, .q8_0, rows, k, x, gu_act, v, true)) |h| _ = self.waitResident(h, out, rows);
                }
                self.sync();
                const rates = probeGateUpRates(D, self, rg, ru, .q8_0, rows, k, x, out, gu_act, gate_up_q8_0_variants.slice());
                p.gate_up_q8_0_variant = fastestVariant(gate_up_q8_0_variants.slice(), &rates);
            }
        }
    } else |_| {}

    // The q4_1 fused `gate_up` is the same spec-constant shader over the q4_1
    // block (the `min` term); measure its variants too (a distinct layout).
    const gate_up_q4_1_variants = catalogGpuVariants(@intFromEnum(grid.AtomicId.gate_up_fused), .q4_1, rows, k);
    const q41_bytes = rows * (k / 32) * 20;
    if (allocator.alloc(u8, q41_bytes)) |w41| {
        defer allocator.free(w41);
        fillWeights(.q4_1, w41, k, rows);
        if (self.uploadResident(w41)) |rg| {
            defer self.freeResident(rg);
            if (self.uploadResident(w41)) |ru| {
                defer self.freeResident(ru);
                for (gate_up_q4_1_variants.slice()) |v| {
                    if (self.matVecGateUpResidentBurstAsync(rg, ru, .q4_1, rows, k, x, gu_act, v, true)) |h| _ = self.waitResident(h, out, rows);
                }
                self.sync();
                const rates = probeGateUpRates(D, self, rg, ru, .q4_1, rows, k, x, out, gu_act, gate_up_q4_1_variants.slice());
                p.gate_up_q4_1_variant = fastestVariant(gate_up_q4_1_variants.slice(), &rates);
            }
        }
    } else |_| {}

    // q4_K: the ALU-bound shader the unroll targets. Measure every catalogued
    // resident variant interleaved; the plan uses the faster per quant.
    const q4_k_probe = probeQuantVariants(D, self, allocator, .q4_k, rows, k, x, out);
    p.q4_k_gbps = q4_k_probe.rates[@intFromEnum(grid.GpuVariant.plain)];
    p.q4_k_u4_gbps = q4_k_probe.rates[@intFromEnum(grid.GpuVariant.u4)];
    p.q4_k_variant = q4_k_probe.variant;

    // Every other quant's variant grid: the same axes, each measured
    // interleaved and its winner recorded for the tuner.
    p.q4_1_variant = probeQuantVariants(D, self, allocator, .q4_1, rows, k, x, out).variant;
    p.q8_0_variant = probeQuantVariants(D, self, allocator, .q8_0, rows, k, x, out).variant;
    p.q6_k_variant = probeQuantVariants(D, self, allocator, .q6_k, rows, k, x, out).variant;
    p.f32_variant = probeQuantVariants(D, self, allocator, .f32, rows, k, x, out).variant;
    p.f16_variant = probeQuantVariants(D, self, allocator, .f16, rows, k, x, out).variant;
    p.bf16_variant = probeQuantVariants(D, self, allocator, .bf16, rows, k, x, out).variant;

    self.sync();
    return p;
}

const ProbeOp = enum { streamed, resident, gemm };

fn probeRate(
    comptime D: type,
    self: *D,
    quant: Quant,
    op: ProbeOp,
    w: []const u8,
    rows: usize,
    k: usize,
    m: usize,
    x: []const f32,
    out: []f32,
    xm: []const f32,
    outm: []f32,
) f64 {
    var res: ?*Resident = null;
    if (op == .resident) {
        res = self.uploadResident(w) orelse return 0;
    }
    defer if (res) |r| self.freeResident(r);

    var best: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < probe_samples) : (i += 1) {
        const t0 = probeNowNs();
        const ran = switch (op) {
            .streamed => self.matVec(quant, w, rows, k, x, out),
            .resident => self.matVecResident(res.?, quant, rows, k, x, out),
            .gemm => self.matMulBatch(quant, w, rows, k, m, xm, outm),
        };
        const t1 = probeNowNs();
        if (!ran) break;
        if (t1 > t0 and t1 - t0 < best) best = t1 - t0;
    }
    if (best == std.math.maxInt(u64) or best == 0) return 0;
    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();
    const bytes = rows * row_bytes;
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best));
}

/// Timed resident GEMV rates (min-of-N), one per catalogued shader variant of
/// `quant`, measured **interleaved** so the RX 580's slow memory-clock ramp
/// cannot flatter whichever variant happens to be measured later. This is the
/// fair comparison the reduction axis needs (a sequential plain/u4/sub order
/// would bias the last-measured variant up). The weight is uploaded once and
/// reused for every variant; every failure leaves the variant's rate `0`.
fn probeResidentVariants(
    comptime D: type,
    self: *D,
    quant: Quant,
    w: []const u8,
    rows: usize,
    k: usize,
    x: []const f32,
    out: []f32,
    variants: []const grid.GpuVariant,
) [grid.variant_slots]f64 {
    var rates = [_]f64{0} ** grid.variant_slots;
    const res = self.uploadResident(w) orelse return rates;
    defer self.freeResident(res);

    var best = [_]u64{std.math.maxInt(u64)} ** grid.variant_slots;
    var i: usize = 0;
    while (i < probe_samples) : (i += 1) {
        for (variants) |v| {
            const t0 = probeNowNs();
            const ran = self.matVecResident(res, quant.variant(v), rows, k, x, out);
            const t1 = probeNowNs();
            if (!ran) continue;
            const idx = @intFromEnum(v);
            if (t1 > t0 and t1 - t0 < best[idx]) best[idx] = t1 - t0;
        }
    }

    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();
    const bytes: f64 = @floatFromInt(rows * row_bytes);
    for (variants) |v| {
        const idx = @intFromEnum(v);
        if (best[idx] == std.math.maxInt(u64) or best[idx] == 0) continue;
        rates[idx] = bytes / @as(f64, @floatFromInt(best[idx]));
    }
    return rates;
}

/// Timed fused `gate_up` rates (min-of-N), one per catalogued shader variant:
/// the two resident weights `rg`/`ru` and their shared activation `x`, effective
/// GB/s over **both** weights' bytes (the fused op reads two per dispatch). The
/// variants are measured **interleaved** so the RX 580's slow memory-clock ramp
/// cannot flatter whichever is measured later (the plain/u4 ordering caveat the
/// single-tensor probe records).
fn probeGateUpRates(
    comptime D: type,
    self: *D,
    rg: *Resident,
    ru: *Resident,
    quant: Quant,
    rows: usize,
    k: usize,
    x: []const f32,
    out: []f32,
    act: GateUpAct,
    variants: []const grid.GpuVariant,
) [grid.variant_slots]f64 {
    var best = [_]u64{std.math.maxInt(u64)} ** grid.variant_slots;
    var i: usize = 0;
    while (i < probe_samples) : (i += 1) {
        for (variants) |v| {
            const t0 = probeNowNs();
            const h = self.matVecGateUpResidentBurstAsync(rg, ru, quant, rows, k, x, act, v, true) orelse return [_]f64{0} ** grid.variant_slots;
            if (!self.waitResident(h, out, rows)) return [_]f64{0} ** grid.variant_slots;
            const t1 = probeNowNs();
            const idx = @intFromEnum(v);
            if (t1 > t0 and t1 - t0 < best[idx]) best[idx] = t1 - t0;
        }
    }
    const row_bytes = (k / quant.blockElems()) * quant.blockBytes();
    const bytes: f64 = @floatFromInt(2 * rows * row_bytes);
    var rates = [_]f64{0} ** grid.variant_slots;
    for (variants) |v| {
        const idx = @intFromEnum(v);
        if (best[idx] == std.math.maxInt(u64) or best[idx] == 0) continue;
        rates[idx] = bytes / @as(f64, @floatFromInt(best[idx]));
    }
    return rates;
}

/// The base `Quant` a ggml type decodes as; the probe builds its synthetic
/// weights and its variant list from this.
fn quantFor(t: ggml.GGMLType) Quant {
    return switch (t) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q4_k => .q4_k,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => .q4_0,
    };
}

/// The probe result for one quant: the winning variant and its per-variant
/// resident rates.
pub const QuantProbe = struct {
    variant: grid.GpuVariant = .plain,
    rates: [grid.variant_slots]f64 = [_]f64{0} ** grid.variant_slots,
};

/// Measure one quant's catalogued resident GEMV variants **interleaved** and
/// return the winner plus the per-variant rates. A shape the quant cannot tile
/// (or an allocation failure) leaves `plain`. The k length is truncated to the
/// quant's block size first (the super-block quants need `k % 256 == 0`).
fn probeQuantVariants(
    comptime D: type,
    self: *D,
    allocator: std.mem.Allocator,
    t: ggml.GGMLType,
    rows: usize,
    k: usize,
    x: []const f32,
    out: []f32,
) QuantProbe {
    const q = quantFor(t);
    var result = QuantProbe{};
    const be = q.blockElems();
    const kq = k - k % be;
    if (kq == 0) return result;
    const bytes = std.math.mul(usize, rows, (kq / be) * q.blockBytes()) catch return result;
    const w = allocator.alloc(u8, bytes) catch return result;
    defer allocator.free(w);
    fillWeights(q, w, kq, rows);
    const variants = catalogGpuVariants(@intFromEnum(grid.AtomicId.mat_vec), t, rows, kq);
    for (variants.slice()) |v| _ = self.matVec(q.variant(v), w, rows, kq, x[0..kq], out);
    self.sync();
    result.rates = probeResidentVariants(D, self, q, w, rows, kq, x[0..kq], out, variants.slice());
    result.variant = fastestVariant(variants.slice(), &result.rates);
    return result;
}

/// The catalogued shader variants for one AtomicOp and base quant, in grid
/// order. The single source of the variant list: the probe enumerates the GPU
/// kernel grid (`gpu/kernels/grid.zig`) instead of a literal list, so a new
/// variant is a grid entry plus a shader.
pub const GpuVariantSet = struct {
    items: [8]grid.GpuVariant = undefined,
    len: usize = 0,

    pub fn slice(self: *const GpuVariantSet) []const grid.GpuVariant {
        return self.items[0..self.len];
    }
};

pub fn catalogGpuVariants(atomic_id: u16, t: ggml.GGMLType, rows: usize, k: usize) GpuVariantSet {
    var set = GpuVariantSet{};
    const code = @intFromEnum(t);
    for (grid.gpu_kernels) |kernel| {
        if (kernel.atomic_id != atomic_id) continue;
        if (kernel.applicable) |f| {
            if (!f(code, rows, k, 0)) continue;
        }
        const v = std.enums.fromInt(grid.GpuVariant, kernel.variant) orelse continue;
        if (set.len < set.items.len) {
            set.items[set.len] = v;
            set.len += 1;
        }
    }
    return set;
}

/// Argmax variant by measured rate, in grid order (strictly greater wins), so
/// an unmeasured set and any tie keep `plain`.
fn fastestVariant(variants: []const grid.GpuVariant, rates: *const [grid.variant_slots]f64) grid.GpuVariant {
    var best: grid.GpuVariant = .plain;
    var best_rate: f64 = 0;
    for (variants) |v| {
        const r = rates[@intFromEnum(v)];
        if (r > best_rate) {
            best_rate = r;
            best = v;
        }
    }
    return best;
}

fn writeF16(dst: []u8, v: f16) void {
    std.mem.writeInt(u16, dst[0..2], @bitCast(v), .little);
}

/// Fill `buf` with a structurally valid quantized tensor. For the block quants
/// the payload is random bytes with the f16 scale/min fields overwritten so no
/// block carries a NaN/Inf scale; for the unquantized types the whole buffer is
/// a deterministic finite pattern (random f32 bits could be NaN/Inf).
pub fn fillWeights(quant: Quant, buf: []u8, k: usize, rows: usize) void {
    const base = quant.base();
    const total = k * rows;
    switch (base) {
        .f32 => {
            for (0..total) |i| {
                const v: f32 = @as(f32, @floatFromInt(i % 13)) * 0.05 - 0.3;
                std.mem.writeInt(u32, buf[i * 4 ..][0..4], @bitCast(v), .little);
            }
            return;
        },
        .f16 => {
            for (0..total) |i| {
                const v: f16 = @floatCast(@as(f32, @floatFromInt(i % 13)) * 0.05 - 0.3);
                writeF16(buf[i * 2 ..], v);
            }
            return;
        },
        .bf16 => {
            for (0..total) |i| {
                const v: f32 = @as(f32, @floatFromInt(i % 13)) * 0.05 - 0.3;
                const bits: u32 = @bitCast(v);
                std.mem.writeInt(u16, buf[i * 2 ..][0..2], @truncate(bits >> 16), .little);
            }
            return;
        },
        else => {},
    }

    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    prng.random().bytes(buf);

    const row_blocks = k / base.blockElems();
    const row_bytes = row_blocks * base.blockBytes();
    for (0..rows) |r| {
        for (0..row_blocks) |b| {
            const off = r * row_bytes + b * base.blockBytes();
            switch (base) {
                .q4_0, .q8_0 => writeF16(buf[off..], 0.05),
                .q4_1 => {
                    writeF16(buf[off..], 0.05);
                    writeF16(buf[off + 2 ..], 0.02);
                },
                .q6_k => writeF16(buf[off + 208 ..], 0.02),
                .q4_k => {
                    writeF16(buf[off..], 0.05);
                    writeF16(buf[off + 2 ..], 0.02);
                },
                else => {},
            }
        }
    }
}

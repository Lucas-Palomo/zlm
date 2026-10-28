const std = @import("std");
const ggml = @import("../gguf/format.zig");
const plan = @import("../plan/root.zig");
const caps_mod = @import("caps.zig");

/// Effective host<->device staging bandwidth for a discrete (PCIe) GPU, GB/s.
/// The runtime path (map, copy into a host-visible buffer, submit, queue wait)
/// sits well below a PCIe link's peak, so this is a deliberately conservative
/// effective figure. Numerically GB/s == bytes/ns, so `bytes / PCIE_BW_GBPS` is
/// a duration in ns.
pub const PCIE_BW_GBPS: f64 = 8.0;

/// Conservative fp32 compute throughput assumed for a discrete GPU, TFLOP/s.
/// A decode GEMV does `2 * rows * k` FLOPs, negligible next to the weight
/// transfer; this only bounds the transfer-free (resident) case.
pub const GPU_TFLOPS: f64 = 3.0;

/// Largest single weight buffer the backend streams in one dispatch, bytes. A
/// weight larger than this is sliced by output rows (`Dispatch.tile`) so each
/// streamed slice fits. Mirrors the backend's streamed weight buffer sizing.
pub const STAGING_LIMIT_BYTES: u64 = 128 * 1024 * 1024;

/// Transfer-aware GPU offload decision for a single-token GEMV (decode).
/// Deterministic, from the device class, the op shape, its raw weight size and
/// residency (no timing): the plan records the outcome, the runtime re-decides
/// nothing.
///
/// - UMA (integrated/unified RAM) has no host<->device transfer penalty, so the
///   op is offloaded for any non-trivial shape (the shape gate).
/// - Discrete (PCIe) pays the weight transfer per streamed dispatch. The decode
///   model is DRAM-bound on both sides, so the CPU time is `length_bytes /
///   cpu_bw` and the GPU time is the transfer (zero when resident) plus the
///   compute plus one sync. A resident weight is transfer-free and is always
///   offloaded (the GPU reads it from VRAM, never on demand); a streamed one
///   reads the same bytes over a slower bus and loses (for a quantized GEMV the
///   CPU is always DRAM-bound, so no separate compute-bound exception is
///   needed).
///
/// Only the quants the backend implements are eligible; anything else stays on
/// the CPU.
pub fn decideOffload(
    is_uma: bool,
    ggml_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    length_bytes: u64,
    resident: bool,
    caps: caps_mod.Caps,
) bool {
    switch (ggml_type) {
        .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 => {},
        else => return false,
    }
    if (rows == 0 or k == 0 or length_bytes == 0) return false;
    if (is_uma) return rows >= 32 and k >= 32;

    // A VRAM-resident weight is transfer-free: the GPU reads it from VRAM with
    // no host transfer, so it is offloaded unconditionally (never on demand).
    // The measured cost gates only streamed (non-resident) ops.
    if (resident) return true;

    const cpu_bw = if (caps.dram_bw_gbps > 0) caps.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
    const cpu_ns = @as(f64, @floatFromInt(length_bytes)) / cpu_bw;

    // A measured device probe (`.zat` gpu coeffs, set at `tune --gpu`) gives the
    // real streamed per-token cost: the streamed rate includes the host<->device
    // transfer. Compare end-to-end and offload only when the device actually
    // wins.
    const measured_rate = caps.gpu.gemv_streamed_gbps;
    if (measured_rate > 0) {
        const gpu_ns = @as(f64, @floatFromInt(length_bytes)) / measured_rate;
        return gpu_ns < cpu_ns;
    }

    // Unmeasured (a CPU-only profile): keep the conservative constants.
    const transfer_ns = @as(f64, @floatFromInt(length_bytes)) / PCIE_BW_GBPS;

    const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(k));
    const compute_ns = flops / (GPU_TFLOPS * 1e12) * 1e9;
    const sync_ns = caps.dispatch_latency_ns;

    return transfer_ns + compute_ns + sync_ns < cpu_ns;
}

/// Transfer-aware GPU offload decision for a batched prefill projection (a
/// prompt-block GEMM). Deterministic, from the device class, the op shape and
/// residency (no timing): the plan records the outcome, the runtime re-decides
/// nothing.
///
/// A batched GEMM reads the weight once and reuses it across all `m` prompt
/// rows, so the host<->device transfer is amortized over the block (a discrete
/// GPU no longer pays one transfer per token). Only the quants with a batched
/// GEMM shader are eligible (`q6_k` has none); anything else stays on the
/// CPU. A resident weight is transfer-free and is always offloaded; for a
/// streamed weight, when the profile carries a measured GEMM rate the device's
/// amortized cost (`length_bytes / matmul_gbps`, transfer included) is compared
/// against the CPU's (`ceil(m/32)` weight reads at the DRAM rate) and the GPU is
/// used only when it wins; an unmeasured profile keeps the streamed shape gate.
pub fn decideOffloadMatMul(
    ggml_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    m: u64,
    length_bytes: u64,
    resident: bool,
    caps: caps_mod.Caps,
) bool {
    switch (ggml_type) {
        .q4_0, .q4_1, .q8_0, .q4_k => {},
        else => return false,
    }
    if (rows == 0 or k == 0) return false;

    // Resident implies offload: the GPU reads the weight from VRAM with no host
    // transfer. The measured GEMM cost gates only streamed ops.
    if (resident) return true;

    // Measured streamed device cost: the GEMM rate includes the weight
    // transfer. The CPU `matMulRows` reuses each weight read across up to 32
    // prompt rows, so its cost is `ceil(m/32)` weight reads at the DRAM rate;
    // the device is used only when its amortized cost is lower.
    if (caps.gpu.matmul_gbps > 0 and length_bytes > 0 and m > 0) {
        const cpu_bw = if (caps.dram_bw_gbps > 0) caps.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
        const reads: u64 = @max((m + 31) / 32, 1);
        const gpu_ns = @as(f64, @floatFromInt(length_bytes)) / caps.gpu.matmul_gbps;
        const cpu_ns = @as(f64, @floatFromInt(reads)) * @as(f64, @floatFromInt(length_bytes)) / cpu_bw;
        return gpu_ns < cpu_ns;
    }

    return m >= 2 and rows >= 32 and k >= 32;
}

/// Whether a fused `gate_up` two-weight GEMV may be offloaded. The device
/// implements the fused shader for q4_0, q4_1 and q8_0, and the runtime's fused
/// path reads both weights from VRAM, so both must be resident and share that
/// quant. Deterministic from the types, shape and residency (no timing), like
/// `decideOffload`.
pub fn decideOffloadGateUp(
    gate_type: ggml.GGMLType,
    up_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    length_bytes: u64,
    resident_both: bool,
) bool {
    if (gate_type != up_type) return false;
    if (gate_type != .q4_0 and gate_type != .q4_1 and gate_type != .q8_0) return false;
    if (rows == 0 or k == 0 or length_bytes == 0) return false;
    return resident_both;
}

/// Whether a fused three-weight MLP may be offloaded. The device runs it as
/// either the single-workgroup fused shader (UMA: the `ffn_mid` in shared
/// memory, all three weights on one block layout) or, on a discrete device,
/// **two multi-workgroup kernels** (the fused `gate_up` then the `down`) with
/// the `ffn_mid` in VRAM, where the `down` may keep its own GEMV quant. The
/// `gate`/`up` must share a fused quant (q4_0, q4_1 or q8_0) and all three be
/// resident. Deterministic from the types, shape and residency (no timing).
/// Unlike the two-weight `gate_up` the MLP is never split: the `ffn_mid` is
/// recomputed per row slice, so a split would duplicate the `gate`/`up` work; it
/// is whole-op `cpu` or `gpu`.
pub fn decideOffloadMlp(
    is_uma: bool,
    gate_type: ggml.GGMLType,
    up_type: ggml.GGMLType,
    down_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    length_bytes: u64,
    resident_all: bool,
) bool {
    if (!resident_all) return false;
    if (gate_type != up_type) return false;
    if (gate_type != .q4_0 and gate_type != .q4_1 and gate_type != .q8_0) return false;
    if (is_uma) {
        if (down_type != gate_type) return false;
    } else switch (down_type) {
        .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 => {},
        else => return false,
    }
    if (rows == 0 or k == 0 or length_bytes == 0) return false;
    return true;
}

/// The `gate_up`/`down` pair's device residency, so `mlpFusionPays` can tell
/// whether the pair would land on **different** devices on a discrete GPU (a
/// PCIe crossing for the `ffn_mid`).
pub const MlpResidency = struct {
    /// Both `gate` and `up` resident (the fused `gate_up` runs on the GPU).
    gate_up: bool = true,
    /// The `down` resident (the `ffn_down` runs on the GPU).
    down: bool = true,
};

/// Whether fusing a `gate_up` + its consuming `ffn_down` into one `mlp_fused`
/// **pays**. On a UMA device the fused op is one dispatch whose `ffn_mid` never
/// leaves the kernel, so it saves the pair's extra dispatch: the pair costs
/// `gate_up_ns + down_ns + 2*setup`, the fused op `fused_ns + setup`, so it pays
/// when `fused_ns < gate_up_ns + down_ns + setup` (both run at the same GEMV
/// rate). On a discrete device the fused op runs as the **two multi-workgroup
/// kernels** (`gate_up` + `down`, same rate), so it costs the same as the pair
/// when both are resident; it only pays when the pair would **cross devices**
/// (`res.gate_up != res.down`) — the atomic MLP keeps the whole op on one device
/// and the `ffn_mid` off the PCIe. When both are resident the separate `gate_up`
/// is already best (keep the pair, byte-identical plan); when neither is, both
/// run on the CPU (no crossing, no gain). Deterministic (no timing).
pub fn mlpFusionPays(
    is_uma: bool,
    gate_type: ggml.GGMLType,
    up_type: ggml.GGMLType,
    down_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    down_rows: u64,
    down_k: u64,
    gate_up_bytes: u64,
    down_bytes: u64,
    res: MlpResidency,
    caps: caps_mod.Caps,
) bool {
    const combined = gate_up_bytes + down_bytes;
    if (!decideOffloadMlp(is_uma, gate_type, up_type, down_type, rows, k, combined, true)) return false;
    const fused = costOptions(.mlp, is_uma, gate_type, up_type, down_type, rows, k, combined, true, caps);
    if (!std.math.isFinite(fused.gpu_ns)) return false;
    if (!is_uma) return res.gate_up != res.down;
    const pair = costOptions(.gate_up, is_uma, gate_type, up_type, down_type, rows, k, gate_up_bytes, true, caps);
    const down = costOptions(.matvec, is_uma, down_type, down_type, down_type, down_rows, down_k, down_bytes, true, caps);
    if (!std.math.isFinite(pair.gpu_ns) or !std.math.isFinite(down.gpu_ns)) return false;
    return fused.gpu_ns < pair.gpu_ns + down.gpu_ns + caps.dispatch_latency_ns;
}

/// Weight metadata the MLP fusion pass needs about one tensor.
pub const WeightMeta = struct {
    ggml_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    length: u64,
};

/// Whether `d` is a fused `gate_up` (its `atomic_id` is `gate_up_atomic`) and
/// `n` the `ffn_down` that consumes its `ffn_mid` output. The Composer emits
/// the pair adjacently (gemma4, qwen3), so the whole-MLP fusion merges two
/// neighbouring dispatches.
fn mlpPair(d: plan.Dispatch, n: plan.Dispatch, gate_up_atomic: u32) bool {
    if (d.op != .matvec or d.tensor2 == plan.NONE or d.tensor3 != plan.NONE) return false;
    if (d.atomic_id != gate_up_atomic) return false;
    if (n.op != .matvec or n.tensor2 != plan.NONE or n.tensor3 != plan.NONE) return false;
    if (d.out.kind != .buffer or n.in.kind != .buffer) return false;
    return d.out.ref == n.in.ref;
}

fn pairPays(
    d: plan.Dispatch,
    n: plan.Dispatch,
    weights: []const ?WeightMeta,
    resident: []const bool,
    is_uma: bool,
    caps: caps_mod.Caps,
) bool {
    if (d.tensor == plan.NONE or d.tensor2 == plan.NONE or n.tensor == plan.NONE) return false;
    if (d.tensor >= weights.len or d.tensor2 >= weights.len or n.tensor >= weights.len) return false;
    const g = weights[d.tensor] orelse return false;
    const u = weights[d.tensor2] orelse return false;
    const dn = weights[n.tensor] orelse return false;
    const res = MlpResidency{
        .gate_up = d.tensor < resident.len and d.tensor2 < resident.len and resident[d.tensor] and resident[d.tensor2],
        .down = n.tensor < resident.len and resident[n.tensor],
    };
    return mlpFusionPays(is_uma, g.ggml_type, u.ggml_type, dn.ggml_type, g.rows, g.k, dn.rows, dn.k, g.length + u.length, dn.length, res, caps);
}

/// Fuse each adjacent `gate_up` + consuming `ffn_down` pair into one
/// `mlp_fused` dispatch **only when `mlpFusionPays` says it pays**. Returns a
/// new slice (the caller owns it); the input is not modified. `weights` maps a
/// tensor index to its metadata (`null` for a non-2-D/absent tensor), and
/// `resident` the predicted VRAM-resident set of the **current** plan (so a
/// discrete pair that would cross devices fuses). The caller passes the
/// `gate_up_fused`/`mlp_fused` atomic ids so this module need not import the
/// kernel catalog.
pub fn fuseMlps(
    allocator: std.mem.Allocator,
    dispatch: []const plan.Dispatch,
    weights: []const ?WeightMeta,
    resident: []const bool,
    gate_up_atomic: u32,
    mlp_atomic: u32,
    is_uma: bool,
    caps: caps_mod.Caps,
) ![]plan.Dispatch {
    var list: std.ArrayList(plan.Dispatch) = .empty;
    errdefer list.deinit(allocator);
    var i: usize = 0;
    while (i < dispatch.len) {
        const d = dispatch[i];
        if (i + 1 < dispatch.len and mlpPair(d, dispatch[i + 1], gate_up_atomic) and
            pairPays(d, dispatch[i + 1], weights, resident, is_uma, caps))
        {
            const n = dispatch[i + 1];
            var m = d;
            m.tensor3 = n.tensor;
            m.atomic_id = mlp_atomic;
            m.out = n.out;
            m.out_len = n.out_len;
            try list.append(allocator, m);
            i += 2;
            continue;
        }
        try list.append(allocator, d);
        i += 1;
    }
    return list.toOwnedSlice(allocator);
}

/// The op class the device oracle is deciding (the eligibility + cost differ).
pub const Op = enum { matvec, matmul, gate_up, mlp };

/// The device a dispatch runs on (recorded in `Dispatch.device`).
pub const Device = enum(u8) { cpu = 0, gpu = 1, split = 2 };

/// The chain's running device byte load, fed back into the oracle so a split
/// balances the **global** totals — not just one op's. The CPU slices and the
/// streamed weights share one memory bandwidth; a split that ignores what the
/// CPU already carries over-loads it.
pub const Load = struct {
    /// Bytes the GPU reads per token (the resident weights + the GPU slices).
    gpu_bytes: u64 = 0,
    /// Bytes the CPU reads per token (the CPU slices + the streamed weights).
    cpu_bytes: u64 = 0,
};

/// The oracle's verdict for one op.
pub const Verdict = struct {
    device: Device,
    /// GPU output rows when `device == .split` (the GPU takes `[0, gpu_rows)`,
    /// the CPU `[gpu_rows, m)`); `0` otherwise.
    gpu_rows: u64 = 0,
    /// Bytes this op leaves on each device, so the caller can update its `Load`.
    gpu_bytes: u64 = 0,
    cpu_bytes: u64 = 0,
};

/// The **device oracle**: the single AOT decision for **every** op — `cpu`,
/// `gpu`, or `split` (the CPU + the GPU over disjoint output rows). It replaces
/// the per-path binary `decideOffload*` calls with one entry point; the per-kind
/// cost is unchanged (a resident GEMV always offloads, a streamed one only when
/// it wins, and so on), and the chain's profile — `cpu`, `gpu`, `split`,
/// `mixed` — emerges from the verdicts.
///
/// `load` is the chain's running byte load. It does not change the `cpu`/`gpu`
/// verdict, but a **split** is sized against it: `g` is the GPU's byte share
/// that balances `(G + g*r)/gpu_rate = (C + (m-g)*r)/cpu_rate`. With an empty
/// load that reduces to the per-op balance `g/m = gpu_rate/(gpu_rate+cpu_rate)`.
pub fn decideDevice(
    op: Op,
    is_uma: bool,
    gate_type: ggml.GGMLType,
    up_type: ggml.GGMLType,
    down_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    m: u64,
    length_bytes: u64,
    resident: bool,
    load: Load,
    caps: caps_mod.Caps,
) Verdict {
    const offload = switch (op) {
        .matvec => decideOffload(is_uma, gate_type, rows, k, length_bytes, resident, caps),
        .matmul => decideOffloadMatMul(gate_type, rows, k, m, length_bytes, resident, caps),
        .gate_up => decideOffloadGateUp(gate_type, up_type, rows, k, length_bytes, resident),
        .mlp => decideOffloadMlp(is_uma, gate_type, up_type, down_type, rows, k, length_bytes, resident),
    };
    if (!offload) return .{ .device = .cpu, .cpu_bytes = length_bytes };
    // A resident decode projection can share the device with the CPU: a plain
    // GEMV (one weight) or a fused `gate_up` (two). (A fused MLP recomputes its
    // intermediate per row slice, so it is never split; a prefill GEMM would
    // need its own row slicing; not yet.)
    if (resident and !is_uma and (op == .matvec or op == .gate_up)) {
        const g = splitRows(rows, length_bytes, load, caps);
        if (g > 0) {
            const row_bytes = length_bytes / rows;
            const gpu_bytes = g * row_bytes;
            return .{ .device = .split, .gpu_rows = g, .gpu_bytes = gpu_bytes, .cpu_bytes = length_bytes - gpu_bytes };
        }
    }
    return .{ .device = .gpu, .gpu_bytes = length_bytes };
}

/// The GPU row count of a split: the slices run concurrently, so the optimum
/// balances their global load, `(gpu + g*r)/gpu_rate = (cpu + (m-g)*r)/cpu_rate`,
/// rounded down to the GEMV row tile (8). `0` = do not split.
///
/// Solving for the GPU's byte share `Xg`:
/// `Xg = (gpu_rate*(C + B) - cpu_rate*G) / (gpu_rate + cpu_rate)`, `g = Xg/r`.
pub fn splitRows(m: u64, length_bytes: u64, load: Load, caps: caps_mod.Caps) u64 {
    const gpu_rate = caps.gpu.gemv_gbps;
    const cpu_rate = if (caps.dram_bw_gbps > 0) caps.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
    if (gpu_rate <= 0 or cpu_rate <= 0 or m < 16 or length_bytes == 0) return 0;
    const row_bytes = length_bytes / m;
    if (row_bytes == 0) return 0;

    const b = @as(f64, @floatFromInt(length_bytes));
    const g_bytes = @as(f64, @floatFromInt(load.gpu_bytes));
    const c_bytes = @as(f64, @floatFromInt(load.cpu_bytes));
    const xg = (gpu_rate * (c_bytes + b) - cpu_rate * g_bytes) / (gpu_rate + cpu_rate);
    if (xg <= 0) return 0;
    var g: u64 = @intFromFloat(xg / @as(f64, @floatFromInt(row_bytes)));
    g -= g % 8;
    if (g == 0 or g >= m) return 0;
    return g;
}

// ---- the chain scheduler (`gpu/11`) ----

/// One op's cost on each device, as the oracle priced it — the chain
/// scheduler's input.
pub const Options = struct {
    /// ns to compute the whole op on the CPU.
    cpu_ns: f64,
    /// ns on the GPU (resident: `bytes/gpu_rate`; streamed: the measured rate).
    gpu_ns: f64,
    /// ns when split across both (the max of the two slices); `inf` = no split.
    split_ns: f64 = std.math.inf(f64),
    /// GPU output rows when a split is available.
    gpu_rows: u64 = 0,
    /// The **RAM the residency spends**, indexed by device (cpu/gpu/split): the
    /// whole weight on the CPU, `0` on the GPU (freed — the KV grows), the CPU
    /// tail on a split.
    ram: [3]u64 = .{ 0, 0, 0 },
    /// The **VRAM the resident set spends**, indexed by device.
    vram: [3]u64 = .{ 0, 0, 0 },
};

fn cpuRate(caps: caps_mod.Caps) f64 {
    return if (caps.dram_bw_gbps > 0) caps.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
}

/// The GPU cost of an op: a resident weight reads VRAM (`gemv_gbps`), a streamed
/// one pays the transfer (`gemv_streamed_gbps`), a batched GEMM amortizes it
/// (`matmul_gbps`). `inf` when the device has no measured rate.
pub fn gpuCostNs(op: Op, length_bytes: u64, resident: bool, caps: caps_mod.Caps) f64 {
    const b: f64 = @floatFromInt(length_bytes);
    if (resident and caps.gpu.gemv_gbps > 0) return b / caps.gpu.gemv_gbps;
    if (op == .matmul and caps.gpu.matmul_gbps > 0) return b / caps.gpu.matmul_gbps;
    if (caps.gpu.gemv_streamed_gbps > 0) return b / caps.gpu.gemv_streamed_gbps;
    return std.math.inf(f64);
}

/// Price one op for the chain scheduler: the CPU cost, the GPU cost, and (for a
/// resident, row-parallel weight) the split cost at the balanced point. Mirrors
/// `decideDevice`'s eligibility + rates, but returns the costs so the scheduler
/// can trade them against the switch/setup.
pub fn costOptions(
    op: Op,
    is_uma: bool,
    gate_type: ggml.GGMLType,
    up_type: ggml.GGMLType,
    down_type: ggml.GGMLType,
    rows: u64,
    k: u64,
    length_bytes: u64,
    resident: bool,
    caps: caps_mod.Caps,
) Options {
    const cpu_ns = @as(f64, @floatFromInt(length_bytes)) / cpuRate(caps);
    const fused_quant = gate_type == .q4_0 or gate_type == .q4_1 or gate_type == .q8_0;
    const eligible = switch (op) {
        .matvec => switch (gate_type) {
            .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 => true,
            else => false,
        },
        .matmul => switch (gate_type) {
            .q4_0, .q4_1, .q8_0 => true,
            else => false,
        },
        .gate_up => gate_type == up_type and fused_quant,
        .mlp => decideOffloadMlp(is_uma, gate_type, up_type, down_type, rows, k, length_bytes, true),
    };
    if (!eligible or rows == 0 or k == 0 or length_bytes == 0) return .{ .cpu_ns = cpu_ns, .gpu_ns = std.math.inf(f64), .ram = .{ length_bytes, 0, 0 } };

    var o = Options{
        .cpu_ns = cpu_ns,
        .gpu_ns = gpuCostNs(op, length_bytes, resident, caps),
        // The CPU reads the whole weight. A **resident** GPU op frees the RAM
        // (the weight lives in VRAM); a **streamed** one keeps the weight in the
        // RAM (it reads it over PCIe), so it does not free the budget — the RAM
        // shadow price must not prefer it. The VRAM is spent only when resident.
        .ram = .{ length_bytes, if (resident) 0 else length_bytes, 0 },
        .vram = .{ 0, if (resident) length_bytes else 0, 0 },
    };
    if (resident and !is_uma and (op == .matvec or op == .gate_up)) {
        const g = splitRows(rows, length_bytes, .{}, caps);
        if (g > 0) {
            const gpu_rate = caps.gpu.gemv_gbps;
            const gpu_b = g * (length_bytes / rows);
            o.gpu_rows = g;
            // A split shares the device with the CPU, but the two memories
            // **contend**: the measured `split_efficiency` (< 1) makes the split
            // rate a fraction of the GPU's, so a split loses to the whole-GPU op.
            // `1` (unmeasured) is the tie; `> 1` is a machine where the bands add.
            const eff = if (caps.gpu.split_efficiency > 0) caps.gpu.split_efficiency else 1.0;
            o.split_ns = @as(f64, @floatFromInt(length_bytes)) / (gpu_rate * eff);
            // A split keeps the CPU tail in the RAM and the GPU rows in the VRAM.
            o.ram[2] = length_bytes - gpu_b;
            o.vram[2] = gpu_b;
        }
    }
    return o;
}

/// The **chain scheduler** (`gpu/11`): a Viterbi DP over the dispatch chain.
/// Given each op's per-device cost it picks a device per op, minimizing the
/// makespan plus the **switch** cost (paid on a device change) and the **setup**
/// cost (a GPU op's staging + submit). The state is the *previous* device, so a
/// crossing is paid **once** — a per-op decision would flip for a marginal
/// saving, the chain does not. The DP is exact for a linear chain (no windowed
/// heuristic: the cost-to-go *is* the look-ahead).
///
/// Writes the chosen device per op into `out` (same length). `scratch` must hold
/// `3 * opts.len` bytes (the DP back-pointers). `ram_ns_per_byte` /
/// `vram_ns_per_byte` are the **shadow prices** of the residency budgets: the
/// scheduler pays them for the RAM/VRAM an op's device leaves spent, so a whole
/// GPU op (which frees the RAM) can beat a marginally cheaper split.
pub fn scheduleDevices(
    opts: []const Options,
    setup_ns: f64,
    switch_ns: f64,
    ram_ns_per_byte: f64,
    vram_ns_per_byte: f64,
    out: []Device,
    scratch: []u8,
) void {
    const n = opts.len;
    if (n == 0 or out.len < n or scratch.len < n * 3) return;
    const devices = [_]Device{ .cpu, .gpu, .split };
    const INF = std.math.inf(f64);

    var prev = [_]f64{ 0, 0, 0 };
    var cur = [_]f64{ INF, INF, INF };
    for (0..n) |i| {
        for (devices, 0..) |D, di| {
            const oc = switch (D) {
                .cpu => opts[i].cpu_ns,
                .gpu => opts[i].gpu_ns,
                .split => opts[i].split_ns,
            };
            if (oc == INF or oc != oc) {
                cur[di] = INF;
                scratch[i * 3 + di] = @intCast(di);
                continue;
            }
            const setup = if (D == .cpu) 0.0 else setup_ns;
            const penalty = ram_ns_per_byte * @as(f64, @floatFromInt(opts[i].ram[di])) +
                vram_ns_per_byte * @as(f64, @floatFromInt(opts[i].vram[di]));
            var best = INF;
            var bp: u8 = @intCast(di);
            for (devices, 0..) |P, pi| {
                const sw = if (i == 0 or P == D) 0.0 else switch_ns;
                const v = prev[pi] + sw;
                if (v < best) {
                    best = v;
                    bp = @intCast(pi);
                }
            }
            cur[di] = best + oc + setup + penalty;
            scratch[i * 3 + di] = bp;
        }
        prev = cur;
    }

    var best_di: usize = 0;
    for (devices, 0..) |_, di| {
        if (prev[di] < prev[best_di]) best_di = di;
    }
    var i = n;
    var d = best_di;
    while (i > 0) {
        i -= 1;
        out[i] = devices[d];
        d = scratch[i * 3 + d];
    }
}

test "chain scheduler: sticky runs, no cheap ping-pong" {
    const testing = std.testing;
    var out: [3]Device = undefined;
    var scratch: [9]u8 = undefined;

    // A cheap-on-CPU op between two GPU ops. A big switch keeps the run (sticky);
    // a small one lets the middle op cross to the CPU.
    const opts = [_]Options{
        .{ .cpu_ns = 20, .gpu_ns = 10 },
        .{ .cpu_ns = 1, .gpu_ns = 10 },
        .{ .cpu_ns = 20, .gpu_ns = 10 },
    };
    scheduleDevices(&opts, 0, 10, 0, 0, &out, &scratch);
    try testing.expectEqual(Device.gpu, out[0]);
    try testing.expectEqual(Device.gpu, out[1]);
    try testing.expectEqual(Device.gpu, out[2]);

    scheduleDevices(&opts, 0, 2, 0, 0, &out, &scratch);
    try testing.expectEqual(Device.gpu, out[0]);
    try testing.expectEqual(Device.cpu, out[1]);
    try testing.expectEqual(Device.gpu, out[2]);

    // The look-ahead: two CPU-cheap ops then a GPU-heavy one; the DP frees the
    // CPU for the cheap pair and crosses once for the heavy op.
    const lookahead = [_]Options{
        .{ .cpu_ns = 1, .gpu_ns = 10 },
        .{ .cpu_ns = 1, .gpu_ns = 10 },
        .{ .cpu_ns = 100, .gpu_ns = 20 },
    };
    scheduleDevices(&lookahead, 0, 5, 0, 0, &out, &scratch);
    try testing.expectEqual(Device.cpu, out[0]);
    try testing.expectEqual(Device.cpu, out[1]);
    try testing.expectEqual(Device.gpu, out[2]);
}

test "chain scheduler: the RAM price keeps a weight resident (no gratuitous split)" {
    const testing = std.testing;
    var out: [1]Device = undefined;
    var scratch: [3]u8 = undefined;
    // A resident op whose split is cheaper, but the split keeps its CPU tail in
    // the RAM (500 B) while the whole-GPU op frees it (0 B). A positive RAM price
    // flips it to the whole-GPU op — the "empty the RAM" policy.
    const opts = [_]Options{
        .{
            .cpu_ns = 100,
            .gpu_ns = 30,
            .split_ns = 20,
            .gpu_rows = 1024,
            .ram = .{ 1000, 0, 500 },
            .vram = .{ 0, 1000, 500 },
        },
    };
    scheduleDevices(&opts, 0, 0, 0, 0, &out, &scratch);
    try testing.expectEqual(Device.split, out[0]);
    // 0.05 ns/byte: the split's 500 B cost 25 ns, so the whole-GPU op wins.
    scheduleDevices(&opts, 0, 0, 0.05, 0, &out, &scratch);
    try testing.expectEqual(Device.gpu, out[0]);
}

test "cost: a streamed GPU op keeps its RAM (the shadow price does not push it off the CPU)" {
    const testing = std.testing;
    const bytes: u64 = 27 << 20; // a q8_0 projection
    const caps = caps_mod.Caps{
        .dram_bw_gbps = 35.0,
        .gpu = .{ .gemv_gbps = 20.0, .gemv_streamed_gbps = 5.0 },
    };

    // Streamed: the weight stays in the RAM (not freed) and does not occupy VRAM.
    const streamed = costOptions(.matvec, false, .q8_0, .q8_0, .q8_0, 5120, 5120, bytes, false, caps);
    try testing.expectEqual(bytes, streamed.ram[1]);
    try testing.expectEqual(@as(u64, 0), streamed.vram[1]);

    // Resident: the GPU frees the RAM and occupies the VRAM.
    const resident = costOptions(.matvec, false, .q8_0, .q8_0, .q8_0, 5120, 5120, bytes, true, caps);
    try testing.expectEqual(@as(u64, 0), resident.ram[1]);
    try testing.expectEqual(bytes, resident.vram[1]);

    // With a RAM shadow price the streamed op is not preferred over the CPU: both
    // keep the same RAM, and the streamed device cost is higher, so the CPU wins.
    var out: [1]Device = undefined;
    var scratch: [3]u8 = undefined;
    const opts = [_]Options{streamed};
    scheduleDevices(&opts, 0, 0, 1.0, 0, &out, &scratch);
    try testing.expectEqual(Device.cpu, out[0]);

    // A fused `gate_up` is eligible for q8_0 too (the q8_0 fused shader ships),
    // so the scheduler can price it on the device.
    const gu = costOptions(.gate_up, false, .q8_0, .q8_0, .q8_0, 5120, 5120, bytes, true, caps);
    try testing.expect(std.math.isFinite(gu.gpu_ns));

    // A fused three-weight MLP is eligible for q8_0 too (the q8_0 fused shader
    // ships, and the two-kernel form runs on a discrete device); all three
    // weights must share the quant.
    const mlp = costOptions(.mlp, true, .q8_0, .q8_0, .q8_0, 5120, 5120, bytes, true, caps);
    try testing.expect(std.math.isFinite(mlp.gpu_ns));
    const mlp_mixed = costOptions(.mlp, true, .q8_0, .q8_0, .q4_0, 5120, 5120, bytes, true, caps);
    try testing.expect(!std.math.isFinite(mlp_mixed.gpu_ns));
    const mlp_discrete = costOptions(.mlp, false, .q8_0, .q8_0, .q8_0, 5120, 5120, bytes, true, caps);
    try testing.expect(std.math.isFinite(mlp_discrete.gpu_ns));
    const mlp_discrete_mixed = costOptions(.mlp, false, .q8_0, .q8_0, .q4_0, 5120, 5120, bytes, true, caps);
    try testing.expect(std.math.isFinite(mlp_discrete_mixed.gpu_ns));
}

/// Rows per GPU slice when a streamed weight exceeds `STAGING_LIMIT_BYTES`
/// (`Dispatch.tile`); `0` means the whole weight. Only streamed (non-resident)
/// weights are sliced: a resident weight already lives in one VRAM buffer and
/// is never re-staged. `rows` is the output width, `length_bytes` the raw weight
/// size; the floor division keeps each slice within the staging limit.
pub fn tileRows(rows: u64, length_bytes: u64, resident: bool) u16 {
    if (resident or rows == 0 or length_bytes <= STAGING_LIMIT_BYTES) return 0;
    const t = (rows * STAGING_LIMIT_BYTES) / length_bytes;
    if (t == 0 or t >= rows) return 0;
    return @intCast(@min(t, std.math.maxInt(u16)));
}

/// Whether a weight tensor is a candidate for the VRAM-resident set: a
/// supported quant and a non-degenerate shape the device can actually run.
/// Residency makes a discrete offload profitable regardless of the per-token
/// payload, so the payload gate is not applied here (only the quant + shape
/// gate, before residency).
pub fn eligibleResident(is_uma: bool, ggml_type: ggml.GGMLType, rows: u64, k: u64) bool {
    switch (ggml_type) {
        .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 => {},
        else => return false,
    }
    if (rows == 0 or k == 0) return false;
    if (is_uma) return rows >= 32 and k >= 32;
    return true;
}

/// One residency candidate: a tensor index, its raw weight bytes and the
/// per-token transfer it saves when resident. `value` is the weight's decode
/// GEMV payload (`length`); it is `0` for a weight only reached by the batched
/// prefill GEMM, whose transfer is amortized over the block and which therefore
/// fills the residual budget after the per-token weights.
///
/// A fused `gate_up` GEMV is modeled as **one** candidate carrying both tensor
/// indices (`tensor_index` = gate, `tensor_index2` = up): `length`/`value` are
/// the combined bytes, so the packer places the pair atomically and charges
/// both buffers against the budget. A fused MLP is one candidate carrying all
/// three (`tensor_index` = gate, `tensor_index2` = up, `tensor_index3` = down).
/// `selectResident` writes only `tensor_index`; the caller (`tune`) expands a
/// chosen candidate by also keeping the partner(s) resident.
/// `tensor_index2`/`tensor_index3 == NONE` for a single-weight candidate.
pub const Candidate = struct {
    tensor_index: u32,
    length: u64,
    value: u64 = 0,
    tensor_index2: u32 = plan.NONE,
    tensor_index3: u32 = plan.NONE,
};

/// Saturating greedy packer over resident weights. Ranked best-first by the
/// per-token transfer saved (`value`), then by size (`length`), then by
/// `tensor_index` for determinism. Packing continues past a candidate that does
/// not fit, so the set saturates `vram_budget -| VRAM_DRIVER_HEADROOM`: the
/// per-token weights are placed first (the ROI pass) and the amortized prefill
/// weights fill the residual budget (the fill pass). The objective is a full
/// budget, not "only what clearly wins". Writes the chosen tensor indices into
/// `out` (which must hold at least `candidates.len` entries) and returns how
/// many were chosen. A budget that leaves no room after the headroom yields an
/// empty set (the GPU degrades, the weights never do).
pub fn selectResident(candidates: []const Candidate, vram_budget: u64, out: []u32) usize {
    const budget = vram_budget -| plan.VRAM_DRIVER_HEADROOM;
    if (budget == 0 or candidates.len == 0 or out.len < candidates.len) return 0;

    const order = out[0..candidates.len];
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, candidates, candidateLess);

    var used: u64 = 0;
    var count: usize = 0;
    for (order) |ci| {
        const c = candidates[ci];
        if (c.length == 0) continue;
        if (used + c.length > budget) continue;
        used += c.length;
        order[count] = c.tensor_index;
        count += 1;
    }
    return count;
}

fn candidateLess(candidates: []const Candidate, a: u32, b: u32) bool {
    const ca = candidates[a];
    const cb = candidates[b];
    if (ca.value != cb.value) return ca.value > cb.value;
    if (ca.length != cb.length) return ca.length > cb.length;
    return ca.tensor_index < cb.tensor_index;
}

test "transfer-aware offload: quant gate, shape gate and discrete transfer" {
    const testing = std.testing;
    const caps = caps_mod.Caps{ .dram_bw_gbps = 50.0, .dispatch_latency_ns = 100.0 };

    // Unsupported quant and degenerate shapes never offload.
    try testing.expect(!decideOffload(false, .q5_k, 4096, 2048, 4 << 20, true, caps));
    try testing.expect(!decideOffload(false, .q4_0, 0, 2048, 4 << 20, true, caps));
    try testing.expect(!decideOffload(false, .q4_0, 4096, 2048, 0, true, caps));

    // The secondary-scope quants are eligible: resident implies offload.
    try testing.expect(decideOffload(false, .q4_k, 262144, 1536, 226 << 20, true, caps));
    try testing.expect(decideOffload(false, .f32, 256, 1536, 1 << 20, true, caps));
    try testing.expect(decideOffload(false, .bf16, 8960, 1536, 27 << 20, true, caps));
    try testing.expect(decideOffload(false, .f16, 256, 1536, 1 << 20, true, caps));

    // UMA: no transfer, so only the shape gate decides (residency is irrelevant).
    try testing.expect(decideOffload(true, .q4_0, 32, 32, 1, false, caps));
    try testing.expect(!decideOffload(true, .q4_0, 16, 16, 1 << 30, true, caps));

    // Discrete, large payload: a resident weight has no per-token transfer and
    // wins; the same weight streamed over PCIe reads the bytes over a slower
    // bus and loses to the CPU's RAM read.
    const bytes: u64 = 4_718_592; // rows=4096, k=2048, q4_0
    try testing.expect(decideOffload(false, .q4_0, 4096, 2048, bytes, true, caps));
    try testing.expect(!decideOffload(false, .q4_0, 4096, 2048, bytes, false, caps));
}

test "measured offload cost: resident always offloads, streamed only when it wins" {
    const testing = std.testing;
    const bytes: u64 = 4_718_592; // rows=4096, k=2048, q4_0

    // A VRAM-resident weight is transfer-free, so it is offloaded even when the
    // device's raw GEMV rate is slower than the CPU's; the streamed op loses.
    const slow_gpu = caps_mod.Caps{
        .dram_bw_gbps = 25.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 2.0, .gemv_streamed_gbps = 1.0 },
    };
    try testing.expect(decideOffload(false, .q4_0, 4096, 2048, bytes, true, slow_gpu));
    try testing.expect(!decideOffload(false, .q4_0, 4096, 2048, bytes, false, slow_gpu));

    // A device that beats the CPU resident but not streamed.
    const fast_res = caps_mod.Caps{
        .dram_bw_gbps = 25.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 50.0, .gemv_streamed_gbps = 1.0 },
    };
    try testing.expect(decideOffload(false, .q4_0, 4096, 2048, bytes, true, fast_res));
    try testing.expect(!decideOffload(false, .q4_0, 4096, 2048, bytes, false, fast_res));

    // The measured path gates the streamed op: a fast CPU bus rejects it even
    // with a measured device rate, while the resident op still offloads.
    const fallback = caps_mod.Caps{ .dram_bw_gbps = 50.0, .dispatch_latency_ns = 100.0 };
    try testing.expect(decideOffload(false, .q4_0, 4096, 2048, bytes, true, fallback));
    const measured = caps_mod.Caps{
        .dram_bw_gbps = 50.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 10.0, .gemv_streamed_gbps = 5.0 },
    };
    try testing.expect(decideOffload(false, .q4_0, 4096, 2048, bytes, true, measured));
    try testing.expect(!decideOffload(false, .q4_0, 4096, 2048, bytes, false, measured));
}

test "transfer-aware offload is a pure function of its inputs" {
    const testing = std.testing;
    const caps = caps_mod.Caps{ .dram_bw_gbps = 32.0, .dispatch_latency_ns = 250.0 };
    const a = decideOffload(false, .q8_0, 1024, 1024, 1 << 20, false, caps);
    const b = decideOffload(false, .q8_0, 1024, 1024, 1 << 20, false, caps);
    try testing.expectEqual(a, b);

    // A faster CPU bus makes the same streamed op even less attractive.
    const faster = caps_mod.Caps{ .dram_bw_gbps = 64.0, .dispatch_latency_ns = 250.0 };
    try testing.expect(!decideOffload(false, .q8_0, 1024, 1024, 1 << 20, false, faster));
}

test "batched-matmul offload: quant gate, block size, shape gate, residency" {
    const testing = std.testing;
    const nb = 4_718_592; // rows=4096, k=2048, q4_0

    // Only the quants with a batched GEMM shader; q6_k has none.
    try testing.expect(!decideOffloadMatMul(.q6_k, 4096, 2048, 8, nb, true, .{}));
    try testing.expect(!decideOffloadMatMul(.f32, 4096, 2048, 8, nb, true, .{}));

    // A single-row block (or degenerate shape) is not a batch.
    try testing.expect(!decideOffloadMatMul(.q4_0, 4096, 2048, 1, nb, false, .{}));
    try testing.expect(!decideOffloadMatMul(.q4_0, 16, 16, 8, nb, false, .{}));
    try testing.expect(!decideOffloadMatMul(.q4_0, 0, 2048, 8, nb, false, .{}));

    // A multi-row block clearing the shape gate is offloaded; residency wins
    // even below the gate.
    try testing.expect(decideOffloadMatMul(.q4_0, 4096, 2048, 8, nb, false, .{}));
    try testing.expect(decideOffloadMatMul(.q8_0, 32, 32, 2, nb, false, .{}));
    try testing.expect(decideOffloadMatMul(.q4_1, 16, 16, 1, nb, true, .{}));

    // Pure function of its inputs.
    try testing.expectEqual(
        decideOffloadMatMul(.q4_0, 1024, 1024, 4, nb, false, .{}),
        decideOffloadMatMul(.q4_0, 1024, 1024, 4, nb, false, .{}),
    );

    // Measured device: offload only when the amortized device cost beats the CPU
    // reading the whole block. A fast CPU bus rejects even a multi-row block; a
    // slow one accepts it.
    const slow_cpu = caps_mod.Caps{ .dram_bw_gbps = 5.0, .gpu = .{ .matmul_gbps = 15.0 } };
    const fast_cpu = caps_mod.Caps{ .dram_bw_gbps = 60.0, .gpu = .{ .matmul_gbps = 15.0 } };
    try testing.expect(decideOffloadMatMul(.q4_0, 4096, 2048, 2, nb, false, slow_cpu));
    try testing.expect(!decideOffloadMatMul(.q4_0, 4096, 2048, 2, nb, false, fast_cpu));
}

test "tile rows: only streamed weights past the staging limit are sliced" {
    const testing = std.testing;
    const limit = STAGING_LIMIT_BYTES;

    // Resident, at/below the limit, or non-degenerate: no slicing.
    try testing.expectEqual(@as(u16, 0), tileRows(1024, 2 * limit, true));
    try testing.expectEqual(@as(u16, 0), tileRows(1024, limit, false));
    try testing.expectEqual(@as(u16, 0), tileRows(0, 2 * limit, false));
    // A single row cannot be sliced below the limit.
    try testing.expectEqual(@as(u16, 0), tileRows(1, 2 * limit, false));

    // Twice the limit -> half the rows per slice, each slice exactly the limit.
    try testing.expectEqual(@as(u16, 512), tileRows(1024, 2 * limit, false));
    // Floor division keeps each slice within the limit.
    try testing.expectEqual(@as(u16, 984), tileRows(1000, limit + limit / 65, false));
}

test "resident candidate gate: supported quant, non-degenerate shape" {
    const testing = std.testing;
    try testing.expect(eligibleResident(false, .q4_0, 64, 64));
    try testing.expect(eligibleResident(false, .q6_k, 1, 1));
    try testing.expect(eligibleResident(false, .q4_k, 262144, 1536));
    try testing.expect(eligibleResident(false, .f32, 256, 1536));
    try testing.expect(eligibleResident(false, .bf16, 8960, 1536));
    try testing.expect(!eligibleResident(false, .q5_k, 4096, 4096));
    try testing.expect(!eligibleResident(false, .q4_0, 0, 64));
    try testing.expect(eligibleResident(true, .q4_0, 32, 32));
    try testing.expect(!eligibleResident(true, .q4_0, 16, 16));
}

test "fused gate_up offload: q4_0/q4_1/q8_0, both weights resident" {
    const testing = std.testing;
    try testing.expect(decideOffloadGateUp(.q4_0, .q4_0, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadGateUp(.q4_1, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadGateUp(.q8_0, .q8_0, 4096, 2048, 1 << 20, true));
    // Both weights must share a fused quant and both be resident.
    try testing.expect(!decideOffloadGateUp(.q4_1, .q4_0, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadGateUp(.q4_0, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadGateUp(.q4_0, .q4_0, 4096, 2048, 1 << 20, false));
    // A quant with no fused shader is rejected.
    try testing.expect(!decideOffloadGateUp(.q6_k, .q6_k, 4096, 2048, 1 << 20, true));
    // Degenerate shape.
    try testing.expect(!decideOffloadGateUp(.q4_0, .q4_0, 0, 2048, 1 << 20, true));
    // Deterministic pure function.
    try testing.expectEqual(
        decideOffloadGateUp(.q4_0, .q4_0, 1024, 1024, 1 << 20, true),
        decideOffloadGateUp(.q4_0, .q4_0, 1024, 1024, 1 << 20, true),
    );
}

test "fused mlp offload: q4_0/q4_1/q8_0, all three weights resident" {
    const testing = std.testing;
    try testing.expect(decideOffloadMlp(true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadMlp(true, .q4_1, .q4_1, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadMlp(true, .q8_0, .q8_0, .q8_0, 4096, 2048, 1 << 20, true));
    // The gate and up must share a fused quant and all three be resident.
    try testing.expect(!decideOffloadMlp(true, .q4_0, .q4_0, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadMlp(true, .q4_1, .q4_0, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadMlp(true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1 << 20, false));
    // A quant with no fused shader is rejected; degenerate shapes too.
    try testing.expect(!decideOffloadMlp(true, .q6_k, .q6_k, .q6_k, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadMlp(true, .q4_0, .q4_0, .q4_0, 0, 2048, 1 << 20, true));
    // A discrete (PCIe) device runs the two-kernel form, so it is eligible and
    // the `down` may carry its own GEMV quant (the two kernels are independent).
    try testing.expect(decideOffloadMlp(false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadMlp(false, .q4_0, .q4_0, .q4_1, 4096, 2048, 1 << 20, true));
    try testing.expect(decideOffloadMlp(false, .q4_0, .q4_0, .q6_k, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadMlp(false, .q4_0, .q4_0, .q5_k, 4096, 2048, 1 << 20, true));
    try testing.expect(!decideOffloadMlp(false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1 << 20, false));
    // Deterministic pure function.
    try testing.expectEqual(
        decideOffloadMlp(true, .q4_0, .q4_0, .q4_0, 1024, 1024, 1 << 20, true),
        decideOffloadMlp(true, .q4_0, .q4_0, .q4_0, 1024, 1024, 1 << 20, true),
    );
}

test "fused mlp fusion: UMA by setup, discrete only to avoid a crossing" {
    const testing = std.testing;
    const caps = caps_mod.Caps{
        .dram_bw_gbps = 35.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 20.0, .gemv_streamed_gbps = 5.0 },
    };
    const gu: u64 = 40 << 20;
    const dn: u64 = 20 << 20;
    const both = MlpResidency{ .gate_up = true, .down = true };

    // UMA: the fused op and the pair share the GEMV rate, so the fused op wins
    // by the pair's extra setup (regardless of the pair's residency).
    try testing.expect(mlpFusionPays(true, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, both, caps));
    // Discrete: a pair already all on the GPU (or all on the CPU) is kept; only
    // a crossing pair (exactly one side resident) fuses.
    try testing.expect(!mlpFusionPays(false, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, both, caps));
    try testing.expect(!mlpFusionPays(false, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, .{ .gate_up = false, .down = false }, caps));
    try testing.expect(mlpFusionPays(false, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, .{ .gate_up = true, .down = false }, caps));
    try testing.expect(mlpFusionPays(false, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, .{ .gate_up = false, .down = true }, caps));
    // A quant with no fused shader never fuses; UMA needs all three equal. A
    // discrete device's two kernels may differ (the `down` keeps its own quant),
    // so a crossing mixed-quant pair fuses there.
    try testing.expect(!mlpFusionPays(true, .q6_k, .q6_k, .q6_k, 15360, 5120, 5120, 15360, gu, dn, both, caps));
    try testing.expect(!mlpFusionPays(true, .q4_0, .q4_0, .q4_1, 15360, 5120, 5120, 15360, gu, dn, both, caps));
    try testing.expect(!mlpFusionPays(false, .q4_0, .q4_0, .q4_1, 15360, 5120, 5120, 15360, gu, dn, both, caps));
    try testing.expect(mlpFusionPays(false, .q4_0, .q4_0, .q4_1, 15360, 5120, 5120, 15360, gu, dn, .{ .gate_up = true, .down = false }, caps));
    // No measured device rate: the fused GPU cost is unknown, so keep the pair.
    try testing.expect(!mlpFusionPays(true, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, both, .{ .dram_bw_gbps = 35.0 }));
    try testing.expect(!mlpFusionPays(false, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, .{ .gate_up = true, .down = false }, .{ .dram_bw_gbps = 35.0 }));
    // Deterministic.
    try testing.expectEqual(
        mlpFusionPays(true, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, both, caps),
        mlpFusionPays(true, .q4_0, .q4_0, .q4_0, 15360, 5120, 5120, 15360, gu, dn, both, caps),
    );
}

test "fused mlp is atomic: never split across devices" {
    const testing = std.testing;
    const bytes: u64 = 60 << 20;
    const caps = caps_mod.Caps{
        .dram_bw_gbps = 25.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 50.0, .gemv_streamed_gbps = 1.0 },
    };
    // Even with a resident set and a measured rate, the fused MLP never offers a
    // split: `costOptions` leaves `split_ns` infinite and `gpu_rows` zero.
    const o = costOptions(.mlp, true, .q4_0, .q4_0, .q4_0, 4096, 2048, bytes, true, caps);
    try testing.expect(std.math.isInf(o.split_ns));
    try testing.expectEqual(@as(u64, 0), o.gpu_rows);
    // The oracle returns whole-op `gpu`/`cpu`, never `split`.
    const v = decideDevice(.mlp, true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps);
    try testing.expect(v.device == .gpu or v.device == .cpu);
}

test "fuse mlps: merges a gate_up + ffn_down pair only when it pays" {
    const testing = std.testing;
    const gate_up_atomic: u32 = 3;
    const mlp_atomic: u32 = 4;
    const buf = plan.Binding{ .kind = .buffer };
    const dispatch = [_]plan.Dispatch{
        .{ .op = .matvec, .atomic_id = gate_up_atomic, .tensor = 0, .tensor2 = 1, .in = buf, .out = .{ .kind = .buffer, .ref = 2 } },
        .{ .op = .matvec, .atomic_id = 0, .tensor = 2, .in = .{ .kind = .buffer, .ref = 2 }, .out = .{ .kind = .buffer, .ref = 0 }, .out_len = 5120 },
        .{ .op = .matvec, .atomic_id = 0, .tensor = 3, .in = buf, .out = buf },
    };
    var weights = [_]?WeightMeta{
        .{ .ggml_type = .q4_0, .rows = 15360, .k = 5120, .length = 40 << 20 },
        .{ .ggml_type = .q4_0, .rows = 15360, .k = 5120, .length = 40 << 20 },
        .{ .ggml_type = .q4_0, .rows = 5120, .k = 15360, .length = 20 << 20 },
        null,
    };
    const uma = caps_mod.Caps{
        .dram_bw_gbps = 35.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 20.0, .gemv_streamed_gbps = 5.0 },
    };

    // No tensor is resident: the UMA device still fuses (the fused op wins by
    // the pair's setup); a discrete device keeps the pair (both on the CPU).
    const none = [_]bool{ false, false, false, false };
    // The `gate`/`up` resident but not the `down`: the pair crosses on a
    // discrete device, so it fuses there too.
    const gate_up_only = [_]bool{ true, true, false, false };
    const all_resident = [_]bool{ true, true, true, false };

    // UMA: the pair collapses into one `mlp` carrying the third weight.
    const fused = try fuseMlps(testing.allocator, &dispatch, &weights, &none, gate_up_atomic, mlp_atomic, true, uma);
    defer testing.allocator.free(fused);
    try testing.expectEqual(@as(usize, 2), fused.len);
    try testing.expectEqual(mlp_atomic, fused[0].atomic_id);
    try testing.expectEqual(@as(u32, 2), fused[0].tensor3);
    try testing.expectEqual(@as(u32, 0), fused[0].out.ref);
    try testing.expectEqual(@as(u32, 5120), fused[0].out_len);
    try testing.expectEqual(@as(u32, 3), fused[1].tensor);

    // Discrete, both on the GPU: the pair is kept (the separate `gate_up` is
    // already best, the plan byte-identical).
    const kept = try fuseMlps(testing.allocator, &dispatch, &weights, &all_resident, gate_up_atomic, mlp_atomic, false, uma);
    defer testing.allocator.free(kept);
    try testing.expectEqual(@as(usize, 3), kept.len);
    try testing.expectEqual(gate_up_atomic, kept[0].atomic_id);
    try testing.expectEqual(plan.NONE, kept[0].tensor3);

    // Discrete, a crossing pair: the atomic MLP keeps the whole op on one device.
    const crossed = try fuseMlps(testing.allocator, &dispatch, &weights, &gate_up_only, gate_up_atomic, mlp_atomic, false, uma);
    defer testing.allocator.free(crossed);
    try testing.expectEqual(@as(usize, 2), crossed.len);
    try testing.expectEqual(mlp_atomic, crossed[0].atomic_id);
    try testing.expectEqual(@as(u32, 2), crossed[0].tensor3);
}

test "device oracle: cpu / gpu / split from the op, shape and residency" {
    const testing = std.testing;
    const bytes: u64 = 4_718_592; // rows=4096, k=2048, q4_0
    const caps = caps_mod.Caps{
        .dram_bw_gbps = 25.0,
        .dispatch_latency_ns = 100.0,
        .gpu = .{ .gemv_gbps = 50.0, .gemv_streamed_gbps = 1.0 },
    };

    // A streamed weight that loses stays on the CPU (and is charged to it).
    const cpu = decideDevice(.matvec, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, false, .{}, caps);
    try testing.expectEqual(Device.cpu, cpu.device);
    try testing.expectEqual(bytes, cpu.cpu_bytes);

    // A resident weight with a measured GPU rate splits; `g` is the middle and
    // the two devices' bytes add to the whole.
    const resident = decideDevice(.matvec, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps);
    try testing.expectEqual(Device.split, resident.device);
    try testing.expect(resident.gpu_rows > 0 and resident.gpu_rows < 4096);
    try testing.expectEqual(bytes, resident.gpu_bytes + resident.cpu_bytes);

    // The global load shifts `g`: a CPU already loaded pushes rows to the GPU, a
    // loaded GPU pushes them to the CPU.
    const loaded_cpu = decideDevice(.matvec, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{ .cpu_bytes = bytes / 4 }, caps);
    try testing.expectEqual(Device.split, loaded_cpu.device);
    try testing.expect(loaded_cpu.gpu_rows > resident.gpu_rows);
    const loaded_gpu = decideDevice(.matvec, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{ .gpu_bytes = bytes / 4 }, caps);
    try testing.expectEqual(Device.split, loaded_gpu.device);
    try testing.expect(loaded_gpu.gpu_rows < resident.gpu_rows);

    // An unmeasured GPU rate cannot size a split, so a resident weight runs
    // whole on the GPU (residency always offloads).
    const unmeasured = caps_mod.Caps{ .dram_bw_gbps = 50.0, .dispatch_latency_ns = 100.0 };
    try testing.expectEqual(Device.gpu, decideDevice(.matvec, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, unmeasured).device);

    // UMA has no transfer penalty: whole-op GPU, never a split.
    try testing.expectEqual(Device.gpu, decideDevice(.matvec, true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, false, .{}, caps).device);

    // A prefill GEMM offloads whole (no row slicing yet); a fused gate_up splits
    // like a plain GEMV when both weights are resident q4_0.
    try testing.expectEqual(Device.gpu, decideDevice(.matmul, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 8, bytes, false, .{}, .{}).device);
    try testing.expectEqual(Device.split, decideDevice(.gate_up, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps).device);
    try testing.expectEqual(Device.cpu, decideDevice(.gate_up, false, .q4_1, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps).device);

    // A fused MLP offloads whole when all three weights are resident and share
    // the quant; it never splits (the intermediate is recomputed per slice).
    // Both UMA (the fused shader) and discrete (the two-kernel form) run it.
    try testing.expectEqual(Device.gpu, decideDevice(.mlp, true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps).device);
    try testing.expectEqual(Device.cpu, decideDevice(.mlp, true, .q4_0, .q4_0, .q4_1, 4096, 2048, 1, bytes, true, .{}, caps).device);
    try testing.expectEqual(Device.gpu, decideDevice(.mlp, false, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, true, .{}, caps).device);
    try testing.expectEqual(Device.cpu, decideDevice(.mlp, true, .q4_0, .q4_0, .q4_0, 4096, 2048, 1, bytes, false, .{}, caps).device);

    // An ineligible quant stays on the CPU.
    try testing.expectEqual(Device.cpu, decideDevice(.matvec, false, .q5_k, .q5_k, .q5_k, 4096, 2048, 1, bytes, true, .{}, caps).device);
}

test "split rows: balances the slices, 8-row tile, no degenerate split" {
    const testing = std.testing;
    const caps = caps_mod.Caps{ .dram_bw_gbps = 35.0, .gpu = .{ .gemv_gbps = 35.0 } };
    const bytes: u64 = 8176 * 120; // q4_0 rows at k=3840
    // g/m = 0.5 -> half the rows, floored to the 8-row tile.
    try testing.expectEqual(@as(u64, 4088), splitRows(8176, bytes, .{}, caps));
    // A loaded CPU pushes rows to the GPU; a loaded GPU takes fewer.
    try testing.expect(splitRows(8176, bytes, .{ .cpu_bytes = bytes / 4 }, caps) > 4088);
    try testing.expect(splitRows(8176, bytes, .{ .gpu_bytes = bytes / 4 }, caps) < 4088);
    // No GPU rate, a tiny op, or a full-op split -> no split.
    try testing.expectEqual(@as(u64, 0), splitRows(4096, 4096 * 120, .{}, .{}));
    try testing.expectEqual(@as(u64, 0), splitRows(8, 8 * 120, .{}, caps));
}

test "resident packer: a fused pair is one candidate charging both weights" {
    const testing = std.testing;
    const headroom = plan.VRAM_DRIVER_HEADROOM;
    var out: [8]u32 = undefined;

    // The pair's combined `length`/`value` (300) outranks the single weight
    // (200) and charges both bytes against the budget, so only the pair fits
    // the 300-byte budget.
    var c = [_]Candidate{
        .{ .tensor_index = 1, .length = 200, .value = 200 },
        .{ .tensor_index = 3, .length = 300, .value = 300, .tensor_index2 = 4 },
    };
    try testing.expectEqual(@as(usize, 1), selectResident(&c, headroom + 300, &out));
    try testing.expectEqualSlices(u32, &[_]u32{3}, out[0..1]);

    // With room for both, the pair is placed first (higher value) then the
    // single weight; the caller expands the pair to tensors 3 and 4.
    try testing.expectEqual(@as(usize, 2), selectResident(&c, headroom + 600, &out));
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 1 }, out[0..2]);
}

test "resident packer: a fused mlp triple is one candidate charging three weights" {
    const testing = std.testing;
    const headroom = plan.VRAM_DRIVER_HEADROOM;
    var out: [8]u32 = undefined;

    // The triple's combined length (300) is placed atomically; the caller reads
    // `tensor_index2`/`tensor_index3` to expand it to tensors 5, 6 and 7.
    var c = [_]Candidate{
        .{ .tensor_index = 5, .length = 300, .value = 300, .tensor_index2 = 6, .tensor_index3 = 7 },
    };
    try testing.expectEqual(@as(usize, 1), selectResident(&c, headroom + 300, &out));
    try testing.expectEqualSlices(u32, &[_]u32{5}, out[0..1]);
    try testing.expectEqual(@as(u32, 6), c[0].tensor_index2);
    try testing.expectEqual(@as(u32, 7), c[0].tensor_index3);

    // The triple does not fit a 200-byte budget (its combined length is 300).
    try testing.expectEqual(@as(usize, 0), selectResident(&c, headroom + 200, &out));
}

test "resident packer: headroom, determinism and overflow skipped" {
    const testing = std.testing;
    const headroom = plan.VRAM_DRIVER_HEADROOM;
    var out: [8]u32 = undefined;

    // Budget at or below the headroom leaves no room.
    var c0 = [_]Candidate{
        .{ .tensor_index = 7, .length = 100 },
    };
    try testing.expectEqual(@as(usize, 0), selectResident(&c0, headroom, &out));
    try testing.expectEqual(@as(usize, 0), selectResident(&c0, headroom / 2, &out));

    // Greedy descending length; a large item that does not fit is skipped for
    // smaller ones that do.
    var c1 = [_]Candidate{
        .{ .tensor_index = 1, .length = 100 },
        .{ .tensor_index = 2, .length = 300 },
        .{ .tensor_index = 3, .length = 50 },
    };
    // budget = 200 after headroom: 300 skipped, 100 + 50 chosen.
    try testing.expectEqual(@as(usize, 2), selectResident(&c1, headroom + 200, &out));
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 3 }, out[0..2]);

    // Equal lengths break ties by tensor_index ascending.
    var c2 = [_]Candidate{
        .{ .tensor_index = 9, .length = 10 },
        .{ .tensor_index = 4, .length = 10 },
    };
    try testing.expectEqual(@as(usize, 2), selectResident(&c2, headroom + 100, &out));
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 9 }, out[0..2]);

    // Zero-length items are never packed.
    var c3 = [_]Candidate{
        .{ .tensor_index = 5, .length = 0 },
        .{ .tensor_index = 6, .length = 10 },
    };
    try testing.expectEqual(@as(usize, 1), selectResident(&c3, headroom + 100, &out));
    try testing.expectEqual(@as(u32, 6), out[0]);

    // The fill continues past a non-fitting item and saturates the budget: 300
    // does not fit 250 at its turn, but 150 + 100 do afterwards, filling it.
    var c4 = [_]Candidate{
        .{ .tensor_index = 11, .length = 300 },
        .{ .tensor_index = 12, .length = 150 },
        .{ .tensor_index = 13, .length = 100 },
    };
    try testing.expectEqual(@as(usize, 2), selectResident(&c4, headroom + 250, &out));
    try testing.expectEqualSlices(u32, &[_]u32{ 12, 13 }, out[0..2]);

    // Per-token value outranks size: a small decode weight is placed before a
    // large amortized (prefill) one, and the fill then packs the prefill weight.
    var c5 = [_]Candidate{
        .{ .tensor_index = 21, .length = 400, .value = 0 },
        .{ .tensor_index = 22, .length = 100, .value = 100 },
    };
    try testing.expectEqual(@as(usize, 2), selectResident(&c5, headroom + 500, &out));
    try testing.expectEqualSlices(u32, &[_]u32{ 22, 21 }, out[0..2]);
}

const std = @import("std");
const plan = @import("../plan/root.zig");
const registry = @import("../kernels/registry.zig");
const graph = @import("../architectures/core/graph.zig");
const ggml = @import("../gguf/format.zig");
const Session = @import("session.zig").Session;
const rope = @import("../kernels/transformer/rope.zig");
const attention = @import("../kernels/transformer/attention.zig");
const norm = @import("../kernels/transformer/norm.zig");
const matvec = @import("../kernels/linear/matvec/root.zig");
const matmul = @import("../kernels/linear/matmul/root.zig");
const cpu = @import("../utils/cpu.zig");
const TensorView = @import("../data/tensor_view.zig").TensorView;
const RowCache = @import("row_cache.zig").RowCache;
const sampler = @import("sampler.zig");
const device = @import("../gpu/root.zig");

const Dispatch = plan.Dispatch;

fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    if (rc != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Sequential loop over the plan's `Dispatch[]`. It decides nothing: for each
/// dispatch it resolves bindings + params (both recorded at tune), builds the
/// family's args and calls `kernels.registry.run(op, atomic_id, args)`.
pub fn execute(s: *Session) !u32 {
    return executeAt(s, 0);
}

/// Run the whole `Dispatch[]` for one position, returning the greedy next
/// token. `pos` is baked into the dispatches by the tuner (rope `params[0]`,
/// write_kv/attention `params[0]`), so it is overridden here before running;
/// everything else stays as tuned.
pub fn executeAt(s: *Session, pos: u32) !u32 {
    return sampler.argmax(try executeLogits(s, pos));
}

/// Run the whole `Dispatch[]` for one position and return the logits row; the
/// caller (the generation sampler) decides the token. Same `pos` override as
/// `executeAt`.
pub fn executeLogits(s: *Session, pos: u32) ![]const f32 {
    if (!s.pinned) {
        if (s.affinityCpu(0)) |c| cpu.pinCurrentThread(c);
        s.pinned = true;
    }
    s.pos = pos;
    const p = s.aot();
    var inflight = Inflight{};
    var i: usize = 0;
    while (i < p.dispatch.len) {
        var d = p.dispatch[i];
        overridePos(&d, pos);
        {
            const r0 = nowNs();
            s.ringEnsure(d);
            s.profile.residency_ns += nowNs() - r0;
        }
        // Dep-driven wait: a dispatch may read a buffer a still-in-flight GPU
        // op is computing. Drain that producer on the device only now, at the
        // point of first use.
        if (d.in.kind == .buffer) {
            drainProducer(s, &inflight, d.in.ref);
            if (d.in.ref2 != plan.NONE) drainProducer(s, &inflight, d.in.ref2);
        }
        // Submit an offloaded resident GEMV without waiting; its result stays
        // on the device until the consumer above (or the drain below) needs it.
        if (submitResidentAsync(s, d, &inflight)) {
            i += 1;
            continue;
        }
        const t0 = nowNs();
        try runDispatch(s, d);
        accountDispatch(s, d, nowNs() - t0);
        i += 1;
    }
    // Every produced buffer must be a host buffer before the logits read.
    drainAll(s, &inflight);

    const logits_full = s.buffer(@intFromEnum(graph.Slot.logits));
    if (logits_full.len == 0) return error.NoLogits;
    return logits_full[0..logitsRow(s, logits_full.len)];
}

/// Apply the per-position overrides the tuner left for the runtime: rope and
/// attention carry `pos` (attention also the sliding-window start).
fn overridePos(d: *Dispatch, pos: u32) void {
    switch (d.op) {
        .rope => d.params[0] = pos,
        .attention => {
            d.params[0] = pos;
            const w = d.params[7];
            const seq = pos + 1;
            d.params[6] = if (w != 0 and seq > w) seq - w else 0;
        },
        else => {},
    }
}

/// Per-dispatch diagnostic accounting (unchanged semantics from the old loop).
fn accountDispatch(s: *Session, d: Dispatch, dt: u64) void {
    s.profile.addOp(d.op, dt);
    if (s.verbose >= 1 and d.layer != plan.NONE and d.layer < s.trace.layer_ns.len) {
        s.trace.layer_ns[d.layer] += dt;
    }
    s.profile.parts_sum += d.parts;
    if (d.parts <= 1) {
        s.profile.serial_dispatches += 1;
        s.profile.serial_ns += dt;
    }
    if (d.op == .matvec) {
        s.profile.matvec_count += 1;
        s.profile.matvec_parts_sum += d.parts;
        s.profile.matvec_ns_only += dt;
    }
    if (d.op == .matvec or d.op == .matmul) {
        if (s.tensorView(d.tensor)) |v| s.profile.weight_bytes += v.byte_length;
        if (d.tensor2 != plan.NONE) {
            if (s.tensorView(d.tensor2)) |v2| s.profile.weight_bytes += v2.byte_length;
        }
        if (d.tensor3 != plan.NONE) {
            if (s.tensorView(d.tensor3)) |v3| s.profile.weight_bytes += v3.byte_length;
        }
    }
}

/// A resident GPU GEMV submitted but not yet drained. Holding its result in
/// the device slot until a dispatch actually reads the output buffer is what
/// overlaps CPU work with the GPU op in flight; `Dispatch.dep` names the
/// consumer, which becomes the drain point. The dispatch is retained for the
/// diagnostic accounting applied at drain.
const Inflight = struct {
    entries: [device.RingSize]Entry = undefined,
    len: usize = 0,
    /// Activation shared by the current run of GPU dispatches. It stays
    /// device-owned across the run, so only the run's first member uploads it.
    cur_in: u32 = plan.NONE,

    const Entry = struct {
        d: Dispatch,
        handle: device.Handle,
        rows: usize,
        t0: u64,
    };
};

/// Wait one in-flight handle, copy its result into the session buffer, and
/// account the dispatch.
fn drainEntry(s: *Session, e: Inflight.Entry) void {
    const g = s.gpu orelse return;
    const out = bufN(s, e.d.out.ref, e.d.out_len);
    _ = g.waitResident(e.handle, out, e.rows);
    accountDispatch(s, e.d, nowNs() - e.t0);
}

/// Drain the in-flight GPU op that wrote `ref`, if any (the dep-driven wait).
fn drainProducer(s: *Session, inflight: *Inflight, ref: u32) void {
    if (ref == plan.NONE) return;
    var i: usize = 0;
    while (i < inflight.len) : (i += 1) {
        if (inflight.entries[i].d.out.ref == ref) {
            drainEntry(s, inflight.entries[i]);
            inflight.entries[i] = inflight.entries[inflight.len - 1];
            inflight.len -= 1;
            return;
        }
    }
}

/// Drain every in-flight handle: the end of the run, a full ring, or a change
/// of shared activation that would otherwise overwrite the device buffer.
fn drainAll(s: *Session, inflight: *Inflight) void {
    while (inflight.len > 0) {
        inflight.len -= 1;
        drainEntry(s, inflight.entries[inflight.len]);
    }
    inflight.cur_in = plan.NONE;
}

/// Whether `d` is a resident GPU GEMV the async path can take: a plain matvec,
/// or a fused `gate_up` two-weight matvec, the tuner offloaded to the device,
/// whose weight(s) are resident and whose shape/quant the backend supports.
fn asyncResidentGemv(s: *Session, d: Dispatch) bool {
    if (d.device == 0 or d.op != .matvec or d.tile != 0) return false;
    // The GPU slice of a split always starts at row 0 (its resident buffer holds
    // the leading rows).
    if (d.row_start != 0) return false;
    if (d.in.kind != .buffer or d.out.kind != .buffer) return false;
    if (d.tensor3 != plan.NONE) {
        // Fused MLP: the gate, up and down weights must all be resident and
        // share the fused shader's quant (q4_0, q4_1 or q8_0).
        if (d.atomic_id != @intFromEnum(matvec.AtomicId.mlp_fused)) return false;
        const gate = s.tensorView(d.tensor) orelse return false;
        const up = s.tensorView(d.tensor2) orelse return false;
        const down = s.tensorView(d.tensor3) orelse return false;
        if (gate.dims.len != 2 or up.dims.len != 2 or down.dims.len != 2) return false;
        if (gate.dims[0] != up.dims[0] or gate.dims[1] != up.dims[1]) return false;
        if (down.dims[0] != gate.dims[1]) return false;
        if (gate.ggml_type != up.ggml_type) return false;
        if (gate.ggml_type != .q4_0 and gate.ggml_type != .q4_1 and gate.ggml_type != .q8_0) return false;
        if (gpuQuant(down.ggml_type, gpuVariant(d)) == null) return false;
        if (s.gpu) |g| {
            if (g.isUma() and down.ggml_type != gate.ggml_type) return false;
        }
        if (s.residentHandle(d.tensor) == null or s.residentHandle(d.tensor2) == null or s.residentHandle(d.tensor3) == null) return false;
        return true;
    }
    if (d.tensor2 != plan.NONE) {
        // Fused `gate_up`: both weights must be resident and share the fused
        // shader's quant (q4_0, q4_1 or q8_0).
        if (d.atomic_id != @intFromEnum(matvec.AtomicId.gate_up_fused)) return false;
        const gate = s.tensorView(d.tensor) orelse return false;
        const up = s.tensorView(d.tensor2) orelse return false;
        if (gate.dims.len != 2 or up.dims.len != 2) return false;
        if (gate.dims[0] != up.dims[0] or gate.dims[1] != up.dims[1]) return false;
        if (gate.ggml_type != up.ggml_type) return false;
        if (gate.ggml_type != .q4_0 and gate.ggml_type != .q4_1 and gate.ggml_type != .q8_0) return false;
        if (s.residentHandle(d.tensor) == null or s.residentHandle(d.tensor2) == null) return false;
        return true;
    }
    const view = s.tensorView(d.tensor) orelse return false;
    if (view.dims.len != 2) return false;
    if (gpuQuant(view.ggml_type, gpuVariant(d)) == null) return false;
    if (s.residentHandle(d.tensor) == null) return false;
    return true;
}

/// Submit `d` on the GPU without waiting. A run of consecutive GPU dispatches
/// that share one input keeps that activation device-owned: only the run's
/// first member uploads it (`first`), the rest read the same device buffer.
/// Returns false when the dispatch is not an async resident GEMV, so the
/// caller runs it on the CPU (or the streamed GPU path).
fn submitResidentAsync(s: *Session, d: Dispatch, inflight: *Inflight) bool {
    if (!asyncResidentGemv(s, d)) return false;
    if (d.tensor3 != plan.NONE) return submitMlpAsync(s, d, inflight);
    if (d.tensor2 != plan.NONE) return submitGateUpAsync(s, d, inflight);
    const g = s.gpu orelse return false;
    const view = s.tensorView(d.tensor).?;
    const quant = gpuQuant(view.ggml_type, gpuVariant(d)).?;
    const res = s.residentHandle(d.tensor).?;
    const k: usize = @intCast(view.dims[0]);
    const m: usize = @intCast(view.dims[1]);
    const rows: usize = if (d.row_count > 0) @min(@as(usize, d.row_count), m) else m;
    if (rows == 0 or k == 0) return false;
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len < rows) return false;
    const x = bufN(s, d.in.ref, d.in_len);
    if (x.len < k) return false;

    // A later writer of a buffer that is still in flight must drain it first.
    drainProducer(s, inflight, d.out.ref);
    const same_input = inflight.len > 0 and inflight.cur_in == d.in.ref;
    const first = !same_input or inflight.len == device.RingSize;
    if (first) drainAll(s, inflight);
    const h = g.matVecResidentBurstAsync(res, quant, rows, k, x[0..k], first) orelse return false;
    inflight.entries[inflight.len] = .{ .d = d, .handle = h, .rows = rows, .t0 = nowNs() };
    inflight.len += 1;
    inflight.cur_in = d.in.ref;
    return true;
}

/// Submit a fused MLP resident dispatch (three resident weights, one dispatch)
/// on the same device-owned activation burst as the single-weight path. On a
/// UMA device the single fused shader keeps the `ffn_mid` in shared memory; on
/// a discrete device the two multi-workgroup kernels run back-to-back (the
/// fused `gate_up` then the `down`) with the `ffn_mid` in a device-local buffer,
/// so all three weights are read at full workgroup parallelism and the
/// intermediate never crosses the PCIe. The gated activation comes from the
/// arch (`params[0]`).
fn submitMlpAsync(s: *Session, d: Dispatch, inflight: *Inflight) bool {
    const g = s.gpu orelse return false;
    const gate = s.tensorView(d.tensor).?;
    const down = s.tensorView(d.tensor3).?;
    const res_g = s.residentHandle(d.tensor).?;
    const res_u = s.residentHandle(d.tensor2).?;
    const res_d = s.residentHandle(d.tensor3).?;
    const k: usize = @intCast(gate.dims[0]);
    const ffn: usize = @intCast(gate.dims[1]);
    const hidden: usize = @intCast(down.dims[1]);
    if (k == 0 or ffn == 0 or hidden == 0) return false;
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len < hidden) return false;
    const x = bufN(s, d.in.ref, d.in_len);
    if (x.len < k) return false;

    drainProducer(s, inflight, d.out.ref);
    const same_input = inflight.len > 0 and inflight.cur_in == d.in.ref;
    const first = !same_input or inflight.len == device.RingSize;
    if (first) drainAll(s, inflight);
    const act: device.GateUpAct = if (d.params[0] == graph.Params.activation_gelu) .gelu else .silu;
    const gu_quant = gpuQuant(gate.ggml_type, gpuVariant(d)) orelse return false;
    const h = if (g.isUma())
        g.matVecMlpResidentBurstAsync(res_g, res_u, res_d, gu_quant, hidden, k, ffn, x[0..k], act, gpuVariant(d), first) orelse return false
    else blk: {
        const down_quant = gpuQuant(down.ggml_type, gpuVariant(d)) orelse return false;
        break :blk g.matVecMlpPairResidentBurstAsync(res_g, res_u, res_d, gu_quant, down_quant, hidden, k, ffn, x[0..k], act, gpuVariant(d), first) orelse return false;
    };
    inflight.entries[inflight.len] = .{ .d = d, .handle = h, .rows = hidden, .t0 = nowNs() };
    inflight.len += 1;
    inflight.cur_in = d.in.ref;
    return true;
}

/// Submit a fused `gate_up` resident GEMV (two resident weights, one dispatch)
/// on the same device-owned activation burst as the single-weight path. The
/// gated activation comes from the arch (`params[0]`, as in `runGateUp`).
fn submitGateUpAsync(s: *Session, d: Dispatch, inflight: *Inflight) bool {
    const g = s.gpu orelse return false;
    const gate = s.tensorView(d.tensor).?;
    const res_g = s.residentHandle(d.tensor).?;
    const res_u = s.residentHandle(d.tensor2).?;
    const k: usize = @intCast(gate.dims[0]);
    const m: usize = @intCast(gate.dims[1]);
    const rows: usize = if (d.row_count > 0) @min(@as(usize, d.row_count), m) else m;
    if (rows == 0 or k == 0) return false;
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len < rows) return false;
    const x = bufN(s, d.in.ref, d.in_len);
    if (x.len < k) return false;

    drainProducer(s, inflight, d.out.ref);
    const same_input = inflight.len > 0 and inflight.cur_in == d.in.ref;
    const first = !same_input or inflight.len == device.RingSize;
    if (first) drainAll(s, inflight);
    const act: device.GateUpAct = if (d.params[0] == graph.Params.activation_gelu) .gelu else .silu;
    const quant = gpuQuant(gate.ggml_type, gpuVariant(d)) orelse return false;
    const h = g.matVecGateUpResidentBurstAsync(res_g, res_u, quant, rows, k, x[0..k], act, gpuVariant(d), first) orelse return false;
    inflight.entries[inflight.len] = .{ .d = d, .handle = h, .rows = rows, .t0 = nowNs() };
    inflight.len += 1;
    inflight.cur_in = d.in.ref;
    return true;
}

/// The logits row width the plan records (the `lm_head`/`softcap` `out_len`).
/// In batched mode the logits buffer holds `prefill_block` rows; decode reads
/// row 0 only.
fn logitsRow(s: *Session, full_len: usize) usize {
    var n: u32 = 0;
    for (s.aot().dispatch) |d| {
        if (d.out.ref == @intFromEnum(graph.Slot.logits)) n = @max(n, d.out_len);
    }
    if (n == 0 or n > full_len) return full_len;
    return n;
}

/// Run the plan's batched prefill forward (`AOTKernel.prefill`) for the `m`
/// prompt tokens starting at `t0` and return the greedy next token. Mirrors
/// `executeAt` but over `m` rows. The plan supplied the policy; this only
/// applies it.
pub fn executePrefill(s: *Session, t0: u32, m: u32) !u32 {
    return sampler.argmax(try executePrefillLogits(s, t0, m));
}

/// Run the batched prefill forward and return the logits row of the last
/// prompt token; the caller (the generation sampler) decides the token. Uses
/// the GEMM family for the projections and block-causal attention.
pub fn executePrefillLogits(s: *Session, t0: u32, m: u32) ![]const f32 {
    if (m == 0) return error.EmptyPrefill;
    if (!s.pinned) {
        if (s.affinityCpu(0)) |c| cpu.pinCurrentThread(c);
        s.pinned = true;
    }
    s.pos = t0 + m - 1;
    const p = s.aot();
    // Prefill reads the same weights as decode; the ring ensures their blocks.
    var vocab: usize = 0;
    for (p.prefill) |d| {
        s.ringEnsure(d);
        if (d.out.ref == @intFromEnum(graph.Slot.logits) and d.out_len != 0) vocab = d.out_len;
        try runPrefill(s, d, t0, m);
    }
    if (vocab == 0) return error.NoLogits;
    const logits = s.buffer(@intFromEnum(graph.Slot.logits));
    const base = @as(usize, m - 1) * vocab;
    if (base + vocab > logits.len) return error.NoLogits;
    return logits[base .. base + vocab];
}

fn runPrefill(s: *Session, d: Dispatch, t0: u32, m: u32) !void {
    switch (d.op) {
        .dequant => return prefillEmbd(s, d, t0, m),
        .matmul => return prefillMatmul(s, d, m),
        .matvec => return prefillLmHead(s, d, m),
        .norm => return prefillNorm(s, d, m),
        .rope => return prefillRope(s, d, t0, m),
        .attention => return prefillAttention(s, d, t0, m),
        .activation => return prefillActivation(s, d, m),
        .elementwise => return prefillElementwise(s, d, m),
    }
}

/// Per-row element width of a slot, read from the prefill program: the first
/// dispatch that writes the slot (its `out_len`), else one that reads it.
fn prefillRowWidth(s: *Session, slot: u32) usize {
    for (s.aot().prefill) |d| {
        if (d.out.ref == slot and d.out_len != 0) return d.out_len;
        if (d.in.ref == slot and d.in_len != 0) return d.in_len;
        if (d.in.ref2 == slot and d.in_len != 0) return d.in_len;
    }
    return 0;
}

/// `m` embedding lookups (one per prompt token) into the M-row `out` buffer.
fn prefillEmbd(s: *Session, d: Dispatch, t0: u32, m: u32) !void {
    const view = s.tensorView(d.tensor) orelse return error.Unsupported;
    const out = s.buffer(d.out.ref);
    if (view.dims.len != 2) return error.ShapeMismatch;
    const row_len: usize = @intCast(view.dims[0]);
    const rows: usize = @intCast(view.dims[1]);
    if (out.len < row_len * m) return error.ShapeMismatch;
    const total = ggml.elementCount(view.dims) catch return error.Unsupported;
    if (total <= row_len) return error.Unsupported;
    const cache: ?*RowCache = if (s.row_cache) |*rc| rc else null;
    for (0..m) |j| {
        const id = s.batch_ids[t0 + j];
        if (id >= rows) return error.Unsupported;
        const dst = out[j * row_len ..][0..row_len];
        try embedRow(cache, d, &view, id, dst);
        applyEmbedScale(d, dst);
    }
}

/// Batched projection: `x` is `m x K`, `out` is `m x N`, the weight `[K, N]`.
fn prefillMatmul(s: *Session, d: Dispatch, m: u32) !void {
    const view = s.tensorView(d.tensor) orelse return error.Unsupported;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const k: usize = @intCast(view.dims[0]);
    const n: usize = @intCast(view.dims[1]);
    const x_all = s.buffer(d.in.ref);
    const out_all = s.buffer(d.out.ref);
    if (x_all.len < k * m or out_all.len < n * m) return error.ShapeMismatch;
    // The tuner marked this batched projection for the GPU co-processor: one
    // GEMM dispatch amortizes the streamed weight over the `m` prompt rows. Fall
    // through to the CPU `matVecBatch` when the device rejects the quant/layout.
    if (d.device != 0) {
        if (s.gpu) |g| {
            if (gpuQuant(view.ggml_type, gpuVariant(d))) |q| {
                if (s.residentHandle(d.tensor)) |res| {
                    if (g.matMulResident(res, q, n, k, m, x_all[0 .. k * m], out_all[0 .. n * m])) return;
                } else if (view.mapped_slice) |data| {
                    if (g.matMulBatch(q, data, n, k, m, x_all[0 .. k * m], out_all[0 .. n * m])) return;
                }
            }
        }
    }
    return s.matVecBatch(
        &view,
        x_all,
        out_all,
        m,
        n,
        k,
        d.parts,
        @intFromEnum(matvec.AtomicId.mat_vec),
        d.variant,
    );
}

/// `lm_head` on the last row only: the tail norm already produced the `m` rows.
fn prefillLmHead(s: *Session, d: Dispatch, m: u32) !void {
    const view = s.tensorView(d.tensor) orelse return error.Unsupported;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const k: usize = @intCast(view.dims[0]);
    const n: usize = @intCast(view.dims[1]);
    const x_all = s.buffer(d.in.ref);
    const out_all = s.buffer(d.out.ref);
    if (x_all.len < k * m or out_all.len < n * m) return error.ShapeMismatch;
    const x = x_all[(m - 1) * k ..][0..k];
    const out = out_all[(m - 1) * n ..][0..n];
    return s.matVecPlan(&view, x, out, n, 0, d.parts, @intCast(d.atomic_id), d.variant, d.simd_width, null, .gelu_tanh);
}

fn prefillNorm(s: *Session, d: Dispatch, m: u32) !void {
    const nm = @import("../kernels/transformer/norm.zig");
    const x_all = s.buffer(d.in.ref);
    const out_all = s.buffer(d.out.ref);
    const row_in: usize = d.in_len;
    const row_out: usize = d.out_len;
    if (row_in == 0 or row_out == 0) return error.ShapeMismatch;
    if (x_all.len < row_in * m or out_all.len < row_out * m) return error.ShapeMismatch;
    const eps = unpackFloat(d.params[0]);
    const head_dim: usize = d.params[1];
    const kind = d.params[2];

    if (kind == graph.Params.norm_scale) {
        const weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
        if (weight.len != 1) return error.ShapeMismatch;
        for (0..m) |j| {
            const x = x_all[j * row_in ..][0..row_in];
            const out = out_all[j * row_out ..][0..row_out];
            for (out, x) |*o, v| o.* = v * weight[0];
        }
        return;
    }

    if (kind == graph.Params.norm_heads or kind == graph.Params.norm_heads_no_weight) {
        if (head_dim == 0 or row_in % head_dim != 0) return error.ShapeMismatch;
        var weight: []const f32 = &.{};
        if (kind == graph.Params.norm_heads) {
            weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
            if (weight.len != head_dim) return error.ShapeMismatch;
        }
        for (0..m) |j| {
            var args = nm.Args{
                .x = x_all[j * row_in ..][0..row_in],
                .eps = eps,
                .head_dim = head_dim,
                .weight = weight,
                .simd_width = d.simd_width,
            };
            try registry.run(.norm, @intCast(d.atomic_id), &args);
        }
        return;
    }

    var weight: []const f32 = &.{};
    if (kind == graph.Params.norm_rms) {
        weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
        if (weight.len != row_in) return error.ShapeMismatch;
    }
    for (0..m) |j| {
        var args = nm.Args{
            .x = x_all[j * row_in ..][0..row_in],
            .out = out_all[j * row_out ..][0..row_out],
            .eps = eps,
            .weight = weight,
            .simd_width = d.simd_width,
        };
        try registry.run(.norm, @intCast(d.atomic_id), &args);
    }
}

fn prefillRope(s: *Session, d: Dispatch, t0: u32, m: u32) !void {
    const buf = s.buffer(d.out.ref);
    const n_rot: usize = d.params[1];
    const head_dim: usize = d.params[3];
    const n_heads: usize = d.params[4];
    if (n_rot == 0 or head_dim == 0 or n_heads == 0) return;
    const per = n_heads * head_dim;
    if (buf.len < per * m) return error.ShapeMismatch;
    const inv = s.ropeInvFreq(d.rope_cfg) orelse return error.Unsupported;
    for (0..m) |j| {
        ropeHeads(buf[j * per ..][0..per], n_heads, head_dim, n_rot, @intCast(t0 + j), inv, s.rope_cos, s.rope_sin);
    }
}

fn prefillAttention(s: *Session, d: Dispatch, t0: u32, m: u32) !void {
    if (d.atomic_id == @intFromEnum(attention.AtomicId.write_kv)) return prefillWriteKv(s, d, t0, m);
    const q_all = s.buffer(d.in.ref);
    const out_all = s.buffer(d.out.ref);
    const kvbuf = s.bufferF16(d.kv);
    const scale = unpackFloat(d.params[1]);
    const n_heads: usize = d.params[3];
    const n_kv_heads: usize = d.params[4];
    const head_dim: usize = d.params[5];
    const window: usize = d.params[7];
    if (n_heads == 0 or head_dim == 0) return error.ShapeMismatch;
    const per = n_heads * head_dim;
    if (q_all.len < per * m or out_all.len < per * m) return error.ShapeMismatch;
    const views = splitKv(kvbuf, n_kv_heads, head_dim, d.max_seq) orelse return error.ShapeMismatch;
    if (s.scores.len < d.max_seq) return error.ShapeMismatch;
    for (0..m) |j| {
        const p = t0 + j;
        const seq = p + 1;
        const t_start: usize = if (window != 0 and seq > window) seq - window else 0;
        try attentionHeads(
            s,
            d,
            q_all[j * per ..][0..per],
            out_all[j * per ..][0..per],
            views.k,
            views.v,
            n_heads,
            n_kv_heads,
            head_dim,
            p,
            d.max_seq,
            scale,
            t_start,
        );
    }
}

fn prefillWriteKv(s: *Session, d: Dispatch, t0: u32, m: u32) !void {
    const k = s.buffer(d.in.ref);
    const v = s.buffer(d.in.ref2);
    const kvbuf = s.bufferF16(d.kv);
    const n_kv_heads: usize = d.params[4];
    const head_dim: usize = d.params[5];
    if (n_kv_heads == 0 or head_dim == 0) return error.ShapeMismatch;
    const per = n_kv_heads * head_dim;
    if (k.len < per * m or v.len < per * m) return error.ShapeMismatch;
    const views = splitKv(kvbuf, n_kv_heads, head_dim, d.max_seq) orelse return error.ShapeMismatch;
    for (0..m) |j| {
        attention.writeKv(
            k[j * per ..][0..per],
            v[j * per ..][0..per],
            views.k,
            views.v,
            n_kv_heads,
            head_dim,
            t0 + j,
            d.max_seq,
        );
    }
}

fn prefillActivation(s: *Session, d: Dispatch, m: u32) !void {
    const ac = @import("../kernels/transformer/activation.zig");
    const ew = @import("../kernels/transformer/elementwise.zig");
    if (d.sem_op == graph.Params.softcap_code) {
        const out_all = s.buffer(d.out.ref);
        const row: usize = d.out_len;
        if (row == 0 or out_all.len < row * m) return error.ShapeMismatch;
        for (0..m) |j| {
            const r = out_all[j * row ..][0..row];
            var args = ew.Args{ .a = r, .out = r, .cap = unpackFloat(d.params[0]), .simd_width = d.simd_width };
            try registry.run(.elementwise, @intCast(d.atomic_id), &args);
        }
        return;
    }
    if (d.atomic_id == @intFromEnum(ac.AtomicId.gelu)) {
        const out_all = s.buffer(d.out.ref);
        const row: usize = d.out_len;
        if (row == 0 or out_all.len < row * m) return error.ShapeMismatch;
        for (0..m) |j| {
            var args = ac.Args{ .out = out_all[j * row ..][0..row] };
            try registry.run(.activation, @intCast(d.atomic_id), &args);
        }
        return;
    }
    const gate_all = s.buffer(d.in.ref);
    const up_all = s.buffer(d.in.ref2);
    const out_all = s.buffer(d.out.ref);
    const row: usize = d.out_len;
    if (row == 0 or gate_all.len < row * m or up_all.len < row * m or out_all.len < row * m) return error.ShapeMismatch;
    for (0..m) |j| {
        var args = ac.Args{
            .gate = gate_all[j * row ..][0..row],
            .up = up_all[j * row ..][0..row],
            .out = out_all[j * row ..][0..row],
            .simd_width = d.simd_width,
        };
        try registry.run(.activation, @intCast(d.atomic_id), &args);
    }
}

fn prefillElementwise(s: *Session, d: Dispatch, m: u32) !void {
    const ew = @import("../kernels/transformer/elementwise.zig");
    const a_all = s.buffer(d.in.ref);
    const out_all = s.buffer(d.out.ref);
    const row: usize = d.out_len;
    if (row == 0 or a_all.len < row * m or out_all.len < row * m) return error.ShapeMismatch;
    for (0..m) |j| {
        const a = a_all[j * row ..][0..row];
        const out = out_all[j * row ..][0..row];
        switch (d.sem_op) {
            graph.Params.mul_code => {
                const b_all = s.buffer(d.in.ref2);
                const b_row = prefillRowWidth(s, d.in.ref2);
                const off: usize = d.params[0];
                if (b_row == 0 or b_all.len < j * b_row + off + row) return error.ShapeMismatch;
                var args = ew.Args{ .a = a, .b = b_all[j * b_row + off ..][0..row], .out = out, .simd_width = d.simd_width };
                try registry.run(.elementwise, @intCast(d.atomic_id), &args);
            },
            graph.Params.scale_code => {
                var args = ew.Args{ .a = a, .out = out, .scale = unpackFloat(d.params[0]), .simd_width = d.simd_width };
                try registry.run(.elementwise, @intCast(d.atomic_id), &args);
            },
            graph.Params.combine_code => {
                const b_all = s.buffer(d.in.ref2);
                if (b_all.len < (j + 1) * row) return error.ShapeMismatch;
                var args = ew.Args{ .a = a, .b = b_all[j * row ..][0..row], .out = out, .scale = unpackFloat(d.params[0]), .simd_width = d.simd_width };
                try registry.run(.elementwise, @intCast(d.atomic_id), &args);
            },
            else => {
                const b_all = s.buffer(d.in.ref2);
                if (b_all.len < (j + 1) * row) return error.ShapeMismatch;
                var args = ew.Args{ .a = a, .b = b_all[j * row ..][0..row], .out = out, .scale = unpackFloat(d.params[0]), .simd_width = d.simd_width };
                try registry.run(.elementwise, @intCast(d.atomic_id), &args);
            },
        }
    }
}

fn runDispatch(s: *Session, d: Dispatch) !void {
    switch (d.op) {
        .dequant => return runDequant(s, d),
        .matvec => return runMatvec(s, d),
        .matmul => return error.Unsupported,
        .norm => return runNorm(s, d),
        .activation => return runActivation(s, d),
        .elementwise => return runElementwise(s, d),
        .rope => return runRope(s, d),
        .attention => return runAttention(s, d),
    }
}

fn unpackFloat(bits: u32) f32 {
    return @bitCast(bits);
}

/// A buffer slice of `len` f32 elements; `len == 0` means the whole buffer
/// (backward-compatible for globals). Plans record `len` at tune per layer.
fn bufN(s: *Session, ref: u32, len: u32) []f32 {
    const full = s.buffer(ref);
    if (len == 0 or len >= full.len) return full;
    return full[0..len];
}

fn runDequant(s: *Session, d: Dispatch) !void {
    const q = @import("../kernels/linear/quant/root.zig");
    const view = s.tensorView(d.tensor) orelse return error.Unsupported;
    const out = bufN(s, d.out.ref, d.out_len);
    const data = view.mapped_slice orelse return error.Unsupported;
    if (out.len == 0) return error.Unsupported;
    // Embedding lookup: the tensor is [row_elems, rows] but `out` holds one
    // row. Dequantize only `s.token_id`'s row so the huge table stays mmap-backed.
    if (view.dims.len == 2 and out.len == view.dims[0]) {
        const total = ggml.elementCount(view.dims) catch return error.Unsupported;
        if (total > out.len) {
            const rows = view.dims[1];
            if (s.token_id >= rows) return error.Unsupported;
            const cache: ?*RowCache = if (s.row_cache) |*rc| rc else null;
            try embedRow(cache, d, &view, s.token_id, out);
            applyEmbedScale(d, out);
            return;
        }
    }
    var args = q.Args{
        .ggml_type = @intFromEnum(view.ggml_type),
        .data = data,
        .out = out,
    };
    try registry.run(.dequant, @intCast(d.atomic_id), &args);
    applyEmbedScale(d, out);
}

/// Dequantize one embedding row `row_id` of `view` into `out` (unscaled),
/// consulting and filling `cache`. The caller applies the embedding scale. A
/// cache miss reads only that row from the mmap, so the table is never pinned
/// whole. The cached value is the pre-scale row, the same bytes the uncached
/// path produces.
fn embedRow(cache: ?*RowCache, d: Dispatch, view: *const TensorView, row_id: u32, out: []f32) !void {
    const q = @import("../kernels/linear/quant/root.zig");
    if (cache) |rc| {
        if (rc.get(d.tensor, row_id, out)) return;
    }
    const data = view.mapped_slice orelse return error.Unsupported;
    const row_bytes = ggml.tensorByteLength(view.ggml_type, view.dims[0]) catch return error.Unsupported;
    const start = @as(u64, row_id) * row_bytes;
    const end = start + row_bytes;
    if (data.len < @as(usize, @intCast(end))) return error.Unsupported;
    var args = q.Args{
        .ggml_type = @intFromEnum(view.ggml_type),
        .data = data[@intCast(start)..@intCast(end)],
        .out = out,
    };
    try registry.run(.dequant, @intCast(d.atomic_id), &args);
    if (cache) |rc| rc.put(d.tensor, row_id, out);
}

/// Post-lookup embedding scale, recorded at tune (`sem_op == embed_code`).
fn applyEmbedScale(d: Dispatch, out: []f32) void {
    if (d.sem_op != graph.Params.embed_code) return;
    const scale = unpackFloat(d.params[0]);
    if (scale == 0) return;
    for (out) |*v| v.* *= scale;
}

fn runMatvec(s: *Session, d: Dispatch) !void {
    if (d.tensor3 != plan.NONE and d.atomic_id == @intFromEnum(matvec.AtomicId.mlp_fused)) {
        return runMlp(s, d);
    }
    if (d.tensor2 != plan.NONE and d.atomic_id == @intFromEnum(matvec.AtomicId.gate_up_fused)) {
        return runGateUp(s, d);
    }
    const view = s.tensorView(d.tensor) orelse return error.Unsupported;
    const x = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len == 0 or x.len == 0) return error.Unsupported;
    if (view.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(view.dims[0]);
    const m: usize = @intCast(view.dims[1]);
    if (x.len < n or out.len < m) return error.ShapeMismatch;
    // Heterogeneous split: the output-row range this dispatch computes. `0`
    // count = the whole tensor. The GPU slice is always `[0, rows)` (its
    // resident buffer holds the leading rows); the CPU slice is
    // `[row_start, row_start + rows)` and writes `out` at those absolute rows.
    const row_start: usize = d.row_start;
    const rows: usize = if (d.row_count > 0) @min(@as(usize, d.row_count), m -| row_start) else m -| row_start;
    if (row_start >= m or rows == 0) return error.ShapeMismatch;
    if (d.device != 0) {
        if (s.gpu) |g| {
            if (gpuQuant(view.ggml_type, gpuVariant(d))) |q| {
                if (s.residentHandle(d.tensor)) |res| {
                    if (row_start == 0 and g.matVecResident(res, q, rows, n, x[0..n], out[0..rows])) return;
                } else if (view.mapped_slice) |data| {
                    const tile: usize = d.tile;
                    if (tile > 0 and tile < m) {
                        if (matVecTiled(g, q, data, m, n, x[0..n], out[0..m], tile)) return;
                    } else if (g.matVec(q, data, m, n, x[0..n], out[0..m])) return;
                }
            }
        }
    }
    return s.matVecPlan(&view, x[0..n], out[0..m], rows, row_start, d.parts, @intCast(d.atomic_id), d.variant, d.simd_width, null, .gelu_tanh);
}

/// Streamed GPU GEMV in `tile`-row slices: each slice stages only its own
/// weight range and writes its own output range, so a weight larger than the
/// backend's staging limit still runs. Returns false (caller falls back to the
/// CPU) if any slice is rejected.
fn matVecTiled(
    g: *device.Device,
    q: device.Quant,
    data: []const u8,
    rows: usize,
    k: usize,
    x: []const f32,
    out: []f32,
    tile: usize,
) bool {
    const block_elems = q.blockElems();
    if (tile == 0 or k == 0 or k % block_elems != 0) return false;
    const row_bytes = (k / block_elems) * q.blockBytes();
    if (data.len < rows * row_bytes or out.len < rows) return false;
    var r0: usize = 0;
    while (r0 < rows) {
        const n = @min(tile, rows - r0);
        const wslice = data[r0 * row_bytes .. (r0 + n) * row_bytes];
        if (!g.matVec(q, wslice, n, k, x, out[r0 .. r0 + n])) return false;
        r0 += n;
    }
    return true;
}

/// The co-processor GEMV variant a `device == 1` dispatch recorded in its
/// `variant` byte. Out-of-range codes fall back to `plain` so a malformed plan
/// degrades instead of redeciding.
fn gpuVariant(d: Dispatch) device.GpuVariant {
    return std.enums.fromInt(device.GpuVariant, d.variant) orelse .plain;
}

/// Map a weight's ggml type to the GPU backend's quant kind for the recorded
/// variant, or null when the device cannot run it (the CPU kernel handles
/// those). The runtime applies the variant and decides nothing.
fn gpuQuant(t: ggml.GGMLType, v: device.GpuVariant) ?device.Quant {
    const base: device.Quant = switch (t) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q4_k => .q4_k,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => return null,
    };
    return base.variant(v);
}

/// `gate_up_fused`: one pass over the shared activation (`in`) reads both the
/// gate weight (`tensor`) and the up weight (`tensor2`), applies the gated
/// activation recorded by the arch (`params[0]`) and writes the result
/// (`out`, the FFN mid buffer).
fn runGateUp(s: *Session, d: Dispatch) !void {
    const gate = s.tensorView(d.tensor) orelse return error.Unsupported;
    const up = s.tensorView(d.tensor2) orelse return error.Unsupported;
    const x = bufN(s, d.in.ref, d.in_len);
    const mid = bufN(s, d.out.ref, d.out_len);
    if (mid.len == 0 or x.len == 0) return error.Unsupported;
    if (gate.dims.len != 2 or up.dims.len != 2) return error.ShapeMismatch;
    const n: usize = @intCast(gate.dims[0]);
    const m: usize = @intCast(gate.dims[1]);
    if (x.len < n or mid.len < m) return error.ShapeMismatch;
    // The heterogeneous split: the row range this dispatch computes.
    const row_start: usize = d.row_start;
    const rows: usize = if (d.row_count > 0) @min(@as(usize, d.row_count), m -| row_start) else m -| row_start;
    if (row_start >= m or rows == 0) return error.ShapeMismatch;
    const act: matvec.ActivationKind = if (d.params[0] == graph.Params.activation_gelu) .gelu_tanh else .silu;
    return s.matVecPlan(&gate, x[0..n], mid[0..m], rows, row_start, d.parts, @intCast(d.atomic_id), d.variant, d.simd_width, &up, act);
}

/// `mlp_fused`: the whole SwiGLU MLP in one dispatch — the `gate`/`up` matvec
/// and the gated activation produce the `ffn_mid` in the session's scratch,
/// which the `down` matvec then consumes. The intermediate never touches a
/// session buffer (no `ffn_mid` round-trip), and the two phases are one plan
/// dispatch instead of two.
fn runMlp(s: *Session, d: Dispatch) !void {
    const gate = s.tensorView(d.tensor) orelse return error.Unsupported;
    const up = s.tensorView(d.tensor2) orelse return error.Unsupported;
    const down = s.tensorView(d.tensor3) orelse return error.Unsupported;
    const x = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    if (x.len == 0 or out.len == 0) return error.Unsupported;
    if (gate.dims.len != 2 or up.dims.len != 2 or down.dims.len != 2) return error.ShapeMismatch;
    const k: usize = @intCast(gate.dims[0]);
    const ffn: usize = @intCast(gate.dims[1]);
    const hidden: usize = @intCast(down.dims[1]);
    if (down.dims[0] != gate.dims[1]) return error.ShapeMismatch;
    if (x.len < k or out.len < hidden) return error.ShapeMismatch;
    if (s.mlp_scratch.len < ffn) {
        if (s.mlp_scratch.len > 0) s.allocator.free(s.mlp_scratch);
        s.mlp_scratch = try s.allocator.alloc(f32, ffn);
    }
    const mid = s.mlp_scratch[0..ffn];
    const act: matvec.ActivationKind = if (d.params[0] == graph.Params.activation_gelu) .gelu_tanh else .silu;
    // Phase 1 reuses the fused `gate_up` atomic (the gated activation in
    // registers); phase 2 runs the `down` matvec over the computed `m`. The two
    // weights may carry different quants, so phase 2 uses the family default
    // (which dispatches on the `down`'s own type), not the gate's CPU plan.
    try s.matVecPlan(&gate, x[0..k], mid, ffn, 0, d.parts, @intFromEnum(matvec.AtomicId.gate_up_fused), d.variant, d.simd_width, &up, act);
    return s.matVecPlan(&down, mid, out[0..hidden], hidden, 0, d.parts, @intFromEnum(matvec.AtomicId.mat_vec), 0, d.simd_width, null, act);
}

fn runNorm(s: *Session, d: Dispatch) !void {
    const nm = @import("../kernels/transformer/norm.zig");
    if (d.atomic_id == @intFromEnum(nm.AtomicId.qk_norm_rope_heads)) return runFusedNormRope(s, d);
    const x = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    if (x.len == 0 or out.len == 0) return error.Unsupported;
    const eps = unpackFloat(d.params[0]);
    const head_dim: usize = d.params[1];
    const kind = d.params[2];

    if (kind == graph.Params.norm_scale) {
        const weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
        if (weight.len != 1) return error.ShapeMismatch;
        for (out, x) |*o, v| o.* = v * weight[0];
        return;
    }

    if (kind == graph.Params.norm_heads or kind == graph.Params.norm_heads_no_weight) {
        if (head_dim == 0 or x.len % head_dim != 0) return error.ShapeMismatch;
        var args = nm.Args{ .x = x, .eps = eps, .head_dim = head_dim, .simd_width = d.simd_width };
        if (kind == graph.Params.norm_heads) {
            const weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
            if (weight.len != head_dim) return error.ShapeMismatch;
            args.weight = weight;
        }
        return registry.run(.norm, @intCast(d.atomic_id), &args);
    }

    var args = nm.Args{ .x = x, .out = out, .eps = eps, .simd_width = d.simd_width };
    if (kind == graph.Params.norm_rms) {
        const weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
        if (weight.len != x.len) return error.ShapeMismatch;
        args.weight = weight;
    }
    return registry.run(.norm, @intCast(d.atomic_id), &args);
}

/// `qk_norm_rope_heads`: per-head RMS norm of q/k followed, in the same pass,
/// by the RoPE rotation. The norm weight + eps come from the norm params; the
/// cos/sin rows are built from the layer's `zcom` inverse frequencies, exactly
/// as the separate rope pass would, so the result equals
/// `qkNormHeads` + `rope`. The rope position is the current forward position.
fn runFusedNormRope(s: *Session, d: Dispatch) !void {
    const x = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    if (x.len == 0 or out.len == 0 or out.len < x.len) return error.Unsupported;
    const eps = unpackFloat(d.params[0]);
    const head_dim: usize = d.params[1];
    if (head_dim == 0 or x.len % head_dim != 0) return error.ShapeMismatch;
    const weight = s.normWeight(d.tensor, d.layer) orelse return error.Unsupported;
    if (weight.len != head_dim) return error.ShapeMismatch;
    const inv = s.ropeInvFreq(d.rope_cfg) orelse return error.Unsupported;
    const n_rot = inv.len * 2;
    if (n_rot == 0 or n_rot > head_dim) return error.ShapeMismatch;
    const angles = n_rot / 2;
    if (s.rope_cos.len < angles or s.rope_sin.len < angles) return error.ShapeMismatch;
    ropeRows(s.pos, inv, angles, s.rope_cos, s.rope_sin);
    var args = norm.Args{
        .x = x,
        .weight = weight,
        .eps = eps,
        .cos_row = s.rope_cos[0..angles],
        .sin_row = s.rope_sin[0..angles],
        .n_heads = x.len / head_dim,
        .head_dim = head_dim,
        .n_rot = n_rot,
        .simd_width = d.simd_width,
    };
    try registry.run(.norm, @intCast(d.atomic_id), &args);
    if (out.ptr != x.ptr) @memcpy(out[0..x.len], x);
}

fn runActivation(s: *Session, d: Dispatch) !void {
    const ac = @import("../kernels/transformer/activation.zig");
    const ew = @import("../kernels/transformer/elementwise.zig");
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len == 0) return error.Unsupported;
    // `softcap` is lowered to `.activation` but uses the elementwise id.
    if (d.sem_op == graph.Params.softcap_code) {
        var eargs = ew.Args{ .a = out, .out = out, .cap = unpackFloat(d.params[0]), .simd_width = d.simd_width };
        return registry.run(.elementwise, @intCast(d.atomic_id), &eargs);
    }
    if (d.atomic_id == @intFromEnum(ac.AtomicId.gelu)) {
        var gargs = ac.Args{ .out = out };
        return registry.run(.activation, @intCast(d.atomic_id), &gargs);
    }
    const gate = bufN(s, d.in.ref, d.in_len);
    const up = bufN(s, d.in.ref2, d.in_len);
    if (gate.len == 0) return error.Unsupported;
    var args = ac.Args{ .gate = gate, .up = up, .out = out, .simd_width = d.simd_width };
    return registry.run(.activation, @intCast(d.atomic_id), &args);
}

fn runElementwise(s: *Session, d: Dispatch) !void {
    const ew = @import("../kernels/transformer/elementwise.zig");
    const a = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    if (out.len == 0) return error.Unsupported;
    switch (d.sem_op) {
        graph.Params.mul_code => {
            const off: usize = d.params[0];
            const full = s.buffer(d.in.ref2);
            if (off + a.len > full.len) return error.ShapeMismatch;
            var args = ew.Args{ .a = a, .b = full[off .. off + a.len], .out = out, .simd_width = d.simd_width };
            return registry.run(.elementwise, @intCast(d.atomic_id), &args);
        },
        graph.Params.scale_code => {
            var args = ew.Args{ .a = a, .out = out, .scale = unpackFloat(d.params[0]), .simd_width = d.simd_width };
            return registry.run(.elementwise, @intCast(d.atomic_id), &args);
        },
        graph.Params.combine_code => {
            const b = bufN(s, d.in.ref2, d.in_len);
            var args = ew.Args{ .a = a, .b = b, .out = out, .scale = unpackFloat(d.params[0]), .simd_width = d.simd_width };
            return registry.run(.elementwise, @intCast(d.atomic_id), &args);
        },
        else => {
            const b = bufN(s, d.in.ref2, d.in_len);
            var args = ew.Args{ .a = a, .b = b, .out = out, .scale = unpackFloat(d.params[0]), .cap = unpackFloat(d.params[0]), .simd_width = d.simd_width };
            return registry.run(.elementwise, @intCast(d.atomic_id), &args);
        },
    }
}

fn runRope(s: *Session, d: Dispatch) !void {
    const buf = bufN(s, d.out.ref, d.out_len);
    if (buf.len == 0) return error.Unsupported;
    const pos: usize = d.params[0];
    const n_rot: usize = d.params[1];
    const head_dim: usize = d.params[3];
    const n_heads: usize = d.params[4];
    if (n_rot == 0 or head_dim == 0 or n_heads == 0) return;
    if (buf.len < n_heads * head_dim) return error.ShapeMismatch;
    const inv = s.ropeInvFreq(d.rope_cfg) orelse return error.Unsupported;
    ropeHeads(buf, n_heads, head_dim, n_rot, pos, inv, s.rope_cos, s.rope_sin);
}

fn runAttention(s: *Session, d: Dispatch) !void {
    if (d.atomic_id == @intFromEnum(attention.AtomicId.write_kv)) return runWriteKv(s, d);
    return runFlashAttention(s, d);
}

/// `write_kv` stores the layer's k/v projections at `pos`. The plan's KV buffer
/// holds the k cache then the v cache, each `n_kv_heads*max_seq*head_dim` f32.
fn runWriteKv(s: *Session, d: Dispatch) !void {
    const k = s.buffer(d.in.ref);
    const v = s.buffer(d.in.ref2);
    const kvbuf = s.bufferF16(d.kv);
    if (k.len == 0 or v.len == 0 or kvbuf.len == 0) return error.Unsupported;
    const pos: usize = d.params[0];
    const n_kv_heads: usize = d.params[4];
    const head_dim: usize = d.params[5];
    const views = splitKv(kvbuf, n_kv_heads, head_dim, d.max_seq) orelse return error.ShapeMismatch;
    attention.writeKv(k, v, views.k, views.v, n_kv_heads, head_dim, pos, d.max_seq);
}

/// `fused_flash_head`: one flash-attention pass per q head over the KV cache.
fn runFlashAttention(s: *Session, d: Dispatch) !void {
    const q = bufN(s, d.in.ref, d.in_len);
    const out = bufN(s, d.out.ref, d.out_len);
    const kvbuf = s.bufferF16(d.kv);
    if (q.len == 0 or out.len == 0 or kvbuf.len == 0) return error.Unsupported;
    const pos: usize = d.params[0];
    const scale = unpackFloat(d.params[1]);
    const n_heads: usize = d.params[3];
    const n_kv_heads: usize = d.params[4];
    const head_dim: usize = d.params[5];
    const t_start: usize = d.params[6];
    if (n_heads == 0 or head_dim == 0) return error.ShapeMismatch;
    if (q.len < n_heads * head_dim or out.len < n_heads * head_dim) return error.ShapeMismatch;
    const views = splitKv(kvbuf, n_kv_heads, head_dim, d.max_seq) orelse return error.ShapeMismatch;
    if (s.scores.len < d.max_seq) return error.ShapeMismatch;
    try attentionHeads(s, d, q, out, views.k, views.v, n_heads, n_kv_heads, head_dim, pos, d.max_seq, scale, t_start);
}

/// Split a plan KV buffer into its k and v halves.
fn splitKv(kvbuf: []f16, n_kv_heads: usize, head_dim: usize, max_seq: usize) ?struct { k: []f16, v: []f16 } {
    if (n_kv_heads == 0 or head_dim == 0 or max_seq == 0) return null;
    const per = n_kv_heads * max_seq * head_dim;
    if (kvbuf.len < 2 * per) return null;
    return .{ .k = kvbuf[0..per], .v = kvbuf[per..][0..per] };
}

/// Fill one `pos` RoPE cos/sin row (`angles` values) from the inverse
/// frequencies. Shared by the separate rope pass and the fused norm+rope pass.
fn ropeRows(pos: usize, inv: []const f32, angles: usize, cos_row: []f32, sin_row: []f32) void {
    std.debug.assert(inv.len >= angles and cos_row.len >= angles and sin_row.len >= angles);
    const t: f32 = @floatFromInt(pos);
    for (0..angles) |i| {
        const angle = t * inv[i];
        cos_row[i] = @cos(angle);
        sin_row[i] = @sin(angle);
    }
}

/// RoPE over `n_heads` heads of width `head_dim`: build cos/sin rows for `pos`
/// from the `zcom` inverse frequencies, then rotate each head in place.
fn ropeHeads(
    buf: []f32,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    inv: []const f32,
    cos_row: []f32,
    sin_row: []f32,
) void {
    const angles = n_rot / 2;
    std.debug.assert(buf.len >= n_heads * head_dim);
    ropeRows(pos, inv, angles, cos_row, sin_row);
    for (0..n_heads) |h| {
        rope.ropeProportionalFromTable(buf[h * head_dim ..][0..head_dim], cos_row[0..angles], sin_row[0..angles], head_dim, n_rot);
    }
}

fn attentionAllHeads(
    q: []const f32,
    out: []f32,
    k_cache: []const f16,
    v_cache: []const f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
    scores: []f32,
) void {
    for (0..n_heads) |q_head| {
        attention.fusedFlashAttentionHead(
            q[q_head * head_dim ..][0..head_dim],
            k_cache,
            v_cache,
            n_heads,
            n_kv_heads,
            head_dim,
            q_head,
            pos,
            max_seq,
            scale,
            t_start,
            scores,
            out[q_head * head_dim ..][0..head_dim],
        );
    }
}

/// Run the plan's selected head-attention AtomicOp for every q head through the
/// registry, so the chosen variant (`fused_flash_head`/`two_pass_head`)
/// executes. The shared `scores` scratch is reused across heads.
fn attentionHeads(
    s: *Session,
    d: Dispatch,
    q: []const f32,
    out: []f32,
    k_cache: []f16,
    v_cache: []f16,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq: usize,
    scale: f32,
    t_start: usize,
) !void {
    if (s.scores.len < max_seq) return error.ShapeMismatch;
    const scores = s.scores[0..max_seq];
    const id: u16 = @intCast(d.atomic_id);
    for (0..n_heads) |q_head| {
        var args = attention.Args{
            .q = q[q_head * head_dim ..][0..head_dim],
            .k_cache = k_cache,
            .v_cache = v_cache,
            .out = out[q_head * head_dim ..][0..head_dim],
            .scores = scores,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .q_head = q_head,
            .pos = pos,
            .max_seq = max_seq,
            .scale = scale,
            .t_start = t_start,
            .simd_width = d.simd_width,
        };
        try registry.run(.attention, id, &args);
    }
}

test "batched kv writes then per-row attention equal token-by-token" {
    const testing = std.testing;
    const n_heads: usize = 4;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 8;
    const max_seq: usize = 8;
    const m: usize = 4;
    const kv_row = n_kv_heads * head_dim;
    const per_q = n_heads * head_dim;
    const per_kv = n_kv_heads * max_seq * head_dim;

    var kproj: [m * kv_row]f32 = undefined;
    var vproj: [m * kv_row]f32 = undefined;
    var q: [m * per_q]f32 = undefined;
    for (&kproj, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.125;
    for (&vproj, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.0625;
    for (&q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.1;

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var batch_out: [m * per_q]f32 = [_]f32{0} ** (m * per_q);
    var token_out: [m * per_q]f32 = [_]f32{0} ** (m * per_q);

    var kvbuf: [2 * per_kv]f16 = [_]f16{0} ** (2 * per_kv);
    const views = splitKv(&kvbuf, n_kv_heads, head_dim, max_seq).?;
    for (0..m) |j| {
        attention.writeKv(kproj[j * kv_row ..][0..kv_row], vproj[j * kv_row ..][0..kv_row], views.k, views.v, n_kv_heads, head_dim, j, max_seq);
    }
    var scores: [max_seq]f32 = undefined;
    for (0..m) |j| {
        attentionAllHeads(q[j * per_q ..][0..per_q], batch_out[j * per_q ..][0..per_q], views.k, views.v, n_heads, n_kv_heads, head_dim, j, max_seq, scale, 0, &scores);
    }

    var kvbuf2: [2 * per_kv]f16 = [_]f16{0} ** (2 * per_kv);
    const views2 = splitKv(&kvbuf2, n_kv_heads, head_dim, max_seq).?;
    for (0..m) |j| {
        attention.writeKv(kproj[j * kv_row ..][0..kv_row], vproj[j * kv_row ..][0..kv_row], views2.k, views2.v, n_kv_heads, head_dim, j, max_seq);
        attentionAllHeads(q[j * per_q ..][0..per_q], token_out[j * per_q ..][0..per_q], views2.k, views2.v, n_heads, n_kv_heads, head_dim, j, max_seq, scale, 0, &scores);
    }

    for (0..m * per_q) |i| try testing.expectApproxEqAbs(token_out[i], batch_out[i], 1e-6);
}

test "executor ropeHeads matches ropeProportional" {
    const testing = std.testing;
    const base: f32 = 10000.0;
    const head_dim: usize = 8;
    const n_rot: usize = 8;
    const angles = n_rot / 2;
    const inv = try @import("../image/zcom.zig").ropeInvFreq(testing.allocator, base, head_dim, n_rot);
    defer testing.allocator.free(inv);

    var cos_row: [angles]f32 = undefined;
    var sin_row: [angles]f32 = undefined;
    var got = [_]f32{ 1, 0.5, -0.3, 2, 0.1, 0.2, -1, 1.5 };
    var want = got;
    ropeHeads(&got, 1, head_dim, n_rot, 3, inv, &cos_row, &sin_row);
    rope.ropeProportional(&want, 3, base, head_dim, n_rot);
    for (0..head_dim) |i| try testing.expectApproxEqAbs(want[i], got[i], 1e-4);
}

test "executor attentionAllHeads single head returns the cached value" {
    const testing = std.testing;
    const n_heads: usize = 1;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 2;
    const max_seq: usize = 4;
    const need = 2 * n_kv_heads * max_seq * head_dim;
    var kvbuf: [need]f16 = [_]f16{0} ** need;
    const views = splitKv(&kvbuf, n_kv_heads, head_dim, max_seq).?;

    const k = [_]f32{ 1, 0 };
    const v = [_]f32{ 2, 3 };
    attention.writeKv(&k, &v, views.k, views.v, n_kv_heads, head_dim, 0, max_seq);

    const q = [_]f32{ 1, 0 };
    var out: [head_dim]f32 = undefined;
    var scores: [max_seq]f32 = undefined;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    attentionAllHeads(&q, &out, views.k, views.v, n_heads, n_kv_heads, head_dim, 0, max_seq, scale, 0, &scores);

    try testing.expectApproxEqAbs(@as(f32, 2), out[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3), out[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1), scores[0], 1e-5);
}

test "embedding lookup hits the row cache and matches the uncached path" {
    const testing = std.testing;
    const q = @import("../kernels/linear/quant/root.zig");

    const row_len: usize = 4;
    const table = [_]f32{
        0.5, -1.0, 2.0,  0.25,
        1.5, 0.0,  -0.5, 4.0,
        3.0, 1.0,  1.0,  -2.0,
    };
    const dims = [_]u64{ row_len, 3 };
    const bytes = std.mem.sliceAsBytes(table[0..]);
    const view = TensorView{
        .name = "token_embd.weight",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = bytes.len,
        .mapped_slice = bytes,
    };
    const d = Dispatch{ .op = .dequant, .atomic_id = @intFromEnum(q.AtomicId.dequant), .tensor = 0 };

    var want: [row_len]f32 = undefined;
    try embedRow(null, d, &view, 1, &want);

    var cache = RowCache.init(testing.allocator, 1024);
    defer cache.deinit();
    var got: [row_len]f32 = undefined;
    try embedRow(&cache, d, &view, 1, &got);
    try testing.expectEqualSlices(f32, &want, &got);
    try testing.expectEqual(@as(usize, 1), cache.len());

    var hit: [row_len]f32 = undefined;
    try embedRow(&cache, d, &view, 1, &hit);
    try testing.expectEqualSlices(f32, &got, &hit);
    try testing.expectEqual(@as(usize, 1), cache.len());
}

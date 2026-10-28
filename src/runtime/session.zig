const std = @import("std");
const Io = std.Io;
const plan = @import("../plan/root.zig");
const zman = @import("../image/zman.zig");
const ztef = @import("../image/ztef.zig");
const zcom = @import("../image/zcom.zig");
const zmeta = @import("../image/zmeta_reader.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const TensorView = @import("../data/tensor_view.zig").TensorView;
const Pool = @import("pool.zig").Pool;
const Job = @import("pool.zig").Job;
const matvec = @import("../kernels/linear/matvec/root.zig");
const registry = @import("../kernels/registry.zig");
const residency_mod = @import("residency.zig");
const Ring = residency_mod.Ring;
const RowCache = @import("row_cache.zig").RowCache;
const Prefetch = @import("prefetch.zig").Prefetch;
const cpu = @import("../utils/cpu.zig");
const prefix_cache = @import("prefix_cache.zig");
const sampler = @import("sampler.zig");
const tools = @import("../architectures/core/tools.zig");

/// Per-request sampling controls (`temperature`/`top-k`/`top-p`). A generation
/// concern, never a plan decision.
pub const Sampling = sampler.Params;

/// A streaming sink: `write` receives UTF-8-safe decoded chunks while
/// `generate` runs. Same callback shape as the tuner/cache builders, so the
/// session stays decoupled from the socket.
pub const Sink = struct {
    ctx: *anyopaque,
    write: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!void,
};

/// UTF-8 sequence length implied by a lead byte: 1 for ASCII/invalid, 2/3/4 for
/// the multi-byte lead ranges.
fn utf8SeqLen(b: u8) usize {
    if (b < 0x80) return 1;
    if (b >> 5 == 0b110) return 2;
    if (b >> 4 == 0b1110) return 3;
    if (b >> 3 == 0b11110) return 4;
    return 1;
}

/// Length of the longest prefix of `bytes` that ends on a complete UTF-8
/// sequence boundary. An incomplete trailing sequence is held back.
fn validUtf8Prefix(bytes: []const u8) usize {
    var i: usize = 0;
    while (i < bytes.len) {
        const len = utf8SeqLen(bytes[i]);
        if (i + len > bytes.len) break;
        i += len;
    }
    return i;
}

/// Flush the longest valid UTF-8 prefix of `pending` through `sink`, holding
/// back an incomplete trailing sequence for the next token.
fn flushUtf8(pending: *std.ArrayList(u8), sink: Sink) !void {
    const n = validUtf8Prefix(pending.items);
    if (n == 0) return;
    try sink.write(sink.ctx, pending.items[0..n]);
    if (n == pending.items.len) {
        pending.clearRetainingCapacity();
    } else {
        const rest = pending.items.len - n;
        std.mem.copyForwards(u8, pending.items[0..rest], pending.items[n..]);
        pending.shrinkRetainingCapacity(rest);
    }
}

/// Diagnostic timing for one `generate` call. Reported to the service's client
/// only; it never feeds a decision, so the AOT invariant holds.
pub const GenerateStats = struct {
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    prefill_tokens: usize = 0,
    decode_tokens: usize = 0,
    /// Prompt tokens served from the KV cache (skipped prefill). Diagnostic only.
    cache_matched: usize = 0,
    /// Per-AtomicOp decode accounting. Diagnostic only (`run --profile`).
    profile: OpProfile = .{},
};

/// Diagnostic trace captured by one `generate` for the `--verbose` levels.
/// Additive and output-only: it never feeds a decision, so the AOT invariant
/// holds. Slices are allocated from the caller's `generate` allocator (the
/// daemon's per-request arena) and are consumed immediately by the caller.
pub const Verbose = struct {
    /// Accumulated decode time per transformer layer, indexed by `Dispatch.layer`.
    layer_ns: []u64 = &.{},
    /// Per generated token: id, chosen logit, and decoded piece (level >= 2).
    tokens: []Token = &.{},
    /// Prompt token ids (level >= 3).
    prompt_ids: []u32 = &.{},
    /// Raw reasoning text, regardless of `show_think` (level >= 3).
    think: []const u8 = &.{},

    pub const Token = struct {
        id: u32,
        logit: f32,
        piece: []const u8,
    };
};

/// Per-AtomicOp wall-clock accounting of one `generate`'s decode, measured at
/// run. Diagnostic only: it never feeds a decision (the AOT invariant holds).
pub const OpProfile = struct {
    matvec_ns: u64 = 0,
    matmul_ns: u64 = 0,
    attention_ns: u64 = 0,
    norm_ns: u64 = 0,
    activation_ns: u64 = 0,
    elementwise_ns: u64 = 0,
    rope_ns: u64 = 0,
    dequant_ns: u64 = 0,
    other_ns: u64 = 0,
    /// Residency (`advance` + `applyDispatch`) time.
    residency_ns: u64 = 0,
    /// Number of dispatches executed (the loop length).
    dispatches: u64 = 0,
    /// Weight bytes streamed by matvec/matmul dispatches (the decode's traffic).
    weight_bytes: u64 = 0,
    /// Sum of the requested thread splits (`parts`) over dispatches; divided by
    /// `dispatches` gives the average requested parallelism.
    parts_sum: u64 = 0,
    /// Dispatches whose `parts` is 1 (run on the calling thread only).
    serial_dispatches: u64 = 0,
    /// Time spent in serial (`parts <= 1`) dispatches.
    serial_ns: u64 = 0,
    /// Matvec dispatches: count, parts sum, and time.
    matvec_count: u64 = 0,
    matvec_parts_sum: u64 = 0,
    matvec_ns_only: u64 = 0,

    pub fn addOp(self: *OpProfile, op: registry.Op, ns: u64) void {
        switch (op) {
            .matvec => self.matvec_ns += ns,
            .matmul => self.matmul_ns += ns,
            .attention => self.attention_ns += ns,
            .norm => self.norm_ns += ns,
            .activation => self.activation_ns += ns,
            .elementwise => self.elementwise_ns += ns,
            .rope => self.rope_ns += ns,
            .dequant => self.dequant_ns += ns,
        }
        self.dispatches += 1;
    }

    /// Sum of the per-op times (excludes residency/dispatch overhead).
    pub fn opNs(self: OpProfile) u64 {
        return self.matvec_ns + self.matmul_ns + self.attention_ns + self.norm_ns +
            self.activation_ns + self.elementwise_ns + self.rope_ns + self.dequant_ns +
            self.other_ns;
    }
};

fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    if (rc != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Deterministic seed for the non-greedy sampler: the same request reproduces
/// the same run. The greedy path never reads the RNG.
fn samplingSeed(prompt: []const u8, p: Sampling) u64 {
    var h = std.hash.Wyhash.init(0x5A4C_4D53_414D_5000);
    h.update(prompt);
    h.update(std.mem.asBytes(&p.temperature));
    h.update(std.mem.asBytes(&p.top_k));
    h.update(std.mem.asBytes(&p.top_p));
    return h.final();
}

/// Sample from a logits row, growing the reusable `scratch` for the non-greedy
/// path only. Greedy (`temperature <= 0`) allocates nothing and is the exact
/// historical argmax.
fn sampleLogits(
    logits: []const f32,
    params: Sampling,
    scratch: *[]u32,
    allocator: std.mem.Allocator,
    rng: std.Random,
) !u32 {
    if (params.temperature <= 0) return sampler.argmax(logits);
    if (scratch.*.len < logits.len) {
        if (scratch.*.len > 0) allocator.free(scratch.*);
        scratch.* = try allocator.alloc(u32, logits.len);
    }
    return sampler.sample(logits, params, scratch.*, rng);
}

/// Compute workers available to the plan: the configured `--threads`. The
/// dedicated prefetch pool is **additional** to it (the plan records
/// `prefetch_threads` separately), so nothing is reserved; clamped so at least
/// one worker remains.
fn computeWorkers(total_threads: u16) u16 {
    return @max(@as(u16, 1), total_threads);
}

/// Arithmetic scope of the K/V: the ISA, the **device** and each dispatch's
/// kernel variant. Deliberately excludes `--threads`, `--ram`, `--vram`, the
/// per-dispatch `parts` and the prefill scheduling knobs: a row-split matvec and
/// a per-head attention leave the K/V unchanged. This is what lets one `.zpc`
/// artifact be **shared** across runtimes of the image that differ only in those
/// knobs. A change to the device, a kernel variant or the ISA invalidates the
/// cache (it changes the bytes).
fn kvArithmeticFingerprint(p: plan.AOTKernel) u64 {
    var h = std.hash.Wyhash.init(0x4B56_4152_4954_4800);
    h.update(std.mem.asBytes(&p.config.cpu_isa));
    h.update(&[_]u8{if (p.config.gpu) 1 else 0});
    h.update(std.mem.asBytes(&p.config.gpu_id));
    for (p.dispatch) |d| hashDispatchArith(&h, d);
    for (p.prefill) |d| hashDispatchArith(&h, d);
    return h.final();
}

fn hashDispatchArith(h: *std.hash.Wyhash, d: plan.Dispatch) void {
    h.update(&[_]u8{@intFromEnum(d.op)});
    h.update(&[_]u8{d.device});
    h.update(std.mem.asBytes(&d.atomic_id));
    h.update(std.mem.asBytes(&d.variant));
    h.update(&[_]u8{d.sem_op});
    h.update(std.mem.asBytes(&d.in_len));
    h.update(std.mem.asBytes(&d.out_len));
}

/// Physical K/V layout scope: the canonical shape list of every buffer the cache
/// stores, in a stable (layer) order. **Without** `max_seq` (the payload is
/// canonical, so the cache is shared across context lengths) and **without** the
/// plan's buffer `slot` (a plan-internal index; hashing it would spuriously miss
/// across plans with the same structure but different buffer numbering).
fn kvLayoutFingerprint(bufs: []const Session.KvBuffer) u64 {
    var h = std.hash.Wyhash.init(0x4B56_4C41_5900_0000);
    for (bufs) |kb| {
        h.update(std.mem.asBytes(&kb.n_kv_heads));
        h.update(std.mem.asBytes(&kb.head_dim));
    }
    return h.final();
}

/// Fraction of the plan's window the prefix cache may hold. The rest stays for
/// the streamed weights, so the cache degrades before the weights do.
const cache_window_div: u64 = 8;

/// Session-level API consumed by the edges: owns the plan, the resolved image
/// readers, the activation buffers and the derived `zcom` data. Decides nothing.
pub const Session = struct {
    allocator: std.mem.Allocator,
    io: Io,
    plan_reader: plan.Reader,
    image_path: []const u8,
    manifest: zman.Manifest,
    /// The image metadata (`zmeta`): architecture config + tokenizer keys.
    zmeta: ?zmeta.Zmeta = null,
    /// Tokenizer over `zmeta`; null when the image has no tokenizer keys.
    tokenizer: ?Tokenizer = null,
    /// Input token row the next embedding lookup must dequantize.
    token_id: u32 = 0,
    /// Prompt ids backing the current batched prefill block; set by `generate`
    /// and read by `executePrefill`.
    batch_ids: []const u32 = &.{},
    /// Current forward position; the executor records it so fused dispatches
    /// (the Composer's `qk_norm_rope_heads`) can build their RoPE rows.
    pos: u32 = 0,
    /// One ZTEF reader per image file (indexed by `ztef.Kind`).
    ztefs: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT,
    zcom_reader: ?zcom.Reader = null,
    /// Activation buffers, indexed by slot.
    buffers: []Buffer = &.{},
    /// Scratch for the executor: attention scores and RoPE cos/sin rows.
    scores: []f32 = &.{},
    rope_cos: []f32 = &.{},
    rope_sin: []f32 = &.{},
    /// GEMM scratch (one dequantized weight row) for batched prefill.
    mm_scratch: []f32 = &.{},
    /// Fused-MLP `ffn_mid` scratch (`ffn` f32), grown on first use. The fused
    /// MLP keeps the gated intermediate here instead of a session buffer, so it
    /// never round-trips through global memory.
    mlp_scratch: []f32 = &.{},
    /// Execution resources: a persistent worker pool, created lazily on the
    /// first dispatch whose plan `parts > 1`; the plan decides the degree.
    pool: ?Pool = null,
    /// Compute workers (including the calling executor thread) carved from
    /// `config.threads`; one worker is reserved for residency prefetch.
    compute_workers: u16 = 1,
    /// Live residency ring, built from the plan's `ResidencyGraph`; null when
    /// the plan carries no ring (`ring_block_bytes == 0`).
    residency: ?Ring = null,
    /// Per-tensor residency range (index into `graph.ranges`), or `plan.NONE`.
    /// Built once at open when the plan has an arena, so `tensorView` can serve
    /// a resident `STAGED` tensor from the arena without rescanning per
    /// dispatch. Empty when the plan has no arena.
    tensor_range: []u32 = &.{},
    /// Bounded row-LRU backing embedding lookups, sized by the plan's
    /// `row_cache_bytes`; null when the plan reserves no row budget.
    row_cache: ?RowCache = null,
    /// GPU co-processor (Vulkan), heap-owned so its address survives the
    /// by-value `open` return. Null when the plan does not enable the GPU.
    gpu: ?*@import("../gpu/root.zig").Device = null,
    /// Persistent VRAM weight handles, aligned with `plan.tensors`; a non-null
    /// entry means the plan kept that tensor resident and the executor reads it
    /// from the device instead of streaming. Empty on a CPU plan.
    resident: []?*@import("../gpu/root.zig").Resident = &.{},
    /// Background page-in worker borrowed by `residency`; heap-owned so its
    /// address survives the by-value `Session` return. Null when unused or
    /// unavailable, in which case `prefetch` nodes run inline.
    prefetch: ?*Prefetch = null,
    /// Set once the executor thread has applied the plan's `affinity[0]`;
    /// pinning happens in `executeAt` so the by-value `open` return is safe.
    pinned: bool = false,
    /// Timing of the most recent `generate` call (diagnostic only).
    last_stats: GenerateStats = .{},
    /// Requested verbose level of the most recent `generate` (0 = off).
    verbose: u8 = 0,
    /// Diagnostic trace of the most recent `generate` (verbose levels).
    trace: Verbose = .{},
    /// Accumulated per-op decode accounting (reset per `generate`).
    profile: OpProfile = .{},
    /// Cross-execution KV cache (prefix radix tree); heap-owned so its address
    /// survives the by-value `open` return. Null only when the plan has no KV or
    /// the backing journal could not be mapped.
    cache: ?*prefix_cache.Memo = null,
    /// Read-only mmap backing the loaded `.zpc` base segments. The cache holds
    /// slices into `cache_base.mmap.data`; it must outlive the cache.
    cache_base: ?@import("zpc.zig").Base = null,
    /// Writable, file-backed mmap journal for runtime-appended K/V segments. The
    /// K/V never lives in anonymous RAM.
    cache_journal: ?@import("zpc.zig").Scratch = null,
    /// Writable mmap holding the segment index (and its free lists); part of the
    /// RAM budget is reserved for it, so the index is not heap-allocated either.
    cache_meta: ?@import("zpc.zig").Scratch = null,
    /// Set when a commit changed the cache since the last `.zpc` flush; the
    /// service flushes on idle so a daemon's appends survive a restart without
    /// paying a write per request.
    cache_dirty: bool = false,
    /// Unique KV buffers the cache snapshots, derived from the plan's attention
    /// dispatches (layers sharing a cache via `source` collapse to one entry).
    kv_buffers: []KvBuffer = &.{},

    /// One physical KV cache: its session buffer slot and the shape its head-major
    /// K/V layout uses (`index = head*max_seq*head_dim + pos*head_dim + dim`).
    pub const KvBuffer = struct {
        slot: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        /// First transformer layer that uses this buffer. Used **only** to order
        /// the buffer list canonically, so the layout fingerprint and the
        /// canonical payload do not depend on the plan's dispatch order or the
        /// buffer numbering.
        layer: u32,

        fn lessByLayer(_: void, a: KvBuffer, b: KvBuffer) bool {
            return a.layer < b.layer;
        }
    };

    pub const Buffer = struct { bytes: []align(64) u8 };

    pub const Options = struct {
        /// Relative path to the image dir (e.g. `.zlm/images/e2b`).
        image_path: []const u8,
    };

    pub fn open(
        allocator: std.mem.Allocator,
        io: Io,
        plan_path: []const u8,
        opts: Options,
    ) !Session {
        var plan_reader = try plan.Reader.openFile(allocator, io, plan_path);
        errdefer plan_reader.deinit();

        var img_dir = try @import("../image/store.zig").openImage(io, opts.image_path);
        defer img_dir.close(io);
        var manifest = try zman.Manifest.open(allocator, img_dir, io);
        errdefer manifest.deinit();

        var self = Session{
            .allocator = allocator,
            .io = io,
            .plan_reader = plan_reader,
            .image_path = opts.image_path,
            .manifest = manifest,
        };
        try self.openZtefs();
        self.zmeta = zmeta.Zmeta.open(allocator, img_dir, io) catch null;
        if (self.zmeta) |*z| self.tokenizer = Tokenizer.fromZmeta(allocator, z) catch null;
        self.zcom_reader = zcom.Reader.open(allocator, img_dir, io) catch null;
        try self.allocBuffers();
        try self.allocScratch();
        self.openCache() catch {};
        // The tuner decided which dispatches run on the GPU; the runtime only
        // brings the device up. Absent/unavailable -> every dispatch stays CPU.
        if (self.plan_reader.plan.config.gpu) {
            if (self.allocator.create(@import("../gpu/root.zig").Device)) |g| {
                g.* = @import("../gpu/root.zig").Device.init(self.allocator, io);
                self.gpu = g;
            } else |_| {}
        }
        try self.openResident();
        self.warmGpu(self.plan_reader.plan.config.gpu_warmup);
        const total_threads: u16 = @max(@as(u16, 1), self.plan_reader.plan.config.threads);
        const prefetch_threads: u16 = self.plan_reader.plan.config.prefetch_threads;
        const has_ring = self.plan_reader.plan.graph.ring_block_bytes > 0 or prefetch_threads > 0;
        // The dedicated prefetch pool is separate from and additional to the
        // compute threads, so the compute pool keeps all `--threads`. The
        // prefetch is opt-in (`--prefetch-threads M`); the runtime only applies
        // the recorded count.
        self.compute_workers = computeWorkers(total_threads);
        if (has_ring) {
            self.residency = try Ring.init(self.allocator, self.plan_reader.plan.graph, prefetch_threads);
            try self.mapRingRanges();
        }
        if (self.plan_reader.plan.graph.row_cache_bytes > 0) {
            self.row_cache = RowCache.init(self.allocator, self.plan_reader.plan.graph.row_cache_bytes);
        }
        if (has_ring and prefetch_threads > 0) {
            // The plan's affinity list is `N + M` entries: the first N for the
            // compute pool, the last M for the prefetch workers.
            const affinity = self.plan_reader.plan.affinity;
            const start = @min(@as(usize, total_threads), affinity.len);
            const end = @min(start + prefetch_threads, affinity.len);
            if (self.residency) |*r| {
                if (self.allocator.create(Prefetch)) |p| {
                    if (Prefetch.init(self.allocator, self.io, prefetch_threads, affinity[start..end])) |pf| {
                        p.* = pf;
                        self.prefetch = p;
                        r.prefetch = p;
                    } else |_| self.allocator.destroy(p);
                } else |_| {}
            }
        }
        return self;
    }

    fn openZtefs(self: *Session) !void {
        var dbuf: [512]u8 = undefined;
        const dir_path = try @import("../image/store.zig").imageDir(&dbuf, self.image_path);
        for (self.manifest.files) |f| {
            var buf: [512]u8 = undefined;
            const zpath = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir_path, f.kind.filename() });
            self.ztefs[@intFromEnum(f.kind)] = ztef.Reader.open(self.allocator, self.io, zpath) catch continue;
        }
    }

    /// Upload the plan's VRAM-resident weights once. A missing device or a
    /// failed upload leaves the handle null, so the executor falls back to the
    /// streamed path (and ultimately the CPU kernel).
    fn openResident(self: *Session) !void {
        const p = self.plan_reader.plan;
        if (p.vram_resident.len == 0) return;
        self.resident = try self.allocator.alloc(?*@import("../gpu/root.zig").Resident, p.tensors.len);
        @memset(self.resident, null);
        const g = self.gpu orelse return;
        if (!g.available()) return;
        for (p.vram_resident) |vr| {
            if (vr.tensor >= p.tensors.len) continue;
            const view = self.tensorView(vr.tensor) orelse continue;
            var data = view.mapped_slice orelse continue;
            // A split op uploads only its GPU rows `[0, rows)`; the rest stays
            // on the CPU (the precise duplication).
            if (vr.rows > 0 and view.dims.len == 2 and data.len > 0) {
                const m: usize = @intCast(view.dims[1]);
                if (m > 0 and vr.rows < m) {
                    const row_bytes = data.len / m;
                    data = data[0 .. row_bytes * vr.rows];
                }
            }
            self.resident[vr.tensor] = g.uploadResident(data);
        }
    }

    /// Heavy-duty GPU warm-up (`gpu/13`): a **sustained** resident-GEMV load for
    /// `duration_ms` keeps the memory bus busy so a discrete device's clock
    /// ramps (a game-like duty cycle). Not N discrete bursts — the async ring is
    /// kept full so the device is never idle between dispatches. Results are
    /// discarded; the plan decided the duration, the runtime only applies it.
    fn warmGpu(self: *Session, duration_ms: u16) void {
        if (duration_ms == 0) return;
        const gpu_mod = @import("../gpu/root.zig");
        const g = self.gpu orelse return;
        if (!g.available()) return;
        // The largest resident weight carries the most memory traffic per op.
        var best: ?usize = null;
        var best_bytes: u64 = 0;
        for (self.resident, 0..) |r, i| {
            const res = r orelse continue;
            if (res.bytes > best_bytes) {
                best_bytes = res.bytes;
                best = i;
            }
        }
        const idx = best orelse return;
        const res = self.resident[idx].?;
        const view = self.tensorView(@intCast(idx)) orelse return;
        if (view.dims.len != 2) return;
        const k: usize = @intCast(view.dims[0]);
        const rows: usize = @intCast(view.dims[1]);
        if (rows == 0 or k == 0) return;
        const quant: gpu_mod.Quant = switch (view.ggml_type) {
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q8_0 => .q8_0,
            .q6_k => .q6_k,
            .q4_k => .q4_k,
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            else => return,
        };
        const x = self.allocator.alloc(f32, k) catch return;
        defer self.allocator.free(x);
        @memset(x, 0.5);
        const deadline = nowNs() +| @as(u64, duration_ms) * std.time.ns_per_ms;
        while (nowNs() < deadline) {
            var i: usize = 0;
            while (i < gpu_mod.RingSize) : (i += 1) {
                if (g.matVecResidentAsync(res, quant, rows, k, x) == null) break;
            }
            g.sync();
        }
    }

    fn allocBuffers(self: *Session) !void {
        const p = self.plan_reader.plan;
        const bufs = try self.allocator.alloc(Buffer, p.buffers.len);
        for (bufs, p.buffers) |*b, pb| {
            b.* = .{ .bytes = try self.allocator.alignedAlloc(u8, .@"64", @intCast(pb.bytes)) };
        }
        self.buffers = bufs;
    }

    /// Size the executor's scratch from the plan: attention needs `max_seq`
    /// scores; rope needs one cos/sin row of `n_rot/2` angles.
    fn allocScratch(self: *Session) !void {
        const p = self.plan_reader.plan;
        const qk_rope = @intFromEnum(@import("../kernels/transformer/norm.zig").AtomicId.qk_norm_rope_heads);
        var max_seq: usize = 0;
        var max_angles: usize = 0;
        var max_k: usize = 0;
        for (p.dispatch) |d| {
            max_seq = @max(max_seq, d.max_seq);
            if (d.op == .rope or d.atomic_id == qk_rope) {
                max_angles = @max(max_angles, d.params[1] / 2);
            }
        }
        for (p.prefill) |d| {
            max_seq = @max(max_seq, d.max_seq);
            if (d.op == .rope or d.atomic_id == qk_rope) {
                max_angles = @max(max_angles, d.params[1] / 2);
            }
            if (d.op == .matmul or d.op == .matvec) {
                max_k = @max(max_k, d.in_len);
            }
        }
        self.scores = try self.allocator.alloc(f32, @max(max_seq, 1));
        self.rope_cos = try self.allocator.alloc(f32, @max(max_angles, 1));
        self.rope_sin = try self.allocator.alloc(f32, @max(max_angles, 1));
        self.mm_scratch = try self.allocator.alloc(f32, @max(max_k, 1));
    }

    pub fn deinit(self: *Session) void {
        if (self.cache) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        if (self.cache_base) |*b| b.deinit();
        if (self.cache_journal) |*j| j.close();
        if (self.cache_meta) |*m| m.close();
        if (self.kv_buffers.len > 0) self.allocator.free(self.kv_buffers);
        if (self.prefetch) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        if (self.residency) |*r| r.deinit();
        if (self.tensor_range.len > 0) self.allocator.free(self.tensor_range);
        if (self.row_cache) |*r| r.deinit();
        if (self.pool) |*p| p.deinit(self.io);
        for (self.buffers) |b| self.allocator.free(b.bytes);
        if (self.buffers.len > 0) self.allocator.free(self.buffers);
        if (self.scores.len > 0) self.allocator.free(self.scores);
        if (self.rope_cos.len > 0) self.allocator.free(self.rope_cos);
        if (self.rope_sin.len > 0) self.allocator.free(self.rope_sin);
        if (self.mm_scratch.len > 0) self.allocator.free(self.mm_scratch);
        if (self.mlp_scratch.len > 0) self.allocator.free(self.mlp_scratch);
        if (self.zcom_reader) |*z| z.deinit();
        if (self.tokenizer) |*t| t.deinit();
        if (self.zmeta) |*z| z.deinit();
        for (&self.ztefs) |*z| {
            if (z.*) |*r| r.deinit();
        }
        for (self.resident) |r| {
            if (r) |res| if (self.gpu) |g| g.freeResident(res);
        }
        if (self.resident.len > 0) self.allocator.free(self.resident);
        if (self.gpu) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
        self.manifest.deinit();
        self.plan_reader.deinit();
    }

    pub fn aot(self: *const Session) plan.AOTKernel {
        return self.plan_reader.plan;
    }

    /// Timing of the most recent `generate` call (diagnostic only).
    pub fn generateStats(self: *const Session) GenerateStats {
        return self.last_stats;
    }

    /// Diagnostic trace captured by the most recent `generate` (verbose levels).
    pub fn verboseTrace(self: *const Session) Verbose {
        return self.trace;
    }

    /// One past the highest `Dispatch.layer` in the decode program (0 when the
    /// plan tags no layers). Sizes the per-layer timing buffer.
    fn maxDispatchLayer(self: *const Session) u32 {
        var n: u32 = 0;
        for (self.aot().dispatch) |d| {
            if (d.layer != plan.NONE) n = @max(n, d.layer + 1);
        }
        return n;
    }

    /// Apply a service residency state to the live engine; a no-op when the
    /// plan carries no residency graph. The window is the plan's tuned budget.
    pub fn setResidencyState(self: *Session, state: residency_mod.State) void {
        if (self.residency) |*r| r.setState(state, self.plan_reader.plan.graph.window_bytes);
    }

    /// The plan's recorded worker -> CPU mapping, indexed with wraparound.
    /// Null when the plan carries no affinity (off Linux or undetected), in
    /// which case the runtime pins nothing.
    pub fn affinityCpu(self: *const Session, index: usize) ?u16 {
        const affinity = self.plan_reader.plan.affinity;
        if (affinity.len == 0) return null;
        return affinity[index % affinity.len];
    }

    /// The plan's KV capacity: the maximum `max_seq` over its dispatches (0
    /// when the plan carries none).
    pub fn maxSeq(self: *const Session) u32 {
        var m: u32 = 0;
        for (self.plan_reader.plan.dispatch) |d| m = @max(m, d.max_seq);
        return m;
    }

    /// Map each plan tensor to the residency range serving it, once at open, by
    /// the `(kind, offset, length)` key the tuner's `findOrAddRange` dedups on.
    /// Only the plan's canonical (content-deduped) tensors have a matching
    /// Build the per-tensor ring range map: `tensor_range[i]` is the index into
    /// `graph.ranges` for a ringed tensor (`Range.block != NONE`); every other
    /// tensor stays `plan.NONE`. A no-op when the plan has no ring.
    fn mapRingRanges(self: *Session) !void {
        const p = self.plan_reader.plan;
        if ((p.graph.ring_block_bytes == 0 and p.config.prefetch_threads == 0) or p.tensors.len == 0) return;
        const map = try self.allocator.alloc(u32, p.tensors.len);
        errdefer self.allocator.free(map);
        @memset(map, plan.NONE);
        for (p.tensors, 0..) |t, i| {
            const r = &(self.ztefs[@intFromEnum(t.kind)] orelse continue);
            const e = r.find(t.name) orelse continue;
            for (p.graph.ranges, 0..) |rg, ri| {
                if (rg.kind != t.kind or rg.offset != e.offset or rg.length != e.length) continue;
                if (rg.block != plan.NONE) map[i] = @intCast(ri);
                break;
            }
        }
        self.tensor_range = map;
    }

    /// Ensure the ring blocks of a dispatch's weights are resident before the
    /// executor reads them; a no-op when the plan has no ring.
    pub fn ringEnsure(self: *Session, d: plan.Dispatch) void {
        const ring = if (self.residency) |*r| r else return;
        if (d.tensor != plan.NONE) self.ringRange(ring, d.tensor);
        if (d.tensor2 != plan.NONE) self.ringRange(ring, d.tensor2);
        if (d.tensor3 != plan.NONE) self.ringRange(ring, d.tensor3);
    }

    fn ringRange(self: *Session, ring: *Ring, tensor: u32) void {
        if (tensor >= self.tensor_range.len) return;
        const ri = self.tensor_range[tensor];
        if (ri == plan.NONE) return;
        ring.ensure(ri, &self.ztefs);
    }

    /// A weight `TensorView` for a plan tensor index: the ring pillar slice when
    /// the tensor's range is ringed and resident, else the image mmap.
    pub fn tensorView(self: *Session, idx: u32) ?TensorView {
        const p = self.plan_reader.plan;
        if (idx == plan.NONE or idx >= p.tensors.len) return null;
        const t = p.tensors[idx];
        const r = &(self.ztefs[@intFromEnum(t.kind)] orelse return null);
        const e = r.find(t.name) orelse return null;
        // Prefer the ring's pillar slice: for a compressed image it is already
        // decompressed, so we must NOT decompress the whole tensor here (that
        // would defeat the bounded ring and re-read the file).
        var mapped: []const u8 = &.{};
        var ringed = false;
        if (idx < self.tensor_range.len and self.tensor_range[idx] != plan.NONE) {
            if (self.residency) |*ring| {
                if (ring.slice(self.tensor_range[idx])) |a| {
                    mapped = a;
                    ringed = true;
                }
            }
        }
        if (!ringed) mapped = r.slice(e);
        return TensorView{
            .name = e.name,
            .dims = e.dims[0..e.n_dims],
            .ggml_type = e.ggml_type,
            .offset = e.offset,
            .byte_length = e.length,
            .mapped_slice = mapped,
        };
    }

    /// The persistent VRAM handle for a plan tensor, or null when it is not
    /// resident (the executor streams it instead).
    pub fn residentHandle(self: *Session, idx: u32) ?*@import("../gpu/root.zig").Resident {
        if (idx >= self.resident.len) return null;
        return self.resident[idx];
    }

    pub fn buffer(self: *Session, slot: u32) []f32 {
        if (slot >= self.buffers.len) return &.{};
        return std.mem.bytesAsSlice(f32, self.buffers[slot].bytes);
    }

    /// A KV-cache buffer as f16 elements (the KV is stored half-precision).
    pub fn bufferF16(self: *Session, slot: u32) []f16 {
        if (slot >= self.buffers.len) return &.{};
        return std.mem.bytesAsSlice(f16, self.buffers[slot].bytes);
    }

    /// Derive the unique KV buffers the memo snapshots from the plan's
    /// attention dispatches (both decode and batched prefill). Layers sharing a
    /// cache via `source` have the same `slot`, so they collapse to one entry.
    /// The result is ordered by the first layer that uses each buffer, so the
    /// layout fingerprint and canonical payload are dispatch/buffer-numbering
    /// independent.
    fn deriveKvBuffers(self: *Session) ![]KvBuffer {
        const p = self.plan_reader.plan;
        var list: std.ArrayList(KvBuffer) = .empty;
        errdefer list.deinit(self.allocator);
        try collectKvBuffers(self.allocator, &list, p.dispatch);
        try collectKvBuffers(self.allocator, &list, p.prefill);
        std.mem.sort(KvBuffer, list.items, {}, KvBuffer.lessByLayer);
        return list.toOwnedSlice(self.allocator);
    }

    fn collectKvBuffers(
        allocator: std.mem.Allocator,
        list: *std.ArrayList(KvBuffer),
        dispatch: []const plan.Dispatch,
    ) !void {
        for (dispatch) |d| {
            if (d.op != .attention or d.kv == plan.NONE) continue;
            const n_kv_heads = d.params[4];
            const head_dim = d.params[5];
            if (n_kv_heads == 0 or head_dim == 0 or d.max_seq == 0) continue;
            var found: ?usize = null;
            for (list.items, 0..) |kb, i| {
                if (kb.slot == d.kv) {
                    found = i;
                    break;
                }
            }
            if (found) |i| {
                if (d.layer < list.items[i].layer) list.items[i].layer = d.layer;
                continue;
            }
            try list.append(allocator, .{
                .slot = d.kv,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .max_seq = d.max_seq,
                .layer = d.layer,
            });
        }
    }

    /// Canonical K/V bytes per token across every buffer (f16).
    fn kvPerTokenBytes(self: *const Session) u64 {
        var n: u64 = 0;
        for (self.kv_buffers) |kb| {
            n += 2 * @as(u64, kb.n_kv_heads) * kb.head_dim * @sizeOf(f16);
        }
        return n;
    }

    /// Map the cache's backing regions and seed it from the `<image>_<fp>.zpc`
    /// when one exists. All K/V bytes live in these mmaps, never on the heap:
    /// - region 0: the writable journal (runtime appends);
    /// - region 1: the read-only `.zpc` base (zero-copy; reused as-is).
    fn openCache(self: *Session) !void {
        self.kv_buffers = try self.deriveKvBuffers();
        if (self.kv_buffers.len == 0) return;
        const zpc = @import("zpc.zig");
        const key = self.cacheKey();
        const per_token = self.kvPerTokenBytes();
        if (per_token == 0) return;

        // The cache competes for the plan's **window** (the budget leftover
        // after the fixed slots + KV + reserve), never a fixed fraction of the
        // whole RAM; it is therefore part of `planBudget`'s account (anti-OOM).
        // A slice of the window is reserved for the mapped index, the rest for
        // the mapped K/V.
        const window = self.plan_reader.plan.graph.window_bytes;
        const total = @max(window / cache_window_div, 4 << 20);
        const meta_bytes: u64 = @min(@max(total / 16, 256 << 10), total / 2);
        const kv_bytes = total - meta_bytes;

        const fp = key.fingerprint();
        const hex = std.fmt.bytesToHex(std.mem.asBytes(&fp), .lower);
        var zbuf: [512]u8 = undefined;
        const base_path = try @import("../image/store.zig").runtimePrefixCachePath(&zbuf, self.plan_reader.plan.image, &hex);

        // Process-unique, immediately-unlinked scratch maps, so `tune` and a
        // running daemon never truncate each other's mapping.
        const pid = std.posix.system.getpid();
        var jbuf: [640]u8 = undefined;
        const jpath = try std.fmt.bufPrint(&jbuf, "{s}.{d}.kv", .{ base_path, pid });
        const kv = zpc.Scratch.create(self.io, jpath, kv_bytes) catch return;
        var mbuf: [640]u8 = undefined;
        const mpath = try std.fmt.bufPrint(&mbuf, "{s}.{d}.meta", .{ base_path, pid });
        const meta = zpc.Scratch.create(self.io, mpath, meta_bytes) catch {
            var kv_map = kv;
            kv_map.close();
            return;
        };

        const m = try self.allocator.create(prefix_cache.Memo);
        m.* = prefix_cache.Memo.init(self.allocator, key, per_token);
        if (m.addMeta(meta.data)) |_| {} else |err| {
            self.allocator.destroy(m);
            var kv_map = kv;
            kv_map.close();
            var meta_map = meta;
            meta_map.close();
            return err;
        }
        if (m.addRegion(kv.data, true)) |_| {} else |err| {
            self.allocator.destroy(m);
            var kv_map = kv;
            kv_map.close();
            var meta_map = meta;
            meta_map.close();
            return err;
        }
        self.cache = m;
        self.cache_journal = kv;
        self.cache_meta = meta;

        const base_opt = zpc.openBase(self.allocator, self.io, base_path) catch null;
        if (base_opt) |base_val| {
            var base = base_val;
            if (prefix_cache.Key.eql(key, base.key)) {
                if (m.addRegion(base.mmap.data, false)) |region| {
                    for (base.records) |r| {
                        _ = m.addRaw(r.parent, r.begin, r.end, region, r.off) catch break;
                    }
                    m.trim();
                    self.cache_base = base;
                } else |_| {
                    base.deinit();
                }
            } else {
                base.deinit();
            }
        }
    }

    /// The cache scope: the weights content root and the K/V structure
    /// (arithmetic + canonical layout). It excludes `--threads`/`--ram`/`--vram`
    /// but includes the device, so runtimes of the image that differ only in the
    /// resource knobs share one `.zpc`.
    fn cacheKey(self: *Session) prefix_cache.Key {
        const p = self.plan_reader.plan;
        const root = self.manifest.content_root;
        return .{
            .model = std.hash.Wyhash.hash(0x6D6F_6465_6C00, &root),
            .arithmetic = kvArithmeticFingerprint(p),
            .layout = kvLayoutFingerprint(self.kv_buffers),
            .format = self.promptFormatFingerprint(),
            .dtype = 0,
        };
    }

    /// Canonical K/V byte size of `[begin, end)`: `(end - begin) * per-token`.
    pub fn kvRangeBytes(self: *const Session, begin: u32, end: u32) usize {
        if (end <= begin) return 0;
        return @intCast(@as(u64, end - begin) * self.kvPerTokenBytes());
    }

    /// Copy the canonical K/V of `[begin, end)` into `out`, **position-major**:
    /// for each position, for each buffer, `K` then `V` (each `n_kv_heads ×
    /// head_dim`, head-major). No `max_seq` stride, so the bytes are independent
    /// of the buffer capacity; and because the position is the outer loop, the
    /// bytes of any `[0, n)` prefix are contiguous, so a partial cache hit can
    /// load the first `n` positions by slicing the payload.
    pub fn saveKvRange(self: *const Session, begin: u32, end: u32, out: []u8) !void {
        var off: usize = 0;
        var pos: usize = begin;
        while (pos < end) : (pos += 1) {
            for (self.kv_buffers) |kb| {
                const buf = std.mem.bytesAsSlice(f16, self.buffers[kb.slot].bytes);
                const per = @as(usize, kb.n_kv_heads) * kb.max_seq * kb.head_dim;
                if (buf.len < 2 * per) return error.CacheKvMismatch;
                var half: usize = 0;
                while (half < 2) : (half += 1) {
                    var h: usize = 0;
                    while (h < kb.n_kv_heads) : (h += 1) {
                        const base = half * per + h * kb.max_seq * kb.head_dim + pos * kb.head_dim;
                        const bytes = std.mem.sliceAsBytes(buf[base .. base + kb.head_dim]);
                        if (off + bytes.len > out.len) return error.CacheKvMismatch;
                        @memcpy(out[off .. off + bytes.len], bytes);
                        off += bytes.len;
                    }
                }
            }
        }
        if (off != out.len) return error.CacheKvMismatch;
    }

    /// Inverse of `saveKvRange`: copy the canonical `[begin, end)` into the
    /// session buffers (re-striding into whatever `max_seq` they have).
    pub fn loadKvRange(self: *Session, begin: u32, end: u32, in: []const u8) !void {
        var off: usize = 0;
        var pos: usize = begin;
        while (pos < end) : (pos += 1) {
            for (self.kv_buffers) |kb| {
                const buf = std.mem.bytesAsSlice(f16, self.buffers[kb.slot].bytes);
                const per = @as(usize, kb.n_kv_heads) * kb.max_seq * kb.head_dim;
                if (buf.len < 2 * per) return error.CacheKvMismatch;
                var half: usize = 0;
                while (half < 2) : (half += 1) {
                    var h: usize = 0;
                    while (h < kb.n_kv_heads) : (h += 1) {
                        const base = half * per + h * kb.max_seq * kb.head_dim + pos * kb.head_dim;
                        const bytes = std.mem.sliceAsBytes(buf[base .. base + kb.head_dim]);
                        if (off + bytes.len > in.len) return error.CacheKvMismatch;
                        @memcpy(bytes, in[off .. off + bytes.len]);
                        off += bytes.len;
                    }
                }
            }
        }
        if (off != in.len) return error.CacheKvMismatch;
    }

    /// Materialize the matched path's K/V into the session buffers. A node may
    /// be matched partially (a divergent continuation branches inside it), so
    /// load only `[token_begin, min(token_end, matched))`; later path segments
    /// overwrite any shadowed range.
    fn loadFromCache(self: *Session, cache: *prefix_cache.Memo, path: []const u32, matched: u32) !void {
        const per = self.kvPerTokenBytes();
        for (path) |id| {
            const seg = cache.segment(id) orelse return error.CacheCorrupt;
            const end = @min(seg.token_end, matched);
            if (seg.token_begin >= end) continue;
            const payload = cache.payloadOf(id) orelse return error.CacheCorrupt;
            const n: usize = @intCast(@as(u64, end - seg.token_begin) * per);
            if (n > payload.len) return error.CacheCorrupt;
            try self.loadKvRange(seg.token_begin, end, payload[0..n]);
        }
    }

    /// Split a matvec/lm_head over `parts` workers; the runtime supplies
    /// resources, the plan already decided `parts`. When there is no pool (or
    /// `parts <= 1`) the range runs on the calling thread. `view_up` + `act`
    /// carry a fused two-weight matvec (`gate_up_fused`); for a plain matvec
    /// they are `null` / default and ignored.
    pub fn matVecPlan(
        self: *Session,
        view: *const TensorView,
        x: []const f32,
        out: []f32,
        rows: usize,
        row_start: usize,
        parts: u16,
        atomic_id: u16,
        plan_id: u16,
        simd_width: u8,
        view_up: ?*const TensorView,
        act: matvec.ActivationKind,
    ) !void {
        if (parts > 1 and self.pool == null) {
            const affinity = self.plan_reader.plan.affinity;
            const cpu_n = @min(@as(usize, self.compute_workers), affinity.len);
            const helpers: []const u16 = if (cpu_n > 1) affinity[1..cpu_n] else &.{};
            self.pool = Pool.init(self.allocator, self.io, self.compute_workers, helpers) catch null;
        }
        const job = Job{ .view = view, .view_up = view_up, .x = x, .out = out, .rows = rows, .row_start = row_start, .atomic_id = atomic_id, .plan = plan_id, .simd_width = simd_width, .act = act };
        if (self.pool) |*p| return p.run(job, @as(usize, parts));
        var args = matvec.Args{
            .view = view,
            .view_up = view_up,
            .x = x,
            .out = out,
            .begin = row_start,
            .end = row_start + rows,
            .plan = plan_id,
            .simd_width = simd_width,
            .act = act,
        };
        return @import("../kernels/registry.zig").run(.matvec, atomic_id, &args);
    }

    /// Batched matvec for prefill: one pool job processes all `m` input rows,
    /// each worker handling its output-row slice for every input row.
    pub fn matVecBatch(
        self: *Session,
        view: *const TensorView,
        x: []const f32,
        out: []f32,
        m: usize,
        n: usize,
        k: usize,
        parts: u16,
        atomic_id: u16,
        plan_id: u16,
    ) !void {
        if (parts > 1 and self.pool == null) {
            const affinity = self.plan_reader.plan.affinity;
            const cpu_n = @min(@as(usize, self.compute_workers), affinity.len);
            const helpers: []const u16 = if (cpu_n > 1) affinity[1..cpu_n] else &.{};
            self.pool = Pool.init(self.allocator, self.io, self.compute_workers, helpers) catch null;
        }
        const job = Job{ .view = view, .x = x, .out = out, .rows = n, .batch = m, .k = k, .atomic_id = atomic_id, .plan = plan_id };
        // A single row is a plain GEMV; the batched family needs `m >= 2`, so it
        // must not go through the pool (a one-token prefill delta can reach here).
        if (m <= 1) {
            var args = matvec.Args{ .view = view, .x = x[0..k], .out = out[0..n], .end = n, .plan = plan_id };
            return @import("../kernels/registry.zig").run(.matvec, atomic_id, &args);
        }
        if (self.pool) |*p| return p.run(job, @as(usize, parts));
        return matvec.matMulRows(view, x[0 .. m * k], out[0 .. m * n], 0, n);
    }

    /// Derived norm weight for a plan tensor index, if `zcom` has it. The match
    /// is by role + layer decoded from the tensor name, so the runtime never
    /// re-derives the weight from the raw (possibly misaligned) ZTEF bytes.
    pub fn normWeight(self: *Session, tensor: u32, layer: u32) ?[]const f32 {
        const z = if (self.zcom_reader) |*z| z else return null;
        const p = self.plan_reader.plan;
        if (tensor == plan.NONE or tensor >= p.tensors.len) return null;
        const parsed = @import("../architectures/root.zig").parse(self.manifest.arch, p.tensors[tensor].name) orelse return null;
        const role_code: u16 = @intFromEnum(parsed.role);
        const norm_layer: u32 = parsed.layer orelse 0xFFFF_FFFF;
        _ = layer;
        const reg = z.find(.norm_f32, role_code, norm_layer) orelse return null;
        const bytes = z.slice(reg);
        const aligned: []align(@alignOf(f32)) const u8 = @alignCast(bytes);
        return std.mem.bytesAsSlice(f32, aligned);
    }

    /// The `zcom` RoPE inverse-frequency table for a config id (keyed by
    /// `role = 0`, `layer = config id`). `n_rot/2` f32 values.
    pub fn ropeInvFreq(self: *Session, cfg: u32) ?[]const f32 {
        if (cfg == plan.NONE) return null;
        const z = if (self.zcom_reader) |*z| z else return null;
        const reg = z.find(.rope_inv_freq, 0, cfg) orelse return null;
        const bytes = z.slice(reg);
        const aligned: []align(@alignOf(f32)) const u8 = @alignCast(bytes);
        return std.mem.bytesAsSlice(f32, aligned);
    }

    /// The effective chat `Template`: the architecture's declared template with
    /// the compile-stored `zlm.chat.*` turn markers overlaid when present. The
    /// stored markers only describe the turn shape; the think decorations always
    /// come from the architecture. Null when neither source exists.
    fn effectiveTemplate(self: *Session) ?@import("../architectures/core/chat.zig").Template {
        const arch = @import("../architectures/root.zig").chatTemplate(self.manifest.arch);
        var template: @import("../architectures/core/chat.zig").Template = arch orelse .{};
        var have_markers = false;
        if (self.zmeta) |*z| {
            const user_prefix = z.getString(zmeta.CHAT_USER_PREFIX);
            const user_suffix = z.getString(zmeta.CHAT_USER_SUFFIX);
            const model_prefix = z.getString(zmeta.CHAT_MODEL_PREFIX);
            if (user_prefix != null and user_suffix != null and model_prefix != null) {
                template.user_prefix = user_prefix.?;
                template.user_suffix = user_suffix.?;
                template.model_prefix = model_prefix.?;
                have_markers = true;
            }
        }
        if (!have_markers and arch == null) return null;
        return template;
    }

    /// Fingerprint of the effective prompt format, used in the prefix-cache key
    /// so a recompile that changes the chat template (with unchanged weights)
    /// invalidates the `.zpc`.
    fn promptFormatFingerprint(self: *Session) u64 {
        var h = std.hash.Wyhash.init(0x5052_4F4D_5054_464D);
        if (self.effectiveTemplate()) |t| {
            h.update(t.user_prefix);
            h.update(t.user_suffix);
            h.update(t.model_prefix);
            h.update(t.think_prelude);
            h.update(t.think_prefill);
            h.update(t.think_round);
        }
        return h.final();
    }

    /// Wrap one user turn with the effective chat template. `think` selects the
    /// architecture's declared think wrapping.
    pub fn formatPrompt(self: *Session, allocator: std.mem.Allocator, text: []const u8, think: bool) ![]u8 {
        return self.formatPromptSoul(allocator, text, think, "");
    }

    /// Like `formatPrompt` with a `--soul` (system) turn prepended when the
    /// architecture declares a system role (`Template.soul_prefix`).
    pub fn formatPromptSoul(self: *Session, allocator: std.mem.Allocator, text: []const u8, think: bool, soul: []const u8) ![]u8 {
        const template = self.effectiveTemplate() orelse return error.UnsupportedArchitecture;
        return template.applyWithSoul(allocator, text, think, soul);
    }

    /// One message in a multi-turn (`chat`) conversation. `text` is the visible
    /// content; assistant turns may carry `calls`, and a tool turn answers one
    /// of them by id (the name is resolved from the preceding assistant turn
    /// when omitted).
    pub const Turn = struct {
        role: enum { user, assistant, tool },
        text: []const u8 = "",
        calls: []const tools.Call = &.{},
        tool_call_id: []const u8 = "",
        name: []const u8 = "",
    };

    /// Format a multi-turn conversation with the chat template: the soul turn
    /// (when the arch declares one), the tool declarations, each user turn
    /// `user_prefix + text + user_suffix`, each assistant turn
    /// `model_prefix + calls + text + user_suffix`, each tool turn as its
    /// dialect's response block, and finally `model_prefix` (+ the think
    /// prefill) to open the reply. The caller feeds the result to `generate`
    /// with `raw = true`.
    ///
    /// `budget` is the token budget for the prompt (the plan's sequence limit
    /// minus the reply reserve, see `maxSeq`); when the conversation exceeds it,
    /// the oldest user turn and the turns that answer it are dropped (the soul,
    /// the tools and the newest turn stay). `budget == 0` disables truncation.
    pub fn formatConversation(
        self: *Session,
        allocator: std.mem.Allocator,
        turns: []const Turn,
        think: bool,
        soul: []const u8,
        budget: usize,
    ) ![]u8 {
        return self.formatConversationWithTools(allocator, turns, think, soul, &.{}, budget);
    }

    /// Like `formatConversation`, with the client's tool declarations. The
    /// architecture's dialect renders the declarations, calls and responses.
    pub fn formatConversationWithTools(
        self: *Session,
        allocator: std.mem.Allocator,
        turns: []const Turn,
        think: bool,
        soul: []const u8,
        tool_list: []const tools.Tool,
        budget: usize,
    ) ![]u8 {
        const t = self.tokenizer orelse return self.buildConversation(allocator, turns, think, soul, tool_list);

        // Start with the whole conversation; drop the oldest user turn (and the
        // assistant/tool turns that belong to it) until it fits, keeping at
        // least the latest turn.
        var start: usize = 0;
        while (true) {
            const out = try self.buildConversation(allocator, turns[start..], think, soul, tool_list);
            if (budget == 0 or start + 1 >= turns.len) return out;
            const ids = t.encodeWithDefaults(allocator, out) catch return out;
            defer allocator.free(ids);
            if (ids.len <= budget) return out;
            allocator.free(out);
            const next = nextTurnStart(turns, start);
            if (next <= start) return self.buildConversation(allocator, turns[start..], think, soul, tool_list);
            start = next;
        }
    }

    /// The next truncation boundary: the first user turn after `start`.
    fn nextTurnStart(turns: []const Turn, start: usize) usize {
        var i = start + 1;
        while (i < turns.len and turns[i].role != .user) : (i += 1) {}
        return @min(i, turns.len - 1);
    }

    fn hasToolTurns(turns: []const Turn) bool {
        for (turns) |turn| {
            if (turn.role == .tool or turn.calls.len > 0) return true;
        }
        return false;
    }

    fn toolNameFor(turns: []const Turn, end: usize, id: []const u8) []const u8 {
        var i = end;
        while (i > 0) {
            i -= 1;
            for (turns[i].calls) |call| {
                if (call.id.len > 0 and std.mem.eql(u8, call.id, id)) return call.name;
            }
        }
        return "";
    }

    /// The formatter without the context truncation (the pure template walk).
    fn buildConversation(
        self: *Session,
        allocator: std.mem.Allocator,
        turns: []const Turn,
        think: bool,
        soul: []const u8,
        tool_list: []const tools.Tool,
    ) ![]u8 {
        const template = self.effectiveTemplate() orelse return error.UnsupportedArchitecture;
        const with_tools = tool_list.len > 0 or hasToolTurns(turns);
        const dialect: ?tools.Dialect = if (with_tools) @import("../architectures/root.zig").toolDialect(self.manifest.arch) else null;
        if (with_tools and dialect == null) return error.UnsupportedArchitecture;

        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);
        // One system turn carries the think token (top), the soul and the tool
        // declarations, matching the model's chat template.
        const has_system = template.soul_prefix.len > 0 and (soul.len > 0 or (think and template.think_prelude.len > 0) or tool_list.len > 0);
        if (has_system) {
            try buf.appendSlice(allocator, template.soul_prefix);
            if (think) try buf.appendSlice(allocator, template.think_prelude);
            try buf.appendSlice(allocator, soul);
            if (dialect) |d| {
                if (tool_list.len > 0) try buf.appendSlice(allocator, try d.renderDeclarations(allocator, tool_list));
            }
            try buf.appendSlice(allocator, template.soul_suffix);
        } else if (think) {
            try buf.appendSlice(allocator, template.think_prelude);
        }

        var model_open = false;
        for (turns, 0..) |turn, idx| {
            switch (turn.role) {
                .user => {
                    if (model_open) {
                        try buf.appendSlice(allocator, template.user_suffix);
                        model_open = false;
                    }
                    try buf.appendSlice(allocator, template.user_prefix);
                    try buf.appendSlice(allocator, std.mem.trim(u8, turn.text, " \t\r\n"));
                    try buf.appendSlice(allocator, template.user_suffix);
                },
                .assistant => {
                    if (!model_open) {
                        try buf.appendSlice(allocator, template.model_prefix);
                        model_open = true;
                    }
                    if (dialect) |d| {
                        for (turn.calls) |call| try buf.appendSlice(allocator, try d.renderCall(allocator, call));
                    }
                    try buf.appendSlice(allocator, std.mem.trim(u8, turn.text, " \t\r\n"));
                    if (turn.calls.len == 0) {
                        try buf.appendSlice(allocator, template.user_suffix);
                        model_open = false;
                    }
                },
                .tool => {
                    const d = dialect orelse return error.UnsupportedArchitecture;
                    if (!model_open) {
                        try buf.appendSlice(allocator, template.model_prefix);
                        model_open = true;
                    }
                    var name = turn.name;
                    if (name.len == 0) name = toolNameFor(turns, idx, turn.tool_call_id);
                    try buf.appendSlice(allocator, try d.renderResult(allocator, name, std.mem.trim(u8, turn.text, " \t\r\n")));
                },
            }
        }

        if (model_open) {
            if (turns.len > 0 and turns[turns.len - 1].role == .tool) {
                // A tool response leaves the model turn open; thinking starts a
                // fresh round inside it instead of a new `<|turn>model`.
                if (think) try buf.appendSlice(allocator, template.think_round);
            } else {
                try buf.appendSlice(allocator, template.user_suffix);
                try buf.appendSlice(allocator, template.model_prefix);
                if (!think) try buf.appendSlice(allocator, template.think_prefill);
            }
        } else {
            try buf.appendSlice(allocator, template.model_prefix);
            if (!think) try buf.appendSlice(allocator, template.think_prefill);
        }
        return buf.toOwnedSlice(allocator);
    }

    /// Split a model reply into visible content and the tool calls it emitted.
    /// `calls` is empty when the architecture has no dialect or the model made
    /// no calls.
    pub fn extractToolCalls(self: *Session, allocator: std.mem.Allocator, text: []const u8) !tools.Parsed {
        const d = @import("../architectures/root.zig").toolDialect(self.manifest.arch) orelse
            return .{ .content = try allocator.dupe(u8, text) };
        return d.parseCalls(allocator, text);
    }

    /// Encode text with the model tokenizer, adding BOS per the image default.
    pub fn encode(self: *Session, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const t = if (self.tokenizer) |*t| t else return error.NoTokenizer;
        return t.encodeWithDefaults(allocator, text);
    }

    /// Decode ids to printable text (special/control pieces dropped).
    pub fn decode(self: *Session, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        const t = if (self.tokenizer) |*t| t else return error.NoTokenizer;
        return t.decodeText(allocator, ids, true);
    }

    /// Execute one forward at `pos` and return the greedy next token id.
    pub fn stepAt(self: *Session, pos: u32) !u32 {
        return @import("executor.zig").executeAt(self, pos);
    }

    /// Execute one decode step at position 0 and return the next token id.
    pub fn step(self: *Session) !u32 {
        return @import("executor.zig").executeAt(self, 0);
    }

    /// A sampled token plus the logit that selected it (for `--verbose=info`).
    const Sampled = struct { id: u32, logit: f32 };

    /// Execute one forward at `pos` and sample the next token under `params`.
    fn sampleAt(
        self: *Session,
        pos: u32,
        params: Sampling,
        scratch: *[]u32,
        allocator: std.mem.Allocator,
        rng: std.Random,
    ) !Sampled {
        const logits = try @import("executor.zig").executeLogits(self, pos);
        const id = try sampleLogits(logits, params, scratch, allocator, rng);
        return .{ .id = id, .logit = logits[id] };
    }

    /// Execute a batched prefill block and sample the next token under `params`.
    fn samplePrefill(
        self: *Session,
        t0: u32,
        m: u32,
        params: Sampling,
        scratch: *[]u32,
        allocator: std.mem.Allocator,
        rng: std.Random,
    ) !Sampled {
        const logits = try @import("executor.zig").executePrefillLogits(self, t0, m);
        const id = try sampleLogits(logits, params, scratch, allocator, rng);
        return .{ .id = id, .logit = logits[id] };
    }

    /// Prompt text -> generated text. Wraps the prompt, encodes it, consults the
    /// KV memo (longest matching prefix resumes at `pos = matched`), prefills
    /// the delta, commits it, then samples (`sampling`) until EOS or `max_tokens`
    /// and detokenizes. `temperature <= 0` is greedy. A memo HIT returns the
    /// identical KV bytes, so the output equals the no-memo path.
    pub fn generate(
        self: *Session,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_tokens: usize,
        max_think: usize,
        sampling: Sampling,
        think: bool,
        show_think: bool,
        verbose: u8,
        soul: []const u8,
        /// When true, `prompt` is already a fully formatted prompt (a multi-turn
        /// conversation via `formatConversation`); skip the single-turn wrap.
        raw: bool,
        sink: ?Sink,
    ) ![]u8 {
        const t = if (self.tokenizer) |*t| t else return error.NoTokenizer;
        const formatted = if (raw) try allocator.dupe(u8, prompt) else try self.formatPromptSoul(allocator, prompt, think, soul);
        defer allocator.free(formatted);
        const prompt_ids = try t.encodeWithDefaults(allocator, formatted);
        defer allocator.free(prompt_ids);

        // A prompt may already end inside a reasoning round (the tool-response
        // template opens `<|channel>thought\n`). Seed the state from the prompt
        // instead of assuming the model emits the opener itself.
        var in_think = false;
        if (t.think_open_id) |o| if (t.think_close_id) |c| {
            for (prompt_ids) |id| {
                if (id == o) in_think = true else if (id == c) in_think = false;
            }
        };
        const prompt_in_think = in_think;

        // Diagnostic capture (verbose levels). Reset every call so a prior
        // request's trace never outlives its arena.
        self.verbose = verbose;
        self.trace = .{};
        if (verbose >= 1) {
            const layers = self.maxDispatchLayer() + 1;
            self.trace.layer_ns = try allocator.alloc(u64, layers);
            @memset(self.trace.layer_ns, 0);
        }
        var trace_tokens: std.ArrayList(Verbose.Token) = .empty;

        var prng = std.Random.DefaultPrng.init(samplingSeed(prompt, sampling));
        const rng = prng.random();
        var sample_scratch: []u32 = &.{};
        defer if (sample_scratch.len > 0) allocator.free(sample_scratch);

        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(allocator);

        const plan_seq = self.maxSeq();
        const limit: u32 = if (plan_seq == 0) std.math.maxInt(u32) else plan_seq;

        const cache = self.cache;
        var path: std.ArrayList(u32) = .empty;
        defer path.deinit(allocator);
        var matched: u32 = 0;
        if (cache) |m| {
            if (prompt_ids.len > 0) {
                matched = m.lookup(prompt_ids, &path) catch 0;
                if (matched > 0) {
                    self.loadFromCache(m, path.items, matched) catch {
                        m.unpin(path.items);
                        path.clearRetainingCapacity();
                        matched = 0;
                    };
                }
            }
        }

        var pos: u32 = matched;
        var next: u32 = undefined;
        var next_logit: f32 = 0;
        const batch = self.plan_reader.plan.config.prefill_mode == @intFromEnum(plan.PrefillMode.batch) and
            self.plan_reader.plan.prefill.len > 0 and
            self.plan_reader.plan.config.prefill_block >= 2;
        const prefill_start = nowNs();
        if (prompt_ids.len == 0) {
            self.token_id = t.bosId() orelse return error.MissingBos;
            const sm = try self.sampleAt(0, sampling, &sample_scratch, allocator, rng);
            next = sm.id;
            next_logit = sm.logit;
            pos = 1;
        } else if (pos < prompt_ids.len and pos < limit) {
            if (batch) {
                self.batch_ids = prompt_ids;
                const block = self.plan_reader.plan.config.prefill_block;
                while (pos < prompt_ids.len and pos < limit) {
                    const remain: u32 = @intCast(prompt_ids.len - pos);
                    const m: u32 = @min(@min(block, remain), limit - pos);
                    if (m == 0) break;
                    if (m == 1) {
                        // A one-token delta is a decode step; the batched GEMM
                        // path needs at least two rows (a cache hit can leave a
                        // single remaining prompt token).
                        self.token_id = prompt_ids[pos];
                        const sm = try self.sampleAt(pos, sampling, &sample_scratch, allocator, rng);
                        next = sm.id;
                        next_logit = sm.logit;
                        pos += 1;
                    } else {
                        const sm = try self.samplePrefill(pos, m, sampling, &sample_scratch, allocator, rng);
                        next = sm.id;
                        next_logit = sm.logit;
                        pos += m;
                    }
                }
            } else {
                while (pos < prompt_ids.len and pos < limit) : (pos += 1) {
                    self.token_id = prompt_ids[pos];
                    const sm = try self.sampleAt(pos, sampling, &sample_scratch, allocator, rng);
                    next = sm.id;
                    next_logit = sm.logit;
                }
            }
        } else if (pos > 0 and pos == prompt_ids.len and pos <= limit) {
            // Full-prefix HIT: the KV is loaded, but logits for the last token
            // were never stored. One forward reproduces them (and rewrites the
            // identical KV at `pos - 1`), so TTFT drops to ~one token.
            self.token_id = prompt_ids[pos - 1];
            const sm = try self.sampleAt(pos - 1, sampling, &sample_scratch, allocator, rng);
            next = sm.id;
            next_logit = sm.logit;
        }
        const prefill_end = nowNs();

        if (cache) |m| {
            const new_end: u32 = @min(pos, @as(u32, @intCast(prompt_ids.len)));
            if (new_end > matched) {
                const parent: u32 = if (path.items.len > 0) path.items[path.items.len - 1] else prefix_cache.NONE;
                var builder = PrefixCacheBuilder{ .s = self };
                m.commit(prompt_ids[0..new_end], parent, matched, new_end, .{
                    .ctx = @ptrCast(&builder),
                    .size = PrefixCacheBuilder.size,
                    .write = PrefixCacheBuilder.write,
                }) catch {};
            }
            if (path.items.len > 0) m.unpin(path.items);
            self.cache_dirty = true;
        }

        const eos = t.eosId();
        self.profile = .{};
        var count: usize = 0;
        var pending: std.ArrayList(u8) = .empty;
        defer pending.deinit(allocator);
        var think_count: usize = 0;
        var answer_count: usize = 0;
        while (pos < limit) {
            if (eos) |e| {
                if (next == e) break;
            }
            // The reasoning budget (`max_think`) is separate from the answer
            // budget (`max_tokens`): once the model has emitted `max_think`
            // reasoning tokens, force the close so the answer still gets its
            // full `max_tokens`.
            if (in_think and think_count >= max_think) {
                if (t.think_close_id) |c| {
                    next = c;
                    next_logit = 0;
                }
            }
            if (!in_think and answer_count >= max_tokens) break;
            try generated.append(allocator, next);
            if (t.think_open_id) |o| if (next == o) {
                in_think = true;
                think_count = 0;
            };
            if (t.think_close_id) |c| if (next == c) {
                in_think = false;
            };
            if (in_think) {
                think_count += 1;
            } else {
                answer_count += 1;
            }
            if (sink) |s| {
                try t.decodeTokenAppend(allocator, next, &pending, &in_think, show_think);
                try flushUtf8(&pending, s);
            }
            if (verbose >= 2) {
                const piece: []const u8 = t.decodeText(allocator, &[_]u32{next}, true) catch "";
                try trace_tokens.append(allocator, .{ .id = next, .logit = next_logit, .piece = piece });
            }
            count += 1;
            self.token_id = next;
            const sm = try self.sampleAt(pos, sampling, &sample_scratch, allocator, rng);
            next = sm.id;
            next_logit = sm.logit;
            pos += 1;
        }
        if (sink) |s| {
            if (pending.items.len > 0) try s.write(s.ctx, pending.items);
            pending.clearRetainingCapacity();
        }
        const decode_end = nowNs();
        if (verbose >= 2) self.trace.tokens = try trace_tokens.toOwnedSlice(allocator);
        if (verbose >= 3) {
            self.trace.prompt_ids = try allocator.dupe(u32, prompt_ids);
            self.trace.think = try t.decodeThink(allocator, generated.items);
        }
        self.last_stats = .{
            .prefill_ns = prefill_end -| prefill_start,
            .decode_ns = decode_end -| prefill_end,
            .prefill_tokens = prompt_ids.len - matched,
            .decode_tokens = count,
            .cache_matched = matched,
            .profile = self.profile,
        };
        return t.decodeToolText(allocator, generated.items, show_think, prompt_in_think);
    }

    /// Prefill `ids` (a known recurring prefix) and commit its K/V to the cache,
    /// without sampling. Used by `tune --prefix-cache` to warm the chat template.
    pub fn warmCache(self: *Session, ids: []const u32) !void {
        const m = self.cache orelse return;
        if (ids.len == 0) return;
        const plan_seq = self.maxSeq();
        const cap: u32 = if (plan_seq == 0) std.math.maxInt(u32) else plan_seq;
        const batch = self.plan_reader.plan.config.prefill_mode == @intFromEnum(plan.PrefillMode.batch) and
            self.plan_reader.plan.prefill.len > 0 and
            self.plan_reader.plan.config.prefill_block >= 2;

        var path: std.ArrayList(u32) = .empty;
        defer path.deinit(self.allocator);
        var matched: u32 = m.lookup(ids, &path) catch 0;
        if (matched > 0) {
            self.loadFromCache(m, path.items, matched) catch {
                m.unpin(path.items);
                path.clearRetainingCapacity();
                matched = 0;
            };
        }

        var pos: u32 = matched;
        if (batch) {
            self.batch_ids = ids;
            const block = self.plan_reader.plan.config.prefill_block;
            while (pos < ids.len and pos < cap) {
                const remain: u32 = @intCast(ids.len - pos);
                const mm: u32 = @min(@min(block, remain), cap - pos);
                if (mm == 0) break;
                _ = try @import("executor.zig").executePrefill(self, pos, mm);
                pos += mm;
            }
        } else {
            while (pos < ids.len and pos < cap) : (pos += 1) {
                self.token_id = ids[pos];
                _ = try self.stepAt(pos);
            }
        }

        if (pos > matched) {
            const parent: u32 = if (path.items.len > 0) path.items[path.items.len - 1] else prefix_cache.NONE;
            var builder = PrefixCacheBuilder{ .s = self };
            try m.commit(ids[0..pos], parent, matched, pos, .{
                .ctx = @ptrCast(&builder),
                .size = PrefixCacheBuilder.size,
                .write = PrefixCacheBuilder.write,
            });
        }
        if (path.items.len > 0) m.unpin(path.items);
    }

    /// Write the current cache to its `<image>_<prefix_cache_fp>.zpc` artifact. Used
    /// by `tune --prefix-cache`; a no-op when the cache is empty.
    pub fn writeCacheFile(self: *Session) !void {
        const m = self.cache orelse return error.NoCache;
        if (m.totalBytes() == 0) return;
        const fp = m.key.fingerprint();
        const hex = std.fmt.bytesToHex(std.mem.asBytes(&fp), .lower);
        var buf: [512]u8 = undefined;
        const path = try @import("../image/store.zig").runtimePrefixCachePath(&buf, self.plan_reader.plan.image, &hex);
        try @import("zpc.zig").writeBase(self.allocator, self.io, path, m);
        self.cache_dirty = false;
    }

    /// Whether a commit changed the cache since the last `.zpc` flush.
    pub fn cacheDirty(self: *const Session) bool {
        return self.cache_dirty;
    }

    /// Total live bytes held by the cache (0 when disabled/empty).
    pub fn cacheBytes(self: *const Session) u64 {
        return if (self.cache) |m| m.totalBytes() else 0;
    }
};

/// Writes a K/V delta into the cache's mmap journal and sizes it, without any
/// heap allocation of the payload.
const PrefixCacheBuilder = struct {
    s: *Session,

    fn size(ctx: *anyopaque, begin: u32, end: u32) u64 {
        const self: *PrefixCacheBuilder = @ptrCast(@alignCast(ctx));
        return self.s.kvRangeBytes(begin, end);
    }

    fn write(ctx: *anyopaque, begin: u32, end: u32, dst: []u8) anyerror!void {
        const self: *PrefixCacheBuilder = @ptrCast(@alignCast(ctx));
        try self.s.saveKvRange(begin, end, dst);
    }
};

test "validUtf8Prefix holds back an incomplete trailing sequence" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 0), validUtf8Prefix(""));
    try testing.expectEqual(@as(usize, 1), validUtf8Prefix("a"));
    try testing.expectEqual(@as(usize, 1), validUtf8Prefix("a\xC3"));
    try testing.expectEqual(@as(usize, 3), validUtf8Prefix("a\xC3\xA9"));
    try testing.expectEqual(@as(usize, 3), validUtf8Prefix("\xE2\x96\x81"));
    try testing.expectEqual(@as(usize, 0), validUtf8Prefix("\xE2\x96"));
    try testing.expectEqual(@as(usize, 4), validUtf8Prefix("\xF0\x9F\x98\x80"));
    try testing.expectEqual(@as(usize, 1), validUtf8Prefix("a\xF0\x9F\x98"));
}

test "flushUtf8 chunks a split multi-byte sequence only when complete" {
    const testing = std.testing;
    const Rec = struct {
        allocator: std.mem.Allocator,
        list: *std.ArrayList(u8),
        fn write(ctx: *anyopaque, bytes: []const u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            try self.list.appendSlice(self.allocator, bytes);
        }
    };
    var collected: std.ArrayList(u8) = .empty;
    defer collected.deinit(testing.allocator);
    var ctx = Rec{ .allocator = testing.allocator, .list = &collected };
    const sink = Sink{ .ctx = @ptrCast(&ctx), .write = Rec.write };

    var pending: std.ArrayList(u8) = .empty;
    defer pending.deinit(testing.allocator);
    try pending.append(testing.allocator, 0xC3);
    try flushUtf8(&pending, sink);
    try testing.expectEqual(@as(usize, 0), collected.items.len);
    try testing.expectEqual(@as(usize, 1), pending.items.len);

    try pending.append(testing.allocator, 0xA9);
    try flushUtf8(&pending, sink);
    try testing.expectEqualStrings("\xC3\xA9", collected.items);
    try testing.expectEqual(@as(usize, 0), pending.items.len);
}

test "computeWorkers keeps all --threads and at least one" {
    const testing = std.testing;
    try testing.expectEqual(@as(u16, 8), computeWorkers(8));
    try testing.expectEqual(@as(u16, 2), computeWorkers(2));
    try testing.expectEqual(@as(u16, 1), computeWorkers(1));
    try testing.expectEqual(@as(u16, 1), computeWorkers(0));
}

test "prefill matVecBatch routes a parts>1 projection through the pool" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const cfg = plan.Config{
        .threads = 4,
        .ram_bytes = 1 << 30,
        .vram_bytes = 0,
        .gpu = false,
        .cpu_isa = 0,
        .gpu_id = 0,
        .prefill_mode = @intFromEnum(plan.PrefillMode.batch),
        .prefill_block = 4,
    };
    try plan.write(testing.allocator, tmp.dir, testing.io, .{
        .config = cfg,
        .fingerprint = cfg.fingerprint(),
        .image = "test",
        .tensors = &.{},
        .buffers = &.{},
        .dispatch = &.{},
        .prefill = &.{},
        .graph = .{ .ranges = &.{}, .window_bytes = 0 },
        .affinity = &.{},
    });

    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = try plan.Reader.open(testing.allocator, tmp.dir, testing.io),
        .image_path = "",
        .manifest = undefined,
        .compute_workers = 4,
    };
    defer {
        if (s.pool) |*p| p.deinit(s.io);
        s.plan_reader.deinit();
    }

    const k: usize = 32;
    const n: usize = 8;
    const m: usize = 3;
    var weights: [k * n]f32 = undefined;
    for (&weights, 0..) |*w, i| {
        const v: i32 = @as(i32, @intCast(i % 7)) - 3;
        w.* = @as(f32, @floatFromInt(v)) * 0.25;
    }
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ k, n };
    const view = TensorView{
        .name = "prefill.w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    var x: [m * k]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const q: i32 = @as(i32, @intCast(i % 11)) - 5;
        v.* = @as(f32, @floatFromInt(q)) * 0.125;
    }
    var expected: [m * n]f32 = [_]f32{0} ** (m * n);
    var got: [m * n]f32 = [_]f32{0} ** (m * n);
    try matvec.matMulRows(&view, &x, &expected, 0, n);

    try testing.expect(s.pool == null);
    try s.matVecBatch(&view, &x, &got, m, n, k, 4, @intFromEnum(matvec.AtomicId.mat_vec), 0);
    try testing.expect(s.pool != null);
    try testing.expect(s.pool.?.spawned > 0);
    try testing.expectEqualSlices(f32, &expected, &got);
}

test "prefill matVecBatch handles a single row (m=1) with a pool" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const cfg = plan.Config{
        .threads = 4,
        .ram_bytes = 1 << 30,
        .vram_bytes = 0,
        .gpu = false,
        .cpu_isa = 0,
        .gpu_id = 0,
        .prefill_mode = @intFromEnum(plan.PrefillMode.batch),
        .prefill_block = 4,
    };
    try plan.write(testing.allocator, tmp.dir, testing.io, .{
        .config = cfg,
        .fingerprint = cfg.fingerprint(),
        .image = "test",
        .tensors = &.{},
        .buffers = &.{},
        .dispatch = &.{},
        .prefill = &.{},
        .graph = .{ .ranges = &.{}, .window_bytes = 0 },
        .affinity = &.{},
    });

    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = try plan.Reader.open(testing.allocator, tmp.dir, testing.io),
        .image_path = "",
        .manifest = undefined,
        .compute_workers = 4,
    };
    defer {
        if (s.pool) |*p| p.deinit(s.io);
        s.plan_reader.deinit();
    }

    const k: usize = 32;
    const n: usize = 8;
    const m: usize = 1;
    var weights: [k * n]f32 = undefined;
    for (&weights, 0..) |*w, i| {
        const v: i32 = @as(i32, @intCast(i % 7)) - 3;
        w.* = @as(f32, @floatFromInt(v)) * 0.25;
    }
    const data = std.mem.sliceAsBytes(weights[0..]);
    const dims = [_]u64{ k, n };
    const view = TensorView{
        .name = "prefill.w",
        .dims = &dims,
        .ggml_type = .f32,
        .offset = 0,
        .byte_length = data.len,
        .mapped_slice = data,
    };
    var x: [m * k]f32 = undefined;
    for (&x, 0..) |*v, i| {
        const q: i32 = @as(i32, @intCast(i % 11)) - 5;
        v.* = @as(f32, @floatFromInt(q)) * 0.125;
    }
    var expected: [m * n]f32 = [_]f32{0} ** (m * n);
    var got: [m * n]f32 = [_]f32{0} ** (m * n);
    try matvec.matMulRows(&view, &x, &expected, 0, n);

    // m == 1 must take the single-row path even with parts > 1 (it previously
    // went through the pool's batched family and returned ShapeMismatch).
    try s.matVecBatch(&view, &x, &got, m, n, k, 4, @intFromEnum(matvec.AtomicId.mat_vec), 0);
    try testing.expectEqualSlices(f32, &expected, &got);
}

test "formatConversation wraps turns with the template and the soul" {
    const testing = std.testing;
    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = undefined,
        .image_path = "",
        .manifest = undefined,
    };
    s.manifest.arch = "gemma4";
    s.zmeta = null;

    const turns = [_]Session.Turn{
        .{ .role = .user, .text = "Hi" },
        .{ .role = .assistant, .text = "Hello" },
        .{ .role = .user, .text = "How are you?" },
    };
    const out = try s.formatConversation(testing.allocator, &turns, false, "Be terse.", 0);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<|turn>system\nBe terse.<turn|>\n" ++
            "<start_of_turn>user\nHi<end_of_turn>\n" ++
            "<start_of_turn>model\nHello<end_of_turn>\n" ++
            "<start_of_turn>user\nHow are you?<end_of_turn>\n" ++
            "<start_of_turn>model\n",
        out,
    );

    // No soul: the system turn is omitted.
    const no_soul = try s.formatConversation(testing.allocator, turns[0..1], false, "", 0);
    defer testing.allocator.free(no_soul);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        no_soul,
    );
}

test "formatPrompt uses the stored markers, else the arch fallback" {
    const testing = std.testing;
    const format = @import("../gguf/format.zig");

    var md = std.StringHashMap(format.MetadataValue).init(testing.allocator);
    defer md.deinit();
    try md.put(zmeta.CHAT_USER_PREFIX, .{ .string = "<|turn>user\n" });
    try md.put(zmeta.CHAT_USER_SUFFIX, .{ .string = "<turn|>\n" });
    try md.put(zmeta.CHAT_MODEL_PREFIX, .{ .string = "<|turn>model\n" });

    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = undefined,
        .image_path = "",
        .manifest = undefined,
    };
    s.manifest.arch = "gemma4";
    s.zmeta = .{
        .allocator = testing.allocator,
        .mmap = undefined,
        .metadata = md,
        .keys = &.{},
    };

    const stored = try s.formatPrompt(testing.allocator, "Hi", false);
    defer testing.allocator.free(stored);
    try testing.expectEqualStrings("<|turn>user\nHi<turn|>\n<|turn>model\n", stored);

    s.zmeta = null;
    const fallback = try s.formatPrompt(testing.allocator, "Hi", false);
    defer testing.allocator.free(fallback);
    try testing.expectEqualStrings("<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n", fallback);
}

test "formatPrompt merges stored markers with the arch think decoration" {
    const testing = std.testing;
    const format = @import("../gguf/format.zig");

    var md = std.StringHashMap(format.MetadataValue).init(testing.allocator);
    defer md.deinit();
    try md.put(zmeta.CHAT_USER_PREFIX, .{ .string = "<|turn>user\n" });
    try md.put(zmeta.CHAT_USER_SUFFIX, .{ .string = "<turn|>\n" });
    try md.put(zmeta.CHAT_MODEL_PREFIX, .{ .string = "<|turn>model\n" });

    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = undefined,
        .image_path = "",
        .manifest = undefined,
    };
    s.manifest.arch = "gemma4";
    s.zmeta = .{
        .allocator = testing.allocator,
        .mmap = undefined,
        .metadata = md,
        .keys = &.{},
    };

    // Stored turn markers + the arch think prelude on think ON.
    const on = try s.formatPrompt(testing.allocator, "Hi", true);
    defer testing.allocator.free(on);
    try testing.expectEqualStrings(
        "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
        on,
    );

    // Fallback to the arch template (markers + think) when no stored markers.
    s.zmeta = null;
    const fallback_on = try s.formatPrompt(testing.allocator, "Hi", true);
    defer testing.allocator.free(fallback_on);
    try testing.expectEqualStrings(
        "<|turn>system\n<|think|>\n<turn|>\n<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        fallback_on,
    );
}

/// Fill a KV buffer half-major (`index = half*per + h*max_seq*head_dim +
/// pos*head_dim + d`) with a deterministic per-position value.
fn fillKvPattern(bytes: []align(64) u8, n_kv_heads: u32, max_seq: u32, head_dim: u32, bias: f32) void {
    const f = std.mem.bytesAsSlice(f16, bytes);
    const per = @as(usize, n_kv_heads) * max_seq * head_dim;
    var half: usize = 0;
    while (half < 2) : (half += 1) {
        var h: usize = 0;
        while (h < n_kv_heads) : (h += 1) {
            var pos: usize = 0;
            while (pos < max_seq) : (pos += 1) {
                var d: usize = 0;
                while (d < head_dim) : (d += 1) {
                    const v: f32 = bias + @as(f32, @floatFromInt(pos * 16 + half * 8 + h * 4 + d));
                    f[half * per + h * max_seq * head_dim + pos * head_dim + d] = @floatCast(v);
                }
            }
        }
    }
}

test "saveKvRange/loadKvRange: a prefix is contiguous across buffers" {
    const testing = std.testing;
    const b0 = try testing.allocator.alignedAlloc(u8, .@"64", 2 * 2 * 8 * 4 * @sizeOf(f16));
    defer testing.allocator.free(b0);
    const b1 = try testing.allocator.alignedAlloc(u8, .@"64", 2 * 1 * 8 * 4 * @sizeOf(f16));
    defer testing.allocator.free(b1);
    fillKvPattern(b0, 2, 8, 4, 0);
    fillKvPattern(b1, 1, 8, 4, 1000);

    var s = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = undefined,
        .image_path = "",
        .manifest = undefined,
    };
    var bufs = [_]Session.Buffer{ .{ .bytes = b0 }, .{ .bytes = b1 } };
    var kbs = [_]Session.KvBuffer{
        .{ .slot = 0, .n_kv_heads = 2, .head_dim = 4, .max_seq = 8, .layer = 0 },
        .{ .slot = 1, .n_kv_heads = 1, .head_dim = 4, .max_seq = 8, .layer = 1 },
    };
    s.buffers = &bufs;
    s.kv_buffers = &kbs;

    const per = s.kvPerTokenBytes();
    const full = try testing.allocator.alloc(u8, 8 * per);
    defer testing.allocator.free(full);
    try s.saveKvRange(0, 8, full);
    const prefix = try testing.allocator.alloc(u8, 4 * per);
    defer testing.allocator.free(prefix);
    try s.saveKvRange(0, 4, prefix);

    // The bytes of [0, 4) are a contiguous slice of the [0, 8) payload.
    try testing.expectEqualSlices(u8, prefix, full[0 .. 4 * per]);

    // Loading just that slice reproduces positions [0, 4) exactly.
    const c0 = try testing.allocator.alignedAlloc(u8, .@"64", b0.len);
    defer testing.allocator.free(c0);
    const c1 = try testing.allocator.alignedAlloc(u8, .@"64", b1.len);
    defer testing.allocator.free(c1);
    @memset(@as([]u8, c0), 0);
    @memset(@as([]u8, c1), 0);
    var s2 = Session{
        .allocator = testing.allocator,
        .io = testing.io,
        .plan_reader = undefined,
        .image_path = "",
        .manifest = undefined,
    };
    var bufs2 = [_]Session.Buffer{ .{ .bytes = c0 }, .{ .bytes = c1 } };
    s2.buffers = &bufs2;
    s2.kv_buffers = &kbs;
    try s2.loadKvRange(0, 4, full[0 .. 4 * per]);
    const back = try testing.allocator.alloc(u8, 4 * per);
    defer testing.allocator.free(back);
    try s2.saveKvRange(0, 4, back);
    try testing.expectEqualSlices(u8, prefix, back);
}

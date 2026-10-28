const std = @import("std");
const Io = std.Io;
const MappedFile = @import("../utils/mmap.zig").MappedFile;
const BinaryReader = @import("../utils/binary_reader.zig").BinaryReader;
const binary_writer = @import("../utils/binary_writer.zig");
const appendU16 = binary_writer.appendU16;
const appendU32 = binary_writer.appendU32;
const appendU64 = binary_writer.appendU64;
const ztef = @import("../image/ztef.zig");
const op = @import("../kernels/registry.zig").Op;

pub const MAGIC = "ZPLN";
/// Bumped when the `MatVecPlan` catalog ordering changes (the dispatch variant
/// is that enum's ordinal), so a stale plan is re-tuned rather than silently
/// interpreted with shifted variant ids. v21 -> v22: q8_0 joined the generated
/// kernel x variant grid (its hand-written `q8_0_u4`/`q8_0_t2` plans retired),
/// which renumbered the generated plan ids. v22 -> v23: q6_k joined the grid
/// (its `q6_k_u4`/`q6_k_t2` plans retired), likewise renumbering the ids.
/// v23 -> v24: the F/BF families (f32/f16/bf16) joined the grid, appending
/// their `F32_fp_*`/`F16_fp_*`/`BF16_fp_*` plans after the Q families.
/// v26 -> v27: the fused SwiGLU MLP — `Dispatch.tensor3` (the third weight)
/// joined the serialized dispatch.
/// v27 -> v28: `Config.residency_workers` (a 0/1 flag) became
/// `Config.prefetch_threads` (an explicit dedicated-worker count).
/// v28 -> v29: `ResidencyGraph.arena_bytes` (the staging arena for the streamed
/// weights) joined the serialized graph, and the `STAGED` range flag.
/// v29 -> v30: `Config.residency_bytes` (the explicit `--residency` ceiling for
/// the resident weight working set) joined the config; `0` = no residency.
/// v30 -> v31: the three-pillar weight ring (`runtime/15`) -- `Range.block` (the
/// consumption-order block ordinal) plus `ResidencyGraph.ring_block_bytes` and
/// `ring_k` joined the graph; `0` = no ring.
pub const VERSION: u32 = 31;
pub const NAME = "plan.zplan";
/// Absent-reference sentinel (tensor/buffer/dep).
pub const NONE: u32 = 0xFFFF_FFFF;
/// VRAM the runtime must leave for the Vulkan driver (anti-OOM headroom). The
/// tuner packs the resident set into `config.vram_bytes -| VRAM_DRIVER_HEADROOM`.
pub const VRAM_DRIVER_HEADROOM: u64 = 64 * 1024 * 1024;

/// Prefill policy the tuner writes and the runtime applies verbatim.
pub const PrefillMode = enum(u8) { token = 0, batch = 1 };

pub const Config = struct {
    threads: u16,
    ram_bytes: u64,
    vram_bytes: u64,
    gpu: bool,
    cpu_isa: u64,
    gpu_id: u32,
    /// Prefill policy chosen by the tuner (`token` = one token at a time,
    /// `batch` = the prompt in `prefill_block`-row forwards). The runtime reads
    /// it and decides nothing.
    prefill_mode: u8 = 0,
    /// Max rows M the batched prefill processes per forward; 0 when unused.
    prefill_block: u32 = 0,
    /// Dedicated residency-prefetch workers (`--prefetch-threads`), separate
    /// from and additional to the compute `threads`: the process runs
    /// `threads + prefetch_threads`. `0` (absent) runs the prefetch inline on
    /// the executor; the runtime reads it and decides nothing.
    prefetch_threads: u8 = 0,
    /// Heavy-duty GPU warm-up duration (ms): at load the runtime runs a
    /// **sustained** resident-GEMV load to ramp the device memory clock
    /// (`gpu/13`); `0` = off. Decided by the tuner when `--gpu`.
    gpu_warmup: u16 = 0,
    /// The explicit `--residency` ceiling for the resident weight working set
    /// (`ResidencyGraph.window_bytes`), in bytes. `0` = no residency: the
    /// runtime streams every weight on demand from the mmap, with no window,
    /// no staging arena and no prefetch. Decided by the tuner (`--residency`).
    residency_bytes: u64 = 0,

    pub fn fingerprint(self: Config) u64 {
        var h = std.hash.Wyhash.init(0x5A4C5A504C414E31);
        h.update(std.mem.asBytes(&self.threads));
        h.update(std.mem.asBytes(&self.ram_bytes));
        h.update(std.mem.asBytes(&self.vram_bytes));
        h.update(&[_]u8{if (self.gpu) 1 else 0});
        h.update(std.mem.asBytes(&self.cpu_isa));
        h.update(std.mem.asBytes(&self.gpu_id));
        h.update(&[_]u8{self.prefill_mode});
        h.update(std.mem.asBytes(&self.prefill_block));
        h.update(&[_]u8{self.prefetch_threads});
        h.update(std.mem.asBytes(&self.residency_bytes));
        return h.final();
    }
};

/// What a dispatch reads/writes: a weight tensor in the image, or an activation
/// buffer in the session. `kind` discriminates; `ref` is an index into
/// `AOTKernel.tensors` (tensor) or `AOTKernel.buffers` (buffer).
pub const Binding = struct {
    pub const Kind = enum(u8) { none = 0, tensor = 1, buffer = 2 };
    pub fn kindOf(k: Kind) Kind {
        return k;
    }

    kind: Kind,
    ref: u32 = 0xFFFF_FFFF,
    /// Additional operand (e.g. the second activation for a fused op).
    ref2: u32 = 0xFFFF_FFFF,
};

/// One item of the AOTKernel: a resolved operation to dispatch.
pub const Dispatch = struct {
    op: op,
    atomic_id: u32,
    device: u8 = 0,
    phase: u8 = 0,
    parts: u16 = 0,
    /// GPU streaming slice: output rows the device processes per dispatch when
    /// the weight is larger than the backend's staging limit. `0` = the whole
    /// weight in one dispatch (the common case). Recorded by the tuner; the
    /// runtime only applies it.
    tile: u16 = 0,
    /// Heterogeneous split: the output-row range this dispatch computes. `0`
    /// means the whole tensor (today's behaviour). A split op is two dispatches
    /// sharing `tensor`/`in`/`out`: the GPU slice `[0, row_count)` and the CPU
    /// slice `[row_start, row_start + row_count)`. Recorded by the tuner; the
    /// runtime only applies it.
    row_start: u32 = 0,
    row_count: u32 = 0,
    /// Chosen kernel variant for this dispatch, interpreted per device:
    /// `device == 0` -> `@intFromEnum(MatVecPlan)`; `device == 1` ->
    /// `@intFromEnum(gpu.GpuVariant)` (a catalogued GPU GEMV AtomicOp).
    /// The single source of the variant; no plan-level bitmask. Recorded by the
    /// Selector, applied by the runtime. `u16` because the kernel x variant
    /// catalog exceeds one byte.
    variant: u16 = 0,
    /// Chosen SIMD width (f32 lanes: 4/8/16) for the compute kernels; `0` means
    /// "unset" (the runtime falls back to the build's native width). Recorded at
    /// tune from the width the calibration measured fastest.
    simd_width: u8 = 0,
    /// Weight tensor this dispatch reads (index into `AOTKernel.tensors`).
    tensor: u32 = 0xFFFF_FFFF,
    /// Second weight tensor a fused two-weight dispatch reads (e.g. the `up`
    /// weight of `gate_up_fused`); index into `AOTKernel.tensors`, or NONE.
    tensor2: u32 = 0xFFFF_FFFF,
    /// Third weight tensor a fused three-weight dispatch reads (the `down`
    /// weight of `mlp_fused`); index into `AOTKernel.tensors`, or NONE.
    tensor3: u32 = 0xFFFF_FFFF,
    /// Transformer layer this dispatch belongs to (global ops: NONE).
    layer: u32 = 0xFFFF_FFFF,
    /// Semantic op from the model graph (for the engine's Args).
    sem_op: u8 = 0,
    /// Op params, resolved at tune (bit-cast per `sem_op`; see the tuner's
    /// `packParams`). The runtime reads them and decides nothing.
    params: [8]u32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    /// Activation input/output buffers.
    in: Binding = .{ .kind = .none },
    out: Binding = .{ .kind = .none },
    /// Dependency: dispatch that must run before this one (index), or NONE.
    dep: u32 = 0xFFFF_FFFF,
    /// KV-cache buffer index for write_kv/attention, or NONE.
    kv: u32 = 0xFFFF_FFFF,
    /// Positions the KV cache must hold (attention `max_seq`); 0 when unused.
    max_seq: u32 = 0,
    /// `zcom` rope config id for a rope dispatch, or NONE.
    rope_cfg: u32 = 0xFFFF_FFFF,
    /// Activation input/output slice lengths (element counts in f32). `0` means
    /// "whole buffer"; per-layer ops carry the layer's own dims, globals stay 0.
    in_len: u32 = 0,
    out_len: u32 = 0,
};

/// A session activation buffer: name + byte size, allocated by the runtime.
pub const Buffer = struct {
    slot: u8,
    bytes: u64,
};

pub const Tensor = struct {
    name: []const u8,
    kind: ztef.Kind,
    layer: u32,
    length: u64,
};

/// One weight unit the residency engine tracks: a whole ZTEF tensor or a
/// sub-range of one. `offset` is into the file's ztef payload.
pub const Range = struct {
    kind: ztef.Kind,
    offset: u64,
    length: u64,
    /// Ring block ordinal in consumption order (`runtime/15`), or `NONE` when
    /// the range is not part of the ring (the mmap streams it). The runtime's
    /// three-pillar ring holds block `block % 3`.
    block: u32 = NONE,
};

/// The compiled residency program stored in the plan: the weight ranges (each
/// with its consumption-order block) and the ring shape the runtime must honor.
pub const ResidencyGraph = struct {
    ranges: []Range,
    window_bytes: u64,
    /// Plan-decided byte budget for the runtime's embedding row cache. The
    /// embedding tables are not pinned whole; the runtime caches individual
    /// dequantized rows within this budget (row-LRU). `0` disables the cache.
    row_cache_bytes: u64 = 0,
    /// Three-pillar ring (`runtime/15`): the pillar size (bytes) and the
    /// `pread` chunks per block (= the `--prefetch-threads` worker count). `0`
    /// disables the ring: with `--prefetch-threads` the workers only issue
    /// `madvise(WILLNEED)` readahead (no buffer), else every range streams from
    /// the mmap. The ring holds exactly three blocks (`block % 3`).
    ring_block_bytes: u64 = 0,
    ring_k: u16 = 0,
};

/// A weight the tuner keeps resident in VRAM. `rows` is the number of output
/// rows uploaded (`0` = the whole tensor). A **split** op uploads only the GPU
/// slice `[0, rows)` and leaves rows `[rows, m)` to the CPU (the precise
/// duplication: only the split ops' ranges, never the whole tensor).
pub const VramResident = struct {
    tensor: u32,
    rows: u32 = 0,
};

/// The compiled plan. `dispatch` is the Executor's program; `graph` says how to
/// read the ZTEF; `config` carries the resources the user set. `prefill` is the
/// batched forward (`prefill_mode == batch`): projections as `matmul`, the
/// other ops recorded per-row; the runtime runs it in `prefill_block` rows.
pub const AOTKernel = struct {
    config: Config,
    fingerprint: u64,
    image: []const u8,
    tensors: []Tensor,
    buffers: []Buffer,
    dispatch: []Dispatch,
    prefill: []Dispatch = &.{},
    graph: ResidencyGraph,
    /// Worker -> CPU mapping measured at tune; empty off Linux or when topology
    /// could not be detected. The runtime only applies it.
    affinity: []u16,
    /// Weights the tuner keeps resident in VRAM (uploaded once instead of
    /// re-streamed every token). Picked by the ROI knapsack within
    /// `config.vram_bytes -| VRAM_DRIVER_HEADROOM`; empty on a CPU plan. A split
    /// op records only its GPU rows.
    vram_resident: []VramResident = &.{},
    /// Total bytes of the `vram_resident` set, for diagnostics.
    vram_resident_bytes: u64 = 0,
};

fn serialize(allocator: std.mem.Allocator, plan: AOTKernel) !std.ArrayList(u8) {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, MAGIC);
    try appendU32(allocator, &buf, VERSION);
    try appendU16(allocator, &buf, plan.config.threads);
    try buf.append(allocator, if (plan.config.gpu) 1 else 0);
    try buf.append(allocator, 0);
    try appendU64(allocator, &buf, plan.config.ram_bytes);
    try appendU64(allocator, &buf, plan.config.vram_bytes);
    try appendU64(allocator, &buf, plan.config.cpu_isa);
    try appendU32(allocator, &buf, plan.config.gpu_id);
    try buf.append(allocator, plan.config.prefill_mode);
    try appendU32(allocator, &buf, plan.config.prefill_block);
    try buf.append(allocator, plan.config.prefetch_threads);
    try appendU16(allocator, &buf, plan.config.gpu_warmup);
    try appendU64(allocator, &buf, plan.config.residency_bytes);
    try appendU64(allocator, &buf, plan.fingerprint);
    try appendStr(allocator, &buf, plan.image);

    try appendU32(allocator, &buf, @intCast(plan.tensors.len));
    for (plan.tensors) |t| {
        try buf.append(allocator, @intCast(@intFromEnum(t.kind)));
        try appendU32(allocator, &buf, t.layer);
        try appendU64(allocator, &buf, t.length);
        try appendStr(allocator, &buf, t.name);
    }

    try appendU32(allocator, &buf, @intCast(plan.buffers.len));
    for (plan.buffers) |b| {
        try buf.append(allocator, b.slot);
        try appendU64(allocator, &buf, b.bytes);
    }

    try appendU32(allocator, &buf, @intCast(plan.dispatch.len));
    for (plan.dispatch) |d| try appendDispatch(allocator, &buf, d);

    try appendU32(allocator, &buf, @intCast(plan.prefill.len));
    for (plan.prefill) |d| try appendDispatch(allocator, &buf, d);

    try appendU32(allocator, &buf, @intCast(plan.graph.ranges.len));
    for (plan.graph.ranges) |r| {
        try buf.append(allocator, @intCast(@intFromEnum(r.kind)));
        try appendU64(allocator, &buf, r.offset);
        try appendU64(allocator, &buf, r.length);
        try appendU32(allocator, &buf, r.block);
    }
    try appendU64(allocator, &buf, plan.graph.window_bytes);
    try appendU64(allocator, &buf, plan.graph.row_cache_bytes);
    try appendU64(allocator, &buf, plan.graph.ring_block_bytes);
    try appendU16(allocator, &buf, plan.graph.ring_k);

    try appendU32(allocator, &buf, @intCast(plan.affinity.len));
    for (plan.affinity) |c| try appendU16(allocator, &buf, c);

    try appendU32(allocator, &buf, @intCast(plan.vram_resident.len));
    for (plan.vram_resident) |vr| {
        try appendU32(allocator, &buf, vr.tensor);
        try appendU32(allocator, &buf, vr.rows);
    }
    try appendU64(allocator, &buf, plan.vram_resident_bytes);

    return buf;
}

fn appendDispatch(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), d: Dispatch) !void {
    try buf.append(allocator, @intCast(@intFromEnum(d.op)));
    try appendU32(allocator, buf, d.atomic_id);
    try buf.append(allocator, d.device);
    try buf.append(allocator, d.phase);
    try appendU16(allocator, buf, d.parts);
    try appendU16(allocator, buf, d.tile);
    try appendU16(allocator, buf, d.variant);
    try buf.append(allocator, d.simd_width);
    try appendU32(allocator, buf, d.row_start);
    try appendU32(allocator, buf, d.row_count);
    try appendU32(allocator, buf, d.tensor);
    try appendU32(allocator, buf, d.tensor2);
    try appendU32(allocator, buf, d.tensor3);
    try appendU32(allocator, buf, d.layer);
    try buf.append(allocator, d.sem_op);
    for (d.params) |p| try appendU32(allocator, buf, p);
    try buf.append(allocator, @intFromEnum(d.in.kind));
    try appendU32(allocator, buf, d.in.ref);
    try appendU32(allocator, buf, d.in.ref2);
    try buf.append(allocator, @intFromEnum(d.out.kind));
    try appendU32(allocator, buf, d.out.ref);
    try appendU32(allocator, buf, d.out.ref2);
    try appendU32(allocator, buf, d.dep);
    try appendU32(allocator, buf, d.kv);
    try appendU32(allocator, buf, d.max_seq);
    try appendU32(allocator, buf, d.rope_cfg);
    try appendU32(allocator, buf, d.in_len);
    try appendU32(allocator, buf, d.out_len);
}

/// Write `plan` to the fixed `plan.zplan` inside `dir`.
pub fn write(allocator: std.mem.Allocator, dir: Io.Dir, io: Io, plan: AOTKernel) !void {
    var buf = try serialize(allocator, plan);
    defer buf.deinit(allocator);
    var f = try dir.createFile(io, NAME, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
}

/// Write `plan` to an arbitrary path (the shared `<image>_<fp>.zplan`).
pub fn writeFile(allocator: std.mem.Allocator, io: Io, path: []const u8, plan: AOTKernel) !void {
    var buf = try serialize(allocator, plan);
    defer buf.deinit(allocator);
    var f = try Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
}

pub const Reader = struct {
    allocator: std.mem.Allocator,
    mmap: MappedFile,
    plan: AOTKernel,

    pub fn open(allocator: std.mem.Allocator, dir: Io.Dir, io: Io) !Reader {
        var mmap = try MappedFile.openIn(dir, io, NAME);
        errdefer mmap.close();
        return build(allocator, mmap);
    }

    /// Open a plan at an arbitrary path (the registry's `<image>_<fp>.zplan`).
    pub fn openFile(allocator: std.mem.Allocator, io: Io, path: []const u8) !Reader {
        var mmap = try MappedFile.open(io, path);
        errdefer mmap.close();
        return build(allocator, mmap);
    }

    fn build(allocator: std.mem.Allocator, mmap: MappedFile) !Reader {
        var br = BinaryReader.init(mmap.data);
        const magic = try br.readSlice(4);
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
        if (try br.readU32() != VERSION) return error.BadVersion;

        const threads = try br.readU16();
        const gpu = (try br.readU8()) != 0;
        _ = try br.readU8();
        const ram_bytes = try br.readU64();
        const vram_bytes = try br.readU64();
        const cpu_isa = try br.readU64();
        const gpu_id = try br.readU32();
        const prefill_mode = try br.readU8();
        const prefill_block = try br.readU32();
        const prefetch_threads = try br.readU8();
        const gpu_warmup = try br.readU16();
        const residency_bytes = try br.readU64();
        const config = Config{
            .threads = threads,
            .ram_bytes = ram_bytes,
            .vram_bytes = vram_bytes,
            .gpu = gpu,
            .cpu_isa = cpu_isa,
            .gpu_id = gpu_id,
            .prefill_mode = prefill_mode,
            .prefill_block = prefill_block,
            .prefetch_threads = prefetch_threads,
            .gpu_warmup = gpu_warmup,
            .residency_bytes = residency_bytes,
        };
        const fingerprint = try br.readU64();
        const image = try readStr(&br);

        const tensor_count = try br.readU32();
        const tensors = try allocator.alloc(Tensor, tensor_count);
        errdefer allocator.free(tensors);
        for (tensors) |*t| {
            const kind: ztef.Kind = @enumFromInt(try br.readU8());
            const layer = try br.readU32();
            const length = try br.readU64();
            const name = try readStr(&br);
            t.* = .{ .name = name, .kind = kind, .layer = layer, .length = length };
        }

        const buffer_count = try br.readU32();
        const buffers = try allocator.alloc(Buffer, buffer_count);
        errdefer allocator.free(buffers);
        for (buffers) |*b| {
            const slot = try br.readU8();
            b.* = .{ .slot = slot, .bytes = try br.readU64() };
        }

        const dispatch_count = try br.readU32();
        const dispatch = try allocator.alloc(Dispatch, dispatch_count);
        errdefer allocator.free(dispatch);
        for (dispatch) |*d| d.* = try readDispatch(&br);

        const prefill_count = try br.readU32();
        const prefill = try allocator.alloc(Dispatch, prefill_count);
        errdefer allocator.free(prefill);
        for (prefill) |*d| d.* = try readDispatch(&br);

        const range_count = try br.readU32();
        const ranges = try allocator.alloc(Range, range_count);
        errdefer allocator.free(ranges);
        for (ranges) |*r| {
            const kind: ztef.Kind = @enumFromInt(try br.readU8());
            const offset = try br.readU64();
            const length = try br.readU64();
            const block = try br.readU32();
            r.* = .{ .kind = kind, .offset = offset, .length = length, .block = block };
        }
        const window_bytes = try br.readU64();
        const row_cache_bytes = try br.readU64();
        const ring_block_bytes = try br.readU64();
        const ring_k = try br.readU16();

        const affinity_count = try br.readU32();
        const affinity = try allocator.alloc(u16, affinity_count);
        errdefer allocator.free(affinity);
        for (affinity) |*c| c.* = try br.readU16();

        const resident_count = try br.readU32();
        const vram_resident = try allocator.alloc(VramResident, resident_count);
        errdefer allocator.free(vram_resident);
        for (vram_resident) |*vr| {
            vr.tensor = try br.readU32();
            vr.rows = try br.readU32();
        }
        const vram_resident_bytes = try br.readU64();

        return .{
            .allocator = allocator,
            .mmap = mmap,
            .plan = .{
                .config = config,
                .fingerprint = fingerprint,
                .image = image,
                .tensors = tensors,
                .buffers = buffers,
                .dispatch = dispatch,
                .prefill = prefill,
                .graph = .{
                    .ranges = ranges,
                    .window_bytes = window_bytes,
                    .row_cache_bytes = row_cache_bytes,
                    .ring_block_bytes = ring_block_bytes,
                    .ring_k = ring_k,
                },
                .affinity = affinity,
                .vram_resident = vram_resident,
                .vram_resident_bytes = vram_resident_bytes,
            },
        };
    }

    pub fn deinit(self: *Reader) void {
        self.allocator.free(self.plan.tensors);
        self.allocator.free(self.plan.buffers);
        self.allocator.free(self.plan.dispatch);
        self.allocator.free(self.plan.prefill);
        self.allocator.free(self.plan.graph.ranges);
        self.allocator.free(self.plan.affinity);
        if (self.plan.vram_resident.len > 0) self.allocator.free(self.plan.vram_resident);
        self.mmap.close();
    }
};

fn readStr(br: *BinaryReader) ![]const u8 {
    return br.readSlice(try br.readU16());
}

fn readDispatch(br: *BinaryReader) !Dispatch {
    const dop: op = @enumFromInt(try br.readU8());
    const atomic_id = try br.readU32();
    const device = try br.readU8();
    const phase = try br.readU8();
    const parts = try br.readU16();
    const tile = try br.readU16();
    const variant = try br.readU16();
    const simd_width = try br.readU8();
    const row_start = try br.readU32();
    const row_count = try br.readU32();
    const tensor = try br.readU32();
    const tensor2 = try br.readU32();
    const tensor3 = try br.readU32();
    const tensor_layer = try br.readU32();
    const sem_op = try br.readU8();
    var params: [8]u32 = undefined;
    for (&params) |*p| p.* = try br.readU32();
    const in_kind: Binding.Kind = @enumFromInt(try br.readU8());
    const in_ref = try br.readU32();
    const in_ref2 = try br.readU32();
    const out_kind: Binding.Kind = @enumFromInt(try br.readU8());
    const out_ref = try br.readU32();
    const out_ref2 = try br.readU32();
    const dep = try br.readU32();
    const kv_ref = try br.readU32();
    const max_seq = try br.readU32();
    const rope_cfg = try br.readU32();
    const in_len = try br.readU32();
    const out_len = try br.readU32();
    return .{
        .op = dop,
        .atomic_id = atomic_id,
        .device = device,
        .phase = phase,
        .parts = parts,
        .tile = tile,
        .variant = variant,
        .simd_width = simd_width,
        .row_start = row_start,
        .row_count = row_count,
        .tensor = tensor,
        .tensor2 = tensor2,
        .tensor3 = tensor3,
        .layer = tensor_layer,
        .sem_op = sem_op,
        .params = params,
        .in = .{ .kind = in_kind, .ref = in_ref, .ref2 = in_ref2 },
        .out = .{ .kind = out_kind, .ref = out_ref, .ref2 = out_ref2 },
        .dep = dep,
        .kv = kv_ref,
        .max_seq = max_seq,
        .rope_cfg = rope_cfg,
        .in_len = in_len,
        .out_len = out_len,
    };
}

fn appendStr(allocator: std.mem.Allocator, l: *std.ArrayList(u8), s: []const u8) !void {
    try appendU16(allocator, l, @intCast(@min(s.len, 0xFFFF)));
    try l.appendSlice(allocator, s[0..@min(s.len, 0xFFFF)]);
}

test "aotkernel round-trip" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var tensors = [_]Tensor{
        .{ .name = "blk.0.attn_q.weight", .kind = .q_proj, .layer = 0, .length = 36 },
        .{ .name = "token_embd.weight", .kind = .embd, .layer = 0xFFFF_FFFF, .length = 18 },
    };
    var dispatch = [_]Dispatch{
        .{ .op = .matvec, .atomic_id = 0, .tensor = 0, .tensor2 = 1, .tensor3 = 1, .max_seq = 99, .rope_cfg = 7, .in_len = 128, .out_len = 64, .variant = 3, .simd_width = 8, .tile = 5 },
        .{ .op = .dequant, .atomic_id = 0, .tensor = 1 },
    };
    var prefill = [_]Dispatch{
        .{ .op = .matmul, .atomic_id = 1, .phase = 1, .tensor = 0, .in_len = 128, .out_len = 64 },
    };
    var buffers = [_]Buffer{
        .{ .slot = 0, .bytes = 4096 },
    };
    var ranges = [_]Range{
        .{ .kind = .q_proj, .offset = 0, .length = 36, .block = 0 },
        .{ .kind = .embd, .offset = 36, .length = 18, .block = 5 },
    };
    const cfg = Config{ .threads = 8, .ram_bytes = 4 << 30, .vram_bytes = 0, .gpu = false, .cpu_isa = 42, .gpu_id = 0, .prefill_mode = 1, .prefill_block = 8, .prefetch_threads = 2 };
    var affinity = [_]u16{ 0, 2, 4, 6 };
    var vram_resident = [_]VramResident{ .{ .tensor = 1 }, .{ .tensor = 0, .rows = 7 } };

    try write(testing.allocator, tmp.dir, testing.io, .{
        .config = cfg,
        .fingerprint = cfg.fingerprint(),
        .image = "e2b",
        .tensors = &tensors,
        .buffers = &buffers,
        .dispatch = &dispatch,
        .prefill = &prefill,
        .graph = .{ .ranges = &ranges, .window_bytes = 1 << 20, .row_cache_bytes = 4096, .ring_block_bytes = 1 << 19, .ring_k = 3 },
        .affinity = &affinity,
        .vram_resident = &vram_resident,
        .vram_resident_bytes = 54,
    });

    var r = try Reader.open(testing.allocator, tmp.dir, testing.io);
    defer r.deinit();
    try testing.expectEqual(@as(u16, 8), r.plan.config.threads);
    try testing.expectEqual(cfg.fingerprint(), r.plan.fingerprint);
    try testing.expectEqualStrings("e2b", r.plan.image);
    try testing.expectEqual(@as(u8, 1), r.plan.config.prefill_mode);
    try testing.expectEqual(@as(u32, 8), r.plan.config.prefill_block);
    try testing.expectEqual(@as(u8, 2), r.plan.config.prefetch_threads);
    try testing.expectEqual(@as(usize, 1), r.plan.prefill.len);
    try testing.expectEqual(op.matmul, r.plan.prefill[0].op);
    try testing.expectEqual(@as(u8, 1), r.plan.prefill[0].phase);
    try testing.expectEqual(@as(u32, 1), r.plan.prefill[0].atomic_id);
    try testing.expectEqual(@as(usize, 2), r.plan.dispatch.len);
    try testing.expectEqual(op.matvec, r.plan.dispatch[0].op);
    try testing.expectEqual(@as(u32, 0), r.plan.dispatch[0].tensor);
    try testing.expectEqual(@as(u32, 1), r.plan.dispatch[0].tensor2);
    try testing.expectEqual(@as(u32, 1), r.plan.dispatch[0].tensor3);
    try testing.expectEqual(NONE, r.plan.dispatch[1].tensor2);
    try testing.expectEqual(NONE, r.plan.dispatch[1].tensor3);
    try testing.expectEqual(@as(u32, 99), r.plan.dispatch[0].max_seq);
    try testing.expectEqual(@as(u32, 7), r.plan.dispatch[0].rope_cfg);
    try testing.expectEqual(@as(u32, 128), r.plan.dispatch[0].in_len);
    try testing.expectEqual(@as(u32, 64), r.plan.dispatch[0].out_len);
    try testing.expectEqual(@as(u16, 3), r.plan.dispatch[0].variant);
    try testing.expectEqual(@as(u16, 0), r.plan.dispatch[1].variant);
    try testing.expectEqual(@as(u8, 8), r.plan.dispatch[0].simd_width);
    try testing.expectEqual(@as(u8, 0), r.plan.dispatch[1].simd_width);
    try testing.expectEqual(@as(u16, 5), r.plan.dispatch[0].tile);
    try testing.expectEqual(@as(u16, 0), r.plan.dispatch[1].tile);
    try testing.expectEqual(op.dequant, r.plan.dispatch[1].op);
    try testing.expectEqual(@as(usize, 2), r.plan.graph.ranges.len);
    try testing.expectEqual(ztef.Kind.q_proj, r.plan.graph.ranges[0].kind);
    try testing.expectEqual(@as(u64, 36), r.plan.graph.ranges[1].offset);
    try testing.expectEqual(@as(u32, 0), r.plan.graph.ranges[0].block);
    try testing.expectEqual(@as(u32, 5), r.plan.graph.ranges[1].block);
    try testing.expectEqual(@as(u64, 1 << 20), r.plan.graph.window_bytes);
    try testing.expectEqual(@as(u64, 4096), r.plan.graph.row_cache_bytes);
    try testing.expectEqual(@as(u64, 1 << 19), r.plan.graph.ring_block_bytes);
    try testing.expectEqual(@as(u16, 3), r.plan.graph.ring_k);
    try testing.expectEqualSlices(u16, &affinity, r.plan.affinity);
    try testing.expectEqualSlices(VramResident, &vram_resident, r.plan.vram_resident);
    try testing.expectEqual(@as(u64, 54), r.plan.vram_resident_bytes);
}

test "plan reader rejects a stale format version" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var tensors = [_]Tensor{.{ .name = "token_embd.weight", .kind = .embd, .layer = 0, .length = 1 }};
    var dispatch = [_]Dispatch{.{ .op = .matvec, .atomic_id = 0, .tensor = 0 }};
    var ranges = [_]Range{.{ .kind = .embd, .offset = 0, .length = 1 }};
    const cfg = Config{ .threads = 1, .ram_bytes = 1 << 20, .vram_bytes = 0, .gpu = false, .cpu_isa = 0, .gpu_id = 0, .prefill_mode = 0, .prefill_block = 0, .prefetch_threads = 0 };

    var buf = try serialize(testing.allocator, .{
        .config = cfg,
        .fingerprint = cfg.fingerprint(),
        .image = "x",
        .tensors = &tensors,
        .buffers = &[_]Buffer{},
        .dispatch = &dispatch,
        .prefill = &[_]Dispatch{},
        .graph = .{ .ranges = &ranges, .window_bytes = 0 },
        .affinity = &[_]u16{},
    });
    defer buf.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, MAGIC, buf.items[0..4]);
    std.mem.writeInt(u32, buf.items[4..8], VERSION - 1, .little);

    var f = try tmp.dir.createFile(testing.io, NAME, .{});
    defer f.close(testing.io);
    try f.writeStreamingAll(testing.io, buf.items);

    try testing.expectError(error.BadVersion, Reader.open(testing.allocator, tmp.dir, testing.io));
}

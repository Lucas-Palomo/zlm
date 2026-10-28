const std = @import("std");
const Io = std.Io;
const io_utils = @import("../../utils/io.zig");
const cpu = @import("../../utils/cpu.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const store = @import("../../image/store.zig");
const zman = @import("../../image/zman.zig");
const ztef = @import("../../image/ztef.zig");
const arch_root = @import("../../architectures/root.zig");
const zmeta = @import("../../image/zmeta_reader.zig");
const graph = @import("../../architectures/core/graph.zig");
const contract = @import("../../architectures/core/contract.zig");
const compile = @import("../../tuner/compile.zig");
const composer = @import("../../tuner/composer.zig");
const calibrate = @import("../../tuner/calibrate.zig");
const caps_mod = @import("../../tuner/caps.zig");
const cost = @import("../../tuner/cost.zig");
const tuner = @import("../../tuner/tuner.zig");
const gpu_policy = @import("../../tuner/gpu.zig");
const device = @import("../../gpu/root.zig");
const attention = @import("../../kernels/transformer/attention.zig");
const matvec = @import("../../kernels/linear/matvec/root.zig");
const zrun = @import("../../image/zrun.zig");
const plan = @import("../../plan/root.zig");
const session_mod = @import("../../runtime/session.zig");
const Context = command.Context;

/// Human label for a recorded co-processor GEMV shader variant.
fn gpuVariantLabel(v: device.GpuVariant) []const u8 {
    return switch (v) {
        .plain => "plain",
        .u4 => "u4",
        .u8 => "u8",
        .sub => "sub",
        .un2 => "un2",
        .un4 => "un4",
        .w32 => "w32",
        .w128 => "w128",
    };
}

/// The recorded GEMV shader variant for a weight's ggml type: the probe's
/// per-quant winner (`plain` for a quant the device does not offload).
fn gpuQuantVariant(gpu: calibrate.GpuCoeffs, t: anytype) device.GpuVariant {
    return switch (t) {
        .q4_0 => gpu.q4_0_variant,
        .q4_1 => gpu.q4_1_variant,
        .q8_0 => gpu.q8_0_variant,
        .q6_k => gpu.q6_k_variant,
        .q4_k => gpu.q4_k_variant,
        .f32 => gpu.f32_variant,
        .f16 => gpu.f16_variant,
        .bf16 => gpu.bf16_variant,
        else => .plain,
    };
}

/// The recorded variant of the fused `gate_up` shader for a weight's quant
/// (q4_0/q4_1/q8_0 each have their own probe-chosen winner).
fn gpuGateUpVariant(gpu: calibrate.GpuCoeffs, t: anytype) device.GpuVariant {
    return switch (t) {
        .q4_0 => gpu.gate_up_variant,
        .q4_1 => gpu.gate_up_q4_1_variant,
        .q8_0 => gpu.gate_up_q8_0_variant,
        else => .plain,
    };
}

pub const help_text =
    \\Tunes an image for this machine and writes a plan.
    \\
    \\Usage: zlm tune <image> <runtime-alias> --threads N --ram <mb|gb>
    \\                                        [--residency <mb|gb>]
    \\                                        [--prefetch-threads M]
    \\                                        [--gpu --vram <mb|gb>] [--force]
    \\                                        [--prefix-cache]
    \\
    \\The plan carries the configuration; the runtime decides nothing.
    \\--threads and --ram are required; --gpu requires --vram.
    \\--threads N is the compute pool. --prefetch-threads M (>= 1) adds M
    \\dedicated residency-prefetch workers, separate from and additional to
    \\the compute threads (the process runs N + M threads); absent = inline.
    \\--residency <mb|gb> is the explicit ceiling for the resident weight ring
    \\(`compute | reserved | prefetch`, filled by --prefetch-threads). It must
    \\be smaller than --ram, and it requires --prefetch-threads. Absent = no
    \\ring. --prefetch-threads alone is the readahead-only mode: the workers
    \\issue madvise(WILLNEED) for the weights ahead, with no ring buffer.
    \\The device calibration profile (.zlm/runtimes/<digest>.zat) is
    \\measured and saved on first tune, then reused; --force re-measures it.
    \\--prefix-cache warms the prefix cache with the chat template and writes
    \\.zlm/runtimes/<image>_<prefix-cache-fp>.zpc (default: no cache file).
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--threads", .short = 't', .kind = .uint, .required = true },
    .{ .long = "--prefetch-threads", .kind = .uint },
    .{ .long = "--ram", .kind = .size, .required = true },
    .{ .long = "--residency", .kind = .size },
    .{ .long = "--vram", .kind = .size },
    .{ .long = "--gpu", .kind = .flag },
    .{ .long = "--force", .kind = .flag },
    .{ .long = "--prefix-cache", .kind = .flag },
};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "tune: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const image_name = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <image>", .{});
        return error.MissingArgument;
    };
    const runtime_alias = opt.positional(1) orelse {
        io_utils.eprintln(ctx.io, "Missing <runtime-alias>", .{});
        return error.MissingArgument;
    };
    if (opt.positional(2) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }
    const n_threads = opt.uint("--threads", u16).?;
    const ram_bytes = opt.size("--ram").?;
    if (n_threads == 0) {
        io_utils.eprintln(ctx.io, "--threads must be greater than 0", .{});
        return error.InvalidValue;
    }
    // `--prefetch-threads` is opt-in: absent = 0 (inline). When present it must
    // be a positive count; the parser already rejected negatives/non-integers.
    const prefetch_threads: u8 = blk: {
        const raw = opt.get("--prefetch-threads") orelse break :blk 0;
        const m = std.fmt.parseInt(u8, raw, 10) catch {
            io_utils.eprintln(ctx.io, "--prefetch-threads must be between 1 and 255", .{});
            return error.InvalidValue;
        };
        if (m == 0) {
            io_utils.eprintln(ctx.io, "--prefetch-threads must be greater than 0", .{});
            return error.InvalidValue;
        }
        break :blk m;
    };
    if (ram_bytes == 0) {
        io_utils.eprintln(ctx.io, "--ram must be greater than 0", .{});
        return error.InvalidValue;
    }
    // `--residency` is the explicit ceiling for the resident weight ring and is
    // opt-in: absent = no ring. It must leave room for the KV/scratch, so it has
    // to be strictly smaller than `--ram`, and it needs the dedicated prefetch
    // workers to fill the pillars, so it requires `--prefetch-threads`.
    // `--prefetch-threads` alone is the readahead-only mode (madvise, no ring).
    const residency_bytes: u64 = opt.size("--residency") orelse 0;
    if (residency_bytes != 0) {
        if (residency_bytes >= ram_bytes) {
            io_utils.eprintln(ctx.io, "--residency must be smaller than --ram", .{});
            return error.InvalidValue;
        }
        if (prefetch_threads == 0) {
            io_utils.eprintln(ctx.io, "--residency requires --prefetch-threads", .{});
            return error.MissingArgument;
        }
    }
    const gpu = opt.has("--gpu");
    const vram = opt.size("--vram");
    if (gpu and vram == null) {
        io_utils.eprintln(ctx.io, "--gpu requires --vram <mb|gb>", .{});
        return error.MissingArgument;
    }
    if (!store.exists(ctx.io, image_name)) {
        io_utils.eprintln(ctx.io, "No such image: {s}", .{image_name});
        return error.NotFound;
    }
    if (!store.isValidName(runtime_alias)) {
        io_utils.eprintln(ctx.io, "Invalid runtime alias: {s}", .{runtime_alias});
        return error.InvalidName;
    }

    // A `--gpu` tune needs a usable device: the co-processor's offload decision
    // is part of the plan, so it must be resolved here (the runtime decides
    // nothing). The device class (UMA vs discrete) sets the offload threshold.
    var gpu_dev: ?device.Device = null;
    defer if (gpu_dev) |*d| d.deinit();
    var gpu_is_uma = false;
    if (gpu) {
        gpu_dev = device.Device.init(ctx.allocator, ctx.io);
        if (!gpu_dev.?.available()) {
            io_utils.eprintln(ctx.io, "--gpu: no usable Vulkan device", .{});
            return error.NoGpuDevice;
        }
        gpu_is_uma = gpu_dev.?.isUma();
    }

    var cfg = plan.Config{
        .threads = n_threads,
        .prefetch_threads = prefetch_threads,
        .ram_bytes = ram_bytes,
        .residency_bytes = residency_bytes,
        .vram_bytes = vram orelse 0,
        .gpu = gpu,
        .cpu_isa = cpu.digest(),
        .gpu_id = 0,
    };
    var fingerprint = cfg.fingerprint();
    var fp_hex = std.fmt.bytesToHex(std.mem.asBytes(&fingerprint), .lower);

    var plan_buf: [512]u8 = undefined;
    var plan_path: []const u8 = try store.runtimePlanPath(&plan_buf, image_name, &fp_hex);

    const device_digest = cpu.digest();
    const digest_hex = std.fmt.bytesToHex(std.mem.asBytes(&device_digest), .lower);
    var dzat_buf: [512]u8 = undefined;
    const zat_path = try store.deviceZatPath(&dzat_buf, &digest_hex);

    var plan_name_buf: [256]u8 = undefined;
    var plan_name: []const u8 = try std.fmt.bufPrint(&plan_name_buf, "{s}_{s}.zplan", .{ image_name, &fp_hex });
    var zat_name_buf: [256]u8 = undefined;
    const zat_name = try std.fmt.bufPrint(&zat_name_buf, "{s}.zat", .{&digest_hex});

    try store.ensureRuntimesDir(ctx.io);
    var rdir = try store.openRuntimes(ctx.io);
    defer rdir.close(ctx.io);

    const force = opt.has("--force");
    var row = zrun.Entry{
        .id = runtime_alias,
        .plan = plan_name,
        .tune = zat_name,
        .image = image_name,
        .fp = fingerprint,
    };

    var dir = try store.openImage(ctx.io, image_name);
    defer dir.close(ctx.io);

    var manifest = zman.Manifest.open(ctx.allocator, dir, ctx.io) catch |err| {
        io_utils.eprintln(ctx.io, "Cannot read image manifest: {s}", .{@errorName(err)});
        return err;
    };
    defer manifest.deinit();
    const arch_name = manifest.arch;

    var dir_buf: [512]u8 = undefined;
    const image_path = try store.imageDir(&dir_buf, image_name);

    var io_path_buf: [512]u8 = undefined;
    const io_path = largestZtefPath(&io_path_buf, image_path, manifest.files);

    var caps = caps_mod.detect(ctx.allocator);
    var coeffs: calibrate.Coeffs = .{};
    if (!force and store.pathExists(ctx.io, zat_path)) {
        if (calibrate.load(ctx.allocator, rdir, ctx.io, &digest_hex)) |profile| {
            caps = profile.caps;
            caps.applyCoeffs(profile.coeffs);
            coeffs = profile.coeffs;
            io_utils.println(ctx.io, "calibrate: reused {s}", .{zat_path});
        } else {
            coeffs = applyCalibration(ctx, rdir, &caps, &digest_hex, io_path, zat_path);
        }
    } else {
        coeffs = applyCalibration(ctx, rdir, &caps, &digest_hex, io_path, zat_path);
    }

    const arch = arch_root.find(arch_name) orelse {
        io_utils.eprintln(ctx.io, "Unsupported architecture: {s}", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    var md = zmeta.Zmeta.open(ctx.allocator, dir, ctx.io) catch |err| {
        io_utils.eprintln(ctx.io, "Cannot read image metadata: {s}", .{@errorName(err)});
        return err;
    };
    defer md.deinit();
    var model_cfg = try arch.loadConfig(&md, ctx.allocator);
    defer model_cfg.deinit();
    const layer_count = layerCountOf(&model_cfg);

    var tensors: std.ArrayList(plan.Tensor) = .empty;
    defer tensors.deinit(ctx.allocator);
    var ranges: std.ArrayList(plan.Range) = .empty;
    defer ranges.deinit(ctx.allocator);
    var tensor_range: std.ArrayList(u32) = .empty;
    defer tensor_range.deinit(ctx.allocator);
    // Shared-weight dedup: `tensor_alias[i]` is the canonical tensor index for
    // tensor `i` (itself when unique). `content_seen` is the first occurrence of
    // each byte-identical tensor (sha256 + type + dims + length, as persisted in
    // the ZTEF); identical tensors at different offsets collapse to one
    // residency range and one read.
    var tensor_alias: std.ArrayList(u32) = .empty;
    defer tensor_alias.deinit(ctx.allocator);
    var content_seen: std.ArrayList(ContentKey) = .empty;
    defer content_seen.deinit(ctx.allocator);
    // Attn q/k/v/out buffer size = the q_proj weight's output features, read
    // from the actual tensor (config's key_length can differ per model). Sliding
    // layers are wider, so track both and size the slots to the maximum.
    var tdim = TensorDims{};

    for (manifest.files) |f| {
        var file_buf: [512]u8 = undefined;
        const zpath = try std.fmt.bufPrint(&file_buf, "{s}/{s}", .{ image_path, f.kind.filename() });
        var r = ztef.Reader.open(ctx.allocator, ctx.io, zpath) catch |err| {
            io_utils.eprintln(ctx.io, "Cannot read {s}: {s}", .{ f.kind.filename(), @errorName(err) });
            return err;
        };
        defer r.deinit();

        for (r.entries) |*e| {
            const tensor_idx: u32 = @intCast(tensors.items.len);
            try tensors.append(ctx.allocator, .{
                .name = try ctx.allocator.dupe(u8, e.name),
                .kind = f.kind,
                .layer = e.layer,
                .length = e.length,
            });
            try tensor_alias.append(ctx.allocator, tensor_idx);
            const key = ContentKey{
                .sha256 = e.sha256,
                .ggml_type = @intFromEnum(e.ggml_type),
                .n_dims = e.n_dims,
                .dims = e.dims,
                .length = e.length,
                .tensor = tensor_idx,
                .range = 0,
            };
            var ridx: ?u32 = null;
            for (content_seen.items) |seen| {
                if (sameContent(seen, key)) {
                    ridx = seen.range;
                    tensor_alias.items[tensor_idx] = seen.tensor;
                    break;
                }
            }
            if (ridx == null) {
                ridx = try findOrAddRange(ctx.allocator, &ranges, f.kind, e.offset, e.length);
                var recorded = key;
                recorded.range = ridx.?;
                try content_seen.append(ctx.allocator, recorded);
            }
            try tensor_range.append(ctx.allocator, ridx.?);
            if (e.n_dims == 2 and e.layer != 0xFFFF_FFFF) {
                if (std.mem.endsWith(u8, e.name, "attn_q.weight")) {
                    tdim.q = @max(tdim.q, e.dims[1]);
                } else if (std.mem.endsWith(u8, e.name, "attn_k.weight")) {
                    tdim.kv = @max(tdim.kv, e.dims[1]);
                } else if (std.mem.endsWith(u8, e.name, "attn_output.weight")) {
                    tdim.attn_out = @max(tdim.attn_out, e.dims[0]);
                } else if (std.mem.endsWith(u8, e.name, "ffn_gate.weight")) {
                    tdim.ffn = @max(tdim.ffn, e.dims[1]);
                }
            }
            // The embedding table is global; its second dim is the vocab size
            // (also the lm_head output width, including tied embeddings).
            if (e.n_dims == 2 and std.mem.eql(u8, e.name, "token_embd.weight")) {
                tdim.vocab = @max(tdim.vocab, e.dims[1]);
            }
        }
    }

    var dims = dimsOf(&model_cfg);
    if (tdim.q != 0) dims.q = tdim.q;
    if (tdim.kv != 0) dims.kv = tdim.kv;
    if (tdim.attn_out != 0) dims.attn_out = tdim.attn_out;
    if (tdim.ffn != 0) dims.ffn = tdim.ffn;
    if (tdim.vocab != 0) dims.vocab = tdim.vocab;
    const policy = choosePrefillPolicy(ram_bytes, dims);
    // GPU co-processor probe: measure the device's real GEMV/GEMM throughput
    // once and record it in the device profile, so the offload decision uses
    // measured numbers instead of the conservative constants. Reused when the
    // profile already carries them; `--force` re-measures. A CPU-only tune never
    // probes, so its profile keeps the all-zero "unmeasured" GPU coeffs.
    if (gpu) {
        if (force or coeffs.gpu.gemv_gbps <= 0 or coeffs.gpu.q4_k_u4_gbps <= 0) {
            const probe = gpu_dev.?.probeGpu(ctx.allocator, @intCast(dims.ffn), @intCast(dims.hidden), @intCast(policy.block));
            coeffs.gpu = .{
                .gemv_gbps = probe.gemv_gbps,
                .gemv_streamed_gbps = probe.gemv_streamed_gbps,
                .matmul_gbps = probe.matmul_gbps,
                .split_efficiency = calibrate.measureSplitEfficiency(ctx.allocator, gpu_dev.?, n_threads) orelse 0,
                .q4_0_variant = probe.q4_0_variant,
                .q4_1_variant = probe.q4_1_variant,
                .q8_0_variant = probe.q8_0_variant,
                .q6_k_variant = probe.q6_k_variant,
                .q4_k_variant = probe.q4_k_variant,
                .f32_variant = probe.f32_variant,
                .f16_variant = probe.f16_variant,
                .bf16_variant = probe.bf16_variant,
                .gate_up_variant = probe.gate_up_variant,
                .gate_up_q4_1_variant = probe.gate_up_q4_1_variant,
                .gate_up_q8_0_variant = probe.gate_up_q8_0_variant,
                .q4_0_u4_gbps = probe.q4_0_u4_gbps,
                .q4_0_sub_gbps = probe.q4_0_sub_gbps,
                .q4_0_un2_gbps = probe.q4_0_un2_gbps,
                .q4_0_un4_gbps = probe.q4_0_un4_gbps,
                .q4_0_w32_gbps = probe.q4_0_w32_gbps,
                .q4_0_w128_gbps = probe.q4_0_w128_gbps,
                .q4_k_gbps = probe.q4_k_gbps,
                .q4_k_u4_gbps = probe.q4_k_u4_gbps,
                .gate_up_gbps = probe.gate_up_gbps,
                .gate_up_u4_gbps = probe.gate_up_u4_gbps,
            };
            caps.applyCoeffs(coeffs);
            calibrate.save(ctx.allocator, rdir, ctx.io, &digest_hex, .{
                .caps = caps,
                .coeffs = coeffs,
            }) catch {};
            io_utils.println(ctx.io, "gpu-probe: gemv {d:.2} GB/s (stream {d:.2}), matmul {d:.2} GB/s, split {d:.2}x, cpu {d:.2} GB/s", .{
                coeffs.gpu.gemv_gbps,
                coeffs.gpu.gemv_streamed_gbps,
                coeffs.gpu.matmul_gbps,
                coeffs.gpu.split_efficiency,
                caps.dram_bw_gbps,
            });
            io_utils.println(ctx.io, "gpu-unroll: q4_0 {d:.2}->{d:.2}/sub {d:.2} GB/s ({s}), q4_k {d:.2}->{d:.2} GB/s ({s})", .{
                coeffs.gpu.gemv_gbps,
                coeffs.gpu.q4_0_u4_gbps,
                coeffs.gpu.q4_0_sub_gbps,
                gpuVariantLabel(coeffs.gpu.q4_0_variant),
                coeffs.gpu.q4_k_gbps,
                coeffs.gpu.q4_k_u4_gbps,
                gpuVariantLabel(coeffs.gpu.q4_k_variant),
            });
            io_utils.println(ctx.io, "gpu-variants: q4_0 {s}, q4_1 {s}, q8_0 {s}, q6_k {s}, q4_k {s}, f32 {s}, f16 {s}, bf16 {s}, gate_up {s}, gate_up_q4_1 {s}, gate_up_q8_0 {s}", .{
                gpuVariantLabel(coeffs.gpu.q4_0_variant),
                gpuVariantLabel(coeffs.gpu.q4_1_variant),
                gpuVariantLabel(coeffs.gpu.q8_0_variant),
                gpuVariantLabel(coeffs.gpu.q6_k_variant),
                gpuVariantLabel(coeffs.gpu.q4_k_variant),
                gpuVariantLabel(coeffs.gpu.f32_variant),
                gpuVariantLabel(coeffs.gpu.f16_variant),
                gpuVariantLabel(coeffs.gpu.bf16_variant),
                gpuVariantLabel(coeffs.gpu.gate_up_variant),
                gpuVariantLabel(coeffs.gpu.gate_up_q4_1_variant),
                gpuVariantLabel(coeffs.gpu.gate_up_q8_0_variant),
            });
            io_utils.println(ctx.io, "gpu-gateup: {d:.2}->{d:.2} GB/s ({s})", .{
                coeffs.gpu.gate_up_gbps,
                coeffs.gpu.gate_up_u4_gbps,
                gpuVariantLabel(coeffs.gpu.gate_up_variant),
            });
            io_utils.println(ctx.io, "gpu-axes: q4_0 base {d:.2}, un2 {d:.2}, un4 {d:.2}, w32 {d:.2}, w128 {d:.2} GB/s", .{
                coeffs.gpu.gemv_gbps,
                coeffs.gpu.q4_0_un2_gbps,
                coeffs.gpu.q4_0_un4_gbps,
                coeffs.gpu.q4_0_w32_gbps,
                coeffs.gpu.q4_0_w128_gbps,
            });
        } else {
            caps.gpu = coeffs.gpu;
        }
    }
    cfg.prefill_mode = @intFromEnum(policy.mode);
    cfg.prefill_block = policy.block;
    fingerprint = cfg.fingerprint();
    fp_hex = std.fmt.bytesToHex(std.mem.asBytes(&fingerprint), .lower);
    plan_path = try store.runtimePlanPath(&plan_buf, image_name, &fp_hex);
    plan_name = try std.fmt.bufPrint(&plan_name_buf, "{s}_{s}.zplan", .{ image_name, &fp_hex });
    row.plan = plan_name;
    row.fp = fingerprint;
    if (!force and store.pathExists(ctx.io, plan_path)) {
        // Only reuse a plan whose on-disk format version is current; a stale
        // `.zplan` (after a format bump) must be re-tuned, else the daemon
        // fails `BadVersion` when it opens the session.
        const current = blk: {
            var r = plan.Reader.openFile(ctx.allocator, ctx.io, plan_path) catch break :blk false;
            r.deinit();
            break :blk true;
        };
        if (current) {
            try zrun.upsert(ctx.allocator, rdir, ctx.io, row);
            io_utils.println(ctx.io, "reuse:   {s}", .{plan_path});
            io_utils.println(ctx.io, "runtime: {s}", .{runtime_alias});
            io_utils.println(ctx.io, "image:   {s}", .{image_name});
            io_utils.println(ctx.io, "fp:      {s}", .{&fp_hex});
            if (opt.has("--prefix-cache")) try warmPrefixCache(ctx, image_name, plan_path, arch);
            return;
        }
        io_utils.println(ctx.io, "stale plan (format changed): re-tuning", .{});
    }

    var sbytes = slotBytes(dims);
    if (policy.mode == .batch) {
        for (&sbytes) |*b| b.* = std.math.mul(u64, b.*, policy.block) catch std.math.maxInt(u64);
    }
    var dims_ctx = LayerDimsCtx{ .arch = arch, .cfg = &model_cfg, .vocab = @intCast(dims.vocab) };

    // KV caches: one buffer per storing layer, appended after the slots.
    var kv = try arch.kvLayout(&model_cfg, ctx.allocator);
    defer ctx.allocator.free(kv.layers);
    const context_length = contextLengthOf(&model_cfg);
    const rope_cfg = try ropeConfigIds(&model_cfg, layer_count, ctx.allocator);
    defer ctx.allocator.free(rope_cfg);
    var slot_total: u64 = 0;
    for (sbytes) |b| slot_total += b;
    var kv_per_token: u64 = 0;
    for (kv.layers) |l| {
        if (!l.stores) continue;
        kv_per_token += 2 * @as(u64, l.kv_heads) * l.head_dim * kv.elem_bytes;
    }
    const reserve = reserveBytes(context_length, dims, caps);

    const pre_kv = ram_bytes -| reserve -| slot_total;
    // Anti-OOM/usable: reserve room for the weights (so they stay resident)
    // before sizing the KV. The context is degraded to fit, never the weights;
    // the remainder above `kv_floor` (a minimum context) reserves the weights.
    //
    // The RAM window covers only the weights the CPU reads: a weight the GPU
    // keeps resident in VRAM (`vram_resident`) is excluded, so the same bytes
    // are never accounted in both tiers. The resident set is chosen over the
    // projection weights, so the budget is first built over every range to
    // compile, then rebuilt over the reduced set; the KV/compile is redone only
    // when the reduced set changes the context.
    const kv_floor = @as(u64, min_context) * kv_per_token;

    // Provisional budget over every range, sizing the KV so the dispatch can be
    // compiled for the resident-set selection.
    var weights_bytes: u64 = 0;
    var embd_bytes: u64 = 0;
    for (ranges.items) |rg| {
        weights_bytes += rg.length;
        if (rg.kind == ztef.Kind.embd) embd_bytes += rg.length;
    }
    const want_window0 = @min(weights_bytes -| embd_bytes, pre_kv -| kv_floor);
    const budget0 = planBudget(ram_bytes, reserve, slot_total, kv_per_token, context_length, want_window0);
    var max_context = budget0.context;

    var kv_bytes: std.ArrayList(u64) = .empty;
    defer kv_bytes.deinit(ctx.allocator);
    // layer -> buffer index; storing layers get a new buffer, others point at
    // their source layer's buffer. Buffer base = slot count.
    const slot_count = sbytes.len;
    var layer_kv = try ctx.allocator.alloc(u32, layer_count);
    defer ctx.allocator.free(layer_kv);
    var kv_stores = try ctx.allocator.alloc(bool, layer_count);
    defer ctx.allocator.free(kv_stores);
    for (0..layer_count) |i| {
        kv_stores[i] = kv.layers[i].stores;
        if (kv.layers[i].stores) {
            layer_kv[i] = @intCast(slot_count + kv_bytes.items.len);
            try kv_bytes.append(ctx.allocator, kv.layerBytes(@intCast(i), max_context));
        } else {
            layer_kv[i] = layer_kv[kv.layers[i].source];
        }
    }

    var lut = TensorLut.init(ctx.allocator, tensors.items);
    defer lut.deinit();
    const layout = LayerLayout{ .count = layer_count, .kv = layer_kv };
    var composer_arena = std.heap.ArenaAllocator.init(ctx.allocator);
    defer composer_arena.deinit();
    const composed = try composer.compose(composer_arena.allocator(), arch.block);
    var base_opts = compile.Options{
        .arch = arch,
        .layer_count = layer_count,
        .threads = n_threads,
        .slot_bytes = &sbytes,
        .extra_buffers = kv_bytes.items,
        .config = &model_cfg,
        .layer_dims = LayerDimsCtx.resolve,
        .layer_dims_ctx = &dims_ctx,
        .kv_index = layer_kv,
        .kv_stores = kv_stores,
        .layer_layout = &layout,
        .max_seq = max_context,
        .rope_config = rope_cfg,
        .lookup = TensorLut.lookup,
        .lookup_ctx = &lut,
        .set_layer = TensorLut.setLayer,
    };
    var compiled = try compile.compileBlock(ctx.allocator, composed.block, base_opts);
    defer ctx.allocator.free(compiled.dispatch);
    defer ctx.allocator.free(compiled.buffers);

    var prefilled = try compile.compileBlock(ctx.allocator, arch.block, prefilledOpts(base_opts));
    defer ctx.allocator.free(prefilled.dispatch);
    defer ctx.allocator.free(prefilled.buffers);

    applyAliases(compiled.dispatch, tensor_alias.items);
    applyAliases(prefilled.dispatch, tensor_alias.items);
    try fuseDecodeMlps(ctx, image_path, tensors.items, &compiled.dispatch, prefilled.dispatch, caps, gpu, gpu_is_uma, vram orelse 0);

    // VRAM-resident set: the device-runnable decode GEMV weights (per-token
    // tier) and prefill GEMM weights (fill tier) the packer keeps within
    // `--vram` minus the driver headroom. Chosen before the RAM budget so the
    // resident bytes can leave the RAM residency. The same pass records the
    // device offload, which the guard below reads.
    var vram_resident: std.ArrayList(plan.VramResident) = .empty;
    defer vram_resident.deinit(ctx.allocator);
    var split_info: std.ArrayList(Split) = .empty;
    defer split_info.deinit(ctx.allocator);
    var splits_applied = false;
    try selectProjectionDispatch(ctx, image_path, tensors.items, compiled.dispatch, prefilled.dispatch, caps, n_threads, gpu, gpu_is_uma, vram orelse 0, pre_kv -| kv_floor, &vram_resident, &split_info);
    try selectPrefillDispatch(ctx, image_path, tensors.items, prefilled.dispatch, caps, n_threads, policy.block, gpu, vram_resident.items);

    // The precise duplication: only the CPU tail `[g, m)` of a split tensor
    // stays in the RAM residency; the GPU rows `[0, g)` live in the VRAM.
    // Shrink the range so the RAM budget (and thus the KV) never accounts for
    // the rows the CPU never reads. A range shared with another tensor is left
    // whole. Applied once; a later re-selection re-records the same splits
    // (deterministic in the rates), so the ranges are not shrunk twice.
    if (!splits_applied) {
        for (split_info.items) |sp| {
            if (sp.gpu_bytes == 0 or sp.tensor >= tensor_range.items.len) continue;
            const ri = tensor_range.items[sp.tensor];
            if (ri == plan.NONE or ri >= ranges.items.len) continue;
            var shared = false;
            for (tensor_range.items, 0..) |tr, ti| {
                if (ti != sp.tensor and tr == ri) {
                    shared = true;
                    break;
                }
            }
            if (shared) continue;
            const rg = &ranges.items[ri];
            if (sp.gpu_bytes >= rg.length) continue;
            rg.offset += sp.gpu_bytes;
            rg.length -= sp.gpu_bytes;
        }
        splits_applied = true;
    }

    // Guard: a resident weight may leave the RAM residency only when the CPU
    // never reads it -- every dispatch that references it runs on the device
    // (`device = 1`). A resident weight that also feeds a CPU dispatch (a
    // non-offloaded fused `gate_up`, a CPU prefill GEMM, an embedding dequant)
    // stays in the RAM residency, else the CPU read would fault it back in.
    const cpu_read = try ctx.allocator.alloc(bool, tensors.items.len);
    defer ctx.allocator.free(cpu_read);
    @memset(cpu_read, false);
    const prefill_slice: []const plan.Dispatch = if (policy.mode == .batch) prefilled.dispatch else &.{};
    markCpuRead(cpu_read, compiled.dispatch, prefill_slice);
    // A split weight's CPU slice reads rows `[g, m)`, so the weight must stay in
    // the RAM residency (the VRAM holds only the GPU rows).
    for (split_info.items) |sp| {
        if (sp.tensor < cpu_read.len) cpu_read[sp.tensor] = true;
    }
    const gpu_only = try ctx.allocator.alloc(bool, tensors.items.len);
    defer ctx.allocator.free(gpu_only);
    @memset(gpu_only, false);
    for (vram_resident.items) |vr| {
        const ti = vr.tensor;
        if (ti < gpu_only.len and !cpu_read[ti]) gpu_only[ti] = true;
    }

    // Drop the GPU-only ranges from the RAM residency and renumber in place, so
    // the budget, the weights list, the pinned set and the nodes never see them.
    const drop_range = try ctx.allocator.alloc(bool, ranges.items.len);
    defer ctx.allocator.free(drop_range);
    @memset(drop_range, false);
    for (gpu_only, 0..) |go, ti| {
        if (go) drop_range[tensor_range.items[ti]] = true;
    }
    const range_map = try ctx.allocator.alloc(u32, ranges.items.len);
    defer ctx.allocator.free(range_map);
    var kept_len: usize = 0;
    for (ranges.items, 0..) |rg, ri| {
        if (drop_range[ri]) {
            range_map[ri] = plan.NONE;
            continue;
        }
        range_map[ri] = @intCast(kept_len);
        ranges.items[kept_len] = rg;
        kept_len += 1;
    }
    for (tensor_range.items) |*tr| tr.* = range_map[tr.*];
    ranges.shrinkRetainingCapacity(kept_len);

    // Final budget over the reduced set: the RAM window serves only the weights
    // the CPU reads (+ the KV/row-cache).
    weights_bytes = 0;
    embd_bytes = 0;
    for (ranges.items) |rg| {
        weights_bytes += rg.length;
        if (rg.kind == ztef.Kind.embd) embd_bytes += rg.length;
    }
    const pin_weights_bytes = weights_bytes -| embd_bytes;
    // The weight budget: the RAM the CPU-read weights may take, clamped so the
    // KV keeps its floor (`pre_kv -| kv_floor`). Without the clamp a `--ram`
    // below the weights collapses the context to 1 (an unusable plan). The two
    // weight layers (`W_hot` + `W_reserved`) partition it; the KV takes the
    // leftover.
    const weight_ceiling = windowBytes(pin_weights_bytes, pre_kv, kv_floor);
    // `--residency` is the explicit ceiling for the resident weight working set.
    // Absent = no residency: nothing is pinned or prefetched, every weight
    // streams on demand from the image mmap. Present = capped at the explicit
    // value, never above the KV-floor ceiling (which keeps the KV room).
    var residency_ceiling = residency_bytes;
    if (residency_ceiling > weight_ceiling) {
        io_utils.eprintln(ctx.io, "warning: --residency {d} MB exceeds the {d} MB ceiling (the RAM left for KV/compute/scratch); residency adjusted to {d} MB", .{ residency_ceiling >> 20, weight_ceiling >> 20, weight_ceiling >> 20 });
        residency_ceiling = weight_ceiling;
    }
    // The RAM reserved for the weights. Without `--residency` the whole ceiling
    // is reserved -- the pre-flag behavior: the KV keeps its floor and the rest
    // backs the image mmap page cache, so the mmap gets the maximum. With the
    // flag the reserved set is capped at the explicit ceiling.
    const weight_budget = if (residency_bytes == 0) weight_ceiling else @min(weight_ceiling, residency_ceiling);
    const budget = planBudget(ram_bytes, reserve, slot_total, kv_per_token, context_length, weight_budget);
    max_context = budget.context;

    // The reduced set frees RAM the budget gives back to the KV, so the context
    // can grow. Rebuild the KV and recompile only when it does.
    const recompiled = max_context != budget0.context;
    if (recompiled) {
        ctx.allocator.free(compiled.dispatch);
        ctx.allocator.free(compiled.buffers);
        ctx.allocator.free(prefilled.dispatch);
        ctx.allocator.free(prefilled.buffers);
        kv_bytes.clearRetainingCapacity();
        for (0..layer_count) |i| {
            if (kv.layers[i].stores) try kv_bytes.append(ctx.allocator, kv.layerBytes(@intCast(i), max_context));
        }
        base_opts.extra_buffers = kv_bytes.items;
        base_opts.max_seq = max_context;
        compiled = try compile.compileBlock(ctx.allocator, composed.block, base_opts);
        prefilled = try compile.compileBlock(ctx.allocator, arch.block, prefilledOpts(base_opts));
        applyAliases(compiled.dispatch, tensor_alias.items);
        applyAliases(prefilled.dispatch, tensor_alias.items);
        try fuseDecodeMlps(ctx, image_path, tensors.items, &compiled.dispatch, prefilled.dispatch, caps, gpu, gpu_is_uma, vram orelse 0);
    }

    var shared_count: usize = 0;
    var shared_bytes: u64 = 0;
    for (tensor_alias.items, 0..) |alias, i| {
        if (alias == i) continue;
        shared_count += 1;
        shared_bytes += tensors.items[i].length;
    }

    var fixed: u64 = 0;
    for (compiled.buffers) |b| fixed += b.bytes;
    const window = ram_bytes -| fixed -| reserve;
    const row_cache = rowCacheBytes(embd_bytes, caps, window);
    // The RAM window the plan records (the RAM left after the fixed slots, the
    // KV and the reserve). The prefix cache sizes itself from a slice of it.
    const window_bytes = window;

    // The residency prefetch is opt-in: `--prefetch-threads M` reserves M
    // dedicated workers, **separate from and additional to** the `--threads`
    // compute pool. Absent = 0 (the prefetch runs inline on the executor). The
    // runtime reads the recorded count and decides nothing.
    cfg.prefetch_threads = prefetch_threads;
    // Heavy-duty GPU warm-up (`gpu/13`): a discrete device's memory clock only
    // ramps under **sustained** load, so a `--gpu` plan records a warm-up the
    // runtime runs at load (a sustained resident-GEMV burst). Without it the
    // decode can run at the idle 400 MHz clock (~4x slow).
    cfg.gpu_warmup = if (gpu) gpu_warmup_ms else 0;

    // The dedicated prefetch workers are additional to the compute pool, so the
    // matvec `parts` always use all `--threads` (compute = N). The resident set
    // and the device offload do not depend on the thread count, so the earlier
    // selection stands unless the dispatch was recompiled.
    const compute_threads: u16 = n_threads;
    if (recompiled) {
        vram_resident.clearRetainingCapacity();
        split_info.clearRetainingCapacity();
        try selectProjectionDispatch(ctx, image_path, tensors.items, compiled.dispatch, prefilled.dispatch, caps, compute_threads, gpu, gpu_is_uma, vram orelse 0, pre_kv -| kv_floor, &vram_resident, &split_info);
        try selectPrefillDispatch(ctx, image_path, tensors.items, prefilled.dispatch, caps, compute_threads, policy.block, gpu, vram_resident.items);
    }

    // Heterogeneous split: a resident decode GEMV the tuner decided to split
    // emits a CPU slice right after the GPU slice. Each dispatch still runs on
    // one device; the runtime applies the row ranges (`row_start`/`row_count`).
    if (split_info.items.len > 0) {
        var list: std.ArrayList(plan.Dispatch) = .empty;
        defer list.deinit(ctx.allocator);
        var si: usize = 0;
        for (compiled.dispatch, 0..) |d, i| {
            try list.append(ctx.allocator, d);
            while (si < split_info.items.len and split_info.items[si].idx == i) : (si += 1) {
                const sp = split_info.items[si];
                var cpu_slice = d;
                cpu_slice.device = 0;
                cpu_slice.row_start = sp.gpu_rows;
                cpu_slice.row_count = 0;
                cpu_slice.variant = sp.cpu_variant;
                cpu_slice.parts = sp.cpu_parts;
                cpu_slice.tile = 0;
                try list.append(ctx.allocator, cpu_slice);
            }
        }
        ctx.allocator.free(compiled.dispatch);
        compiled.dispatch = try list.toOwnedSlice(ctx.allocator);
    }
    selectAttentionDispatch(compiled.dispatch, caps, compute_threads);
    selectAttentionDispatch(prefilled.dispatch, caps, compute_threads);
    assignSimdWidth(compiled.dispatch, caps.simd_width);
    assignSimdWidth(prefilled.dispatch, caps.simd_width);
    assignDeps(compiled.dispatch);
    assignDeps(prefilled.dispatch);

    var vram_resident_bytes: u64 = 0;
    for (vram_resident.items) |vr| {
        const full = tensors.items[vr.tensor].length;
        vram_resident_bytes += if (vr.rows > 0) full / 2 else full;
    }

    if (gpu) {
        var offloaded: usize = 0;
        var split_ops: usize = 0;
        for (compiled.dispatch) |d| {
            if (d.device != 0) {
                offloaded += 1;
                if (d.row_count > 0) split_ops += 1;
            }
        }
        var pf_offloaded: usize = 0;
        for (prefilled.dispatch) |d| {
            if (d.device != 0) pf_offloaded += 1;
        }
        io_utils.println(ctx.io, "gpu:      {d}/{d} decode dispatches offloaded, {d}/{d} MB resident (budget)", .{
            offloaded,
            compiled.dispatch.len,
            vram_resident_bytes >> 20,
            ((vram orelse 0) -| plan.VRAM_DRIVER_HEADROOM) >> 20,
        });
        io_utils.println(ctx.io, "gpu:      {d}/{d} prefill dispatches offloaded", .{ pf_offloaded, prefilled.dispatch.len });
        // The device profile the oracle produced. `split` dispatches also use the
        // GPU (their lead rows), so they are counted apart from `whole-gpu`.
        const cpu_ops = compiled.dispatch.len - offloaded;
        const whole_gpu = offloaded - split_ops;
        const profile: []const u8 = if (offloaded == 0)
            "cpu"
        else if (whole_gpu == 0 and split_ops == 0)
            "gpu"
        else if (split_ops == 0)
            "mixed"
        else if (whole_gpu == 0)
            "split"
        else
            "mixed+split";
        io_utils.println(ctx.io, "profile:  {s} (whole-gpu {d}, split {d}, cpu-only {d})", .{ profile, whole_gpu, split_ops, cpu_ops });
    }

    // The affinity list has `N + M` entries: the first N for the compute pool,
    // the last M for the dedicated prefetch workers.
    const affinity = try cpu.affinityPlan(ctx.allocator, n_threads + @as(u16, prefetch_threads));
    defer ctx.allocator.free(affinity);

    // The three-pillar ring shape (`runtime/15`): assign every streamed range a
    // consumption-order block ordinal (its layer) and size the pillar to the
    // largest block. `0` (no residency) leaves the ranges un-ringed (the mmap
    // stream). The runtime derives the pillar from `block % 3`.
    const want_blocks = residency_bytes > 0 or prefetch_threads > 0;
    const ring_max = try assignRingBlocks(ctx.allocator, tensors.items, tensor_range.items, ranges.items, want_blocks);
    // Never let the three pillars exceed the explicit ceiling (anti-OOM): a
    // block larger than `R/3` is clamped, and the ranges past the pillar fall
    // back to the mmap (`Ring.slice` returns null for them).
    const ring_block_bytes: u64 = if (residency_bytes == 0) 0 else @min(ring_max, residency_bytes / 3);
    const ring_k: u16 = prefetch_threads;

    try plan.writeFile(ctx.allocator, ctx.io, plan_path, .{
        .config = cfg,
        .fingerprint = fingerprint,
        .image = image_name,
        .tensors = tensors.items,
        .buffers = compiled.buffers,
        .dispatch = compiled.dispatch,
        .prefill = if (policy.mode == .batch) prefilled.dispatch else prefilled.dispatch[0..0],
        .graph = .{
            .ranges = ranges.items,
            .window_bytes = window_bytes,
            .row_cache_bytes = row_cache,
            .ring_block_bytes = ring_block_bytes,
            .ring_k = ring_k,
        },
        .affinity = affinity,
        .vram_resident = vram_resident.items,
        .vram_resident_bytes = vram_resident_bytes,
    });
    try zrun.upsert(ctx.allocator, rdir, ctx.io, row);

    if (opt.has("--prefix-cache")) {
        try warmPrefixCache(ctx, image_name, plan_path, arch);
    }

    io_utils.println(ctx.io, "runtime:  {s}", .{runtime_alias});
    io_utils.println(ctx.io, "image:    {s}", .{image_name});
    io_utils.println(ctx.io, "threads:  {d}", .{n_threads});
    io_utils.println(ctx.io, "ram:      {d} MB", .{ram_bytes >> 20});
    io_utils.println(ctx.io, "residency: {d} MB", .{residency_bytes >> 20});
    io_utils.println(ctx.io, "vram:     {d} MB", .{(vram orelse 0) >> 20});
    io_utils.println(ctx.io, "gpu:      {s}", .{if (gpu) "on" else "off"});
    io_utils.println(ctx.io, "fp:       {s}", .{&fp_hex});
    io_utils.println(ctx.io, "tensors:  {d}", .{tensors.items.len});
    io_utils.println(ctx.io, "shared:   {d} tensors ({d} MB)", .{ shared_count, shared_bytes >> 20 });
    io_utils.println(ctx.io, "dispatch: {d}", .{compiled.dispatch.len});
    io_utils.println(ctx.io, "prefill:  {s} block {d} dispatch {d}", .{
        @tagName(policy.mode),
        policy.block,
        prefilled.dispatch.len,
    });
    io_utils.println(ctx.io, "fused:    {d} recipes/layer", .{composed.fused});
    io_utils.println(ctx.io, "buffers:  {d}", .{compiled.buffers.len});
    io_utils.println(ctx.io, "layers:   {d}", .{layer_count});
    io_utils.println(ctx.io, "ranges:   {d}", .{ranges.items.len});
    if (ring_block_bytes > 0) {
        io_utils.println(ctx.io, "ring:     block {d} MB, k {d}", .{ ring_block_bytes >> 20, ring_k });
    }
    if (cfg.prefetch_threads > 0) {
        io_utils.println(ctx.io, "prefetch: {d} dedicated worker(s)", .{cfg.prefetch_threads});
    } else {
        io_utils.println(ctx.io, "prefetch: inline", .{});
    }
    io_utils.println(ctx.io, "budget:   weights {d} MB fixed {d} MB reserve {d} MB row-cache {d} MB window {d} MB", .{
        weights_bytes >> 20,
        fixed >> 20,
        reserve >> 20,
        row_cache >> 20,
        window_bytes >> 20,
    });
    if (max_context < context_length) {
        io_utils.println(ctx.io, "degraded: context {d} -> {d}", .{ context_length, max_context });
    }
    if (window == 0) {
        io_utils.println(ctx.io, "degraded: window=0, weights stream from disk", .{});
    }
}

/// Natively measure the device coefficients, apply them to `caps`, and persist
/// the resulting profile as `<digest>.zat`. Measurement failures fall back to
/// conservative constants; a failed save leaves the in-memory profile. Returns
/// the measured coefficients so the caller can fold in a GPU probe and re-save.
fn applyCalibration(
    ctx: *Context,
    rdir: Io.Dir,
    caps: *caps_mod.Caps,
    digest_hex: []const u8,
    sample_path: []const u8,
    zat_path: []const u8,
) calibrate.Coeffs {
    const sample: ?[]const u8 = if (sample_path.len > 0) sample_path else null;
    const coeffs = calibrate.measure(ctx.allocator, ctx.io, sample);
    caps.applyCoeffs(coeffs);
    calibrate.save(ctx.allocator, rdir, ctx.io, digest_hex, .{
        .caps = caps.*,
        .coeffs = coeffs,
    }) catch {};
    io_utils.println(ctx.io, "calibrate: measured {s}", .{zat_path});
    return coeffs;
}

/// Warm the prefix cache with the architecture's chat-template opening: prefill
/// the deterministic recurring prefix through a session on the just-written
/// plan and write the `.zpc`. Random/synthetic prompts are never warmed.
fn warmPrefixCache(ctx: *Context, image_name: []const u8, plan_path: []const u8, arch: contract.Arch) !void {
    if (arch.chat.isEmpty()) {
        io_utils.println(ctx.io, "prefix-cache: skipped (no chat template)", .{});
        return;
    }
    var session = session_mod.Session.open(ctx.allocator, ctx.io, plan_path, .{ .image_path = image_name }) catch |err| {
        io_utils.eprintln(ctx.io, "prefix-cache: cannot open session: {s}", .{@errorName(err)});
        return err;
    };
    defer session.deinit();

    const ids = try session.encode(ctx.allocator, arch.chat.user_prefix);
    defer ctx.allocator.free(ids);
    try session.warmCache(ids);
    try session.writeCacheFile();
    io_utils.println(ctx.io, "prefix-cache: warmed {d} tok ({d} KB)", .{ ids.len, session.cacheBytes() >> 10 });
}

/// Record the machine's measured SIMD width (f32 lanes) on every compute
/// dispatch, so the kernels run the width the calibration found fastest. `0`
/// (unset) leaves the native width.
fn assignSimdWidth(dispatch: []plan.Dispatch, width: u8) void {
    for (dispatch) |*d| d.simd_width = width;
}

/// Populate `Dispatch.dep`: the most recent earlier dispatch that wrote this
/// dispatch's input buffer, or `NONE`. Deterministic (a backwards scan), stored
/// by the format already; the executor reads it as the dependency boundary — a
/// GPU result is drained before its first consumer. Independent siblings that
/// share one activation (q/k/v) all depend on the norm that produced it and on
/// nothing inside the run, which is what makes them batchable.
fn assignDeps(dispatch: []plan.Dispatch) void {
    for (dispatch, 0..) |*d, i| {
        d.dep = plan.NONE;
        if (d.in.kind != .buffer) continue;
        var j = i;
        while (j > 0) {
            j -= 1;
            const p = dispatch[j];
            if (p.out.kind == .buffer and p.out.ref == d.in.ref) {
                d.dep = @intCast(j);
                break;
            }
        }
    }
}

/// Mark every tensor the CPU reads: any dispatch that references it
/// (`tensor`/`tensor2`) while running on the CPU (`device == 0`). A
/// VRAM-resident weight the CPU also reads cannot leave the RAM residency --
/// excluding it would fault the CPU read back in.
fn markCpuRead(out: []bool, decode: []const plan.Dispatch, prefill: []const plan.Dispatch) void {
    for (decode) |d| {
        if (d.device != 0) continue;
        if (d.tensor != plan.NONE and d.tensor < out.len) out[d.tensor] = true;
        if (d.tensor2 != plan.NONE and d.tensor2 < out.len) out[d.tensor2] = true;
        if (d.tensor3 != plan.NONE and d.tensor3 < out.len) out[d.tensor3] = true;
    }
    for (prefill) |d| {
        if (d.device != 0) continue;
        if (d.tensor != plan.NONE and d.tensor < out.len) out[d.tensor] = true;
        if (d.tensor2 != plan.NONE and d.tensor2 < out.len) out[d.tensor2] = true;
        if (d.tensor3 != plan.NONE and d.tensor3 < out.len) out[d.tensor3] = true;
    }
}

/// Fuse each adjacent `gate_up` + `ffn_down` pair into one `mlp_fused`
/// dispatch **only when the cost model says it pays** (`gpu_policy.fuseMlps`).
/// On a UMA device the pairs that win become a single atomic dispatch; on a
/// discrete device a pair fuses only when its two ops would cross devices (the
/// `gate_up` resident, the `down` not, or the reverse), so the atomic MLP keeps
/// the whole op on one device and the `ffn_mid` off the PCIe — a pair already
/// all on the GPU is kept, so its decode program is the pre-`mlp` one (no
/// regression). The residency the cost model reads is predicted from the
/// current plan; fusing a pair changes what the packer keeps, so the pass
/// iterates to a fixed point (each round fuses at least one pair or stops, so it
/// terminates). Both kernels stay; the choice is per instruction. Replaces
/// `dispatch.*` with the result (the caller owns the new slice).
fn fuseDecodeMlps(
    ctx: *Context,
    image_path: []const u8,
    tensors: []const plan.Tensor,
    dispatch: *[]plan.Dispatch,
    prefill: []const plan.Dispatch,
    hw: caps_mod.Caps,
    gpu: bool,
    gpu_is_uma: bool,
    vram_bytes: u64,
) !void {
    if (!gpu or dispatch.len == 0) return;
    var readers: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT;
    defer for (&readers) |*r| {
        if (r.*) |*x| x.deinit();
    };
    const weights = try ctx.allocator.alloc(?gpu_policy.WeightMeta, tensors.len);
    defer ctx.allocator.free(weights);
    @memset(weights, null);
    for (tensors, 0..) |t, i| {
        const r = readerFor(ctx, image_path, &readers, t.kind) orelse continue;
        const e = r.find(t.name) orelse continue;
        if (e.n_dims != 2) continue;
        weights[i] = .{ .ggml_type = e.ggml_type, .rows = e.dims[1], .k = e.dims[0], .length = e.length };
    }
    var current: []plan.Dispatch = dispatch.*;
    var owned = false;
    var round: usize = 0;
    while (round < 64) : (round += 1) {
        const resident = try ctx.allocator.alloc(bool, tensors.len);
        defer ctx.allocator.free(resident);
        try selectResidentSet(ctx, image_path, tensors, current, prefill, gpu, gpu_is_uma, vram_bytes, resident, null);
        const next = try gpu_policy.fuseMlps(
            ctx.allocator,
            current,
            weights,
            resident,
            @intFromEnum(matvec.AtomicId.gate_up_fused),
            @intFromEnum(matvec.AtomicId.mlp_fused),
            gpu_is_uma,
            hw,
        );
        if (next.len == current.len) {
            ctx.allocator.free(next);
            break;
        }
        if (owned) ctx.allocator.free(current);
        current = next;
        owned = true;
    }
    if (owned) {
        ctx.allocator.free(dispatch.*);
        dispatch.* = current;
    }
}

/// Assign each projection dispatch its deterministic kernel choice and `parts`
/// from the hardware capabilities and the analytical cost model
/// (`tuner.selectProjection`), with no timing. Tuning runs the decode phase
/// (M=1), so every projection resolves to `matvec` and the produced plan is
/// unchanged; `parts`/`variant` come from the matvec selector. Dispatches
/// without a mapped native 2-D tensor keep the compiler's `partsFor` result.
/// A heterogeneous split the tuner recorded for a resident decode GEMV: the GPU
/// dispatch at `idx` computes rows `[0, gpu_rows)` and a CPU slice is emitted
/// right after it for rows `[gpu_rows, m)`. `cpu_variant`/`cpu_parts` are the
/// CPU selection the GPU offload overwrote.
const Split = struct {
    idx: usize,
    tensor: u32,
    gpu_rows: u32,
    /// Bytes of the GPU rows `[0, g)` — the part uploaded to the VRAM and
    /// removed from the RAM residency (the precise duplication).
    gpu_bytes: u64,
    cpu_variant: u16,
    cpu_parts: u16,
};

/// One weight op priced for the chain scheduler (`gpu/11`): its per-device cost,
/// the CPU selection and the GPU variant, so the apply pass needs no reader.
const Decision = struct {
    idx: usize,
    opts: gpu_policy.Options,
    rows: u64,
    length: u64,
    is_resident: bool,
    cpu_variant: u16,
    cpu_parts: u16,
    gpu_variant: u16,
    tensor: u32,
    tensor2: u32,
};

/// The VRAM-resident set the packer chooses for `dispatch`/`prefill`. The decode
/// GEMV weights (non-fused `matvec`) are the ROI tier -- residency saves their
/// per-token transfer -- and are ranked first; the batched-prefill `matmul`
/// weights are the fill tier. The packer saturates `vram_bytes` minus the driver
/// headroom, best-first. Fills `resident` (all false on entry) and, when
/// `resident_out` is non-null, appends the chosen tensors (with fused
/// pairs/triples expanded). Deterministic; tune-time only. Extracted so the
/// fusion pass can predict, on the current dispatch, which weights the packer
/// would keep -- a `gate_up` whose `down` is not resident crosses devices.
fn selectResidentSet(
    ctx: *Context,
    image_path: []const u8,
    tensors: []const plan.Tensor,
    dispatch: []const plan.Dispatch,
    prefill: []const plan.Dispatch,
    gpu: bool,
    gpu_is_uma: bool,
    vram_bytes: u64,
    resident: []bool,
    resident_out: ?*std.ArrayList(plan.VramResident),
) !void {
    @memset(resident, false);
    if (!gpu or tensors.len == 0) return;

    var readers: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT;
    defer for (&readers) |*r| {
        if (r.*) |*x| x.deinit();
    };

    var candidates: std.ArrayList(gpu_policy.Candidate) = .empty;
    defer candidates.deinit(ctx.allocator);
    const seen = try ctx.allocator.alloc(bool, tensors.len);
    defer ctx.allocator.free(seen);
    @memset(seen, false);
    for (dispatch) |d| {
        if (d.op != .matvec) continue;
        if (d.tensor3 != plan.NONE) {
            try addResidentTripleCandidate(ctx, image_path, &readers, tensors, seen, &candidates, gpu_is_uma, d.tensor, d.tensor2, d.tensor3);
        } else if (d.tensor2 != plan.NONE) {
            try addResidentPairCandidate(ctx, image_path, &readers, tensors, seen, &candidates, gpu_is_uma, d.tensor, d.tensor2);
        } else {
            try addResidentCandidate(ctx, image_path, &readers, tensors, seen, &candidates, gpu_is_uma, d.tensor, true);
        }
    }
    for (prefill) |d| {
        if (d.op != .matmul) continue;
        try addResidentCandidate(ctx, image_path, &readers, tensors, seen, &candidates, gpu_is_uma, d.tensor, false);
    }
    const chosen = try ctx.allocator.alloc(u32, tensors.len);
    defer ctx.allocator.free(chosen);
    const n = gpu_policy.selectResident(candidates.items, vram_bytes, chosen);
    for (chosen[0..n]) |ti| {
        resident[ti] = true;
        if (resident_out) |ro| try ro.append(ctx.allocator, .{ .tensor = ti });
    }
    // Expand a chosen fused `gate_up` pair or fused MLP triple: the packer
    // selected the gate and charged the combined bytes, so its partner(s) are
    // resident too.
    for (candidates.items) |c| {
        if (!resident[c.tensor_index]) continue;
        if (c.tensor_index2 != plan.NONE and !resident[c.tensor_index2]) {
            resident[c.tensor_index2] = true;
            if (resident_out) |ro| try ro.append(ctx.allocator, .{ .tensor = c.tensor_index2 });
        }
        if (c.tensor_index3 != plan.NONE and !resident[c.tensor_index3]) {
            resident[c.tensor_index3] = true;
            if (resident_out) |ro| try ro.append(ctx.allocator, .{ .tensor = c.tensor_index3 });
        }
    }
}

fn selectProjectionDispatch(
    ctx: *Context,
    image_path: []const u8,
    tensors: []const plan.Tensor,
    dispatch: []plan.Dispatch,
    prefill: []const plan.Dispatch,
    hw: caps_mod.Caps,
    threads: u16,
    gpu: bool,
    gpu_is_uma: bool,
    vram_bytes: u64,
    weight_ram_budget: u64,
    resident_out: *std.ArrayList(plan.VramResident),
    split_out: *std.ArrayList(Split),
) !void {
    var readers: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT;
    defer for (&readers) |*r| {
        if (r.*) |*x| x.deinit();
    };

    // First pass: the VRAM-resident set. Candidates are every weight the device
    // can run across both phases. The decode GEMV weights (non-fused `matvec`)
    // are the ROI tier -- residency saves their per-token transfer -- and are
    // ranked first; the batched-prefill `matmul` weights (which also carry
    // `ffn_up`, and `ffn_gate` when the decode path fuses `gate_up`) are the
    // fill tier, executed on the GPU in prefill. The packer saturates `--vram`
    // minus the driver headroom, best-first; a budget too small for the headroom
    // yields an empty set (the GPU degrades, the weights never do).
    const resident = try ctx.allocator.alloc(bool, tensors.len);
    defer ctx.allocator.free(resident);
    try selectResidentSet(ctx, image_path, tensors, dispatch, prefill, gpu, gpu_is_uma, vram_bytes, resident, resident_out);

    // ---- The chain scheduler (`gpu/11`): price every weight op, resolve the
    // residency budget (a bounded Lagrangian search for the RAM shadow price),
    // run the Viterbi, then apply below. ----
    var decisions: std.ArrayList(Decision) = .empty;
    defer decisions.deinit(ctx.allocator);
    for (dispatch, 0..) |d, i| {
        if (d.op != .matvec or d.tensor == plan.NONE or d.tensor >= tensors.len) continue;
        const t = tensors[d.tensor];
        const reader = readerFor(ctx, image_path, &readers, t.kind) orelse continue;
        const e = reader.find(t.name) orelse continue;
        if (e.n_dims != 2) continue;
        const sel = switch (tuner.selectProjection(hw, .decode, e.ggml_type, e.dims[0], e.dims[1], 1, threads)) {
            .matvec => |s| s,
            .matmul => continue,
        };
        var up_type = e.ggml_type;
        var down_type = e.ggml_type;
        var is_resident = resident[d.tensor];
        var op: gpu_policy.Op = .matvec;
        var combined_bytes = e.length;
        var gpu_variant: u16 = @intFromEnum(gpuQuantVariant(hw.gpu, e.ggml_type));
        if (d.tensor2 != plan.NONE) {
            op = .gate_up;
            const up_valid = d.tensor2 < tensors.len;
            const up_t = if (up_valid) tensors[d.tensor2] else undefined;
            const up_reader = if (up_valid) readerFor(ctx, image_path, &readers, up_t.kind) else null;
            const up_e = if (up_reader) |r| r.find(up_t.name) else null;
            if (up_e) |ue| up_type = ue.ggml_type;
            is_resident = up_valid and resident[d.tensor] and resident[d.tensor2];
            gpu_variant = @intFromEnum(gpuGateUpVariant(hw.gpu, e.ggml_type));
            if (up_e) |ue| combined_bytes += ue.length;
        }
        if (d.tensor3 != plan.NONE) {
            op = .mlp;
            const down_valid = d.tensor3 < tensors.len;
            const down_t = if (down_valid) tensors[d.tensor3] else undefined;
            const down_reader = if (down_valid) readerFor(ctx, image_path, &readers, down_t.kind) else null;
            const down_e = if (down_reader) |r| r.find(down_t.name) else null;
            if (down_e) |de| {
                down_type = de.ggml_type;
                combined_bytes += de.length;
            }
            is_resident = down_valid and resident[d.tensor] and resident[d.tensor2] and resident[d.tensor3];
            gpu_variant = @intFromEnum(gpuGateUpVariant(hw.gpu, e.ggml_type));
        }
        const opts = if (gpu)
            gpu_policy.costOptions(op, gpu_is_uma, e.ggml_type, up_type, down_type, e.dims[1], e.dims[0], combined_bytes, is_resident, hw)
        else
            gpu_policy.Options{ .cpu_ns = 1, .gpu_ns = std.math.inf(f64), .ram = .{ combined_bytes, 0, 0 } };
        try decisions.append(ctx.allocator, .{
            .idx = i,
            .opts = opts,
            .rows = e.dims[1],
            .length = combined_bytes,
            .is_resident = is_resident,
            .cpu_variant = @intCast(@intFromEnum(sel.plan)),
            .cpu_parts = sel.parts,
            .gpu_variant = gpu_variant,
            .tensor = d.tensor,
            .tensor2 = d.tensor2,
        });
    }

    const opts_a = try ctx.allocator.alloc(gpu_policy.Options, decisions.items.len);
    defer ctx.allocator.free(opts_a);
    const devs = try ctx.allocator.alloc(gpu_policy.Device, decisions.items.len);
    defer ctx.allocator.free(devs);
    const dp_scratch = try ctx.allocator.alloc(u8, decisions.items.len * 3 + 3);
    defer ctx.allocator.free(dp_scratch);
    for (decisions.items, 0..) |dec, j| opts_a[j] = dec.opts;
    const hop_ns = hw.dispatch_latency_ns;
    // The RAM/VRAM shadow prices are the **per-op** consequence of the device: an
    // op pays for the residency bytes its device leaves spent. The RAM price is
    // the **context's value** — a byte of RAM is worth the CPU's read time
    // (`1/cpu_rate`) — so freeing the RAM (a whole-GPU op) beats a marginally
    // cheaper split that keeps its CPU tail. It never collapses to zero, so a mix
    // (whole-GPU + split) stays possible; only a budget overflow raises it.
    const cpu_rate = if (hw.dram_bw_gbps > 0) hw.dram_bw_gbps else caps_mod.default_dram_bw_gbps;
    var ram_price: f64 = if (cpu_rate > 0) 1.0 / cpu_rate else 0;
    var dp_it: usize = 0;
    while (dp_it < 16) : (dp_it += 1) {
        gpu_policy.scheduleDevices(opts_a, hop_ns, hop_ns, ram_price, 0, devs, dp_scratch);
        var used: u64 = 0;
        for (opts_a, 0..) |o, j| used += o.ram[@intFromEnum(devs[j])];
        if (used <= weight_ram_budget) break;
        ram_price = ram_price * 4;
    }

    // The apply pass walks the scheduled decisions in order.
    var jd: usize = 0;
    for (dispatch, 0..) |*d, i| {
        if (d.op != .matvec or d.tensor == plan.NONE or d.tensor >= tensors.len) continue;
        const t = tensors[d.tensor];
        const reader = readerFor(ctx, image_path, &readers, t.kind) orelse continue;
        const e = reader.find(t.name) orelse continue;
        if (e.n_dims != 2) continue;
        switch (tuner.selectProjection(hw, .decode, e.ggml_type, e.dims[0], e.dims[1], 1, threads)) {
            .matvec => |sel| {
                d.variant = @intCast(@intFromEnum(sel.plan));
                d.simd_width = hw.simd_width;
                d.parts = sel.parts;
                // The CPU selection, kept so a split can hand it to the CPU
                // slice (the GPU offload below overwrites `variant`/`parts`).
                const cpu_variant: u16 = d.variant;
                const cpu_parts: u16 = d.parts;
                // The device oracle: one decision for every op -- cpu, gpu, or
                // split (the GPU rows `[0, g)`, the CPU `[g, m)`). A resident
                // decode GEMV may split; a streamed one is offloaded only when
                // the measured cost beats the CPU; a fused two-weight `gate_up`
                // only when both weights are resident q4_0/q4_1/q8_0, and a
                // fused three-weight MLP only when all three are. A streamed
                // weight past
                // the staging limit is sliced by output rows (`tile`).
                if (d.tensor3 != plan.NONE) {
                    const down_valid = d.tensor3 < tensors.len;
                    const down_t = if (down_valid) tensors[d.tensor3] else undefined;
                    const down_reader = if (down_valid) readerFor(ctx, image_path, &readers, down_t.kind) else null;
                    const down_e = if (down_reader) |r| r.find(down_t.name) else null;
                    const verdict = if (gpu and down_e != null and jd < devs.len)
                        gpu_policy.Verdict{ .device = devs[jd] }
                    else
                        gpu_policy.Verdict{ .device = .cpu };
                    jd += 1;
                    if (verdict.device != .cpu) {
                        d.device = 1;
                        d.parts = 1;
                        // The fused MLP is a distinct shader; it reuses the fused
                        // `gate_up` variant the probe chose per quant (the same
                        // spec-constant axes).
                        d.variant = @intFromEnum(switch (e.ggml_type) {
                            .q4_1 => hw.gpu.gate_up_q4_1_variant,
                            .q8_0 => hw.gpu.gate_up_q8_0_variant,
                            else => hw.gpu.gate_up_variant,
                        });
                    }
                    continue;
                }
                if (d.tensor2 != plan.NONE) {
                    const up_valid = d.tensor2 < tensors.len;
                    const up_t = if (up_valid) tensors[d.tensor2] else undefined;
                    const up_reader = if (up_valid) readerFor(ctx, image_path, &readers, up_t.kind) else null;
                    const up_e = if (up_reader) |r| r.find(up_t.name) else null;
                    const verdict = if (gpu and up_e != null and jd < devs.len)
                        gpu_policy.Verdict{ .device = devs[jd], .gpu_rows = opts_a[jd].gpu_rows }
                    else
                        gpu_policy.Verdict{ .device = .cpu };
                    jd += 1;
                    if (verdict.device != .cpu) {
                        d.device = 1;
                        d.parts = 1;
                        // The fused op is a distinct shader; the probe chose its
                        // variant per quant.
                        d.variant = @intFromEnum(switch (e.ggml_type) {
                            .q4_1 => hw.gpu.gate_up_q4_1_variant,
                            .q8_0 => hw.gpu.gate_up_q8_0_variant,
                            else => hw.gpu.gate_up_variant,
                        });
                        if (verdict.device == .split) {
                            d.row_count = @intCast(verdict.gpu_rows);
                            const row_bytes: u64 = if (e.dims[1] > 0) e.length / e.dims[1] else 0;
                            try split_out.append(ctx.allocator, .{
                                .idx = i,
                                .tensor = d.tensor,
                                .gpu_rows = @intCast(verdict.gpu_rows),
                                .gpu_bytes = verdict.gpu_rows * row_bytes,
                                .cpu_variant = cpu_variant,
                                .cpu_parts = cpu_parts,
                            });
                            // The pair's GPU rows are the split's lead rows.
                            for (resident_out.items) |*vr| {
                                if (vr.tensor == d.tensor or vr.tensor == d.tensor2) vr.rows = @intCast(verdict.gpu_rows);
                            }
                        }
                    }
                    continue;
                }
                const is_resident = resident[d.tensor];
                const verdict = if (gpu and jd < devs.len)
                    gpu_policy.Verdict{ .device = devs[jd], .gpu_rows = opts_a[jd].gpu_rows }
                else
                    gpu_policy.Verdict{ .device = .cpu };
                jd += 1;
                if (verdict.device != .cpu) {
                    d.device = 1;
                    d.parts = 1;
                    d.tile = gpu_policy.tileRows(e.dims[1], e.length, is_resident);
                    // Offloaded: the per-dispatch variant is the co-processor's,
                    // not the CPU matvec plan resolved above.
                    d.variant = @intFromEnum(gpuQuantVariant(hw.gpu, e.ggml_type));
                    // A split: the GPU slices rows `[0, g)` and the caller emits
                    // a CPU slice for `[g, m)` right after it.
                    if (verdict.device == .split) {
                        d.row_count = @intCast(verdict.gpu_rows);
                        d.tile = 0;
                        const row_bytes: u64 = if (e.dims[1] > 0) e.length / e.dims[1] else 0;
                        try split_out.append(ctx.allocator, .{
                            .idx = i,
                            .tensor = d.tensor,
                            .gpu_rows = @intCast(verdict.gpu_rows),
                            .gpu_bytes = verdict.gpu_rows * row_bytes,
                            .cpu_variant = cpu_variant,
                            .cpu_parts = cpu_parts,
                        });
                        for (resident_out.items) |*vr| {
                            if (vr.tensor == d.tensor) vr.rows = @intCast(verdict.gpu_rows);
                        }
                    }
                }
            },
            .matmul => {},
        }
    }
}

/// Append `ti` to the residency candidate list when it is a device-runnable
/// 2-D weight not already seen. Deduplicated by (canonical) tensor index, so a
/// content-shared alias contributes once. `per_token` marks a weight whose
/// decode GEMV residency saves a per-token transfer (the ROI tier); a prefill-
/// only weight carries `value = 0` (the fill tier).
fn addResidentCandidate(
    ctx: *Context,
    image_path: []const u8,
    readers: *[ztef.COUNT]?ztef.Reader,
    tensors: []const plan.Tensor,
    seen: []bool,
    candidates: *std.ArrayList(gpu_policy.Candidate),
    gpu_is_uma: bool,
    ti: u32,
    per_token: bool,
) !void {
    if (ti == plan.NONE or ti >= tensors.len or seen[ti]) return;
    const t = tensors[ti];
    const reader = readerFor(ctx, image_path, readers, t.kind) orelse return;
    const e = reader.find(t.name) orelse return;
    if (e.n_dims != 2) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, e.ggml_type, e.dims[1], e.dims[0])) return;
    seen[ti] = true;
    try candidates.append(ctx.allocator, .{
        .tensor_index = ti,
        .length = e.length,
        .value = if (per_token) e.length else 0,
    });
}

/// Append a fused `gate_up` pair as one residency candidate when both tensors
/// are device-runnable 2-D weights not already seen. `length`/`value` are the
/// combined raw bytes, so the packer places the pair atomically within the
/// budget; the caller expands a chosen pair to keep both resident.
fn addResidentPairCandidate(
    ctx: *Context,
    image_path: []const u8,
    readers: *[ztef.COUNT]?ztef.Reader,
    tensors: []const plan.Tensor,
    seen: []bool,
    candidates: *std.ArrayList(gpu_policy.Candidate),
    gpu_is_uma: bool,
    gate: u32,
    up: u32,
) !void {
    if (gate == plan.NONE or up == plan.NONE or gate == up) return;
    if (gate >= tensors.len or up >= tensors.len or seen[gate] or seen[up]) return;
    const tg = tensors[gate];
    const tu = tensors[up];
    const rg = readerFor(ctx, image_path, readers, tg.kind) orelse return;
    const eg = rg.find(tg.name) orelse return;
    const ru = readerFor(ctx, image_path, readers, tu.kind) orelse return;
    const eu = ru.find(tu.name) orelse return;
    if (eg.n_dims != 2 or eu.n_dims != 2) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, eg.ggml_type, eg.dims[1], eg.dims[0])) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, eu.ggml_type, eu.dims[1], eu.dims[0])) return;
    seen[gate] = true;
    seen[up] = true;
    const combined = eg.length + eu.length;
    try candidates.append(ctx.allocator, .{
        .tensor_index = gate,
        .length = combined,
        .value = combined,
        .tensor_index2 = up,
    });
}

/// Append a fused MLP triple as one residency candidate when all three tensors
/// are device-runnable 2-D weights not already seen. `length`/`value` are the
/// combined raw bytes of `gate`/`up`/`down`, so the packer places the triple
/// atomically within the budget; the caller expands a chosen triple to keep all
/// three resident.
fn addResidentTripleCandidate(
    ctx: *Context,
    image_path: []const u8,
    readers: *[ztef.COUNT]?ztef.Reader,
    tensors: []const plan.Tensor,
    seen: []bool,
    candidates: *std.ArrayList(gpu_policy.Candidate),
    gpu_is_uma: bool,
    gate: u32,
    up: u32,
    down: u32,
) !void {
    if (gate == plan.NONE or up == plan.NONE or down == plan.NONE) return;
    if (gate == up or gate == down or up == down) return;
    if (gate >= tensors.len or up >= tensors.len or down >= tensors.len) return;
    if (seen[gate] or seen[up] or seen[down]) return;
    const tg = tensors[gate];
    const tu = tensors[up];
    const td = tensors[down];
    const rg = readerFor(ctx, image_path, readers, tg.kind) orelse return;
    const eg = rg.find(tg.name) orelse return;
    const ru = readerFor(ctx, image_path, readers, tu.kind) orelse return;
    const eu = ru.find(tu.name) orelse return;
    const rd = readerFor(ctx, image_path, readers, td.kind) orelse return;
    const ed = rd.find(td.name) orelse return;
    if (eg.n_dims != 2 or eu.n_dims != 2 or ed.n_dims != 2) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, eg.ggml_type, eg.dims[1], eg.dims[0])) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, eu.ggml_type, eu.dims[1], eu.dims[0])) return;
    if (!gpu_policy.eligibleResident(gpu_is_uma, ed.ggml_type, ed.dims[1], ed.dims[0])) return;
    seen[gate] = true;
    seen[up] = true;
    seen[down] = true;
    const combined = eg.length + eu.length + ed.length;
    try candidates.append(ctx.allocator, .{
        .tensor_index = gate,
        .length = combined,
        .value = combined,
        .tensor_index2 = up,
        .tensor_index3 = down,
    });
}

/// A copy of `opts` with the prefill phase enabled (projections become `matmul`).
fn prefilledOpts(opts: compile.Options) compile.Options {
    var o = opts;
    o.prefill = true;
    return o;
}

/// Assign each batched-prefill projection (`phase = prefill`, `op = matmul`)
/// its deterministic GEMM variant and `parts` from the cost model, with M =
/// `block`. The op mapping is already baked in by the compiler; this only
/// resolves the `matmul` `AtomicId` (and its parallelism).
fn selectPrefillDispatch(
    ctx: *Context,
    image_path: []const u8,
    tensors: []const plan.Tensor,
    dispatch: []plan.Dispatch,
    hw: caps_mod.Caps,
    threads: u16,
    block: u32,
    gpu: bool,
    resident: []const plan.VramResident,
) !void {
    if (block < 2) return;
    var readers: [ztef.COUNT]?ztef.Reader = [_]?ztef.Reader{null} ** ztef.COUNT;
    defer for (&readers) |*r| {
        if (r.*) |*x| x.deinit();
    };

    for (dispatch) |*d| {
        if (d.op != .matmul or d.tensor == plan.NONE or d.tensor >= tensors.len) continue;
        const t = tensors[d.tensor];
        const reader = readerFor(ctx, image_path, &readers, t.kind) orelse continue;
        const e = reader.find(t.name) orelse continue;
        if (e.n_dims != 2) continue;
        const sel = tuner.selectMatvec(hw, e.ggml_type, e.dims[0], e.dims[1], threads);
        d.variant = @intFromEnum(sel.plan);
        d.parts = sel.parts;
        // Co-processor offload (AOT): a batched GEMM amortizes the streamed
        // weight over the `m` prompt rows, so a batched projection whose quant
        // and shape the device supports runs on the GPU (device = 1, serial).
        // With a measured device GEMM rate it runs only when that beats the CPU;
        // otherwise the residency/shape gate decides. The GPU never slices a
        // prefill weight (`tile` stays 0).
        const verdict = if (gpu)
            gpu_policy.decideDevice(.matmul, false, e.ggml_type, e.ggml_type, e.ggml_type, e.dims[1], e.dims[0], block, e.length, isResident(resident, d.tensor), .{}, hw)
        else
            gpu_policy.Verdict{ .device = .cpu, .cpu_bytes = e.length };
        if (verdict.device != .cpu) {
            d.device = 1;
            d.parts = 1;
            // The batched GEMM shaders are unvaried, so a GPU prefill dispatch
            // records the plain variant (the decode `variant` is irrelevant here).
            d.variant = 0;
        }
    }
}

/// Whether `tensor` is in the plan's VRAM-resident set (a decode-GEMV weight the
/// ROI knapsack kept). Linear scan over a short list; tune-time only.
fn isResident(resident: []const plan.VramResident, tensor: u32) bool {
    for (resident) |vr| {
        if (vr.tensor == tensor) return true;
    }
    return false;
}

/// Resolve each head-attention dispatch through the generic Selector (the same
/// path as projections): the attention family's candidates carry `applicable`/
/// cost, so the choice is `argmin opCost` rather than a fixed `AtomicId`. The
/// KV-store atomics are filtered out by their `applicable` predicate, so only
/// the head variants are considered. The sequence length is the plan's max
/// context, a stable representative shape for the AOT choice.
fn selectAttentionDispatch(dispatch: []plan.Dispatch, hw: caps_mod.Caps, threads: u16) void {
    const flash = @intFromEnum(attention.AtomicId.fused_flash_head);
    const two_pass = @intFromEnum(attention.AtomicId.two_pass_head);
    for (dispatch) |*d| {
        if (d.op != .attention or d.atomic_id != flash) continue;
        const sel = tuner.select(.attention, .{
            .caps = hw,
            .type_code = 0,
            .m = d.max_seq,
            .n = d.params[5],
            .k = 0,
            .rows = 0,
            .threads = threads,
        });
        if (sel.atomic_id == flash or sel.atomic_id == two_pass) {
            d.atomic_id = sel.atomic_id;
        }
    }
}

/// A lazily-opened ZTEF reader per kind, reused across the selection pass.
fn readerFor(
    ctx: *Context,
    image_path: []const u8,
    readers: *[ztef.COUNT]?ztef.Reader,
    kind: ztef.Kind,
) ?*ztef.Reader {
    const slot = &readers[@intFromEnum(kind)];
    if (slot.* == null) {
        var file_buf: [512]u8 = undefined;
        const zpath = std.fmt.bufPrint(&file_buf, "{s}/{s}", .{ image_path, kind.filename() }) catch return null;
        slot.* = ztef.Reader.open(ctx.allocator, ctx.io, zpath) catch return null;
    }
    return if (slot.*) |*x| x else null;
}

/// Minimum context (tokens) the budget keeps before it starts degrading, so a
/// large model still leaves room to pin its weights instead of an all-KV plan.
const min_context: u32 = 4096;

/// Heavy-duty GPU warm-up duration (ms) recorded in a `--gpu` plan (`gpu/13`):
/// a **sustained** load long enough to ramp a discrete device's memory clock.
const gpu_warmup_ms: u16 = 3000;

/// Computed budget after the KV/window inversion.
const Budget = struct {
    context: u32,
    fixed: u64,
    window: u64,
};

/// Scratch the runtime allocates at `positions` context: attention scores (one
/// f32 per position) plus the rope cos/sin rows, plus one L1 cache worth of
/// allocator/fragmentation headroom (the compute scratch is L1-resident).
/// Passed the model's full context as a safe upper bound.
fn reserveBytes(positions: u32, dims: Dims, caps: caps_mod.Caps) u64 {
    const scores = @as(u64, positions) * @sizeOf(f32);
    const rope = 2 * (dims.head_dim / 2) * @sizeOf(f32);
    return scores + rope + caps.l1_bytes;
}

/// The pinned-weight window target: the weights the CPU reads, but never more
/// than what leaves the KV its floor (`pre_kv -| kv_floor`). The clamp keeps the
/// context from collapsing below `min_context` when `--ram` is smaller than the
/// weights — the GPU-aware budget included.
fn windowBytes(pin_weights_bytes: u64, pre_kv: u64, kv_floor: u64) u64 {
    return @min(pin_weights_bytes, pre_kv -| kv_floor);
}

/// KV/window inversion: reserve `want_window` (the weight budget) from the RAM
/// first, then cap the context so slots + KV + reserve + want_window fit `ram`.
/// `want_window = 0` reproduces the KV-first result.
fn planBudget(
    ram: u64,
    reserve: u64,
    slot_total: u64,
    kv_per_token: u64,
    context_length: u32,
    want_window: u64,
) Budget {
    const pre_kv = ram -| reserve -| slot_total;
    const kv_budget = pre_kv -| want_window;
    var context: u32 = context_length;
    if (kv_per_token != 0) {
        const fits = kv_budget / kv_per_token;
        context = @intCast(@max(@min(@as(u64, context_length), fits), 1));
    }
    const fixed = slot_total + @as(u64, context) * kv_per_token;
    return .{ .context = context, .fixed = fixed, .window = ram -| fixed -| reserve };
}

/// Plan-decided budget for the runtime's embedding row cache: the smaller of
/// the embeddings' total bytes, one L3 cache (the natural size of a hot-row
/// working set), and a quarter of the window (so the row cache cannot crowd out
/// the pinned weights). 0 when there are no embeddings or no window.
/// Deterministic from `caps`; the runtime only reads it.
fn rowCacheBytes(embd_bytes: u64, caps: caps_mod.Caps, window: u64) u64 {
    if (embd_bytes == 0 or window == 0) return 0;
    const cap = @min(caps.l3_bytes, window / 4);
    return @min(embd_bytes, cap);
}

/// One tensor's content identity: the per-tensor sha256 persisted in the ZTEF
/// by `image compile`, plus the shape/type that make two equal-hash tensors
/// interchangeable. `tensor`/`range` point at the first occurrence, so later
/// duplicates alias it. Using the stored sha means the tuner never re-reads the
/// payload.
const ContentKey = struct {
    sha256: [32]u8,
    ggml_type: u32,
    n_dims: u32,
    dims: [4]u64,
    length: u64,
    tensor: u32,
    range: u32,
};

/// Whether two content keys are interchangeable. The exact shape/type/length
/// plus the content sha never aliases a tensor read with a different shape.
fn sameContent(a: ContentKey, b: ContentKey) bool {
    return std.mem.eql(u8, &a.sha256, &b.sha256) and
        a.ggml_type == b.ggml_type and
        a.n_dims == b.n_dims and
        a.length == b.length and
        std.mem.eql(u64, a.dims[0..a.n_dims], b.dims[0..b.n_dims]);
}

/// Assign each streamed range a ring block ordinal (`runtime/15`) from the
/// consumption order (the layer of the first tensor that reads it; embeddings
/// are excluded, they use the row cache) and return the pillar size (the
/// largest block). `0` when there is no residency (no ring).
fn assignRingBlocks(
    allocator: std.mem.Allocator,
    tensors: []const plan.Tensor,
    tensor_range: []const u32,
    ranges: []plan.Range,
    enable: bool,
) !u64 {
    if (!enable) return 0;
    var max_layer: u32 = 0;
    for (tensor_range, 0..) |ri, ti| {
        if (ri == plan.NONE or ri >= ranges.len) continue;
        if (ranges[ri].kind == ztef.Kind.embd) continue;
        const layer = tensors[ti].layer;
        if (layer == plan.NONE) continue;
        if (ranges[ri].block == plan.NONE or layer < ranges[ri].block) ranges[ri].block = layer;
        if (layer > max_layer) max_layer = layer;
    }
    const sums = try allocator.alloc(u64, max_layer + 1);
    defer allocator.free(sums);
    @memset(sums, 0);
    for (ranges) |rg| {
        if (rg.block == plan.NONE or rg.block >= sums.len) continue;
        sums[rg.block] += rg.length;
    }
    var max: u64 = 0;
    for (sums) |s| {
        if (s > max) max = s;
    }
    return max;
}

/// Return the existing residency range for `(kind, offset, length)` or append a
/// new one. This is the offset-level dedup; the caller first collapses
/// byte-identical tensors (same sha256/shape) to the same range.
fn findOrAddRange(
    allocator: std.mem.Allocator,
    ranges: *std.ArrayList(plan.Range),
    kind: ztef.Kind,
    offset: u64,
    length: u64,
) !u32 {
    for (ranges.items, 0..) |rg, i| {
        if (rg.kind == kind and rg.offset == offset and rg.length == length) return @intCast(i);
    }
    try ranges.append(allocator, .{
        .kind = kind,
        .offset = offset,
        .length = length,
    });
    return @intCast(ranges.items.len - 1);
}

/// Rewrite each dispatch's tensor references to their canonical (content-shared)
/// tensor, so the runtime reads and the residency prefetches one copy.
fn applyAliases(dispatch: []plan.Dispatch, alias: []const u32) void {
    for (dispatch) |*d| {
        if (d.tensor != plan.NONE and d.tensor < alias.len) d.tensor = alias[d.tensor];
        if (d.tensor2 != plan.NONE and d.tensor2 < alias.len) d.tensor2 = alias[d.tensor2];
        if (d.tensor3 != plan.NONE and d.tensor3 < alias.len) d.tensor3 = alias[d.tensor3];
    }
}

test "cpu-read guard marks the tensors a CPU dispatch reads" {
    const testing = std.testing;
    const D = plan.Dispatch;
    var out = [_]bool{ false, false, false, false, false, false };
    const decode = [_]D{
        .{ .op = .matvec, .atomic_id = 0, .device = 1, .tensor = 0 },
        .{ .op = .dequant, .atomic_id = 0, .device = 0, .tensor = 1 },
        .{ .op = .matvec, .atomic_id = 0, .device = 0, .tensor = 5, .tensor2 = 3 },
        .{ .op = .matvec, .atomic_id = 0, .device = 1, .tensor = 4 },
    };
    const prefill = [_]D{
        .{ .op = .matmul, .atomic_id = 0, .device = 1, .tensor = 4 },
        .{ .op = .matmul, .atomic_id = 0, .device = 0, .tensor = 2 },
    };
    markCpuRead(&out, &decode, &prefill);
    try testing.expect(!out[0]); // a device=1 weight only: safe to exclude
    try testing.expect(out[1]); // CPU embedding dequant
    try testing.expect(out[2]); // CPU prefill GEMM
    try testing.expect(out[3]); // tensor2 of a CPU dispatch
    try testing.expect(!out[4]); // device=1 in both phases: safe to exclude
    try testing.expect(out[5]); // CPU fused `gate_up`
}

test "sameContent matches only interchangeable tensors" {
    const testing = std.testing;
    const s0: [32]u8 = [_]u8{0} ** 32;
    var s1: [32]u8 = s0;
    s1[0] = 1;
    const base = ContentKey{ .sha256 = s0, .ggml_type = 1, .n_dims = 2, .dims = .{ 64, 8, 0, 0 }, .length = 4096, .tensor = 0, .range = 0 };
    try testing.expect(sameContent(base, base));
    var same = base;
    same.tensor = 5;
    same.range = 3;
    try testing.expect(sameContent(base, same));
    var other_dims = base;
    other_dims.dims = .{ 8, 64, 0, 0 };
    try testing.expect(!sameContent(base, other_dims));
    var other_len = base;
    other_len.length = 4097;
    try testing.expect(!sameContent(base, other_len));
    var other_type = base;
    other_type.ggml_type = 2;
    try testing.expect(!sameContent(base, other_type));
    var other_sha = base;
    other_sha.sha256 = s1;
    try testing.expect(!sameContent(base, other_sha));
}

/// Pick the largest manifest file (the most representative cold-read sample).
fn largestZtefPath(buf: []u8, image_path: []const u8, files: []const zman.FileRec) []const u8 {
    var best: ?zman.FileRec = null;
    for (files) |f| {
        if (best == null or f.byte_length > best.?.byte_length) best = f;
    }
    const f = best orelse return "";
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ image_path, f.kind.filename() }) catch "";
}

test "budget inversion keeps a positive window and reduces the context" {
    const testing = std.testing;
    const ram: u64 = 1 << 30;
    const reserve: u64 = 32 << 20;
    const slot_total: u64 = 100_000_000;
    const kv_per_token: u64 = 8192;
    const context_length: u32 = 131072;
    const want_window: u64 = 8 << 20;

    const kv_first = planBudget(ram, reserve, slot_total, kv_per_token, context_length, 0);
    const inverted = planBudget(ram, reserve, slot_total, kv_per_token, context_length, want_window);
    try testing.expect(kv_first.window < want_window);
    try testing.expect(inverted.context < kv_first.context);
    try testing.expect(inverted.context >= 1);
    try testing.expect(inverted.window > 0);
    try testing.expect(inverted.window >= want_window);
}

test "weight window never starves the KV floor (--ram below the weights)" {
    const testing = std.testing;
    const ram: u64 = 8 << 30;
    const reserve: u64 = 16 << 20;
    const slot_total: u64 = 16 << 20;
    const kv_per_token: u64 = 163840;
    const context_length: u32 = 40960;
    const pre_kv = ram -| reserve -| slot_total;
    const kv_floor = @as(u64, min_context) * kv_per_token;
    // The 14B Q8_0 weights: far larger than the 8 GB budget.
    const pin_weights_bytes: u64 = 14_854 << 20;

    // The unclamped window (the old GPU-aware branch) collapses the context to 1.
    const collapsed = planBudget(ram, reserve, slot_total, kv_per_token, context_length, pin_weights_bytes);
    try testing.expect(collapsed.context < min_context);

    // The clamped window keeps the KV floor, so the context stays usable.
    const want_window = windowBytes(pin_weights_bytes, pre_kv, kv_floor);
    try testing.expect(want_window <= pre_kv -| kv_floor);
    const budget = planBudget(ram, reserve, slot_total, kv_per_token, context_length, want_window);
    try testing.expect(budget.context >= min_context);
}

test "choosePrefillPolicy: batch when the M-row footprint fits, token otherwise" {
    const testing = std.testing;
    const dims = Dims{
        .hidden = 1024,
        .q = 1024,
        .kv = 256,
        .attn_out = 1024,
        .ffn = 4096,
        .vocab = 32000,
        .ple = 256,
        .ple_all = 256 * 32,
        .head_dim = 128,
    };

    const big = choosePrefillPolicy(8 << 30, dims);
    try testing.expectEqual(plan.PrefillMode.batch, big.mode);
    try testing.expect(big.block >= 2);
    const again = choosePrefillPolicy(8 << 30, dims);
    try testing.expectEqual(big.mode, again.mode);
    try testing.expectEqual(big.block, again.block);

    const tiny = choosePrefillPolicy(64 * 1024, dims);
    try testing.expectEqual(plan.PrefillMode.token, tiny.mode);
    try testing.expectEqual(@as(u32, 0), tiny.block);
}

/// Model dimensions the Scheduler needs to size activation buffers. Values
/// marked "tensor" are read from the actual ztef entries when present.
const Dims = struct {
    hidden: u64,
    /// q projection output (n_heads * head_dim). Tensor.
    q: u64,
    /// k/v projection output (n_kv_heads * head_dim). Tensor.
    kv: u64,
    /// attn_output projection output (hidden). Tensor.
    attn_out: u64,
    /// feed-forward width. Tensor.
    ffn: u64,
    /// lm_head output rows (vocab). Tensor.
    vocab: u64,
    /// per-layer embedding width (`embedding_length_per_layer_input`).
    ple: u64,
    /// `block_count * ple`; width of the global PLE buffers.
    ple_all: u64,
    /// attention head width (`key_length`), an upper bound on the rope rows.
    head_dim: u64,
};

/// 2-D dims captured from the image's ztef entries (0 = not seen), each the
/// maximum over all layers (sliding layers are wider than full ones).
const TensorDims = struct {
    q: u64 = 0,
    kv: u64 = 0,
    attn_out: u64 = 0,
    ffn: u64 = 0,
    vocab: u64 = 0,
};

fn dimsOf(cfg: *const @import("../../architectures/core/root.zig").Config) Dims {
    return switch (cfg.*) {
        .gemma4 => |g| .{
            .hidden = g.embedding_length,
            .q = g.head_count * g.key_length,
            .kv = (g.head_count_kv.get(0) orelse g.head_count) * g.key_length,
            .attn_out = g.embedding_length,
            .ffn = g.feed_forward_length.get(0) orelse g.embedding_length,
            .vocab = g.embedding_length,
            .ple = g.embedding_length_per_layer_input,
            .ple_all = @as(u64, g.embedding_length_per_layer_input) * g.block_count,
            .head_dim = g.key_length,
        },
        .qwen3 => |g| .{
            .hidden = g.embedding_length,
            .q = g.head_count * g.key_length,
            .kv = g.head_count_kv * g.key_length,
            .attn_out = g.embedding_length,
            .ffn = g.feed_forward_length,
            .vocab = g.embedding_length,
            .ple = 0,
            .ple_all = 0,
            .head_dim = g.key_length,
        },
    };
}

const LayerLayout = compile.LayerLayout;

/// Bridges the arch's per-layer dims with the tensor-derived vocab (which the
/// config may not carry) for the tuner's `in_len`/`out_len`.
const LayerDimsCtx = struct {
    arch: contract.Arch,
    cfg: *const @import("../../architectures/core/root.zig").Config,
    vocab: u32,

    fn resolve(ctx_ptr: ?*anyopaque, layer: u32) contract.LayerDims {
        const self: *LayerDimsCtx = @ptrCast(@alignCast(ctx_ptr.?));
        var d = self.arch.layerDims(self.cfg, layer);
        if (self.vocab != 0) d.vocab = self.vocab;
        return d;
    }
};

fn contextLengthOf(cfg: *const @import("../../architectures/core/root.zig").Config) u32 {
    return switch (cfg.*) {
        .gemma4 => |g| g.context_length,
        .qwen3 => |g| g.context_length,
    };
}

/// layer -> `zcom` rope config id, using the same first-appearance assignment
/// as the image compiler (`GemmaConfig.ropeConfigId`), so rope dispatches point
/// at the right `rope_inv_freq` region.
fn ropeConfigIds(
    cfg: *const @import("../../architectures/core/root.zig").Config,
    layer_count: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    const ids = try allocator.alloc(u32, layer_count);
    switch (cfg.*) {
        .gemma4 => |*g| for (ids, 0..) |*id, layer| {
            id.* = g.ropeConfigId(layer);
        },
        // qwen3 is uniform: a single rope config, id 0 for every layer.
        .qwen3 => {
            for (ids) |*id| id.* = 0;
        },
    }
    return ids;
}

fn layerCountOf(cfg: *const @import("../../architectures/core/root.zig").Config) u32 {
    return switch (cfg.*) {
        .gemma4 => |g| g.block_count,
        .qwen3 => |g| g.block_count,
    };
}

/// The plan's prefill policy: the mode plus the max rows per batched forward.
const PrefillPolicy = struct {
    mode: plan.PrefillMode,
    block: u32,
};

/// Deterministic prefill policy from the cost model: batched prefill pays off
/// only when a GEMM exists (the `matmul` family is registered) and the M-row
/// activation footprint fits a fraction of RAM. The block ladder keeps the plan
/// reproducible; `token` is the fallback, so the runtime default is unchanged.
fn choosePrefillPolicy(ram_bytes: u64, d: Dims) PrefillPolicy {
    const base = slotBytes(d);
    var row_bytes: u64 = 0;
    for (base) |b| row_bytes += b;
    const cap = ram_bytes / 8;
    var block: u32 = 0;
    for ([_]u32{ 2, 4, 8, 16 }) |m| {
        const need = std.math.mul(u64, row_bytes, m) catch std.math.maxInt(u64);
        if (need <= cap) block = m;
    }
    if (block < 2) return .{ .mode = .token, .block = 0 };
    return .{ .mode = .batch, .block = block };
}

/// Activation slot byte sizes, index = `@intFromEnum(graph.Slot)`.
fn slotBytes(d: Dims) [15]u64 {
    const f32sz = @sizeOf(f32);
    const hidden = d.hidden * f32sz;
    const q = d.q * f32sz;
    const kv = d.kv * f32sz;
    const attn_out = d.attn_out * f32sz;
    const ffn = d.ffn * f32sz;
    const logits = d.vocab * f32sz;
    const ple = d.ple * f32sz;
    const ple_all = d.ple_all * f32sz;
    // Slot order = `graph.Slot`: hidden, normed, q, k, v, attn_out, ffn_a,
    // ffn_b, ffn_mid, logits, emb_row, ple_token, ple_projected, ple_combined,
    // ple_a.
    return .{
        hidden, // hidden
        hidden, // normed
        q, // q
        kv, // k
        kv, // v
        attn_out, // attn_out
        ffn, // ffn_a
        ffn, // ffn_b
        ffn, // ffn_mid
        logits, // logits
        hidden, // emb_row
        ple_all,
        ple_all,
        ple_all,
        ple,
    };
}

const TensorLut = struct {
    names: []const plan.Tensor,
    allocator: std.mem.Allocator,
    table: std.StringHashMapUnmanaged(u32) = .empty,
    /// Set when resolving per-layer names so `blk.<layer>.<suffix>` maps to the
    /// concrete tensor (the block always declares layer 0).
    layer: u32 = 0,

    fn init(allocator: std.mem.Allocator, names: []const plan.Tensor) TensorLut {
        var lut = TensorLut{ .names = names, .allocator = allocator };
        for (names, 0..) |t, i| {
            lut.table.put(allocator, t.name, @intCast(i)) catch {};
        }
        return lut;
    }

    fn deinit(self: *TensorLut) void {
        self.table.deinit(self.allocator);
    }

    fn lookup(ctx: *anyopaque, name: []const u8) ?u32 {
        const self: *TensorLut = @ptrCast(@alignCast(ctx));
        if (self.table.get(name)) |idx| return idx;
        // Retry with the current layer spliced into `blk.N.`.
        if (std.mem.startsWith(u8, name, "blk.0.")) {
            var buf: [512]u8 = undefined;
            const remapped = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ self.layer, name["blk.0.".len..] }) catch return null;
            return self.table.get(remapped);
        }
        return null;
    }

    fn setLayer(ctx: *anyopaque, layer: u32) void {
        const self: *TensorLut = @ptrCast(@alignCast(ctx));
        self.layer = layer;
    }
};

test "rowCacheBytes caps at the L3 and a quarter of the window" {
    const testing = std.testing;
    var caps = caps_mod.Caps{};
    caps.l3_bytes = 16 * 1024 * 1024;

    try testing.expectEqual(@as(u64, 0), rowCacheBytes(0, caps, 1 << 30));
    try testing.expectEqual(@as(u64, 0), rowCacheBytes(1 << 30, caps, 0));

    const embd: u64 = 2 << 30;
    try testing.expectEqual(@as(u64, 16 * 1024 * 1024), rowCacheBytes(embd, caps, 1 << 30));
    try testing.expectEqual(@as(u64, 8 * 1024 * 1024), rowCacheBytes(embd, caps, 32 * 1024 * 1024));

    const small: u64 = 4 * 1024 * 1024;
    try testing.expectEqual(small, rowCacheBytes(small, caps, 1 << 30));
}

fn testCtx(items: []const []const u8) Context {
    return .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .args = @import("../args.zig").Args.init(items),
        .commands = &.{},
    };
}

test "residency requires prefetch-threads; prefetch-threads alone is allowed" {
    const testing = std.testing;
    // `--residency` without `--prefetch-threads` is rejected before any I/O.
    const only_residency = [_][]const u8{ "img", "rt", "--threads", "2", "--ram", "8gb", "--residency", "2gb" };
    var ctx_res = testCtx(&only_residency);
    try testing.expectError(error.MissingArgument, run(&ctx_res));

    // `--prefetch-threads` alone passes validation (the madvise-only mode); it
    // then fails on the missing image, not on the flags.
    const only_prefetch = [_][]const u8{ "img", "rt", "--threads", "2", "--ram", "8gb", "--prefetch-threads", "2" };
    var ctx_pf = testCtx(&only_prefetch);
    try testing.expectError(error.NotFound, run(&ctx_pf));
}

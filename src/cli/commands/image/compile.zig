const std = @import("std");
const Io = std.Io;
const io_utils = @import("../../../utils/io.zig");
const command = @import("../../command.zig");
const options = @import("../../options.zig");
const gguf = @import("../../../gguf/root.zig");
const gguf_format = @import("../../../gguf/format.zig");
const store = @import("../../../image/store.zig");
const ztef = @import("../../../image/ztef.zig");
const zman = @import("../../../image/zman.zig");
const zmeta = @import("../../../image/zmeta.zig");
const zmeta_reader = @import("../../../image/zmeta_reader.zig");
const zhash = @import("../../../image/zhash.zig");
const zcom = @import("../../../image/zcom.zig");
const quant = @import("../../../kernels/linear/quant/root.zig");
const contract = @import("../../../architectures/core/contract.zig");
const arch_root = @import("../../../architectures/root.zig");
const Context = command.Context;

/// The projection kinds, in the order the decode consumes them within a layer.
const projection_kinds = [_]ztef.Kind{ .q_proj, .k_proj, .v_proj, .o_proj, .gate_proj, .up_proj, .down_proj };

const ProjectionItem = struct { kind: ztef.Kind, item: ztef.Item };

fn projectionRank(k: ztef.Kind) u8 {
    for (projection_kinds, 0..) |pk, i| {
        if (pk == k) return @intCast(i);
    }
    return std.math.maxInt(u8);
}

fn projectionLess(_: void, a: ProjectionItem, b: ProjectionItem) bool {
    if (a.item.layer != b.item.layer) return a.item.layer < b.item.layer;
    return projectionRank(a.kind) < projectionRank(b.kind);
}

/// Merge the seven projection buckets into `misc`, ordered layer-major in decode
/// consumption order (layer asc, then q/k/v/o/gate/up/down). The decode then
/// reads one sequential `misc.ztef` stream instead of interleaving seven
/// per-kind files. `embd` and `norms` keep their own `.ztef`: the tuner's
/// row-cache/window and HOT decisions key on those kinds. The emptied per-kind
/// buckets write no file.
fn mergeProjectionBuckets(allocator: std.mem.Allocator, buckets: *[ztef.COUNT]std.ArrayList(ztef.Item)) !void {
    var merged: std.ArrayList(ProjectionItem) = .empty;
    defer merged.deinit(allocator);
    for (projection_kinds) |kind| {
        for (buckets[@intFromEnum(kind)].items) |item| {
            try merged.append(allocator, .{ .kind = kind, .item = item });
        }
    }
    std.mem.sort(ProjectionItem, merged.items, {}, projectionLess);
    for (merged.items) |entry| {
        try buckets[@intFromEnum(ztef.Kind.misc)].append(allocator, entry.item);
    }
    for (projection_kinds) |kind| buckets[@intFromEnum(kind)].clearRetainingCapacity();
}

pub const help_text =
    \\Compiles a GGUF model into a zlm image.
    \\
    \\Creates the image '<name>' with model.zman, model.zmeta and the *.ztef
    \\tensor files. Compilation is deterministic and hardware-independent.
    \\
    \\Options:
    \\  --check   parse the GGUF and print a summary; write nothing
    \\  --force   overwrite an existing image
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--check", .kind = .flag },
    .{ .long = "--force", .kind = .flag },
};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "compile: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const model_name = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <name>", .{});
        return error.MissingArgument;
    };
    const path = opt.positional(1) orelse {
        io_utils.eprintln(ctx.io, "Missing <model.gguf>", .{});
        return error.MissingArgument;
    };
    if (opt.positional(2) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }
    const check = opt.has("--check");
    const force = opt.has("--force");
    if (!store.isValidName(model_name)) {
        io_utils.eprintln(ctx.io, "Invalid image name: {s}", .{model_name});
        return error.InvalidName;
    }

    const t0 = Io.Timestamp.now(ctx.io, .awake);

    var model = gguf.Gguf.open(ctx.allocator, ctx.io, path) catch |err| {
        io_utils.eprintln(ctx.io, "Failed to read {s}: {s}", .{ path, @errorName(err) });
        return err;
    };
    defer model.deinit();

    io_utils.println(ctx.io, "name:      {s}", .{model_name});
    io_utils.println(ctx.io, "gguf:      {s}", .{path});
    io_utils.println(ctx.io, "arch:      {s}", .{model.architecture()});
    io_utils.println(ctx.io, "tensors:   {d}", .{model.header.tensor_count});
    io_utils.println(ctx.io, "metadata:  {d}", .{model.header.metadata_kv_count});
    io_utils.println(ctx.io, "size:      {d} B", .{model.mmap.size});

    if (check) return;

    if (store.exists(ctx.io, model_name) and !force) {
        io_utils.eprintln(ctx.io, "Image '{s}' already exists; use 'zlm image remove {s}' or --force", .{ model_name, model_name });
        return error.AlreadyExists;
    }

    try store.ensureBase(ctx.io);

    var temp_buf: [256]u8 = undefined;
    const temp_path = try store.tempDir(&temp_buf, model_name);
    Io.Dir.cwd().deleteTree(ctx.io, temp_path) catch {};
    var temp = try Io.Dir.cwd().createDirPathOpen(ctx.io, temp_path, .{});
    defer temp.close(ctx.io);
    errdefer Io.Dir.cwd().deleteTree(ctx.io, temp_path) catch {};

    var buckets: [ztef.COUNT]std.ArrayList(ztef.Item) = undefined;
    for (&buckets) |*b| b.* = .empty;
    defer for (&buckets) |*b| b.deinit(ctx.allocator);

    // A tensor with no declared role bucket is kept in `misc` rather than
    // silently dropped; `unbucketed` reports how many took that fallback.
    var unbucketed: usize = 0;
    for (model.tensors) |*t| {
        const bucket = ztef.bucketOfName(t.name) orelse blk: {
            unbucketed += 1;
            break :blk ztef.Bucket{ .kind = .misc, .layer = ztef.GLOBAL_LAYER };
        };
        const bytes = model.tensorBytes(t) catch |err| {
            io_utils.eprintln(ctx.io, "Cannot read tensor {s}: {s}", .{ t.name, @errorName(err) });
            return err;
        };
        try buckets[@intFromEnum(bucket.kind)].append(ctx.allocator, .{
            .name = t.name,
            .layer = bucket.layer,
            .ggml_type = t.ggml_type,
            .n_dims = t.n_dims,
            .dims = t.dims,
            .bytes = bytes,
        });
    }

    try mergeProjectionBuckets(ctx.allocator, &buckets);

    var files: std.ArrayList(zman.FileRec) = .empty;
    defer files.deinit(ctx.allocator);
    var hashes: std.ArrayList(zhash.Entry) = .empty;
    defer hashes.deinit(ctx.allocator);
    var all_shas: std.ArrayList(ztef.TensorSha) = .empty;
    defer all_shas.deinit(ctx.allocator);

    var total_bytes: u64 = 0;
    var written_tensors: u64 = 0;
    for (std.meta.tags(ztef.Kind)) |kind| {
        const items = buckets[@intFromEnum(kind)].items;
        if (items.len == 0) continue;
        const w = try ztef.write(ctx.allocator, temp, ctx.io, kind, items);
        defer ctx.allocator.free(w.tensors);
        for (w.tensors) |ts| try all_shas.append(ctx.allocator, ts);
        try files.append(ctx.allocator, .{
            .kind = kind,
            .entry_count = @intCast(items.len),
            .byte_length = w.bytes,
        });
        try hashes.append(ctx.allocator, .{ .kind = zhash.ofZtef(kind), .sha256 = w.sha256 });
        total_bytes += w.bytes;
        written_tensors += items.len;
    }

    std.mem.sort(ztef.TensorSha, all_shas.items, {}, ztef.shaLessByName);
    var meta_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(model.mmap.data[0..@intCast(model.data_offset)], &meta_hash, .{});
    var root_h = std.crypto.hash.sha2.Sha256.init(.{});
    root_h.update(&meta_hash);
    for (all_shas.items) |ts| {
        root_h.update(ts.name);
        root_h.update(&ts.sha256);
    }
    var content_root: [32]u8 = undefined;
    root_h.final(&content_root);

    const zman_out = try zman.write(ctx.allocator, temp, ctx.io, .{
        .content_root = content_root,
        .gguf_size = model.mmap.size,
        .arch = model.architecture(),
        .tensor_count = @intCast(written_tensors),
        .metadata_kv_count = model.header.metadata_kv_count,
        .files = files.items,
    });
    total_bytes += zman_out.bytes;

    const zmeta_out = try zmeta.write(ctx.allocator, temp, ctx.io, &model);
    total_bytes += zmeta_out.bytes;

    const zcom_out = try buildZcom(ctx, temp, &model);
    total_bytes += zcom_out.bytes;

    try hashes.append(ctx.allocator, .{ .kind = .zman, .sha256 = zman_out.sha256 });
    try hashes.append(ctx.allocator, .{ .kind = .zmeta, .sha256 = zmeta_out.sha256 });
    try hashes.append(ctx.allocator, .{ .kind = .zcom, .sha256 = zcom_out.sha256 });
    try zhash.write(ctx.allocator, temp, ctx.io, hashes.items);

    var final_buf: [256]u8 = undefined;
    const final_path = try store.imageDir(&final_buf, model_name);
    try store.install(ctx.io, model_name);

    const ms = @divTrunc(t0.durationTo(Io.Timestamp.now(ctx.io, .awake)).nanoseconds, 1_000_000);
    var msbuf: [32]u8 = undefined;
    const elapsed = io_utils.formatDuration(&msbuf, @floatFromInt(ms));
    const hex = std.fmt.bytesToHex(content_root, .lower);
    io_utils.println(ctx.io, "", .{});
    io_utils.println(ctx.io, "image:     {s}", .{final_path});
    io_utils.println(ctx.io, "content:   {s}", .{&hex});
    io_utils.println(ctx.io, "tensors:   {d} ({d} not bucketed)", .{ written_tensors, unbucketed });
    io_utils.println(ctx.io, "files:     {d} ztef + zman + zmeta + zcom", .{files.items.len});
    io_utils.println(ctx.io, "zcom:      {d} B", .{zcom_out.bytes});
    io_utils.println(ctx.io, "bytes:     {d}", .{total_bytes});
    io_utils.println(ctx.io, "elapsed:   {s}", .{elapsed});
}

/// Derive the image's `zcom` data: norm weights in f32 and one RoPE
/// `inv_freq` table per distinct `(freq_base, head_dim, n_rot)` configuration.
/// Deterministic (no hardware): the source bytes plus the image's own metadata.
fn buildZcom(ctx: *Context, dir: Io.Dir, model: *gguf.Gguf) !zcom.Written {
    const arch_name = model.architecture();
    var sources: std.ArrayList(zcom.Source) = .empty;
    defer sources.deinit(ctx.allocator);
    var owned: std.ArrayList([]f32) = .empty;
    defer {
        for (owned.items) |o| ctx.allocator.free(o);
        owned.deinit(ctx.allocator);
    }

    for (model.tensors) |*t| {
        const role = roleOf(arch_name, t.name) orelse continue;
        const kind: zcom.Kind = switch (role) {
            .attn_norm, .attn_q_norm, .attn_k_norm, .post_attention_norm, .ffn_norm, .post_ffw_norm, .post_norm, .output_norm, .per_layer_proj_norm, .layer_output_scale => .norm_f32,
            // Embedding tables are large; dequantizing the whole table to f32
            // is not the derived form we want (rows are looked up on demand).
            // Deferred: a row-indexed `embedding_f32` cut later.
            else => continue,
        };
        const bytes = try model.tensorBytes(t);
        const n = try gguf_format.elementCount(t.dimsSlice());
        const out = try ctx.allocator.alloc(f32, @intCast(n));
        quant.dequantTo(t.ggml_type, bytes, out) catch {
            ctx.allocator.free(out);
            continue;
        };
        try owned.append(ctx.allocator, out);
        try sources.append(ctx.allocator, .{
            .kind = kind,
            .role = @intFromEnum(role),
            .layer = layerOf(arch_name, t.name),
            .bytes = std.mem.sliceAsBytes(out),
        });
    }

    try appendRopeRegions(ctx, dir, arch_name, &sources, &owned);

    return zcom.write(ctx.allocator, dir, ctx.io, sources.items);
}

const gemma_config = @import("../../../architectures/impl/gemma4/config.zig");

/// Append one `rope_inv_freq` region per distinct RoPE config. The region's
/// `layer` is the config id (order of first appearance across the layers) and
/// `role` is 0; the payload is `n_rot/2` f32 inverse frequencies. Storing the
/// `inv_freq` (not full cos/sin) keeps `zcom` at a few KB instead of ~268 MB.
/// The id assignment (`GemmaConfig.ropeConfigId`) is shared with the tuner, so
/// the runtime resolves a layer's region by the serialized id.
fn appendRopeRegions(
    ctx: *Context,
    dir: Io.Dir,
    arch_name: []const u8,
    sources: *std.ArrayList(zcom.Source),
    owned: *std.ArrayList([]f32),
) !void {
    var md = try zmeta_reader.Zmeta.open(ctx.allocator, dir, ctx.io);
    defer md.deinit();
    var cfg = try arch_root.loadConfig(arch_name, &md, ctx.allocator);
    defer cfg.deinit();

    switch (cfg) {
        .gemma4 => |*g| {
            var next_id: u32 = 0;
            for (0..g.block_count) |layer| {
                const id = g.ropeConfigId(layer);
                if (id == gemma_config.GemmaConfig.no_rope_config) continue;
                if (id < next_id) continue;
                const rc = g.ropeConfig(layer);
                const inv = try zcom.ropeInvFreq(ctx.allocator, rc.freq_base, rc.head_dim, rc.n_rot);
                try owned.append(ctx.allocator, inv);
                try sources.append(ctx.allocator, .{
                    .kind = .rope_inv_freq,
                    .role = 0,
                    .layer = id,
                    .bytes = std.mem.sliceAsBytes(inv),
                });
                next_id += 1;
            }
        },
        // qwen3 is uniform: full rotary (`n_rot = head_dim`) with one config.
        .qwen3 => |*q| {
            const hd = q.headDim();
            const inv = try zcom.ropeInvFreq(ctx.allocator, q.rope_freq_base, hd, hd);
            try owned.append(ctx.allocator, inv);
            try sources.append(ctx.allocator, .{
                .kind = .rope_inv_freq,
                .role = 0,
                .layer = 0,
                .bytes = std.mem.sliceAsBytes(inv),
            });
        },
    }
}

fn roleOf(arch_name: []const u8, name: []const u8) ?contract.TensorRole {
    const parsed = arch_root.parse(arch_name, name) orelse return null;
    return parsed.role;
}

fn layerOf(arch_name: []const u8, name: []const u8) u32 {
    const parsed = arch_root.parse(arch_name, name) orelse return 0;
    return parsed.layer orelse 0xFFFF_FFFF;
}

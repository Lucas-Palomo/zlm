const std = @import("std");
const io_utils = @import("../../utils/io.zig");
const command = @import("../command.zig");
const options = @import("../options.zig");
const store = @import("../../image/store.zig");
const plan = @import("../../plan/root.zig");
const calibrate = @import("../../tuner/calibrate.zig");
const device = @import("../../gpu/root.zig");
const MatVecPlan = @import("../../kernels/linear/matvec/types.zig").MatVecPlan;
const MatMulAtomic = @import("../../kernels/linear/matmul/root.zig").AtomicId;
const Context = command.Context;

pub const summary = "Inspect a tuned runtime plan and device profile";
pub const usage = "zlm inspect <runtime-alias> [--plan|--image|--profile|--all]";
pub const help_text =
    \\Prints the resolved plan (config, tensors, dispatches, residency graph)
    \\and, with --profile, the device calibration profile the plan was tuned
    \\against. Read-only: it never changes the plan.
    \\
    \\  zlm inspect <runtime-alias>            plan summary
    \\  zlm inspect <runtime-alias> --plan     config + dispatches + residency
    \\  zlm inspect <runtime-alias> --image    image tensors
    \\  zlm inspect <runtime-alias> --profile  device caps + coeffs
    \\  zlm inspect <runtime-alias> --all      everything
    \\
;

const specs = [_]options.Spec{
    .{ .long = "--plan", .kind = .flag },
    .{ .long = "--image", .kind = .flag },
    .{ .long = "--profile", .kind = .flag },
    .{ .long = "--all", .kind = .flag },
};

pub const Sections = struct {
    plan: bool = false,
    image: bool = false,
    profile: bool = false,
};

pub fn run(ctx: *Context) anyerror!void {
    var opt = options.parse(ctx.allocator, &ctx.args, &specs) catch |err| {
        io_utils.eprintln(ctx.io, "inspect: {s}", .{@errorName(err)});
        return err;
    };
    defer opt.deinit();

    const alias = opt.positional(0) orelse {
        io_utils.eprintln(ctx.io, "Missing <runtime-alias>", .{});
        return error.MissingArgument;
    };
    if (opt.positional(1) != null) {
        io_utils.eprintln(ctx.io, "Too many arguments", .{});
        return error.TooManyArgs;
    }

    const all = opt.has("--all");
    const sel_plan = opt.has("--plan");
    const sel_image = opt.has("--image");
    const sel_profile = opt.has("--profile");
    const any_selected = sel_plan or sel_image or sel_profile;
    const sections = Sections{
        .plan = all or sel_plan or !any_selected,
        .image = all or sel_image,
        .profile = all or sel_profile,
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rt = store.resolveRuntime(arena.allocator(), ctx.io, alias) orelse {
        io_utils.eprintln(ctx.io, "No such runtime: {s}; run 'zlm tune <image> {s} ...'", .{ alias, alias });
        return error.NotFound;
    };

    var plan_buf: [std.fs.max_path_bytes]u8 = undefined;
    const plan_path = rt.planPath(&plan_buf) catch return error.PathTooLong;
    if (!store.pathExists(ctx.io, plan_path)) {
        io_utils.eprintln(ctx.io, "Runtime artifact missing: {s}; re-run 'zlm tune <image> {s} ...'", .{ plan_path, alias });
        return error.NotFound;
    }

    var reader = plan.Reader.openFile(ctx.allocator, ctx.io, plan_path) catch |err| {
        io_utils.eprintln(ctx.io, "inspect: cannot read plan '{s}': {s}", .{ plan_path, @errorName(err) });
        return err;
    };
    defer reader.deinit();

    var profile: ?calibrate.Profile = null;
    if (sections.profile) {
        profile = loadProfile(ctx, rt);
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(ctx.allocator);
    try render(ctx.allocator, &out, rt, reader.plan, profile, sections);
    io_utils.writeAll(ctx.io, io_utils.stdout(), out.items);
}

fn loadProfile(ctx: *Context, rt: store.Runtime) ?calibrate.Profile {
    const digest = if (std.mem.endsWith(u8, rt.tune, ".zat"))
        rt.tune[0 .. rt.tune.len - ".zat".len]
    else
        rt.tune;
    var rdir = store.openRuntimes(ctx.io) catch return null;
    defer rdir.close(ctx.io);
    return calibrate.load(ctx.allocator, rdir, ctx.io, digest);
}

/// Render the requested sections of a loaded plan into `out`.
pub fn render(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    rt: store.Runtime,
    kernel: plan.AOTKernel,
    profile: ?calibrate.Profile,
    sections: Sections,
) !void {
    try out.print(allocator, "runtime: {s}\n", .{rt.alias});
    try out.print(allocator, "image:   {s}\n", .{rt.image});
    if (sections.image) try renderTensors(allocator, out, kernel);
    if (sections.plan) try renderPlan(allocator, out, kernel);
    if (sections.profile) try renderProfile(allocator, out, rt, profile);
}

fn renderConfig(allocator: std.mem.Allocator, out: *std.ArrayList(u8), kernel: plan.AOTKernel) !void {
    const c = kernel.config;
    try out.print(allocator, "\nconfig:\n", .{});
    try out.print(allocator, "  threads:       {d}\n", .{c.threads});
    try out.print(allocator, "  ram:           {d} MB\n", .{c.ram_bytes >> 20});
    if (c.gpu) try out.print(allocator, "  vram:          {d} MB\n", .{c.vram_bytes >> 20});
    const mode: []const u8 = if (c.prefill_mode == @intFromEnum(plan.PrefillMode.batch)) "batch" else "token";
    try out.print(allocator, "  prefill_mode:  {s} (block {d})\n", .{ mode, c.prefill_block });
    try out.print(allocator, "  fingerprint:   {x}\n", .{kernel.fingerprint});
}

fn maxSeq(kernel: plan.AOTKernel) u32 {
    var m: u32 = 0;
    for (kernel.dispatch) |d| m = @max(m, d.max_seq);
    for (kernel.prefill) |d| m = @max(m, d.max_seq);
    return m;
}

fn renderTensors(allocator: std.mem.Allocator, out: *std.ArrayList(u8), kernel: plan.AOTKernel) !void {
    try out.print(allocator, "\ntensors ({d}):\n", .{kernel.tensors.len});
    for (kernel.tensors, 0..) |t, i| {
        try out.print(allocator, "  [{d:>4}] {s:<40} kind={s:<10} layer={d} len={d}\n", .{
            i, t.name, @tagName(t.kind), t.layer, t.length,
        });
    }
}

fn renderPlan(allocator: std.mem.Allocator, out: *std.ArrayList(u8), kernel: plan.AOTKernel) !void {
    try renderConfig(allocator, out, kernel);
    try out.print(allocator, "\nplan:\n", .{});
    try out.print(allocator, "  tensors:  {d}\n", .{kernel.tensors.len});
    try out.print(allocator, "  buffers:  {d}\n", .{kernel.buffers.len});
    try out.print(allocator, "  context:  {d} tokens\n", .{maxSeq(kernel)});
    try out.print(allocator, "  decode:   {d} dispatches\n", .{kernel.dispatch.len});
    try out.print(allocator, "  prefill:  {d} dispatches\n", .{kernel.prefill.len});
    try renderDispatchList(allocator, out, "decode", kernel.dispatch, kernel.tensors);
    if (kernel.prefill.len > 0) {
        try renderDispatchList(allocator, out, "prefill", kernel.prefill, kernel.tensors);
    }
    try out.print(allocator, "\nresidency:\n", .{});
    try out.print(allocator, "  ranges:   {d}\n", .{kernel.graph.ranges.len});
    try out.print(allocator, "  ring:     {d} bytes, k {d}\n", .{ kernel.graph.ring_block_bytes, kernel.graph.ring_k });
    try out.print(allocator, "  window:   {d} bytes\n", .{kernel.graph.window_bytes});
}

fn renderDispatchList(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    label: []const u8,
    list: []const plan.Dispatch,
    tensors: []const plan.Tensor,
) !void {
    try out.print(allocator, "  {s}:\n", .{label});
    for (list, 0..) |d, i| {
        try out.print(allocator, "    [{d:>5}] op={s:<12} atomic={d} variant={s:<12} simd={d} parts={d} phase={s} tensor={s}\n", .{
            i,
            @tagName(d.op),
            d.atomic_id,
            variantName(d),
            d.simd_width,
            d.parts,
            if (d.phase == 1) "prefill" else "decode",
            tensorName(tensors, d.tensor),
        });
    }
}

fn renderProfile(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    rt: store.Runtime,
    profile: ?calibrate.Profile,
) !void {
    try out.print(allocator, "\nprofile:\n", .{});
    const p = profile orelse {
        try out.print(allocator, "  (no calibration profile '{s}')\n", .{rt.tune});
        return;
    };
    const c = p.caps;
    try out.print(allocator, "  caps:\n", .{});
    try out.print(allocator, "    simd_lanes:     {d}\n", .{c.simd_lanes});
    try out.print(allocator, "    avx512:         {s}\n", .{yesNo(c.avx512)});
    try out.print(allocator, "    avx2:           {s}\n", .{yesNo(c.avx2)});
    try out.print(allocator, "    neon:           {s}\n", .{yesNo(c.neon)});
    try out.print(allocator, "    vnni:           {s}\n", .{yesNo(c.vnni)});
    try out.print(allocator, "    bf16:           {s}\n", .{yesNo(c.bf16)});
    try out.print(allocator, "    physical_cores: {d}\n", .{c.physical_cores});
    try out.print(allocator, "    logical_cpus:   {d}\n", .{c.logical_cpus});
    try out.print(allocator, "    smt:            {s}\n", .{yesNo(c.smt)});
    try out.print(allocator, "    numa_nodes:     {d}\n", .{c.numa_nodes});
    try out.print(allocator, "    l1_bytes:       {d}\n", .{c.l1_bytes});
    try out.print(allocator, "    l2_bytes:       {d}\n", .{c.l2_bytes});
    try out.print(allocator, "    l3_bytes:       {d}\n", .{c.l3_bytes});
    try out.print(allocator, "    cache_line:     {d}\n", .{c.cache_line});
    try out.print(allocator, "    base_freq_hz:   {d}\n", .{c.base_freq_hz});
    try out.print(allocator, "    cgroup_cpus:    {d}\n", .{c.cgroup_cpus});
    try out.print(allocator, "    available_cpus: {d}\n", .{c.available_cpus});
    try out.print(allocator, "  coeffs:\n", .{});
    try out.print(allocator, "    dram_bw_gbps:        {d:.2}\n", .{p.coeffs.dram_bw_gbps});
    try out.print(allocator, "    dram_bw_1t_gbps:     {d:.2}\n", .{p.coeffs.dram_bw_1t_gbps});
    try out.print(allocator, "    matvec_bw_1t_gbps:   {d:.2}\n", .{p.coeffs.matvec_bw_1t_gbps});
    try out.print(allocator, "    disk_bw_gbps:        {d:.2}\n", .{p.coeffs.disk_bw_gbps});
    try out.print(allocator, "    dispatch_latency_ns: {d:.2}\n", .{p.coeffs.dispatch_latency_ns});
    if (p.coeffs.gpu.gemv_gbps > 0 or p.coeffs.gpu.gemv_streamed_gbps > 0 or p.coeffs.gpu.matmul_gbps > 0) {
        try out.print(allocator, "    gpu_gemv_gbps:        {d:.2}\n", .{p.coeffs.gpu.gemv_gbps});
        try out.print(allocator, "    gpu_gemv_streamed:    {d:.2}\n", .{p.coeffs.gpu.gemv_streamed_gbps});
        try out.print(allocator, "    gpu_matmul_gbps:      {d:.2}\n", .{p.coeffs.gpu.matmul_gbps});
    }
    try out.print(allocator, "  variant_eff:\n", .{});
    for (std.enums.values(MatVecPlan)) |mv| {
        const e = p.coeffs.variant_eff.matvec[@intFromEnum(mv)];
        if (e != 1.0) try out.print(allocator, "    matvec.{s}: {d:.4}\n", .{ @tagName(mv), e });
    }
}

fn variantName(d: plan.Dispatch) []const u8 {
    // The variant byte is device-scoped: `device == 0` is a `MatVecPlan`,
    // `device == 1` a co-processor `GpuVariant`.
    if (d.op == .matvec and d.device == 1) {
        return @tagName(std.enums.fromInt(device.GpuVariant, d.variant) orelse .plain);
    }
    return switch (d.op) {
        .matvec => if (std.enums.fromInt(MatVecPlan, d.variant)) |p| @tagName(p) else "-",
        .matmul => if (std.math.cast(u16, d.atomic_id)) |id|
            (if (std.enums.fromInt(MatMulAtomic, id)) |a| @tagName(a) else "-")
        else
            "-",
        else => "-",
    };
}

fn tensorName(tensors: []const plan.Tensor, idx: u32) []const u8 {
    if (idx == plan.NONE or idx >= tensors.len) return "-";
    return tensors[idx].name;
}

fn yesNo(v: bool) []const u8 {
    return if (v) "yes" else "no";
}

fn testCtx(items: []const []const u8) Context {
    return .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
        .args = @import("../args.zig").Args.init(items),
        .commands = &.{},
    };
}

fn syntheticKernel(tmp: *std.testing.TmpDir) !plan.Reader {
    var tensors = [_]plan.Tensor{
        .{ .name = "blk.0.attn_q.weight", .kind = .q_proj, .layer = 0, .length = 36 },
        .{ .name = "token_embd.weight", .kind = .embd, .layer = plan.NONE, .length = 18 },
    };
    var dispatch = [_]plan.Dispatch{
        .{
            .op = .matvec,
            .atomic_id = 0,
            .tensor = 0,
            .parts = 3,
            .variant = @intFromEnum(MatVecPlan.Q4_0_fp_w4t1un1_p0),
            .in_len = 128,
            .out_len = 64,
        },
    };
    var prefill = [_]plan.Dispatch{
        .{
            .op = .matmul,
            .atomic_id = @intFromEnum(MatMulAtomic.blocked),
            .phase = 1,
            .tensor = 0,
            .parts = 2,
        },
    };
    var ranges = [_]plan.Range{
        .{ .kind = .q_proj, .offset = 0, .length = 36 },
    };
    const cfg = plan.Config{
        .threads = 8,
        .ram_bytes = 4 << 30,
        .vram_bytes = 0,
        .gpu = false,
        .cpu_isa = 42,
        .gpu_id = 0,
        .prefill_mode = 1,
        .prefill_block = 8,
    };
    var affinity = [_]u16{0};

    try plan.write(std.testing.allocator, tmp.dir, std.testing.io, .{
        .config = cfg,
        .fingerprint = cfg.fingerprint(),
        .image = "e2b",
        .tensors = &tensors,
        .buffers = &.{},
        .dispatch = &dispatch,
        .prefill = &prefill,
        .graph = .{ .ranges = &ranges, .window_bytes = 1 << 20 },
        .affinity = &affinity,
    });
    return plan.Reader.open(std.testing.allocator, tmp.dir, std.testing.io);
}

test "inspect renders the resolved plan selections" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var reader = try syntheticKernel(&tmp);
    defer reader.deinit();

    const rt = store.Runtime{ .alias = "e2b", .image = "e2b", .plan = "e2b_ab.zplan", .key = "e2b_ab", .tune = "deadbeef.zat" };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try render(testing.allocator, &out, rt, reader.plan, null, .{ .plan = true, .image = true });

    try testing.expect(std.mem.indexOf(u8, out.items, "Q4_0_fp_w4t1un1_p0") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "parts=3") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "blocked") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "blk.0.attn_q.weight") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "prefill_mode:  batch") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "residency:") != null);
    try testing.expect(std.mem.indexOf(u8, out.items, "window:   1048576 bytes") != null);
}

test "inspect reports a missing profile" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var reader = try syntheticKernel(&tmp);
    defer reader.deinit();

    const rt = store.Runtime{ .alias = "e2b", .image = "e2b", .plan = "e2b_ab.zplan", .key = "e2b_ab", .tune = "deadbeef.zat" };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try render(testing.allocator, &out, rt, reader.plan, null, .{ .profile = true });
    try testing.expect(std.mem.indexOf(u8, out.items, "no calibration profile") != null);
}

test "inspect rejects too many arguments" {
    const testing = std.testing;
    const args = [_][]const u8{ "rt-a", "extra" };
    var ctx = testCtx(&args);
    try testing.expectError(error.TooManyArgs, run(&ctx));
}

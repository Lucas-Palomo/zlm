const std = @import("std");
const dequant = @import("../quant/root.zig");
const activation = @import("../../transformer/activation.zig");
const simd = @import("../../simd.zig");

pub const ActivationKind = activation.ActivationKind;

pub const max_activation_cache_blocks: usize = 2048;

pub const KernelError = dequant.KernelError || error{
    ShapeMismatch,
    InvalidRange,
    UnsupportedPlan,
    UnsupportedLayout,
    NoMappedData,
    GpuDispatchFailed,
    OutOfMemory,
};

/// The matvec catalog grid — the L3 sweep axes shared by every generated kernel
/// catalog. A **kernel** is an L1 weight family member x an L2 dot; a **variant**
/// is this sweep. The full catalog is the product of the kernel catalog and the
/// variant grid, with two prunes applied; the position in the effective product
/// is the plan id. The whole catalog is generated from them at comptime (see
/// `MatVecPlan` / `candidates`), so the grid is the only edit needed to add or
/// drop a variant.
///
///   width    ∈ {4,8,16}   f32 lanes (fp) or i32 lanes (int)
///   tile     ∈ {1,2,4,8,16}   output rows per block sweep
///   unroll   ∈ {1,2,4,8,16}   blocks per iteration (independent accumulators)
///   prefetch ∈ {0,1}      `@prefetch` the next blocks
///   budget   = tile * unroll * ceil(width/8) ≤ 16   (AVX2: 16 YMM accumulators)
///
/// The full grid is 160 combinations; P1 (comptime ISA prune) and P2 (tune-time
/// prefetch heuristic) yield the effective catalog.
pub const grid_widths = [_]u8{ 4, 8, 16 };
pub const grid_tiles = [_]u8{ 1, 2, 4, 8, 16 };
pub const grid_unrolls = [_]u8{ 1, 2, 4, 8, 16 };
pub const grid_prefetches = [_]u8{ 0, 1 };

/// AVX2's 16 architectural YMM registers bound the live accumulators: each
/// accumulator costs `ceil(width/8)` of them, so
/// `tile * unroll * ceil(width/8) <= 16`.
pub const accumulator_budget: usize = 16;

fn widthUnits(width: u8) usize {
    return (@as(usize, width) + 7) / 8;
}

fn withinBudget(width: u8, tile: u8, unroll: u8) bool {
    return @as(usize, tile) * @as(usize, unroll) * widthUnits(width) <= accumulator_budget;
}

/// L2 — the dot algorithm. `fp` uses the L1 `fpBlock` step (a f32 partial) with
/// an f32 accumulator; `int` uses the L1 `intBlock` step (an i32 partial) against
/// a q8-quantized activation. `int` exists only for the kernels that decode
/// nibbles/signed bytes with an integer decode, so it is a property of the
/// kernel, not the variant.
pub const Dot = enum { fp, int };

/// L1 — the weight decode family members implemented as comptime types (the Q
/// family in `kernels/quantized.zig`, the F/BF families in `kernels/float.zig`
/// and `kernels/bfloat.zig`). The catalog pairs each with a dot (L2) to form a
/// kernel.
pub const KernelTag = enum { q4_0, q4_1, q4_k, q8_0, q6_k, f32, f16, bf16 };

/// One kernel = an L1 weight type x an L2 dot, with its specialization prefix.
/// `kernel = L1 x L2`; the specialization name is `<prefix>_w{W}t{T}un{U}_p{P}`.
pub const KernelSpec = struct {
    tag: KernelTag,
    dot: Dot,
    prefix: []const u8,
};

/// The kernel catalog (L1 x L2). The comptime loop over this product drives the
/// plan generation; adding a weight family member is a type + an entry here.
pub const kernel_catalog = [_]KernelSpec{
    .{ .tag = .q4_0, .dot = .fp, .prefix = "Q4_0_fp" },
    .{ .tag = .q4_0, .dot = .int, .prefix = "Q4_0_int" },
    .{ .tag = .q4_1, .dot = .fp, .prefix = "Q4_1_fp" },
    .{ .tag = .q4_1, .dot = .int, .prefix = "Q4_1_int" },
    .{ .tag = .q4_k, .dot = .fp, .prefix = "Q4_K_fp" },
    .{ .tag = .q4_k, .dot = .int, .prefix = "Q4_K_int" },
    .{ .tag = .q8_0, .dot = .fp, .prefix = "Q8_0_fp" },
    .{ .tag = .q8_0, .dot = .int, .prefix = "Q8_0_int" },
    .{ .tag = .q6_k, .dot = .fp, .prefix = "Q6_K_fp" },
    .{ .tag = .f32, .dot = .fp, .prefix = "F32_fp" },
    .{ .tag = .f16, .dot = .fp, .prefix = "F16_fp" },
    .{ .tag = .bf16, .dot = .fp, .prefix = "BF16_fp" },
};

/// P1 — the comptime ISA prune, from `simd.native` (the `-Dcpu` target).
///
/// The integer dot exists per ISA width (`pmaddwd`: SSE2 4, AVX2 8, AVX-512 16);
/// a narrower one uses half the register file, a wider one does not exist, so
/// `int` is generated only at `width == simd.native`. A f32 width wider than
/// native is lowered by *splitting*, which the Tuner measures and avoids, so it
/// is not generated either. Exact for a `-Dcpu=native` build (the perf build);
/// the runtime `applicable(caps)` gate stays as a belt-and-suspenders.
fn intGenerated(comptime width: u8) bool {
    return width == simd.native.lanes();
}

fn f32Generated(comptime width: u8) bool {
    return width <= simd.native.lanes();
}

fn variantGenerated(comptime dot: Dot, comptime width: u8) bool {
    return switch (dot) {
        .fp => f32Generated(width),
        .int => intGenerated(width),
    };
}

/// The generated name of one (kernel, variant): `<prefix>_w{W}t{T}un{U}_p{P}`.
fn planName(comptime ks: KernelSpec, comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("{s}_w{d}t{d}un{d}_p{d}", .{
        ks.prefix, v.width, v.tile, v.unroll, @intFromBool(v.prefetch),
    });
}

/// The hand-written (non-grid) plans, in serialized order. The generated
/// kernel x variant catalog is appended after these, so `production` stays `0`
/// and the base ids are stable. q8_0 and q6_k are grid families (their plans are
/// generated), so only `production` remains hand-written here.
const base_plan_names = [_][]const u8{
    "production",
};

/// The full grid (the definition, before any prune): 40 in-budget
/// `(width,tile,unroll)` points x 2 (dot) x 2 (prefetch) = 160. Shared by every
/// generated kernel catalog (the axes and prunes are dot/quant-independent).
pub const grid_total: usize = blk: {
    var n: usize = 0;
    for (grid_widths) |width| {
        for (grid_tiles) |tile| {
            for (grid_unrolls) |unroll| {
                if (withinBudget(width, tile, unroll)) n += 1;
            }
        }
    }
    break :blk n * 2 * grid_prefetches.len;
};

/// The effective variant count of one `fp` kernel after P1: every in-budget
/// point at a generated f32 width, with both prefetch settings.
pub const fp_variant_count: usize = blk: {
    @setEvalBranchQuota(100_000);
    var n: usize = 0;
    for (grid_widths) |width| {
        if (!f32Generated(width)) continue;
        for (grid_tiles) |tile| {
            for (grid_unrolls) |unroll| {
                if (withinBudget(width, tile, unroll)) n += grid_prefetches.len;
            }
        }
    }
    break :blk n;
};

/// The effective variant count of one `int` kernel after P1: every in-budget
/// point at the native i32 width, with both prefetch settings.
pub const int_variant_count: usize = blk: {
    @setEvalBranchQuota(100_000);
    var n: usize = 0;
    for (grid_widths) |width| {
        if (!intGenerated(width)) continue;
        for (grid_tiles) |tile| {
            for (grid_unrolls) |unroll| {
                if (withinBudget(width, tile, unroll)) n += grid_prefetches.len;
            }
        }
    }
    break :blk n;
};

/// The effective variant count of one kernel after P1 (`fp` or `int`).
fn variantsPerKernel(comptime dot: Dot) usize {
    return if (dot == .fp) fp_variant_count else int_variant_count;
}

/// The effective variant count of a weight family that has both an `fp` and an
/// `int` kernel (every grid family). `candidateCountForTag` is the general form.
pub const grid_count: usize = fp_variant_count + int_variant_count;

/// The generated kernel x variant product (before the plan ids): the comptime
/// loop over the kernel catalog x the variant grid, with P1 applied.
const Gen = struct {
    kernel: KernelSpec,
    variant: Variant,
};

const generated: [generated_count]Gen = blk: {
    @setEvalBranchQuota(1_000_000);
    var arr: [generated_count]Gen = undefined;
    var idx: usize = 0;
    for (kernel_catalog) |ks| {
        for (grid_widths) |width| {
            for (grid_tiles) |tile| {
                for (grid_unrolls) |unroll| {
                    if (!withinBudget(width, tile, unroll)) continue;
                    if (!variantGenerated(ks.dot, width)) continue;
                    for (grid_prefetches) |p| {
                        arr[idx] = .{
                            .kernel = ks,
                            .variant = .{ .width = width, .tile = tile, .unroll = unroll, .prefetch = p != 0 },
                        };
                        idx += 1;
                    }
                }
            }
        }
    }
    break :blk arr;
};

/// The number of generated plans (the kernel x variant product).
pub const generated_count: usize = blk: {
    var n: usize = 0;
    for (kernel_catalog) |ks| n += variantsPerKernel(ks.dot);
    break :blk n;
};

pub const plan_count: usize = base_plan_names.len + generated_count;

const plan_names = blk: {
    @setEvalBranchQuota(1_000_000);
    var names: [plan_count][]const u8 = undefined;
    for (base_plan_names, 0..) |name, i| names[i] = name;
    for (generated, 0..) |g, i| names[base_plan_names.len + i] = planName(g.kernel, g.variant);
    break :blk names;
};

/// The plan catalog, generated at comptime: the hand-written plans plus the
/// kernel x variant product. The enum ordinal is the serialized id (see
/// `plan.Dispatch.variant`); the generated ids are the base plan count plus the
/// product index. `production` is `0`, so an unset dispatch variant still means
/// production. The backing type is `u16` because the q4 family product exceeds
/// one byte.
pub const MatVecPlan = @Enum(u16, .exhaustive, &plan_names, &std.simd.iota(u16, plan_count));

/// The serialized label of a plan (its tag name).
pub fn planLabel(plan: MatVecPlan) []const u8 {
    return @tagName(plan);
}

/// Comptime descriptor of one generated matvec specialization (the L3 sweep the
/// generic body in `native.zig` is instantiated over). `width` is the f32 lane
/// count (4/8/16) or the i32 lane count for an `int` kernel, or `0` for the
/// build's native width; `tile` is the number of output rows advanced per block
/// sweep; `unroll` is the number of blocks processed per iteration with
/// independent accumulators; `prefetch` issues a `@prefetch` of the next blocks
/// in the sweep. The dot (L2) is a property of the kernel, not the variant.
pub const Variant = struct {
    width: u8 = 8,
    tile: u8 = 1,
    unroll: u8 = 1,
    prefetch: bool = false,
};

/// One generated candidate: its kernel (L1 x L2), its sweep descriptor, and the
/// `MatVecPlan` id it is serialized as.
pub const Candidate = struct {
    kernel: KernelSpec,
    variant: Variant,
    plan: MatVecPlan,
};

/// The generated kernel x variant catalog, in kernel-catalog order. The single
/// source the registry, the Selector, the runtime dispatch, the calibration and
/// `bench` all read.
pub const candidates: [generated_count]Candidate = blk: {
    var arr: [generated_count]Candidate = undefined;
    for (generated, 0..) |g, i| {
        arr[i] = .{
            .kernel = g.kernel,
            .variant = g.variant,
            .plan = @enumFromInt(base_plan_names.len + i),
        };
    }
    break :blk arr;
};

/// The number of generated candidates of one weight family: its `fp` kernel's
/// variants, plus its `int` kernel's when the family defines one.
pub fn candidateCountForTag(comptime tag: KernelTag) usize {
    var n: usize = 0;
    for (kernel_catalog) |ks| {
        if (ks.tag == tag) n += variantsPerKernel(ks.dot);
    }
    return n;
}

/// The generated candidates for one weight family, in canonical order (the
/// `fp` kernel's variants, then the `int` kernel's), with the comptime ISA prune
/// (P1) applied. This is the single source of the family's catalog — the
/// registry's kernels, the Selector's applicability, the runtime dispatch and
/// the calibration all read it, so the grid/catalog is the only edit. The
/// generated plan id is carried on each candidate.
fn candidatesForTag(comptime tag: KernelTag) [candidateCountForTag(tag)]Candidate {
    var arr: [candidateCountForTag(tag)]Candidate = undefined;
    var idx: usize = 0;
    for (candidates) |cand| {
        if (cand.kernel.tag == tag) {
            arr[idx] = cand;
            idx += 1;
        }
    }
    return arr;
}

pub const q4_0_candidates = candidatesForTag(.q4_0);
pub const q4_1_candidates = candidatesForTag(.q4_1);
pub const q4_k_candidates = candidatesForTag(.q4_k);
pub const q8_0_candidates = candidatesForTag(.q8_0);
pub const q6_k_candidates = candidatesForTag(.q6_k);
pub const f32_candidates = candidatesForTag(.f32);
pub const f16_candidates = candidatesForTag(.f16);
pub const bf16_candidates = candidatesForTag(.bf16);

/// Whether `plan` is a generated `prefetch = 1` variant (the P2 prune excludes
/// these when the op is not a streaming, memory-bound case). Covers every
/// generated kernel.
pub fn isPrefetchPlan(plan: MatVecPlan) bool {
    inline for (candidates) |cand| {
        if (plan == cand.plan) return cand.variant.prefetch;
    }
    return false;
}

/// The native-width f32 tag of one `fp` kernel, for a legacy alias
/// (`u4`/`experimental`/`q4_k_u8`).
fn nativeTag(comptime prefix: []const u8, comptime tile: u8, comptime unroll: u8) MatVecPlan {
    return std.meta.stringToEnum(
        MatVecPlan,
        std.fmt.comptimePrint("{s}_w{d}t{d}un{d}_p0", .{ prefix, simd.native.lanes(), tile, unroll }),
    ).?;
}

pub const OperationId = enum {
    mat_vec,

    pub fn label(self: OperationId) []const u8 {
        return switch (self) {
            .mat_vec => "mat_vec",
        };
    }
};

pub const MacroKernelId = enum {
    row_loop,
    q4_0_unroll4,
    q8_0_unroll4,
    q4_1_unroll4,
    q6_k_unroll4,
    q4_0_tile_rows,
    q8_0_tile_rows,
    q4_1_tile_rows,
    q6_k_tile_rows,

    pub fn label(self: MacroKernelId) []const u8 {
        return switch (self) {
            .row_loop => "row_loop",
            .q4_0_unroll4 => "q4_0_unroll4",
            .q8_0_unroll4 => "q8_0_unroll4",
            .q4_1_unroll4 => "q4_1_unroll4",
            .q6_k_unroll4 => "q6_k_unroll4",
            .q4_0_tile_rows => "q4_0_tile_rows",
            .q8_0_tile_rows => "q8_0_tile_rows",
            .q4_1_tile_rows => "q4_1_tile_rows",
            .q6_k_tile_rows => "q6_k_tile_rows",
        };
    }
};

pub const MicroKernelId = enum {
    production,
    q4_0_simd_f32x32,
    q8_0_simd_f32x32,
    q4_1_simd_f32x32,
    q4_0_scale_split_f32x32,
    q4_1_scale_split_f32x32,

    pub fn label(self: MicroKernelId) []const u8 {
        return switch (self) {
            .production => "production",
            .q4_0_simd_f32x32 => "q4_0_simd_f32x32",
            .q8_0_simd_f32x32 => "q8_0_simd_f32x32",
            .q4_1_simd_f32x32 => "q4_1_simd_f32x32",
            .q4_0_scale_split_f32x32 => "q4_0_scale_split_f32x32",
            .q4_1_scale_split_f32x32 => "q4_1_scale_split_f32x32",
        };
    }
};

pub const PackFormat = enum {
    none,

    pub fn label(self: PackFormat) []const u8 {
        return switch (self) {
            .none => "none",
        };
    }
};

/// Parse a plan name back to its plan. Any generated label matches its tag
/// name; the historical short aliases are kept for the CLI/API surface.
pub fn parseMatVecPlan(name: []const u8) ?MatVecPlan {
    if (std.mem.eql(u8, name, "prod")) return .production;
    if (std.mem.eql(u8, name, "u4")) return comptime nativeTag("Q4_0_fp", 1, 4);
    if (std.mem.eql(u8, name, "experimental")) return comptime nativeTag("Q4_0_fp", 2, 1);
    // The subsumed q4_k 8x plan aliases to its generated grid equivalent.
    if (std.mem.eql(u8, name, "q4_k_u8")) return comptime nativeTag("Q4_K_fp", 1, 8);
    return std.meta.stringToEnum(MatVecPlan, name);
}

const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../../../simd.zig");
const layout = @import("../layout.zig");

const KernelError = types.KernelError;
const Activation = layout.Activation;
const q4KXSum = layout.q4KXSum;
const max_activation_cache_blocks = types.max_activation_cache_blocks;

/// The generic single-row dot — the `single` form with m = 1: the L1 `fpBlock`
/// step accumulated over the row's blocks, then one horizontal reduce
/// (`out = reduce(acc) + corr`). `K` is an L1 weight type (`block_elems`,
/// `block_bytes`, `has_min`, `fpBlock`), so there is no per-quant body.
///
/// The `has_min` kernels fold a min term whose activation sums (`xsum`, the
/// per-32-chunk `sum(x)`) are row-independent. When `xsum` is supplied it is
/// consumed directly — the caller computes it once and reuses it across rows
/// (`lmHeadRows`), which is what keeps the q4_k path equivalent to the old
/// precomputed-xsum dot. When omitted, the sums are computed once here.
pub fn dot(comptime K: type, row: []const u8, x: []const f32, xsum: ?[]const f32) KernelError!f32 {
    if (x.len % K.block_elems != 0) return error.ShapeMismatch;
    const n_blocks = x.len / K.block_elems;
    if (row.len < n_blocks * K.block_bytes) return error.TruncatedData;

    if (comptime K.has_min) {
        if (xsum) |xs| {
            const chunks = n_blocks * (K.block_elems / 32);
            if (xs.len < chunks) return error.ShapeMismatch;
            return dotXsum(K, row, x, xs, n_blocks);
        }
        return dotComputed(K, row, x, n_blocks);
    }
    return dotXsum(K, row, x, &.{}, n_blocks);
}

/// The shared block sweep: the `lanes`-wide `fpBlock` accumulator over the row,
/// then one reduce. `xsum` is the activation sums the `has_min` kernels read
/// (empty for the others).
fn dotXsum(comptime K: type, row: []const u8, x: []const f32, xsum: []const f32, n_blocks: usize) f32 {
    const lanes: usize = comptime simd.native.lanes();
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var corr: f32 = 0;
    const act = Activation{ .x = x, .xsum = xsum };
    var rb: usize = 0;
    while (rb < n_blocks) : (rb += 1) {
        K.fpBlock(lanes, row[rb * K.block_bytes ..][0..K.block_bytes], act, rb, &acc, &corr);
    }
    return @reduce(.Add, acc) + corr;
}

/// The no-precompute `has_min` path: compute the per-32-chunk activation sums
/// once into a bounded cache, then sweep. Cold — `lmHeadRows` always supplies
/// the precompute.
fn dotComputed(comptime K: type, row: []const u8, x: []const f32, n_blocks: usize) KernelError!f32 {
    const chunks = x.len / 32;
    if (chunks > max_activation_cache_blocks) return error.ShapeMismatch;
    var buf: [max_activation_cache_blocks]f32 = undefined;
    q4KXSum(x, buf[0..chunks]);
    return dotXsum(K, row, x, buf[0..chunks], n_blocks);
}

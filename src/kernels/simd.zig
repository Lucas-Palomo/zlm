//! Compile-time SIMD width variants for the f32 kernels.
//!
//! The source is written once, parameterized by `comptime lanes`; the build
//! instantiates one specialization per `Width` for the target, and the Tuner
//! picks the fastest on the machine (recorded in the `.zplan`). `@Vector(N,T)`
//! does not enable ISA features — the target (`-mcpu`) does; a width wider than
//! the target's native width is lowered by splitting, which the Tuner measures
//! and avoids. See `sdd/tasks/kernels/11-kernels-isa.md`.

const std = @import("std");
const builtin = @import("builtin");

/// A compile-time width variant. The raw value is the f32 lane count and is the
/// plan code stored in the `.zplan`.
pub const Width = enum(u8) {
    s4 = 4,
    s8 = 8,
    s16 = 16,

    pub fn lanes(self: Width) usize {
        return @intFromEnum(self);
    }
};

/// Every width the build instantiates (the candidate set the Tuner measures).
pub const all = [_]Width{ .s4, .s8, .s16 };

/// The build target's natural f32 width: the fallback / default when the Tuner
/// has not measured. 16 on AVX-512, 8 on AVX2/AVX, 4 on NEON/SSE, else 4.
pub const native: Width = blk: {
    const f = builtin.cpu.features;
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => {
            if (std.Target.x86.featureSetHas(f, .avx512f)) break :blk .s16;
            if (std.Target.x86.featureSetHas(f, .avx2)) break :blk .s8;
        },
        .aarch64 => break :blk .s4,
        else => {},
    }
    break :blk .s4;
};

/// Decode a plan code (the raw lane count, or `0` = unset) to a `Width`,
/// falling back to the build's native width.
pub fn fromCode(raw: u8) Width {
    return std.enums.fromInt(Width, raw) orelse native;
}

/// Whether the build target has hardware fused multiply-add. `@mulAdd` on a
/// target without it lowers to a libcall (`fmaf`), so `fma` only fuses here.
pub const has_fma: bool = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .aarch64, .aarch64_be => true, // NEON `fmla`
    else => false,
};

/// `a*b + c` as a single fused multiply-add when the target has FMA (`vfmadd`
/// on AVX2/AVX-512, `fmla` on NEON), else plain `a*b + c`. The width decides the
/// register: 8 lanes → ymm on AVX2, 16 → zmm on AVX-512 (split on AVX2).
pub inline fn fma(
    comptime lanes: usize,
    a: @Vector(lanes, f32),
    b: @Vector(lanes, f32),
    c: @Vector(lanes, f32),
) @Vector(lanes, f32) {
    if (comptime has_fma) return @mulAdd(@Vector(lanes, f32), a, b, c);
    return a * b + c;
}

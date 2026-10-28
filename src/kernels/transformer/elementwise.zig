const vector = @import("../linear/vector.zig");
const simd = @import("../simd.zig");

/// Elementwise family: residual/scale combines over activation buffers.
pub const AtomicId = enum(u16) {
    /// `out = a + b`
    add,
    /// `out = a + scale * b`
    add_scaled,
    /// `out = cap * tanh(out / cap)`, in place (logit softcap).
    softcap,
    /// `out = a * b`
    mul,
    /// `out = a * scale`, in place when a == out.
    scale,
    /// `out = (a + b) * scale`
    combine,
};

pub const Args = struct {
    a: []const f32 = &.{},
    b: []const f32 = &.{},
    out: []f32 = &.{},
    scale: f32 = 1.0,
    cap: f32 = 0,
    /// Chosen SIMD width (f32 lanes 4/8/16; `0` = build's native).
    simd_width: u8 = 0,
};

fn run(id: AtomicId, args: *const Args) @import("../api.zig").KernelError!void {
    const w = simd.fromCode(args.simd_width);
    switch (id) {
        .add => switch (w) {
            .s4 => vector.addLanes(4, args.a, args.b, args.out),
            .s8 => vector.addLanes(8, args.a, args.b, args.out),
            .s16 => vector.addLanes(16, args.a, args.b, args.out),
        },
        .add_scaled => switch (w) {
            .s4 => vector.addAndScaleLanes(4, args.a, args.b, args.scale, args.out),
            .s8 => vector.addAndScaleLanes(8, args.a, args.b, args.scale, args.out),
            .s16 => vector.addAndScaleLanes(16, args.a, args.b, args.scale, args.out),
        },
        .softcap => @import("activation.zig").softcapInPlace(args.out, args.cap),
        .mul => switch (w) {
            .s4 => vector.mulLanes(4, args.a, args.b, args.out),
            .s8 => vector.mulLanes(8, args.a, args.b, args.out),
            .s16 => vector.mulLanes(16, args.a, args.b, args.out),
        },
        .scale => switch (w) {
            .s4 => vector.scaleByLanes(4, args.a, args.scale, args.out),
            .s8 => vector.scaleByLanes(8, args.a, args.scale, args.out),
            .s16 => vector.scaleByLanes(16, args.a, args.scale, args.out),
        },
        .combine => switch (w) {
            .s4 => vector.addAndScaleLanes(4, args.a, args.b, args.scale, args.out),
            .s8 => vector.addAndScaleLanes(8, args.a, args.b, args.scale, args.out),
            .s16 => vector.addAndScaleLanes(16, args.a, args.b, args.scale, args.out),
        },
    }
}

const kernels = [_]@import("../api.zig").Kernel{
    .{ .id = "add", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.add) },
    .{ .id = "add_scaled", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.add_scaled) },
    .{ .id = "softcap", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.softcap) },
    .{ .id = "mul", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.mul) },
    .{ .id = "scale", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.scale) },
    .{ .id = "combine", .op = .elementwise, .atomic_id = @intFromEnum(AtomicId.combine) },
};

pub const set = @import("../api.zig").Family{
    .op = .elementwise,
    .name = "elementwise",
    .kernels = &kernels,
    .run = @import("../api.zig").erased(AtomicId, Args, run),
};

const ggml = @import("../../../gguf/format.zig");
const api = @import("../../api.zig");
const types = @import("types.zig");
const standard = @import("standard.zig");
const k_quants = @import("k_quants.zig");
const vector = @import("../vector.zig");
const quantized = @import("../matvec/kernels/quantized.zig");
const float = @import("../matvec/kernels/float.zig");
const bfloat = @import("../matvec/kernels/bfloat.zig");

pub const encodeQ8_0Block = standard.encodeQ8_0Block;
pub const encodeQ4_0Block = standard.encodeQ4_0Block;
pub const encodeQ4_1Block = standard.encodeQ4_1Block;

/// AtomicOps of the quant family (one atomic_id per routine).
pub const AtomicId = enum(u16) { dequant, dequant_scale };

pub const Args = struct { ggml_type: u32, data: []const u8, out: []f32, scale: f32 = 1.0 };

/// The generic dequant: decode `out.len` f32 from the L1 weight type `K`, one
/// block at a time. `K` supplies `block_elems`/`block_bytes` and the L1
/// `dequantBlock` step (the same block granularity the matvec `fpBlock` uses),
/// so a weight family member gets a dequant from its type — including the
/// dequant-only `q5_k` — with no per-type body here.
pub fn dequant(comptime K: type, data: []const u8, out: []f32) api.KernelError!void {
    const be: usize = K.block_elems;
    const full = out.len / be;
    const rem = out.len % be;
    var pos: usize = 0;
    var b: usize = 0;
    while (b < full) : (b += 1) {
        try K.dequantBlock(data[pos..], out[b * be ..][0..be]);
        pos += K.block_bytes;
    }
    if (rem != 0) {
        var tail: [be]f32 = undefined;
        try K.dequantBlock(data[pos..], tail[0..rem]);
        @memcpy(out[full * be ..], tail[0..rem]);
    }
}

/// The type-keyed dequant entry point the runtime needs: map the ggml type to
/// its L1 weight type and delegate to the generic `dequant`.
pub fn dequantTo(ggml_type: ggml.GGMLType, data: []const u8, out: []f32) api.KernelError!void {
    return switch (ggml_type) {
        .f32 => dequant(float.F32, data, out),
        .f16 => dequant(float.F16, data, out),
        .bf16 => dequant(bfloat.BF16, data, out),
        .q8_0 => dequant(quantized.Q8_0, data, out),
        .q4_0 => dequant(quantized.Q4_0, data, out),
        .q4_1 => dequant(quantized.Q4_1, data, out),
        .q4_k => dequant(quantized.Q4_K, data, out),
        .q5_k => dequant(quantized.Q5_K, data, out),
        .q6_k => dequant(quantized.Q6_K, data, out),
        else => error.UnsupportedType,
    };
}

pub fn dequantScaledTo(ggml_type: ggml.GGMLType, data: []const u8, scale: f32, out: []f32) api.KernelError!void {
    try dequantTo(ggml_type, data, out);
    vector.scaleBy(out, scale, out);
}

fn run(id: AtomicId, args: *const Args) api.KernelError!void {
    switch (id) {
        .dequant => {
            const t = ggml.GGMLType.fromInt(args.ggml_type) orelse return error.UnsupportedType;
            return dequantTo(t, args.data, args.out);
        },
        .dequant_scale => {
            const t = ggml.GGMLType.fromInt(args.ggml_type) orelse return error.UnsupportedType;
            return dequantScaledTo(t, args.data, args.scale, args.out);
        },
    }
}

const kernels = [_]api.Kernel{
    .{ .id = "dequant", .op = .dequant, .atomic_id = @intFromEnum(AtomicId.dequant) },
    .{ .id = "dequant_scale", .op = .dequant, .atomic_id = @intFromEnum(AtomicId.dequant_scale), .fusable = true },
};

pub const set = api.Family{
    .op = .dequant,
    .name = "quant",
    .kernels = &kernels,
    .run = api.erased(AtomicId, Args, run),
};

// ---- family surface (helpers consumed by matvec) ----
pub const KernelError = api.KernelError;
pub const elementCount = types.elementCount;
pub const q4_0RowBytes = types.q4_0RowBytes;
pub const q4_1RowBytes = types.q4_1RowBytes;
pub const q8_0RowBytes = types.q8_0RowBytes;
pub const q6KRowBytes = types.q6KRowBytes;
pub const q4KRowBytes = types.q4KRowBytes;
pub const q5KRowBytes = types.q5KRowBytes;
pub const getScaleMinK4 = k_quants.getScaleMinK4;

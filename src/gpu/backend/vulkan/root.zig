const std = @import("std");
const Io = std.Io;
const contract = @import("../root.zig");
const grid = @import("../../kernels/grid.zig");
const context = @import("context.zig");
const pipeline = @import("pipeline.zig");

pub const Quant = contract.Quant;
pub const GpuVariant = grid.GpuVariant;
pub const Resident = contract.Resident;
pub const GateUpAct = contract.GateUpAct;
pub const Handle = contract.Handle;
pub const ResidentOp = contract.ResidentOp;
pub const RingSize = contract.RingSize;
pub const Probe = contract.Probe;
pub const Backend = enum { none, vulkan };

/// Backend-agnostic co-processor handle. Ops are streamed to the GPU one at a
/// time and every op falls back to the CPU when the device is unavailable or
/// rejects the dispatch. On a machine without a working Vulkan device this is a
/// valid, inert object (`available() == false`); construction never fails.
pub const Device = struct {
    allocator: std.mem.Allocator = undefined,
    backend: Backend = .none,
    vk_ctx: ?context.VulkanContext = null,
    vk_pipe: ?pipeline.VulkanPipeline = null,

    pub fn init(allocator: std.mem.Allocator, io: Io) Device {
        var vk_ctx = context.VulkanContext.init(allocator, io);
        if (vk_ctx.status == .vulkan) {
            if (pipeline.VulkanPipeline.init(&vk_ctx)) |vk_pipe| {
                return Device{
                    .allocator = allocator,
                    .backend = .vulkan,
                    .vk_ctx = vk_ctx,
                    .vk_pipe = vk_pipe,
                };
            }
        }
        if (vk_ctx.status == .vulkan) vk_ctx.deinit();
        return Device{ .allocator = allocator };
    }

    pub fn deinit(self: *Device) void {
        if (self.vk_pipe) |*pipe| {
            pipe.deinit();
            self.vk_pipe = null;
        }
        if (self.vk_ctx) |*vk| {
            vk.deinit();
            self.vk_ctx = null;
        }
        self.backend = .none;
    }

    pub fn available(self: *const Device) bool {
        return self.vk_pipe != null;
    }

    pub fn backendName(self: *const Device) []const u8 {
        return switch (self.backend) {
            .none => "off",
            .vulkan => "vulkan",
        };
    }

    pub fn deviceName(self: *const Device) []const u8 {
        if (self.vk_ctx) |*vk| return vk.deviceNameSlice();
        return "off";
    }

    pub fn isUma(self: *const Device) bool {
        if (self.vk_ctx) |*vk| return vk.isUma();
        return false;
    }

    /// GEMV: `out[row] = dot(dequant(weight[row]), x)` for `rows` rows, where
    /// `weight_bytes` is the raw ggml quantized tensor (`rows * k` elements) and
    /// `x` has `k` f32. Returns false when the op did not run on the GPU, so the
    /// caller can use the CPU kernel.
    pub fn matVec(self: *Device, quant: Quant, weight_bytes: []const u8, rows: usize, k: usize, x: []const f32, out: []f32) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.dispatchMatVec(quant, weight_bytes, rows, k, x, out);
        }
        return false;
    }

    /// Batched GEMM: `out[t*rows + r] = dot(dequant(weight[r]), x[t])`, where
    /// `weight_bytes` is the raw quantized `[k, rows]` tensor, `x` has `m*k`
    /// f32, and `out` has `m*rows` f32. Mirrors `runPrefillMatmul`/`matVecBatch`
    /// but computes the whole prompt block in one dispatch, amortizing the
    /// weight transfer over the `m` tokens. Returns false when the op did not
    /// run on the GPU, so the caller can use the CPU kernel.
    pub fn matMulBatch(self: *Device, quant: Quant, weight_bytes: []const u8, rows: usize, k: usize, m: usize, x: []const f32, out: []f32) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.dispatchMatMul(quant, weight_bytes, rows, k, m, x, out);
        }
        return false;
    }

    /// Batched GEMM over a persistent resident weight buffer (`uploadResident`):
    /// only the activation/result are staged per call, so a prefill projection
    /// whose weight the decode ROI kept resident is not re-uploaded. Returns
    /// false when the op did not run on the GPU, so the caller can use the CPU
    /// kernel or the streamed `matMulBatch` path.
    pub fn matMulResident(self: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, m: usize, x: []const f32, out: []f32) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.dispatchMatMulResident(res, quant, rows, k, m, x, out);
        }
        return false;
    }

    /// GEMV over a persistent resident weight buffer (`uploadResident`): only
    /// the activation/result are staged per call. Returns false when the op did
    /// not run on the GPU, so the caller can use the CPU kernel.
    pub fn matVecResident(self: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, out: []f32) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.dispatchMatVecResident(res, quant, rows, k, x, out);
        }
        return false;
    }

    /// Submit a resident GEMV without waiting, returning a handle to drain with
    /// `waitResident`. Returns null when the device is unavailable or rejects
    /// the dispatch. This is the async seam the executor uses to overlap GPU
    /// work with the next dispatch.
    pub fn matVecResidentAsync(self: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32) ?Handle {
        if (self.vk_pipe) |*pipe| {
            return pipe.submitResidentGemv(res, quant, rows, k, x);
        }
        return null;
    }

    /// Submit a resident GEMV that reads the shared burst activation: `first`
    /// stages `x` into the device-owned buffer (one upload for the whole run of
    /// GEMVs that share the input), later members reuse it. `x` must be the
    /// same slice the burst's first submission carried.
    pub fn matVecResidentBurstAsync(self: *Device, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, first: bool) ?Handle {
        if (self.vk_pipe) |*pipe| {
            return pipe.submitResidentGemvShared(res, quant, rows, k, x, first);
        }
        return null;
    }

    /// Submit a **fused** resident GEMV that reads two resident weights (the
    /// `gate` and `up` projections) over the shared burst activation and applies
    /// the gated `act` (`out[i] = act(dot(gate_i, x), dot(up_i, x))`). `first`
    /// stages `x` exactly like `matVecResidentBurstAsync`; later members of the
    /// same burst reuse it. `quant` selects the fused shader (q4_0 or q8_0);
    /// `variant` selects the shader variant (q4_0 only). Returns null when the
    /// device is unavailable or rejects the dispatch.
    pub fn matVecGateUpResidentBurstAsync(self: *Device, res_gate: *Resident, res_up: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.vk_pipe) |*pipe| {
            return pipe.submitResidentGateUpShared(res_gate, res_up, quant, rows, k, x, act, variant, first);
        }
        return null;
    }

    /// Submit a **fused MLP** resident dispatch that reads three resident
    /// weights — the `gate`/`up` projections (`[k, ffn]`) and the `down`
    /// projection (`[ffn, rows]`) — over the shared burst activation. The gated
    /// activation and the `ffn_mid` stay inside the kernel, fed straight to the
    /// `down`, so no intermediate round-trips. `first` stages `x` exactly like
    /// `matVecResidentBurstAsync`. Returns null when the device is unavailable
    /// or rejects the dispatch.
    pub fn matVecMlpResidentBurstAsync(self: *Device, res_gate: *Resident, res_up: *Resident, res_down: *Resident, quant: Quant, rows: usize, k: usize, ffn: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.vk_pipe) |*pipe| {
            return pipe.submitResidentMlpShared(res_gate, res_up, res_down, quant, rows, k, ffn, x, act, variant, first);
        }
        return null;
    }

    /// Submit the whole SwiGLU MLP as **two multi-workgroup kernels on one
    /// device**: the fused `gate_up` (gate/up + gated activation, `gu_quant`)
    /// writes the `ffn_mid` into a device-local buffer, then the `down` GEMV
    /// (`down_quant`, which may differ from `gu_quant`) reads it. The
    /// intermediate never leaves VRAM, and both kernels keep full workgroup
    /// parallelism (unlike the single-workgroup fused shader). `first` stages `x`
    /// exactly like `matVecResidentBurstAsync`. Returns null when the device is
    /// unavailable or rejects the dispatch.
    pub fn matVecMlpPairResidentBurstAsync(self: *Device, res_gate: *Resident, res_up: *Resident, res_down: *Resident, gu_quant: Quant, down_quant: Quant, rows: usize, k: usize, ffn: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.vk_pipe) |*pipe| {
            return pipe.submitResidentMlpPairShared(res_gate, res_up, res_down, gu_quant, down_quant, rows, k, ffn, x, act, variant, first);
        }
        return null;
    }

    /// Wait an async handle and copy its `rows`-f32 result into `out`.
    pub fn waitResident(self: *Device, h: Handle, out: []f32, rows: usize) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.waitHandle(h, out, rows);
        }
        return false;
    }

    /// Submit a run of resident GEMVs that all read the same `x` and drain them
    /// with one fence wait (the decode q/k/v overlap). Returns false when any
    /// item is rejected, so the caller can fall back to the per-op path.
    pub fn matVecResidentBatch(self: *Device, k: usize, x: []const f32, ops: []const ResidentOp) bool {
        if (self.vk_pipe) |*pipe| {
            return pipe.dispatchMatVecResidentBatch(k, x, ops);
        }
        return false;
    }

    /// Upload `bytes` into a persistent GPU buffer, returned as an owned
    /// `Resident` handle (free with `freeResident`). Returns null when the
    /// device is unavailable or the allocation fails.
    pub fn uploadResident(self: *Device, bytes: []const u8) ?*Resident {
        const pipe = if (self.vk_pipe) |*p| p else return null;
        const res = self.allocator.create(Resident) catch return null;
        if (!pipe.initResident(self.allocator, res, bytes)) {
            self.allocator.destroy(res);
            return null;
        }
        return res;
    }

    /// Release a resident weight buffer and its handle.
    pub fn freeResident(self: *Device, res: *Resident) void {
        if (self.vk_pipe) |*pipe| pipe.deinitResident(self.allocator, res);
        self.allocator.destroy(res);
    }

    /// Block until all submitted GPU work has completed.
    pub fn sync(self: *Device) void {
        if (self.vk_pipe) |*pipe| pipe.sync();
    }

    /// Measure representative co-processor throughput at `tune --gpu` time.
    /// Delegates to the contract's generic probe (`gpu/backend/root.zig`),
    /// which is built on this device's ops.
    pub fn probeGpu(self: *Device, allocator: std.mem.Allocator, rows_in: usize, k_in: usize, m_in: usize) Probe {
        return contract.probeGpu(@This(), self, allocator, rows_in, k_in, m_in);
    }
};

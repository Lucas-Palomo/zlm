//! The gpu-facing API the runtime uses.
//!
//! Three layers: this file (the stable surface + the backend selection), the
//! backend contract (`backend/root.zig`, the shared types + the ops), and the
//! backend implementation (`backend/vulkan/`, selected at build time — a
//! comptime choice, no runtime plugin). A new backend is a new
//! `backend/<name>/` implementing the contract; the names below and the
//! runtime stay untouched.

const contract = @import("backend/root.zig");
const grid = @import("kernels/grid.zig");

/// The selected backend. Build-time selection: Vulkan today (the one backend
/// that exists); a future macOS Metal backend would be selected here by
/// `builtin.os.tag`. `contract.check` asserts it implements the contract.
pub const backend = @import("backend/vulkan/root.zig");

comptime {
    contract.check(backend);
}

/// Quantized weight layouts the GPU GEMV supports.
pub const Quant = contract.Quant;
/// Co-processor GEMV shader variant recorded per dispatch (`Dispatch.variant`).
pub const GpuVariant = grid.GpuVariant;
/// Gated activation for the fused `gate_up` GEMV (`gate_up_fused`).
pub const GateUpAct = contract.GateUpAct;
/// A handle to an asynchronously submitted resident dispatch.
pub const Handle = contract.Handle;
/// A persistent GPU weight buffer handle.
pub const Resident = contract.Resident;
/// One resident GEMV in a batch submit.
pub const ResidentOp = contract.ResidentOp;
/// In-flight resident dispatch depth of the backend ring.
pub const RingSize = contract.RingSize;
/// A measured co-processor throughput probe.
pub const Probe = contract.Probe;

/// The backend's device: a valid, inert object when no usable GPU exists.
pub const Device = backend.Device;

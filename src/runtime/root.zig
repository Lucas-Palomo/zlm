const plan = @import("../plan/root.zig");

/// Runtime layer: owns the `AOTKernel` (plan) and all session state; exposes
/// only a session-level API to the edges (CLI/HTTP).
pub const Session = @import("session.zig").Session;
pub const Executor = @import("executor.zig").Executor;
pub const AOTKernel = plan.AOTKernel;

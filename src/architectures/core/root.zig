const std = @import("std");
pub const contract = @import("contract.zig");
pub const Arch = contract.Arch;
pub const TensorRef = contract.TensorRef;
pub const TensorRole = contract.TensorRole;

const gemma4 = @import("../impl/gemma4/root.zig");
const qwen3 = @import("../impl/qwen3/root.zig");

/// Loaded architecture config (union so the registry stays uniform).
pub const Config = union(enum) {
    gemma4: gemma4.GemmaConfig,
    qwen3: qwen3.Qwen3Config,

    pub fn deinit(self: *Config) void {
        switch (self.*) {
            inline else => |*c| c.deinit(),
        }
    }
};

/// The registered architectures (dev-time catalog).
pub const archs = [_]Arch{
    gemma4.arch,
    qwen3.arch,
};

pub fn find(arch: []const u8) ?Arch {
    for (archs) |a| {
        if (a.matches(arch)) return a;
    }
    return null;
}

pub fn isSupported(arch: []const u8) bool {
    return find(arch) != null;
}

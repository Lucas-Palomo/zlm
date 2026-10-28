const std = @import("std");
const api = @import("api.zig");

test "erased decodes a valid id and rejects an unknown one" {
    const Args = struct { x: f32 = 0 };
    const Id = enum(u16) { only };
    const S = struct {
        fn run(id: Id, args: *const Args) api.KernelError!void {
            _ = id;
            _ = args;
        }
    };
    const f = api.erased(Id, Args, S.run);
    var args = Args{};
    try f(0, &args);
    try std.testing.expectError(error.Unsupported, f(1, &args));
}

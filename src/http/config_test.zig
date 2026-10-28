const std = @import("std");
const config = @import("config.zig");

test "config parse copies strings out of the read buffer" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const json =
        \\{"version":1,"host":"127.0.0.1","default":"m","models":[{"id":"m","runtime":"m","soul":"ok"}]}
    ;
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = json });

    const path = try std.fmt.allocPrint(
        testing.allocator,
        ".zig-cache/tmp/{s}/config.json",
        .{tmp.sub_path},
    );
    defer testing.allocator.free(path);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const cfg = try config.parse(arena.allocator(), testing.io, path);

    var scratch: [1 << 20]u8 = undefined;
    @memset(&scratch, 'x');

    try testing.expectEqualStrings("m", cfg.models[0].id);
    try testing.expectEqualStrings("ok", cfg.models[0].soul);
    _ = &scratch;
}

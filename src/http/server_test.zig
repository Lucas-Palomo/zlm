const std = @import("std");
const server = @import("server.zig");

test "the SSE stream terminates with a finish_reason chunk" {
    const testing = std.testing;
    var aw = std.Io.Writer.Allocating.init(testing.allocator);
    defer aw.deinit();
    try server.writeSseFinish(&aw.writer, "stop");
    const frame = try aw.toOwnedSlice();
    defer testing.allocator.free(frame);

    try testing.expect(std.mem.startsWith(u8, frame, "data: "));
    try testing.expect(std.mem.endsWith(u8, frame, "\n\n"));

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const chunk = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        frame["data: ".len .. frame.len - 2],
        .{},
    );
    const choices = chunk.object.get("choices").?.array;
    try testing.expectEqual(@as(usize, 1), choices.items.len);
    const choice = choices.items[0].object;
    try testing.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try testing.expect(choice.get("delta").? == .object);
}

test "the terminal frame reports a tool call" {
    const testing = std.testing;
    var aw = std.Io.Writer.Allocating.init(testing.allocator);
    defer aw.deinit();
    try server.writeSseFinish(&aw.writer, "tool_calls");
    const frame = aw.written();
    try testing.expect(std.mem.indexOf(u8, frame, "\"finish_reason\":\"tool_calls\"") != null);
}

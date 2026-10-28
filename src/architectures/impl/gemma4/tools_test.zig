const std = @import("std");
const tools = @import("../../core/tools.zig");
const gemma = @import("tools.zig");

test "declarations follow the gemma tool DSL" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const params = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"where\"}},\"required\":[\"path\"]}",
        .{},
    );
    const out = try gemma.renderDeclarations(testing.allocator, &.{.{
        .name = "read",
        .description = "Read a file",
        .parameters = params,
    }});
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<|tool>declaration:read{description:<|\"|>Read a file<|\"|>,parameters:{properties:{path:{description:<|\"|>where<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>",
        out,
    );
}

test "calls and responses round-trip through the DSL" {
    const testing = std.testing;
    const call = try gemma.renderCall(testing.allocator, .{ .name = "read", .arguments = "{\"path\":\"a.zig\",\"n\":2}" });
    defer testing.allocator.free(call);
    try testing.expectEqualStrings("<|tool_call>call:read{n:2,path:<|\"|>a.zig<|\"|>}<tool_call|>", call);

    const res = try gemma.renderResult(testing.allocator, "read", "line 1");
    defer testing.allocator.free(res);
    try testing.expectEqualStrings("<|tool_response>response:read{value:<|\"|>line 1<|\"|>}<tool_response|>", res);
}

test "parseCalls extracts multiple calls and keeps the surrounding text" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const text = "Checking.\n<|tool_call>call:read{path:<|\"|>a.zig<|\"|>,n:2}<tool_call|>\n<|tool_call>call:grep{pattern:<|\"|>x<|\"|>}<tool_call|>";
    const parsed = try gemma.parseCalls(arena.allocator(), text);
    try testing.expectEqualStrings("Checking.", parsed.content);
    try testing.expectEqual(@as(usize, 2), parsed.calls.len);
    try testing.expectEqualStrings("read", parsed.calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"a.zig\",\"n\":2}", parsed.calls[0].arguments);
    try testing.expectEqualStrings("grep", parsed.calls[1].name);
    try testing.expectEqualStrings("{\"pattern\":\"x\"}", parsed.calls[1].arguments);
}

test "parseCalls accepts JSON-shaped arguments" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try gemma.parseCalls(arena.allocator(), "<|tool_call>call:run{\"command\":\"ls -la\",\"timeout\":5}<tool_call|>");
    try testing.expectEqualStrings("", parsed.content);
    try testing.expectEqual(@as(usize, 1), parsed.calls.len);
    try testing.expectEqualStrings("{\"command\":\"ls -la\",\"timeout\":5}", parsed.calls[0].arguments);
}

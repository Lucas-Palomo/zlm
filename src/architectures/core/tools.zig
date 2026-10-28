const std = @import("std");

/// A tool (function) declaration from the client, in the OpenAI shape:
/// `{"type":"function","function":{"name","description","parameters"}}`.
pub const Tool = struct {
    name: []const u8,
    description: []const u8 = "",
    /// The JSON Schema object (`function.parameters`), borrowed from the
    /// request arena.
    parameters: ?std.json.Value = null,
};

/// One tool call in an OpenAI conversation turn (`message.tool_calls[]`).
pub const Call = struct {
    id: []const u8 = "",
    name: []const u8,
    /// The arguments as a JSON object text (OpenAI `function.arguments`).
    arguments: []const u8 = "{}",
};

/// One tool result (`role: "tool"` message).
pub const Result = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    content: []const u8,
};

/// A model reply split into its user-visible content and the tool calls it
/// emitted. `content` is allocated (possibly empty); `calls` is allocated and
/// owned by the caller.
pub const Parsed = struct {
    content: []const u8 = "",
    calls: []Call = &.{},
};

/// The architecture's tool-call prompt dialect. Architectures declare the
/// literal syntax (Gemma 4's `<|tool_call>` DSL, for example); the runtime and
/// the edge only move structured calls in and out.
pub const Dialect = struct {
    renderDeclarations: *const fn (allocator: std.mem.Allocator, tools: []const Tool) anyerror![]u8,
    renderCall: *const fn (allocator: std.mem.Allocator, call: Call) anyerror![]u8,
    renderResult: *const fn (allocator: std.mem.Allocator, name: []const u8, content: []const u8) anyerror![]u8,
    parseCalls: *const fn (allocator: std.mem.Allocator, text: []const u8) anyerror!Parsed,
};

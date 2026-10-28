const std = @import("std");

/// The shared single-turn prompt shape. Architectures supply their own literal
/// prefix/suffix strings; the runtime fills in the user text. The edge never
/// builds a prompt (CONSTITUTION "an edge talks only to the runtime").
pub const Template = struct {
    /// Opening of the user turn (e.g. `"<start_of_turn>user\n"`).
    user_prefix: []const u8 = "",
    /// Closing of the user turn (e.g. `"<end_of_turn>\n"`).
    user_suffix: []const u8 = "",
    /// Opening of the model turn that the model must continue.
    model_prefix: []const u8 = "",
    /// Opening/closing of the `--soul` (system) turn. Empty when the architecture
    /// has no native system role; the soul is then skipped.
    soul_prefix: []const u8 = "",
    soul_suffix: []const u8 = "",
    /// Emitted at the top of the system turn (before the soul) when thinking is
    /// ON (gemma-4: `<|think|>\n`). Empty when the arch has no think mode.
    think_prelude: []const u8 = "",
    /// Emitted after `model_prefix` when thinking is OFF (Qwen3: the pre-closed
    /// `" thinking\n\n response\n\n"`). Empty when not needed.
    think_prefill: []const u8 = "",
    /// Emitted inside an open model turn to start another reasoning round
    /// (gemma-4: `<|channel>thought\n`), e.g. after tool responses.
    think_round: []const u8 = "",

    /// Wrap one trimmed user turn and open the model reply (thinking off).
    pub fn apply(self: Template, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        return self.applyWith(allocator, text, false);
    }

    /// Wrap one trimmed user turn and open the model reply, emitting the think
    /// decorations the architecture declared for the requested mode.
    pub fn applyWith(self: Template, allocator: std.mem.Allocator, text: []const u8, think: bool) ![]u8 {
        return self.applyWithSoul(allocator, text, think, "");
    }

    /// Like `applyWith`, but prepend a `--soul` (system) turn when the
    /// architecture declares a system role and `soul` is non-empty, or when
    /// thinking is ON. The think token is emitted at the top of that same
    /// system turn (the Gemma-4 template shape: `<|turn>system\n<|think|>\n…
    /// <turn|>`). A template with no `soul_prefix`/`soul_suffix` has no native
    /// system role; it emits a bare `think_prelude` before the user turn.
    pub fn applyWithSoul(self: Template, allocator: std.mem.Allocator, text: []const u8, think: bool, soul: []const u8) ![]u8 {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        const has_system = self.soul_prefix.len > 0 and (soul.len > 0 or (think and self.think_prelude.len > 0));
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);
        if (has_system) {
            try buf.appendSlice(allocator, self.soul_prefix);
            if (think) try buf.appendSlice(allocator, self.think_prelude);
            try buf.appendSlice(allocator, soul);
            try buf.appendSlice(allocator, self.soul_suffix);
        } else if (think) {
            try buf.appendSlice(allocator, self.think_prelude);
        }
        try buf.appendSlice(allocator, self.user_prefix);
        try buf.appendSlice(allocator, trimmed);
        try buf.appendSlice(allocator, self.user_suffix);
        try buf.appendSlice(allocator, self.model_prefix);
        if (!think) try buf.appendSlice(allocator, self.think_prefill);
        return buf.toOwnedSlice(allocator);
    }

    pub fn isEmpty(self: Template) bool {
        return self.user_prefix.len == 0 and self.user_suffix.len == 0 and self.model_prefix.len == 0;
    }

    /// Whether this template declares any think decoration.
    pub fn hasThink(self: Template) bool {
        return self.think_prelude.len > 0 or self.think_prefill.len > 0;
    }
};

test "template wraps a single user turn" {
    const testing = std.testing;
    const t = Template{
        .user_prefix = "<start_of_turn>user\n",
        .user_suffix = "<end_of_turn>\n",
        .model_prefix = "<start_of_turn>model\n",
    };
    const out = try t.apply(testing.allocator, "  Hi  ");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        out,
    );
}

test "applyWith emits think decorations only for the matching mode" {
    const testing = std.testing;
    const t = Template{
        .user_prefix = "<u>",
        .user_suffix = "</u>",
        .model_prefix = "<m>",
        .think_prelude = "<sys><|think|>\n",
        .think_prefill = " thinking\n\n</think>\n\n",
    };

    const on = try t.applyWith(testing.allocator, " Hi ", true);
    defer testing.allocator.free(on);
    try testing.expectEqualStrings("<sys><|think|>\n<u>Hi</u><m>", on);

    const off = try t.applyWith(testing.allocator, " Hi ", false);
    defer testing.allocator.free(off);
    try testing.expectEqualStrings("<u>Hi</u><m> thinking\n\n</think>\n\n", off);

    try testing.expect(t.hasThink());
}

test "applyWith on a neutral template ignores think" {
    const testing = std.testing;
    const t = Template{ .user_prefix = "<u>", .model_prefix = "<m>" };
    try testing.expect(!t.hasThink());

    const on = try t.applyWith(testing.allocator, "Hi", true);
    defer testing.allocator.free(on);
    const off = try t.applyWith(testing.allocator, "Hi", false);
    defer testing.allocator.free(off);
    try testing.expectEqualStrings("<u>Hi<m>", on);
    try testing.expectEqualStrings(on, off);
}

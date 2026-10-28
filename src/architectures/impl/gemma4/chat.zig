const std = @import("std");
const chat = @import("../../core/chat.zig");

/// gemma4's single-turn prompt. The think ON variant injects the system turn
/// that carries `<|think|>` (matching the unsloth `<|turn>` template); think
/// OFF emits the template default. BOS is added by the tokenizer when
/// `tokenizer.ggml.add_bos_token` says so.
pub const template = chat.Template{
    .user_prefix = "<start_of_turn>user\n",
    .user_suffix = "<end_of_turn>\n",
    .model_prefix = "<start_of_turn>model\n",
    .soul_prefix = "<|turn>system\n",
    .soul_suffix = "<turn|>\n",
    .think_prelude = "<|think|>\n",
    .think_round = "<|channel>thought\n",
};

test "gemma4 template matches the single-turn wrap" {
    const testing = std.testing;
    const out = try template.apply(testing.allocator, "Hi");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        out,
    );
}

test "gemma4 think off is byte-identical, think on prepends the system turn" {
    const testing = std.testing;
    const off = try template.applyWith(testing.allocator, "Hi", false);
    defer testing.allocator.free(off);
    try testing.expectEqualStrings(
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        off,
    );

    const on = try template.applyWith(testing.allocator, "Hi", true);
    defer testing.allocator.free(on);
    try testing.expectEqualStrings(
        "<|turn>system\n<|think|>\n<turn|>\n<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        on,
    );
    try testing.expect(template.hasThink());
}

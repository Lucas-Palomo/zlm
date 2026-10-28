const std = @import("std");
const chat = @import("../../core/chat.zig");

/// qwen3's single-turn ChatML prompt. `think` OFF pre-closes the reasoning
/// block (`think_prefill`); `think` ON lets the model open ` thinking` itself.
/// `tokenizer.ggml.add_bos_token = false`, so no BOS is prepended.
pub const template = chat.Template{
    .user_prefix = "<|im_start|>user\n",
    .user_suffix = "<|im_end|>\n",
    .model_prefix = "<|im_start|>assistant\n",
    .soul_prefix = "<|im_start|>system\n",
    .soul_suffix = "<|im_end|>\n",
    .think_prefill = " thinking\n\n</think>\n\n",
};

test "qwen3 template wraps ChatML" {
    const testing = std.testing;
    const out = try template.applyWith(testing.allocator, "Hi", true);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        out,
    );
}

test "qwen3 think off pre-closes the reasoning block" {
    const testing = std.testing;
    const out = try template.applyWith(testing.allocator, "Hi", false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n",
        out,
    );
}

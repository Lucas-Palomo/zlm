const command = @import("../command.zig");
const compile = @import("image/compile.zig");
const verify = @import("image/verify.zig");
const remove = @import("image/remove.zig");
const list = @import("image/list.zig");
const info = @import("image/info.zig");

pub const summary = "Compile and manage model images";
pub const usage = "zlm image <subcommand>";
pub const help_text =
    \\An image is a compiled, self-contained model store: the model's tensors
    \\(ZTEF), manifest (zman) and tokenizer (zmeta), extracted from a GGUF.
    \\Images are immutable; execution state lives elsewhere.
    \\
;

pub const subcommands = [_]command.Command{
    .{
        .name = "compile",
        .summary = "Compile a GGUF model into an image",
        .usage = "zlm image compile <name> <model.gguf>",
        .help = compile.help_text,
        .run = compile.run,
    },
    .{
        .name = "verify",
        .summary = "Verify an image against its source GGUF",
        .usage = "zlm image verify <name> <model.gguf>",
        .help = verify.help_text,
        .run = verify.run,
    },
    .{
        .name = "remove",
        .summary = "Remove an image",
        .usage = "zlm image remove <name>",
        .help = remove.help_text,
        .run = remove.run,
    },
    .{
        .name = "list",
        .summary = "List compiled images",
        .usage = "zlm image list",
        .help = list.help_text,
        .run = list.run,
    },
    .{
        .name = "info",
        .summary = "Show details about an image",
        .usage = "zlm image info <name>",
        .help = info.help_text,
        .run = info.run,
    },
};

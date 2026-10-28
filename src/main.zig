const std = @import("std");
const cli = @import("cli/root.zig");
const store = @import("image/store.zig");
const io_utils = @import("utils/io.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    const home = store.resolveHome(init.environ_map) orelse {
        io_utils.eprintln(io, "cannot resolve store home: set ZLM_HOME or HOME", .{});
        std.process.exit(1);
    };
    store.configure(home);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.skip();

    var args_buf: [64][]const u8 = undefined;
    var argc: usize = 0;
    while (it.next()) |arg| {
        if (argc >= args_buf.len) break;
        args_buf[argc] = arg;
        argc += 1;
    }

    std.process.exit(cli.run(allocator, io, args_buf[0..argc]));
}

test {
    _ = @import("utils/io.zig");
    _ = @import("utils/mmap.zig");
    _ = @import("utils/binary_reader.zig");
    _ = @import("utils/binary_writer.zig");
    _ = @import("utils/cpu.zig");
    _ = @import("gguf/format.zig");
    _ = @import("gguf/reader.zig");
    _ = @import("gguf/root.zig");
    _ = @import("image/store.zig");
    _ = @import("image/ztef.zig");
    _ = @import("image/zman.zig");
    _ = @import("image/zrun.zig");
    _ = @import("image/zhash.zig");
    _ = @import("image/zmeta.zig");
    _ = @import("image/zmeta_reader.zig");
    _ = @import("image/zcom.zig");
    _ = @import("plan/root.zig");
    _ = @import("runtime/root.zig");
    _ = @import("runtime/prefix_cache.zig");
    _ = @import("runtime/zpc.zig");
    _ = @import("runtime/session.zig");
    _ = @import("runtime/sampler.zig");
    _ = @import("runtime/pool.zig");
    _ = @import("runtime/service.zig");
    _ = @import("runtime/residency.zig");
    _ = @import("runtime/prefetch.zig");
    _ = @import("runtime/executor.zig");
    _ = @import("kernels/test.zig");
    _ = @import("data/tensor_view.zig");
    _ = @import("gpu/test.zig");
    _ = @import("architectures/root.zig");
    _ = @import("tuner/compile.zig");
    _ = @import("tuner/composer.zig");
    _ = @import("tuner/bench.zig");
    _ = @import("tuner/calibrate.zig");
    _ = @import("tuner/caps.zig");
    _ = @import("tuner/cost.zig");
    _ = @import("tuner/tuner.zig");
    _ = @import("tuner/gpu.zig");
    _ = @import("architectures/impl/gemma4/root.zig");
    _ = @import("architectures/core/root.zig");
    _ = @import("architectures/impl/gemma4/arch.zig");
    _ = @import("architectures/core/graph.zig");
    _ = @import("architectures/impl/gemma4/block.zig");
    _ = @import("architectures/impl/gemma4/tools_test.zig");
    _ = @import("cli/args.zig");
    _ = @import("cli/options.zig");
    _ = @import("cli/command.zig");
    _ = @import("cli/usage.zig");
    _ = @import("cli/root.zig");
    _ = @import("cli/commands/help.zig");
    _ = @import("cli/commands/image.zig");
    _ = @import("cli/commands/image/compile.zig");
    _ = @import("cli/commands/image/verify.zig");
    _ = @import("cli/commands/image/remove.zig");
    _ = @import("cli/commands/image/list.zig");
    _ = @import("cli/commands/image/info.zig");
    _ = @import("cli/commands/list.zig");
    _ = @import("cli/commands/inspect.zig");
    _ = @import("cli/commands/unload.zig");
    _ = @import("cli/commands/tune.zig");
    _ = @import("cli/commands/load.zig");
    _ = @import("cli/commands/run.zig");
    _ = @import("cli/commands/serve.zig");
    _ = @import("http/server_test.zig");
    _ = @import("http/config_test.zig");
}

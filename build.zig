const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_model = .native,
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    // A build fingerprint: a hash of the sources (and build config) computed at
    // configure time and injected as a comptime constant. It changes on any code
    // change, so the runtime can detect a daemon left over from a previous build
    // without a manual version to bump.
    const opts = b.addOptions();
    opts.addOption(u64, "build_fingerprint", sourceFingerprint(b, target, optimize));

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addOptions("build_options", opts);

    const exe = b.addExecutable(.{
        .name = "zlm",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.setEnvironmentVariable("ZLM_HOME", b.pathFromRoot(".zlm"));
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the zlm CLI");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

/// Deterministic sha256 (first 8 bytes) over every `src/**` file plus
/// `build.zig`, with the target/optimize folded in. Pure configure-time work;
/// the result is baked into the binary as `build_options.build_fingerprint`.
fn sourceFingerprint(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) u64 {
    const io = b.graph.io;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(target.result.cpu.model.name);
    h.update(@tagName(target.result.os.tag));
    h.update(@tagName(optimize));

    hashFile(b, io, &h, "build.zig");

    var dir = std.Io.Dir.cwd().openDir(io, "src", .{ .iterate = true }) catch return finish(&h);
    defer dir.close(io);
    var walker = dir.walk(b.allocator) catch return finish(&h);
    defer walker.deinit();

    var paths: std.ArrayList([]const u8) = .empty;
    defer {
        for (paths.items) |p| b.allocator.free(p);
        paths.deinit(b.allocator);
    }
    while (walker.next(io) catch null) |e| {
        if (e.kind != .file) continue;
        const p = b.allocator.dupe(u8, e.path) catch continue;
        paths.append(b.allocator, p) catch b.allocator.free(p);
    }
    std.mem.sort([]const u8, paths.items, {}, lessThan);

    for (paths.items) |p| {
        const full = std.fmt.allocPrint(b.allocator, "src/{s}", .{p}) catch continue;
        defer b.allocator.free(full);
        h.update(p);
        hashFile(b, io, &h, full);
    }
    return finish(&h);
}

fn hashFile(b: *std.Build, io: std.Io, h: *std.crypto.hash.sha2.Sha256, path: []const u8) void {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, b.allocator, .limited(64 << 20)) catch return;
    defer b.allocator.free(data);
    h.update(data);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn finish(h: *std.crypto.hash.sha2.Sha256) u64 {
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.mem.readInt(u64, digest[0..8], .little);
}

const std = @import("std");
const Io = std.Io;
const zrun = @import("zrun.zig");

pub const DEFAULT_BASE = ".zlm";

var base_dir: []const u8 = DEFAULT_BASE;
var home_buf: [std.fs.max_path_bytes]u8 = undefined;

/// Store root: `$ZLM_HOME`, else `$HOME/.zlm`.
pub fn resolveHome(env: *std.process.Environ.Map) ?[]const u8 {
    if (env.get("ZLM_HOME")) |h| {
        if (h.len > 0) return h;
    }
    const home = env.get("HOME") orelse return null;
    if (home.len == 0) return null;
    return std.fmt.bufPrint(&home_buf, "{s}/.zlm", .{home}) catch null;
}

/// Store root: `$ZLM_HOME` when set, else the project's `.zlm`.
pub fn configure(root: []const u8) void {
    base_dir = if (root.len > 0) root else DEFAULT_BASE;
}

pub fn base() []const u8 {
    return base_dir;
}

pub fn imagesDir(buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/images", .{base_dir});
}

test "paths follow the configured root" {
    const testing = std.testing;
    configure("/tmp/zlmroot");
    defer configure(DEFAULT_BASE);
    try testing.expectEqualStrings("/tmp/zlmroot", base());
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("/tmp/zlmroot/images/x", try imageDir(&buf, "x"));
    try testing.expectEqualStrings("/tmp/zlmroot/images/.tmp-x", try tempDir(&buf, "x"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/x_ab.zplan", try runtimePlanPath(&buf, "x", "ab"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/deadbeef.zat", try deviceZatPath(&buf, "deadbeef"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/manifest.zrun", try registryPath(&buf));
    try testing.expectEqualStrings("x_ab", planKeyFromPlanName("x_ab.zplan"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/x_ab.socket", try serviceSocketPath(&buf, "x_ab"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/x_ab.pid", try servicePidPath(&buf, "x_ab"));
    try testing.expectEqualStrings("/tmp/zlmroot/runtimes/x_ab.log", try serviceLogPath(&buf, "x_ab"));
}

test "plan refcount follows shared artifacts" {
    const testing = std.testing;
    const entries = [_]zrun.Entry{
        .{ .id = "a", .plan = "img_ab.zplan", .tune = "img_ab.zat", .image = "img", .fp = 1 },
        .{ .id = "b", .plan = "img_ab.zplan", .tune = "img_ab.zat", .image = "img", .fp = 1 },
        .{ .id = "c", .plan = "img_cd.zplan", .tune = "img_cd.zat", .image = "img", .fp = 2 },
    };
    try testing.expect(planInEntries(&entries, "img_ab.zplan"));
    try testing.expect(planInEntries(&entries, "img_cd.zplan"));
    try testing.expect(!planInEntries(&entries, "img_ef.zplan"));
    try testing.expect(!planInEntries(&entries, "img_ab.zat"));
}

test "swapIn preserves the old image when the install fails" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var images = try tmp.dir.createDirPathOpen(testing.io, "images", .{});
    defer images.close(testing.io);

    var old = try images.createDirPathOpen(testing.io, "keep", .{});
    const f = try old.createFile(testing.io, "old.bin", .{});
    try f.writeStreamingAll(testing.io, "old");
    f.close(testing.io);
    old.close(testing.io);

    // No `.tmp-keep` to install: the move fails, so the old image must return.
    try testing.expectError(error.FileNotFound, swapIn(testing.io, images, ".tmp-keep", "keep", ".bak-keep"));
    const st = try images.statFile(testing.io, "keep/old.bin", .{});
    try testing.expectEqual(@as(u64, 3), st.size);
    try testing.expectError(error.FileNotFound, images.statFile(testing.io, ".bak-keep", .{}));
}

test "swapIn replaces the old image and drops the backup" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var images = try tmp.dir.createDirPathOpen(testing.io, "images", .{});
    defer images.close(testing.io);

    var old = try images.createDirPathOpen(testing.io, "keep", .{});
    const old_f = try old.createFile(testing.io, "old.bin", .{});
    old_f.close(testing.io);
    old.close(testing.io);

    var temp = try images.createDirPathOpen(testing.io, ".tmp-keep", .{});
    const new_f = try temp.createFile(testing.io, "new.bin", .{});
    new_f.close(testing.io);
    temp.close(testing.io);

    try swapIn(testing.io, images, ".tmp-keep", "keep", ".bak-keep");
    _ = try images.statFile(testing.io, "keep/new.bin", .{});
    try testing.expectError(error.FileNotFound, images.statFile(testing.io, "keep/old.bin", .{}));
    try testing.expectError(error.FileNotFound, images.statFile(testing.io, ".bak-keep", .{}));
    try testing.expectError(error.FileNotFound, images.statFile(testing.io, ".tmp-keep", .{}));
}

pub const MAX_NAME = 128;

pub fn isValidName(name: []const u8) bool {
    if (name.len == 0 or name.len > MAX_NAME) return false;
    if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    for (name) |c| {
        if (c == '/' or c == '\\' or c == 0) return false;
    }
    return true;
}

pub fn imageDir(buf: []u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/images/{s}", .{ base_dir, name });
}

pub fn tempDir(buf: []u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/images/.tmp-{s}", .{ base_dir, name });
}

pub fn ensureBase(io: Io) !void {
    var buf: [512]u8 = undefined;
    const images = try imagesDir(&buf);
    _ = Io.Dir.cwd().createDirPathOpen(io, images, .{}) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

pub fn exists(io: Io, name: []const u8) bool {
    var buf: [256]u8 = undefined;
    const path = imageDir(&buf, name) catch return false;
    _ = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return true;
}

pub fn pathExists(io: Io, path: []const u8) bool {
    _ = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return true;
}

pub fn remove(io: Io, name: []const u8) !void {
    var buf: [256]u8 = undefined;
    const path = try imageDir(&buf, name);
    try Io.Dir.cwd().deleteTree(io, path);
}

/// Install the image built at the temporary `.tmp-<name>` directory as `<name>`,
/// replacing any existing image. The previous image is moved aside to
/// `.bak-<name>` until the new one is in place, so a failed install cannot lose
/// it (see `swapIn`).
pub fn install(io: Io, name: []const u8) !void {
    var path_buf: [512]u8 = undefined;
    const images = try imagesDir(&path_buf);
    var dir = try Io.Dir.cwd().openDir(io, images, .{});
    defer dir.close(io);

    var temp_buf: [256]u8 = undefined;
    const temp_name = try std.fmt.bufPrint(&temp_buf, ".tmp-{s}", .{name});
    var backup_buf: [256]u8 = undefined;
    const backup_name = try std.fmt.bufPrint(&backup_buf, ".bak-{s}", .{name});
    try swapIn(io, dir, temp_name, name, backup_name);
}

/// Move `temp_name` into `final_name` inside `dir`. When `final_name` already
/// exists it is renamed to `backup_name` first and deleted only after the move
/// succeeds; if the move fails, the backup is restored. Renames stay within
/// `dir`, so the swap is atomic.
pub fn swapIn(io: Io, dir: Io.Dir, temp_name: []const u8, final_name: []const u8, backup_name: []const u8) !void {
    const had_old = blk: {
        _ = dir.statFile(io, final_name, .{}) catch break :blk false;
        break :blk true;
    };
    if (had_old) {
        dir.deleteTree(io, backup_name) catch {};
        try Io.Dir.rename(dir, final_name, dir, backup_name, io);
    }

    Io.Dir.rename(dir, temp_name, dir, final_name, io) catch |err| {
        if (had_old) Io.Dir.rename(dir, backup_name, dir, final_name, io) catch {};
        return err;
    };

    if (had_old) dir.deleteTree(io, backup_name) catch {};
}

/// Delete the image's prefix-cache files (`<image>_*.zpc`) under `runtimes/`.
/// A `.zpc` is tied to the image, so removing the image reclaims it (otherwise
/// it would linger until a later `unload --prune`).
pub fn removeImageCaches(io: Io, image: []const u8) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    const path = runtimesDir(&buf) catch return;
    var dir = Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var victims: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zpc")) continue;
        if (entry.name.len <= image.len + 1) continue;
        if (!std.mem.startsWith(u8, entry.name, image)) continue;
        if (entry.name[image.len] != '_') continue;
        victims.append(a, a.dupe(u8, entry.name) catch continue) catch continue;
    }
    for (victims.items) |n| dir.deleteFile(io, n) catch {};
}

pub fn openImage(io: Io, name: []const u8) !Io.Dir {
    var buf: [256]u8 = undefined;
    const path = try imageDir(&buf, name);
    return Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
}

pub fn runtimesDir(buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes", .{base_dir});
}

pub fn registryPath(buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}", .{ base_dir, zrun.NAME });
}

pub fn runtimePlanPath(buf: []u8, image_name: []const u8, fp_hex: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}_{s}.zplan", .{ base_dir, image_name, fp_hex });
}

/// Prefix-cache artifact (`tune --prefix-cache`), keyed by the K/V fingerprint so
/// runtimes of the image that differ only in `--threads`/`--ram`/`--vram` share it.
pub fn runtimePrefixCachePath(buf: []u8, image_name: []const u8, fp_hex: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}_{s}.zpc", .{ base_dir, image_name, fp_hex });
}

/// Plan key `<image>_<fp>` derived from a shared plan artifact filename.
pub fn planKeyFromPlanName(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".zplan")) return name[0 .. name.len - ".zplan".len];
    return name;
}

pub fn serviceSocketPath(buf: []u8, key: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}.socket", .{ base_dir, key });
}

pub fn servicePidPath(buf: []u8, key: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}.pid", .{ base_dir, key });
}

pub fn serviceLogPath(buf: []u8, key: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}.log", .{ base_dir, key });
}

/// Device calibration profile: shared by all images on the same device.
pub fn deviceZatPath(buf: []u8, digest_hex: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/runtimes/{s}.zat", .{ base_dir, digest_hex });
}

pub fn ensureRuntimesDir(io: Io) !void {
    var buf: [512]u8 = undefined;
    const path = try runtimesDir(&buf);
    _ = Io.Dir.cwd().createDirPathOpen(io, path, .{}) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

pub fn openRuntimes(io: Io) !Io.Dir {
    var buf: [512]u8 = undefined;
    const path = try runtimesDir(&buf);
    return Io.Dir.cwd().openDir(io, path, .{});
}

/// Chat sessions: one `<uuid>.json` per conversation, under `sessions/`.
pub fn sessionsDir(buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/sessions", .{base_dir});
}

pub fn sessionPath(buf: []u8, id: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/sessions/{s}.json", .{ base_dir, id });
}

pub fn ensureSessionsDir(io: Io) !void {
    var buf: [512]u8 = undefined;
    const path = try sessionsDir(&buf);
    _ = Io.Dir.cwd().createDirPathOpen(io, path, .{}) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

pub fn openSessions(io: Io) !Io.Dir {
    var buf: [512]u8 = undefined;
    const path = try sessionsDir(&buf);
    return Io.Dir.cwd().openDir(io, path, .{});
}

/// Load `manifest.zrun` into `arena`, or null when the store/registry is
/// absent. The returned registry owns its strings via `arena`.
pub fn loadRegistry(arena: std.mem.Allocator, io: Io) ?zrun.Registry {
    var buf: [512]u8 = undefined;
    const path = runtimesDir(&buf) catch return null;
    var dir = Io.Dir.cwd().openDir(io, path, .{}) catch return null;
    defer dir.close(io);
    return zrun.load(arena, dir, io);
}

/// Resolve an alias to the full path of its shared plan artifact, or null when
/// the registry or the alias is absent. The returned bytes live in `buf`.
pub fn resolvePlan(buf: []u8, io: Io, alias: []const u8) ?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var reg = loadRegistry(arena.allocator(), io) orelse return null;
    defer reg.deinit();
    const e = reg.find(alias) orelse return null;
    var dbuf: [512]u8 = undefined;
    const dir = runtimesDir(&dbuf) catch return null;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, e.plan }) catch null;
}

/// Whether the registry carries a row for `alias`.
pub fn aliasExists(io: Io, alias: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var reg = loadRegistry(arena.allocator(), io) orelse return false;
    defer reg.deinit();
    return reg.find(alias) != null;
}

/// A resolved runtime alias: the shared plan artifact and its per-plan key.
pub const Runtime = struct {
    alias: []const u8,
    image: []const u8,
    plan: []const u8,
    key: []const u8,
    /// Device calibration artifact filename (`<digest>.zat`) this plan was
    /// tuned against.
    tune: []const u8,

    pub fn planPath(self: Runtime, buf: []u8) ![]const u8 {
        var dbuf: [512]u8 = undefined;
        const dir = try runtimesDir(&dbuf);
        return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, self.plan });
    }

    pub fn tunePath(self: Runtime, buf: []u8) ![]const u8 {
        var dbuf: [512]u8 = undefined;
        const dir = try runtimesDir(&dbuf);
        return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, self.tune });
    }
};

/// Resolve an alias to its runtime record. Strings live in `arena`.
pub fn resolveRuntime(arena: std.mem.Allocator, io: Io, alias: []const u8) ?Runtime {
    var reg = loadRegistry(arena, io) orelse return null;
    const e = reg.find(alias) orelse return null;
    return .{
        .alias = e.id,
        .image = e.image,
        .plan = e.plan,
        .key = planKeyFromPlanName(e.plan),
        .tune = e.tune,
    };
}

/// Whether any registry row still references the shared `plan` artifact.
pub fn planHasAliases(arena: std.mem.Allocator, io: Io, plan: []const u8) bool {
    var reg = loadRegistry(arena, io) orelse return false;
    defer reg.deinit();
    return planInEntries(reg.entries, plan);
}

/// Whether `entries` carries a row for the shared `plan` artifact.
pub fn planInEntries(entries: []const zrun.Entry, plan: []const u8) bool {
    for (entries) |e| {
        if (std.mem.eql(u8, e.plan, plan)) return true;
    }
    return false;
}

/// Rewrite `manifest.zrun` atomically from `entries`.
pub fn writeRegistry(allocator: std.mem.Allocator, io: Io, entries: []const zrun.Entry) !void {
    try ensureRuntimesDir(io);
    var buf: [512]u8 = undefined;
    const path = try runtimesDir(&buf);
    var dir = try Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    try zrun.write(allocator, dir, io, entries);
}

/// Drop the registry row for `alias`, rewriting the registry. Returns whether
/// the alias existed.
pub fn removeAlias(allocator: std.mem.Allocator, io: Io, alias: []const u8) !bool {
    var buf: [512]u8 = undefined;
    const path = runtimesDir(&buf) catch return false;
    var dir = Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    defer dir.close(io);
    return zrun.remove(allocator, dir, io, alias);
}

/// Delete `.zplan`/`.zat` files under `runtimes/` referenced by no registry
/// entry. Returns the number of files removed.
pub fn pruneOrphans(allocator: std.mem.Allocator, io: Io) !usize {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const entries: []const zrun.Entry = if (loadRegistry(arena.allocator(), io)) |reg| reg.entries else &[_]zrun.Entry{};
    // `.zpc` files belong to the image, so they are pruned only when the image
    // is gone — not when it merely has no runtime.
    const images: []const []const u8 = list(arena.allocator(), io) catch &.{};

    var buf: [512]u8 = undefined;
    const path = runtimesDir(&buf) catch return 0;
    var dir = Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer dir.close(io);
    return zrun.pruneOrphans(allocator, dir, io, entries, images);
}

/// Names of existing images.
pub fn list(allocator: std.mem.Allocator, io: Io) ![][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    var buf: [512]u8 = undefined;
    const images = try imagesDir(&buf);
    var dir = Io.Dir.cwd().openDir(io, images, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return names.toOwnedSlice(allocator),
        else => return err,
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .directory) continue;
        if (std.mem.startsWith(u8, entry.name, ".tmp-")) continue;
        if (std.mem.startsWith(u8, entry.name, ".bak-")) continue;
        const copy = try allocator.dupe(u8, entry.name);
        try names.append(allocator, copy);
    }
    return names.toOwnedSlice(allocator);
}

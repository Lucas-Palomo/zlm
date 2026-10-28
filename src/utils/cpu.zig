const std = @import("std");
const builtin = @import("builtin");

const cpu_set_words = @typeInfo(std.os.linux.cpu_set_t).array.len;
const max_cpus = cpu_set_words * @bitSizeOf(usize);

pub fn digest() u64 {
    var h = std.hash.Wyhash.init(0x5A4C5A43505521);
    h.update(@tagName(builtin.cpu.arch));
    h.update(builtin.cpu.model.name);
    h.update(builtin.cpu.features.asBytes());
    return h.final();
}

/// A worker -> CPU mapping measured from the machine's topology: up to `threads`
/// distinct CPU ids from the process's allowed set, preferring one CPU per
/// physical core before adding SMT siblings, and capped by the cgroup CPU quota.
/// Empty off Linux.
pub fn affinityPlan(allocator: std.mem.Allocator, threads_in: u16) ![]u16 {
    if (builtin.os.tag != .linux) {
        return allocator.alloc(u16, 0);
    } else {
        return affinityPlanLinux(allocator, threads_in);
    }
}

/// Pin the calling thread to `cpu` with `sched_setaffinity`. No-op off Linux.
pub fn pinCurrentThread(cpu: u16) void {
    if (builtin.os.tag != .linux) {
        return;
    } else {
        var set: std.os.linux.cpu_set_t = [_]usize{0} ** cpu_set_words;
        const bits = @bitSizeOf(usize);
        const word = cpu / bits;
        if (word >= cpu_set_words) return;
        set[word] = @as(usize, 1) << @intCast(cpu % bits);
        std.os.linux.sched_setaffinity(0, &set) catch {};
    }
}

/// Snapshot the calling thread's allowed-CPU mask, or null off Linux. Pair with
/// `restoreAffinity` to undo a temporary `pinCurrentThread` instead of leaving
/// the process pinned for the rest of the run.
pub fn currentAffinity() ?std.os.linux.cpu_set_t {
    return allowedCpus();
}

/// Restore a mask captured by `currentAffinity`. No-op off Linux.
pub fn restoreAffinity(set: *const std.os.linux.cpu_set_t) void {
    if (builtin.os.tag != .linux) {
        return;
    } else {
        std.os.linux.sched_setaffinity(0, set) catch {};
    }
}

fn affinityPlanLinux(allocator: std.mem.Allocator, threads_in: u16) ![]u16 {
    const threads: usize = @max(@as(usize, 1), @as(usize, threads_in));

    const set = allowedCpus() orelse return fallbackPlan(allocator, threads);

    var candidates: [max_cpus]u16 = undefined;
    var cand_count: usize = 0;
    for (0..max_cpus) |cpu| {
        if (!isCpuSet(&set, cpu)) continue;
        candidates[cand_count] = @intCast(cpu);
        cand_count += 1;
    }
    if (cand_count == 0) return fallbackPlan(allocator, threads);

    var group_keys: [max_cpus]u64 = undefined;
    var group_of: [max_cpus]usize = undefined;
    var occurrence: [max_cpus]usize = undefined;
    var group_count: usize = 0;
    for (candidates[0..cand_count], 0..) |cpu, i| {
        const key = topologyKey(cpu);
        var found: ?usize = null;
        for (group_keys[0..group_count], 0..) |k, g| {
            if (k == key) {
                found = g;
                break;
            }
        }
        const g = found orelse blk: {
            group_keys[group_count] = key;
            group_count += 1;
            break :blk group_count - 1;
        };
        group_of[i] = g;
    }
    for (0..cand_count) |i| {
        var o: usize = 0;
        for (0..i) |j| {
            if (group_of[j] == group_of[i]) o += 1;
        }
        occurrence[i] = o;
    }

    var limit = @min(threads, cand_count);
    if (quotaCpus()) |q| limit = @min(limit, q);
    if (limit == 0) limit = 1;

    const out = try allocator.alloc(u16, limit);
    var n: usize = 0;
    outer: for (0..cand_count) |pass| {
        for (0..cand_count) |i| {
            if (occurrence[i] != pass) continue;
            if (n >= limit) break :outer;
            out[n] = candidates[i];
            n += 1;
        }
    }
    return out;
}

/// Degraded plan when affinity or topology is unavailable: cpu ids `0..n`.
fn fallbackPlan(allocator: std.mem.Allocator, threads: usize) ![]u16 {
    const n = @min(threads, max_cpus);
    const out = try allocator.alloc(u16, n);
    for (out, 0..) |*c, i| c.* = @intCast(i);
    return out;
}

fn allowedCpus() ?std.os.linux.cpu_set_t {
    var set: std.os.linux.cpu_set_t = [_]usize{0} ** cpu_set_words;
    const rc = std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &set);
    if (@as(isize, @bitCast(rc)) < 0) return null;
    return set;
}

fn isCpuSet(set: *const std.os.linux.cpu_set_t, cpu: usize) bool {
    const bits = @bitSizeOf(usize);
    const word = cpu / bits;
    if (word >= cpu_set_words) return false;
    return (set[word] & (@as(usize, 1) << @intCast(cpu % bits))) != 0;
}

fn cpuCount(set: *const std.os.linux.cpu_set_t) usize {
    var n: usize = 0;
    for (set.*) |w| n += @popCount(w);
    return n;
}

/// Physical-core key for a CPU: `(package_id << 32) | core_id`. When the core
/// id is unreadable the CPU gets a unique high-bit key so no false siblings form.
fn topologyKey(cpu: u16) u64 {
    const core = readTopologyInt(cpu, "core_id") orelse
        return 0x8000_0000_0000_0000 | @as(u64, cpu);
    const package: u64 = if (readTopologyInt(cpu, "physical_package_id")) |p|
        @intCast(p)
    else
        0;
    return (package << 32) | @as(u64, @intCast(core));
}

fn readTopologyInt(cpu: u16, name: []const u8) ?i32 {
    var path_buf: [128]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/{s}", .{ cpu, name }) catch return null;
    var data_buf: [64]u8 = undefined;
    const data = readFileSmall(path, &data_buf) orelse return null;
    const trimmed = std.mem.trim(u8, data, " \t\r\n");
    return std.fmt.parseInt(i32, trimmed, 10) catch null;
}

/// Distinct physical cores among the online CPUs, from Linux sysfs topology.
/// Null off Linux or when the online set is unreadable. CPUs whose `core_id` is
/// unreadable count as separate cores, so the result never undercounts.
pub fn physicalCoreCount() ?u32 {
    if (builtin.os.tag != .linux) {
        return null;
    } else {
        var buf: [256]u8 = undefined;
        const online = readFileSmall("/sys/devices/system/cpu/online", &buf) orelse return null;
        var keys: [max_cpus]u64 = undefined;
        var n: usize = 0;
        var it = std.mem.tokenizeAny(u8, online, ", \t\r\n");
        while (it.next()) |tok| {
            var lo: usize = 0;
            var hi: usize = 0;
            if (std.mem.indexOfScalar(u8, tok, '-')) |dash| {
                lo = std.fmt.parseInt(usize, tok[0..dash], 10) catch continue;
                hi = std.fmt.parseInt(usize, tok[dash + 1 ..], 10) catch continue;
            } else {
                lo = std.fmt.parseInt(usize, tok, 10) catch continue;
                hi = lo;
            }
            if (lo >= max_cpus) continue;
            if (hi >= max_cpus) hi = max_cpus - 1;
            var cpu = lo;
            while (cpu <= hi) : (cpu += 1) {
                const key = topologyKey(@intCast(cpu));
                var seen = false;
                for (keys[0..n]) |k| {
                    if (k == key) {
                        seen = true;
                        break;
                    }
                }
                if (!seen and n < keys.len) {
                    keys[n] = key;
                    n += 1;
                }
            }
        }
        if (n == 0) return null;
        return @intCast(n);
    }
}

/// CPU quota in cores from cgroup v2 (`cpu.max`) or v1 (`cpu.cfs_quota_us` +
/// `cpu.cfs_period_us`). Null when unset, unlimited, or unreadable.
pub fn quotaCpus() ?usize {
    var buf: [64]u8 = undefined;
    if (readFileSmall("/sys/fs/cgroup/cpu.max", &buf)) |data| {
        var it = std.mem.tokenizeAny(u8, data, " \t\r\n");
        const quota_s = it.next() orelse return null;
        if (std.mem.eql(u8, quota_s, "max")) return null;
        const quota = std.fmt.parseInt(u64, quota_s, 10) catch return null;
        const period = if (it.next()) |p| (std.fmt.parseInt(u64, p, 10) catch return null) else 100000;
        if (period == 0) return null;
        return @intCast(@max(@as(u64, 1), (quota + period - 1) / period));
    }
    if (readFileSmall("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &buf)) |data| {
        const quota = std.fmt.parseInt(i64, std.mem.trim(u8, data, " \t\r\n"), 10) catch return null;
        if (quota <= 0) return null;
        var period_buf: [64]u8 = undefined;
        const period = if (readFileSmall("/sys/fs/cgroup/cpu/cpu.cfs_period_us", &period_buf)) |pd|
            (std.fmt.parseInt(u64, std.mem.trim(u8, pd, " \t\r\n"), 10) catch return null)
        else
            return null;
        if (period == 0) return null;
        return @intCast(@max(@as(u64, 1), (@as(u64, @intCast(quota)) + period - 1) / period));
    }
    return null;
}

/// Read a small pseudo-file into `buf` (caller-owned). Null off Linux or on any
/// open/read error; the returned slice points into `buf`.
pub fn readFileSmall(path: []const u8, buf: []u8) ?[]const u8 {
    if (builtin.os.tag != .linux) {
        return null;
    } else {
        const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{ .CLOEXEC = true }, 0) catch return null;
        defer _ = std.os.linux.close(@intCast(fd));
        const n = std.posix.read(fd, buf) catch return null;
        return buf[0..n];
    }
}

test "affinityPlan returns distinct ids bounded by threads and allowed cpus" {
    const testing = std.testing;
    const got = try affinityPlan(testing.allocator, 4);
    defer testing.allocator.free(got);
    if (builtin.os.tag != .linux) {
        try testing.expectEqual(@as(usize, 0), got.len);
        return;
    }
    try testing.expect(got.len <= 4);
    if (allowedCpus()) |set| {
        try testing.expect(got.len <= cpuCount(&set));
    }
    for (got, 0..) |c, i| {
        for (got[i + 1 ..]) |d| try testing.expect(c != d);
    }
}

test "affinityPlan never exceeds the requested thread count" {
    const testing = std.testing;
    const got = try affinityPlan(testing.allocator, 1);
    defer testing.allocator.free(got);
    try testing.expect(got.len <= 1);
}

const std = @import("std");
const posix = std.posix;
const Io = std.Io;

pub const MappedFile = struct {
    data: []align(std.heap.page_size_min) u8,
    size: u64,
    /// The file descriptor the mapping was created from. Kept open so a caller
    /// (the residency arena's `pread` workers) can read from the file without
    /// reopening it; closed by `close`.
    handle: std.posix.fd_t,

    pub fn open(io: Io, path: []const u8) !MappedFile {
        return openIn(Io.Dir.cwd(), io, path);
    }

    pub fn openIn(dir: Io.Dir, io: Io, sub_path: []const u8) !MappedFile {
        const file = try dir.openFile(io, sub_path, .{});
        errdefer _ = std.os.linux.close(file.handle);

        const file_size = try file.length(io);
        const mapped = try posix.mmap(
            null,
            @intCast(file_size),
            posix.PROT{ .READ = true },
            posix.MAP{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        return .{ .data = mapped, .size = file_size, .handle = file.handle };
    }

    pub fn adviseWillNeed(self: MappedFile) void {
        _ = posix.madvise(self.data.ptr, self.data.len, posix.MADV.WILLNEED) catch {};
    }

    pub fn close(self: MappedFile) void {
        posix.munmap(self.data);
        _ = std.os.linux.close(self.handle);
    }
};

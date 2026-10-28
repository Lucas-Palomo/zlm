const std = @import("std");
const Zmeta = @import("../image/zmeta_reader.zig").Zmeta;
const format = @import("../gguf/format.zig");

/// A directed pair of vocab pieces; a BPE merge rule keys on `(a, b)`.
const Pair = struct { a: []const u8, b: []const u8 };

const PairContext = struct {
    pub fn hash(_: PairContext, p: Pair) u64 {
        var h = std.hash.Wyhash.init(0x4250_454D_4552_4745);
        h.update(p.a);
        h.update(&[_]u8{0});
        h.update(p.b);
        return h.final();
    }

    pub fn eql(_: PairContext, x: Pair, y: Pair) bool {
        return std.mem.eql(u8, x.a, y.a) and std.mem.eql(u8, x.b, y.b);
    }
};

const PairMap = std.HashMap(Pair, u32, PairContext, std.hash_map.default_max_load_percentage);

/// A piece that must never be split by BPE (registered special or control type).
const Special = struct { piece: []const u8, id: u32 };

pub const Error = error{
    MissingVocab,
    MissingMerges,
    MissingBos,
    InvalidTokenId,
    OutOfMemory,
};

/// How pieces map to text: SentencePiece (`▁` for space, `<0xXX>` fallback) or
/// GPT-2 byte-level (a byte-to-unicode map; Qwen/`tokenizer.ggml.model = gpt2`).
pub const VocabStyle = enum { sentencepiece, byte_level };

pub const EncodeOptions = struct {
    add_bos: bool = false,
    add_eos: bool = false,
};

/// The fields the tokenizer needs, decoupled from `zmeta` so it is testable
/// with synthetic vocab/merges. `tokens`/`merges`/`types` are copied (the
/// underlying string bytes are borrowed, not owned).
pub const Loaded = struct {
    tokens: []const []const u8,
    types: []const i32 = &.{},
    merges: []const []const u8,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    unk_id: ?u32 = null,
    pad_id: ?u32 = null,
    mask_id: ?u32 = null,
    add_bos: bool = false,
    add_space_prefix: bool = false,
    style: VocabStyle = .sentencepiece,
};

/// SentencePiece-BPE tokenizer (`tokenizer.ggml.model` == "gemma"/"gemma4"):
/// normalize spaces to `▁`, greedy BPE by merge rank, `<0xXX>` byte fallback.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    tokens: []const []const u8,
    types: []const i32,
    merges: []const []const u8,
    specials: []const Special,
    token_to_id: std.StringHashMap(u32),
    merge_ranks: PairMap,
    bos_id: ?u32,
    eos_id: ?u32,
    unk_id: ?u32,
    pad_id: ?u32,
    mask_id: ?u32,
    add_bos: bool,
    add_space_prefix: bool,
    style: VocabStyle,
    /// Token ids that open/close the reasoning block, resolved from the vocab
    /// (`<|channel>`/`<channel|>` for gemma4, ` thinking`/`</think>` for qwen3).
    /// Null on a model with no think turn; the output path then shows everything.
    think_open_id: ?u32 = null,
    think_close_id: ?u32 = null,
    /// Tool-call markers kept by `decodeToolText` so the runtime can parse the
    /// model's native call block (gemma4: `<|tool_call>` / `<tool_call|>` and
    /// the `<|"|>` string delimiter).
    tool_call_open_id: ?u32 = null,
    tool_call_close_id: ?u32 = null,
    tool_string_id: ?u32 = null,

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.types);
        self.allocator.free(self.merges);
        self.allocator.free(self.specials);
        self.token_to_id.deinit();
        self.merge_ranks.deinit();
    }

    /// Read the tokenizer from the image metadata. The GGUF tokenizer keys are
    /// re-encoded into `zmeta` at compile, so this is the runtime's only source.
    pub fn fromZmeta(allocator: std.mem.Allocator, z: *const Zmeta) Error!Tokenizer {
        const tok_arr = z.getArray("tokenizer.ggml.tokens") orelse return error.MissingVocab;
        if (tok_arr.elem != .string) return error.MissingVocab;
        const tokens = try allocator.alloc([]const u8, tok_arr.items.len);
        defer allocator.free(tokens);
        for (tok_arr.items, 0..) |item, i| tokens[i] = switch (item) {
            .string => |s| s,
            else => return error.MissingVocab,
        };

        const merge_arr = z.getArray("tokenizer.ggml.merges") orelse return error.MissingMerges;
        if (merge_arr.elem != .string) return error.MissingMerges;
        const merges = try allocator.alloc([]const u8, merge_arr.items.len);
        defer allocator.free(merges);
        for (merge_arr.items, 0..) |item, i| merges[i] = switch (item) {
            .string => |s| s,
            else => return error.MissingMerges,
        };

        var types_buf: []i32 = &.{};
        if (z.getArray("tokenizer.ggml.token_type")) |tt| {
            types_buf = try allocator.alloc(i32, tt.items.len);
            for (tt.items, types_buf) |item, *o| o.* = switch (item) {
                .uint8 => |x| x,
                .int8 => |x| x,
                .uint16 => |x| x,
                .int16 => |x| x,
                .uint32 => |x| @intCast(x),
                .int32 => |x| x,
                else => 1,
            };
        }
        defer if (types_buf.len > 0) allocator.free(types_buf);

        return init(allocator, .{
            .tokens = tokens,
            .types = types_buf,
            .merges = merges,
            .bos_id = if (z.getUint("tokenizer.ggml.bos_token_id")) |v| @intCast(v) else null,
            .eos_id = if (z.getUint("tokenizer.ggml.eos_token_id")) |v| @intCast(v) else null,
            .unk_id = if (z.getUint("tokenizer.ggml.unknown_token_id")) |v| @intCast(v) else null,
            .pad_id = if (z.getUint("tokenizer.ggml.padding_token_id")) |v| @intCast(v) else null,
            .mask_id = if (z.getUint("tokenizer.ggml.mask_token_id")) |v| @intCast(v) else null,
            .add_bos = z.getBool("tokenizer.ggml.add_bos_token") orelse false,
            .add_space_prefix = z.getBool("tokenizer.ggml.add_space_prefix") orelse false,
            .style = styleFromModel(z.getString("tokenizer.ggml.model")),
        });
    }

    pub fn init(allocator: std.mem.Allocator, l: Loaded) Error!Tokenizer {
        const tokens = try allocator.dupe([]const u8, l.tokens);
        errdefer allocator.free(tokens);
        const merges = try allocator.dupe([]const u8, l.merges);
        errdefer allocator.free(merges);
        const types = try allocator.dupe(i32, l.types);
        errdefer allocator.free(types);

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();
        try token_to_id.ensureTotalCapacity(@intCast(tokens.len));
        for (tokens, 0..) |t, id| try token_to_id.put(t, @intCast(id));

        var merge_ranks = PairMap.init(allocator);
        errdefer merge_ranks.deinit();
        try merge_ranks.ensureTotalCapacity(@intCast(merges.len));
        for (merges, 0..) |m, rank| {
            const sp = std.mem.indexOfScalar(u8, m, ' ') orelse continue;
            if (sp == 0 or sp + 1 >= m.len) continue;
            try merge_ranks.put(.{ .a = m[0..sp], .b = m[sp + 1 ..] }, @intCast(rank));
        }

        var self = Tokenizer{
            .allocator = allocator,
            .tokens = tokens,
            .types = types,
            .merges = merges,
            .specials = &.{},
            .token_to_id = token_to_id,
            .merge_ranks = merge_ranks,
            .bos_id = l.bos_id,
            .eos_id = l.eos_id,
            .unk_id = l.unk_id,
            .pad_id = l.pad_id,
            .mask_id = l.mask_id,
            .add_bos = l.add_bos,
            .add_space_prefix = l.add_space_prefix,
            .style = l.style,
        };
        self.specials = try self.collectSpecials(allocator);
        self.resolveThinkMarkers();
        return self;
    }

    /// Resolve the reasoning block's open/close token ids from the vocab. A
    /// model uses one pair; the rest are ignored.
    fn resolveThinkMarkers(self: *Tokenizer) void {
        const pairs = [_][2][]const u8{
            .{ "<|channel>", "<channel|>" },
            .{ " thinking", " response" },
        };
        for (pairs) |pair| {
            const o = self.token_to_id.get(pair[0]) orelse continue;
            const c = self.token_to_id.get(pair[1]) orelse continue;
            self.think_open_id = o;
            self.think_close_id = c;
            break;
        }
        self.tool_call_open_id = self.token_to_id.get("<|tool_call>");
        self.tool_call_close_id = self.token_to_id.get("<tool_call|>");
        self.tool_string_id = self.token_to_id.get("<|\"|>");
    }

    fn collectSpecials(self: *const Tokenizer, allocator: std.mem.Allocator) Error![]Special {
        var list: std.ArrayList(Special) = .empty;
        errdefer list.deinit(allocator);
        for (self.tokens, 0..) |tok, i| {
            if (!self.isSpecialType(@intCast(i)) and !self.isRegistered(@intCast(i))) continue;
            try list.append(allocator, .{ .piece = tok, .id = @intCast(i) });
        }
        return list.toOwnedSlice(allocator);
    }

    fn isSpecialType(self: *const Tokenizer, id: u32) bool {
        if (id >= self.types.len) return false;
        return switch (self.types[id]) {
            1, 6 => false,
            else => true,
        };
    }

    fn isRegistered(self: *const Tokenizer, id: u32) bool {
        inline for (.{ "bos_id", "eos_id", "unk_id", "pad_id", "mask_id" }) |field| {
            if (@field(self, field)) |v| {
                if (v == id) return true;
            }
        }
        return false;
    }

    fn piece(self: *const Tokenizer, id: u32) ?[]const u8 {
        if (id >= self.tokens.len) return null;
        return self.tokens[id];
    }

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) Error![]u32 {
        return self.encodeOptions(allocator, text, .{});
    }

    /// Encode with the model's own defaults (adds BOS when the image says so).
    pub fn encodeWithDefaults(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) Error![]u32 {
        return self.encodeOptions(allocator, text, .{ .add_bos = self.add_bos });
    }

    pub fn encodeOptions(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, opts: EncodeOptions) Error![]u32 {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var ids: std.ArrayList(u32) = .empty;
        if (opts.add_bos) {
            try ids.append(a, self.bos_id orelse return error.MissingBos);
        }
        try self.encodeSegments(a, text, &ids);
        if (opts.add_eos) {
            if (self.eos_id) |eos| try ids.append(a, eos);
        }
        return allocator.dupe(u32, ids.items);
    }

    fn encodeSegments(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, ids: *std.ArrayList(u32)) Error!void {
        var i: usize = 0;
        var seg_start: usize = 0;
        while (i < text.len) {
            var matched: ?Special = null;
            for (self.specials) |sp| {
                if (sp.piece.len == 0 or i + sp.piece.len > text.len) continue;
                if (!std.mem.eql(u8, text[i .. i + sp.piece.len], sp.piece)) continue;
                if (matched == null or sp.piece.len > matched.?.piece.len) matched = sp;
            }
            if (matched) |sp| {
                if (seg_start < i) try self.bpeAppend(a, text[seg_start..i], ids);
                try ids.append(a, sp.id);
                i += sp.piece.len;
                seg_start = i;
                continue;
            }
            i += 1;
        }
        if (seg_start < text.len) try self.bpeAppend(a, text[seg_start..], ids);
    }

    fn bpeAppend(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, ids: *std.ArrayList(u32)) Error!void {
        if (text.len == 0) return;
        var symbols: std.ArrayList([]const u8) = .empty;
        switch (self.style) {
            .sentencepiece => try self.splitCodepoints(a, text, &symbols),
            .byte_level => try self.splitByteLevel(a, text, &symbols),
        }

        while (symbols.items.len >= 2) {
            var best_rank: ?u32 = null;
            var best_idx: usize = 0;
            for (0..symbols.items.len - 1) |idx| {
                const p = Pair{ .a = symbols.items[idx], .b = symbols.items[idx + 1] };
                if (self.merge_ranks.get(p)) |r| {
                    if (best_rank == null or r < best_rank.?) {
                        best_rank = r;
                        best_idx = idx;
                    }
                }
            }
            if (best_rank == null) break;

            const best_a = symbols.items[best_idx];
            const best_b = symbols.items[best_idx + 1];
            var out: std.ArrayList([]const u8) = .empty;
            var idx: usize = 0;
            while (idx < symbols.items.len) {
                if (idx + 1 < symbols.items.len and
                    std.mem.eql(u8, symbols.items[idx], best_a) and
                    std.mem.eql(u8, symbols.items[idx + 1], best_b))
                {
                    const merged = try a.alloc(u8, best_a.len + best_b.len);
                    @memcpy(merged[0..best_a.len], best_a);
                    @memcpy(merged[best_a.len..], best_b);
                    try out.append(a, merged);
                    idx += 2;
                } else {
                    try out.append(a, symbols.items[idx]);
                    idx += 1;
                }
            }
            symbols = out;
        }

        for (symbols.items) |sym| {
            if (self.token_to_id.get(sym)) |id| {
                try ids.append(a, id);
            } else {
                for (sym) |b| try ids.append(a, try self.byteFallback(b));
            }
        }
    }

    /// Normalize spaces to `▁` (optionally with a leading dummy prefix) and
    /// split into UTF-8 codepoint symbols.
    fn splitCodepoints(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, symbols: *std.ArrayList([]const u8)) Error!void {
        var spaces: usize = 0;
        for (text) |c| {
            if (c == ' ') spaces += 1;
        }
        const prefix: usize = if (self.add_space_prefix) 3 else 0;
        const buf = try a.alloc(u8, prefix + text.len + spaces * 2);
        var bi: usize = 0;
        if (self.add_space_prefix) {
            buf[0] = 0xE2;
            buf[1] = 0x96;
            buf[2] = 0x81;
            bi = 3;
        }
        for (text) |c| {
            if (c == ' ') {
                buf[bi] = 0xE2;
                buf[bi + 1] = 0x96;
                buf[bi + 2] = 0x81;
                bi += 3;
            } else {
                buf[bi] = c;
                bi += 1;
            }
        }
        const normalized = buf[0..bi];
        var i: usize = 0;
        while (i < normalized.len) {
            const b = normalized[i];
            const len: usize = if (b < 0x80)
                1
            else if (b >> 5 == 0b110)
                2
            else if (b >> 4 == 0b1110)
                3
            else if (b >> 3 == 0b11110)
                4
            else
                1;
            const end = @min(i + len, normalized.len);
            try symbols.append(a, normalized[i..end]);
            i = end;
        }
    }

    /// GPT-2 byte-level split: map each input byte to its byte-to-unicode
    /// codepoint (UTF-8 encoded) so the shared BPE merge loop runs over the
    /// byte-level alphabet (then `pieceToText` inverse-maps on the way out).
    fn splitByteLevel(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, symbols: *std.ArrayList([]const u8)) Error!void {
        _ = self;
        for (text) |b| {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(byteToCodepoint(b), &buf) catch unreachable;
            const sym = try a.dupe(u8, buf[0..len]);
            try symbols.append(a, sym);
        }
    }

    fn byteFallback(self: *const Tokenizer, byte: u8) Error!u32 {
        var lower: [6]u8 = undefined;
        const l = std.fmt.bufPrint(&lower, "<0x{x:0>2}>", .{byte}) catch return error.InvalidTokenId;
        if (self.token_to_id.get(l)) |id| return id;
        var upper: [6]u8 = undefined;
        const u = std.fmt.bufPrint(&upper, "<0x{X:0>2}>", .{byte}) catch return error.InvalidTokenId;
        if (self.token_to_id.get(u)) |id| return id;
        return self.unk_id orelse error.InvalidTokenId;
    }

    /// Faithful detokenization of every piece (specials included).
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            const p = self.piece(id) orelse return error.InvalidTokenId;
            try self.pieceToText(allocator, p, &out);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Detokenize one piece according to the vocabulary style.
    fn pieceToText(self: *const Tokenizer, allocator: std.mem.Allocator, p: []const u8, out: *std.ArrayList(u8)) Error!void {
        switch (self.style) {
            .sentencepiece => try decodePiece(allocator, p, out),
            .byte_level => try decodePieceByteLevel(allocator, p, out),
        }
    }

    /// Detokenize for display: special/control pieces are dropped. A piece is
    /// dropped when the GGUF marks it control/special, when it is a registered
    /// id (bos/eos/unk/pad/mask), or when it is a known control marker that the
    /// GGUF may (incorrectly) tag `NORMAL` (e.g. `<end_of_turn>`). Encoding is
    /// untouched; only the displayed text changes.
    pub fn decodeText(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, show_think: bool) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        var in_think = false;
        for (ids) |id| {
            if (self.think_open_id) |o| if (id == o) {
                in_think = true;
                continue;
            };
            if (self.think_close_id) |c| if (id == c) {
                in_think = false;
                continue;
            };
            if (in_think and !show_think) continue;
            if (self.isSpecialType(id) or self.isRegistered(id)) continue;
            const p = self.piece(id) orelse continue;
            if (isControlMarker(p)) continue;
            try self.pieceToText(allocator, p, &out);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Like `decodeText`, but keeps the native tool-call markers so the
    /// dialect parser can see the model's `<|tool_call>…<tool_call|>` block.
    /// Thinking is still suppressed unless `show_think`.
    pub fn decodeToolText(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, show_think: bool, initial_in_think: bool) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        var in_think = initial_in_think;
        for (ids) |id| {
            if (self.think_open_id) |o| if (id == o) {
                in_think = true;
                continue;
            };
            if (self.think_close_id) |c| if (id == c) {
                in_think = false;
                continue;
            };
            if (in_think and !show_think) continue;
            if (self.isToolMarker(id)) {
                const p = self.piece(id) orelse continue;
                try out.appendSlice(allocator, p);
                continue;
            }
            if (self.isSpecialType(id) or self.isRegistered(id)) continue;
            const p = self.piece(id) orelse continue;
            if (isControlMarker(p)) continue;
            try self.pieceToText(allocator, p, &out);
        }
        return out.toOwnedSlice(allocator);
    }

    fn isToolMarker(self: *const Tokenizer, id: u32) bool {
        if (self.tool_call_open_id) |v| if (id == v) return true;
        if (self.tool_call_close_id) |v| if (id == v) return true;
        if (self.tool_string_id) |v| if (id == v) return true;
        return false;
    }

    /// Detokenize only the reasoning (think block) content, regardless of
    /// `show_think`: the printable text between the open and close think tokens.
    /// Diagnostic only (`--verbose=all`); it never reaches normal output.
    pub fn decodeThink(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        var in_think = false;
        for (ids) |id| {
            if (self.think_open_id) |o| if (id == o) {
                in_think = true;
                continue;
            };
            if (self.think_close_id) |c| if (id == c) {
                in_think = false;
                continue;
            };
            if (!in_think) continue;
            if (self.isSpecialType(id) or self.isRegistered(id)) continue;
            const p = self.piece(id) orelse continue;
            if (isControlMarker(p)) continue;
            try self.pieceToText(allocator, p, &out);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Streaming counterpart of `decodeText`: detokenize a single id into `out`
    /// with the same special/control filtering, letting the caller chunk the
    /// result across tokens as bytes arrive. `in_think` carries the reasoning
    /// state across tokens; while inside and `show_think` is false, the content
    /// is suppressed.
    pub fn decodeTokenAppend(self: *const Tokenizer, allocator: std.mem.Allocator, id: u32, out: *std.ArrayList(u8), in_think: *bool, show_think: bool) Error!void {
        if (self.think_open_id) |o| if (id == o) {
            in_think.* = true;
            return;
        };
        if (self.think_close_id) |c| if (id == c) {
            in_think.* = false;
            return;
        };
        if (in_think.* and !show_think) return;
        if (self.isSpecialType(id) or self.isRegistered(id)) return;
        const p = self.piece(id) orelse return;
        if (isControlMarker(p)) return;
        try self.pieceToText(allocator, p, out);
    }

    /// Chat control markers that must never reach display text. GGUF converters
    /// sometimes tag these `NORMAL` instead of `CONTROL`, so they are dropped by
    /// piece as well; ordinary `<...>` tokens (HTML, byte fallback) are not.
    fn isControlMarker(p: []const u8) bool {
        const markers = [_][]const u8{
            "<start_of_turn>", "<end_of_turn>",
            "<|turn>",         "<turn|>",
            "<eos>",           "<bos>",
            "<|eot_id|>",      "<|eom_id|>",
            "<|im_start|>",    "<|im_end|>",
            "<|endoftext|>",   "<|end_of_text|>",
        };
        for (markers) |m| {
            if (std.mem.eql(u8, p, m)) return true;
        }
        return false;
    }

    fn decodePiece(allocator: std.mem.Allocator, p: []const u8, out: *std.ArrayList(u8)) Error!void {
        var i: usize = 0;
        while (i < p.len) {
            if (i + 2 < p.len and p[i] == 0xE2 and p[i + 1] == 0x96 and p[i + 2] == 0x81) {
                try out.append(allocator, ' ');
                i += 3;
                continue;
            }
            if (i + 5 < p.len and p[i] == '<' and p[i + 1] == '0' and p[i + 2] == 'x' and hexVal(p[i + 3]) != null and hexVal(p[i + 4]) != null and p[i + 5] == '>') {
                try out.append(allocator, (hexVal(p[i + 3]).? << 4) | hexVal(p[i + 4]).?);
                i += 6;
                continue;
            }
            try out.append(allocator, p[i]);
            i += 1;
        }
    }

    fn hexVal(c: u8) ?u8 {
        return switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => null,
        };
    }

    pub fn eosId(self: *const Tokenizer) ?u32 {
        return self.eos_id;
    }

    pub fn bosId(self: *const Tokenizer) ?u32 {
        return self.bos_id;
    }
};

fn testTokenizer(allocator: std.mem.Allocator) !Tokenizer {
    const tokens = [_][]const u8{
        "<pad>",  "<eos>",         "<bos>",  "<unk>", "<mask>",
        "h",      "e",             "l",      "o",     "he",
        "hel",    "hell",          "hello",
        "▁",
        "w",      "r",             "d",
        "▁w",
        "▁wo",
        "▁wor",
        "▁worl",
        "▁world",
        "<0x41>", "<end_of_turn>", "<html>",
    };
    const types = [_]i32{ 3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1 };
    const merges = [_][]const u8{
        "h e", "he l", "hel l", "hell o",
        "▁ w",
        "▁w o",
        "▁wo r",
        "▁wor l",
        "▁worl d",
    };
    return Tokenizer.init(allocator, .{
        .tokens = &tokens,
        .types = &types,
        .merges = &merges,
        .bos_id = 2,
        .eos_id = 1,
        .unk_id = 3,
        .pad_id = 0,
        .mask_id = 4,
        .add_bos = true,
        .add_space_prefix = false,
    });
}

test "tokenizer encode/decode round-trip" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "hello");
    defer testing.allocator.free(ids);
    try testing.expect(ids.len > 0);

    const text = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello", text);
}

test "tokenizer space maps to the marker" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "hello world");
    defer testing.allocator.free(ids);
    const text = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello world", text);
}

test "tokenizer byte fallback and decode" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const ids = try tok.encode(testing.allocator, "A");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(@as(u32, 22), ids[0]);

    const text = try tok.decode(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("A", text);
}

test "tokenizer specials, bos default and decodeText" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const ids = try tok.encodeWithDefaults(testing.allocator, "hello");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(u32, 2), ids[0]);

    const text = try tok.decodeText(testing.allocator, ids, true);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello", text);
}

test "decodeTokenAppend matches decodeText token by token" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const hello = tok.token_to_id.get("hello").?;
    const eot = tok.token_to_id.get("<end_of_turn>").?;
    const html = tok.token_to_id.get("<html>").?;

    const ids = [_]u32{ 2, hello, eot, html, hello };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    var in_think = false;
    for (ids) |id| try tok.decodeTokenAppend(testing.allocator, id, &out, &in_think, true);

    const batch = try tok.decodeText(testing.allocator, &ids, true);
    defer testing.allocator.free(batch);
    try testing.expectEqualStrings(batch, out.items);
    try testing.expectEqualStrings("hello<html>hello", out.items);
}

test "think block is hidden by default and shown with show_think" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();
    // Synthesize a think pair in the vocab.
    tok.think_open_id = tok.token_to_id.get("<html>").?;
    tok.think_close_id = tok.token_to_id.get("<end_of_turn>").?;
    const open = tok.think_open_id.?;
    const close = tok.think_close_id.?;
    const hello = tok.token_to_id.get("hello").?;
    const ids = [_]u32{ hello, open, hello, close, hello };

    const hidden = try tok.decodeText(testing.allocator, &ids, false);
    defer testing.allocator.free(hidden);
    try testing.expectEqualStrings("hellohello", hidden);

    const shown = try tok.decodeText(testing.allocator, &ids, true);
    defer testing.allocator.free(shown);
    try testing.expectEqualStrings("hellohellohello", shown);

    // Streaming path agrees.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    var in_think = false;
    for (ids) |id| try tok.decodeTokenAppend(testing.allocator, id, &out, &in_think, false);
    try testing.expectEqualStrings("hellohello", out.items);
}

test "decodeThink returns only the reasoning content" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();
    tok.think_open_id = tok.token_to_id.get("<html>").?;
    tok.think_close_id = tok.token_to_id.get("<end_of_turn>").?;
    const open = tok.think_open_id.?;
    const close = tok.think_close_id.?;
    const hello = tok.token_to_id.get("hello").?;
    const ids = [_]u32{ hello, open, hello, hello, close, hello };

    const think = try tok.decodeThink(testing.allocator, &ids);
    defer testing.allocator.free(think);
    try testing.expectEqualStrings("hellohello", think);

    const none = try tok.decodeThink(testing.allocator, &[_]u32{hello});
    defer testing.allocator.free(none);
    try testing.expectEqualStrings("", none);
}

test "decodeText drops a mislabeled control marker, keeps ordinary pieces" {
    const testing = std.testing;
    var tok = try testTokenizer(testing.allocator);
    defer tok.deinit();

    const hello = tok.token_to_id.get("hello").?;
    const eot = tok.token_to_id.get("<end_of_turn>").?;
    const html = tok.token_to_id.get("<html>").?;

    const ids = [_]u32{ hello, eot, html, hello };
    const text = try tok.decodeText(testing.allocator, &ids, true);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("hello<html>hello", text);
}

/// The byte-level style from `tokenizer.ggml.model` (`gpt2` -> byte_level).
fn styleFromModel(model: ?[]const u8) VocabStyle {
    const m = model orelse return .sentencepiece;
    if (std.mem.eql(u8, m, "gpt2")) return .byte_level;
    return .sentencepiece;
}

/// GPT-2 byte-level identity range: bytes that map to themselves.
fn isByteLevelIdentity(b: u8) bool {
    return (b >= 0x21 and b <= 0x7e) or (b >= 0xa1 and b <= 0xac) or (b >= 0xae and b <= 0xff);
}

/// GPT-2 byte -> unicode codepoint (identity for the printable range; else
/// `256 + n` for the `n`th non-identity byte).
fn byteToCodepoint(b: u8) u21 {
    if (isByteLevelIdentity(b)) return @intCast(b);
    var n: u16 = 0;
    var x: u16 = 0;
    while (x <= b) : (x += 1) {
        const bx: u8 = @intCast(x);
        if (!isByteLevelIdentity(bx)) {
            if (bx == b) return @intCast(256 + n);
            n += 1;
        }
    }
    unreachable;
}

/// Inverse of `byteToCodepoint`; null when `cp` is not a byte-level codepoint.
fn codepointToByte(cp: u21) ?u8 {
    if (cp <= 0xff) {
        const b: u8 = @intCast(cp);
        if (isByteLevelIdentity(b)) return b;
        return null;
    }
    if (cp < 256) return null;
    const want = cp - 256;
    var n: u21 = 0;
    var x: u16 = 0;
    while (x <= 0xff) : (x += 1) {
        const bx: u8 = @intCast(x);
        if (!isByteLevelIdentity(bx)) {
            if (n == want) return bx;
            n += 1;
        }
    }
    return null;
}

fn nextUtf8Codepoint(s: []const u8, i: *usize) ?u21 {
    if (i.* >= s.len) return null;
    const len = std.unicode.utf8ByteSequenceLength(s[i.*]) catch return null;
    if (i.* + len > s.len) return null;
    const cp = std.unicode.utf8Decode(s[i.* .. i.* + len]) catch return null;
    i.* += len;
    return cp;
}

/// Detokenize a byte-level piece: each codepoint maps back to its byte; a
/// non-byte-level codepoint is emitted verbatim.
fn decodePieceByteLevel(allocator: std.mem.Allocator, piece: []const u8, out: *std.ArrayList(u8)) !void {
    var i: usize = 0;
    while (i < piece.len) {
        const before = i;
        const cp = nextUtf8Codepoint(piece, &i) orelse {
            try out.append(allocator, piece[before]);
            i = before + 1;
            continue;
        };
        if (codepointToByte(cp)) |b| {
            try out.append(allocator, b);
        } else {
            try out.appendSlice(allocator, piece[before..i]);
        }
    }
}

test "byte-level map round-trips all 256 bytes" {
    const testing = std.testing;
    var b: u16 = 0;
    while (b <= 0xff) : (b += 1) {
        const byte: u8 = @intCast(b);
        const cp = byteToCodepoint(byte);
        try testing.expectEqual(byte, codepointToByte(cp).?);
    }
}

test "byte-level piece decodes back to the original bytes" {
    const testing = std.testing;
    const text = "A b\n\t\xff";
    var piece: std.ArrayList(u8) = .empty;
    defer piece.deinit(testing.allocator);
    for (text) |byte| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(byteToCodepoint(byte), &buf) catch unreachable;
        try piece.appendSlice(testing.allocator, buf[0..n]);
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try decodePieceByteLevel(testing.allocator, piece.items, &out);
    try testing.expectEqualStrings(text, out.items);
}

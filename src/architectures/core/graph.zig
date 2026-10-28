const contract = @import("contract.zig");

/// Activation buffers the model uses; the scheduler resolves them to session
/// memory. A model declares which slots it needs and their sizes.
pub const Slot = enum {
    hidden,
    normed,
    q,
    k,
    v,
    attn_out,
    ffn_a,
    ffn_b,
    ffn_mid,
    logits,
    emb_row,
    ple_token,
    ple_projected,
    ple_combined,
    ple_a,
};

/// A semantic instruction: what to do, over which role (to find the tensor),
/// with which activation slots as inputs/outputs. Abstract — no offsets,
/// no atomic kernel yet.
pub const Instr = struct {
    op: Op,
    role: ?contract.TensorRole = null,
    /// Second weight role for a fused two-weight op (e.g. the `up` weight of
    /// `.gate_up`); null for single-weight ops.
    role2: ?contract.TensorRole = null,
    /// Third weight role for a fused three-weight op (e.g. the `down` weight of
    /// `.mlp`); null otherwise.
    role3: ?contract.TensorRole = null,
    layer: ?u32 = null,
    in: ?Slot = null,
    /// Second operand (e.g. the residual `add`'s second term).
    in2: ?Slot = null,
    out: ?Slot = null,
    /// Operation-specific parameters (eps, pos, scale, n_rot, ...). Kept as a
    /// tagged union so the scheduler/engine can switch on it.
    params: Params = .none,
    /// Part of the KV-store path: emitted only for layers that store a cache.
    /// The tuner drops it for shared-KV layers, so the runtime never sees it.
    store_only: bool = false,
    /// Execute this `norm` and, in the same pass, the `rope` that would
    /// otherwise follow on the same activation slot. The block leaves this
    /// false; the tuner's Composer sets it when it fuses a matching
    /// `norm`+`rope` pair (emitted as the `qk_norm_rope_heads` atomic). Keeping
    /// it false means the declared block is unchanged.
    fuse_rope: bool = false,
    /// Configurable presence: `.ple` runs only when the layer has per-layer
    /// embeddings, `.no_ple` only when it does not. `.always` ignores the arch
    /// dims. The tuner resolves `dims.ple` and drops the instruction.
    gate: Gate = .always,

    pub const Gate = enum { always, ple, no_ple };
};

pub const Op = enum {
    embd_lookup,
    dequant,
    norm,
    v_norm,
    matvec,
    /// Fused FFN: the `ffn_gate` matvec + `ffn_up` matvec + gated activation
    /// collapse into one instruction (`role` = gate, `role2` = up, `in` = the
    /// shared activation, `out` = the gated result). Emitted by the Composer.
    gate_up,
    /// Fused SwiGLU MLP: the whole FFN — `gate`/`up` + gated activation + the
    /// `down` matvec — into one instruction (`role` = gate, `role2` = up,
    /// `role3` = down, `in` = the block activation, `out` = the MLP output).
    /// The `ffn_mid` intermediate never leaves the kernel. The Composer leaves
    /// the pair separate; the tuner emits `.mlp` when the per-op cost model
    /// says the fusion pays (`tuner/gpu.zig`'s `fuseMlps`).
    mlp,
    rope,
    attention,
    activation,
    write_kv,
    lm_head,
    softcap,
    /// Residual add (`out = a + b`); optional `params.scale` scales b.
    add,
    /// Ungated `gelu(a)` in place (`out`).
    gelu,
    /// `out = a * b` (`params.mul.off` indexes `in2`).
    mul,
    /// `out = a * params.scale.s`.
    scale,
    /// `out = (a + b) * params.combine.s`.
    combine,
};

pub const Params = union(enum) {
    none,
    norm: struct { eps: f32, head_dim: u32 = 0, kind: u32 = norm_rms },
    rope: struct {
        pos: u32,
        n_rot: u32,
        freq_base: f32,
        /// Per-head dimension of the buffer rope acts on (heads are this wide).
        head_dim: u32 = 0,
        /// Number of heads in the buffer (q heads for q, kv heads for k).
        n_heads: u32 = 0,
    },
    attention: struct {
        pos: u32,
        scale: f32,
        sliding: bool,
        n_heads: u32 = 1,
        n_kv_heads: u32 = 1,
        head_dim: u32 = 0,
        t_start: u32 = 0,
        window: u32 = 0,
    },
    activation: struct { kind: u32 },
    softcap: struct { cap: f32 },
    /// `out = a + scale * b` (scale = 1 for a plain residual).
    add: struct { scale: f32 = 1.0 },
    /// Post-lookup embedding scale (`hidden *= scale`), resolved from config.
    embed: struct { scale: f32 },
    /// `out = a * b`; `off` is the element offset into the `in2` buffer.
    mul: struct { off: u32 = 0 },
    /// `out = a * s`.
    scale: struct { s: f32 },
    /// `out = (a + b) * s`.
    combine: struct { s: f32 },

    /// Stable numeric code for the plan (`Dispatch.sem_op`); the runtime
    /// compares `sem_op` against these without reading the tagged union.
    pub const softcap_code: u8 = 5;
    pub const embed_code: u8 = 7;
    pub const mul_code: u8 = 8;
    pub const scale_code: u8 = 9;
    pub const combine_code: u8 = 10;
    /// Activation kinds recorded on an `.activation` instruction.
    pub const activation_silu: u32 = 0;
    pub const activation_gelu: u32 = 1;
    /// Norm kinds recorded on a `.norm` instruction; the executor dispatches
    /// on the recorded kind instead of inferring it from tensor shapes.
    pub const norm_rms: u32 = 0;
    pub const norm_rms_no_weight: u32 = 1;
    pub const norm_heads: u32 = 2;
    pub const norm_heads_no_weight: u32 = 3;
    pub const norm_scale: u32 = 4;

    /// Stable numeric code for the plan (`Dispatch.sem_op`).
    pub fn code(self: Params) u8 {
        return switch (self) {
            .none => 0,
            .norm => 1,
            .rope => 2,
            .attention => 3,
            .activation => 4,
            .softcap => 5,
            .add => 6,
            .embed => 7,
            .mul => 8,
            .scale => 9,
            .combine => 10,
        };
    }
};

/// A model's forward, declared as a semantic instruction sequence.
pub const Block = struct {
    /// Activation slots the model needs (with byte sizes filled by the
    /// scheduler from config); declared here as an ordered set.
    slots: []const Slot,
    /// Global instructions (embed, final norm, lm_head).
    head: []const Instr,
    /// Per-layer instructions (run `block_count` times).
    layer: []const Instr,
    tail: []const Instr,
};

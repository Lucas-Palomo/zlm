const std = @import("std");
const types = @import("types.zig");
const context_mod = @import("context.zig");
const contract = @import("../root.zig");
const grid = @import("../../kernels/grid.zig");
const shaders = @import("../../kernels/shaders.zig");

/// Co-processor GEMV shader variant recorded per dispatch (for a `device == 1`
/// dispatch the plan's `Dispatch.variant` is one of these codes). `plain` is the
/// base shader; `u4`/`u8` are the unrolled variants. Defined in the GPU kernel
/// grid (`gpu/kernels/grid.zig`) so the grid and this backend share one enum;
/// this module is the AtomicOps' implementation, not a variant list.
pub const GpuVariant = grid.GpuVariant;

/// Quantized weight layouts the GPU GEMV supports, from the backend contract.
pub const Quant = contract.Quant;

/// The single-weight GEMV quant families, each shipping **one** spec-constant
/// shader; the pipeline table is indexed by `(family, variant)`.
const matvec_quants = [_]Quant{ .q4_0, .q4_1, .q8_0, .q6_k, .q4_k, .f32, .f16, .bf16 };
/// The fused two-weight `gate_up` families (both weights share the quant).
const gate_up_quants = [_]Quant{ .q4_0, .q4_1, .q8_0 };
/// The fused three-weight MLP families (all three weights share the quant).
const mlp_quants = [_]Quant{ .q4_0, .q4_1, .q8_0 };
/// The variant axes every GEMV family is specialized on (the grid's variant
/// set; `u8` is unused by the spec-constant shaders).
const shader_variants = [_]GpuVariant{ .plain, .u4, .sub, .un2, .un4, .w32, .w128 };

/// Largest MLP output width the fused shader's shared accumulator supports
/// (`MAX_OUT` in `matvec_mlp_*.comp`). A wider output is rejected so the
/// caller falls back to the CPU rather than reading past shared memory.
pub const max_mlp_out: usize = 8192;

/// The variant a `Quant` was specialized on, derived from the forward
/// `Quant.variant` map (the base quant returns `plain`).
fn variantOf(quant: Quant) GpuVariant {
    const base = quant.base();
    inline for (std.enums.values(GpuVariant)) |v| {
        if (base.variant(v) == quant) return v;
    }
    return .plain;
}

/// The `matvec_quants` index of a base quant, or null when it has no GEMV
/// shader.
fn quantIndex(base: Quant) ?usize {
    for (matvec_quants, 0..) |q, i| {
        if (q == base) return i;
    }
    return null;
}

/// Gated activation for the fused `gate_up` GEMV.
pub const GateUpAct = contract.GateUpAct;
/// Number of in-flight resident dispatches the ring supports.
pub const RingSize = contract.RingSize;
/// A handle to a submitted resident dispatch (the ring slot).
pub const Handle = contract.Handle;
/// One resident GEMV in a batch.
pub const ResidentOp = contract.ResidentOp;
/// Vulkan 1.1 core version: subgroup arithmetic (`subgroupAdd`) is core from
/// 1.1, so the subgroup-reduction pipeline is only created when the device
/// reports at least this (the prune before pipeline creation).
const vk_api_version_1_1: u32 = (1 << 22) | (1 << 12) | 0;

/// The gpu-facing resident handle the backend fills.
pub const Resident = contract.Resident;

pub const PushConsts = extern struct {
    num_rows: u32,
    num_blocks_per_row: u32,
    /// Workgroups dispatched along x; a GEMV workgroup's row is
    /// `group_id.y * groups_x + group_id.x`.
    groups_x: u32,
    /// Gated activation for the fused `gate_up` GEMV: `0` = silu, `1` =
    /// gelu_tanh (see `GateUpAct`). Ignored by every other shader.
    act: u32,
};

/// Push constants for the batched GEMM compute shaders. `num_tokens` is the
/// activation-row count M, `num_rows` the output-feature count N, and
/// `num_blocks_per_row` the quant blocks (K / block elems) of one weight row.
pub const PushConstsMatMul = extern struct {
    num_tokens: u32,
    num_rows: u32,
    num_blocks_per_row: u32,
};

/// Push constants for the fused MLP compute shader. `num_rows` is the output
/// width (the `down` rows), `num_blocks_per_row` the `gate`/`up` input blocks
/// (k / block elems), `ffn_rows` the number of `gate`/`up` rows (the `down`
/// input width) and `down_blocks_per_row` the `down` blocks (`ffn_rows / block
/// elems`). `act` selects the gated activation (see `GateUpAct`).
pub const PushConstsMlp = extern struct {
    num_rows: u32,
    num_blocks_per_row: u32,
    groups_x: u32,
    act: u32,
    ffn_rows: u32,
    down_blocks_per_row: u32,
    pad0: u32,
    pad1: u32,
};

/// GEMV workgroup grid: each workgroup owns `rows_per_group` output rows, whose
/// 64 lanes split the row's k-blocks (so adjacent lanes read adjacent weight
/// blocks). `rows` is split across y when the group count exceeds the x dispatch
/// limit (Vulkan's guaranteed minimum of 65535); the shader folds y back into
/// the group index.
fn gemvGroups(rows: usize, rows_per_group: usize) struct { x: u32, y: u32 } {
    const r = @max(rows_per_group, 1);
    const groups = (rows + r - 1) / r;
    const max_x: usize = 65535;
    const x: u32 = @intCast(@min(groups, max_x));
    const y: u32 = @intCast((groups + max_x - 1) / max_x);
    return .{ .x = x, .y = y };
}

/// Rows a GEMV workgroup owns for `quant`; must match the shader's
/// `ROWS_PER_GROUP` (q6_k/q4_k and the unquantized types are one row per
/// workgroup, the others four).
fn gemvRowsPerGroup(quant: Quant) usize {
    return quant.rowsPerGroup();
}

/// Rows the fused `gate_up` workgroup owns; must match the shader's
/// `ROWS_PER_GROUP` (the `u4` tile is eight rows, the rest four).
fn gateUpRowsPerGroup(quant: Quant, v: GpuVariant) usize {
    return grid.specConsts(quant.ggmlType(), v).rows_per_group;
}

/// The fused `gate_up` GEMV pipeline for the recorded quant and variant, from
/// the per-family spec-constant table. Null when the shader module failed to
/// build or the variant was pruned.
fn gateUpPipeline(self: *const VulkanPipeline, quant: Quant, v: GpuVariant) types.VkPipeline {
    const qi = gateUpIndex(quant.base()) orelse return null;
    return self.pipeline_gate_up[qi][@intFromEnum(v)];
}

/// The `gate_up_quants` index of a base quant, or null when it has no fused
/// shader.
fn gateUpIndex(base: Quant) ?usize {
    for (gate_up_quants, 0..) |q, i| {
        if (q == base) return i;
    }
    return null;
}

/// The fused three-weight MLP pipeline for the recorded quant and variant.
fn mlpPipeline(self: *const VulkanPipeline, quant: Quant, v: GpuVariant) types.VkPipeline {
    const qi = mlpIndex(quant.base()) orelse return null;
    return self.pipeline_mlp[qi][@intFromEnum(v)];
}

/// The `mlp_quants` index of a base quant, or null when it has no fused shader.
fn mlpIndex(base: Quant) ?usize {
    for (mlp_quants, 0..) |q, i| {
        if (q == base) return i;
    }
    return null;
}

/// The GEMV pipeline for `quant` (the spec-constant variant the plan selected),
/// from the per-family table. Null when the shader module failed to build or
/// the variant was pruned.
fn gemvPipeline(self: *const VulkanPipeline, quant: Quant) types.VkPipeline {
    const qi = quantIndex(quant.base()) orelse return null;
    return self.pipeline_matvec[qi][@intFromEnum(variantOf(quant))];
}

/// Backend-private storage behind a contract `Resident`: the Vulkan buffer and
/// its bound memory. Allocated by `initResident` and freed by `deinitResident`;
/// the public `Resident` carries the pointer in its `device` field.
const VulkanResident = struct {
    buffer: types.VkBuffer = null,
    memory: types.VkDeviceMemory = null,
};

/// One ring slot: an independent command buffer, fence, descriptor set and a
/// per-slot activation/result staging pair, so several dispatches may be in
/// flight at once without sharing a mutable binding. Buffers are host-visible
/// and persistently mapped, so a dispatch costs no `vkMapMemory`/`vkUnmapMemory`
/// and the readback is a plain memcpy after the fence.
pub const Slot = struct {
    cmd: types.VkCommandBuffer = null,
    fence: types.VkFence = null,
    desc: types.VkDescriptorSet = null,
    /// Second descriptor set, used by the two-kernel MLP: the fused `gate_up`
    /// dispatch reads it as `x`/`gate`/`up` and writes the `ffn_mid`, the `down`
    /// dispatch reads the `ffn_mid` and writes the slot's `y`. A descriptor set
    /// cannot be rebound between two dispatches recorded in one command buffer,
    /// so each kernel gets its own.
    desc2: types.VkDescriptorSet = null,
    x_buffer: types.VkBuffer = null,
    x_memory: types.VkDeviceMemory = null,
    x_size: u64 = 0,
    x_mapped: ?*anyopaque = null,
    /// Host-visible staging for `x`; copied into the device-local `x_buffer` per
    /// dispatch so the shader reads the activation from VRAM, not over PCIe.
    x_stage_buffer: types.VkBuffer = null,
    x_stage_memory: types.VkDeviceMemory = null,
    x_stage_size: u64 = 0,
    x_stage_mapped: ?*anyopaque = null,
    y_buffer: types.VkBuffer = null,
    y_memory: types.VkDeviceMemory = null,
    y_size: u64 = 0,
    y_mapped: ?*anyopaque = null,
};

/// The Vulkan storage behind a contract `Resident`.
fn vres(res: *const Resident) *VulkanResident {
    return @ptrCast(@alignCast(res.device.?));
}

/// Four quantized `matvec` compute pipelines plus the streamed weight/activation
/// buffers. Streamed dispatches stay synchronous (submit + wait); resident
/// dispatches run fence-asynchronous on a command-buffer ring so consecutive
/// independent GEMVs are submitted back-to-back and drained with one wait.
pub const VulkanPipeline = struct {
    device: types.VkDevice = null,
    physical_device: types.VkPhysicalDevice = null,
    compute_queue: types.VkQueue = null,
    descriptor_layout: types.VkDescriptorSetLayout = null,
    pipeline_layout: types.VkPipelineLayout = null,

    /// The single-weight GEMV pipelines, one per `(matvec_quants, variant)`,
    /// created by specializing each family's **one** shader module with
    /// `VkSpecializationInfo` (the axes). A pruned/unbuildable entry stays null.
    pipeline_matvec: [matvec_quants.len][grid.variant_slots]types.VkPipeline = [_][grid.variant_slots]types.VkPipeline{[_]types.VkPipeline{null} ** grid.variant_slots} ** matvec_quants.len,
    /// The fused `gate_up` pipelines (two weight buffers + gated activation),
    /// one per `(gate_up_quants, variant)`.
    pipeline_gate_up: [gate_up_quants.len][grid.variant_slots]types.VkPipeline = [_][grid.variant_slots]types.VkPipeline{[_]types.VkPipeline{null} ** grid.variant_slots} ** gate_up_quants.len,
    /// The fused three-weight MLP pipelines (gate/up/down + gated activation),
    /// one per `(mlp_quants, variant)`.
    pipeline_mlp: [mlp_quants.len][grid.variant_slots]types.VkPipeline = [_][grid.variant_slots]types.VkPipeline{[_]types.VkPipeline{null} ** grid.variant_slots} ** mlp_quants.len,

    pipeline_matmul_q4_0: types.VkPipeline = null,
    pipeline_matmul_q4_1: types.VkPipeline = null,
    pipeline_matmul_q8_0: types.VkPipeline = null,
    pipeline_matmul_q4_k: types.VkPipeline = null,

    /// One shader module per GEMV family (the driver specializes it into the
    /// variant pipelines); one per fused `gate_up` family.
    shader_matvec: [matvec_quants.len]types.VkShaderModule = [_]types.VkShaderModule{null} ** matvec_quants.len,
    shader_gate_up: [gate_up_quants.len]types.VkShaderModule = [_]types.VkShaderModule{null} ** gate_up_quants.len,
    shader_mlp: [mlp_quants.len]types.VkShaderModule = [_]types.VkShaderModule{null} ** mlp_quants.len,

    shader_matmul_q4_0: types.VkShaderModule = null,
    shader_matmul_q4_1: types.VkShaderModule = null,
    shader_matmul_q8_0: types.VkShaderModule = null,
    shader_matmul_q4_k: types.VkShaderModule = null,

    descriptor_pool: types.VkDescriptorPool = null,
    descriptor_set: types.VkDescriptorSet = null,
    command_pool: types.VkCommandPool = null,
    command_buffer: types.VkCommandBuffer = null,
    /// Shared synchronous-path fence (replaces `vkQueueWaitIdle`).
    fence: types.VkFence = null,

    /// Resident async ring (see `Slot`).
    slots: [RingSize]Slot = [_]Slot{.{}} ** RingSize,
    next_slot: u8 = 0,

    /// Shared activation for a burst of resident GEMVs that all read the same
    /// `x` (e.g. the q/k/v projections of one layer). It lives device-local and
    /// is staged once by the burst's first dispatch, so the run uploads the
    /// activation a single time and the shaders read it from VRAM. Valid only
    /// while the burst is in flight; the executor drains the burst before a new
    /// activation replaces it.
    burst_x: types.VkBuffer = null,
    burst_x_memory: types.VkDeviceMemory = null,
    burst_x_size: u64 = 0,
    burst_x_stage: types.VkBuffer = null,
    burst_x_stage_memory: types.VkDeviceMemory = null,
    burst_x_stage_size: u64 = 0,
    burst_x_stage_mapped: ?*anyopaque = null,

    /// Device-local `ffn_mid` of the two-kernel MLP (`gate_up` -> `down`). The
    /// activation/result of the intermediate stays in VRAM; the two dispatches
    /// are recorded back-to-back in one command buffer, so it never crosses the
    /// PCIe. Valid only while that submission is in flight (the executor drains
    /// the prior MLP before a new one starts).
    burst_mid: types.VkBuffer = null,
    burst_mid_memory: types.VkDeviceMemory = null,
    burst_mid_size: u64 = 0,

    buffer_x: types.VkBuffer = null,
    memory_x: types.VkDeviceMemory = null,
    size_x: u64 = 0,

    buffer_w: types.VkBuffer = null,
    memory_w: types.VkDeviceMemory = null,
    size_w: u64 = 0,

    buffer_y: types.VkBuffer = null,
    memory_y: types.VkDeviceMemory = null,
    size_y: u64 = 0,

    /// Persistent host maps of the shared staging/readback buffers, so a
    /// dispatch never pays a per-call `vkMapMemory`/`vkUnmapMemory`.
    mapped_x: ?*anyopaque = null,
    mapped_w: ?*anyopaque = null,
    mapped_y: ?*anyopaque = null,

    vkCreateShaderModule: ?types.PFN_vkCreateShaderModule = null,
    vkDestroyShaderModule: ?types.PFN_vkDestroyShaderModule = null,
    vkCreateDescriptorSetLayout: ?types.PFN_vkCreateDescriptorSetLayout = null,
    vkDestroyDescriptorSetLayout: ?types.PFN_vkDestroyDescriptorSetLayout = null,
    vkCreatePipelineLayout: ?types.PFN_vkCreatePipelineLayout = null,
    vkDestroyPipelineLayout: ?types.PFN_vkDestroyPipelineLayout = null,
    vkCreateComputePipelines: ?types.PFN_vkCreateComputePipelines = null,
    vkDestroyPipeline: ?types.PFN_vkDestroyPipeline = null,
    vkCreateDescriptorPool: ?types.PFN_vkCreateDescriptorPool = null,
    vkDestroyDescriptorPool: ?types.PFN_vkDestroyDescriptorPool = null,
    vkAllocateDescriptorSets: ?types.PFN_vkAllocateDescriptorSets = null,
    vkUpdateDescriptorSets: ?types.PFN_vkUpdateDescriptorSets = null,
    vkCreateBuffer: ?types.PFN_vkCreateBuffer = null,
    vkDestroyBuffer: ?types.PFN_vkDestroyBuffer = null,
    vkGetBufferMemoryRequirements: ?types.PFN_vkGetBufferMemoryRequirements = null,
    vkGetPhysicalDeviceMemoryProperties: ?types.PFN_vkGetPhysicalDeviceMemoryProperties = null,
    vkAllocateMemory: ?types.PFN_vkAllocateMemory = null,
    vkFreeMemory: ?types.PFN_vkFreeMemory = null,
    vkBindBufferMemory: ?types.PFN_vkBindBufferMemory = null,
    vkMapMemory: ?types.PFN_vkMapMemory = null,
    vkUnmapMemory: ?types.PFN_vkUnmapMemory = null,
    vkCreateCommandPool: ?types.PFN_vkCreateCommandPool = null,
    vkDestroyCommandPool: ?types.PFN_vkDestroyCommandPool = null,
    vkAllocateCommandBuffers: ?types.PFN_vkAllocateCommandBuffers = null,
    vkFreeCommandBuffers: ?types.PFN_vkFreeCommandBuffers = null,
    vkBeginCommandBuffer: ?types.PFN_vkBeginCommandBuffer = null,
    vkEndCommandBuffer: ?types.PFN_vkEndCommandBuffer = null,
    vkCmdBindPipeline: ?types.PFN_vkCmdBindPipeline = null,
    vkCmdBindDescriptorSets: ?types.PFN_vkCmdBindDescriptorSets = null,
    vkCmdPushConstants: ?types.PFN_vkCmdPushConstants = null,
    vkCmdDispatch: ?types.PFN_vkCmdDispatch = null,
    vkCmdCopyBuffer: ?types.PFN_vkCmdCopyBuffer = null,
    vkCmdPipelineBarrier: ?types.PFN_vkCmdPipelineBarrier = null,
    vkQueueSubmit: ?types.PFN_vkQueueSubmit = null,
    vkQueueWaitIdle: ?types.PFN_vkQueueWaitIdle = null,
    vkCreateFence: ?types.PFN_vkCreateFence = null,
    vkDestroyFence: ?types.PFN_vkDestroyFence = null,
    vkResetFences: ?types.PFN_vkResetFences = null,
    vkWaitForFences: ?types.PFN_vkWaitForFences = null,

    pub fn init(ctx: *context_mod.VulkanContext) ?VulkanPipeline {
        if (ctx.status != .vulkan or ctx.device == null or ctx.compute_queue == null) return null;
        var lib = ctx.lib orelse return null;

        var pipe = VulkanPipeline{
            .device = ctx.device,
            .physical_device = ctx.physical_device,
            .compute_queue = ctx.compute_queue,
        };

        pipe.vkCreateShaderModule = lib.lookup(types.PFN_vkCreateShaderModule, "vkCreateShaderModule");
        pipe.vkDestroyShaderModule = lib.lookup(types.PFN_vkDestroyShaderModule, "vkDestroyShaderModule");
        pipe.vkCreateDescriptorSetLayout = lib.lookup(types.PFN_vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
        pipe.vkDestroyDescriptorSetLayout = lib.lookup(types.PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
        pipe.vkCreatePipelineLayout = lib.lookup(types.PFN_vkCreatePipelineLayout, "vkCreatePipelineLayout");
        pipe.vkDestroyPipelineLayout = lib.lookup(types.PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
        pipe.vkCreateComputePipelines = lib.lookup(types.PFN_vkCreateComputePipelines, "vkCreateComputePipelines");
        pipe.vkDestroyPipeline = lib.lookup(types.PFN_vkDestroyPipeline, "vkDestroyPipeline");
        pipe.vkCreateDescriptorPool = lib.lookup(types.PFN_vkCreateDescriptorPool, "vkCreateDescriptorPool");
        pipe.vkDestroyDescriptorPool = lib.lookup(types.PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
        pipe.vkAllocateDescriptorSets = lib.lookup(types.PFN_vkAllocateDescriptorSets, "vkAllocateDescriptorSets");
        pipe.vkUpdateDescriptorSets = lib.lookup(types.PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
        pipe.vkCreateBuffer = lib.lookup(types.PFN_vkCreateBuffer, "vkCreateBuffer");
        pipe.vkDestroyBuffer = lib.lookup(types.PFN_vkDestroyBuffer, "vkDestroyBuffer");
        pipe.vkGetBufferMemoryRequirements = lib.lookup(types.PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
        pipe.vkGetPhysicalDeviceMemoryProperties = lib.lookup(types.PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
        pipe.vkAllocateMemory = lib.lookup(types.PFN_vkAllocateMemory, "vkAllocateMemory");
        pipe.vkFreeMemory = lib.lookup(types.PFN_vkFreeMemory, "vkFreeMemory");
        pipe.vkBindBufferMemory = lib.lookup(types.PFN_vkBindBufferMemory, "vkBindBufferMemory");
        pipe.vkMapMemory = lib.lookup(types.PFN_vkMapMemory, "vkMapMemory");
        pipe.vkUnmapMemory = lib.lookup(types.PFN_vkUnmapMemory, "vkUnmapMemory");
        pipe.vkCreateCommandPool = lib.lookup(types.PFN_vkCreateCommandPool, "vkCreateCommandPool");
        pipe.vkDestroyCommandPool = lib.lookup(types.PFN_vkDestroyCommandPool, "vkDestroyCommandPool");
        pipe.vkAllocateCommandBuffers = lib.lookup(types.PFN_vkAllocateCommandBuffers, "vkAllocateCommandBuffers");
        pipe.vkFreeCommandBuffers = lib.lookup(types.PFN_vkFreeCommandBuffers, "vkFreeCommandBuffers");
        pipe.vkBeginCommandBuffer = lib.lookup(types.PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer");
        pipe.vkEndCommandBuffer = lib.lookup(types.PFN_vkEndCommandBuffer, "vkEndCommandBuffer");
        pipe.vkCmdBindPipeline = lib.lookup(types.PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
        pipe.vkCmdBindDescriptorSets = lib.lookup(types.PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
        pipe.vkCmdPushConstants = lib.lookup(types.PFN_vkCmdPushConstants, "vkCmdPushConstants");
        pipe.vkCmdDispatch = lib.lookup(types.PFN_vkCmdDispatch, "vkCmdDispatch");
        pipe.vkCmdCopyBuffer = lib.lookup(types.PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");
        pipe.vkCmdPipelineBarrier = lib.lookup(types.PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
        pipe.vkQueueSubmit = lib.lookup(types.PFN_vkQueueSubmit, "vkQueueSubmit");
        pipe.vkQueueWaitIdle = lib.lookup(types.PFN_vkQueueWaitIdle, "vkQueueWaitIdle");
        pipe.vkCreateFence = lib.lookup(types.PFN_vkCreateFence, "vkCreateFence");
        pipe.vkDestroyFence = lib.lookup(types.PFN_vkDestroyFence, "vkDestroyFence");
        pipe.vkResetFences = lib.lookup(types.PFN_vkResetFences, "vkResetFences");
        pipe.vkWaitForFences = lib.lookup(types.PFN_vkWaitForFences, "vkWaitForFences");

        if (pipe.vkCreateShaderModule == null or pipe.vkCreateDescriptorSetLayout == null or pipe.vkCreatePipelineLayout == null or pipe.vkCreateComputePipelines == null or pipe.vkCreateBuffer == null or pipe.vkAllocateMemory == null or pipe.vkMapMemory == null or pipe.vkQueueSubmit == null) {
            return null;
        }
        if (pipe.vkCreateFence == null or pipe.vkDestroyFence == null or pipe.vkResetFences == null or pipe.vkWaitForFences == null) {
            return null;
        }

        const matvec_spv = [_][]align(4) const u8{
            &shaders.matvec_q4_0_spv,
            &shaders.matvec_q4_1_spv,
            &shaders.matvec_q8_0_spv,
            &shaders.matvec_q6_k_spv,
            &shaders.matvec_q4_k_spv,
            &shaders.matvec_f32_spv,
            &shaders.matvec_f16_spv,
            &shaders.matvec_bf16_spv,
        };
        for (&pipe.shader_matvec, matvec_spv) |*m, spv| m.* = pipe.createShaderModule(spv) orelse return null;
        const gate_up_spv = [_][]align(4) const u8{
            &shaders.matvec_gate_up_q4_0_spv,
            &shaders.matvec_gate_up_q4_1_spv,
            &shaders.matvec_gate_up_q8_0_spv,
        };
        for (&pipe.shader_gate_up, gate_up_spv) |*m, spv| m.* = pipe.createShaderModule(spv) orelse return null;
        const mlp_spv = [_][]align(4) const u8{
            &shaders.matvec_mlp_q4_0_spv,
            &shaders.matvec_mlp_q4_1_spv,
            &shaders.matvec_mlp_q8_0_spv,
        };
        for (&pipe.shader_mlp, mlp_spv) |*m, spv| m.* = pipe.createShaderModule(spv) orelse return null;

        pipe.shader_matmul_q4_0 = pipe.createShaderModule(&shaders.matmul_q4_0_spv) orelse return null;
        pipe.shader_matmul_q4_1 = pipe.createShaderModule(&shaders.matmul_q4_1_spv) orelse return null;
        pipe.shader_matmul_q8_0 = pipe.createShaderModule(&shaders.matmul_q8_0_spv) orelse return null;
        pipe.shader_matmul_q4_k = pipe.createShaderModule(&shaders.matmul_q4_k_spv) orelse return null;

        const bindings = [_]types.VkDescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT },
            .{ .binding = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT },
            .{ .binding = 2, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT },
            .{ .binding = 3, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT },
            .{ .binding = 4, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT },
        };

        const dsl_info = types.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = bindings.len,
            .pBindings = &bindings,
        };

        if (pipe.vkCreateDescriptorSetLayout.?(ctx.device, &dsl_info, null, &pipe.descriptor_layout) != .SUCCESS) return null;

        // Both the GEMV and the batched GEMM pipelines share this layout, so the
        // range must hold the larger of the two push-constant blocks (the fused
        // `gate_up` GEMV adds the activation selector). A GEMV still uploads
        // only its own `PushConsts`.
        const push_range = types.VkPushConstantRange{
            .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = @max(@sizeOf(PushConsts), @max(@sizeOf(PushConstsMatMul), @sizeOf(PushConstsMlp))),
        };

        const dsl_arr = [_]types.VkDescriptorSetLayout{pipe.descriptor_layout};
        const pl_info = types.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &dsl_arr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = @ptrCast(&push_range),
        };

        if (pipe.vkCreatePipelineLayout.?(ctx.device, &pl_info, null, &pipe.pipeline_layout) != .SUCCESS) return null;

        // One shader module per family, specialized into its variant pipelines:
        // the plain (4-row tile, shared reduction), the 8-row tile (`u4`), the
        // subgroup/wave reduction (`sub`), the inner k-block unroll (`un2`/`un4`)
        // and the workgroup sizes (`w32`/`w128`). The modules declare Vulkan 1.1
        // subgroup arithmetic (the `sub` path), so on a 1.0 device they cannot be
        // created: the GEMV pipelines are then absent and the GEMV falls back to
        // the CPU. A variant the device rejects (a workgroup size, the subgroup
        // path) leaves its slot null (the prune).
        if (ctx.api_version >= vk_api_version_1_1) {
            for (matvec_quants, 0..) |q, qi| {
                for (shader_variants) |v| {
                    pipe.pipeline_matvec[qi][@intFromEnum(v)] = pipe.createSpecializedPipeline(pipe.shader_matvec[qi], grid.specConsts(q.ggmlType(), v)) orelse null;
                }
            }
            for (gate_up_quants, 0..) |q, qi| {
                for (shader_variants) |v| {
                    pipe.pipeline_gate_up[qi][@intFromEnum(v)] = pipe.createSpecializedPipeline(pipe.shader_gate_up[qi], grid.specConsts(q.ggmlType(), v)) orelse null;
                }
            }
            for (mlp_quants, 0..) |q, qi| {
                for (shader_variants) |v| {
                    pipe.pipeline_mlp[qi][@intFromEnum(v)] = pipe.createSpecializedPipeline(pipe.shader_mlp[qi], grid.specConsts(q.ggmlType(), v)) orelse null;
                }
            }
        }

        pipe.pipeline_matmul_q4_0 = pipe.createComputePipeline(pipe.shader_matmul_q4_0) orelse return null;
        pipe.pipeline_matmul_q4_1 = pipe.createComputePipeline(pipe.shader_matmul_q4_1) orelse return null;
        pipe.pipeline_matmul_q8_0 = pipe.createComputePipeline(pipe.shader_matmul_q8_0) orelse return null;
        pipe.pipeline_matmul_q4_k = pipe.createComputePipeline(pipe.shader_matmul_q4_k) orelse return null;

        if (!pipe.ensureBuffer(&pipe.buffer_x, &pipe.memory_x, &pipe.size_x, &pipe.mapped_x, 16 * 1024 * 1024)) return null;
        if (!pipe.ensureBuffer(&pipe.buffer_w, &pipe.memory_w, &pipe.size_w, &pipe.mapped_w, 128 * 1024 * 1024)) return null;
        if (!pipe.ensureBuffer(&pipe.buffer_y, &pipe.memory_y, &pipe.size_y, &pipe.mapped_y, 16 * 1024 * 1024)) return null;

        const pool_sizes = [_]types.VkDescriptorPoolSize{
            .{ .type = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 5 * (2 * RingSize + 1) },
        };

        const pool_info = types.VkDescriptorPoolCreateInfo{
            .maxSets = 2 * RingSize + 1,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        };

        if (pipe.vkCreateDescriptorPool.?(ctx.device, &pool_info, null, &pipe.descriptor_pool) != .SUCCESS) return null;

        const alloc_info = types.VkDescriptorSetAllocateInfo{
            .descriptorPool = pipe.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &dsl_arr,
        };

        if (pipe.vkAllocateDescriptorSets.?(ctx.device, &alloc_info, @ptrCast(&pipe.descriptor_set)) != .SUCCESS) return null;

        pipe.updateDescriptorSet();

        const cp_info = types.VkCommandPoolCreateInfo{
            .queueFamilyIndex = ctx.queue_family_index,
        };

        if (pipe.vkCreateCommandPool.?(ctx.device, &cp_info, null, &pipe.command_pool) != .SUCCESS) return null;

        const cb_info = types.VkCommandBufferAllocateInfo{
            .commandPool = pipe.command_pool,
            .commandBufferCount = 1,
        };

        if (pipe.vkAllocateCommandBuffers.?(ctx.device, &cb_info, @ptrCast(&pipe.command_buffer)) != .SUCCESS) return null;

        const ring_layouts = [_]types.VkDescriptorSetLayout{pipe.descriptor_layout} ** (2 * RingSize);
        const ring_alloc = types.VkDescriptorSetAllocateInfo{
            .descriptorPool = pipe.descriptor_pool,
            .descriptorSetCount = 2 * RingSize,
            .pSetLayouts = &ring_layouts,
        };
        var ring_sets: [2 * RingSize]types.VkDescriptorSet = undefined;
        if (pipe.vkAllocateDescriptorSets.?(ctx.device, &ring_alloc, &ring_sets) != .SUCCESS) return null;

        var ring_cmds: [RingSize]types.VkCommandBuffer = undefined;
        const rcb_info = types.VkCommandBufferAllocateInfo{
            .commandPool = pipe.command_pool,
            .commandBufferCount = RingSize,
        };
        if (pipe.vkAllocateCommandBuffers.?(ctx.device, &rcb_info, &ring_cmds) != .SUCCESS) return null;

        for (&pipe.slots, 0..) |*s, i| {
            s.desc = ring_sets[2 * i];
            s.desc2 = ring_sets[2 * i + 1];
            s.cmd = ring_cmds[i];
            if (!pipe.createFence(&s.fence, true)) return null;
        }
        if (!pipe.createFence(&pipe.fence, false)) return null;

        return pipe;
    }

    pub fn deinit(self: *VulkanPipeline) void {
        const d = self.device orelse return;

        if (self.mapped_x != null) if (self.vkUnmapMemory) |f| f(d, self.memory_x);
        if (self.mapped_w != null) if (self.vkUnmapMemory) |f| f(d, self.memory_w);
        if (self.mapped_y != null) if (self.vkUnmapMemory) |f| f(d, self.memory_y);
        self.mapped_x = null;
        self.mapped_w = null;
        self.mapped_y = null;

        for (&self.slots) |*s| {
            if (s.x_mapped != null) if (self.vkUnmapMemory) |f| f(d, s.x_memory);
            if (s.x_stage_mapped != null) if (self.vkUnmapMemory) |f| f(d, s.x_stage_memory);
            if (s.y_mapped != null) if (self.vkUnmapMemory) |f| f(d, s.y_memory);
            if (s.x_buffer) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
            if (s.x_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);
            if (s.x_stage_buffer) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
            if (s.x_stage_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);
            if (s.y_buffer) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
            if (s.y_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);
            if (s.fence) |fe| if (self.vkDestroyFence) |f| f(d, fe, null);
        }
        if (self.fence) |fe| if (self.vkDestroyFence) |f| f(d, fe, null);

        if (self.burst_x_stage_mapped != null) if (self.vkUnmapMemory) |f| f(d, self.burst_x_stage_memory);
        if (self.burst_x_stage) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.burst_x_stage_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);
        if (self.burst_x) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.burst_x_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);

        if (self.burst_mid) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.burst_mid_memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);

        if (self.buffer_x) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.memory_x) |m| if (self.vkFreeMemory) |f| f(d, m, null);

        if (self.buffer_w) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.memory_w) |m| if (self.vkFreeMemory) |f| f(d, m, null);

        if (self.buffer_y) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
        if (self.memory_y) |m| if (self.vkFreeMemory) |f| f(d, m, null);

        if (self.command_pool) |cp| if (self.vkDestroyCommandPool) |f| f(d, cp, null);
        if (self.descriptor_pool) |dp| if (self.vkDestroyDescriptorPool) |f| f(d, dp, null);

        for (&self.pipeline_matvec) |*row| {
            for (row) |p| if (p) |pp| if (self.vkDestroyPipeline) |f| f(d, pp, null);
        }
        for (&self.pipeline_gate_up) |*row| {
            for (row) |p| if (p) |pp| if (self.vkDestroyPipeline) |f| f(d, pp, null);
        }
        for (&self.pipeline_mlp) |*row| {
            for (row) |p| if (p) |pp| if (self.vkDestroyPipeline) |f| f(d, pp, null);
        }

        if (self.pipeline_matmul_q4_0) |p| if (self.vkDestroyPipeline) |f| f(d, p, null);
        if (self.pipeline_matmul_q4_1) |p| if (self.vkDestroyPipeline) |f| f(d, p, null);
        if (self.pipeline_matmul_q8_0) |p| if (self.vkDestroyPipeline) |f| f(d, p, null);
        if (self.pipeline_matmul_q4_k) |p| if (self.vkDestroyPipeline) |f| f(d, p, null);

        for (self.shader_matvec) |s| if (s) |ss| if (self.vkDestroyShaderModule) |f| f(d, ss, null);
        for (self.shader_gate_up) |s| if (s) |ss| if (self.vkDestroyShaderModule) |f| f(d, ss, null);
        for (self.shader_mlp) |s| if (s) |ss| if (self.vkDestroyShaderModule) |f| f(d, ss, null);

        if (self.shader_matmul_q4_0) |s| if (self.vkDestroyShaderModule) |f| f(d, s, null);
        if (self.shader_matmul_q4_1) |s| if (self.vkDestroyShaderModule) |f| f(d, s, null);
        if (self.shader_matmul_q8_0) |s| if (self.vkDestroyShaderModule) |f| f(d, s, null);
        if (self.shader_matmul_q4_k) |s| if (self.vkDestroyShaderModule) |f| f(d, s, null);

        if (self.pipeline_layout) |pl| if (self.vkDestroyPipelineLayout) |f| f(d, pl, null);
        if (self.descriptor_layout) |dsl| if (self.vkDestroyDescriptorSetLayout) |f| f(d, dsl, null);
    }

    fn createShaderModule(self: *VulkanPipeline, spv: []align(4) const u8) ?types.VkShaderModule {
        const info = types.VkShaderModuleCreateInfo{
            .codeSize = spv.len,
            .pCode = @ptrCast(spv.ptr),
        };
        var module: types.VkShaderModule = null;
        if (self.vkCreateShaderModule.?(self.device, &info, null, &module) != .SUCCESS) return null;
        return module;
    }

    fn createComputePipeline(self: *VulkanPipeline, shader: types.VkShaderModule) ?types.VkPipeline {
        return self.createComputePipelineSpec(shader, null);
    }

    /// Create a pipeline from `shader` with the specialization constants `spec`
    /// (the GPU variant axes: rows/unroll/reduction). One shader module thus
    /// becomes N pipelines at `vkCreateComputePipelines`: the driver bakes the
    /// constant values in, so no extra shader is needed per variant.
    fn createSpecializedPipeline(self: *VulkanPipeline, shader: types.VkShaderModule, spec: grid.SpecConsts) ?types.VkPipeline {
        const values = [4]u32{ spec.rows_per_group, spec.unroll, @intFromEnum(spec.reduction), spec.workgroup_size };
        const entries = [4]types.VkSpecializationMapEntry{
            .{ .constantID = 0, .offset = 0, .size = @sizeOf(u32) },
            .{ .constantID = 1, .offset = @sizeOf(u32), .size = @sizeOf(u32) },
            .{ .constantID = 2, .offset = 2 * @sizeOf(u32), .size = @sizeOf(u32) },
            .{ .constantID = 3, .offset = 3 * @sizeOf(u32), .size = @sizeOf(u32) },
        };
        const info = types.VkSpecializationInfo{
            .mapEntryCount = entries.len,
            .pMapEntries = &entries,
            .dataSize = @sizeOf(@TypeOf(values)),
            .pData = @ptrCast(&values),
        };
        return self.createComputePipelineSpec(shader, &info);
    }

    fn createComputePipelineSpec(self: *VulkanPipeline, shader: types.VkShaderModule, spec: ?*const types.VkSpecializationInfo) ?types.VkPipeline {
        const stage = types.VkPipelineShaderStageCreateInfo{
            .stage = types.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pSpecializationInfo = spec,
        };
        const info = types.VkComputePipelineCreateInfo{
            .stage = stage,
            .layout = self.pipeline_layout,
        };
        var pipe: types.VkPipeline = null;
        if (self.vkCreateComputePipelines.?(self.device, null, 1, @ptrCast(&info), null, @ptrCast(&pipe)) != .SUCCESS) return null;
        return pipe;
    }

    fn findMemoryType(self: *VulkanPipeline, type_bits: u32, props: u32) ?u32 {
        var mem_props: types.VkPhysicalDeviceMemoryProperties = undefined;
        self.vkGetPhysicalDeviceMemoryProperties.?(self.physical_device, &mem_props);
        for (0..mem_props.memoryTypeCount) |i| {
            if ((type_bits & (@as(u32, 1) << @intCast(i))) != 0) {
                if ((mem_props.memoryTypes[i].propertyFlags & props) == props) {
                    return @intCast(i);
                }
            }
        }
        return null;
    }

    fn createFence(self: *VulkanPipeline, fence_ptr: *types.VkFence, signaled: bool) bool {
        const info = types.VkFenceCreateInfo{
            .flags = if (signaled) types.VK_FENCE_CREATE_SIGNALED_BIT else 0,
        };
        var f: types.VkFence = null;
        if (self.vkCreateFence.?(self.device, &info, null, &f) != .SUCCESS) return false;
        fence_ptr.* = f;
        return true;
    }

    fn ensureBuffer(
        self: *VulkanPipeline,
        buffer_ptr: *types.VkBuffer,
        memory_ptr: *types.VkDeviceMemory,
        current_size: *u64,
        mapped_ptr: ?*?*anyopaque,
        min_size: u64,
    ) bool {
        const d = self.device orelse return false;

        if (current_size.* >= min_size and buffer_ptr.* != null) {
            if (mapped_ptr) |mp| {
                if (mp.* == null) {
                    var m: ?*anyopaque = null;
                    if (self.vkMapMemory.?(d, memory_ptr.*, 0, current_size.*, 0, &m) != .SUCCESS or m == null) return false;
                    mp.* = m;
                }
            }
            return true;
        }

        if (mapped_ptr) |mp| {
            if (mp.* != null) {
                self.vkUnmapMemory.?(d, memory_ptr.*);
                mp.* = null;
            }
        }

        if (buffer_ptr.*) |b| self.vkDestroyBuffer.?(d, b, null);
        if (memory_ptr.*) |m| self.vkFreeMemory.?(d, m, null);

        buffer_ptr.* = null;
        memory_ptr.* = null;

        const b_info = types.VkBufferCreateInfo{
            .size = min_size,
            .usage = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        };

        var buf: types.VkBuffer = null;
        if (self.vkCreateBuffer.?(d, &b_info, null, &buf) != .SUCCESS) return false;

        var reqs: types.VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements.?(d, buf, &reqs);

        const mem_type = self.findMemoryType(reqs.memoryTypeBits, types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) orelse return false;

        const m_info = types.VkMemoryAllocateInfo{
            .allocationSize = reqs.size,
            .memoryTypeIndex = mem_type,
        };

        var mem: types.VkDeviceMemory = null;
        if (self.vkAllocateMemory.?(d, &m_info, null, &mem) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            return false;
        }

        if (self.vkBindBufferMemory.?(d, buf, mem, 0) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            self.vkFreeMemory.?(d, mem, null);
            return false;
        }

        if (mapped_ptr) |mp| {
            var m: ?*anyopaque = null;
            if (self.vkMapMemory.?(d, mem, 0, reqs.size, 0, &m) != .SUCCESS or m == null) {
                self.vkDestroyBuffer.?(d, buf, null);
                self.vkFreeMemory.?(d, mem, null);
                return false;
            }
            mp.* = m;
        }

        buffer_ptr.* = buf;
        memory_ptr.* = mem;
        current_size.* = min_size;
        return true;
    }

    /// Like `ensureBuffer` but allocates **device-local** memory and never maps
    /// it: used for the resident GEMV activation, which is staged through a
    /// host-visible buffer with `vkCmdCopyBuffer` so the shader reads it from
    /// VRAM instead of over PCIe. Returns false if the allocation fails.
    fn ensureDeviceBuffer(
        self: *VulkanPipeline,
        buffer_ptr: *types.VkBuffer,
        memory_ptr: *types.VkDeviceMemory,
        current_size: *u64,
        min_size: u64,
    ) bool {
        const d = self.device orelse return false;
        if (current_size.* >= min_size and buffer_ptr.* != null) return true;

        if (buffer_ptr.*) |b| self.vkDestroyBuffer.?(d, b, null);
        if (memory_ptr.*) |m| self.vkFreeMemory.?(d, m, null);
        buffer_ptr.* = null;
        memory_ptr.* = null;

        const b_info = types.VkBufferCreateInfo{
            .size = min_size,
            .usage = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | types.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        var buf: types.VkBuffer = null;
        if (self.vkCreateBuffer.?(d, &b_info, null, &buf) != .SUCCESS) return false;

        var reqs: types.VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements.?(d, buf, &reqs);
        const mem_type = self.findMemoryType(reqs.memoryTypeBits, types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) orelse {
            self.vkDestroyBuffer.?(d, buf, null);
            return false;
        };
        const m_info = types.VkMemoryAllocateInfo{ .allocationSize = reqs.size, .memoryTypeIndex = mem_type };
        var mem: types.VkDeviceMemory = null;
        if (self.vkAllocateMemory.?(d, &m_info, null, &mem) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            return false;
        }
        if (self.vkBindBufferMemory.?(d, buf, mem, 0) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            self.vkFreeMemory.?(d, mem, null);
            return false;
        }
        buffer_ptr.* = buf;
        memory_ptr.* = mem;
        current_size.* = min_size;
        return true;
    }

    fn updateDescriptorSet(self: *VulkanPipeline) void {
        self.updateDescriptorSetWithWeight(self.buffer_w, self.size_w);
    }

    fn updateDescriptorSetWithWeight(self: *VulkanPipeline, w_buffer: types.VkBuffer, w_range: u64) void {
        const d_info_x = [_]types.VkDescriptorBufferInfo{.{ .buffer = self.buffer_x, .offset = 0, .range = self.size_x }};
        const d_info_w = [_]types.VkDescriptorBufferInfo{.{ .buffer = w_buffer, .offset = 0, .range = w_range }};
        const d_info_y = [_]types.VkDescriptorBufferInfo{.{ .buffer = self.buffer_y, .offset = 0, .range = self.size_y }};

        const writes = [_]types.VkWriteDescriptorSet{
            .{ .dstSet = self.descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_x },
            .{ .dstSet = self.descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w },
            .{ .dstSet = self.descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_y },
            // Binding 3 is the second weight of a fused `gate_up` GEMV and
            // binding 4 the third weight of a fused MLP; unused by the
            // single-weight shaders, bound to the same buffer so every
            // descriptor set is fully specified.
            .{ .dstSet = self.descriptor_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w },
            .{ .dstSet = self.descriptor_set, .dstBinding = 4, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w },
        };

        self.vkUpdateDescriptorSets.?(self.device, writes.len, &writes, 0, null);
    }

    fn allocBuffer(self: *VulkanPipeline, size: u64) ?VulkanResident {
        const d = self.device orelse return null;

        const b_info = types.VkBufferCreateInfo{
            .size = size,
            .usage = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        };

        var buf: types.VkBuffer = null;
        if (self.vkCreateBuffer.?(d, &b_info, null, &buf) != .SUCCESS) return null;

        var reqs: types.VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements.?(d, buf, &reqs);

        const mem_type = self.findMemoryType(reqs.memoryTypeBits, types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) orelse {
            self.vkDestroyBuffer.?(d, buf, null);
            return null;
        };

        const m_info = types.VkMemoryAllocateInfo{
            .allocationSize = reqs.size,
            .memoryTypeIndex = mem_type,
        };

        var mem: types.VkDeviceMemory = null;
        if (self.vkAllocateMemory.?(d, &m_info, null, &mem) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            return null;
        }

        if (self.vkBindBufferMemory.?(d, buf, mem, 0) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            self.vkFreeMemory.?(d, mem, null);
            return null;
        }

        return .{ .buffer = buf, .memory = mem };
    }

    /// Upload `bytes` into a persistent **device-local** GPU buffer (VRAM on a
    /// discrete GPU) reused across every GEMV that reads the weight. The bytes
    /// are staged through a host-visible buffer and copied with `vkCmdCopyBuffer`
    /// followed by a transfer->shader barrier, so the weight is read from VRAM at
    /// full bandwidth instead of over PCIe. Falls back to a host-visible mapping
    /// when the device-local/copy path is unavailable. Returns false if both
    /// fail.
    pub fn initResident(self: *VulkanPipeline, allocator: std.mem.Allocator, res: *Resident, bytes: []const u8) bool {
        const d = self.device orelse return false;
        if (bytes.len == 0) return false;

        const storage = allocator.create(VulkanResident) catch return false;
        storage.* = .{};

        if (self.allocDeviceLocal(bytes.len)) |dst| {
            if (self.uploadDeviceLocal(dst, bytes)) {
                storage.* = dst;
                res.* = .{ .bytes = bytes.len, .device = storage };
                return true;
            }
            if (dst.buffer) |b| self.vkDestroyBuffer.?(d, b, null);
            if (dst.memory) |m| self.vkFreeMemory.?(d, m, null);
        }
        if (self.initResidentMapped(storage, bytes)) {
            res.* = .{ .bytes = bytes.len, .device = storage };
            return true;
        }
        allocator.destroy(storage);
        return false;
    }

    /// Host-visible fallback: allocate a mappable buffer and memcpy the weight in.
    fn initResidentMapped(self: *VulkanPipeline, storage: *VulkanResident, bytes: []const u8) bool {
        const d = self.device orelse return false;
        const buf = self.allocBuffer(bytes.len) orelse return false;

        var mapped: ?*anyopaque = null;
        if (self.vkMapMemory.?(d, buf.memory, 0, bytes.len, 0, &mapped) != .SUCCESS or mapped == null) {
            if (buf.buffer) |b| self.vkDestroyBuffer.?(d, b, null);
            if (buf.memory) |m| self.vkFreeMemory.?(d, m, null);
            return false;
        }
        @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..bytes.len], bytes);
        self.vkUnmapMemory.?(d, buf.memory);

        storage.* = buf;
        return true;
    }

    /// A device-local (VRAM) storage buffer, transfer destination; null when the
    /// copy commands are unavailable or the allocation fails.
    fn allocDeviceLocal(self: *VulkanPipeline, size: u64) ?VulkanResident {
        const d = self.device orelse return null;
        if (self.vkCmdCopyBuffer == null or self.vkCmdPipelineBarrier == null) return null;

        const b_info = types.VkBufferCreateInfo{
            .size = size,
            .usage = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | types.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        var buf: types.VkBuffer = null;
        if (self.vkCreateBuffer.?(d, &b_info, null, &buf) != .SUCCESS) return null;

        var reqs: types.VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements.?(d, buf, &reqs);

        const mem_type = self.findMemoryType(reqs.memoryTypeBits, types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) orelse {
            self.vkDestroyBuffer.?(d, buf, null);
            return null;
        };

        const m_info = types.VkMemoryAllocateInfo{
            .allocationSize = reqs.size,
            .memoryTypeIndex = mem_type,
        };
        var mem: types.VkDeviceMemory = null;
        if (self.vkAllocateMemory.?(d, &m_info, null, &mem) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            return null;
        }
        if (self.vkBindBufferMemory.?(d, buf, mem, 0) != .SUCCESS) {
            self.vkDestroyBuffer.?(d, buf, null);
            self.vkFreeMemory.?(d, mem, null);
            return null;
        }
        return .{ .buffer = buf, .memory = mem };
    }

    /// Stage `bytes` through a host-visible buffer, copy into the device-local
    /// `dst` and barrier transfer-write -> shader-read; synchronous one-shot.
    fn uploadDeviceLocal(self: *VulkanPipeline, dst: VulkanResident, bytes: []const u8) bool {
        const d = self.device orelse return false;
        const staging = self.allocBuffer(bytes.len) orelse return false;
        defer {
            if (staging.buffer) |b| self.vkDestroyBuffer.?(d, b, null);
            if (staging.memory) |m| self.vkFreeMemory.?(d, m, null);
        }

        var mapped: ?*anyopaque = null;
        if (self.vkMapMemory.?(d, staging.memory, 0, bytes.len, 0, &mapped) != .SUCCESS or mapped == null) return false;
        @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..bytes.len], bytes);
        self.vkUnmapMemory.?(d, staging.memory);

        const cmd = self.command_buffer;
        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(cmd, &begin_info) != .SUCCESS) return false;

        const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = bytes.len }};
        self.vkCmdCopyBuffer.?(cmd, staging.buffer, dst.buffer, 1, &regions);

        const barriers = [_]types.VkBufferMemoryBarrier{.{
            .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
            .buffer = dst.buffer,
            .offset = 0,
            .size = types.VK_WHOLE_SIZE,
        }};
        self.vkCmdPipelineBarrier.?(
            cmd,
            types.VK_PIPELINE_STAGE_TRANSFER_BIT,
            types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            null,
            1,
            @ptrCast(&barriers),
            0,
            null,
        );

        if (self.vkEndCommandBuffer.?(cmd) != .SUCCESS) return false;
        return self.submitShared();
    }

    /// Release a resident weight buffer and its backend storage.
    pub fn deinitResident(self: *VulkanPipeline, allocator: std.mem.Allocator, res: *Resident) void {
        const d = self.device orelse return;
        if (res.device) |p| {
            const storage: *VulkanResident = @ptrCast(@alignCast(p));
            if (storage.buffer) |b| if (self.vkDestroyBuffer) |f| f(d, b, null);
            if (storage.memory) |m| if (self.vkFreeMemory) |f| f(d, m, null);
            allocator.destroy(storage);
        }
        res.* = .{};
    }

    fn stageMapped(self: *VulkanPipeline, mapped: ?*anyopaque, data: []const u8) bool {
        _ = self;
        const m = mapped orelse return false;
        @memcpy(@as([*]u8, @ptrCast(m))[0..data.len], data);
        return true;
    }

    /// Submit the shared-path command buffer (streamed GEMV/GEMM) and wait its
    /// fence. No `vkQueueWaitIdle`; the readback buffer is persistently mapped.
    fn submitShared(self: *VulkanPipeline) bool {
        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&self.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&self.command_buffer),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, self.fence) != .SUCCESS) return false;
        if (self.vkWaitForFences.?(self.device, 1, @ptrCast(&self.fence), types.VK_TRUE, std.math.maxInt(u64)) != .SUCCESS) return false;
        return true;
    }

    fn submitGemv(self: *VulkanPipeline, pipe: types.VkPipeline, quant: Quant, rows: usize, num_blocks: usize, out: []f32) bool {
        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(self.command_buffer, &begin_info) != .SUCCESS) return false;

        self.vkCmdBindPipeline.?(self.command_buffer, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

        const dsets = [_]types.VkDescriptorSet{self.descriptor_set};
        self.vkCmdBindDescriptorSets.?(self.command_buffer, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);

        const groups = gemvGroups(rows, gemvRowsPerGroup(quant));
        const push = PushConsts{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = groups.x,
            .act = 0,
        };
        self.vkCmdPushConstants.?(self.command_buffer, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &push);

        self.vkCmdDispatch.?(self.command_buffer, groups.x, groups.y, 1);

        if (self.vkEndCommandBuffer.?(self.command_buffer) != .SUCCESS) return false;
        if (!self.submitShared()) return false;

        const y_bytes_count = rows * @sizeOf(f32);
        const mapped_y = self.mapped_y orelse return false;
        @memcpy(std.mem.sliceAsBytes(out[0..rows]), @as([*]const u8, @ptrCast(mapped_y))[0..y_bytes_count]);
        return true;
    }

    /// Record + submit + wait the batched GEMM (`y[t*rows + n] = dot(weight[n],
    /// x[t])`) and copy the `tokens*rows` f32 result back into `out`. The X and
    /// weight buffers have already been staged and the descriptor set updated.
    /// Per-shader workgroup tile for the batched GEMM: `x` output rows per
    /// workgroup, `y` tokens per workgroup. Must match the shader's local size
    /// and its per-lane token count (q4_0/q4_k: 4x4; the tiled shaders: 16x16).
    fn matmulTile(quant: Quant) struct { x: u32, y: u32 } {
        return switch (quant.base()) {
            .q4_0, .q4_k => .{ .x = 4, .y = 4 },
            else => .{ .x = 16, .y = 16 },
        };
    }

    fn submitMatMul(self: *VulkanPipeline, pipe: types.VkPipeline, tokens: usize, rows: usize, num_blocks: usize, tile_x: u32, tile_y: u32, out: []f32) bool {
        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(self.command_buffer, &begin_info) != .SUCCESS) return false;

        self.vkCmdBindPipeline.?(self.command_buffer, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

        const dsets = [_]types.VkDescriptorSet{self.descriptor_set};
        self.vkCmdBindDescriptorSets.?(self.command_buffer, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);

        const push = PushConstsMatMul{
            .num_tokens = @intCast(tokens),
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
        };
        self.vkCmdPushConstants.?(self.command_buffer, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstsMatMul), &push);

        const group_x: u32 = @intCast((rows + tile_x - 1) / tile_x);
        const group_y: u32 = @intCast((tokens + tile_y - 1) / tile_y);
        self.vkCmdDispatch.?(self.command_buffer, group_x, group_y, 1);

        if (self.vkEndCommandBuffer.?(self.command_buffer) != .SUCCESS) return false;
        if (!self.submitShared()) return false;

        const y_bytes_count = tokens * rows * @sizeOf(f32);
        const mapped_y = self.mapped_y orelse return false;
        @memcpy(std.mem.sliceAsBytes(out[0 .. tokens * rows]), @as([*]const u8, @ptrCast(mapped_y))[0..y_bytes_count]);
        return true;
    }

    /// Batched GEMM: `out[t*rows + n] = dot(dequant(weight[n]), x[t])` for `m`
    /// activation rows and `rows` output features, where `x` is `m*k` f32, the
    /// weight is `[k, rows]` quantized, and `out` is `m*rows` f32. This mirrors
    /// `runPrefillMatmul`'s `matVecBatch` shape but runs the whole batch in one
    /// dispatch, amortizing the weight transfer over `m` tokens. Only the quants
    /// with a batched GEMM shader are accepted (`q6_k` has none). Returns false
    /// when the op is unsupported or the device rejects it (caller falls back).
    pub fn dispatchMatMul(
        self: *VulkanPipeline,
        quant: Quant,
        weight_bytes: []const u8,
        rows: usize,
        k: usize,
        m: usize,
        x: []const f32,
        out: []f32,
    ) bool {
        if (self.device == null or self.compute_queue == null) return false;

        const pipe = switch (quant.base()) {
            .q4_0 => self.pipeline_matmul_q4_0,
            .q4_1 => self.pipeline_matmul_q4_1,
            .q8_0 => self.pipeline_matmul_q8_0,
            .q4_k => self.pipeline_matmul_q4_k,
            else => return false,
        };
        if (pipe == null) return false;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or m == 0 or k % block_elems != 0) return false;
        if (x.len != m * k or out.len < m * rows) return false;

        const num_blocks = k / block_elems;
        const row_bytes = num_blocks * quant.blockBytes();
        if (weight_bytes.len < rows * row_bytes) return false;

        const x_bytes_count = x.len * @sizeOf(f32);
        const y_bytes_count = m * rows * @sizeOf(f32);

        if (!self.ensureBuffer(&self.buffer_x, &self.memory_x, &self.size_x, &self.mapped_x, x_bytes_count)) return false;
        if (!self.ensureBuffer(&self.buffer_w, &self.memory_w, &self.size_w, &self.mapped_w, weight_bytes.len)) return false;
        if (!self.ensureBuffer(&self.buffer_y, &self.memory_y, &self.size_y, &self.mapped_y, y_bytes_count)) return false;

        self.updateDescriptorSet();

        if (!self.stageMapped(self.mapped_x, std.mem.sliceAsBytes(x))) return false;
        if (!self.stageMapped(self.mapped_w, weight_bytes)) return false;

        const tile = matmulTile(quant);
        return self.submitMatMul(pipe, m, rows, num_blocks, tile.x, tile.y, out);
    }

    /// Like `dispatchMatMul` but reads the weight from a persistent resident
    /// buffer uploaded once via `initResident`; only the activation is staged
    /// per call. This keeps the batched prefill from re-uploading weights the
    /// decode phase already kept in VRAM. Returns false when the shape does not
    /// fit the resident buffer or the device rejects the dispatch.
    pub fn dispatchMatMulResident(
        self: *VulkanPipeline,
        res: *Resident,
        quant: Quant,
        rows: usize,
        k: usize,
        m: usize,
        x: []const f32,
        out: []f32,
    ) bool {
        if (self.device == null or self.compute_queue == null) return false;

        const pipe = switch (quant.base()) {
            .q4_0 => self.pipeline_matmul_q4_0,
            .q4_1 => self.pipeline_matmul_q4_1,
            .q8_0 => self.pipeline_matmul_q8_0,
            .q4_k => self.pipeline_matmul_q4_k,
            else => return false,
        };
        if (pipe == null) return false;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or m == 0 or k % block_elems != 0) return false;
        if (x.len != m * k or out.len < m * rows) return false;

        const num_blocks = k / block_elems;
        const row_bytes = num_blocks * quant.blockBytes();
        if (res.bytes < rows * row_bytes) return false;

        const x_bytes_count = x.len * @sizeOf(f32);
        const y_bytes_count = m * rows * @sizeOf(f32);

        if (!self.ensureBuffer(&self.buffer_x, &self.memory_x, &self.size_x, &self.mapped_x, x_bytes_count)) return false;
        if (!self.ensureBuffer(&self.buffer_y, &self.memory_y, &self.size_y, &self.mapped_y, y_bytes_count)) return false;

        self.updateDescriptorSetWithWeight(vres(res).buffer, res.bytes);

        if (!self.stageMapped(self.mapped_x, std.mem.sliceAsBytes(x))) return false;

        const tile = matmulTile(quant);
        return self.submitMatMul(pipe, m, rows, num_blocks, tile.x, tile.y, out);
    }

    /// Stream a raw quantized weight to the GPU and compute `out[row] =
    /// dot(dequant(weight[row]), x)` for `rows` output rows. Returns false if
    /// the op is unsupported or the device rejects it (caller falls back).
    pub fn dispatchMatVec(
        self: *VulkanPipeline,
        quant: Quant,
        weight_bytes: []const u8,
        rows: usize,
        k: usize,
        x: []const f32,
        out: []f32,
    ) bool {
        if (self.device == null or self.compute_queue == null) return false;

        const pipe = gemvPipeline(self, quant);
        if (pipe == null) return false;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or x.len != k or out.len < rows) return false;
        if (k % block_elems != 0) return false;

        const num_blocks = k / block_elems;
        const row_bytes = num_blocks * quant.blockBytes();
        if (weight_bytes.len < rows * row_bytes) return false;

        const x_bytes_count = k * @sizeOf(f32);
        const y_bytes_count = rows * @sizeOf(f32);

        if (!self.ensureBuffer(&self.buffer_x, &self.memory_x, &self.size_x, &self.mapped_x, x_bytes_count)) return false;
        if (!self.ensureBuffer(&self.buffer_w, &self.memory_w, &self.size_w, &self.mapped_w, weight_bytes.len)) return false;
        if (!self.ensureBuffer(&self.buffer_y, &self.memory_y, &self.size_y, &self.mapped_y, y_bytes_count)) return false;

        self.updateDescriptorSet();

        if (!self.stageMapped(self.mapped_x, std.mem.sliceAsBytes(x))) return false;
        if (!self.stageMapped(self.mapped_w, weight_bytes)) return false;

        return self.submitGemv(pipe, quant, rows, num_blocks, out);
    }

    fn updateSlotDescriptor(self: *VulkanPipeline, s: *Slot, w_buffer: types.VkBuffer, w_range: u64) void {
        self.updateSlotDescriptorWithX(s, w_buffer, w_range, s.x_buffer, s.x_size);
    }

    /// Bind a slot's descriptor set with an explicit activation buffer (the
    /// shared burst `x`), so a run of GEMVs sharing one input reads the same
    /// device-local activation instead of each staging its own copy.
    fn updateSlotDescriptorWithX(self: *VulkanPipeline, s: *Slot, w_buffer: types.VkBuffer, w_range: u64, x_buffer: types.VkBuffer, x_range: u64) void {
        self.updateSlotDescriptorFull(s, x_buffer, x_range, w_buffer, w_range, w_buffer, w_range);
    }

    /// Bind a slot's full descriptor set. `w2` is the second weight of a fused
    /// `gate_up` GEMV (binding 3); single-weight ops pass `w` again so binding 3
    /// is always valid even though their shader never reads it.
    fn updateSlotDescriptorFull(self: *VulkanPipeline, s: *Slot, x_buffer: types.VkBuffer, x_range: u64, w_buffer: types.VkBuffer, w_range: u64, w2_buffer: types.VkBuffer, w2_range: u64) void {
        self.writeDescriptorFull(s.desc, x_buffer, x_range, w_buffer, w_range, s.y_buffer, s.y_size, w2_buffer, w2_range);
    }

    /// Write a full descriptor set (`x`/`w`/`y`/`w2` on bindings 0..3, binding 4
    /// mirrors `w2`). `y_buffer` is explicit so a two-kernel MLP can bind its
    /// `gate_up` dispatch's output to the device-local `ffn_mid` and its `down`
    /// dispatch's input to the same buffer, each on its own set.
    fn writeDescriptorFull(self: *VulkanPipeline, dst: types.VkDescriptorSet, x_buffer: types.VkBuffer, x_range: u64, w_buffer: types.VkBuffer, w_range: u64, y_buffer: types.VkBuffer, y_range: u64, w2_buffer: types.VkBuffer, w2_range: u64) void {
        const d_info_x = [_]types.VkDescriptorBufferInfo{.{ .buffer = x_buffer, .offset = 0, .range = x_range }};
        const d_info_w = [_]types.VkDescriptorBufferInfo{.{ .buffer = w_buffer, .offset = 0, .range = w_range }};
        const d_info_y = [_]types.VkDescriptorBufferInfo{.{ .buffer = y_buffer, .offset = 0, .range = y_range }};
        const d_info_w2 = [_]types.VkDescriptorBufferInfo{.{ .buffer = w2_buffer, .offset = 0, .range = w2_range }};

        const writes = [_]types.VkWriteDescriptorSet{
            .{ .dstSet = dst, .dstBinding = 0, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_x },
            .{ .dstSet = dst, .dstBinding = 1, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w },
            .{ .dstSet = dst, .dstBinding = 2, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_y },
            .{ .dstSet = dst, .dstBinding = 3, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w2 },
            .{ .dstSet = dst, .dstBinding = 4, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_w2 },
        };

        self.vkUpdateDescriptorSets.?(self.device, writes.len, &writes, 0, null);
    }

    /// Bind a slot's descriptor set for the fused MLP: `x` (the shared burst
    /// activation), the `gate`/`up` weights (bindings 1/3) and the `down` weight
    /// (binding 4). The `down` is the only three-weight shader's third operand.
    fn updateSlotDescriptorMlp(self: *VulkanPipeline, s: *Slot, x_buffer: types.VkBuffer, x_range: u64, gate_buffer: types.VkBuffer, gate_range: u64, up_buffer: types.VkBuffer, up_range: u64, down_buffer: types.VkBuffer, down_range: u64) void {
        const d_info_x = [_]types.VkDescriptorBufferInfo{.{ .buffer = x_buffer, .offset = 0, .range = x_range }};
        const d_info_g = [_]types.VkDescriptorBufferInfo{.{ .buffer = gate_buffer, .offset = 0, .range = gate_range }};
        const d_info_y = [_]types.VkDescriptorBufferInfo{.{ .buffer = s.y_buffer, .offset = 0, .range = s.y_size }};
        const d_info_u = [_]types.VkDescriptorBufferInfo{.{ .buffer = up_buffer, .offset = 0, .range = up_range }};
        const d_info_d = [_]types.VkDescriptorBufferInfo{.{ .buffer = down_buffer, .offset = 0, .range = down_range }};

        const writes = [_]types.VkWriteDescriptorSet{
            .{ .dstSet = s.desc, .dstBinding = 0, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_x },
            .{ .dstSet = s.desc, .dstBinding = 1, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_g },
            .{ .dstSet = s.desc, .dstBinding = 2, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_y },
            .{ .dstSet = s.desc, .dstBinding = 3, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_u },
            .{ .dstSet = s.desc, .dstBinding = 4, .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &d_info_d },
        };

        self.vkUpdateDescriptorSets.?(self.device, writes.len, &writes, 0, null);
    }

    /// Record a resident GEMV into the next ring slot, stage `x` and submit it
    /// with the slot's fence, without waiting. `out` is drained later by
    /// `waitHandle`. Returns null when the shape is unsupported or the ring
    /// cannot stage it.
    pub fn submitResidentGemv(self: *VulkanPipeline, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32) ?Handle {
        if (self.device == null or self.compute_queue == null) return null;
        const pipe = gemvPipeline(self, quant);
        if (pipe == null) return null;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or x.len != k or k % block_elems != 0) return null;
        const num_blocks = k / block_elems;
        if (res.bytes < rows * num_blocks * quant.blockBytes()) return null;

        const slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % RingSize;
        const s = &self.slots[slot];

        const x_bytes = k * @sizeOf(f32);
        const y_bytes = rows * @sizeOf(f32);
        // Prefer a device-local activation (staged through a host-visible buffer
        // per dispatch) so the shader reads `x` from VRAM, not over PCIe. Fall
        // back to a directly-mapped host-visible buffer without the copy commands.
        const device_local_x = self.vkCmdCopyBuffer != null and self.vkCmdPipelineBarrier != null;
        if (device_local_x) {
            if (!self.ensureDeviceBuffer(&s.x_buffer, &s.x_memory, &s.x_size, x_bytes)) return null;
            if (!self.ensureBuffer(&s.x_stage_buffer, &s.x_stage_memory, &s.x_stage_size, &s.x_stage_mapped, x_bytes)) return null;
            if (!self.stageMapped(s.x_stage_mapped, std.mem.sliceAsBytes(x))) return null;
        } else {
            if (!self.ensureBuffer(&s.x_buffer, &s.x_memory, &s.x_size, &s.x_mapped, x_bytes)) return null;
            if (!self.stageMapped(s.x_mapped, std.mem.sliceAsBytes(x))) return null;
        }
        if (!self.ensureBuffer(&s.y_buffer, &s.y_memory, &s.y_size, &s.y_mapped, y_bytes)) return null;

        self.updateSlotDescriptor(s, vres(res).buffer, res.bytes);

        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(s.cmd, &begin_info) != .SUCCESS) return null;
        if (device_local_x) {
            const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = x_bytes }};
            self.vkCmdCopyBuffer.?(s.cmd, s.x_stage_buffer, s.x_buffer, 1, &regions);
            const barriers = [_]types.VkBufferMemoryBarrier{.{
                .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
                .buffer = s.x_buffer,
                .offset = 0,
                .size = types.VK_WHOLE_SIZE,
            }};
            self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_TRANSFER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&barriers), 0, null);
        }
        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const dsets = [_]types.VkDescriptorSet{s.desc};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);
        const groups = gemvGroups(rows, gemvRowsPerGroup(quant));
        const push = PushConsts{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = groups.x,
            .act = 0,
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &push);
        self.vkCmdDispatch.?(s.cmd, groups.x, groups.y, 1);
        if (self.vkEndCommandBuffer.?(s.cmd) != .SUCCESS) return null;

        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&s.cmd),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, s.fence) != .SUCCESS) return null;
        return .{ .slot = slot };
    }

    /// Record a resident GEMV that reads the shared burst activation (`x`
    /// staged once by the burst's first member) into the next ring slot and
    /// submit it without waiting. `first` stages `x` into the shared
    /// device-local buffer and records the copy + transfer->shader barrier in
    /// this slot's command buffer; later members only record the dispatch. The
    /// barrier's second scope covers later submissions on the queue, so the
    /// whole burst reads the activation safely. Returns null when the shape is
    /// unsupported or the buffers cannot be staged.
    pub fn submitResidentGemvShared(self: *VulkanPipeline, res: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, first: bool) ?Handle {
        if (self.device == null or self.compute_queue == null) return null;
        const pipe = gemvPipeline(self, quant);
        if (pipe == null) return null;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or x.len != k or k % block_elems != 0) return null;
        const num_blocks = k / block_elems;
        if (res.bytes < rows * num_blocks * quant.blockBytes()) return null;

        const slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % RingSize;
        const s = &self.slots[slot];

        const x_bytes = k * @sizeOf(f32);
        const y_bytes = rows * @sizeOf(f32);
        if (!self.ensureDeviceBuffer(&self.burst_x, &self.burst_x_memory, &self.burst_x_size, x_bytes)) return null;
        if (first) {
            if (!self.ensureBuffer(&self.burst_x_stage, &self.burst_x_stage_memory, &self.burst_x_stage_size, &self.burst_x_stage_mapped, x_bytes)) return null;
            if (!self.stageMapped(self.burst_x_stage_mapped, std.mem.sliceAsBytes(x))) return null;
        }
        if (!self.ensureBuffer(&s.y_buffer, &s.y_memory, &s.y_size, &s.y_mapped, y_bytes)) return null;

        self.updateSlotDescriptorWithX(s, vres(res).buffer, res.bytes, self.burst_x, self.burst_x_size);

        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(s.cmd, &begin_info) != .SUCCESS) return null;
        if (first) {
            const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = x_bytes }};
            self.vkCmdCopyBuffer.?(s.cmd, self.burst_x_stage, self.burst_x, 1, &regions);
            const barriers = [_]types.VkBufferMemoryBarrier{.{
                .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
                .buffer = self.burst_x,
                .offset = 0,
                .size = types.VK_WHOLE_SIZE,
            }};
            self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_TRANSFER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&barriers), 0, null);
        }
        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const dsets = [_]types.VkDescriptorSet{s.desc};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);
        const groups = gemvGroups(rows, gemvRowsPerGroup(quant));
        const push = PushConsts{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = groups.x,
            .act = 0,
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &push);
        self.vkCmdDispatch.?(s.cmd, groups.x, groups.y, 1);
        if (self.vkEndCommandBuffer.?(s.cmd) != .SUCCESS) return null;

        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&s.cmd),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, s.fence) != .SUCCESS) return null;
        return .{ .slot = slot };
    }

    /// Record a fused `gate_up` resident GEMV (q4_0 or q8_0): one dispatch reads
    /// two resident weight buffers (`res_gate` at binding 1, `res_up` at
    /// binding 3) over the shared burst activation, applies the gated `act` and
    /// writes `rows` f32. `quant` selects the block layout (both weights share
    /// it); `variant` selects the shader variant (`plain`/`u4`, q4_0 only).
    /// Shares the burst activation/buffer machinery with
    /// `submitResidentGemvShared`. Returns null when unsupported.
    pub fn submitResidentGateUpShared(self: *VulkanPipeline, res_gate: *Resident, res_up: *Resident, quant: Quant, rows: usize, k: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.device == null or self.compute_queue == null) return null;
        const pipe = gateUpPipeline(self, quant, variant);
        if (pipe == null) return null;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or x.len != k or k % block_elems != 0) return null;
        const num_blocks = k / block_elems;
        const need = rows * num_blocks * quant.blockBytes();
        if (res_gate.bytes < need or res_up.bytes < need) return null;

        const slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % RingSize;
        const s = &self.slots[slot];

        const x_bytes = k * @sizeOf(f32);
        const y_bytes = rows * @sizeOf(f32);
        if (!self.ensureDeviceBuffer(&self.burst_x, &self.burst_x_memory, &self.burst_x_size, x_bytes)) return null;
        if (first) {
            if (!self.ensureBuffer(&self.burst_x_stage, &self.burst_x_stage_memory, &self.burst_x_stage_size, &self.burst_x_stage_mapped, x_bytes)) return null;
            if (!self.stageMapped(self.burst_x_stage_mapped, std.mem.sliceAsBytes(x))) return null;
        }
        if (!self.ensureBuffer(&s.y_buffer, &s.y_memory, &s.y_size, &s.y_mapped, y_bytes)) return null;

        self.updateSlotDescriptorFull(s, self.burst_x, self.burst_x_size, vres(res_gate).buffer, res_gate.bytes, vres(res_up).buffer, res_up.bytes);

        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(s.cmd, &begin_info) != .SUCCESS) return null;
        if (first) {
            const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = x_bytes }};
            self.vkCmdCopyBuffer.?(s.cmd, self.burst_x_stage, self.burst_x, 1, &regions);
            const barriers = [_]types.VkBufferMemoryBarrier{.{
                .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
                .buffer = self.burst_x,
                .offset = 0,
                .size = types.VK_WHOLE_SIZE,
            }};
            self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_TRANSFER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&barriers), 0, null);
        }
        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const dsets = [_]types.VkDescriptorSet{s.desc};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);
        const groups = gemvGroups(rows, gateUpRowsPerGroup(quant, variant));
        const push = PushConsts{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = groups.x,
            .act = @intFromEnum(act),
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &push);
        self.vkCmdDispatch.?(s.cmd, groups.x, groups.y, 1);
        if (self.vkEndCommandBuffer.?(s.cmd) != .SUCCESS) return null;

        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&s.cmd),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, s.fence) != .SUCCESS) return null;
        return .{ .slot = slot };
    }

    /// Record a fused MLP resident dispatch (q4_0/q4_1/q8_0): one dispatch reads
    /// the `gate`/`up` weights (`[k, ffn]`, bindings 1/3) and the `down` weight
    /// (`[ffn, rows]`, binding 4) over the shared burst activation, applies the
    /// gated `act`, keeps the `ffn_mid` in shared memory and writes `rows` f32.
    /// The whole SwiGLU MLP is one dispatch, so the intermediate never
    /// round-trips through global memory. Shares the burst activation machinery
    /// with `submitResidentGemvShared`. Returns null when unsupported.
    pub fn submitResidentMlpShared(self: *VulkanPipeline, res_gate: *Resident, res_up: *Resident, res_down: *Resident, quant: Quant, rows: usize, k: usize, ffn: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.device == null or self.compute_queue == null) return null;
        const pipe = mlpPipeline(self, quant, variant);
        if (pipe == null) return null;

        const block_elems = quant.blockElems();
        if (rows == 0 or k == 0 or ffn == 0 or x.len != k) return null;
        if (k % block_elems != 0 or ffn % block_elems != 0) return null;
        if (rows > max_mlp_out) return null;
        const num_blocks = k / block_elems;
        const down_blocks = ffn / block_elems;
        const gate_need = ffn * num_blocks * quant.blockBytes();
        const down_need = rows * down_blocks * quant.blockBytes();
        if (res_gate.bytes < gate_need or res_up.bytes < gate_need or res_down.bytes < down_need) return null;

        const slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % RingSize;
        const s = &self.slots[slot];

        const x_bytes = k * @sizeOf(f32);
        const y_bytes = rows * @sizeOf(f32);
        if (!self.ensureDeviceBuffer(&self.burst_x, &self.burst_x_memory, &self.burst_x_size, x_bytes)) return null;
        if (first) {
            if (!self.ensureBuffer(&self.burst_x_stage, &self.burst_x_stage_memory, &self.burst_x_stage_size, &self.burst_x_stage_mapped, x_bytes)) return null;
            if (!self.stageMapped(self.burst_x_stage_mapped, std.mem.sliceAsBytes(x))) return null;
        }
        if (!self.ensureBuffer(&s.y_buffer, &s.y_memory, &s.y_size, &s.y_mapped, y_bytes)) return null;

        self.updateSlotDescriptorMlp(s, self.burst_x, self.burst_x_size, vres(res_gate).buffer, res_gate.bytes, vres(res_up).buffer, res_up.bytes, vres(res_down).buffer, res_down.bytes);

        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(s.cmd, &begin_info) != .SUCCESS) return null;
        if (first) {
            const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = x_bytes }};
            self.vkCmdCopyBuffer.?(s.cmd, self.burst_x_stage, self.burst_x, 1, &regions);
            const barriers = [_]types.VkBufferMemoryBarrier{.{
                .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
                .buffer = self.burst_x,
                .offset = 0,
                .size = types.VK_WHOLE_SIZE,
            }};
            self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_TRANSFER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&barriers), 0, null);
        }
        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const dsets = [_]types.VkDescriptorSet{s.desc};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &dsets, 0, null);
        const push = PushConstsMlp{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = 1,
            .act = @intFromEnum(act),
            .ffn_rows = @intCast(ffn),
            .down_blocks_per_row = @intCast(down_blocks),
            .pad0 = 0,
            .pad1 = 0,
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstsMlp), &push);
        self.vkCmdDispatch.?(s.cmd, 1, 1, 1);
        if (self.vkEndCommandBuffer.?(s.cmd) != .SUCCESS) return null;

        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&s.cmd),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, s.fence) != .SUCCESS) return null;
        return .{ .slot = slot };
    }

    /// Record the whole SwiGLU MLP as **two multi-workgroup dispatches** in one
    /// command buffer on one device: the fused `gate_up` (gate/up + gated
    /// activation) reads the shared burst `x` and writes the `ffn_mid` into a
    /// device-local buffer; a shader-write -> shader-read barrier makes it
    /// visible; then the `down` GEMV reads the `ffn_mid` and writes the slot's
    /// `y`. Both kernels keep full workgroup parallelism and the intermediate
    /// never leaves VRAM, so on a discrete (PCIe) device the MLP is competitive
    /// (the single-workgroup fused shader is not). Shares the burst activation
    /// machinery with `submitResidentGemvShared`. Returns null when unsupported.
    pub fn submitResidentMlpPairShared(self: *VulkanPipeline, res_gate: *Resident, res_up: *Resident, res_down: *Resident, gu_quant: Quant, down_quant: Quant, rows: usize, k: usize, ffn: usize, x: []const f32, act: GateUpAct, variant: GpuVariant, first: bool) ?Handle {
        if (self.device == null or self.compute_queue == null) return null;
        const gu_pipe = gateUpPipeline(self, gu_quant, variant);
        if (gu_pipe == null) return null;
        const dn_q = down_quant.base().variant(variant);
        const down_pipe = gemvPipeline(self, dn_q);
        if (down_pipe == null) return null;

        const gu_block = gu_quant.blockElems();
        const dn_block = down_quant.blockElems();
        if (rows == 0 or k == 0 or ffn == 0 or x.len != k) return null;
        if (k % gu_block != 0 or ffn % dn_block != 0) return null;
        const num_blocks = k / gu_block;
        const down_blocks = ffn / dn_block;
        const gate_need = ffn * num_blocks * gu_quant.blockBytes();
        const down_need = rows * down_blocks * down_quant.blockBytes();
        if (res_gate.bytes < gate_need or res_up.bytes < gate_need or res_down.bytes < down_need) return null;

        const slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % RingSize;
        const s = &self.slots[slot];

        const x_bytes = k * @sizeOf(f32);
        const mid_bytes = ffn * @sizeOf(f32);
        const y_bytes = rows * @sizeOf(f32);
        if (!self.ensureDeviceBuffer(&self.burst_x, &self.burst_x_memory, &self.burst_x_size, x_bytes)) return null;
        if (!self.ensureDeviceBuffer(&self.burst_mid, &self.burst_mid_memory, &self.burst_mid_size, mid_bytes)) return null;
        if (first) {
            if (!self.ensureBuffer(&self.burst_x_stage, &self.burst_x_stage_memory, &self.burst_x_stage_size, &self.burst_x_stage_mapped, x_bytes)) return null;
            if (!self.stageMapped(self.burst_x_stage_mapped, std.mem.sliceAsBytes(x))) return null;
        }
        if (!self.ensureBuffer(&s.y_buffer, &s.y_memory, &s.y_size, &s.y_mapped, y_bytes)) return null;

        self.writeDescriptorFull(s.desc, self.burst_x, self.burst_x_size, vres(res_gate).buffer, res_gate.bytes, self.burst_mid, self.burst_mid_size, vres(res_up).buffer, res_up.bytes);
        self.writeDescriptorFull(s.desc2, self.burst_mid, self.burst_mid_size, vres(res_down).buffer, res_down.bytes, s.y_buffer, s.y_size, vres(res_down).buffer, res_down.bytes);

        const begin_info = types.VkCommandBufferBeginInfo{};
        if (self.vkBeginCommandBuffer.?(s.cmd, &begin_info) != .SUCCESS) return null;
        if (first) {
            const regions = [_]types.VkBufferCopy{.{ .srcOffset = 0, .dstOffset = 0, .size = x_bytes }};
            self.vkCmdCopyBuffer.?(s.cmd, self.burst_x_stage, self.burst_x, 1, &regions);
            const barriers = [_]types.VkBufferMemoryBarrier{.{
                .srcAccessMask = types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
                .buffer = self.burst_x,
                .offset = 0,
                .size = types.VK_WHOLE_SIZE,
            }};
            self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_TRANSFER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&barriers), 0, null);
        }

        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, gu_pipe);
        const gu_dsets = [_]types.VkDescriptorSet{s.desc};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &gu_dsets, 0, null);
        const gu_groups = gemvGroups(ffn, gateUpRowsPerGroup(gu_quant, variant));
        const gu_push = PushConsts{
            .num_rows = @intCast(ffn),
            .num_blocks_per_row = @intCast(num_blocks),
            .groups_x = gu_groups.x,
            .act = @intFromEnum(act),
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &gu_push);
        self.vkCmdDispatch.?(s.cmd, gu_groups.x, gu_groups.y, 1);

        const mid_barriers = [_]types.VkBufferMemoryBarrier{.{
            .srcAccessMask = types.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
            .buffer = self.burst_mid,
            .offset = 0,
            .size = types.VK_WHOLE_SIZE,
        }};
        self.vkCmdPipelineBarrier.?(s.cmd, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, @ptrCast(&mid_barriers), 0, null);

        self.vkCmdBindPipeline.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, down_pipe);
        const down_dsets = [_]types.VkDescriptorSet{s.desc2};
        self.vkCmdBindDescriptorSets.?(s.cmd, types.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &down_dsets, 0, null);
        const down_groups = gemvGroups(rows, gemvRowsPerGroup(dn_q));
        const down_push = PushConsts{
            .num_rows = @intCast(rows),
            .num_blocks_per_row = @intCast(down_blocks),
            .groups_x = down_groups.x,
            .act = 0,
        };
        self.vkCmdPushConstants.?(s.cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConsts), &down_push);
        self.vkCmdDispatch.?(s.cmd, down_groups.x, down_groups.y, 1);
        if (self.vkEndCommandBuffer.?(s.cmd) != .SUCCESS) return null;

        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        const submits = [_]types.VkSubmitInfo{
            .{
                .commandBufferCount = 1,
                .pCommandBuffers = @ptrCast(&s.cmd),
            },
        };
        if (self.vkQueueSubmit.?(self.compute_queue, 1, &submits, s.fence) != .SUCCESS) return null;
        return .{ .slot = slot };
    }

    /// Wait `h`'s fence and copy its `rows`-f32 result into `out`, then reset
    /// the slot's fence for reuse.
    pub fn waitHandle(self: *VulkanPipeline, h: Handle, out: []f32, rows: usize) bool {
        if (self.device == null) return false;
        const s = &self.slots[h.slot];
        if (self.vkWaitForFences.?(self.device, 1, @ptrCast(&s.fence), types.VK_TRUE, std.math.maxInt(u64)) != .SUCCESS) return false;
        const mapped = s.y_mapped orelse return false;
        @memcpy(std.mem.sliceAsBytes(out[0..rows]), @as([*]const u8, @ptrCast(mapped))[0 .. rows * @sizeOf(f32)]);
        _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        return true;
    }

    fn drainHandles(self: *VulkanPipeline, handles: []const Handle) void {
        for (handles) |h| {
            const s = &self.slots[h.slot];
            _ = self.vkWaitForFences.?(self.device, 1, @ptrCast(&s.fence), types.VK_TRUE, std.math.maxInt(u64));
            _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        }
    }

    /// Submit a run of resident GEMVs that all read the same `x` and drain them
    /// with **one** fence wait. This is the decode overlap win: e.g. the q/k/v
    /// projections of a layer share the normed activation, so N dispatches pay
    /// one submission wait instead of N `submit + wait` round-trips. The
    /// dispatches stay serialized on the queue but independent (distinct output
    /// buffers), so their results are correct regardless of completion order.
    pub fn dispatchMatVecResidentBatch(self: *VulkanPipeline, k: usize, x: []const f32, ops: []const ResidentOp) bool {
        if (self.device == null or self.compute_queue == null) return false;
        if (ops.len == 0 or ops.len > RingSize or x.len != k) return false;

        var handles: [RingSize]Handle = undefined;
        var n: usize = 0;
        for (ops) |op| {
            if (op.rows == 0 or op.out.len < op.rows) {
                self.drainHandles(handles[0..n]);
                return false;
            }
            const h = self.submitResidentGemv(op.res, op.quant, op.rows, k, x) orelse {
                self.drainHandles(handles[0..n]);
                return false;
            };
            handles[n] = h;
            n += 1;
        }

        var fences: [RingSize]types.VkFence = undefined;
        for (handles[0..n], 0..) |h, i| fences[i] = self.slots[h.slot].fence;
        if (self.vkWaitForFences.?(self.device, @intCast(n), &fences, types.VK_TRUE, std.math.maxInt(u64)) != .SUCCESS) {
            self.drainHandles(handles[0..n]);
            return false;
        }
        for (handles[0..n], 0..) |h, i| {
            const s = &self.slots[h.slot];
            const mapped = s.y_mapped orelse return false;
            @memcpy(std.mem.sliceAsBytes(ops[i].out[0..ops[i].rows]), @as([*]const u8, @ptrCast(mapped))[0 .. ops[i].rows * @sizeOf(f32)]);
            _ = self.vkResetFences.?(self.device, 1, @ptrCast(&s.fence));
        }
        return true;
    }

    /// Like `dispatchMatVec` but reads the weight from a persistent resident
    /// buffer uploaded once via `initResident`; only the activation and result
    /// are staged per call. Asynchronous under the hood: submitted on the ring
    /// and drained immediately, so the streamed and resident paths share the
    /// fence machinery. Returns false when the shape does not fit the resident
    /// buffer or the device rejects the dispatch (caller falls back).
    pub fn dispatchMatVecResident(
        self: *VulkanPipeline,
        res: *Resident,
        quant: Quant,
        rows: usize,
        k: usize,
        x: []const f32,
        out: []f32,
    ) bool {
        const h = self.submitResidentGemv(res, quant, rows, k, x) orelse return false;
        return self.waitHandle(h, out, rows);
    }

    /// Block until the compute queue is idle. Dispatches already wait, but the
    /// runtime may call this to fence a batch of offloaded work.
    pub fn sync(self: *VulkanPipeline) void {
        if (self.compute_queue == null) return;
        if (self.vkQueueWaitIdle) |f| _ = f(self.compute_queue);
    }
};

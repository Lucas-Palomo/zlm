const std = @import("std");

pub const VkResult = enum(i32) {
    SUCCESS = 0,
    NOT_READY = 1,
    TIMEOUT = 2,
    EVENT_SET = 3,
    EVENT_RESET = 4,
    INCOMPLETE = 5,
    ERROR_OUT_OF_HOST_MEMORY = -1,
    ERROR_OUT_OF_DEVICE_MEMORY = -2,
    ERROR_INITIALIZATION_FAILED = -3,
    ERROR_DEVICE_LOST = -4,
    ERROR_MEMORY_MAP_FAILED = -5,
    ERROR_LAYER_NOT_PRESENT = -6,
    ERROR_EXTENSION_NOT_PRESENT = -7,
    ERROR_FEATURE_NOT_PRESENT = -8,
    ERROR_INCOMPATIBLE_DRIVER = -9,
    ERROR_TOO_MANY_OBJECTS = -10,
    ERROR_FORMAT_NOT_SUPPORTED = -11,
    ERROR_FRAGMENTED_POOL = -12,
    ERROR_UNKNOWN = -13,
    _,
};

pub const VkStructureType = enum(i32) {
    APPLICATION_INFO = 0,
    INSTANCE_CREATE_INFO = 1,
    DEVICE_QUEUE_CREATE_INFO = 2,
    DEVICE_CREATE_INFO = 3,
    SUBMIT_INFO = 4,
    MEMORY_ALLOCATE_INFO = 5,
    BUFFER_CREATE_INFO = 12,
    SHADER_MODULE_CREATE_INFO = 16,
    COMPUTE_PIPELINE_CREATE_INFO = 29,
    PIPELINE_LAYOUT_CREATE_INFO = 30,
    FENCE_CREATE_INFO = 8,
    DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    DESCRIPTOR_POOL_CREATE_INFO = 33,
    DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    WRITE_DESCRIPTOR_SET = 35,
    COMMAND_POOL_CREATE_INFO = 39,
    COMMAND_BUFFER_ALLOCATE_INFO = 40,
    COMMAND_BUFFER_BEGIN_INFO = 42,
    BUFFER_MEMORY_BARRIER = 44,
    PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    _,
};

pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};
pub const VkCommandPool = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkBuffer = ?*opaque {};
pub const VkDeviceMemory = ?*opaque {};
pub const VkShaderModule = ?*opaque {};
pub const VkPipeline = ?*opaque {};
pub const VkPipelineLayout = ?*opaque {};
pub const VkDescriptorSetLayout = ?*opaque {};
pub const VkDescriptorPool = ?*opaque {};
pub const VkDescriptorSet = ?*opaque {};
pub const VkFence = ?*opaque {};

/// `VkFenceCreateInfo.flags`: the fence starts signaled, so the first reuse of a
/// ring slot does not wait a fence that was never submitted.
pub const VK_FENCE_CREATE_SIGNALED_BIT: u32 = 0x00000001;
/// `vkWaitForFences(..., waitAll, ...)`.
pub const VK_TRUE: u32 = 1;

pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

/// `VkPipelineStageFlagBits`: source/destination stages for a barrier.
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x00001000;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;
/// `VkAccessFlagBits`: the transfer-write -> compute-shader-read barrier that
/// makes a staged, device-local resident upload visible to the GEMV shaders.
pub const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x00001000;
pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;
/// `VkAccessFlagBits`: the compute-shader-write -> compute-shader-read barrier
/// that makes one dispatch's device-local output visible to the next (the
/// two-kernel MLP's `ffn_mid`).
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000008;
pub const VK_QUEUE_FAMILY_IGNORED: u32 = ~@as(u32, 0);
pub const VK_WHOLE_SIZE: u64 = ~@as(u64, 0);

pub const VkApplicationInfo = extern struct {
    sType: VkStructureType = .APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: [*:0]const u8 = "zlm",
    applicationVersion: u32 = 1,
    pEngineName: [*:0]const u8 = "zlm_gpu",
    engineVersion: u32 = 1,
    apiVersion: u32 = (1 << 22) | (0 << 12) | 0,
};

pub const VkInstanceCreateInfo = extern struct {
    sType: VkStructureType = .INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: VkStructureType = .DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: [*]const f32,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: VkStructureType = .DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: [*]const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

pub const VkExtent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: VkExtent3D,
};

pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: u64,
    sparseAddressSpaceSize: u64,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: u64,
    minUniformBufferOffsetAlignment: u64,
    minStorageBufferOffsetAlignment: u64,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: u32,
    framebufferDepthSampleCounts: u32,
    framebufferStencilSampleCounts: u32,
    framebufferNoAttachmentsSampleCounts: u32,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: u32,
    sampledImageIntegerSampleCounts: u32,
    sampledImageDepthSampleCounts: u32,
    sampledImageStencilSampleCounts: u32,
    storageImageSampleCounts: u32,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: u32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: u32,
    standardSampleLocations: u32,
    optimalBufferCopyOffsetAlignment: u64,
    optimalBufferCopyRowPitchAlignment: u64,
    nonCoherentAtomSize: u64,
};

pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: u32,
    residencyStandard2DMultisampleBlockShape: u32,
    residencyStandard3DBlockShape: u32,
    residencyAlignedMipSize: u32,
    residencyNonAlignedMipSize: u32,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: u32,
    deviceName: [VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    pipelineCacheUUID: [VK_UUID_SIZE]u8,
    limits: VkPhysicalDeviceLimits,
    sparseProperties: VkPhysicalDeviceSparseProperties,
};

pub const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

pub const VkMemoryHeap = extern struct {
    size: u64,
    flags: u32,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap,
};

pub const VkMemoryRequirements = extern struct {
    size: u64,
    alignment: u64,
    memoryTypeBits: u32,
};

pub const VkMemoryAllocateInfo = extern struct {
    sType: VkStructureType = .MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: u64,
    memoryTypeIndex: u32,
};

pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType = .BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64,
    usage: u32,
    sharingMode: u32 = 0,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*const u32 = null,
};

pub const VkBufferCopy = extern struct {
    srcOffset: u64,
    dstOffset: u64,
    size: u64,
};

pub const VkBufferMemoryBarrier = extern struct {
    sType: VkStructureType = .BUFFER_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32,
    dstAccessMask: u32,
    srcQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    buffer: VkBuffer,
    offset: u64,
    size: u64,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: VkStructureType = .SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: VkStructureType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: [*]const VkDescriptorSetLayoutBinding,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: u32,
    offset: u32,
    size: u32,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: VkStructureType = .PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32,
    pSetLayouts: [*]const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: [*]const VkPushConstantRange,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: VkStructureType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32,
    module: VkShaderModule,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

/// One specialization constant: the shader's `constant_id`, its byte offset in
/// the specialization data and its size. Lets one shader become N pipelines.
pub const VkSpecializationMapEntry = extern struct {
    constantID: u32,
    offset: u32,
    size: usize,
};

pub const VkSpecializationInfo = extern struct {
    mapEntryCount: u32,
    pMapEntries: [*]const VkSpecializationMapEntry,
    dataSize: usize,
    pData: *const anyopaque,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: VkStructureType = .COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

pub const VkDescriptorPoolSize = extern struct {
    type: u32,
    descriptorCount: u32,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: VkStructureType = .DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: [*]const VkDescriptorPoolSize,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: VkStructureType = .DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: [*]const VkDescriptorSetLayout,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: u64,
    range: u64,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: VkStructureType = .WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32 = 0,
    descriptorCount: u32,
    descriptorType: u32,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?[*]const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: VkStructureType = .COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: VkStructureType = .COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool,
    level: u32 = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32 = 1,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: VkStructureType = .COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: VkStructureType = .FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};

pub const VkSubmitInfo = extern struct {
    sType: VkStructureType = .SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const u32 = null,
    commandBufferCount: u32 = 1,
    pCommandBuffers: [*]const VkCommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

pub const PFN_vkGetInstanceProcAddr = *const fn (instance: VkInstance, pName: [*:0]const u8) callconv(.c) ?*const anyopaque;
pub const PFN_vkCreateInstance = *const fn (pCreateInfo: *const VkInstanceCreateInfo, pAllocator: ?*const anyopaque, pInstance: *VkInstance) callconv(.c) VkResult;
pub const PFN_vkDestroyInstance = *const fn (instance: VkInstance, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkEnumeratePhysicalDevices = *const fn (instance: VkInstance, pPhysicalDeviceCount: *u32, pPhysicalDevices: ?[*]VkPhysicalDevice) callconv(.c) VkResult;
pub const PFN_vkGetPhysicalDeviceProperties = *const fn (physicalDevice: VkPhysicalDevice, pProperties: *VkPhysicalDeviceProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceQueueFamilyProperties = *const fn (physicalDevice: VkPhysicalDevice, pQueueFamilyPropertyCount: *u32, pQueueFamilyProperties: ?[*]VkQueueFamilyProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceMemoryProperties = *const fn (physicalDevice: VkPhysicalDevice, pMemoryProperties: *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
pub const PFN_vkCreateDevice = *const fn (physicalDevice: VkPhysicalDevice, pCreateInfo: *const VkDeviceCreateInfo, pAllocator: ?*const anyopaque, pDevice: *VkDevice) callconv(.c) VkResult;
pub const PFN_vkDestroyDevice = *const fn (device: VkDevice, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkGetDeviceQueue = *const fn (device: VkDevice, queueFamilyIndex: u32, queueIndex: u32, pQueue: *VkQueue) callconv(.c) void;

pub const PFN_vkCreateShaderModule = *const fn (device: VkDevice, pCreateInfo: *const VkShaderModuleCreateInfo, pAllocator: ?*const anyopaque, pShaderModule: *VkShaderModule) callconv(.c) VkResult;
pub const PFN_vkDestroyShaderModule = *const fn (device: VkDevice, shaderModule: VkShaderModule, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkCreateDescriptorSetLayout = *const fn (device: VkDevice, pCreateInfo: *const VkDescriptorSetLayoutCreateInfo, pAllocator: ?*const anyopaque, pSetLayout: *VkDescriptorSetLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorSetLayout = *const fn (device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkCreatePipelineLayout = *const fn (device: VkDevice, pCreateInfo: *const VkPipelineLayoutCreateInfo, pAllocator: ?*const anyopaque, pPipelineLayout: *VkPipelineLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyPipelineLayout = *const fn (device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkCreateComputePipelines = *const fn (device: VkDevice, pipelineCache: ?*anyopaque, createInfoCount: u32, pCreateInfos: [*]const VkComputePipelineCreateInfo, pAllocator: ?*const anyopaque, pPipelines: [*]VkPipeline) callconv(.c) VkResult;
pub const PFN_vkDestroyPipeline = *const fn (device: VkDevice, pipeline: VkPipeline, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkCreateDescriptorPool = *const fn (device: VkDevice, pCreateInfo: *const VkDescriptorPoolCreateInfo, pAllocator: ?*const anyopaque, pDescriptorPool: *VkDescriptorPool) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorPool = *const fn (device: VkDevice, descriptorPool: VkDescriptorPool, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkAllocateDescriptorSets = *const fn (device: VkDevice, pAllocateInfo: *const VkDescriptorSetAllocateInfo, pDescriptorSets: [*]VkDescriptorSet) callconv(.c) VkResult;
pub const PFN_vkUpdateDescriptorSets = *const fn (device: VkDevice, descriptorWriteCount: u32, pDescriptorWrites: [*]const VkWriteDescriptorSet, descriptorCopyCount: u32, pDescriptorCopies: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkCreateBuffer = *const fn (device: VkDevice, pCreateInfo: *const VkBufferCreateInfo, pAllocator: ?*const anyopaque, pBuffer: *VkBuffer) callconv(.c) VkResult;
pub const PFN_vkDestroyBuffer = *const fn (device: VkDevice, buffer: VkBuffer, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkGetBufferMemoryRequirements = *const fn (device: VkDevice, buffer: VkBuffer, pMemoryRequirements: *VkMemoryRequirements) callconv(.c) void;

pub const PFN_vkAllocateMemory = *const fn (device: VkDevice, pAllocateInfo: *const VkMemoryAllocateInfo, pAllocator: ?*const anyopaque, pMemory: *VkDeviceMemory) callconv(.c) VkResult;
pub const PFN_vkFreeMemory = *const fn (device: VkDevice, memory: VkDeviceMemory, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkBindBufferMemory = *const fn (device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, memoryOffset: u64) callconv(.c) VkResult;

pub const PFN_vkMapMemory = *const fn (device: VkDevice, memory: VkDeviceMemory, offset: u64, size: u64, flags: u32, ppData: *?*anyopaque) callconv(.c) VkResult;
pub const PFN_vkUnmapMemory = *const fn (device: VkDevice, memory: VkDeviceMemory) callconv(.c) void;

pub const PFN_vkCreateCommandPool = *const fn (device: VkDevice, pCreateInfo: *const VkCommandPoolCreateInfo, pAllocator: ?*const anyopaque, pCommandPool: *VkCommandPool) callconv(.c) VkResult;
pub const PFN_vkDestroyCommandPool = *const fn (device: VkDevice, commandPool: VkCommandPool, pAllocator: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkAllocateCommandBuffers = *const fn (device: VkDevice, pAllocateInfo: *const VkCommandBufferAllocateInfo, pCommandBuffers: [*]VkCommandBuffer) callconv(.c) VkResult;
pub const PFN_vkFreeCommandBuffers = *const fn (device: VkDevice, commandPool: VkCommandPool, commandBufferCount: u32, pCommandBuffers: [*]const VkCommandBuffer) callconv(.c) void;

pub const PFN_vkBeginCommandBuffer = *const fn (commandBuffer: VkCommandBuffer, pBeginInfo: *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub const PFN_vkEndCommandBuffer = *const fn (commandBuffer: VkCommandBuffer) callconv(.c) VkResult;

pub const PFN_vkCmdBindPipeline = *const fn (commandBuffer: VkCommandBuffer, pipelineBindPoint: u32, pipeline: VkPipeline) callconv(.c) void;
pub const PFN_vkCmdBindDescriptorSets = *const fn (commandBuffer: VkCommandBuffer, pipelineBindPoint: u32, layout: VkPipelineLayout, firstSet: u32, descriptorSetCount: u32, pDescriptorSets: [*]const VkDescriptorSet, dynamicOffsetCount: u32, pDynamicOffsets: ?*const u32) callconv(.c) void;
pub const PFN_vkCmdPushConstants = *const fn (commandBuffer: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: u32, offset: u32, size: u32, pValues: *const anyopaque) callconv(.c) void;
pub const PFN_vkCmdDispatch = *const fn (commandBuffer: VkCommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) callconv(.c) void;
pub const PFN_vkCmdCopyBuffer = *const fn (commandBuffer: VkCommandBuffer, srcBuffer: VkBuffer, dstBuffer: VkBuffer, regionCount: u32, pRegions: [*]const VkBufferCopy) callconv(.c) void;
pub const PFN_vkCmdPipelineBarrier = *const fn (commandBuffer: VkCommandBuffer, srcStageMask: u32, dstStageMask: u32, dependencyFlags: u32, memoryBarrierCount: u32, pMemoryBarriers: ?*const anyopaque, bufferMemoryBarrierCount: u32, pBufferMemoryBarriers: ?*const VkBufferMemoryBarrier, imageMemoryBarrierCount: u32, pImageMemoryBarriers: ?*const anyopaque) callconv(.c) void;

pub const PFN_vkQueueSubmit = *const fn (queue: VkQueue, submitCount: u32, pSubmits: [*]const VkSubmitInfo, fence: ?*anyopaque) callconv(.c) VkResult;
pub const PFN_vkQueueWaitIdle = *const fn (queue: VkQueue) callconv(.c) VkResult;

pub const PFN_vkCreateFence = *const fn (device: VkDevice, pCreateInfo: *const VkFenceCreateInfo, pAllocator: ?*const anyopaque, pFence: *VkFence) callconv(.c) VkResult;
pub const PFN_vkDestroyFence = *const fn (device: VkDevice, fence: VkFence, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkResetFences = *const fn (device: VkDevice, fenceCount: u32, pFences: [*]const VkFence) callconv(.c) VkResult;
pub const PFN_vkWaitForFences = *const fn (device: VkDevice, fenceCount: u32, pFences: [*]const VkFence, waitAll: u32, timeout: u64) callconv(.c) VkResult;

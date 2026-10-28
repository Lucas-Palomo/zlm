const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const types = @import("types.zig");

pub const VulkanStatus = enum {
    off,
    vulkan,
};

/// A live Vulkan instance/device/queue, loaded dynamically (no link-time
/// dependency on `libvulkan`). `status == .off` means no usable GPU.
pub const VulkanContext = struct {
    lib: ?std.DynLib = null,
    instance: types.VkInstance = null,
    physical_device: types.VkPhysicalDevice = null,
    device: types.VkDevice = null,
    compute_queue: types.VkQueue = null,
    queue_family_index: u32 = 0,
    status: VulkanStatus = .off,

    device_name: [256]u8 = [_]u8{0} ** 256,
    device_name_len: usize = 0,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    driver_version: u32 = 0,
    api_version: u32 = 0,
    device_type: u32 = 0,
    is_uma: bool = false,

    vkDestroyInstance: ?types.PFN_vkDestroyInstance = null,
    vkEnumeratePhysicalDevices: ?types.PFN_vkEnumeratePhysicalDevices = null,
    vkGetPhysicalDeviceProperties: ?types.PFN_vkGetPhysicalDeviceProperties = null,
    vkGetPhysicalDeviceQueueFamilyProperties: ?types.PFN_vkGetPhysicalDeviceQueueFamilyProperties = null,
    vkCreateDevice: ?types.PFN_vkCreateDevice = null,
    vkDestroyDevice: ?types.PFN_vkDestroyDevice = null,
    vkGetDeviceQueue: ?types.PFN_vkGetDeviceQueue = null,

    /// Never fails: on any problem (no library, no device, no compute queue) the
    /// returned context has `status == .off`.
    pub fn init(allocator: std.mem.Allocator, io: Io) VulkanContext {
        _ = allocator;
        var ctx = VulkanContext{};

        const lib_names = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"vulkan-1.dll"},
            .macos => &[_][]const u8{ "libvulkan.1.dylib", "libvulkan.dylib" },
            else => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
        };

        var loaded_lib: ?std.DynLib = null;
        for (lib_names) |lib_name| {
            if (std.DynLib.open(lib_name)) |l| {
                loaded_lib = l;
                break;
            } else |_| {}
        }

        if (loaded_lib == null) {
            _ = io;
            return ctx;
        }

        var lib = loaded_lib.?;
        ctx.lib = lib;

        const vkCreateInstanceFn = lib.lookup(types.PFN_vkCreateInstance, "vkCreateInstance") orelse {
            ctx.deinit();
            return VulkanContext{};
        };

        ctx.vkDestroyInstance = lib.lookup(types.PFN_vkDestroyInstance, "vkDestroyInstance");
        ctx.vkEnumeratePhysicalDevices = lib.lookup(types.PFN_vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices");
        ctx.vkGetPhysicalDeviceProperties = lib.lookup(types.PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
        ctx.vkGetPhysicalDeviceQueueFamilyProperties = lib.lookup(types.PFN_vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties");
        ctx.vkCreateDevice = lib.lookup(types.PFN_vkCreateDevice, "vkCreateDevice");
        ctx.vkDestroyDevice = lib.lookup(types.PFN_vkDestroyDevice, "vkDestroyDevice");
        ctx.vkGetDeviceQueue = lib.lookup(types.PFN_vkGetDeviceQueue, "vkGetDeviceQueue");

        if (ctx.vkEnumeratePhysicalDevices == null or ctx.vkGetPhysicalDeviceProperties == null or ctx.vkGetPhysicalDeviceQueueFamilyProperties == null or ctx.vkCreateDevice == null or ctx.vkGetDeviceQueue == null) {
            ctx.deinit();
            return VulkanContext{};
        }

        // Request Vulkan 1.1 so the subgroup-reduction shader is valid; fall
        // back to 1.0 when the loader/device cannot (the subgroup pipeline is
        // then pruned in `pipeline.zig`).
        var app_info = types.VkApplicationInfo{};
        app_info.apiVersion = (1 << 22) | (1 << 12) | 0;
        const instance_info = types.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
        };

        var instance: types.VkInstance = null;
        if (vkCreateInstanceFn(&instance_info, null, &instance) != .SUCCESS or instance == null) {
            app_info.apiVersion = (1 << 22) | 0;
            if (vkCreateInstanceFn(&instance_info, null, &instance) != .SUCCESS or instance == null) {
                ctx.deinit();
                return VulkanContext{};
            }
        }
        ctx.instance = instance;

        var device_count: u32 = 0;
        if (ctx.vkEnumeratePhysicalDevices.?(instance, &device_count, null) != .SUCCESS or device_count == 0) {
            ctx.deinit();
            return VulkanContext{};
        }

        var devices: [16]types.VkPhysicalDevice = undefined;
        const actual_count = @min(device_count, devices.len);
        if (ctx.vkEnumeratePhysicalDevices.?(instance, &device_count, &devices) != .SUCCESS) {
            ctx.deinit();
            return VulkanContext{};
        }

        var selected_physical: types.VkPhysicalDevice = null;
        var selected_queue_family: u32 = 0;
        var selected_props_buf: [4096]u8 align(8) = [_]u8{0} ** 4096;
        var found_discrete = false;

        for (devices[0..actual_count]) |pdev| {
            var props_buf: [4096]u8 align(8) = [_]u8{0} ** 4096;
            ctx.vkGetPhysicalDeviceProperties.?(pdev, @ptrCast(&props_buf));

            const dev_type = std.mem.readInt(u32, props_buf[16..20], .little);

            var queue_count: u32 = 0;
            ctx.vkGetPhysicalDeviceQueueFamilyProperties.?(pdev, &queue_count, null);
            if (queue_count == 0) continue;

            var queue_props: [16]types.VkQueueFamilyProperties = undefined;
            const actual_qcount = @min(queue_count, queue_props.len);
            ctx.vkGetPhysicalDeviceQueueFamilyProperties.?(pdev, &queue_count, &queue_props);

            for (queue_props[0..actual_qcount], 0..) |qp, qidx| {
                if ((qp.queueFlags & types.VK_QUEUE_COMPUTE_BIT) != 0) {
                    const is_discrete = dev_type == types.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                    if (selected_physical == null or (is_discrete and !found_discrete)) {
                        selected_physical = pdev;
                        selected_queue_family = @intCast(qidx);
                        selected_props_buf = props_buf;
                        found_discrete = is_discrete;
                    }
                    break;
                }
            }
        }

        if (selected_physical == null) {
            ctx.deinit();
            return VulkanContext{};
        }

        ctx.physical_device = selected_physical;
        ctx.queue_family_index = selected_queue_family;

        const priority = [_]f32{1.0};
        const queue_info = [_]types.VkDeviceQueueCreateInfo{
            types.VkDeviceQueueCreateInfo{
                .queueFamilyIndex = selected_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            },
        };

        const device_info = types.VkDeviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
        };

        var device: types.VkDevice = null;
        if (ctx.vkCreateDevice.?(selected_physical, &device_info, null, &device) != .SUCCESS or device == null) {
            ctx.deinit();
            return VulkanContext{};
        }

        ctx.device = device;

        var queue: types.VkQueue = null;
        ctx.vkGetDeviceQueue.?(device, selected_queue_family, 0, &queue);
        if (queue == null) {
            ctx.deinit();
            return VulkanContext{};
        }
        ctx.compute_queue = queue;

        ctx.api_version = std.mem.readInt(u32, selected_props_buf[0..4], .little);
        ctx.driver_version = std.mem.readInt(u32, selected_props_buf[4..8], .little);
        ctx.vendor_id = std.mem.readInt(u32, selected_props_buf[8..12], .little);
        ctx.device_id = std.mem.readInt(u32, selected_props_buf[12..16], .little);
        ctx.device_type = std.mem.readInt(u32, selected_props_buf[16..20], .little);
        ctx.is_uma = (ctx.device_type == types.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU or ctx.device_type == 4);

        const raw_name = selected_props_buf[20..276];
        var name_len: usize = 0;
        while (name_len < raw_name.len and raw_name[name_len] != 0) : (name_len += 1) {}
        @memcpy(ctx.device_name[0..name_len], raw_name[0..name_len]);
        ctx.device_name_len = name_len;
        ctx.status = .vulkan;

        return ctx;
    }

    pub fn deinit(self: *VulkanContext) void {
        if (self.device) |d| {
            if (self.vkDestroyDevice) |destroyDev| {
                destroyDev(d, null);
            }
            self.device = null;
        }
        if (self.instance) |inst| {
            if (self.vkDestroyInstance) |destroyInst| {
                destroyInst(inst, null);
            }
            self.instance = null;
        }
        if (self.lib) |*l| {
            l.close();
            self.lib = null;
        }
        self.status = .off;
    }

    pub fn deviceNameSlice(self: *const VulkanContext) []const u8 {
        if (self.status == .off or self.device_name_len == 0) return "off";
        return self.device_name[0..self.device_name_len];
    }

    pub fn isUma(self: *const VulkanContext) bool {
        return self.status == .vulkan and self.is_uma;
    }
};

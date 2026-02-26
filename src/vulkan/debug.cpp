#include <vksdl/debug.hpp>

#include <string>

namespace vksdl {

void debugName(VkDevice device, VkObjectType type, std::uint64_t handle,
               std::string_view name) {
    auto pfn = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (!pfn) return;

    std::string nameStr(name);

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = nameStr.c_str();

    pfn(device, &info);
}

void debugName(VkDevice device, VkImage image, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_IMAGE,
              reinterpret_cast<std::uint64_t>(image), name);
}

void debugName(VkDevice device, VkImageView view, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_IMAGE_VIEW,
              reinterpret_cast<std::uint64_t>(view), name);
}

void debugName(VkDevice device, VkBuffer buffer, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_BUFFER,
              reinterpret_cast<std::uint64_t>(buffer), name);
}

void debugName(VkDevice device, VkPipeline pipeline, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_PIPELINE,
              reinterpret_cast<std::uint64_t>(pipeline), name);
}

void debugName(VkDevice device, VkSampler sampler, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_SAMPLER,
              reinterpret_cast<std::uint64_t>(sampler), name);
}

void debugName(VkDevice device, VkDescriptorSet set, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_DESCRIPTOR_SET,
              reinterpret_cast<std::uint64_t>(set), name);
}

void debugName(VkDevice device, VkCommandBuffer cmd, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_COMMAND_BUFFER,
              reinterpret_cast<std::uint64_t>(cmd), name);
}

void debugName(VkDevice device, VkCommandPool pool, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_COMMAND_POOL,
              reinterpret_cast<std::uint64_t>(pool), name);
}

void debugName(VkDevice device, VkSemaphore semaphore, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_SEMAPHORE,
              reinterpret_cast<std::uint64_t>(semaphore), name);
}

void debugName(VkDevice device, VkFence fence, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_FENCE,
              reinterpret_cast<std::uint64_t>(fence), name);
}

void debugName(VkDevice device, VkSwapchainKHR swapchain, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_SWAPCHAIN_KHR,
              reinterpret_cast<std::uint64_t>(swapchain), name);
}

void debugName(VkDevice device, VkEvent event, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_EVENT,
              reinterpret_cast<std::uint64_t>(event), name);
}

void debugName(VkDevice device, VkDescriptorSetLayout layout, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
              reinterpret_cast<std::uint64_t>(layout), name);
}

void debugName(VkDevice device, VkPipelineLayout layout, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT,
              reinterpret_cast<std::uint64_t>(layout), name);
}

void debugName(VkDevice device, VkShaderModule module, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_SHADER_MODULE,
              reinterpret_cast<std::uint64_t>(module), name);
}

void debugName(VkDevice device, VkAccelerationStructureKHR as, std::string_view name) {
    debugName(device, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
              reinterpret_cast<std::uint64_t>(as), name);
}

} // namespace vksdl

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string_view>

namespace vksdl {

// Tag Vulkan objects with debug names visible in validation layers and
// GPU debuggers (RenderDoc, Nsight). No-op when VK_EXT_debug_utils is
// not enabled on the device.

// Core -- any object type.
void debugName(VkDevice device, VkObjectType type, std::uint64_t handle, std::string_view name);

// Convenience overloads for common handle types.
void debugName(VkDevice device, VkImage image, std::string_view name);
void debugName(VkDevice device, VkImageView view, std::string_view name);
void debugName(VkDevice device, VkBuffer buffer, std::string_view name);
void debugName(VkDevice device, VkPipeline pipeline, std::string_view name);
void debugName(VkDevice device, VkSampler sampler, std::string_view name);
void debugName(VkDevice device, VkDescriptorSet set, std::string_view name);
void debugName(VkDevice device, VkCommandBuffer cmd, std::string_view name);
void debugName(VkDevice device, VkCommandPool pool, std::string_view name);
void debugName(VkDevice device, VkSemaphore semaphore, std::string_view name);
void debugName(VkDevice device, VkFence fence, std::string_view name);
void debugName(VkDevice device, VkSwapchainKHR swapchain, std::string_view name);
void debugName(VkDevice device, VkEvent event, std::string_view name);
void debugName(VkDevice device, VkDescriptorSetLayout layout, std::string_view name);
void debugName(VkDevice device, VkPipelineLayout layout, std::string_view name);
void debugName(VkDevice device, VkShaderModule module, std::string_view name);
void debugName(VkDevice device, VkAccelerationStructureKHR as, std::string_view name);

} // namespace vksdl

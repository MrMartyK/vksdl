#pragma once

#include <vulkan/vulkan.h>

#include <atomic>

namespace vksdl::detail {

struct PipelineHandleImpl {
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline baseline = VK_NULL_HANDLE;
    std::atomic<VkPipeline> optimized = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    bool ownsLayout = true;
    VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    std::atomic<bool> destroyed = false; // set before delete, checked by bg thread
};

} // namespace vksdl::detail

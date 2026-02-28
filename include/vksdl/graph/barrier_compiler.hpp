#pragma once

#include <vksdl/graph/resource_state.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl::graph {

// Pre-allocated batch of barriers for one pass boundary.
// Holds raw Vulkan barrier structs. All memory owned by this struct.
struct BarrierBatch {
    std::vector<VkImageMemoryBarrier2> imageBarriers;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    std::vector<VkMemoryBarrier2> memoryBarriers;

    // Build VkDependencyInfo pointing into the vectors above.
    // The returned struct references this batch's storage --
    // BarrierBatch must outlive the VkDependencyInfo.
    [[nodiscard]] VkDependencyInfo dependencyInfo() const;

    [[nodiscard]] bool empty() const {
        return imageBarriers.empty() && bufferBarriers.empty() && memoryBarriers.empty();
    }

    void clear();
};

// Request to compute a barrier for an image subresource transition.
struct ImageBarrierRequest {
    VkImage image = VK_NULL_HANDLE;
    SubresourceRange range;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    ResourceState src;
    ResourceState dst;  // desired state (only lastWrite* and currentLayout matter)
    bool isRead = true; // whether the dst access is a read
};

// Request to compute a barrier for a buffer transition.
struct BufferBarrierRequest {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = VK_WHOLE_SIZE;
    ResourceState src;
    ResourceState dst;
    bool isRead = true;
};

// Append a barrier for this image transition to the batch.
// No-op when no barrier is needed.
void appendImageBarrier(BarrierBatch& batch, const ImageBarrierRequest& req);

// Append a barrier for this buffer transition to the batch.
// No-op when no barrier is needed.
void appendBufferBarrier(BarrierBatch& batch, const BufferBarrierRequest& req);

// Check if an access mask contains any write operations.
[[nodiscard]] bool isWriteAccess(VkAccessFlags2 access);

} // namespace vksdl::graph

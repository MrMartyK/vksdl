#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

// Fluent helper for push descriptors (VK_KHR_push_descriptor).
// Mirrors the DescriptorWriter API but calls vkCmdPushDescriptorSetKHR
// instead of vkUpdateDescriptorSets. No descriptor pool or set allocation
// needed -- descriptors are pushed directly into the command buffer.
//
// Thread safety: thread-confined.
//
// Requires Device::hasPushDescriptors() == true.
//
// Usage:
//   PushDescriptorWriter(pipelineLayout, /*set=*/0)
//       .image(0, view, layout, sampler)
//       .buffer(1, ubo, sizeof(UBO))
//       .push(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
class PushDescriptorWriter {
public:
    explicit PushDescriptorWriter(VkPipelineLayout layout, std::uint32_t set);

    // Combined image sampler (most common image binding).
    PushDescriptorWriter& image(std::uint32_t binding, VkImageView view,
                                VkImageLayout imageLayout, VkSampler sampler,
                                VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    // Storage image.
    PushDescriptorWriter& storageImage(std::uint32_t binding, VkImageView view,
                                        VkImageLayout imageLayout);

    // Buffer binding (uniform, storage, dynamic variants).
    PushDescriptorWriter& buffer(std::uint32_t binding, VkBuffer buf,
                                  VkDeviceSize size, VkDeviceSize offset = 0,
                                  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    // Push all accumulated descriptors into the command buffer.
    // bindPoint: VK_PIPELINE_BIND_POINT_GRAPHICS or VK_PIPELINE_BIND_POINT_COMPUTE.
    void push(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint);

private:
    VkPipelineLayout layout_;
    std::uint32_t    set_;

    enum class InfoKind : std::uint8_t { Image, Buffer };

    struct PendingWrite {
        std::uint32_t    binding;
        VkDescriptorType type;
        InfoKind         kind;
        std::uint32_t    infoIndex;
    };

    std::vector<VkDescriptorImageInfo>  imageInfos_;
    std::vector<VkDescriptorBufferInfo> bufferInfos_;
    std::vector<PendingWrite>           pending_;
};

} // namespace vksdl

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// Fluent helper for writing descriptor set bindings.
// Accumulates writes and issues one vkUpdateDescriptorSets call.
//
// Usage:
//   DescriptorWriter(set)
//       .buffer(0, ubo, sizeof(UBO))
//       .image(1, view, layout, sampler)
//       .write(device);
class DescriptorWriter {
public:
    explicit DescriptorWriter(VkDescriptorSet set);

    // Combined image sampler (most common image binding).
    DescriptorWriter& image(std::uint32_t binding, VkImageView view,
                            VkImageLayout layout, VkSampler sampler,
                            VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    // Storage image.
    DescriptorWriter& storageImage(std::uint32_t binding, VkImageView view,
                                    VkImageLayout layout);

    // Buffer binding (uniform, storage, dynamic variants).
    DescriptorWriter& buffer(std::uint32_t binding, VkBuffer buf,
                             VkDeviceSize size, VkDeviceSize offset = 0,
                             VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    // Issue all accumulated writes in one vkUpdateDescriptorSets call.
    void write(VkDevice device);
    void write(const Device& device);

private:
    VkDescriptorSet set_;

    // Deferred build: stores info structs and pending writes.
    // VkWriteDescriptorSet array is built at write() time to avoid
    // pointer invalidation from push_back during accumulation.
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

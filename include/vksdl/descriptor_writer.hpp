#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vksdl {

class Device;
class DescriptorPool;
class Pipeline;

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
    explicit DescriptorWriter(VkDescriptorSet set, std::vector<std::pair<std::uint32_t, VkDescriptorType>> bindingTypes);

    [[nodiscard]] static Result<DescriptorWriter> forReflected(
        const Pipeline& pipeline, DescriptorPool& pool, std::uint32_t setIndex = 0);

    [[nodiscard]] VkDescriptorSet descriptorSet() const { return set_; }

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
    [[nodiscard]] VkDescriptorType reflectedTypeOr(
        std::uint32_t binding, VkDescriptorType fallback) const;

    VkDescriptorSet set_;
    std::vector<std::pair<std::uint32_t, VkDescriptorType>> bindingTypes_;

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

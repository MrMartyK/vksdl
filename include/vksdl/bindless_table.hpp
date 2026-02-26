#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vksdl {

class Device;

// Large partially-bound descriptor array for bindless rendering.
// Uses UPDATE_AFTER_BIND + PARTIALLY_BOUND so slots can be written at any
// time and uninitialized slots won't cause validation errors.
//
// Requires Device::hasBindless() == true.
//
// Usage:
//   auto table = BindlessTable::create(device, 4096).value();
//   table.writeImage(0, view0, layout0, sampler);
//   table.writeImage(7, view7, layout7, sampler);
//   table.bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0);
//   // In shader: texture(textures[nonuniformEXT(index)], uv)
// Thread safety: thread-confined. Descriptor updates are not
// externally synchronized even with UPDATE_AFTER_BIND.
class BindlessTable {
public:
    [[nodiscard]] static Result<BindlessTable> create(
        const Device& device,
        std::uint32_t capacity,
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VkShaderStageFlags stages = VK_SHADER_STAGE_ALL);

    ~BindlessTable();
    BindlessTable(BindlessTable&&) noexcept;
    BindlessTable& operator=(BindlessTable&&) noexcept;
    BindlessTable(const BindlessTable&) = delete;
    BindlessTable& operator=(const BindlessTable&) = delete;

    // Write a single slot. Immediate -- calls vkUpdateDescriptorSets.
    void writeImage(std::uint32_t index, VkImageView view,
                    VkImageLayout layout, VkSampler sampler);
    void writeStorageImage(std::uint32_t index, VkImageView view,
                           VkImageLayout layout);
    void writeBuffer(std::uint32_t index, VkBuffer buffer,
                     VkDeviceSize size, VkDeviceSize offset = 0);

    // Bind this table to a command buffer at the given set index.
    void bind(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint,
              VkPipelineLayout pipelineLayout, std::uint32_t set = 0) const;

    [[nodiscard]] VkDescriptorSet       native()                const { return set_; }
    [[nodiscard]] VkDescriptorSet       vkDescriptorSet()       const { return native(); }
    [[nodiscard]] VkDescriptorSetLayout vkDescriptorSetLayout() const { return layout_; }
    [[nodiscard]] std::uint32_t         capacity()              const { return capacity_; }
    [[nodiscard]] VkDescriptorType      descriptorType()        const { return type_; }

private:
    BindlessTable() = default;
    void destroy();

    VkDevice              device_   = VK_NULL_HANDLE;
    VkDescriptorPool      pool_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_   = VK_NULL_HANDLE;
    VkDescriptorSet       set_      = VK_NULL_HANDLE;
    std::uint32_t         capacity_ = 0;
    VkDescriptorType      type_     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
};

} // namespace vksdl

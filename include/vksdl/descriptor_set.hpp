#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// RAII descriptor set. Owns VkDescriptorSetLayout, VkDescriptorPool, and
// VkDescriptorSet. Pool is sized for exactly one set.
// Destruction: pool implicitly frees the set, then destroy pool, then layout.
//
// Descriptor safety: do not destroy while a command buffer referencing the
// set is still pending. Do not update bindings while the set is in-flight.
//
// Thread safety: thread-confined. Updates via vkUpdateDescriptorSets are
// not externally synchronized by Vulkan.
class DescriptorSet {
public:
    ~DescriptorSet();
    DescriptorSet(DescriptorSet&&) noexcept;
    DescriptorSet& operator=(DescriptorSet&&) noexcept;
    DescriptorSet(const DescriptorSet&) = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;

    [[nodiscard]] VkDescriptorSet       native()                const { return set_; }
    [[nodiscard]] VkDescriptorSet       vkDescriptorSet()       const { return native(); }
    [[nodiscard]] VkDescriptorSetLayout vkDescriptorSetLayout() const { return layout_; }
    [[nodiscard]] VkDescriptorPool      vkDescriptorPool()      const { return pool_; }

    void updateBuffer(std::uint32_t binding, VkBuffer buffer,
                      VkDeviceSize size, VkDeviceSize offset = 0);

    void updateImage(std::uint32_t binding, VkImageView view,
                     VkImageLayout layout,
                     VkSampler sampler = VK_NULL_HANDLE);

    void updateAccelerationStructure(std::uint32_t binding,
                                     VkAccelerationStructureKHR as);

private:
    friend class DescriptorSetBuilder;
    DescriptorSet() = default;

    VkDevice              device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool      pool_   = VK_NULL_HANDLE;
    VkDescriptorSet       set_    = VK_NULL_HANDLE;

    // Binding metadata for update methods to look up descriptor type.
    struct BindingInfo {
        std::uint32_t    binding;
        VkDescriptorType type;
    };
    std::vector<BindingInfo> bindings_;
};

class DescriptorSetBuilder {
public:
    explicit DescriptorSetBuilder(const Device& device);

    DescriptorSetBuilder& addUniformBuffer(std::uint32_t binding,
                                           VkShaderStageFlags stageFlags);
    DescriptorSetBuilder& addDynamicUniformBuffer(std::uint32_t binding,
                                                  VkShaderStageFlags stageFlags);
    DescriptorSetBuilder& addStorageBuffer(std::uint32_t binding,
                                           VkShaderStageFlags stageFlags);
    DescriptorSetBuilder& addStorageImage(std::uint32_t binding,
                                          VkShaderStageFlags stageFlags);
    DescriptorSetBuilder& addCombinedImageSampler(std::uint32_t binding,
                                                  VkShaderStageFlags stageFlags);
    DescriptorSetBuilder& addAccelerationStructure(std::uint32_t binding,
                                                    VkShaderStageFlags stageFlags);

    // Escape hatch -- any descriptor type
    DescriptorSetBuilder& addBinding(std::uint32_t binding,
                                     VkDescriptorType type,
                                     VkShaderStageFlags stageFlags,
                                     std::uint32_t count = 1);

    [[nodiscard]] Result<DescriptorSet> build();

private:
    VkDevice device_ = VK_NULL_HANDLE;

    struct BindingEntry {
        VkDescriptorSetLayoutBinding layoutBinding;
        VkDescriptorType             type;
    };
    std::vector<BindingEntry> entries_;
};

} // namespace vksdl

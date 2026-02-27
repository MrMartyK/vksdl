#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace vksdl {

class Device;

// Reusable descriptor set layout object with typed binding metadata.
//
// Thread safety: immutable after construction.
class DescriptorLayout {
  public:
    ~DescriptorLayout();
    DescriptorLayout(DescriptorLayout&&) noexcept;
    DescriptorLayout& operator=(DescriptorLayout&&) noexcept;
    DescriptorLayout(const DescriptorLayout&) = delete;
    DescriptorLayout& operator=(const DescriptorLayout&) = delete;

    [[nodiscard]] VkDescriptorSetLayout vkDescriptorSetLayout() const {
        return layout_;
    }

    struct BindingInfo {
        std::uint32_t binding = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        std::uint32_t count = 0;
        VkShaderStageFlags stageFlags = 0;
    };

    [[nodiscard]] std::optional<BindingInfo> bindingInfo(std::uint32_t binding) const;

  private:
    friend class DescriptorLayoutBuilder;
    DescriptorLayout() = default;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    std::vector<BindingInfo> bindings_;
};

class DescriptorLayoutBuilder {
  public:
    explicit DescriptorLayoutBuilder(const Device& device);

    DescriptorLayoutBuilder& addUniformBuffer(std::uint32_t binding, VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addDynamicUniformBuffer(std::uint32_t binding,
                                                     VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addStorageBuffer(std::uint32_t binding, VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addStorageImage(std::uint32_t binding, VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addCombinedImageSampler(std::uint32_t binding,
                                                     VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addAccelerationStructure(std::uint32_t binding,
                                                      VkShaderStageFlags stageFlags);
    DescriptorLayoutBuilder& addBinding(std::uint32_t binding, VkDescriptorType type,
                                        VkShaderStageFlags stageFlags, std::uint32_t count = 1);

    [[nodiscard]] Result<DescriptorLayout> build();

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<DescriptorLayout::BindingInfo> bindings_;
};

} // namespace vksdl

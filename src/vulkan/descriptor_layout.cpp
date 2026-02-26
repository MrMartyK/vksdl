#include <vksdl/descriptor_layout.hpp>
#include <vksdl/device.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace vksdl {

DescriptorLayout::~DescriptorLayout() {
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }
}

DescriptorLayout::DescriptorLayout(DescriptorLayout&& o) noexcept
    : device_(o.device_)
    , layout_(o.layout_)
    , bindings_(std::move(o.bindings_)) {
    o.device_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
}

DescriptorLayout& DescriptorLayout::operator=(DescriptorLayout&& o) noexcept {
    if (this != &o) {
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        }
        device_ = o.device_;
        layout_ = o.layout_;
        bindings_ = std::move(o.bindings_);
        o.device_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
    }
    return *this;
}

std::optional<DescriptorLayout::BindingInfo> DescriptorLayout::bindingInfo(
    std::uint32_t binding) const {
    auto it = std::find_if(bindings_.begin(), bindings_.end(),
        [binding](const BindingInfo& b) { return b.binding == binding; });
    if (it == bindings_.end()) {
        return std::nullopt;
    }
    return *it;
}

DescriptorLayoutBuilder::DescriptorLayoutBuilder(const Device& device)
    : device_(device.vkDevice()) {}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addUniformBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addDynamicUniformBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addStorageBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addStorageImage(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addCombinedImageSampler(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addAccelerationStructure(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, stageFlags);
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addBinding(
    std::uint32_t binding, VkDescriptorType type,
    VkShaderStageFlags stageFlags, std::uint32_t count) {
    DescriptorLayout::BindingInfo info{};
    info.binding = binding;
    info.type = type;
    info.count = count;
    info.stageFlags = stageFlags;
    bindings_.push_back(info);
    return *this;
}

Result<DescriptorLayout> DescriptorLayoutBuilder::build() {
    if (bindings_.empty()) {
        return Error{"create descriptor layout", 0,
                     "no bindings added -- call addUniformBuffer(), addStorageImage(), etc."};
    }

    DescriptorLayout layout;
    layout.device_ = device_;

    std::vector<VkDescriptorSetLayoutBinding> vkBindings;
    vkBindings.reserve(bindings_.size());
    for (const auto& b : bindings_) {
        VkDescriptorSetLayoutBinding vkBinding{};
        vkBinding.binding = b.binding;
        vkBinding.descriptorType = b.type;
        vkBinding.descriptorCount = b.count;
        vkBinding.stageFlags = b.stageFlags;
        vkBindings.push_back(vkBinding);
    }

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<std::uint32_t>(vkBindings.size());
    ci.pBindings = vkBindings.data();

    VkResult vr = vkCreateDescriptorSetLayout(device_, &ci, nullptr, &layout.layout_);
    if (vr != VK_SUCCESS) {
        return Error{"create descriptor layout", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorSetLayout failed"};
    }

    layout.bindings_ = bindings_;
    return layout;
}

} // namespace vksdl

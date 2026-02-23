#include <vksdl/descriptor_set.hpp>
#include <vksdl/device.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace vksdl {

DescriptorSet::~DescriptorSet() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }
}

DescriptorSet::DescriptorSet(DescriptorSet&& o) noexcept
    : device_(o.device_), layout_(o.layout_), pool_(o.pool_),
      set_(o.set_), bindings_(std::move(o.bindings_)) {
    o.device_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.pool_   = VK_NULL_HANDLE;
    o.set_    = VK_NULL_HANDLE;
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& o) noexcept {
    if (this != &o) {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, pool_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        }
        device_   = o.device_;
        layout_   = o.layout_;
        pool_     = o.pool_;
        set_      = o.set_;
        bindings_ = std::move(o.bindings_);
        o.device_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
        o.pool_   = VK_NULL_HANDLE;
        o.set_    = VK_NULL_HANDLE;
    }
    return *this;
}

void DescriptorSet::updateBuffer(std::uint32_t binding, VkBuffer buffer,
                                  VkDeviceSize size, VkDeviceSize offset) {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for (const auto& b : bindings_) {
        if (b.binding == binding) {
            type = b.type;
            break;
        }
    }

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = buffer;
    bufInfo.offset = offset;
    bufInfo.range  = size;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = set_;
    write.dstBinding      = binding;
    write.descriptorCount = 1;
    write.descriptorType  = type;
    write.pBufferInfo     = &bufInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorSet::updateAccelerationStructure(
    std::uint32_t binding, VkAccelerationStructureKHR as) {

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures    = &as;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext           = &asInfo;
    write.dstSet          = set_;
    write.dstBinding      = binding;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorSet::updateImage(std::uint32_t binding, VkImageView view,
                                 VkImageLayout layout, VkSampler sampler) {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (const auto& b : bindings_) {
        if (b.binding == binding) {
            type = b.type;
            break;
        }
    }

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = view;
    imgInfo.imageLayout = layout;
    imgInfo.sampler     = sampler;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = set_;
    write.dstBinding      = binding;
    write.descriptorCount = 1;
    write.descriptorType  = type;
    write.pImageInfo      = &imgInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

DescriptorSetBuilder::DescriptorSetBuilder(const Device& device)
    : device_(device.vkDevice()) {}

DescriptorSetBuilder& DescriptorSetBuilder::addUniformBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addDynamicUniformBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addStorageBuffer(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addStorageImage(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addCombinedImageSampler(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addAccelerationStructure(
    std::uint32_t binding, VkShaderStageFlags stageFlags) {
    return addBinding(binding, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, stageFlags);
}

DescriptorSetBuilder& DescriptorSetBuilder::addBinding(
    std::uint32_t binding, VkDescriptorType type,
    VkShaderStageFlags stageFlags, std::uint32_t count) {
    VkDescriptorSetLayoutBinding lb{};
    lb.binding         = binding;
    lb.descriptorType  = type;
    lb.descriptorCount = count;
    lb.stageFlags      = stageFlags;

    entries_.push_back({lb, type});
    return *this;
}

Result<DescriptorSet> DescriptorSetBuilder::build() {
    if (entries_.empty()) {
        return Error{"create descriptor set", 0,
                     "no bindings added -- call addUniformBuffer(), addStorageImage(), etc."};
    }

    DescriptorSet ds;
    ds.device_ = device_;

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(entries_.size());
    for (const auto& e : entries_) {
        layoutBindings.push_back(e.layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
    dslCI.pBindings    = layoutBindings.data();

    VkResult vr = vkCreateDescriptorSetLayout(device_, &dslCI, nullptr, &ds.layout_);
    if (vr != VK_SUCCESS) {
        return Error{"create descriptor set layout", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorSetLayout failed"};
    }

    // Aggregate pool sizes by descriptor type, sized exactly for one set.
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto& e : entries_) {
        bool found = false;
        for (auto& ps : poolSizes) {
            if (ps.type == e.layoutBinding.descriptorType) {
                ps.descriptorCount += e.layoutBinding.descriptorCount;
                found = true;
                break;
            }
        }
        if (!found) {
            poolSizes.push_back({e.layoutBinding.descriptorType,
                                 e.layoutBinding.descriptorCount});
        }
    }

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolCI.pPoolSizes    = poolSizes.data();

    vr = vkCreateDescriptorPool(device_, &poolCI, nullptr, &ds.pool_);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, ds.layout_, nullptr);
        ds.layout_ = VK_NULL_HANDLE;
        return Error{"create descriptor pool", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorPool failed"};
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = ds.pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &ds.layout_;

    vr = vkAllocateDescriptorSets(device_, &allocInfo, &ds.set_);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorPool(device_, ds.pool_, nullptr);
        ds.pool_ = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(device_, ds.layout_, nullptr);
        ds.layout_ = VK_NULL_HANDLE;
        return Error{"allocate descriptor set", static_cast<std::int32_t>(vr),
                     "vkAllocateDescriptorSets failed"};
    }

    ds.bindings_.reserve(entries_.size());
    for (const auto& e : entries_) {
        ds.bindings_.push_back({e.layoutBinding.binding, e.type});
    }

    return ds;
}

} // namespace vksdl

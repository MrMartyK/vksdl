#include <vksdl/bindless_table.hpp>
#include <vksdl/device.hpp>

#include <cassert>

namespace vksdl {

void BindlessTable::destroy() {
    if (device_ == VK_NULL_HANDLE)
        return;
    if (pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    if (layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    device_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_ = VK_NULL_HANDLE;
}

BindlessTable::~BindlessTable() {
    destroy();
}

BindlessTable::BindlessTable(BindlessTable&& o) noexcept
    : device_(o.device_), pool_(o.pool_), layout_(o.layout_), set_(o.set_), capacity_(o.capacity_),
      type_(o.type_) {
    o.device_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.set_ = VK_NULL_HANDLE;
}

BindlessTable& BindlessTable::operator=(BindlessTable&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pool_ = o.pool_;
        layout_ = o.layout_;
        set_ = o.set_;
        capacity_ = o.capacity_;
        type_ = o.type_;
        o.device_ = VK_NULL_HANDLE;
        o.pool_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
        o.set_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<BindlessTable> BindlessTable::create(const Device& device, std::uint32_t capacity,
                                            VkDescriptorType type, VkShaderStageFlags stages) {
    if (!device.hasBindless()) {
        return Error{"create BindlessTable", 0,
                     "Bindless descriptors not available on this device.\n"
                     "Check Device::hasBindless() before creating a BindlessTable."};
    }
    if (capacity == 0) {
        return Error{"create BindlessTable", 0, "Capacity must be greater than zero."};
    }

    VkDevice dev = device.vkDevice();

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = type;
    binding.descriptorCount = capacity;
    binding.stageFlags = stages;

    VkDescriptorBindingFlags bindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCI{};
    flagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsCI.bindingCount = 1;
    flagsCI.pBindingFlags = &bindingFlags;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.pNext = &flagsCI;
    layoutCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutCI.bindingCount = 1;
    layoutCI.pBindings = &binding;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr, &layout);
    if (vr != VK_SUCCESS) {
        return Error{"create BindlessTable layout", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorSetLayout failed for bindless table"};
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = type;
    poolSize.descriptorCount = capacity;

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    vr = vkCreateDescriptorPool(dev, &poolCI, nullptr, &pool);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(dev, layout, nullptr);
        return Error{"create BindlessTable pool", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorPool failed for bindless table"};
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    vr = vkAllocateDescriptorSets(dev, &allocInfo, &set);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorPool(dev, pool, nullptr);
        vkDestroyDescriptorSetLayout(dev, layout, nullptr);
        return Error{"create BindlessTable set", static_cast<std::int32_t>(vr),
                     "vkAllocateDescriptorSets failed for bindless table"};
    }

    BindlessTable table;
    table.device_ = dev;
    table.pool_ = pool;
    table.layout_ = layout;
    table.set_ = set;
    table.capacity_ = capacity;
    table.type_ = type;
    return table;
}

void BindlessTable::writeImage(std::uint32_t index, VkImageView view, VkImageLayout layout,
                               VkSampler sampler) {
    assert(index < capacity_);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = type_;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void BindlessTable::writeStorageImage(std::uint32_t index, VkImageView view, VkImageLayout layout) {
    assert(index < capacity_);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = type_;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void BindlessTable::writeBuffer(std::uint32_t index, VkBuffer buffer, VkDeviceSize size,
                                VkDeviceSize offset) {
    assert(index < capacity_);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = type_;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void BindlessTable::bind(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint,
                         VkPipelineLayout pipelineLayout, std::uint32_t setIndex) const {
    vkCmdBindDescriptorSets(cmd, bindPoint, pipelineLayout, setIndex, 1, &set_, 0, nullptr);
}

} // namespace vksdl

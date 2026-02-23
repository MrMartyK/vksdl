#include <vksdl/descriptor_writer.hpp>
#include <vksdl/device.hpp>

namespace vksdl {

DescriptorWriter::DescriptorWriter(VkDescriptorSet set)
    : set_(set) {}

DescriptorWriter& DescriptorWriter::image(std::uint32_t binding,
                                           VkImageView view,
                                           VkImageLayout layout,
                                           VkSampler sampler,
                                           VkDescriptorType type) {
    VkDescriptorImageInfo info{};
    info.sampler     = sampler;
    info.imageView   = view;
    info.imageLayout = layout;
    auto idx = static_cast<std::uint32_t>(imageInfos_.size());
    imageInfos_.push_back(info);
    pending_.push_back({binding, type, InfoKind::Image, idx});
    return *this;
}

DescriptorWriter& DescriptorWriter::storageImage(std::uint32_t binding,
                                                   VkImageView view,
                                                   VkImageLayout layout) {
    return image(binding, view, layout, VK_NULL_HANDLE,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

DescriptorWriter& DescriptorWriter::buffer(std::uint32_t binding,
                                            VkBuffer buf,
                                            VkDeviceSize size,
                                            VkDeviceSize offset,
                                            VkDescriptorType type) {
    VkDescriptorBufferInfo info{};
    info.buffer = buf;
    info.offset = offset;
    info.range  = size;
    auto idx = static_cast<std::uint32_t>(bufferInfos_.size());
    bufferInfos_.push_back(info);
    pending_.push_back({binding, type, InfoKind::Buffer, idx});
    return *this;
}

void DescriptorWriter::write(VkDevice device) {
    // Info vectors are stable now (no more push_back), so pointers are valid.
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(pending_.size());

    for (const auto& pw : pending_) {
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set_;
        w.dstBinding      = pw.binding;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType  = pw.type;

        if (pw.kind == InfoKind::Image) {
            w.pImageInfo = &imageInfos_[pw.infoIndex];
        } else {
            w.pBufferInfo = &bufferInfos_[pw.infoIndex];
        }

        writes.push_back(w);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device,
            static_cast<std::uint32_t>(writes.size()),
            writes.data(), 0, nullptr);
    }
}

void DescriptorWriter::write(const Device& device) {
    write(device.vkDevice());
}

} // namespace vksdl

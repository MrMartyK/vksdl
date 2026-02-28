#include <vksdl/push_descriptor_writer.hpp>

namespace vksdl {

PushDescriptorWriter::PushDescriptorWriter(VkPipelineLayout layout, std::uint32_t set)
    : layout_(layout), set_(set) {}

PushDescriptorWriter& PushDescriptorWriter::image(std::uint32_t binding, VkImageView view,
                                                  VkImageLayout imageLayout, VkSampler sampler,
                                                  VkDescriptorType type) {
    VkDescriptorImageInfo info{};
    info.sampler = sampler;
    info.imageView = view;
    info.imageLayout = imageLayout;
    auto idx = static_cast<std::uint32_t>(imageInfos_.size());
    imageInfos_.push_back(info);
    pending_.push_back({binding, type, InfoKind::Image, idx});
    return *this;
}

PushDescriptorWriter& PushDescriptorWriter::storageImage(std::uint32_t binding, VkImageView view,
                                                         VkImageLayout imageLayout) {
    return image(binding, view, imageLayout, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

PushDescriptorWriter& PushDescriptorWriter::buffer(std::uint32_t binding, VkBuffer buf,
                                                   VkDeviceSize size, VkDeviceSize offset,
                                                   VkDescriptorType type) {
    VkDescriptorBufferInfo info{};
    info.buffer = buf;
    info.offset = offset;
    info.range = size;
    auto idx = static_cast<std::uint32_t>(bufferInfos_.size());
    bufferInfos_.push_back(info);
    pending_.push_back({binding, type, InfoKind::Buffer, idx});
    return *this;
}

void PushDescriptorWriter::push(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint) {
    // Info vectors are stable now (no more push_back), so pointers are valid.
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(pending_.size());

    for (const auto& pw : pending_) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = VK_NULL_HANDLE; // not used by push descriptors
        w.dstBinding = pw.binding;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = pw.type;

        if (pw.kind == InfoKind::Image) {
            w.pImageInfo = &imageInfos_[pw.infoIndex];
        } else {
            w.pBufferInfo = &bufferInfos_[pw.infoIndex];
        }

        writes.push_back(w);
    }

    if (!writes.empty()) {
        // KHR extension function -- not in the static loader lib.
        // Load via vkGetInstanceProcAddr(NULL) to get the loader trampoline.
        // Per-call loading matches the pattern in debug.cpp and rt_functions.cpp.
        static auto pfn = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCmdPushDescriptorSetKHR"));
        if (pfn) {
            pfn(cmd, bindPoint, layout_, set_, static_cast<std::uint32_t>(writes.size()),
                writes.data());
        }
    }
}

} // namespace vksdl

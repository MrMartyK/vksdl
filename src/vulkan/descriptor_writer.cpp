#include <vksdl/descriptor_pool.hpp>
#include <vksdl/descriptor_writer.hpp>
#include <vksdl/device.hpp>
#include <vksdl/pipeline.hpp>

#include <string>
#include <utility>

namespace vksdl {

DescriptorWriter::DescriptorWriter(VkDescriptorSet set) : set_(set) {}

DescriptorWriter::DescriptorWriter(
    VkDescriptorSet set, std::vector<std::pair<std::uint32_t, VkDescriptorType>> bindingTypes)
    : set_(set), bindingTypes_(std::move(bindingTypes)) {}

Result<DescriptorWriter> DescriptorWriter::forReflected(const Pipeline& pipeline,
                                                        DescriptorPool& pool,
                                                        std::uint32_t setIndex) {
    const auto& reflectedLayouts = pipeline.reflectedSetLayouts();
    if (setIndex >= reflectedLayouts.size()) {
        return Error{"create reflected descriptor writer", 0,
                     "set index " + std::to_string(setIndex) +
                         " is out of range for reflectedSetLayouts()"};
    }

    auto ds = pool.allocate(reflectedLayouts[setIndex]);
    if (!ds.ok()) {
        return ds.error();
    }

    std::vector<std::pair<std::uint32_t, VkDescriptorType>> bindingTypes;
    if (const auto* reflected = pipeline.reflectedLayout()) {
        for (const auto& binding : reflected->bindings) {
            if (binding.set == setIndex) {
                bindingTypes.emplace_back(binding.binding, binding.type);
            }
        }
    }

    return DescriptorWriter(ds.value(), std::move(bindingTypes));
}

VkDescriptorType DescriptorWriter::reflectedTypeOr(std::uint32_t binding,
                                                   VkDescriptorType fallback) const {
    for (const auto& p : bindingTypes_) {
        if (p.first == binding) {
            return p.second;
        }
    }
    return fallback;
}

DescriptorWriter& DescriptorWriter::image(std::uint32_t binding, VkImageView view,
                                          VkImageLayout layout, VkSampler sampler,
                                          VkDescriptorType type) {
    VkDescriptorImageInfo info{};
    info.sampler = sampler;
    info.imageView = view;
    info.imageLayout = layout;
    auto idx = static_cast<std::uint32_t>(imageInfos_.size());
    imageInfos_.push_back(info);
    pending_.push_back({binding, reflectedTypeOr(binding, type), InfoKind::Image, idx});
    return *this;
}

DescriptorWriter& DescriptorWriter::storageImage(std::uint32_t binding, VkImageView view,
                                                 VkImageLayout layout) {
    return image(binding, view, layout, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

DescriptorWriter& DescriptorWriter::buffer(std::uint32_t binding, VkBuffer buf, VkDeviceSize size,
                                           VkDeviceSize offset, VkDescriptorType type) {
    VkDescriptorBufferInfo info{};
    info.buffer = buf;
    info.offset = offset;
    info.range = size;
    auto idx = static_cast<std::uint32_t>(bufferInfos_.size());
    bufferInfos_.push_back(info);
    pending_.push_back({binding, reflectedTypeOr(binding, type), InfoKind::Buffer, idx});
    return *this;
}

void DescriptorWriter::write(VkDevice device) {
    // Info vectors are stable now (no more push_back), so pointers are valid.
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(pending_.size());

    for (const auto& pw : pending_) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set_;
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
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);
    }
}

void DescriptorWriter::write(const Device& device) {
    write(device.vkDevice());
}

} // namespace vksdl

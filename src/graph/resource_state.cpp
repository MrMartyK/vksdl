#include <vksdl/graph/resource_state.hpp>

#include <algorithm>

namespace vksdl::graph {

bool SubresourceRange::contains(const SubresourceRange& other) const {
    return other.baseMipLevel >= baseMipLevel && other.mipEnd() <= mipEnd() &&
           other.baseArrayLayer >= baseArrayLayer && other.layerEnd() <= layerEnd();
}

bool SubresourceRange::overlaps(const SubresourceRange& other) const {
    if (other.baseMipLevel >= mipEnd() || baseMipLevel >= other.mipEnd())
        return false;
    if (other.baseArrayLayer >= layerEnd() || baseArrayLayer >= other.layerEnd())
        return false;
    return true;
}

ImageSubresourceMap::ImageSubresourceMap(std::uint32_t mipLevels, std::uint32_t arrayLayers,
                                         ResourceState initialState) {
    slices_.push_back(
        SubresourceSlice{SubresourceRange{0, mipLevels, 0, arrayLayers}, initialState});
}

void ImageSubresourceMap::resetState(std::uint32_t mipLevels, std::uint32_t arrayLayers,
                                     const ResourceState& state) {
    slices_.clear(); // keeps capacity
    slices_.push_back(SubresourceSlice{SubresourceRange{0, mipLevels, 0, arrayLayers}, state});
}

ResourceState ImageSubresourceMap::queryState(const SubresourceRange& range) const {
    ResourceState merged{};
    bool first = true;
    bool layoutUniform = true;
    bool queueUniform = true;

    for (const auto& slice : slices_) {
        if (!slice.range.overlaps(range))
            continue;

        if (first) {
            merged = slice.state;
            first = false;
        } else {
            merged.lastWriteStage |= slice.state.lastWriteStage;
            merged.lastWriteAccess |= slice.state.lastWriteAccess;
            merged.readStagesSinceWrite |= slice.state.readStagesSinceWrite;
            merged.readAccessSinceWrite |= slice.state.readAccessSinceWrite;

            if (merged.currentLayout != slice.state.currentLayout)
                layoutUniform = false;
            if (merged.queueFamily != slice.state.queueFamily)
                queueUniform = false;
        }
    }

    if (!layoutUniform)
        merged.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!queueUniform)
        merged.queueFamily = VK_QUEUE_FAMILY_IGNORED;

    return merged;
}

void ImageSubresourceMap::setState(const SubresourceRange& range, const ResourceState& newState) {
    std::vector<SubresourceSlice> result;
    result.reserve(slices_.size() + 4);

    for (const auto& slice : slices_) {
        if (!slice.range.overlaps(range)) {
            // No overlap -- keep unchanged.
            result.push_back(slice);
            continue;
        }

        // Split non-overlapping remainders from the existing slice,
        // then the overlapping portion gets the new state.
        const auto& sr = slice.range;

        // Mip levels below the target range.
        if (sr.baseMipLevel < range.baseMipLevel) {
            result.push_back(SubresourceSlice{SubresourceRange{sr.baseMipLevel,
                                                               range.baseMipLevel - sr.baseMipLevel,
                                                               sr.baseArrayLayer, sr.layerCount},
                                              slice.state});
        }

        // Mip levels above the target range.
        if (sr.mipEnd() > range.mipEnd()) {
            result.push_back(
                SubresourceSlice{SubresourceRange{range.mipEnd(), sr.mipEnd() - range.mipEnd(),
                                                  sr.baseArrayLayer, sr.layerCount},
                                 slice.state});
        }

        // Compute the overlapping mip range.
        std::uint32_t overlapMipBase = std::max(sr.baseMipLevel, range.baseMipLevel);
        std::uint32_t overlapMipEnd = std::min(sr.mipEnd(), range.mipEnd());
        std::uint32_t overlapMipCount = overlapMipEnd - overlapMipBase;

        // Array layers below the target range (within the overlapping mip range).
        if (sr.baseArrayLayer < range.baseArrayLayer) {
            result.push_back(SubresourceSlice{
                SubresourceRange{overlapMipBase, overlapMipCount, sr.baseArrayLayer,
                                 range.baseArrayLayer - sr.baseArrayLayer},
                slice.state});
        }

        // Array layers above the target range (within the overlapping mip range).
        if (sr.layerEnd() > range.layerEnd()) {
            result.push_back(
                SubresourceSlice{SubresourceRange{overlapMipBase, overlapMipCount, range.layerEnd(),
                                                  sr.layerEnd() - range.layerEnd()},
                                 slice.state});
        }

        // The actual overlap region gets the new state.
        std::uint32_t overlapLayerBase = std::max(sr.baseArrayLayer, range.baseArrayLayer);
        std::uint32_t overlapLayerEnd = std::min(sr.layerEnd(), range.layerEnd());
        std::uint32_t overlapLayerCount = overlapLayerEnd - overlapLayerBase;

        result.push_back(SubresourceSlice{
            SubresourceRange{overlapMipBase, overlapMipCount, overlapLayerBase, overlapLayerCount},
            newState});
    }

    slices_ = std::move(result);
}

VkImageAspectFlags aspectFromFormat(VkFormat format) {
    switch (format) {
    // Depth-only formats.
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;

    // Depth + stencil formats.
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    // Stencil-only.
    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;

    // Everything else is color.
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

} // namespace vksdl::graph

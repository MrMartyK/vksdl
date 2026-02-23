#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl::graph {

// Vulkan synchronization state for one resource (or subresource range).
// Tracks the last write and all reads since that write separately,
// so the barrier compiler can handle multi-reader fan-out correctly.
//
// Example: pass A writes (fragment), pass B reads (compute), pass C reads
// (fragment). After A->B barrier, readStagesSinceWrite = COMPUTE_SHADER.
// When C arrives, the compiler sees that FRAGMENT_SHADER is not covered
// by readStagesSinceWrite, so it emits an execution dependency from A's
// write stage to C's read stage. Without the split, the naive single-state
// model would see B's read state and skip the barrier -- corrupting on AMD.
struct ResourceState {
    VkPipelineStageFlags2 lastWriteStage        = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        lastWriteAccess        = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 readStagesSinceWrite   = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        readAccessSinceWrite   = VK_ACCESS_2_NONE;
    VkImageLayout         currentLayout          = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t         queueFamily            = VK_QUEUE_FAMILY_IGNORED;

    [[nodiscard]] bool operator==(const ResourceState&) const = default;
};

// A contiguous range of mip levels and array layers.
struct SubresourceRange {
    std::uint32_t baseMipLevel   = 0;
    std::uint32_t levelCount     = 1;
    std::uint32_t baseArrayLayer = 0;
    std::uint32_t layerCount     = 1;

    [[nodiscard]] bool contains(const SubresourceRange& other) const;
    [[nodiscard]] bool overlaps(const SubresourceRange& other) const;
    [[nodiscard]] bool operator==(const SubresourceRange&) const = default;

    // End indices (exclusive) for interval arithmetic.
    [[nodiscard]] std::uint32_t mipEnd()   const { return baseMipLevel + levelCount; }
    [[nodiscard]] std::uint32_t layerEnd() const { return baseArrayLayer + layerCount; }
};

// One subresource range and its current state.
struct SubresourceSlice {
    SubresourceRange range;
    ResourceState    state;
};

// Sparse subresource state map for images.
//
// Starts with one entry covering the full image. Splits only when a pass
// touches a sub-range that doesn't match an existing entry (mipgen, shadow
// cascades, cube faces).
//
// Invariant: slices are non-overlapping and together cover the full image.
class ImageSubresourceMap {
public:
    // Initialize with full-image range in the given state.
    explicit ImageSubresourceMap(std::uint32_t mipLevels,
                                std::uint32_t arrayLayers,
                                ResourceState initialState = {});

    // Look up the state for a given range. If the range spans multiple
    // slices with different states, returns a merged state:
    //   - OR of all write stages/access
    //   - OR of all read stages/access
    //   - layout = common if uniform, else UNDEFINED
    //   - queueFamily = common if uniform, else VK_QUEUE_FAMILY_IGNORED
    [[nodiscard]] ResourceState queryState(const SubresourceRange& range) const;

    // Update the state for a given range. Splits existing slices as needed
    // to maintain the non-overlapping invariant.
    void setState(const SubresourceRange& range, const ResourceState& newState);

    // Reset to a single slice covering the full image with the given state.
    // Reuses internal vector capacity (no heap allocation).
    void resetState(std::uint32_t mipLevels, std::uint32_t arrayLayers,
                    const ResourceState& state);

    // Number of tracked slices (for diagnostics/testing).
    [[nodiscard]] std::uint32_t sliceCount() const {
        return static_cast<std::uint32_t>(slices_.size());
    }

    // Direct access to slices (for barrier compiler iteration).
    [[nodiscard]] const std::vector<SubresourceSlice>& slices() const {
        return slices_;
    }

private:
    std::vector<SubresourceSlice> slices_;
};

// Derive aspect flags from image format.
// Depth formats -> DEPTH_BIT, depth+stencil -> DEPTH|STENCIL, else COLOR.
[[nodiscard]] VkImageAspectFlags aspectFromFormat(VkFormat format);

} // namespace vksdl::graph

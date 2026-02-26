#pragma once

#include <vksdl/descriptor_allocator.hpp>
#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// Frame-aware descriptor allocator that manages N DescriptorAllocators
// internally (one per frame-in-flight). Hides the per-frame vector +
// indexing boilerplate that every Vulkan app writes identically.
//
// Usage:
//   auto fda = FrameDescriptorAllocator::create(device, framesInFlight);
//   // each frame:
//   fda.resetFrame(frameIndex);   // after fence wait
//   auto set = fda.allocate(frameIndex, layout);
//
// Descriptor safety: each frame slot is independent. resetFrame(i) only
// invalidates sets from slot i. Safe pattern: wait fence, reset, allocate,
// write, bind, submit.
//
// Thread safety: thread-confined (render loop thread).
class FrameDescriptorAllocator {
public:
    [[nodiscard]] static Result<FrameDescriptorAllocator> create(
        const Device& device, std::uint32_t framesInFlight,
        std::uint32_t maxSetsPerPool = 256);

    ~FrameDescriptorAllocator() = default;
    FrameDescriptorAllocator(FrameDescriptorAllocator&&) noexcept = default;
    FrameDescriptorAllocator& operator=(FrameDescriptorAllocator&&) noexcept = default;
    FrameDescriptorAllocator(const FrameDescriptorAllocator&) = delete;
    FrameDescriptorAllocator& operator=(const FrameDescriptorAllocator&) = delete;

    [[nodiscard]] Result<VkDescriptorSet> allocate(
        std::uint32_t frameIndex, VkDescriptorSetLayout layout);

    // Call after fence wait for the given frame index.
    void resetFrame(std::uint32_t frameIndex);

    [[nodiscard]] std::uint32_t frameCount() const {
        return static_cast<std::uint32_t>(allocators_.size());
    }

    [[nodiscard]] std::uint32_t allocatedSetCount(std::uint32_t frameIndex) const;

private:
    FrameDescriptorAllocator() = default;

    std::vector<DescriptorAllocator> allocators_;
};

} // namespace vksdl

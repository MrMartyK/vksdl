#include <vksdl/frame_descriptor_allocator.hpp>
#include <vksdl/device.hpp>

#include <cassert>

namespace vksdl {

Result<FrameDescriptorAllocator> FrameDescriptorAllocator::create(
    const Device& device, std::uint32_t framesInFlight,
    std::uint32_t maxSetsPerPool) {
    assert(framesInFlight > 0 && "framesInFlight must be at least 1");

    FrameDescriptorAllocator fda;
    fda.allocators_.reserve(framesInFlight);

    for (std::uint32_t i = 0; i < framesInFlight; ++i) {
        auto alloc = DescriptorAllocator::create(device, maxSetsPerPool);
        if (!alloc.ok()) return alloc.error();
        fda.allocators_.push_back(std::move(alloc).value());
    }

    return fda;
}

Result<VkDescriptorSet> FrameDescriptorAllocator::allocate(
    std::uint32_t frameIndex, VkDescriptorSetLayout layout) {
    assert(frameIndex < allocators_.size() && "frameIndex out of range");
    return allocators_[frameIndex].allocate(layout);
}

void FrameDescriptorAllocator::resetFrame(std::uint32_t frameIndex) {
    assert(frameIndex < allocators_.size() && "frameIndex out of range");
    allocators_[frameIndex].resetPools();
}

std::uint32_t FrameDescriptorAllocator::allocatedSetCount(
    std::uint32_t frameIndex) const {
    assert(frameIndex < allocators_.size() && "frameIndex out of range");
    return allocators_[frameIndex].allocatedSetCount();
}

} // namespace vksdl

#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// Frame-scoped descriptor set allocator with fixed-size pool chain.
// Designed for per-frame ephemeral descriptor sets. Pools are never freed
// individually -- call resetPools() once per frame when the fence signals.
//
// Differs from DescriptorPool:
//   - DescriptorPool: 2x doubling, for persistent sets across frames
//   - DescriptorAllocator: fixed-size chain, for ephemeral per-frame sets
//
// Descriptor safety: sets are ephemeral -- valid from allocate() until
// the next resetPools() call. Bound resources must outlive the frame's
// GPU submission (wait for fence before resetting).
//
// Thread safety: thread-confined. resetPools() after fence, allocate() during recording.
class DescriptorAllocator {
public:
    [[nodiscard]] static Result<DescriptorAllocator> create(
        const Device& device, std::uint32_t maxSetsPerPool = 256);

    ~DescriptorAllocator();
    DescriptorAllocator(DescriptorAllocator&&) noexcept;
    DescriptorAllocator& operator=(DescriptorAllocator&&) noexcept;
    DescriptorAllocator(const DescriptorAllocator&) = delete;
    DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;

    // Chains a new pool on exhaustion (no doubling -- fixed-size chain).
    [[nodiscard]] Result<VkDescriptorSet> allocate(VkDescriptorSetLayout layout);

    // Moves all pools to the free list for reuse. Call once per frame after fence wait.
    void resetPools();

    [[nodiscard]] std::uint32_t allocatedSetCount() const { return allocatedSets_; }
    [[nodiscard]] std::uint32_t poolCount() const;

private:
    DescriptorAllocator() = default;
    void destroy();
    [[nodiscard]] Result<VkDescriptorPool> grabPool();
    [[nodiscard]] Result<VkDescriptorPool> createPool();

    VkDevice                        device_          = VK_NULL_HANDLE;
    std::uint32_t                   maxSetsPerPool_  = 256;
    std::uint32_t                   allocatedSets_   = 0;
    VkDescriptorPool                currentPool_     = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool>   usedPools_;
    std::vector<VkDescriptorPool>   freePools_;
};

} // namespace vksdl

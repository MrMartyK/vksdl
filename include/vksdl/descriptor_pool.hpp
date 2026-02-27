#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;
class DescriptorLayout;

// Growable descriptor pool allocator for allocating many descriptor sets.
// When the current pool runs out, a new pool with 2x capacity is created.
// Individual sets cannot be freed -- use reset() to reclaim all at once.
//
// Descriptor safety: allocated VkDescriptorSets are valid until reset().
// Resources bound via DescriptorWriter must outlive any command buffer
// submission that references the set.
//
// Thread safety: thread-confined.
class DescriptorPool {
  public:
    [[nodiscard]] static Result<DescriptorPool> create(const Device& device,
                                                       std::uint32_t maxSetsPerPool = 64);

    ~DescriptorPool();
    DescriptorPool(DescriptorPool&&) noexcept;
    DescriptorPool& operator=(DescriptorPool&&) noexcept;
    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // Grows automatically on exhaustion (2x capacity per new pool).
    [[nodiscard]] Result<VkDescriptorSet> allocate(VkDescriptorSetLayout layout);
    [[nodiscard]] Result<VkDescriptorSet> allocate(const DescriptorLayout& layout);
    [[nodiscard]] Result<std::vector<VkDescriptorSet>> allocateMany(VkDescriptorSetLayout layout,
                                                                    std::uint32_t count);
    [[nodiscard]] Result<std::vector<VkDescriptorSet>> allocateMany(const DescriptorLayout& layout,
                                                                    std::uint32_t count);

    // Reclaims all descriptor sets for reuse. Does not free VkDescriptorPools.
    void reset();

    [[nodiscard]] std::uint32_t allocatedSetCount() const {
        return allocatedSets_;
    }

    [[nodiscard]] std::uint32_t poolCount() const {
        return static_cast<std::uint32_t>(pools_.size());
    }

  private:
    DescriptorPool() = default;
    void destroy();
    Result<void> addPool();

    VkDevice device_ = VK_NULL_HANDLE;
    std::uint32_t maxSetsPerPool_ = 64;
    std::uint32_t nextPoolScale_ = 1; // doubles each time
    std::uint32_t allocatedSets_ = 0;
    std::vector<VkDescriptorPool> pools_;
};

} // namespace vksdl

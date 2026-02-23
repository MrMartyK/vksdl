#include <vksdl/descriptor_pool.hpp>
#include <vksdl/device.hpp>

#include <cstdint>

namespace vksdl {

void DescriptorPool::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    for (auto p : pools_) vkDestroyDescriptorPool(device_, p, nullptr);
    pools_.clear();

    device_ = VK_NULL_HANDLE;
}

DescriptorPool::~DescriptorPool() { destroy(); }

DescriptorPool::DescriptorPool(DescriptorPool&& o) noexcept
    : device_(o.device_), maxSetsPerPool_(o.maxSetsPerPool_),
      nextPoolScale_(o.nextPoolScale_), allocatedSets_(o.allocatedSets_),
      pools_(std::move(o.pools_)) {
    o.device_ = VK_NULL_HANDLE;
}

DescriptorPool& DescriptorPool::operator=(DescriptorPool&& o) noexcept {
    if (this != &o) {
        destroy();
        device_         = o.device_;
        maxSetsPerPool_ = o.maxSetsPerPool_;
        nextPoolScale_  = o.nextPoolScale_;
        allocatedSets_  = o.allocatedSets_;
        pools_          = std::move(o.pools_);
        o.device_       = VK_NULL_HANDLE;
    }
    return *this;
}

Result<DescriptorPool> DescriptorPool::create(const Device& device,
                                               std::uint32_t maxSetsPerPool) {
    DescriptorPool dp;
    dp.device_         = device.vkDevice();
    dp.maxSetsPerPool_ = maxSetsPerPool;
    dp.nextPoolScale_  = 1;

    auto res = dp.addPool();
    if (!res.ok()) return res.error();

    return dp;
}

Result<void> DescriptorPool::addPool() {
    // Each pool has generous per-type counts so any layout combination can
    // allocate. Counts scale with nextPoolScale_ (doubles each time).
    std::uint32_t count = 16 * nextPoolScale_;
    std::uint32_t sets  = maxSetsPerPool_ * nextPoolScale_;

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          count},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,  count},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          count},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,  count},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  count},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           count},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,           count},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                 count},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,         count},
    };

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = sets;
    poolCI.poolSizeCount = static_cast<std::uint32_t>(
        sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolCI.pPoolSizes    = poolSizes;
    // No FREE_DESCRIPTOR_SET_BIT -- bulk reset only.

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorPool(device_, &poolCI, nullptr, &pool);
    if (vr != VK_SUCCESS) {
        return Error{"create descriptor pool", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorPool failed"};
    }

    pools_.push_back(pool);
    nextPoolScale_ *= 2;

    return {};
}

Result<VkDescriptorSet> DescriptorPool::allocate(VkDescriptorSetLayout layout) {
    // Try allocating from the last pool.
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = pools_.back();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult vr = vkAllocateDescriptorSets(device_, &allocInfo, &set);

    if (vr == VK_SUCCESS) {
        ++allocatedSets_;
        return set;
    }

    // Pool exhausted -- grow and retry.
    if (vr == VK_ERROR_OUT_OF_POOL_MEMORY ||
        vr == VK_ERROR_FRAGMENTED_POOL) {
        auto res = addPool();
        if (!res.ok()) return res.error();

        allocInfo.descriptorPool = pools_.back();
        vr = vkAllocateDescriptorSets(device_, &allocInfo, &set);
        if (vr == VK_SUCCESS) {
            ++allocatedSets_;
            return set;
        }
    }

    return Error{"allocate descriptor set", static_cast<std::int32_t>(vr),
                 "vkAllocateDescriptorSets failed after pool growth"};
}

void DescriptorPool::reset() {
    for (auto p : pools_) {
        vkResetDescriptorPool(device_, p, 0);
    }
    allocatedSets_ = 0;
}

} // namespace vksdl

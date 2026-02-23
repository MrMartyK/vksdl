#include <vksdl/descriptor_allocator.hpp>
#include <vksdl/device.hpp>

#include <cstdint>

namespace vksdl {

void DescriptorAllocator::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    for (auto p : usedPools_) vkDestroyDescriptorPool(device_, p, nullptr);
    for (auto p : freePools_) vkDestroyDescriptorPool(device_, p, nullptr);
    if (currentPool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, currentPool_, nullptr);

    usedPools_.clear();
    freePools_.clear();
    currentPool_ = VK_NULL_HANDLE;
    device_      = VK_NULL_HANDLE;
}

DescriptorAllocator::~DescriptorAllocator() { destroy(); }

DescriptorAllocator::DescriptorAllocator(DescriptorAllocator&& o) noexcept
    : device_(o.device_), maxSetsPerPool_(o.maxSetsPerPool_),
      allocatedSets_(o.allocatedSets_), currentPool_(o.currentPool_),
      usedPools_(std::move(o.usedPools_)),
      freePools_(std::move(o.freePools_)) {
    o.device_      = VK_NULL_HANDLE;
    o.currentPool_ = VK_NULL_HANDLE;
}

DescriptorAllocator& DescriptorAllocator::operator=(DescriptorAllocator&& o) noexcept {
    if (this != &o) {
        destroy();
        device_         = o.device_;
        maxSetsPerPool_ = o.maxSetsPerPool_;
        allocatedSets_  = o.allocatedSets_;
        currentPool_    = o.currentPool_;
        usedPools_      = std::move(o.usedPools_);
        freePools_      = std::move(o.freePools_);
        o.device_       = VK_NULL_HANDLE;
        o.currentPool_  = VK_NULL_HANDLE;
    }
    return *this;
}

Result<DescriptorAllocator> DescriptorAllocator::create(
    const Device& device, std::uint32_t maxSetsPerPool) {
    DescriptorAllocator da;
    da.device_         = device.vkDevice();
    da.maxSetsPerPool_ = maxSetsPerPool;

    auto pool = da.createPool();
    if (!pool.ok()) return pool.error();
    da.currentPool_ = pool.value();

    return da;
}

Result<VkDescriptorPool> DescriptorAllocator::createPool() {
    // Fixed type multipliers per pool, generous for any layout mix.
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxSetsPerPool_ * 4},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         maxSetsPerPool_ * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         maxSetsPerPool_ * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          maxSetsPerPool_ * 2},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          maxSetsPerPool_ * 2},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                maxSetsPerPool_ * 1},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,        maxSetsPerPool_ * 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, maxSetsPerPool_ * 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, maxSetsPerPool_ * 1},
    };

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = maxSetsPerPool_;
    poolCI.poolSizeCount = static_cast<std::uint32_t>(
        sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolCI.pPoolSizes    = poolSizes;
    // No FREE_DESCRIPTOR_SET_BIT -- bulk reset only.

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorPool(device_, &poolCI, nullptr, &pool);
    if (vr != VK_SUCCESS) {
        return Error{"create descriptor pool", static_cast<std::int32_t>(vr),
                     "vkCreateDescriptorPool failed for DescriptorAllocator"};
    }
    return pool;
}

Result<VkDescriptorPool> DescriptorAllocator::grabPool() {
    if (!freePools_.empty()) {
        VkDescriptorPool pool = freePools_.back();
        freePools_.pop_back();
        return pool;
    }
    return createPool();
}

Result<VkDescriptorSet> DescriptorAllocator::allocate(VkDescriptorSetLayout layout) {
    // Ensure we have a current pool.
    if (currentPool_ == VK_NULL_HANDLE) {
        auto pool = grabPool();
        if (!pool.ok()) return pool.error();
        currentPool_ = pool.value();
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = currentPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult vr = vkAllocateDescriptorSets(device_, &allocInfo, &set);

    if (vr == VK_SUCCESS) {
        ++allocatedSets_;
        return set;
    }

    // Pool exhausted -- retire current and grab a new one.
    if (vr == VK_ERROR_OUT_OF_POOL_MEMORY ||
        vr == VK_ERROR_FRAGMENTED_POOL) {
        usedPools_.push_back(currentPool_);
        currentPool_ = VK_NULL_HANDLE;

        auto newPool = grabPool();
        if (!newPool.ok()) return newPool.error();
        currentPool_ = newPool.value();

        allocInfo.descriptorPool = currentPool_;
        vr = vkAllocateDescriptorSets(device_, &allocInfo, &set);
        if (vr == VK_SUCCESS) {
            ++allocatedSets_;
            return set;
        }
    }

    return Error{"allocate descriptor set", static_cast<std::int32_t>(vr),
                 "vkAllocateDescriptorSets failed after pool chain growth"};
}

void DescriptorAllocator::resetPools() {
    for (auto p : usedPools_) {
        vkResetDescriptorPool(device_, p, 0);
        freePools_.push_back(p);
    }
    usedPools_.clear();

    if (currentPool_ != VK_NULL_HANDLE) {
        vkResetDescriptorPool(device_, currentPool_, 0);
        freePools_.push_back(currentPool_);
        currentPool_ = VK_NULL_HANDLE;
    }

    allocatedSets_ = 0;
}

std::uint32_t DescriptorAllocator::poolCount() const {
    return static_cast<std::uint32_t>(
        usedPools_.size() + freePools_.size()
        + (currentPool_ != VK_NULL_HANDLE ? 1 : 0));
}

} // namespace vksdl

#include <vksdl/transfer_queue.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/buffer.hpp>
#include <vksdl/device.hpp>

#include <vk_mem_alloc.h>

#include <cstring>

namespace vksdl {

// Cast void* allocator_ back to VmaAllocator for internal use.
static VmaAllocator toVma(void* p) { return static_cast<VmaAllocator>(p); }

void TransferQueue::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (timeline_ != VK_NULL_HANDLE)
        vkDestroySemaphore(device_, timeline_, nullptr);
    if (pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, pool_, nullptr);

    device_ = VK_NULL_HANDLE;
}

TransferQueue::~TransferQueue() { destroy(); }

TransferQueue::TransferQueue(TransferQueue&& o) noexcept
    : device_(o.device_), queue_(o.queue_), pool_(o.pool_),
      timeline_(o.timeline_), srcFamily_(o.srcFamily_),
      dstFamily_(o.dstFamily_), crossFamily_(o.crossFamily_),
      counter_(o.counter_), allocator_(o.allocator_) {
    o.device_   = VK_NULL_HANDLE;
    o.pool_     = VK_NULL_HANDLE;
    o.timeline_ = VK_NULL_HANDLE;
}

TransferQueue& TransferQueue::operator=(TransferQueue&& o) noexcept {
    if (this != &o) {
        destroy();
        device_      = o.device_;
        queue_       = o.queue_;
        pool_        = o.pool_;
        timeline_    = o.timeline_;
        srcFamily_   = o.srcFamily_;
        dstFamily_   = o.dstFamily_;
        crossFamily_ = o.crossFamily_;
        counter_     = o.counter_;
        allocator_   = o.allocator_;
        o.device_    = VK_NULL_HANDLE;
        o.pool_      = VK_NULL_HANDLE;
        o.timeline_  = VK_NULL_HANDLE;
    }
    return *this;
}

Result<TransferQueue> TransferQueue::create(const Device& device,
                                             const Allocator& alloc) {
    TransferQueue tq;
    tq.device_    = device.vkDevice();
    tq.allocator_ = static_cast<void*>(alloc.vmaAllocator());
    tq.dstFamily_ = device.queueFamilies().graphics;

    if (device.hasDedicatedTransfer()) {
        tq.srcFamily_   = device.queueFamilies().transfer;
        tq.queue_       = device.transferQueue();
        tq.crossFamily_ = true;
    } else {
        tq.srcFamily_ = device.queueFamilies().graphics;
        tq.queue_     = device.graphicsQueue();
        tq.crossFamily_ = false;
    }

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = tq.srcFamily_;

    VkResult vr = vkCreateCommandPool(tq.device_, &poolCI, nullptr, &tq.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create transfer command pool", static_cast<std::int32_t>(vr),
                     "vkCreateCommandPool failed"};
    }

    VkSemaphoreTypeCreateInfo timelineCI{};
    timelineCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCI.initialValue  = 0;

    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semCI.pNext = &timelineCI;

    vr = vkCreateSemaphore(tq.device_, &semCI, nullptr, &tq.timeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create transfer timeline", static_cast<std::int32_t>(vr),
                     "vkCreateSemaphore failed"};
    }

    return tq;
}

Result<PendingTransfer> TransferQueue::uploadAsync(const Buffer& dst,
                                                     const void* data,
                                                     VkDeviceSize size) {
    VkBufferCreateInfo stagingCI{};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size  = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingInfo{};

    VkResult vr = vmaCreateBuffer(toVma(allocator_), &stagingCI, &allocCI,
                                   &stagingBuf, &stagingAlloc, &stagingInfo);
    if (vr != VK_SUCCESS) {
        return Error{"async upload", static_cast<std::int32_t>(vr),
                     "failed to create staging buffer"};
    }

    std::memcpy(stagingInfo.pMappedData, data, static_cast<std::size_t>(size));

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = pool_;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(device_, &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(toVma(allocator_), stagingBuf, stagingAlloc);
        return Error{"async upload", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, stagingBuf, dst.vkBuffer(), 1, &region);

    // Release barrier: transfer ownership from transfer queue to graphics queue.
    if (crossFamily_) {
        VkBufferMemoryBarrier2 release{};
        release.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        release.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        release.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        release.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
        release.dstAccessMask = VK_ACCESS_2_NONE;
        release.srcQueueFamilyIndex = srcFamily_;
        release.dstQueueFamilyIndex = dstFamily_;
        release.buffer = dst.vkBuffer();
        release.offset = 0;
        release.size   = VK_WHOLE_SIZE;

        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers    = &release;

        vkCmdPipelineBarrier2(cmd, &dep);
    }

    vkEndCommandBuffer(cmd);

    ++counter_;
    std::uint64_t signalValue = counter_;

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues    = &signalValue;

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext                = &timelineInfo;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &timeline_;

    vr = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(toVma(allocator_), stagingBuf, stagingAlloc);
        return Error{"async upload", static_cast<std::int32_t>(vr),
                     "vkQueueSubmit failed"};
    }

    // CPU-blocks until transfer completes. Graphics queue remains unstalled.
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores    = &timeline_;
    waitInfo.pValues        = &signalValue;
    vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);

    vmaDestroyBuffer(toVma(allocator_), stagingBuf, stagingAlloc);

    PendingTransfer result;
    result.timelineValue          = signalValue;
    result.buffer                 = dst.vkBuffer();
    result.srcFamily              = srcFamily_;
    result.dstFamily              = dstFamily_;
    result.needsOwnershipTransfer = crossFamily_;

    return result;
}

void TransferQueue::waitIdle() {
    if (counter_ == 0) return;

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores    = &timeline_;
    waitInfo.pValues        = &counter_;
    vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);
}

bool TransferQueue::isComplete(std::uint64_t value) const {
    std::uint64_t completed = 0;
    vkGetSemaphoreCounterValue(device_, timeline_, &completed);
    return completed >= value;
}

void TransferQueue::insertAcquireBarrier(VkCommandBuffer cmd,
                                          const PendingTransfer& transfer) {
    if (!transfer.needsOwnershipTransfer) return;

    VkBufferMemoryBarrier2 acquire{};
    acquire.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    acquire.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    acquire.srcAccessMask = VK_ACCESS_2_NONE;
    acquire.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                            VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    acquire.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                            VK_ACCESS_2_INDEX_READ_BIT;
    acquire.srcQueueFamilyIndex = transfer.srcFamily;
    acquire.dstQueueFamilyIndex = transfer.dstFamily;
    acquire.buffer = transfer.buffer;
    acquire.offset = 0;
    acquire.size   = VK_WHOLE_SIZE;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &acquire;

    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace vksdl

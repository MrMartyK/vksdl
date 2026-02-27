#include <vksdl/compute_queue.hpp>
#include <vksdl/device.hpp>
#include "device_lost.hpp"

namespace vksdl {

void ComputeQueue::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (timeline_ != VK_NULL_HANDLE)
        vkDestroySemaphore(device_, timeline_, nullptr);
    if (pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, pool_, nullptr);

    device_ = VK_NULL_HANDLE;
}

ComputeQueue::~ComputeQueue() { destroy(); }

ComputeQueue::ComputeQueue(ComputeQueue&& o) noexcept
    : device_(o.device_), queue_(o.queue_), pool_(o.pool_),
      timeline_(o.timeline_), srcFamily_(o.srcFamily_),
      dstFamily_(o.dstFamily_), crossFamily_(o.crossFamily_),
      counter_(o.counter_), devicePtr_(o.devicePtr_) {
    o.device_    = VK_NULL_HANDLE;
    o.pool_      = VK_NULL_HANDLE;
    o.timeline_  = VK_NULL_HANDLE;
    o.devicePtr_ = nullptr;
}

ComputeQueue& ComputeQueue::operator=(ComputeQueue&& o) noexcept {
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
        devicePtr_   = o.devicePtr_;
        o.device_    = VK_NULL_HANDLE;
        o.pool_      = VK_NULL_HANDLE;
        o.timeline_  = VK_NULL_HANDLE;
        o.devicePtr_ = nullptr;
    }
    return *this;
}

Result<ComputeQueue> ComputeQueue::create(const Device& device) {
    ComputeQueue cq;
    cq.device_    = device.vkDevice();
    cq.devicePtr_ = &device;
    cq.dstFamily_ = device.queueFamilies().graphics;

    if (device.hasDedicatedCompute()) {
        cq.srcFamily_   = device.queueFamilies().compute;
        cq.queue_       = device.computeQueue();
        cq.crossFamily_ = true;
    } else {
        cq.srcFamily_   = device.queueFamilies().graphics;
        cq.queue_       = device.graphicsQueue();
        cq.crossFamily_ = false;
    }

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = cq.srcFamily_;

    VkResult vr = vkCreateCommandPool(cq.device_, &poolCI, nullptr, &cq.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create compute command pool", static_cast<std::int32_t>(vr),
                     "vkCreateCommandPool failed"};
    }

    VkSemaphoreTypeCreateInfo timelineCI{};
    timelineCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCI.initialValue  = 0;

    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semCI.pNext = &timelineCI;

    vr = vkCreateSemaphore(cq.device_, &semCI, nullptr, &cq.timeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create compute timeline", static_cast<std::int32_t>(vr),
                     "vkCreateSemaphore failed"};
    }

    return cq;
}

Result<PendingCompute> ComputeQueue::submit(
    std::function<void(VkCommandBuffer)> record) {

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = pool_;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult vr = vkAllocateCommandBuffers(device_, &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        return Error{"compute submit", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    record(cmd);

    vkEndCommandBuffer(cmd);

    return submitInternal(cmd);
}

Result<PendingCompute> ComputeQueue::submit(VkCommandBuffer preRecorded) {
    return submitInternal(preRecorded);
}

Result<PendingCompute> ComputeQueue::submitInternal(VkCommandBuffer cmd) {
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

    VkResult vr = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        if (devicePtr_) detail::checkDeviceLost(*devicePtr_, vr);
        return Error{"compute submit", static_cast<std::int32_t>(vr),
                     "vkQueueSubmit failed"};
    }

    PendingCompute result;
    result.timelineValue          = signalValue;
    result.srcFamily              = srcFamily_;
    result.dstFamily              = dstFamily_;
    result.needsOwnershipTransfer = crossFamily_;

    return result;
}

void ComputeQueue::waitIdle() {
    if (counter_ == 0) return;

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores    = &timeline_;
    waitInfo.pValues        = &counter_;
    // VKSDL_BLOCKING_WAIT: explicit queue drain requested by caller.
    vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);
}

bool ComputeQueue::isComplete(std::uint64_t value) const {
    std::uint64_t completed = 0;
    vkGetSemaphoreCounterValue(device_, timeline_, &completed);
    return completed >= value;
}

void ComputeQueue::waitFor(std::uint64_t value) const {
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores    = &timeline_;
    waitInfo.pValues        = &value;
    // VKSDL_BLOCKING_WAIT: caller requested explicit wait on a specific value.
    VkResult vr = vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);
    if (vr != VK_SUCCESS && devicePtr_) {
        detail::checkDeviceLost(*devicePtr_, vr);
    }
}

void ComputeQueue::insertBufferAcquireBarrier(VkCommandBuffer cmd,
                                               VkBuffer buffer,
                                               VkPipelineStageFlags2 dstStage,
                                               VkAccessFlags2 dstAccess,
                                               const PendingCompute& pending) {
    if (!pending.needsOwnershipTransfer) return;

    VkBufferMemoryBarrier2 acquire{};
    acquire.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    acquire.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    acquire.srcAccessMask       = VK_ACCESS_2_NONE;
    acquire.dstStageMask        = dstStage;
    acquire.dstAccessMask       = dstAccess;
    acquire.srcQueueFamilyIndex = pending.srcFamily;
    acquire.dstQueueFamilyIndex = pending.dstFamily;
    acquire.buffer              = buffer;
    acquire.offset              = 0;
    acquire.size                = VK_WHOLE_SIZE;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &acquire;

    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace vksdl

#include <vksdl/timeline_sync.hpp>
#include <vksdl/window.hpp>
#include "device_lost.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace vksdl {

void TimelineSync::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (timeline_ != VK_NULL_HANDLE)
        vkDestroySemaphore(device_, timeline_, nullptr);
    for (auto s : renderDone_) vkDestroySemaphore(device_, s, nullptr);

    if (pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, pool_, nullptr);

    device_ = VK_NULL_HANDLE;
}

TimelineSync::~TimelineSync() { destroy(); }

TimelineSync::TimelineSync(TimelineSync&& o) noexcept
    : device_(o.device_), devicePtr_(o.devicePtr_),
      pool_(o.pool_), timeline_(o.timeline_),
      count_(o.count_), current_(o.current_), counter_(o.counter_),
      cmds_(std::move(o.cmds_)),
      renderDone_(std::move(o.renderDone_)) {
    o.device_    = VK_NULL_HANDLE;
    o.devicePtr_ = nullptr;
    o.pool_      = VK_NULL_HANDLE;
    o.timeline_  = VK_NULL_HANDLE;
}

TimelineSync& TimelineSync::operator=(TimelineSync&& o) noexcept {
    if (this != &o) {
        destroy();
        device_     = o.device_;
        devicePtr_  = o.devicePtr_;
        pool_       = o.pool_;
        timeline_   = o.timeline_;
        count_      = o.count_;
        current_    = o.current_;
        counter_    = o.counter_;
        cmds_       = std::move(o.cmds_);
        renderDone_ = std::move(o.renderDone_);
        o.device_    = VK_NULL_HANDLE;
        o.devicePtr_ = nullptr;
        o.pool_      = VK_NULL_HANDLE;
        o.timeline_  = VK_NULL_HANDLE;
    }
    return *this;
}

Result<TimelineSync> TimelineSync::create(const Device& device,
                                           std::uint32_t framesInFlight) {
    if (device.queueFamilies().graphics == UINT32_MAX) {
        return Error{"create timeline sync", 0,
                     "Device has no graphics queue family"};
    }

    TimelineSync ts;
    ts.device_    = device.vkDevice();
    ts.devicePtr_ = &device;
    ts.count_     = framesInFlight;

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = device.queueFamilies().graphics;

    VkResult vr = vkCreateCommandPool(ts.device_, &poolCI, nullptr, &ts.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create command pool", static_cast<std::int32_t>(vr),
                     "vkCreateCommandPool failed"};
    }

    ts.cmds_.resize(framesInFlight);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = ts.pool_;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = framesInFlight;

    vr = vkAllocateCommandBuffers(ts.device_, &allocInfo, ts.cmds_.data());
    if (vr != VK_SUCCESS) {
        return Error{"allocate command buffers", static_cast<std::int32_t>(vr),
                     "vkAllocateCommandBuffers failed"};
    }

    // One timeline semaphore replaces N per-frame fences; monotonic counter
    // allows waiting on the oldest in-flight frame without knowing which slot it used.
    VkSemaphoreTypeCreateInfo timelineCI{};
    timelineCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCI.initialValue  = 0;

    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semCI.pNext = &timelineCI;

    vr = vkCreateSemaphore(ts.device_, &semCI, nullptr, &ts.timeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create timeline semaphore", static_cast<std::int32_t>(vr),
                     "vkCreateSemaphore failed"};
    }

    // vkQueuePresentKHR requires binary semaphores; timeline semaphores cannot
    // be used for presentation signaling.
    VkSemaphoreCreateInfo binSemCI{};
    binSemCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    ts.renderDone_.resize(framesInFlight);
    for (std::uint32_t i = 0; i < framesInFlight; ++i) {
        vr = vkCreateSemaphore(ts.device_, &binSemCI, nullptr, &ts.renderDone_[i]);
        if (vr != VK_SUCCESS) {
            return Error{"create semaphore", static_cast<std::int32_t>(vr),
                         "failed for renderDone[" + std::to_string(i) + "]"};
        }
    }

    return ts;
}

Result<TimelineFrame> TimelineSync::nextFrame() {
    std::uint32_t i = current_;

    // Wait for the oldest in-flight frame (counter_ - count_ + 1).
    // First N frames skip the wait since no work has been submitted yet.
    if (counter_ >= count_) {
        std::uint64_t waitValue = counter_ - count_ + 1;

        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &timeline_;
        waitInfo.pValues        = &waitValue;

        // VKSDL_BLOCKING_WAIT: frame-slot timeline wait before command reuse.
        VkResult vr = vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);
        if (vr != VK_SUCCESS) {
            if (devicePtr_) detail::checkDeviceLost(*devicePtr_, vr);
            return Error{"wait for timeline semaphore",
                         static_cast<std::int32_t>(vr),
                         "vkWaitSemaphores failed for frame " + std::to_string(i)};
        }
    }

    VkResult vr = vkResetCommandBuffer(cmds_[i], 0);
    if (vr != VK_SUCCESS) {
        return Error{"reset command buffer", static_cast<std::int32_t>(vr),
                     "vkResetCommandBuffer failed"};
    }

    ++counter_;

    TimelineFrame frame;
    frame.cmd   = cmds_[i];
    frame.value = counter_;
    frame.index = i;

    current_ = (current_ + 1) % count_;

    return frame;
}

void submitTimelineFrame(VkQueue queue, const TimelineSync& sync,
                         const TimelineFrame& frame,
                         VkSemaphore imageReady, VkPipelineStageFlags waitStage) {
    VkSemaphore waitSemaphores[] = {imageReady};
    VkPipelineStageFlags waitStages[] = {waitStage};

    VkSemaphore signalSemaphores[] = {
        sync.vkTimelineSemaphore(),
        sync.renderDoneSemaphore(frame.index),
    };

    // Binary semaphore values are ignored by the spec (must be 0).
    std::uint64_t waitValues[]   = {0};
    std::uint64_t signalValues[] = {frame.value, 0};

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount   = 1;
    timelineInfo.pWaitSemaphoreValues      = waitValues;
    timelineInfo.signalSemaphoreValueCount = 2;
    timelineInfo.pSignalSemaphoreValues    = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext                = &timelineInfo;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &frame.cmd;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    // No fence needed -- timeline semaphore handles CPU-GPU sync.
    VkResult vr = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    (void)vr; // timeline semaphore signals completion or device lost
}

Result<TimelineAcquiredFrame> acquireTimelineFrame(
    Swapchain& swapchain, TimelineSync& sync,
    const Device& device, const Window& window) {

    auto frameRes = sync.nextFrame();
    if (!frameRes.ok()) return frameRes.error();

    auto img = swapchain.nextImage();
    if (!img.ok()) {
        // Out-of-date: recreate and retry once.
        device.waitIdle();
        auto recreateRes = swapchain.recreate(window.pixelSize());
        if (!recreateRes.ok()) return recreateRes.error();

        img = swapchain.nextImage();
        if (!img.ok()) return img.error();
    }

    return TimelineAcquiredFrame{frameRes.value(), img.value()};
}

void presentTimelineFrame(const Device& device, Swapchain& swapchain,
                          const Window& window, TimelineSync& sync,
                          const TimelineFrame& frame,
                          const SwapchainImage& image,
                          VkPipelineStageFlags waitStage) {

    submitTimelineFrame(device.graphicsQueue(), sync, frame,
                        image.imageReady, waitStage);

    VkResult result = swapchain.present(
        device.presentQueue(), image.index,
        sync.renderDoneSemaphore(frame.index));

    if (result != VK_SUCCESS) {
        if (detail::checkDeviceLost(device, result)) {
            return;
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            device.waitIdle();
            auto recreateRes = swapchain.recreate(window.pixelSize());
#ifndef NDEBUG
            if (!recreateRes.ok()) {
                std::fprintf(stderr, "vksdl: presentTimelineFrame: swapchain recreate failed: %s\n",
                             recreateRes.error().format().c_str());
            }
#else
            (void)recreateRes;
#endif
        }
    }
}

} // namespace vksdl

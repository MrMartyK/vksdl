#include <vksdl/frames.hpp>
#include <vksdl/window.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

namespace vksdl {

void FrameSync::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    for (auto s : drawDone_) vkDestroySemaphore(device_, s, nullptr);
    for (auto f : fences_)   vkDestroyFence(device_, f, nullptr);

    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, pool_, nullptr);
    }

    device_ = VK_NULL_HANDLE;
}

FrameSync::~FrameSync() { destroy(); }

FrameSync::FrameSync(FrameSync&& o) noexcept
    : device_(o.device_), pool_(o.pool_), count_(o.count_), current_(o.current_),
      cmds_(std::move(o.cmds_)),
      drawDone_(std::move(o.drawDone_)),
      fences_(std::move(o.fences_)) {
    o.device_ = VK_NULL_HANDLE;
    o.pool_   = VK_NULL_HANDLE;
}

FrameSync& FrameSync::operator=(FrameSync&& o) noexcept {
    if (this != &o) {
        destroy();
        device_   = o.device_;
        pool_     = o.pool_;
        count_    = o.count_;
        current_  = o.current_;
        cmds_     = std::move(o.cmds_);
        drawDone_ = std::move(o.drawDone_);
        fences_   = std::move(o.fences_);
        o.device_ = VK_NULL_HANDLE;
        o.pool_   = VK_NULL_HANDLE;
    }
    return *this;
}

Result<FrameSync> FrameSync::create(const Device& device, std::uint32_t count) {
    if (device.queueFamilies().graphics == UINT32_MAX) {
        return Error{"create command pool", 0,
                     "Device has no graphics queue family"};
    }

    FrameSync fs;
    fs.device_ = device.vkDevice();
    fs.count_  = count;

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = device.queueFamilies().graphics;

    VkResult vr = vkCreateCommandPool(fs.device_, &poolCI, nullptr, &fs.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create command pool", static_cast<std::int32_t>(vr),
                     "vkCreateCommandPool failed"};
    }

    fs.cmds_.resize(count);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = fs.pool_;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = count;

    vr = vkAllocateCommandBuffers(fs.device_, &allocInfo, fs.cmds_.data());
    if (vr != VK_SUCCESS) {
        return Error{"allocate command buffers", static_cast<std::int32_t>(vr),
                     "vkAllocateCommandBuffers failed"};
    }

    // Image-acquire semaphores live in the Swapchain (one per swapchain image),
    // not here, to avoid reuse hazards during presentation.
    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    fs.drawDone_.resize(count);
    fs.fences_.resize(count);

    for (std::uint32_t i = 0; i < count; ++i) {
        vr = vkCreateSemaphore(fs.device_, &semCI, nullptr, &fs.drawDone_[i]);
        if (vr != VK_SUCCESS) {
            return Error{"create semaphore", static_cast<std::int32_t>(vr),
                         "failed for drawDone[" + std::to_string(i) + "]"};
        }

        vr = vkCreateFence(fs.device_, &fenceCI, nullptr, &fs.fences_[i]);
        if (vr != VK_SUCCESS) {
            return Error{"create fence", static_cast<std::int32_t>(vr),
                         "failed for fence[" + std::to_string(i) + "]"};
        }
    }

    return fs;
}

Result<Frame> FrameSync::nextFrame() {
    std::uint32_t i = current_;

    // VKSDL_BLOCKING_WAIT: frame-slot fence wait before command/fence reuse.
    VkResult vr = vkWaitForFences(device_, 1, &fences_[i], VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) {
        return Error{"wait for fence", static_cast<std::int32_t>(vr),
                     "vkWaitForFences failed for frame " + std::to_string(i)};
    }

    vr = vkResetFences(device_, 1, &fences_[i]);
    if (vr != VK_SUCCESS) {
        return Error{"reset fence", static_cast<std::int32_t>(vr),
                     "vkResetFences failed"};
    }

    vr = vkResetCommandBuffer(cmds_[i], 0);
    if (vr != VK_SUCCESS) {
        return Error{"reset command buffer", static_cast<std::int32_t>(vr),
                     "vkResetCommandBuffer failed"};
    }

    Frame frame;
    frame.cmd      = cmds_[i];
    frame.drawDone = drawDone_[i];
    frame.fence    = fences_[i];
    frame.index    = i;

    current_ = (current_ + 1) % count_;

    return frame;
}

void beginOneTimeCommands(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
}

void endCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
}

Result<void> endSubmitOneShotBlocking(
    VkQueue queue, VkCommandBuffer cmd, VkFence fence) {
    VkResult vr = vkEndCommandBuffer(cmd);
    if (vr != VK_SUCCESS) {
        return Error{"end one-shot command buffer",
                     static_cast<std::int32_t>(vr),
                     "vkEndCommandBuffer failed"};
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vr = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (vr != VK_SUCCESS) {
        return Error{"submit one-shot command buffer",
                     static_cast<std::int32_t>(vr),
                     "vkQueueSubmit failed"};
    }

    // VKSDL_BLOCKING_WAIT: explicit one-shot helper; waits for completion by design.
    vr = vkQueueWaitIdle(queue);
    if (vr != VK_SUCCESS) {
        return Error{"wait one-shot queue idle",
                     static_cast<std::int32_t>(vr),
                     "vkQueueWaitIdle failed"};
    }

    return {};
}

void submitFrame(VkQueue queue, const Frame& frame,
                 VkSemaphore imageReady, VkPipelineStageFlags waitStage) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = &imageReady;
    submitInfo.pWaitDstStageMask    = &waitStage;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &frame.cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &frame.drawDone;
    VkResult vr = vkQueueSubmit(queue, 1, &submitInfo, frame.fence);
    (void)vr; // fire-and-forget; fence signals completion or device lost
}

Result<AcquiredFrame> acquireFrame(
    Swapchain& swapchain, FrameSync& frames,
    const Device& device, const Window& window) {

    auto frameRes = frames.nextFrame();
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

    return AcquiredFrame{frameRes.value(), img.value()};
}

void presentFrame(const Device& device, Swapchain& swapchain,
                  const Window& window, const Frame& frame,
                  const SwapchainImage& image,
                  VkPipelineStageFlags waitStage) {

    submitFrame(device.graphicsQueue(), frame, image.imageReady, waitStage);

    VkResult result = swapchain.present(
        device.presentQueue(), image.index, frame.drawDone);

    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_SUBOPTIMAL_KHR) {
        device.waitIdle();
        auto recreateRes = swapchain.recreate(window.pixelSize());
#ifndef NDEBUG
        if (!recreateRes.ok()) {
            std::fprintf(stderr, "vksdl: presentFrame: swapchain recreate failed: %s\n",
                         recreateRes.error().format().c_str());
        }
#else
        (void)recreateRes;
#endif
    }
}

} // namespace vksdl

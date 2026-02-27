#pragma once

#include <vksdl/device.hpp>
#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

// Per-frame resources returned by TimelineSync::nextFrame().
// Plain data -- does not own anything. TimelineSync manages lifetime.
struct TimelineFrame {
    VkCommandBuffer cmd   = VK_NULL_HANDLE; // already reset, not yet begun
    std::uint64_t   value = 0;              // timeline value for this frame
    std::uint32_t   index = 0;              // which frame-in-flight slot (0..N-1)
};

// Bundled result of acquireTimelineFrame(). Contains both frame sync objects
// and the acquired swapchain image. Plain data -- does not own anything.
struct TimelineAcquiredFrame {
    TimelineFrame  frame;
    SwapchainImage image;
};

// Timeline semaphore based frame synchronization (Vulkan 1.3 core).
// Replaces N fences with one monotonic timeline semaphore for CPU-GPU sync.
// Binary semaphores are still required for present (Vulkan spec requirement),
// so this class holds N round-robin binary "renderDone" semaphores internally.
// Image-acquire semaphores live in the Swapchain, same as FrameSync.
//
// Thread safety: thread-confined (render loop thread).
class TimelineSync {
public:
    [[nodiscard]] static Result<TimelineSync> create(const Device& device,
                                                      std::uint32_t framesInFlight = 2);

    ~TimelineSync();
    TimelineSync(TimelineSync&&) noexcept;
    TimelineSync& operator=(TimelineSync&&) noexcept;
    TimelineSync(const TimelineSync&) = delete;
    TimelineSync& operator=(const TimelineSync&) = delete;

    // Wait for this frame slot to be available, reset its command buffer.
    // Returns the frame's handles. Command buffer is reset but NOT begun --
    // call vkBeginCommandBuffer yourself (you choose the flags).
    [[nodiscard]] Result<TimelineFrame> nextFrame();

    [[nodiscard]] VkSemaphore    vkTimelineSemaphore() const { return timeline_; }
    [[nodiscard]] std::uint64_t  currentValue()        const { return counter_; }
    [[nodiscard]] std::uint32_t  count()               const { return count_; }

    // Binary renderDone semaphore for the given frame slot (for present).
    [[nodiscard]] VkSemaphore renderDoneSemaphore(std::uint32_t index) const {
        return renderDone_[index];
    }

private:
    TimelineSync() = default;
    void destroy();

    VkDevice                     device_    = VK_NULL_HANDLE;
    const Device*                devicePtr_ = nullptr;
    VkCommandPool                pool_      = VK_NULL_HANDLE;
    VkSemaphore                  timeline_ = VK_NULL_HANDLE;
    std::uint32_t                count_    = 0;
    std::uint32_t                current_  = 0;
    std::uint64_t                counter_  = 0; // monotonic, incremented each frame
    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore>     renderDone_; // binary, for present
};

// Submit a recorded command buffer for this frame using timeline semaphore.
// Waits on imageReady at waitStage, signals timeline + binary renderDone.
void submitTimelineFrame(VkQueue queue, const TimelineSync& sync,
                         const TimelineFrame& frame,
                         VkSemaphore imageReady, VkPipelineStageFlags waitStage);

// Acquire a frame + swapchain image in one call (timeline variant).
// Handles out-of-date by recreating the swapchain and retrying once.
[[nodiscard]] Result<TimelineAcquiredFrame> acquireTimelineFrame(
    Swapchain& swapchain, TimelineSync& sync,
    const Device& device, const Window& window);

// Submit + present in one call (timeline variant).
// Handles out-of-date/suboptimal by recreating the swapchain.
void presentTimelineFrame(const Device& device, Swapchain& swapchain,
                          const Window& window, TimelineSync& sync,
                          const TimelineFrame& frame,
                          const SwapchainImage& image,
                          VkPipelineStageFlags waitStage);

} // namespace vksdl

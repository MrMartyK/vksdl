#pragma once

#include <vksdl/device.hpp>
#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

// Per-frame resources returned by FrameSync::nextFrame().
// Plain data -- does not own anything. FrameSync manages lifetime.
// Note: image-acquire semaphore lives in SwapchainImage, not here.
struct Frame {
    VkCommandBuffer cmd = VK_NULL_HANDLE;  // already reset, not yet begun
    VkSemaphore drawDone = VK_NULL_HANDLE; // signal when rendering finishes
    VkFence fence = VK_NULL_HANDLE;        // CPU waits before reusing this slot
    std::uint32_t index = 0;               // which frame-in-flight slot (0..N-1)
};

// Bundled result of acquireFrame(). Contains both frame sync objects and
// the acquired swapchain image. Plain data -- does not own anything.
struct AcquiredFrame {
    Frame frame;
    SwapchainImage image;
};

// Owns a command pool, N command buffers, and N sets of sync objects
// (semaphores + fences) for frames-in-flight.
// Pre-allocates everything at creation -- zero per-frame allocations.
//
// Thread safety: thread-confined (render loop thread).
class FrameSync {
  public:
    [[nodiscard]] static Result<FrameSync> create(const Device& device, std::uint32_t count = 2);

    ~FrameSync();
    FrameSync(FrameSync&&) noexcept;
    FrameSync& operator=(FrameSync&&) noexcept;
    FrameSync(const FrameSync&) = delete;
    FrameSync& operator=(const FrameSync&) = delete;

    // Wait for this frame's fence, reset it, reset the command buffer.
    // Returns the frame's handles. Command buffer is reset but NOT begun --
    // call vkBeginCommandBuffer yourself (you choose the flags).
    [[nodiscard]] Result<Frame> nextFrame();

    [[nodiscard]] std::uint32_t count() const {
        return count_;
    }

  private:
    FrameSync() = default;
    void destroy();

    VkDevice device_ = VK_NULL_HANDLE;
    const Device* devicePtr_ = nullptr;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    std::uint32_t count_ = 0;
    std::uint32_t current_ = 0;
    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> drawDone_;
    std::vector<VkFence> fences_;
};

// Begin a command buffer with ONE_TIME_SUBMIT flag.
void beginOneTimeCommands(VkCommandBuffer cmd);

// End a command buffer (symmetric counterpart to beginOneTimeCommands).
void endCommands(VkCommandBuffer cmd);

// End, submit, and wait for one-shot command buffers.
// Explicitly blocking by design.
[[nodiscard]] Result<void> endSubmitOneShotBlocking(VkQueue queue, VkCommandBuffer cmd,
                                                    VkFence fence = VK_NULL_HANDLE);

// Submit a recorded command buffer for this frame.
// Waits on imageReady at waitStage, signals frame.drawDone, fences frame.fence.
void submitFrame(VkQueue queue, const Frame& frame, VkSemaphore imageReady,
                 VkPipelineStageFlags waitStage);

// Acquire a frame + swapchain image in one call. Handles out-of-date by
// recreating the swapchain and retrying once. Returns error only on hard failure.
[[nodiscard]] Result<AcquiredFrame> acquireFrame(Swapchain& swapchain, FrameSync& frames,
                                                 const Device& device, const Window& window);

// Submit + present in one call. The SDL_GL_SwapWindow equivalent.
// Handles out-of-date/suboptimal by recreating the swapchain.
void presentFrame(const Device& device, Swapchain& swapchain, const Window& window,
                  const Frame& frame, const SwapchainImage& image, VkPipelineStageFlags waitStage);

} // namespace vksdl

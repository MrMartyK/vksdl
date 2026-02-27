#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>

namespace vksdl {

class Device;

// Result of a non-blocking compute submission. Contains the info needed for an
// ownership transfer barrier on the graphics queue (when the compute queue
// family differs from the graphics queue family).
struct PendingCompute {
    std::uint64_t timelineValue          = 0;
    std::uint32_t srcFamily              = UINT32_MAX; // compute queue family
    std::uint32_t dstFamily              = UINT32_MAX; // graphics queue family
    bool          needsOwnershipTransfer = false;
};

// Async compute queue with timeline semaphore synchronization.
// Uses the dedicated compute queue when available, falls back to graphics.
// submit() is non-blocking -- returns PendingCompute immediately.
//
// Thread safety: thread-confined. Async internally but single-threaded API.
class ComputeQueue {
public:
    [[nodiscard]] static Result<ComputeQueue> create(const Device& device);

    ~ComputeQueue();
    ComputeQueue(ComputeQueue&&) noexcept;
    ComputeQueue& operator=(ComputeQueue&&) noexcept;
    ComputeQueue(const ComputeQueue&) = delete;
    ComputeQueue& operator=(const ComputeQueue&) = delete;

    // Submit a compute workload via lambda. Allocates a command buffer,
    // calls record(cmd), submits, returns immediately. Non-blocking.
    [[nodiscard]] Result<PendingCompute> submit(
        std::function<void(VkCommandBuffer)> record);

    // Submit a pre-recorded command buffer. Zero-overhead path for callers
    // who manage their own command buffers. Non-blocking.
    [[nodiscard]] Result<PendingCompute> submit(VkCommandBuffer preRecorded);

    void waitIdle();

    [[nodiscard]] bool         isComplete(std::uint64_t value) const;
    void                       waitFor(std::uint64_t value) const;

    [[nodiscard]] VkSemaphore   vkTimelineSemaphore() const { return timeline_; }
    [[nodiscard]] std::uint64_t currentValue()        const { return counter_; }
    [[nodiscard]] VkCommandPool vkCommandPool()       const { return pool_; }
    [[nodiscard]] std::uint32_t queueFamily()         const { return srcFamily_; }

    // Insert an acquire barrier in a graphics command buffer for a buffer
    // produced by a compute submission. No-op when same queue family.
    static void insertBufferAcquireBarrier(VkCommandBuffer cmd,
                                           VkBuffer buffer,
                                           VkPipelineStageFlags2 dstStage,
                                           VkAccessFlags2 dstAccess,
                                           const PendingCompute& pending);

private:
    ComputeQueue() = default;
    void destroy();

    [[nodiscard]] Result<PendingCompute> submitInternal(VkCommandBuffer cmd);

    VkDevice      device_      = VK_NULL_HANDLE;
    VkQueue       queue_       = VK_NULL_HANDLE;
    VkCommandPool pool_        = VK_NULL_HANDLE;
    VkSemaphore   timeline_    = VK_NULL_HANDLE;
    std::uint32_t srcFamily_   = UINT32_MAX;
    std::uint32_t dstFamily_   = UINT32_MAX;
    bool          crossFamily_ = false;
    std::uint64_t counter_     = 0;
};

} // namespace vksdl

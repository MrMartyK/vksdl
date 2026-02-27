#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vksdl {

class Allocator;
class Buffer;
class Device;

// Result of an async transfer. Contains the info needed for an ownership
// transfer barrier on the graphics queue (when the transfer queue family
// differs from the graphics queue family).
struct PendingTransfer {
    std::uint64_t timelineValue          = 0;
    VkBuffer      buffer                 = VK_NULL_HANDLE;
    std::uint32_t srcFamily              = UINT32_MAX; // transfer queue family
    std::uint32_t dstFamily              = UINT32_MAX; // graphics queue family
    bool          needsOwnershipTransfer = false;
};

// Asynchronous transfer queue using a dedicated transfer family (if available).
// Uses a timeline semaphore for synchronization. Falls back to the graphics
// queue when no dedicated transfer family exists.
//
// Thread safety: thread-confined. Async internally but single-threaded API.
class TransferQueue {
public:
    [[nodiscard]] static Result<TransferQueue> create(const Device& device,
                                                       const Allocator& alloc);

    ~TransferQueue();
    TransferQueue(TransferQueue&&) noexcept;
    TransferQueue& operator=(TransferQueue&&) noexcept;
    TransferQueue(const TransferQueue&) = delete;
    TransferQueue& operator=(const TransferQueue&) = delete;

    // CPU-blocking: waits for the transfer to complete before returning.
    // The returned PendingTransfer is used for cross-family ownership transfer.
    [[nodiscard]] Result<PendingTransfer> uploadAsync(const Buffer& dst,
                                                       const void* data,
                                                       VkDeviceSize size);

    void waitIdle();

    [[nodiscard]] bool isComplete(std::uint64_t value) const;

    [[nodiscard]] VkSemaphore vkTimelineSemaphore() const { return timeline_; }
    [[nodiscard]] std::uint64_t currentValue()      const { return counter_; }

    // Insert an acquire barrier in a graphics command buffer to take ownership
    // of a buffer transferred from the transfer queue.
    // No-op when needsOwnershipTransfer is false (same queue family).
    static void insertAcquireBarrier(VkCommandBuffer cmd,
                                      const PendingTransfer& transfer);

private:
    TransferQueue() = default;
    void destroy();

    VkDevice          device_       = VK_NULL_HANDLE;
    VkQueue           queue_        = VK_NULL_HANDLE;
    VkCommandPool     pool_         = VK_NULL_HANDLE;
    VkSemaphore       timeline_     = VK_NULL_HANDLE;
    std::uint32_t     srcFamily_    = UINT32_MAX;
    std::uint32_t     dstFamily_    = UINT32_MAX;
    bool              crossFamily_  = false;
    std::uint64_t     counter_      = 0;

    // VMA allocator handle stored as void* to avoid pulling vk_mem_alloc.h
    // into the public header. Cast to VmaAllocator in the .cpp file.
    void*             allocator_    = nullptr;
    const Device*     devicePtr_    = nullptr; // non-owning, for device-lost reporting
};

} // namespace vksdl

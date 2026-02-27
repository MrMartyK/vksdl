#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

namespace vksdl {

class Device;

// RAII wrapper for VkEvent. Created with VK_EVENT_CREATE_DEVICE_ONLY_BIT
// for optimal GPU-only synchronization.
//
// Named GpuEvent to avoid collision with vksdl::Event (window input event).
//
// Thread safety: thread-confined. Record commands on the same command buffer.
class GpuEvent {
public:
    [[nodiscard]] static Result<GpuEvent> create(const Device& device);

    ~GpuEvent();
    GpuEvent(GpuEvent&&) noexcept;
    GpuEvent& operator=(GpuEvent&&) noexcept;
    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    [[nodiscard]] VkEvent native()  const { return event_; }
    [[nodiscard]] VkEvent vkEvent() const { return native(); }

    // Record vkCmdSetEvent2. The event is signaled when srcStage/srcAccess
    // operations on preceding commands complete.
    void set(VkCommandBuffer cmd,
             VkPipelineStageFlags2 srcStage,
             VkAccessFlags2 srcAccess) const;

    // Record vkCmdWaitEvents2. The command buffer waits until the event is
    // signaled, then makes dstStage/dstAccess operations visible.
    void wait(VkCommandBuffer cmd,
              VkPipelineStageFlags2 dstStage,
              VkAccessFlags2 dstAccess) const;

    // Full-control wait using a caller-provided VkDependencyInfo.
    void wait(VkCommandBuffer cmd,
              const VkDependencyInfo& depInfo) const;

    // Record vkCmdResetEvent2.
    void reset(VkCommandBuffer cmd,
               VkPipelineStageFlags2 stageMask) const;

private:
    GpuEvent() = default;
    void destroy();

    VkDevice device_ = VK_NULL_HANDLE;
    VkEvent  event_  = VK_NULL_HANDLE;
};

} // namespace vksdl

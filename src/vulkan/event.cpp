#include <vksdl/device.hpp>
#include <vksdl/event.hpp>

namespace vksdl {

void GpuEvent::destroy() {
    if (device_ == VK_NULL_HANDLE)
        return;
    if (event_ != VK_NULL_HANDLE)
        vkDestroyEvent(device_, event_, nullptr);
    device_ = VK_NULL_HANDLE;
    event_ = VK_NULL_HANDLE;
}

GpuEvent::~GpuEvent() {
    destroy();
}

GpuEvent::GpuEvent(GpuEvent&& o) noexcept : device_(o.device_), event_(o.event_) {
    o.device_ = VK_NULL_HANDLE;
    o.event_ = VK_NULL_HANDLE;
}

GpuEvent& GpuEvent::operator=(GpuEvent&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        event_ = o.event_;
        o.device_ = VK_NULL_HANDLE;
        o.event_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<GpuEvent> GpuEvent::create(const Device& device) {
    VkEventCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    // Device-only: no host set/reset, avoids cache flush overhead on host ops.
    ci.flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT;

    GpuEvent ev;
    ev.device_ = device.vkDevice();

    VkResult vr = vkCreateEvent(ev.device_, &ci, nullptr, &ev.event_);
    if (vr != VK_SUCCESS) {
        return Error{"create gpu event", static_cast<std::int32_t>(vr), "vkCreateEvent failed"};
    }

    return ev;
}

void GpuEvent::set(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage,
                   VkAccessFlags2 srcAccess) const {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask = VK_ACCESS_2_NONE;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;

    vkCmdSetEvent2(cmd, event_, &dep);
}

void GpuEvent::wait(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;

    vkCmdWaitEvents2(cmd, 1, &event_, &dep);
}

void GpuEvent::wait(VkCommandBuffer cmd, const VkDependencyInfo& depInfo) const {
    vkCmdWaitEvents2(cmd, 1, &event_, &depInfo);
}

void GpuEvent::reset(VkCommandBuffer cmd, VkPipelineStageFlags2 stageMask) const {
    vkCmdResetEvent2(cmd, event_, stageMask);
}

} // namespace vksdl

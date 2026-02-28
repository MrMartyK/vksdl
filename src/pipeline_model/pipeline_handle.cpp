#include <vksdl/pipeline_model/pipeline_handle.hpp>

#include "pipeline_handle_impl.hpp"

#include <vulkan/vulkan.h>

namespace vksdl {

PipelineHandle::~PipelineHandle() {
    destroy();
}

PipelineHandle::PipelineHandle(PipelineHandle&& o) noexcept : impl_(o.impl_) {
    o.impl_ = nullptr;
}

PipelineHandle& PipelineHandle::operator=(PipelineHandle&& o) noexcept {
    if (this != &o) {
        destroy();
        impl_ = o.impl_;
        o.impl_ = nullptr;
    }
    return *this;
}

void PipelineHandle::destroy() {
    if (!impl_)
        return;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);

    // Signal background thread to not store into this handle, then
    // atomically grab whatever optimized pipeline was stored. This
    // closes the race where the worker stores between our flag set
    // and our read.
    impl->destroyed.store(true, std::memory_order_release);

    VkPipeline opt = impl->optimized.exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
    if (opt != VK_NULL_HANDLE && opt != impl->baseline) {
        vkDestroyPipeline(impl->device, opt, nullptr);
    }
    if (impl->baseline != VK_NULL_HANDLE) {
        vkDestroyPipeline(impl->device, impl->baseline, nullptr);
    }
    if (impl->layout != VK_NULL_HANDLE && impl->ownsLayout) {
        vkDestroyPipelineLayout(impl->device, impl->layout, nullptr);
    }

    delete impl;
    impl_ = nullptr;
}

void PipelineHandle::bind(VkCommandBuffer cmd) const {
    if (!impl_)
        return;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);
    VkPipeline p = impl->optimized.load(std::memory_order_acquire);
    if (p == VK_NULL_HANDLE)
        p = impl->baseline;
    vkCmdBindPipeline(cmd, impl->bindPoint, p);
}

bool PipelineHandle::isOptimized() const {
    if (!impl_)
        return false;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);
    return impl->optimized.load(std::memory_order_acquire) != VK_NULL_HANDLE;
}

bool PipelineHandle::isReady() const {
    if (!impl_)
        return false;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);
    return impl->baseline != VK_NULL_HANDLE;
}

VkPipeline PipelineHandle::vkPipeline() const {
    if (!impl_)
        return VK_NULL_HANDLE;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);
    VkPipeline p = impl->optimized.load(std::memory_order_acquire);
    return (p != VK_NULL_HANDLE) ? p : impl->baseline;
}

VkPipelineLayout PipelineHandle::vkPipelineLayout() const {
    if (!impl_)
        return VK_NULL_HANDLE;
    auto* impl = static_cast<detail::PipelineHandleImpl*>(impl_);
    return impl->layout;
}

} // namespace vksdl

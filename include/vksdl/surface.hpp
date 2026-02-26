#pragma once

#include <vksdl/error.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/result.hpp>
#include <vksdl/window.hpp>

#include <vulkan/vulkan.h>

namespace vksdl {

// RAII wrapper for VkSurfaceKHR.
// Must be destroyed before the VkInstance it was created from.
//
// Thread safety: immutable after construction.
class Surface {
public:
    ~Surface();
    Surface(Surface&&) noexcept;
    Surface& operator=(Surface&&) noexcept;
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    [[nodiscard]] static Result<Surface> create(const Instance& instance,
                                                 const Window& window);

    [[nodiscard]] VkSurfaceKHR native()    const { return surface_; }
    [[nodiscard]] VkSurfaceKHR vkSurface() const { return native(); }
    [[nodiscard]] VkInstance   vkInstance() const { return instance_; }

private:
    Surface() = default;

    VkInstance   instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_  = VK_NULL_HANDLE;
};

} // namespace vksdl

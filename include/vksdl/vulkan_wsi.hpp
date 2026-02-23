#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/window.hpp>

#include <vulkan/vulkan.h>

#include <vector>

namespace vksdl::wsi {

// Returns the instance extensions required for Vulkan surface creation.
// SDL3 provides these without needing a window handle.
[[nodiscard]] std::vector<const char*> requiredInstanceExtensions();

// Creates a VkSurfaceKHR for the given window via SDL_Vulkan_CreateSurface.
// Caller gets RAII Surface from Surface::create() -- this is the raw helper.
[[nodiscard]] Result<VkSurfaceKHR> createSurface(VkInstance instance,
                                                  const Window& window);

} // namespace vksdl::wsi

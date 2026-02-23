#include <vksdl/vulkan_wsi.hpp>

#include <SDL3/SDL_vulkan.h>

#include <string>

namespace vksdl::wsi {

std::vector<const char*> requiredInstanceExtensions() {
    Uint32 count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);

    if (!names || count == 0) {
        // Fallback -- shouldn't happen with a working Vulkan driver
        return {VK_KHR_SURFACE_EXTENSION_NAME};
    }

    return std::vector<const char*>(names, names + count);
}

Result<VkSurfaceKHR> createSurface(VkInstance instance, const Window& window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    if (!SDL_Vulkan_CreateSurface(window.sdlWindow(), instance, nullptr, &surface)) {
        return Error{"create surface", 0,
                     std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError()};
    }

    return surface;
}

} // namespace vksdl::wsi

#include <vksdl/surface.hpp>
#include <vksdl/vulkan_wsi.hpp>

#include <SDL3/SDL_vulkan.h>

namespace vksdl {

Surface::~Surface() {
    if (surface_ != VK_NULL_HANDLE) {
        SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
    }
}

Surface::Surface(Surface&& o) noexcept : instance_(o.instance_), surface_(o.surface_) {
    o.instance_ = VK_NULL_HANDLE;
    o.surface_ = VK_NULL_HANDLE;
}

Surface& Surface::operator=(Surface&& o) noexcept {
    if (this != &o) {
        if (surface_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
        }
        instance_ = o.instance_;
        surface_ = o.surface_;
        o.instance_ = VK_NULL_HANDLE;
        o.surface_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<Surface> Surface::create(const Instance& instance, const Window& window) {
    auto* vkInst = instance.vkInstance();
    auto result = wsi::createSurface(vkInst, window);
    if (!result.ok()) {
        return result.error();
    }

    Surface s;
    s.instance_ = vkInst;
    s.surface_ = result.value();
    return s;
}

} // namespace vksdl

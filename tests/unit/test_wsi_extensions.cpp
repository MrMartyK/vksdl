#include <vksdl/app.hpp>
#include <vksdl/vulkan_wsi.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    // SDL must be initialized for Vulkan extension query
    auto appResult = vksdl::App::create();
    assert(appResult.ok());
    auto app = std::move(appResult.value());

    auto exts = vksdl::wsi::requiredInstanceExtensions();

    // Must contain at least VK_KHR_surface
    assert(!exts.empty());

    bool hasSurface = false;
    for (auto* name : exts) {
        if (std::strcmp(name, VK_KHR_SURFACE_EXTENSION_NAME) == 0)
            hasSurface = true;
    }

    assert(hasSurface);

    std::printf("wsi extensions test passed (count=%zu)\n", exts.size());
    return 0;
}

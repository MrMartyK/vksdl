#include <vksdl/app.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto appResult = vksdl::App::create();
    assert(appResult.ok());
    auto app = std::move(appResult.value());

    auto winResult = app.createWindow("surface test", 640, 480);
    assert(winResult.ok());
    auto window = std::move(winResult.value());

    // Create instance with window support
    auto instResult = vksdl::InstanceBuilder{}
        .appName("surface_test")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instResult.ok());
    auto instance = std::move(instResult.value());

    // Create RAII surface
    auto surfResult = vksdl::Surface::create(instance, window);
    assert(surfResult.ok() && "Surface::create failed");
    auto surface = std::move(surfResult.value());

    assert(surface.vkSurface() != VK_NULL_HANDLE);
    assert(surface.vkInstance() == instance.vkInstance());

    std::printf("surface test passed\n");

    // RAII cleanup: surface destroyed before instance
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("unified layouts test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_unified_layouts")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .build();
    assert(device.ok());

    {
        bool supported = device.value().hasUnifiedImageLayouts();
        std::printf("  unified image layouts: %s\n",
                    supported ? "supported" : "not supported");
    }

    {
        bool original = device.value().hasUnifiedImageLayouts();
        // The device was already moved-into by Result; just verify it's consistent.
        assert(device.value().hasUnifiedImageLayouts() == original);
        std::printf("  value consistent after access: ok\n");
    }

    device.value().waitIdle();
    std::printf("unified layouts test passed\n");
    return 0;
}

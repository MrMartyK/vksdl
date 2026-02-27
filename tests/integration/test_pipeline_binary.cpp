#include <vksdl/vksdl.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline binary test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_pipeline_binary")
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

    // hasPipelineBinary() is always safe to call regardless of support.
    bool hasBinary = device.value().hasPipelineBinary();
    std::printf("  VK_KHR_pipeline_binary: %s\n",
                hasBinary ? "available" : "not available");

    // Accessor must be idempotent.
    assert(device.value().hasPipelineBinary() == hasBinary);

    device.value().waitIdle();
    std::printf("pipeline binary detection test passed\n");
    return 0;
}

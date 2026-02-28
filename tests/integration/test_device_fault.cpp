#include <vksdl/vksdl.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("device fault test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_device_fault")
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
        bool supported = device.value().hasDeviceFault();
        std::printf("  hasDeviceFault: %s\n", supported ? "true" : "false");
    }

    {
        auto fault = device.value().queryDeviceFault();
        // No fault has occurred, so it should be empty (or unsupported -> also empty).
        assert(fault.empty() && "no fault should be reported on a healthy device");
        std::printf("  queryDeviceFault (no fault): ok (empty)\n");
    }

    device.value().waitIdle();
    std::printf("device fault test passed\n");
    return 0;
}

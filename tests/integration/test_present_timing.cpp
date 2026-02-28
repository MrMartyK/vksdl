#include <vksdl/vksdl.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("present timing test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_present_timing")
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

    bool hasTiming = device.value().hasPresentTiming();
    std::printf("  VK_EXT_present_timing:       %s\n",
                device.value().hasExtPresentTiming() ? "available" : "not available");
    std::printf("  VK_GOOGLE_display_timing:    %s\n",
                device.value().hasGoogleDisplayTiming() ? "available" : "not available");
    std::printf("  hasPresentTiming:            %s\n", hasTiming ? "available" : "not available");

    // Accessor must be idempotent.
    assert(device.value().hasPresentTiming() == hasTiming);

    // Individual flags are consistent with the combined flag.
    assert(hasTiming ==
           (device.value().hasExtPresentTiming() || device.value().hasGoogleDisplayTiming()));

    device.value().waitIdle();
    std::printf("present timing detection test passed\n");
    return 0;
}

#include <vksdl/app.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto appResult = vksdl::App::create();
    assert(appResult.ok());
    auto app = std::move(appResult.value());

    auto winResult = app.createWindow("device test", 640, 480);
    assert(winResult.ok());
    auto window = std::move(winResult.value());

    auto instResult = vksdl::InstanceBuilder{}
                          .appName("device_test")
                          .requireVulkan(1, 3)
                          .validation(vksdl::Validation::Off)
                          .enableWindowSupport()
                          .build();
    assert(instResult.ok());
    auto instance = std::move(instResult.value());

    auto surfResult = vksdl::Surface::create(instance, window);
    assert(surfResult.ok());
    auto surface = std::move(surfResult.value());

    // Default: discrete preferred, swapchain via named method
    {
        auto result = vksdl::DeviceBuilder(instance, surface).needSwapchain().build();

        assert(result.ok() && "device creation failed");
        auto device = std::move(result.value());

        assert(device.vkDevice() != VK_NULL_HANDLE);
        assert(device.vkPhysicalDevice() != VK_NULL_HANDLE);
        assert(device.graphicsQueue() != VK_NULL_HANDLE);
        assert(device.presentQueue() != VK_NULL_HANDLE);
        assert(device.queueFamilies().valid());

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device.vkPhysicalDevice(), &props);
        std::printf("  selected GPU: %s\n", props.deviceName);
        std::printf("  queue families: graphics=%u present=%u (%s)\n",
                    device.queueFamilies().graphics, device.queueFamilies().present,
                    device.queueFamilies().shared() ? "shared" : "separate");

        device.waitIdle();
        std::printf("  default device: ok\n");
    }

    // Named feature methods (dynamic rendering + sync2)
    {
        auto result = vksdl::DeviceBuilder(instance, surface)
                          .needSwapchain()
                          .needDynamicRendering()
                          .needSync2()
                          .build();

        assert(result.ok() && "device with named features failed");
        std::printf("  device with needDynamicRendering + needSync2: ok\n");
    }

    // Preset method for common graphics setup.
    {
        auto result =
            vksdl::DeviceBuilder(instance, surface).graphicsDefaults().preferDiscreteGpu().build();

        assert(result.ok() && "device with graphicsDefaults failed");
        std::printf("  device with graphicsDefaults: ok\n");
    }

    // Escape hatch: lambda still works
    {
        auto result = vksdl::DeviceBuilder(instance, surface)
                          .needSwapchain()
                          .requireFeatures([](VkPhysicalDeviceVulkan13Features& f) {
                              f.dynamicRendering = VK_TRUE;
                          })
                          .build();

        assert(result.ok() && "device with requireFeatures lambda failed");
        std::printf("  device with requireFeatures escape hatch: ok\n");
    }

    // Bogus extension should fail
    {
        auto result = vksdl::DeviceBuilder(instance, surface)
                          .requireExtension("VK_KHR_does_not_exist_device")
                          .build();

        assert(!result.ok());
        auto msg = result.error().format();
        assert(msg.find("No suitable GPU") != std::string::npos);
        std::printf("  bogus device extension rejected: ok\n");
    }

    std::printf("device builder test passed\n");
    return 0;
}

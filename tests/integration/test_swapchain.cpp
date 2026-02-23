#include <vksdl/app.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/swapchain.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto appResult = vksdl::App::create();
    assert(appResult.ok());
    auto app = std::move(appResult.value());

    auto winResult = app.createWindow("swapchain test", 800, 600);
    assert(winResult.ok());
    auto window = std::move(winResult.value());

    auto instResult = vksdl::InstanceBuilder{}
        .appName("swapchain_test")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instResult.ok());
    auto instance = std::move(instResult.value());

    auto surfResult = vksdl::Surface::create(instance, window);
    assert(surfResult.ok());
    auto surface = std::move(surfResult.value());

    auto devResult = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .build();
    assert(devResult.ok());
    auto device = std::move(devResult.value());

    // Create swapchain with defaults
    {
        auto scResult = vksdl::SwapchainBuilder(device, surface)
            .size(window.pixelSize())
            .build();

        assert(scResult.ok() && "swapchain creation failed");
        auto swapchain = std::move(scResult.value());

        assert(swapchain.vkSwapchain() != VK_NULL_HANDLE);
        assert(swapchain.format() != VK_FORMAT_UNDEFINED);
        assert(swapchain.extent().width > 0);
        assert(swapchain.extent().height > 0);
        assert(swapchain.imageCount() >= 2);
        assert(swapchain.images().size() == swapchain.imageViews().size());

        std::printf("  format: %d\n", swapchain.format());
        std::printf("  extent: %ux%u\n", swapchain.extent().width, swapchain.extent().height);
        std::printf("  images: %u\n", swapchain.imageCount());
        std::printf("  default swapchain: ok\n");

        // Test recreate
        device.waitIdle();
        auto recreateResult = swapchain.recreate({640, 480});
        assert(recreateResult.ok() && "recreate failed");

        assert(swapchain.imageCount() >= 2);
        assert(swapchain.images().size() == swapchain.imageViews().size());

        std::printf("  after recreate: %ux%u, %u images\n",
                    swapchain.extent().width,
                    swapchain.extent().height,
                    swapchain.imageCount());
        std::printf("  recreate: ok\n");
    }

    device.waitIdle();
    std::printf("swapchain test passed\n");
    return 0;
}

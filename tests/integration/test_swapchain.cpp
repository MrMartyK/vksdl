#include <vksdl/app.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/swapchain.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

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

    auto devResult = vksdl::DeviceBuilder(instance, surface).needSwapchain().build();
    assert(devResult.ok());
    auto device = std::move(devResult.value());

    // Create swapchain with defaults
    {
        auto scResult = vksdl::SwapchainBuilder(device, surface).forWindow(window).build();

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

        std::printf("  after recreate: %ux%u, %u images\n", swapchain.extent().width,
                    swapchain.extent().height, swapchain.imageCount());
        std::printf("  recreate: ok\n");

        // Zero-size recreate should be a no-op for minimized-window handling.
        const auto beforeZeroSwapchain = swapchain.vkSwapchain();
        const auto beforeZeroExtent = swapchain.extent();
        const auto beforeZeroCount = swapchain.imageCount();
        device.waitIdle();
        auto zeroResult = swapchain.recreate({0, 0});
        assert(zeroResult.ok() && "zero-size recreate failed");
        assert(swapchain.vkSwapchain() == beforeZeroSwapchain);
        assert(swapchain.extent().width == beforeZeroExtent.width);
        assert(swapchain.extent().height == beforeZeroExtent.height);
        assert(swapchain.imageCount() == beforeZeroCount);

        // Stress repeated recreates across varying extents.
        const std::vector<vksdl::Size> stressSizes = {
            {640, 480}, {800, 600}, {1024, 576}, {1280, 720}, {960, 540}, {640, 360},
        };
        for (int i = 0; i < 3; ++i) {
            for (const auto s : stressSizes) {
                device.waitIdle();
                auto stressResult = swapchain.recreate(s);
                assert(stressResult.ok() && "stress recreate failed");
                assert(swapchain.extent().width > 0);
                assert(swapchain.extent().height > 0);
                assert(swapchain.imageCount() >= 2);
                assert(swapchain.images().size() == swapchain.imageViews().size());
            }
        }
        std::printf("  stress recreate: ok\n");
    }

    device.waitIdle();
    std::printf("swapchain test passed\n");
    return 0;
}

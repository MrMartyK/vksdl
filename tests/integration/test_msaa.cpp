#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("msaa test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_msaa")
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

    auto allocator = vksdl::Allocator::create(instance.value(), device.value());
    assert(allocator.ok());

    auto swapchain = vksdl::SwapchainBuilder(device.value(), surface.value())
        .size(window.value().pixelSize())
        .build();
    assert(swapchain.ok());

    // 1. maxMsaaSamples returns at least 2x (all GPUs support this)
    {
        auto maxSamples = device.value().maxMsaaSamples();
        assert(maxSamples >= VK_SAMPLE_COUNT_2_BIT);
        std::printf("  maxMsaaSamples (%dx): ok\n", static_cast<int>(maxSamples));
    }

    // 2. msaaColorAttachment creates MSAA image with correct sample count
    {
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(64, 64)
            .format(swapchain.value().format())
            .msaaColorAttachment()
            .samples(VK_SAMPLE_COUNT_4_BIT)
            .build();
        assert(img.ok());
        assert(img.value().vkImage() != VK_NULL_HANDLE);
        assert(img.value().vkImageView() != VK_NULL_HANDLE);
        assert(img.value().samples() == VK_SAMPLE_COUNT_4_BIT);
        std::printf("  msaaColorAttachment: ok\n");
    }

    // 3. samples() accessor preserved after move
    {
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(64, 64)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .msaaColorAttachment()
            .samples(VK_SAMPLE_COUNT_4_BIT)
            .build();
        assert(img.ok());

        VkImage handle = img.value().vkImage();
        vksdl::Image moved = std::move(img.value());
        assert(moved.vkImage() == handle);
        assert(moved.samples() == VK_SAMPLE_COUNT_4_BIT);
        std::printf("  samples after move: ok\n");
    }

    // 4. MSAA pipeline creation
    {
        std::filesystem::path shaderDir =
            std::filesystem::path(SDL_GetBasePath()) / "shaders";

        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .samples(VK_SAMPLE_COUNT_4_BIT)
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  msaa pipeline: ok\n");
    }

    std::printf("all msaa tests passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    {
        assert(vksdl::calculateMipLevels(1024, 512) == 11);
        assert(vksdl::calculateMipLevels(1, 1) == 1);
        assert(vksdl::calculateMipLevels(256, 256) == 9);
        assert(vksdl::calculateMipLevels(0, 0) == 1);
        assert(vksdl::calculateMipLevels(2, 2) == 2);
        assert(vksdl::calculateMipLevels(1, 512) == 10);
        std::printf("  calculateMipLevels: ok\n");
    }

    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("mipmap test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_mipmaps")
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

    {
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(256, 256)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .sampled()
            .mipmapped()
            .build();
        assert(img.ok());
        assert(img.value().mipLevels() == 9);
        std::printf("  mipmapped image mipLevels: ok\n");
    }

    {
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(256, 256)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .sampled()
            .build();
        assert(img.ok());
        assert(img.value().mipLevels() == 1);
        std::printf("  non-mipmapped image mipLevels: ok\n");
    }

    {
        std::filesystem::path assetDir =
            std::filesystem::path(SDL_GetBasePath()) / "assets";
        auto imgData = vksdl::loadImage(assetDir / "test_2x2.png");
        assert(imgData.ok());

        auto gpuImage = vksdl::ImageBuilder(allocator.value())
            .size(imgData.value().width, imgData.value().height)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .sampled()
            .mipmapped()
            .build();
        assert(gpuImage.ok());
        assert(gpuImage.value().mipLevels() == 2);

        auto uploadResult = vksdl::uploadToImage(
            allocator.value(), device.value(), gpuImage.value(),
            imgData.value().pixels, imgData.value().sizeBytes());
        assert(uploadResult.ok());

        // Record generateMipmaps in one-shot command buffer
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = device.value().queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkResult vr = vkCreateCommandPool(device.value().vkDevice(), &poolCI,
                                           nullptr, &cmdPool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool        = cmdPool;
        cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device.value().vkDevice(), &cmdAI, &cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::generateMipmaps(cmd, gpuImage.value());

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &cmd;

        vkQueueSubmit(device.value().graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(device.value().graphicsQueue());

        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        std::printf("  upload + generateMipmaps: ok\n");
    }

    {
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(64, 64)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .sampled()
            .mipmapped()
            .build();
        assert(img.ok());
        std::uint32_t levels = img.value().mipLevels();

        vksdl::Image moved = std::move(img).value();
        assert(moved.mipLevels() == levels);
        std::printf("  mipLevels after move: ok\n");
    }

    device.value().waitIdle();
    std::printf("all mipmap tests passed\n");
    return 0;
}

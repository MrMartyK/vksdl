#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("texture upload test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_texture_upload")
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

    // 1. loadImage -- load test_2x2.png
    {
        std::filesystem::path assetDir =
            std::filesystem::path(SDL_GetBasePath()) / "assets";
        auto img = vksdl::loadImage(assetDir / "test_2x2.png");
        assert(img.ok());
        assert(img.value().width == 2);
        assert(img.value().height == 2);
        assert(img.value().channels == 4);
        assert(img.value().pixels != nullptr);
        assert(img.value().sizeBytes() == 2 * 2 * 4);
        std::printf("  loadImage: ok\n");
    }

    // 2. loadImage -- missing file returns error
    {
        auto img = vksdl::loadImage("nonexistent_file.png");
        assert(!img.ok());
        std::printf("  loadImage missing file: ok\n");
    }

    // 3. uploadToImage -- load + create GPU image + upload
    {
        std::filesystem::path assetDir =
            std::filesystem::path(SDL_GetBasePath()) / "assets";
        auto imgData = vksdl::loadImage(assetDir / "test_2x2.png");
        assert(imgData.ok());

        auto gpuImage = vksdl::ImageBuilder(allocator.value())
            .size(imgData.value().width, imgData.value().height)
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .sampled()
            .build();
        assert(gpuImage.ok());

        auto uploadResult = vksdl::uploadToImage(
            allocator.value(),
            device.value(),
            gpuImage.value(),
            imgData.value().pixels,
            imgData.value().sizeBytes());
        assert(uploadResult.ok());
        std::printf("  uploadToImage: ok\n");
    }

    // 4. ImageData move semantics
    {
        std::filesystem::path assetDir =
            std::filesystem::path(SDL_GetBasePath()) / "assets";
        auto imgData = vksdl::loadImage(assetDir / "test_2x2.png");
        assert(imgData.ok());

        unsigned char* origPixels = imgData.value().pixels;
        vksdl::ImageData moved = std::move(imgData.value());
        assert(moved.pixels == origPixels);
        assert(moved.width == 2);
        // moved-from destructor is safe (null pixels)
        std::printf("  ImageData move: ok\n");
    }

    device.value().waitIdle();
    std::printf("all texture upload tests passed\n");
    return 0;
}

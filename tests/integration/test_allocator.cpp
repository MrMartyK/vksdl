#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("allocator test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_allocator")
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
    assert(allocator.value().vmaAllocator() != nullptr);
    assert(allocator.value().vkDevice() == device.value().vkDevice());
    std::printf("  allocator: ok\n");

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(1024).vertexBuffer().build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        assert(buf.value().size() == 1024);
        assert(buf.value().mappedData() == nullptr);
        std::printf("  vertex buffer: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(256).stagingBuffer().build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        assert(buf.value().mappedData() != nullptr);
        std::printf("  staging buffer: ok\n");
    }

    {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto buf =
            vksdl::BufferBuilder(allocator.value()).size(sizeof(data)).stagingBuffer().build();
        assert(buf.ok());
        std::memcpy(buf.value().mappedData(), data, sizeof(data));
        assert(std::memcmp(buf.value().mappedData(), data, sizeof(data)) == 0);
        std::printf("  write to mapped buffer: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(64).uniformBuffer().build();
        assert(buf.ok());
        assert(buf.value().mappedData() != nullptr);
        std::printf("  uniform buffer: ok\n");
    }

    {
        float vertices[] = {0.0f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f};

        auto vbuf =
            vksdl::BufferBuilder(allocator.value()).size(sizeof(vertices)).vertexBuffer().build();
        assert(vbuf.ok());

        auto uploadResult = vksdl::uploadToBuffer(allocator.value(), device.value(), vbuf.value(),
                                                  vertices, sizeof(vertices));
        assert(uploadResult.ok());
        std::printf("  staged upload: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).vertexBuffer().build();
        assert(!buf.ok());
        std::printf("  zero-size rejected: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(64).build();
        assert(!buf.ok());
        std::printf("  no-usage rejected: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(128).stagingBuffer().build();
        assert(buf.ok());
        auto b1 = std::move(buf).value();
        auto b2 = std::move(b1);
        assert(b2.vkBuffer() != VK_NULL_HANDLE);
        std::printf("  buffer move: ok\n");
    }

    {
        auto img = vksdl::ImageBuilder(allocator.value())
                       .size(800, 600)
                       .format(VK_FORMAT_R8G8B8A8_UNORM)
                       .colorAttachment()
                       .build();
        assert(img.ok());
        assert(img.value().vkImage() != VK_NULL_HANDLE);
        assert(img.value().vkImageView() != VK_NULL_HANDLE);
        assert(img.value().format() == VK_FORMAT_R8G8B8A8_UNORM);
        assert(img.value().extent().width == 800);
        assert(img.value().extent().height == 600);
        std::printf("  color attachment image: ok\n");
    }

    {
        auto img = vksdl::ImageBuilder(allocator.value()).size(800, 600).depthAttachment().build();
        assert(img.ok());
        assert(img.value().vkImage() != VK_NULL_HANDLE);
        assert(img.value().vkImageView() != VK_NULL_HANDLE);
        assert(img.value().format() == VK_FORMAT_D32_SFLOAT);
        std::printf("  depth attachment image: ok\n");
    }

    {
        auto img = vksdl::ImageBuilder(allocator.value())
                       .size(64, 64)
                       .format(VK_FORMAT_R8G8B8A8_UNORM)
                       .colorAttachment()
                       .build();
        assert(img.ok());
        auto i1 = std::move(img).value();
        auto i2 = std::move(i1);
        assert(i2.vkImage() != VK_NULL_HANDLE);
        std::printf("  image move: ok\n");
    }

    device.value().waitIdle();
    std::printf("allocator test passed\n");
    return 0;
}

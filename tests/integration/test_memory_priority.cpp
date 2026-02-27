#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("memory priority test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_memory_priority")
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

    std::printf("  VK_EXT_memory_priority: %s\n",
                device.value().hasMemoryPriority() ? "supported" : "not supported");

    {
        // High-priority buffer -- resident under memory pressure.
        auto buf = vksdl::BufferBuilder(allocator.value())
            .storageBuffer()
            .size(4096)
            .memoryPriority(1.0f)
            .build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        assert(buf.value().size() == 4096);
        std::printf("  high-priority buffer (1.0): ok\n");
    }

    {
        // Low-priority buffer -- evicted first under pressure.
        auto buf = vksdl::BufferBuilder(allocator.value())
            .storageBuffer()
            .size(4096)
            .memoryPriority(0.0f)
            .build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        std::printf("  low-priority buffer (0.0): ok\n");
    }

    {
        // Default priority buffer (no explicit call).
        auto buf = vksdl::BufferBuilder(allocator.value())
            .vertexBuffer()
            .size(1024)
            .build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        std::printf("  default-priority buffer: ok\n");
    }

    {
        // High-priority image.
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(256, 256)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .colorAttachment()
            .memoryPriority(1.0f)
            .build();
        assert(img.ok());
        assert(img.value().vkImage() != VK_NULL_HANDLE);
        std::printf("  high-priority image (1.0): ok\n");
    }

    {
        // Low-priority image.
        auto img = vksdl::ImageBuilder(allocator.value())
            .size(64, 64)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .sampled()
            .memoryPriority(0.25f)
            .build();
        assert(img.ok());
        assert(img.value().vkImage() != VK_NULL_HANDLE);
        std::printf("  low-priority image (0.25): ok\n");
    }

    device.value().waitIdle();
    std::printf("memory priority test passed\n");
    return 0;
}

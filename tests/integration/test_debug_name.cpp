#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("debug name test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_debug_name")
        .requireVulkan(1, 3)
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
        auto buf = vksdl::BufferBuilder(allocator.value())
            .size(64)
            .uniformBuffer()
            .build();
        assert(buf.ok());

        vksdl::debugName(device.value().vkDevice(), buf.value().vkBuffer(),
                         "test uniform buffer");
        std::printf("  debugName buffer: ok\n");
    }

    {
        auto sampler = vksdl::SamplerBuilder(device.value())
            .nearest()
            .repeat()
            .build();
        assert(sampler.ok());

        vksdl::debugName(device.value().vkDevice(), sampler.value().vkSampler(),
                         "test sampler");
        std::printf("  debugName sampler: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value())
            .size(128)
            .uniformBuffer()
            .build();
        assert(buf.ok());

        vksdl::debugName(device.value().vkDevice(), VK_OBJECT_TYPE_BUFFER,
                         reinterpret_cast<std::uint64_t>(buf.value().vkBuffer()),
                         "test buffer via core overload");
        std::printf("  debugName core overload: ok\n");
    }

    device.value().waitIdle();
    std::printf("all debug name tests passed\n");
    return 0;
}

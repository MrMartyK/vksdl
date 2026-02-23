#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("sampler test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_sampler")
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

    // 1. Default sampler (linear + repeat defaults)
    {
        auto sampler = vksdl::SamplerBuilder(device.value()).build();
        assert(sampler.ok());
        assert(sampler.value().vkSampler() != VK_NULL_HANDLE);
        std::printf("  default sampler: ok\n");
    }

    // 2. Linear + repeat (explicit)
    {
        auto sampler = vksdl::SamplerBuilder(device.value())
            .linear()
            .repeat()
            .build();
        assert(sampler.ok());
        assert(sampler.value().vkSampler() != VK_NULL_HANDLE);
        std::printf("  linear + repeat: ok\n");
    }

    // 3. Nearest + clamp to edge
    {
        auto sampler = vksdl::SamplerBuilder(device.value())
            .nearest()
            .clampToEdge()
            .build();
        assert(sampler.ok());
        assert(sampler.value().vkSampler() != VK_NULL_HANDLE);
        std::printf("  nearest + clamp: ok\n");
    }

    // 4. Anisotropy
    {
        auto sampler = vksdl::SamplerBuilder(device.value())
            .anisotropy(16.0f)
            .build();
        assert(sampler.ok());
        assert(sampler.value().vkSampler() != VK_NULL_HANDLE);
        std::printf("  anisotropy: ok\n");
    }

    // 5. Escape hatches
    {
        auto sampler = vksdl::SamplerBuilder(device.value())
            .magFilter(VK_FILTER_NEAREST)
            .minFilter(VK_FILTER_LINEAR)
            .addressModeU(VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT)
            .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
            .build();
        assert(sampler.ok());
        assert(sampler.value().vkSampler() != VK_NULL_HANDLE);
        std::printf("  escape hatches: ok\n");
    }

    // 6. Move semantics
    {
        auto sampler = vksdl::SamplerBuilder(device.value()).build();
        assert(sampler.ok());

        VkSampler handle = sampler.value().vkSampler();
        vksdl::Sampler moved = std::move(sampler.value());
        assert(moved.vkSampler() == handle);
        // moved-from destructor is safe (null handle)
        std::printf("  move semantics: ok\n");
    }

    std::printf("all sampler tests passed\n");
    return 0;
}

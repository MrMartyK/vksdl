#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("sampler cache test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_sampler_cache")
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

    std::printf("sampler cache test\n");

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());
        assert(cache.value().size() == 0);
        std::printf("  1. create empty cache: ok\n");
    }

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());

        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.maxLod = VK_LOD_CLAMP_NONE;

        auto s = cache.value().get(ci);
        assert(s.ok());
        assert(s.value() != VK_NULL_HANDLE);
        assert(cache.value().size() == 1);
        std::printf("  2. get creates sampler: ok\n");
    }

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());

        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_NEAREST;
        ci.minFilter = VK_FILTER_NEAREST;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = VK_LOD_CLAMP_NONE;

        auto s1 = cache.value().get(ci);
        auto s2 = cache.value().get(ci);
        assert(s1.ok() && s2.ok());
        assert(s1.value() == s2.value());
        assert(cache.value().size() == 1);
        std::printf("  3. deduplication (same handle): ok\n");
    }

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());

        VkSamplerCreateInfo linear{};
        linear.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        linear.magFilter = VK_FILTER_LINEAR;
        linear.minFilter = VK_FILTER_LINEAR;
        linear.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        linear.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        linear.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        linear.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        linear.maxLod = VK_LOD_CLAMP_NONE;

        VkSamplerCreateInfo nearest{};
        nearest.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        nearest.magFilter = VK_FILTER_NEAREST;
        nearest.minFilter = VK_FILTER_NEAREST;
        nearest.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        nearest.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearest.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearest.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        nearest.maxLod = VK_LOD_CLAMP_NONE;

        auto s1 = cache.value().get(linear);
        auto s2 = cache.value().get(nearest);
        assert(s1.ok() && s2.ok());
        assert(s1.value() != s2.value());
        assert(cache.value().size() == 2);
        std::printf("  4. different configs -> different samplers: ok\n");
    }

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());

        VkSamplerCreateInfo base{};
        base.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        base.magFilter = VK_FILTER_LINEAR;
        base.minFilter = VK_FILTER_LINEAR;
        base.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        base.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        base.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        base.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        base.maxLod = VK_LOD_CLAMP_NONE;

        VkSamplerCreateInfo aniso = base;
        aniso.anisotropyEnable = VK_TRUE;
        aniso.maxAnisotropy = 16.0f;

        auto s1 = cache.value().get(base);
        auto s2 = cache.value().get(aniso);
        assert(s1.ok() && s2.ok());
        assert(s1.value() != s2.value());
        assert(cache.value().size() == 2);

        // Same aniso config returns same handle.
        auto s3 = cache.value().get(aniso);
        assert(s3.ok());
        assert(s3.value() == s2.value());
        assert(cache.value().size() == 2);
        std::printf("  5. anisotropy affects identity: ok\n");
    }

    {
        auto cache = vksdl::SamplerCache::create(device.value());
        assert(cache.ok());

        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.maxLod = VK_LOD_CLAMP_NONE;

        auto s1 = cache.value().get(ci);
        assert(s1.ok());

        // Move-construct.
        vksdl::SamplerCache moved = std::move(cache.value());
        assert(moved.size() == 1);

        // Moved-to still returns same handle.
        auto s2 = moved.get(ci);
        assert(s2.ok());
        assert(s2.value() == s1.value());
        assert(moved.size() == 1);

        // Move-assign.
        auto cache2 = vksdl::SamplerCache::create(device.value());
        assert(cache2.ok());
        cache2.value() = std::move(moved);
        assert(cache2.value().size() == 1);

        std::printf("  6. move semantics: ok\n");
    }

    device.value().waitIdle();
    std::printf("sampler cache test passed\n");
    return 0;
}

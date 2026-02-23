#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    {
        assert(vksdl::alignUp(64, 256) == 256);
        assert(vksdl::alignUp(256, 256) == 256);
        assert(vksdl::alignUp(257, 256) == 512);
        assert(vksdl::alignUp(0, 256) == 0);
        assert(vksdl::alignUp(1, 1) == 1);
        assert(vksdl::alignUp(100, 64) == 128);
        std::printf("  alignUp: ok\n");
    }

    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("dynamic ubo test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_dynamic_ubo")
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

    {
        VkDeviceSize alignment = device.value().minUniformBufferOffsetAlignment();
        assert(alignment > 0);
        std::printf("  minUniformBufferOffsetAlignment = %llu: ok\n",
                    static_cast<unsigned long long>(alignment));
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addDynamicUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorSetLayout() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorPool() != VK_NULL_HANDLE);
        std::printf("  addDynamicUniformBuffer: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .addDynamicUniformBuffer(1, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        std::printf("  mixed uniform + dynamic: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addDynamicUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());
        auto d1 = std::move(ds).value();
        auto d2 = std::move(d1);
        assert(d2.vkDescriptorSet() != VK_NULL_HANDLE);
        assert(d1.vkDescriptorSet() == VK_NULL_HANDLE);
        std::printf("  dynamic UBO move: ok\n");
    }

    device.value().waitIdle();
    std::printf("all dynamic UBO tests passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("descriptor pool test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_descriptor_pool")
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

    // Create a simple layout for testing.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 1;
    layoutCI.pBindings    = &binding;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorSetLayout(device.value().vkDevice(),
                                               &layoutCI, nullptr, &layout);
    assert(vr == VK_SUCCESS);

    {
        auto pool = vksdl::DescriptorPool::create(device.value());
        assert(pool.ok() && "DescriptorPool creation failed");
        assert(pool.value().poolCount() == 1);
        assert(pool.value().allocatedSetCount() == 0);
        std::printf("  create pool: ok\n");
    }

    {
        auto pool = vksdl::DescriptorPool::create(device.value());
        assert(pool.ok());

        auto set = pool.value().allocate(layout);
        assert(set.ok() && "first allocation failed");
        assert(set.value() != VK_NULL_HANDLE);
        assert(pool.value().allocatedSetCount() == 1);
        std::printf("  allocate single set: ok\n");
    }

    {
        auto pool = vksdl::DescriptorPool::create(device.value(), 4);
        assert(pool.ok());

        // Allocate well beyond the initial pool's maxSets.
        for (int i = 0; i < 100; ++i) {
            auto set = pool.value().allocate(layout);
            assert(set.ok() && "batch allocation failed");
            assert(set.value() != VK_NULL_HANDLE);
        }

        assert(pool.value().allocatedSetCount() == 100);
        assert(pool.value().poolCount() > 1);
        std::printf("  allocate 100 sets (pool count: %u): ok\n",
                    pool.value().poolCount());
    }

    {
        auto pool = vksdl::DescriptorPool::create(device.value(), 16);
        assert(pool.ok());

        for (int i = 0; i < 32; ++i) {
            auto set = pool.value().allocate(layout);
            assert(set.ok());
        }

        assert(pool.value().allocatedSetCount() == 32);

        pool.value().reset();
        assert(pool.value().allocatedSetCount() == 0);

        // Allocate again after reset.
        for (int i = 0; i < 32; ++i) {
            auto set = pool.value().allocate(layout);
            assert(set.ok());
        }

        assert(pool.value().allocatedSetCount() == 32);
        std::printf("  reset and reallocate: ok\n");
    }

    {
        auto pool = vksdl::DescriptorPool::create(device.value());
        assert(pool.ok());

        auto set1 = pool.value().allocate(layout);
        assert(set1.ok());

        vksdl::DescriptorPool moved = std::move(pool.value());
        assert(moved.allocatedSetCount() == 1);
        assert(moved.poolCount() == 1);

        auto set2 = moved.allocate(layout);
        assert(set2.ok());
        assert(moved.allocatedSetCount() == 2);
        std::printf("  move semantics: ok\n");
    }

    vkDestroyDescriptorSetLayout(device.value().vkDevice(), layout, nullptr);

    device.value().waitIdle();
    std::printf("descriptor pool test passed\n");
    return 0;
}

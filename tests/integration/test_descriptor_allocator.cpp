#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("descriptor allocator test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_descriptor_allocator")
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

    std::printf("descriptor allocator test\n");

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value());
        assert(alloc.ok() && "DescriptorAllocator creation failed");
        assert(alloc.value().allocatedSetCount() == 0);
        assert(alloc.value().poolCount() == 1);
        std::printf("  1. create allocator: ok\n");
    }

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value());
        assert(alloc.ok());

        auto set = alloc.value().allocate(layout);
        assert(set.ok() && "first allocation failed");
        assert(set.value() != VK_NULL_HANDLE);
        assert(alloc.value().allocatedSetCount() == 1);
        std::printf("  2. allocate single set: ok\n");
    }

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value(), 256);
        assert(alloc.ok());

        for (int i = 0; i < 300; ++i) {
            auto set = alloc.value().allocate(layout);
            assert(set.ok() && "batch allocation failed");
            assert(set.value() != VK_NULL_HANDLE);
        }

        assert(alloc.value().allocatedSetCount() == 300);
        assert(alloc.value().poolCount() > 1);
        std::printf("  3. allocate 300 sets (pool count: %u): ok\n",
                    alloc.value().poolCount());
    }

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value(), 128);
        assert(alloc.ok());

        for (int i = 0; i < 200; ++i) {
            auto set = alloc.value().allocate(layout);
            assert(set.ok());
        }

        std::uint32_t poolsBefore = alloc.value().poolCount();
        assert(alloc.value().allocatedSetCount() == 200);

        alloc.value().resetPools();
        assert(alloc.value().allocatedSetCount() == 0);
        // All pools moved to free list -- poolCount stays the same.
        assert(alloc.value().poolCount() == poolsBefore);

        // Re-allocate same number -- should reuse pools, not create new ones.
        for (int i = 0; i < 200; ++i) {
            auto set = alloc.value().allocate(layout);
            assert(set.ok());
        }

        assert(alloc.value().allocatedSetCount() == 200);
        assert(alloc.value().poolCount() == poolsBefore);
        std::printf("  4. reset and reallocate (pool reuse): ok\n");
    }

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value());
        assert(alloc.ok());

        auto set1 = alloc.value().allocate(layout);
        assert(set1.ok());

        // Move-construct.
        vksdl::DescriptorAllocator moved = std::move(alloc.value());
        assert(moved.allocatedSetCount() == 1);
        assert(moved.poolCount() == 1);

        auto set2 = moved.allocate(layout);
        assert(set2.ok());
        assert(moved.allocatedSetCount() == 2);

        // Move-assign.
        auto alloc2 = vksdl::DescriptorAllocator::create(device.value());
        assert(alloc2.ok());
        alloc2.value() = std::move(moved);
        assert(alloc2.value().allocatedSetCount() == 2);

        std::printf("  5. move semantics: ok\n");
    }

    {
        auto alloc = vksdl::DescriptorAllocator::create(device.value());
        assert(alloc.ok());

        auto set = alloc.value().allocate(layout);
        assert(set.ok());

        // Create a dummy buffer to write.
        VkBufferCreateInfo bufCI{};
        bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufCI.size  = 64;
        bufCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VkBuffer buf = VK_NULL_HANDLE;
        vr = vkCreateBuffer(device.value().vkDevice(), &bufCI, nullptr, &buf);
        assert(vr == VK_SUCCESS);

        // Allocate memory for the buffer.
        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(device.value().vkDevice(), buf, &memReq);

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(device.value().vkPhysicalDevice(), &memProps);

        std::uint32_t memTypeIdx = UINT32_MAX;
        for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memTypeIdx = i;
                break;
            }
        }
        assert(memTypeIdx != UINT32_MAX);

        VkMemoryAllocateInfo memAI{};
        memAI.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAI.allocationSize  = memReq.size;
        memAI.memoryTypeIndex = memTypeIdx;

        VkDeviceMemory mem = VK_NULL_HANDLE;
        vr = vkAllocateMemory(device.value().vkDevice(), &memAI, nullptr, &mem);
        assert(vr == VK_SUCCESS);
        vr = vkBindBufferMemory(device.value().vkDevice(), buf, mem, 0);
        assert(vr == VK_SUCCESS);

        // Write a uniform buffer binding using DescriptorWriter.
        vksdl::DescriptorWriter(set.value())
            .buffer(0, buf, 64)
            .write(device.value());

        std::printf("  6. DescriptorWriter: ok\n");

        vkDestroyBuffer(device.value().vkDevice(), buf, nullptr);
        vkFreeMemory(device.value().vkDevice(), mem, nullptr);
    }

    vkDestroyDescriptorSetLayout(device.value().vkDevice(), layout, nullptr);

    device.value().waitIdle();
    std::printf("descriptor allocator test passed\n");
    return 0;
}

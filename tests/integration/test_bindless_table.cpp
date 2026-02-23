#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("bindless_table test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_bindless_table")
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

    std::printf("  bindless available: %s\n",
                device.value().hasBindless() ? "yes" : "no");

    if (!device.value().hasBindless()) {
        std::printf("all bindless_table tests skipped (no bindless support)\n");
        return 0;
    }

    {
        auto table = vksdl::BindlessTable::create(device.value(), 1024);
        assert(table.ok());
        assert(table.value().vkDescriptorSet() != VK_NULL_HANDLE);
        assert(table.value().vkDescriptorSetLayout() != VK_NULL_HANDLE);
        assert(table.value().capacity() == 1024);
        assert(table.value().descriptorType() == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        std::printf("  create combined image sampler table (1024): ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(
            device.value(), 256,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_COMPUTE_BIT);
        assert(table.ok());
        assert(table.value().capacity() == 256);
        assert(table.value().descriptorType() == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::printf("  create storage buffer table (256): ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(device.value(), 64);
        assert(table.ok());

        auto img = vksdl::ImageBuilder(allocator.value())
            .size(16, 16)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .sampled()
            .build();
        assert(img.ok());

        auto sampler = vksdl::SamplerBuilder(device.value())
            .linear().clampToEdge().build();
        assert(sampler.ok());

        // Write to slot 0 and slot 42 (partial binding -- skip everything between).
        table.value().writeImage(0, img.value().vkImageView(),
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 sampler.value().vkSampler());
        table.value().writeImage(42, img.value().vkImageView(),
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 sampler.value().vkSampler());
        std::printf("  write image to sparse slots: ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(
            device.value(), 32, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        assert(table.ok());

        auto img = vksdl::ImageBuilder(allocator.value())
            .size(16, 16)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .storage()
            .build();
        assert(img.ok());

        table.value().writeStorageImage(0, img.value().vkImageView(),
                                        VK_IMAGE_LAYOUT_GENERAL);
        std::printf("  write storage image: ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(
            device.value(), 128, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        assert(table.ok());

        auto buf = vksdl::BufferBuilder(allocator.value())
            .size(256)
            .storageBuffer()
            .build();
        assert(buf.ok());

        table.value().writeBuffer(0, buf.value().vkBuffer(), 256);
        table.value().writeBuffer(100, buf.value().vkBuffer(), 128, 64);
        std::printf("  write buffer to sparse slots: ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(device.value(), 64);
        assert(table.ok());

        // Create a pipeline layout that matches the bindless layout.
        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 1;
        VkDescriptorSetLayout dsl = table.value().vkDescriptorSetLayout();
        plCI.pSetLayouts    = &dsl;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkResult vr = vkCreatePipelineLayout(device.value().vkDevice(),
                                              &plCI, nullptr, &pipelineLayout);
        assert(vr == VK_SUCCESS);

        VkCommandPoolCreateInfo cpCI{};
        cpCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpCI.queueFamilyIndex = device.value().queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        vr = vkCreateCommandPool(device.value().vkDevice(), &cpCI, nullptr, &cmdPool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo cbAI{};
        cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAI.commandPool        = cmdPool;
        cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device.value().vkDevice(), &cbAI, &cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        table.value().bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0);

        vkEndCommandBuffer(cmd);

        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        vkDestroyPipelineLayout(device.value().vkDevice(), pipelineLayout, nullptr);
        std::printf("  bind to command buffer: ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(device.value(), 32);
        assert(table.ok());

        auto t1 = std::move(table).value();
        assert(t1.vkDescriptorSet() != VK_NULL_HANDLE);

        auto t2 = std::move(t1);
        assert(t2.vkDescriptorSet() != VK_NULL_HANDLE);
        assert(t1.vkDescriptorSet() == VK_NULL_HANDLE);

        vksdl::BindlessTable t3 = vksdl::BindlessTable::create(device.value(), 16).value();
        t3 = std::move(t2);
        assert(t3.vkDescriptorSet() != VK_NULL_HANDLE);
        assert(t3.capacity() == 32);
        assert(t2.vkDescriptorSet() == VK_NULL_HANDLE);
        std::printf("  move semantics: ok\n");
    }

    {
        auto table = vksdl::BindlessTable::create(device.value(), 0);
        assert(!table.ok());
        std::printf("  zero capacity rejected: ok\n");
    }

    device.value().waitIdle();
    std::printf("all bindless_table tests passed\n");
    return 0;
}

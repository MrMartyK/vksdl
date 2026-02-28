#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("push_descriptor_writer test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_push_descriptor_writer")
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

    std::printf("  push descriptors available: %s\n",
                device.value().hasPushDescriptors() ? "yes" : "no");

    if (!device.value().hasPushDescriptors()) {
        std::printf("all push_descriptor_writer tests skipped (no VK_KHR_push_descriptor)\n");
        return 0;
    }

    VkDevice dev = device.value().vkDevice();

    // Create a descriptor set layout with PUSH_DESCRIPTOR flag.
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslCI.bindingCount = 2;
    dslCI.pBindings = bindings;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
    assert(vr == VK_SUCCESS);

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &dsl;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vr = vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
    assert(vr == VK_SUCCESS);

    // Create a command pool + buffer for recording push descriptor commands.
    VkCommandPoolCreateInfo cpCI{};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpCI.queueFamilyIndex = device.value().queueFamilies().graphics;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(dev, &cpCI, nullptr, &cmdPool);
    assert(vr == VK_SUCCESS);

    VkCommandBufferAllocateInfo cbAI{};
    cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAI.commandPool = cmdPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(dev, &cbAI, &cmd);
    assert(vr == VK_SUCCESS);

    // Create test resources.
    auto ubo = vksdl::BufferBuilder(allocator.value()).size(64).uniformBuffer().build();
    assert(ubo.ok());

    auto img = vksdl::ImageBuilder(allocator.value())
                   .size(16, 16)
                   .format(VK_FORMAT_R8G8B8A8_UNORM)
                   .sampled()
                   .build();
    assert(img.ok());

    auto sampler = vksdl::SamplerBuilder(device.value()).linear().clampToEdge().build();
    assert(sampler.ok());

    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::PushDescriptorWriter(pipelineLayout, 0)
            .buffer(0, ubo.value().vkBuffer(), 64)
            .push(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

        vkEndCommandBuffer(cmd);
        vkResetCommandBuffer(cmd, 0);
        std::printf("  push buffer descriptor: ok\n");
    }

    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::PushDescriptorWriter(pipelineLayout, 0)
            .image(1, img.value().vkImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sampler.value().vkSampler())
            .push(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

        vkEndCommandBuffer(cmd);
        vkResetCommandBuffer(cmd, 0);
        std::printf("  push image descriptor: ok\n");
    }

    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::PushDescriptorWriter(pipelineLayout, 0)
            .buffer(0, ubo.value().vkBuffer(), 64)
            .image(1, img.value().vkImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sampler.value().vkSampler())
            .push(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

        vkEndCommandBuffer(cmd);
        vkResetCommandBuffer(cmd, 0);
        std::printf("  push multiple descriptors: ok\n");
    }

    {
        auto storageImg = vksdl::ImageBuilder(allocator.value())
                              .size(16, 16)
                              .format(VK_FORMAT_R8G8B8A8_UNORM)
                              .storage()
                              .build();
        assert(storageImg.ok());

        // Need a layout with storage image binding for this test.
        VkDescriptorSetLayoutBinding storageBind{};
        storageBind.binding = 0;
        storageBind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        storageBind.descriptorCount = 1;
        storageBind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo storageDslCI{};
        storageDslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        storageDslCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        storageDslCI.bindingCount = 1;
        storageDslCI.pBindings = &storageBind;

        VkDescriptorSetLayout storageDsl = VK_NULL_HANDLE;
        vr = vkCreateDescriptorSetLayout(dev, &storageDslCI, nullptr, &storageDsl);
        assert(vr == VK_SUCCESS);

        VkPipelineLayoutCreateInfo storagePlCI{};
        storagePlCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        storagePlCI.setLayoutCount = 1;
        storagePlCI.pSetLayouts = &storageDsl;

        VkPipelineLayout storagePL = VK_NULL_HANDLE;
        vr = vkCreatePipelineLayout(dev, &storagePlCI, nullptr, &storagePL);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::PushDescriptorWriter(storagePL, 0)
            .storageImage(0, storageImg.value().vkImageView(), VK_IMAGE_LAYOUT_GENERAL)
            .push(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);

        vkEndCommandBuffer(cmd);
        vkResetCommandBuffer(cmd, 0);

        vkDestroyPipelineLayout(dev, storagePL, nullptr);
        vkDestroyDescriptorSetLayout(dev, storageDsl, nullptr);
        std::printf("  push storage image descriptor: ok\n");
    }

    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::PushDescriptorWriter(pipelineLayout, 0).push(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

        vkEndCommandBuffer(cmd);
        vkResetCommandBuffer(cmd, 0);
        std::printf("  empty push no-op: ok\n");
    }

    // Cleanup
    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);

    device.value().waitIdle();
    std::printf("all push_descriptor_writer tests passed\n");
    return 0;
}

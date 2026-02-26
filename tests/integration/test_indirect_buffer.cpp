#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("indirect buffer test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_indirect_buffer")
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

    // Create indirect buffer via builder
    {
        auto buf = vksdl::BufferBuilder(allocator.value())
            .indirectBuffer()
            .size(sizeof(VkDrawIndirectCommand))
            .build();
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        assert(buf.value().size() == sizeof(VkDrawIndirectCommand));
        assert(buf.value().deviceAddress() != 0);
        std::printf("  indirect buffer create: ok\n");
    }

    // Upload VkDrawIndirectCommand via uploadIndirectBuffer
    {
        VkDrawIndirectCommand cmd{};
        cmd.vertexCount   = 3;
        cmd.instanceCount = 1;
        cmd.firstVertex   = 0;
        cmd.firstInstance = 0;

        auto buf = vksdl::uploadIndirectBuffer(
            allocator.value(), device.value(),
            &cmd, sizeof(cmd));
        assert(buf.ok());
        assert(buf.value().vkBuffer() != VK_NULL_HANDLE);
        assert(buf.value().deviceAddress() != 0);
        std::printf("  uploadIndirectBuffer: ok\n");
    }

    // Record barriers (validate no errors)
    {
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = device.value().queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkResult vr = vkCreateCommandPool(device.value().vkDevice(), &poolCI,
                                          nullptr, &cmdPool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool        = cmdPool;
        cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device.value().vkDevice(), &cmdAI, &cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::barrierComputeToIndirectRead(cmd);
        vksdl::barrierComputeToVertexRead(cmd);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &cmd;

        vr = vkQueueSubmit(device.value().graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);
        vkQueueWaitIdle(device.value().graphicsQueue());

        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        std::printf("  barriers submit: ok\n");
    }

    device.value().waitIdle();
    std::printf("indirect buffer test passed\n");
    return 0;
}

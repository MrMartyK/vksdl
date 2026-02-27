#include <vksdl/vksdl.hpp>
#include <vksdl/barriers.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

// Helper: allocate a one-time command buffer from the given pool.
static VkCommandBuffer allocCmd(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult vr = vkAllocateCommandBuffers(device, &ai, &cmd);
    assert(vr == VK_SUCCESS);
    return cmd;
}

static void beginOneTime(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("queue ownership test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_queue_ownership")
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

    std::printf("  dedicated transfer: %s (family %u)\n",
                device.value().hasDedicatedTransfer() ? "yes" : "no",
                device.value().queueFamilies().transfer);

    auto& dev = device.value();
    auto& alloc = allocator.value();

    uint32_t graphicsFamily = dev.queueFamilies().graphics;
    uint32_t transferFamily = dev.hasDedicatedTransfer()
                              ? dev.queueFamilies().transfer
                              : dev.queueFamilies().graphics;

    // Create command pools on both queue families.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkCommandPool graphicsPool = VK_NULL_HANDLE;
    poolCI.queueFamilyIndex = graphicsFamily;
    VkResult vr = vkCreateCommandPool(dev.vkDevice(), &poolCI, nullptr, &graphicsPool);
    assert(vr == VK_SUCCESS);

    VkCommandPool transferPool = VK_NULL_HANDLE;
    poolCI.queueFamilyIndex = transferFamily;
    vr = vkCreateCommandPool(dev.vkDevice(), &poolCI, nullptr, &transferPool);
    assert(vr == VK_SUCCESS);

    {
        // Buffer ownership transfer: record both release and acquire barriers.
        // With a single queue family (no dedicated transfer), the barriers are
        // effectively no-ops (srcFamily == dstFamily), but the API must still
        // accept them without error.
        auto buf = vksdl::BufferBuilder(alloc)
            .storageBuffer()
            .size(1024)
            .build();
        assert(buf.ok());

        // Record release on transfer side.
        VkCommandBuffer releaseCmd = allocCmd(dev.vkDevice(), transferPool);
        beginOneTime(releaseCmd);
        vksdl::barrierQueueRelease(releaseCmd,
            buf.value().vkBuffer(), 0, VK_WHOLE_SIZE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            transferFamily, graphicsFamily);
        vkEndCommandBuffer(releaseCmd);

        // Record acquire on graphics side.
        VkCommandBuffer acquireCmd = allocCmd(dev.vkDevice(), graphicsPool);
        beginOneTime(acquireCmd);
        vksdl::barrierQueueAcquire(acquireCmd,
            buf.value().vkBuffer(), 0, VK_WHOLE_SIZE,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            transferFamily, graphicsFamily);
        vkEndCommandBuffer(acquireCmd);

        // Submit release signaling a semaphore, acquire waiting on it.
        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore xferDone = VK_NULL_HANDLE;
        vr = vkCreateSemaphore(dev.vkDevice(), &semCI, nullptr, &xferDone);
        assert(vr == VK_SUCCESS);

        VkSubmitInfo relSI{};
        relSI.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        relSI.commandBufferCount   = 1;
        relSI.pCommandBuffers      = &releaseCmd;
        relSI.signalSemaphoreCount = 1;
        relSI.pSignalSemaphores    = &xferDone;
        vr = vkQueueSubmit(dev.transferQueue(), 1, &relSI, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkSubmitInfo acqSI{};
        acqSI.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        acqSI.waitSemaphoreCount   = 1;
        acqSI.pWaitSemaphores      = &xferDone;
        acqSI.pWaitDstStageMask    = &waitStage;
        acqSI.commandBufferCount   = 1;
        acqSI.pCommandBuffers      = &acquireCmd;
        vr = vkQueueSubmit(dev.graphicsQueue(), 1, &acqSI, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);
        vkQueueWaitIdle(dev.graphicsQueue());

        vkDestroySemaphore(dev.vkDevice(), xferDone, nullptr);

        std::printf("  buffer queue ownership transfer: ok\n");
    }

    {
        // Image ownership transfer: record release and acquire with layout preserved.
        auto img = vksdl::ImageBuilder(alloc)
            .size(64, 64)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .colorAttachment()
            .build();
        assert(img.ok());

        VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkCommandBuffer releaseCmd = allocCmd(dev.vkDevice(), transferPool);
        beginOneTime(releaseCmd);
        // Transition to the layout first (UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL).
        vksdl::transitionToColorAttachment(releaseCmd, img.value().vkImage());
        vksdl::barrierQueueRelease(releaseCmd,
            img.value().vkImage(), layout,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            transferFamily, graphicsFamily);
        vkEndCommandBuffer(releaseCmd);

        VkCommandBuffer acquireCmd = allocCmd(dev.vkDevice(), graphicsPool);
        beginOneTime(acquireCmd);
        vksdl::barrierQueueAcquire(acquireCmd,
            img.value().vkImage(), layout,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            transferFamily, graphicsFamily);
        vkEndCommandBuffer(acquireCmd);

        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore imgXferDone = VK_NULL_HANDLE;
        vr = vkCreateSemaphore(dev.vkDevice(), &semCI, nullptr, &imgXferDone);
        assert(vr == VK_SUCCESS);

        VkSubmitInfo relSI{};
        relSI.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        relSI.commandBufferCount   = 1;
        relSI.pCommandBuffers      = &releaseCmd;
        relSI.signalSemaphoreCount = 1;
        relSI.pSignalSemaphores    = &imgXferDone;
        vr = vkQueueSubmit(dev.transferQueue(), 1, &relSI, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkSubmitInfo acqSI{};
        acqSI.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        acqSI.waitSemaphoreCount   = 1;
        acqSI.pWaitSemaphores      = &imgXferDone;
        acqSI.pWaitDstStageMask    = &waitStage;
        acqSI.commandBufferCount   = 1;
        acqSI.pCommandBuffers      = &acquireCmd;
        vr = vkQueueSubmit(dev.graphicsQueue(), 1, &acqSI, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);
        vkQueueWaitIdle(dev.graphicsQueue());

        vkDestroySemaphore(dev.vkDevice(), imgXferDone, nullptr);

        std::printf("  image queue ownership transfer: ok\n");
    }

    vkDestroyCommandPool(dev.vkDevice(), graphicsPool, nullptr);
    vkDestroyCommandPool(dev.vkDevice(), transferPool, nullptr);

    dev.waitIdle();
    std::printf("queue ownership test passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("async compute test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_async_compute")
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
                      .needAsyncCompute()
                      .preferDiscreteGpu()
                      .build();
    assert(device.ok());

    auto allocator = vksdl::Allocator::create(instance.value(), device.value());
    assert(allocator.ok());

    auto& dev = device.value();

    std::printf("  dedicated compute: %s (family %u, graphics %u)\n",
                dev.hasDedicatedCompute() ? "yes" : "no (using graphics)",
                dev.queueFamilies().compute, dev.queueFamilies().graphics);

    // hasAsyncCompute() and hasDedicatedCompute() must agree.
    assert(dev.hasAsyncCompute() == dev.hasDedicatedCompute());

    // computeQueue() must always return a non-null handle.
    assert(dev.computeQueue() != VK_NULL_HANDLE);

    // When no dedicated compute exists, compute queue == graphics queue.
    if (!dev.hasDedicatedCompute()) {
        assert(dev.computeQueue() == dev.graphicsQueue());
    }

    std::printf("  device accessors: ok\n");

    {
        // ComputeQueue creation.
        auto cq = vksdl::ComputeQueue::create(dev);
        assert(cq.ok() && "ComputeQueue creation failed");
        assert(cq.value().vkTimelineSemaphore() != VK_NULL_HANDLE);
        assert(cq.value().vkCommandPool() != VK_NULL_HANDLE);
        assert(cq.value().currentValue() == 0);
        std::printf("  create ComputeQueue: ok\n");
    }

    {
        // submit(lambda) with no-op workload -- non-blocking.
        auto cq = vksdl::ComputeQueue::create(dev);
        assert(cq.ok());

        auto pending = cq.value().submit([](VkCommandBuffer) {
            // intentional no-op -- just verify the submit path
        });
        assert(pending.ok() && "submit(lambda) failed");
        assert(pending.value().timelineValue == 1);
        assert(cq.value().currentValue() == 1);
        std::printf("  submit(lambda) non-blocking: ok\n");

        // Wait and verify completion.
        cq.value().waitFor(pending.value().timelineValue);
        assert(cq.value().isComplete(pending.value().timelineValue));
        std::printf("  waitFor + isComplete: ok\n");
    }

    {
        // submit(preRecorded) path.
        auto cq = vksdl::ComputeQueue::create(dev);
        assert(cq.ok());

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool = cq.value().vkCommandPool();
        cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkResult vr = vkAllocateCommandBuffers(dev.vkDevice(), &cmdAI, &cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        vkEndCommandBuffer(cmd);

        auto pending = cq.value().submit(cmd);
        assert(pending.ok() && "submit(preRecorded) failed");
        assert(pending.value().timelineValue == 1);

        cq.value().waitIdle();
        assert(cq.value().isComplete(1));
        std::printf("  submit(preRecorded): ok\n");
    }

    {
        // insertBufferAcquireBarrier no-op path (needsOwnershipTransfer = false).
        auto cq = vksdl::ComputeQueue::create(dev);
        assert(cq.ok());

        auto buf = vksdl::BufferBuilder(allocator.value()).storageBuffer().size(256).build();
        assert(buf.ok());

        // Submit a no-op compute workload.
        auto pending = cq.value().submit([](VkCommandBuffer) {});
        assert(pending.ok());
        cq.value().waitFor(pending.value().timelineValue);

        // Create a command buffer on the graphics queue to receive the barrier.
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = dev.queueFamilies().graphics;

        VkCommandPool graphicsPool = VK_NULL_HANDLE;
        VkResult vr = vkCreateCommandPool(dev.vkDevice(), &poolCI, nullptr, &graphicsPool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool = graphicsPool;
        cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer gfxCmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(dev.vkDevice(), &cmdAI, &gfxCmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(gfxCmd, &beginInfo);

        // No-op when needsOwnershipTransfer is false (same-family fallback).
        vksdl::PendingCompute noop;
        noop.needsOwnershipTransfer = false;
        vksdl::ComputeQueue::insertBufferAcquireBarrier(gfxCmd, buf.value().vkBuffer(),
                                                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                        VK_ACCESS_2_SHADER_STORAGE_READ_BIT, noop);

        vkEndCommandBuffer(gfxCmd);
        vkDestroyCommandPool(dev.vkDevice(), graphicsPool, nullptr);

        std::printf("  insertBufferAcquireBarrier no-op: ok\n");
    }

    {
        // Move semantics.
        auto cq = vksdl::ComputeQueue::create(dev);
        assert(cq.ok());

        VkSemaphore original = cq.value().vkTimelineSemaphore();
        vksdl::ComputeQueue moved = std::move(cq.value());
        assert(moved.vkTimelineSemaphore() == original);
        std::printf("  move semantics: ok\n");
    }

    dev.waitIdle();
    std::printf("async compute test passed\n");
    return 0;
}

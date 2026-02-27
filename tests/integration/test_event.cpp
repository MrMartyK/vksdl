#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

// Helper: allocate a one-shot command buffer from a pool.
static VkCommandBuffer allocCmd(VkDevice dev, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult vr = vkAllocateCommandBuffers(dev, &ai, &cmd);
    assert(vr == VK_SUCCESS);
    return cmd;
}

// Helper: begin one-shot recording.
static void beginCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

// Helper: end, submit, and wait for idle.
static void submitAndWait(VkCommandBuffer cmd, VkQueue queue, VkDevice dev) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    VkResult vr = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    assert(vr == VK_SUCCESS);
    vkQueueWaitIdle(queue);
}

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("event test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_event")
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

    VkDevice dev = device.value().vkDevice();
    VkQueue queue = device.value().graphicsQueue();

    // Command pool shared by all sub-tests.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags =
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = device.value().queueFamilies().graphics;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkResult vr = vkCreateCommandPool(dev, &poolCI, nullptr, &pool);
    assert(vr == VK_SUCCESS);

    // --- Test 1: create GpuEvent, verify handle ---
    {
        auto ev = vksdl::GpuEvent::create(device.value());
        assert(ev.ok() && "GpuEvent creation failed");
        assert(ev.value().native() != VK_NULL_HANDLE);
        assert(ev.value().vkEvent() != VK_NULL_HANDLE);
        assert(ev.value().native() == ev.value().vkEvent());
        std::printf("  create GpuEvent: ok\n");
    }

    // --- Test 2: set / wait / reset on a command buffer ---
    {
        auto ev = vksdl::GpuEvent::create(device.value());
        assert(ev.ok());

        // Both set and wait must be in the same command buffer (Vulkan rule for
        // GPU-only events without a pipeline barrier between submissions).
        VkCommandBuffer cmd = allocCmd(dev, pool);
        beginCmd(cmd);

        ev.value().set(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT);

        ev.value().wait(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);

        ev.value().reset(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        submitAndWait(cmd, queue, dev);
        std::printf("  set/wait/reset on command buffer: ok\n");
    }

    // --- Test 3: full-control wait() overload with custom VkDependencyInfo ---
    {
        auto ev = vksdl::GpuEvent::create(device.value());
        assert(ev.ok());

        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        mb.srcAccessMask = VK_ACCESS_2_NONE;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;

        VkCommandBuffer cmd = allocCmd(dev, pool);
        beginCmd(cmd);

        ev.value().set(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT);

        // Use the full-control overload.
        ev.value().wait(cmd, dep);

        ev.value().reset(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

        submitAndWait(cmd, queue, dev);
        std::printf("  full-control wait() overload: ok\n");
    }

    // --- Test 4: move constructor ---
    {
        auto ev = vksdl::GpuEvent::create(device.value());
        assert(ev.ok());

        VkEvent original = ev.value().native();
        vksdl::GpuEvent moved = std::move(ev.value());
        assert(moved.native() == original);
        // Moved-from handle should be null (destroy() is a no-op on it).
        assert(ev.value().native() == VK_NULL_HANDLE);
        std::printf("  move constructor: ok\n");
    }

    // --- Test 5: move-assign ---
    {
        auto ev1 = vksdl::GpuEvent::create(device.value());
        auto ev2 = vksdl::GpuEvent::create(device.value());
        assert(ev1.ok() && ev2.ok());

        VkEvent h2 = ev2.value().native();
        ev1.value() = std::move(ev2.value());
        assert(ev1.value().native() == h2);
        assert(ev2.value().native() == VK_NULL_HANDLE);
        std::printf("  move-assign: ok\n");
    }

    vkDestroyCommandPool(dev, pool, nullptr);
    device.value().waitIdle();
    std::printf("event test passed\n");
    return 0;
}

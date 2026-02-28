#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("query pool test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_query_pool")
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

    // 1. Create timestamp query pool
    {
        auto pool = vksdl::QueryPool::create(device.value(), VK_QUERY_TYPE_TIMESTAMP, 2);
        assert(pool.ok());
        assert(pool.value().vkQueryPool() != VK_NULL_HANDLE);
        std::printf("  create timestamp pool: ok\n");
    }

    // 2. Accessors
    {
        auto pool = vksdl::QueryPool::create(device.value(), VK_QUERY_TYPE_TIMESTAMP, 4);
        assert(pool.ok());
        assert(pool.value().type() == VK_QUERY_TYPE_TIMESTAMP);
        assert(pool.value().count() == 4);
        assert(pool.value().vkQueryPool() != VK_NULL_HANDLE);
        std::printf("  query pool accessors: ok\n");
    }

    // 3. Move semantics
    {
        auto pool = vksdl::QueryPool::create(device.value(), VK_QUERY_TYPE_TIMESTAMP, 2);
        assert(pool.ok());

        VkQueryPool handle = pool.value().vkQueryPool();
        vksdl::QueryPool moved = std::move(pool.value());
        assert(moved.vkQueryPool() == handle);
        assert(moved.type() == VK_QUERY_TYPE_TIMESTAMP);
        assert(moved.count() == 2);
        std::printf("  query pool move: ok\n");
    }

    // 4. resetQueries + writeTimestamp + getResults
    {
        auto pool = vksdl::QueryPool::create(device.value(), VK_QUERY_TYPE_TIMESTAMP, 2);
        assert(pool.ok());

        // Create one-shot command buffer
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = device.value().queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkResult vr = vkCreateCommandPool(device.value().vkDevice(), &poolCI, nullptr, &cmdPool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool = cmdPool;
        cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device.value().vkDevice(), &cmdAI, &cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vksdl::resetQueries(cmd, pool.value(), 0, 2);
        vksdl::writeTimestamp(cmd, pool.value(), VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0);
        vksdl::writeTimestamp(cmd, pool.value(), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 1);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(device.value().graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(device.value().graphicsQueue());

        auto results = pool.value().getResults(0, 2, VK_QUERY_RESULT_WAIT_BIT);
        assert(results.ok());
        assert(results.value().size() == 2);
        assert(results.value()[1] >= results.value()[0]);

        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        std::printf("  resetQueries + writeTimestamp + getResults: ok\n");
    }

    // 5. timestampPeriod
    {
        float period = device.value().timestampPeriod();
        assert(period > 0.0f);
        std::printf("  timestampPeriod (%.2f ns): ok\n", static_cast<double>(period));
    }

    device.value().waitIdle();
    std::printf("all query pool tests passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("transfer queue test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_transfer_queue")
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

    std::printf("  dedicated transfer queue: %s (family %u -> %u)\n",
                device.value().hasDedicatedTransfer() ? "yes" : "no (using graphics)",
                device.value().queueFamilies().transfer,
                device.value().queueFamilies().graphics);

    {
        auto tq = vksdl::TransferQueue::create(device.value(), allocator.value());
        assert(tq.ok() && "TransferQueue creation failed");
        assert(tq.value().vkTimelineSemaphore() != VK_NULL_HANDLE);
        assert(tq.value().currentValue() == 0);
        std::printf("  create TransferQueue: ok\n");
    }

    {
        auto tq = vksdl::TransferQueue::create(device.value(), allocator.value());
        assert(tq.ok());

        float vertices[] = {0.0f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f};

        auto buf = vksdl::BufferBuilder(allocator.value())
            .vertexBuffer()
            .size(sizeof(vertices))
            .build();
        assert(buf.ok());

        auto pending = tq.value().uploadAsync(buf.value(), vertices,
                                                sizeof(vertices));
        assert(pending.ok());
        assert(pending.value().buffer == buf.value().vkBuffer());
        assert(tq.value().currentValue() == 1);

        // Transfer should already be complete (we wait inside uploadAsync).
        assert(tq.value().isComplete(pending.value().timelineValue));
        std::printf("  async upload: ok (ownership transfer: %s)\n",
                    pending.value().needsOwnershipTransfer ? "yes" : "no");
    }

    {
        auto tq = vksdl::TransferQueue::create(device.value(), allocator.value());
        assert(tq.ok());

        std::vector<vksdl::Buffer> buffers;
        for (int i = 0; i < 5; ++i) {
            float data[] = {static_cast<float>(i)};
            auto buf = vksdl::BufferBuilder(allocator.value())
                .storageBuffer()
                .size(sizeof(data))
                .build();
            assert(buf.ok());

            auto pending = tq.value().uploadAsync(buf.value(), data, sizeof(data));
            assert(pending.ok());

            buffers.push_back(std::move(buf.value()));
        }

        assert(tq.value().currentValue() == 5);
        tq.value().waitIdle();
        std::printf("  multiple uploads (5): ok\n");
    }

    {
        auto tq = vksdl::TransferQueue::create(device.value(), allocator.value());
        assert(tq.ok());

        VkSemaphore original = tq.value().vkTimelineSemaphore();
        vksdl::TransferQueue moved = std::move(tq.value());
        assert(moved.vkTimelineSemaphore() == original);
        std::printf("  move semantics: ok\n");
    }

    {
        // Test the no-op path of insertAcquireBarrier (same queue family).
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

        vksdl::PendingTransfer transfer;
        transfer.needsOwnershipTransfer = false;
        vksdl::TransferQueue::insertAcquireBarrier(cmd, transfer);

        vkEndCommandBuffer(cmd);
        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        std::printf("  insertAcquireBarrier no-op: ok\n");
    }

    device.value().waitIdle();
    std::printf("transfer queue test passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("debug name test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_debug_name")
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

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(64).uniformBuffer().build();
        assert(buf.ok());

        vksdl::debugName(device.value().vkDevice(), buf.value().vkBuffer(), "test uniform buffer");
        std::printf("  debugName buffer: ok\n");
    }

    {
        auto sampler = vksdl::SamplerBuilder(device.value()).nearest().repeat().build();
        assert(sampler.ok());

        vksdl::debugName(device.value().vkDevice(), sampler.value().vkSampler(), "test sampler");
        std::printf("  debugName sampler: ok\n");
    }

    {
        auto buf = vksdl::BufferBuilder(allocator.value()).size(128).uniformBuffer().build();
        assert(buf.ok());

        vksdl::debugName(device.value().vkDevice(), VK_OBJECT_TYPE_BUFFER,
                         reinterpret_cast<std::uint64_t>(buf.value().vkBuffer()),
                         "test buffer via core overload");
        std::printf("  debugName core overload: ok\n");
    }

    // Test new overloads: command pool, semaphore, fence, event,
    // descriptor set layout, pipeline layout, shader module.
    {
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = device.value().queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkResult vr = vkCreateCommandPool(device.value().vkDevice(), &poolCI, nullptr, &cmdPool);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), cmdPool, "test command pool");
        vkDestroyCommandPool(device.value().vkDevice(), cmdPool, nullptr);
        std::printf("  debugName command pool: ok\n");
    }

    {
        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkSemaphore sem = VK_NULL_HANDLE;
        VkResult vr = vkCreateSemaphore(device.value().vkDevice(), &semCI, nullptr, &sem);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), sem, "test semaphore");
        vkDestroySemaphore(device.value().vkDevice(), sem, nullptr);
        std::printf("  debugName semaphore: ok\n");
    }

    {
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence = VK_NULL_HANDLE;
        VkResult vr = vkCreateFence(device.value().vkDevice(), &fenceCI, nullptr, &fence);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), fence, "test fence");
        vkDestroyFence(device.value().vkDevice(), fence, nullptr);
        std::printf("  debugName fence: ok\n");
    }

    {
        VkEventCreateInfo eventCI{};
        eventCI.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;

        VkEvent event = VK_NULL_HANDLE;
        VkResult vr = vkCreateEvent(device.value().vkDevice(), &eventCI, nullptr, &event);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), event, "test event");
        vkDestroyEvent(device.value().vkDevice(), event, nullptr);
        std::printf("  debugName event: ok\n");
    }

    {
        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

        VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
        VkResult vr =
            vkCreateDescriptorSetLayout(device.value().vkDevice(), &layoutCI, nullptr, &dsLayout);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), dsLayout, "test descriptor set layout");
        vkDestroyDescriptorSetLayout(device.value().vkDevice(), dsLayout, nullptr);
        std::printf("  debugName descriptor set layout: ok\n");
    }

    {
        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
        VkResult vr =
            vkCreatePipelineLayout(device.value().vkDevice(), &plCI, nullptr, &pipeLayout);
        assert(vr == VK_SUCCESS);

        vksdl::debugName(device.value().vkDevice(), pipeLayout, "test pipeline layout");
        vkDestroyPipelineLayout(device.value().vkDevice(), pipeLayout, nullptr);
        std::printf("  debugName pipeline layout: ok\n");
    }

    device.value().waitIdle();
    std::printf("all debug name tests passed\n");
    return 0;
}

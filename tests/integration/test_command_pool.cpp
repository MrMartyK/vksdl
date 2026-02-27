#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("command pool test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_command_pool")
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

    std::uint32_t graphics = device.value().queueFamilies().graphics;

    std::printf("command pool test\n");

    {
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok() && "CommandPool creation failed");
        assert(pool.value().native() != VK_NULL_HANDLE);
        assert(pool.value().vkCommandPool() == pool.value().native());
        std::printf("  1. create pool: ok\n");
    }

    {
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok());

        auto cmd = pool.value().allocate();
        assert(cmd.ok() && "single allocate failed");
        assert(cmd.value() != VK_NULL_HANDLE);
        std::printf("  2. allocate single buffer: ok\n");
    }

    {
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok());

        auto cmds = pool.value().allocate(4u);
        assert(cmds.ok() && "batch allocate failed");
        assert(cmds.value().size() == 4);
        for (VkCommandBuffer cb : cmds.value()) {
            assert(cb != VK_NULL_HANDLE);
        }
        std::printf("  3. allocate 4 buffers: ok\n");
    }

    {
        // allocate(0) must return an empty vector without error.
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok());

        auto cmds = pool.value().allocate(0u);
        assert(cmds.ok());
        assert(cmds.value().empty());
        std::printf("  4. allocate(0): ok\n");
    }

    {
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok());

        auto cmd = pool.value().allocate();
        assert(cmd.ok());

        pool.value().reset();

        // After reset the pool is still valid and we can allocate again.
        auto cmd2 = pool.value().allocate();
        assert(cmd2.ok());
        assert(cmd2.value() != VK_NULL_HANDLE);
        std::printf("  5. reset pool: ok\n");
    }

    {
        auto pool = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool.ok());

        VkCommandPool original = pool.value().native();

        vksdl::CommandPool moved = std::move(pool.value());
        assert(moved.native() == original);
        // pool.value() is now in moved-from state -- pool_ is VK_NULL_HANDLE.
        // We cannot safely check pool.value().native() after a move of the inner value.

        // Move assignment.
        auto pool2 = vksdl::CommandPool::create(device.value(), graphics);
        assert(pool2.ok());
        pool2.value() = std::move(moved);
        assert(pool2.value().native() == original);
        std::printf("  6. move semantics: ok\n");
    }

    device.value().waitIdle();
    std::printf("command pool test passed\n");
    return 0;
}

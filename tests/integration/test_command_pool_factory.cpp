#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("command pool factory test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_command_pool_factory")
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

    std::printf("command pool factory test\n");

    {
        auto factory = vksdl::CommandPoolFactory::create(device.value(), graphics);
        assert(factory.ok() && "CommandPoolFactory creation failed");

        vksdl::CommandPool& pool = factory.value().getForCurrentThread();
        assert(pool.native() != VK_NULL_HANDLE);
        std::printf("  1. create factory, get pool: ok\n");
    }

    {
        auto factory = vksdl::CommandPoolFactory::create(device.value(), graphics);
        assert(factory.ok());

        vksdl::CommandPool& p1 = factory.value().getForCurrentThread();
        vksdl::CommandPool& p2 = factory.value().getForCurrentThread();
        // Same thread must yield the same pool both times.
        assert(&p1 == &p2);
        std::printf("  2. idempotent get: ok\n");
    }

    {
        auto factory = vksdl::CommandPoolFactory::create(device.value(), graphics);
        assert(factory.ok());

        vksdl::CommandPool& pool = factory.value().getForCurrentThread();
        auto cmd = pool.allocate();
        assert(cmd.ok());
        assert(cmd.value() != VK_NULL_HANDLE);

        factory.value().resetAll();

        // After resetAll, pool is still valid and can allocate again.
        auto cmd2 = pool.allocate();
        assert(cmd2.ok());
        assert(cmd2.value() != VK_NULL_HANDLE);
        std::printf("  3. resetAll: ok\n");
    }

    {
        // Two threads each get their own distinct pool.
        auto factory = vksdl::CommandPoolFactory::create(device.value(), graphics);
        assert(factory.ok());

        vksdl::CommandPool* mainPool = &factory.value().getForCurrentThread();
        vksdl::CommandPool* threadPool = nullptr;

        std::thread worker([&]() { threadPool = &factory.value().getForCurrentThread(); });
        worker.join();

        assert(mainPool != nullptr);
        assert(threadPool != nullptr);
        assert(mainPool != threadPool);
        assert(mainPool->native() != VK_NULL_HANDLE);
        assert(threadPool->native() != VK_NULL_HANDLE);
        std::printf("  4. separate pools per thread: ok\n");
    }

    {
        auto factory = vksdl::CommandPoolFactory::create(device.value(), graphics);
        assert(factory.ok());

        // Populate the factory from both the main thread and a worker.
        (void) factory.value().getForCurrentThread();
        std::thread([&]() { (void) factory.value().getForCurrentThread(); }).join();

        // Move-construct.
        vksdl::CommandPoolFactory moved = std::move(factory.value());
        // After move, the moved-from factory is gone; moved owns the pools.
        moved.resetAll(); // must not crash
        std::printf("  5. move semantics: ok\n");
    }

    device.value().waitIdle();
    std::printf("command pool factory test passed\n");
    return 0;
}

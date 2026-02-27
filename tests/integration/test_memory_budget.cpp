#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <vector>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("memory budget test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_memory_budget")
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

    std::printf("  VK_EXT_memory_budget: %s\n",
                device.value().hasMemoryBudget() ? "supported" : "not supported");
    std::printf("  allocator hasMemoryBudget: %s\n",
                allocator.value().hasMemoryBudget() ? "yes" : "no");

    assert(allocator.value().hasMemoryBudget() == device.value().hasMemoryBudget());

    {
        auto budgets = allocator.value().queryBudget();
        assert(!budgets.empty());
        std::printf("  queryBudget: %zu heap(s)\n", budgets.size());
        for (std::size_t i = 0; i < budgets.size(); ++i) {
            std::printf("    heap[%zu]: usage=%" PRIu64 " B  budget=%" PRIu64 " B\n",
                        i,
                        static_cast<unsigned long long>(budgets[i].usage),
                        static_cast<unsigned long long>(budgets[i].budget));
            // Budget must be non-zero (at minimum equals heap size).
            assert(budgets[i].budget > 0);
        }
    }

    {
        float pct = allocator.value().gpuMemoryUsagePercent();
        std::printf("  gpuMemoryUsagePercent: %.1f%%\n", static_cast<double>(pct));
        // Percentage is in [0, 100].
        assert(pct >= 0.0f);
        assert(pct <= 100.0f);
    }

    device.value().waitIdle();
    std::printf("memory budget test passed\n");
    return 0;
}

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

        std::uint64_t totalDeviceLocalUsage = 0;
        std::uint64_t totalDeviceLocalBudget = 0;

        for (std::size_t i = 0; i < budgets.size(); ++i) {
            std::printf("    heap[%zu]: usage=%" PRIu64 " B  budget=%" PRIu64 " B"
                        "  heapSize=%" PRIu64 " B  flags=0x%x\n",
                        i, static_cast<unsigned long long>(budgets[i].usage),
                        static_cast<unsigned long long>(budgets[i].budget),
                        static_cast<unsigned long long>(budgets[i].heapSize),
                        static_cast<unsigned>(budgets[i].flags));
            // Budget must be non-zero (at minimum equals heap size).
            assert(budgets[i].budget > 0);
            // Heap size must be non-zero.
            assert(budgets[i].heapSize > 0);

            if (budgets[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                totalDeviceLocalUsage += budgets[i].usage;
                totalDeviceLocalBudget += budgets[i].budget;
            }
        }

        // Verify gpuMemoryUsagePercent sums ALL device-local heaps, not just
        // the largest one. On ReBAR systems this matters because the BAR region
        // is a separate device-local heap.
        float pct = allocator.value().gpuMemoryUsagePercent();
        std::printf("  gpuMemoryUsagePercent: %.1f%%\n", static_cast<double>(pct));

        if (totalDeviceLocalBudget > 0) {
            float expected = static_cast<float>(totalDeviceLocalUsage) /
                             static_cast<float>(totalDeviceLocalBudget) * 100.0f;
            float diff = pct - expected;
            if (diff < 0.0f)
                diff = -diff;
            // Allow 0.1% tolerance for floating-point and timing between queries.
            std::printf("  expected (summed): %.1f%%  diff: %.4f%%\n",
                        static_cast<double>(expected), static_cast<double>(diff));
            assert(diff < 0.1f);
        }

        assert(pct >= 0.0f);
    }

    device.value().waitIdle();
    std::printf("memory budget test passed\n");
    return 0;
}

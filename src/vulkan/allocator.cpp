#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

#include <cstdint>

namespace vksdl {

Allocator::~Allocator() {
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
}

Allocator::Allocator(Allocator&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_),
      hasMemoryBudget_(o.hasMemoryBudget_) {
    o.allocator_       = nullptr;
    o.device_          = VK_NULL_HANDLE;
    o.hasMemoryBudget_ = false;
}

Allocator& Allocator::operator=(Allocator&& o) noexcept {
    if (this != &o) {
        if (allocator_ != nullptr) {
            vmaDestroyAllocator(allocator_);
        }
        allocator_       = o.allocator_;
        device_          = o.device_;
        hasMemoryBudget_ = o.hasMemoryBudget_;
        o.allocator_       = nullptr;
        o.device_          = VK_NULL_HANDLE;
        o.hasMemoryBudget_ = false;
    }
    return *this;
}

Result<Allocator> Allocator::create(const Instance& instance, const Device& device) {
    VmaAllocatorCreateInfo ci{};
    ci.instance         = instance.vkInstance();
    ci.physicalDevice   = device.vkPhysicalDevice();
    ci.device           = device.vkDevice();
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    // Always enable BDA support. bufferDeviceAddress is a required Vulkan 1.2
    // feature (promoted from VK_KHR_buffer_device_address), so every Vulkan 1.3
    // driver supports it. Zero overhead when BDA is not actually used. Avoids
    // a footgun where RT code silently fails without this flag.
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (device.hasMemoryBudget()) {
        ci.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (device.hasMemoryPriority()) {
        ci.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

    Allocator a;
    a.device_          = device.vkDevice();
    a.hasMemoryBudget_ = device.hasMemoryBudget();

    VkResult vr = vmaCreateAllocator(&ci, &a.allocator_);
    if (vr != VK_SUCCESS) {
        return Error{"create allocator", static_cast<std::int32_t>(vr),
                     "vmaCreateAllocator failed"};
    }

    return a;
}

std::vector<HeapBudget> Allocator::queryBudget() const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(
        [&]() -> VkPhysicalDevice {
            // Retrieve the physical device handle from VMA.
            VmaAllocatorInfo info{};
            vmaGetAllocatorInfo(allocator_, &info);
            return info.physicalDevice;
        }(),
        &memProps);

    std::uint32_t heapCount = memProps.memoryHeapCount;

    // VMA fills one VmaBudget per heap.
    std::vector<VmaBudget> vmaBudgets(heapCount);
    vmaGetHeapBudgets(allocator_, vmaBudgets.data());

    std::vector<HeapBudget> result(heapCount);
    for (std::uint32_t i = 0; i < heapCount; ++i) {
        result[i].usage  = vmaBudgets[i].usage;
        result[i].budget = vmaBudgets[i].budget;
    }
    return result;
}

float Allocator::gpuMemoryUsagePercent() const {
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(allocator_, &info);
        physDev = info.physicalDevice;
    }
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    // Find the largest DEVICE_LOCAL heap.
    std::uint32_t bestHeap  = UINT32_MAX;
    VkDeviceSize  bestSize  = 0;
    for (std::uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (!(memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            continue;
        if (memProps.memoryHeaps[i].size > bestSize) {
            bestSize = memProps.memoryHeaps[i].size;
            bestHeap = i;
        }
    }

    if (bestHeap == UINT32_MAX || bestSize == 0) return 0.0f;

    std::vector<VmaBudget> vmaBudgets(memProps.memoryHeapCount);
    vmaGetHeapBudgets(allocator_, vmaBudgets.data());

    VkDeviceSize usage  = vmaBudgets[bestHeap].usage;
    VkDeviceSize budget = vmaBudgets[bestHeap].budget;

    if (budget == 0) return 0.0f;
    float pct = static_cast<float>(usage) / static_cast<float>(budget) * 100.0f;
    return pct < 0.0f ? 0.0f : (pct > 100.0f ? 100.0f : pct);
}

} // namespace vksdl

#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

// Forward-declare VMA handle to avoid pulling vk_mem_alloc.h into user code.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace vksdl {

class Instance;
class Device;

// Per-heap memory budget snapshot.
// usage    -- bytes currently allocated (includes all VMA pools + implicit objects).
// budget   -- bytes estimated available to this process by the OS.
//             Exceeding budget may trigger eviction or stalls.
// heapSize -- physical heap size from VkMemoryHeap::size.
// flags    -- VkMemoryHeapFlags from VkMemoryHeap::flags
//             (e.g. VK_MEMORY_HEAP_DEVICE_LOCAL_BIT).
// When hasMemoryBudget() is false, usage equals blockBytes and budget equals
// heapSize (both derived from VkPhysicalDeviceMemoryProperties, no OS query).
struct HeapBudget {
    std::uint64_t usage = 0;
    std::uint64_t budget = 0;
    std::uint64_t heapSize = 0;
    VkMemoryHeapFlags flags = 0;
};

// Thread safety: thread-confined. VMA allocations require external
// synchronization unless VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT
// is NOT set (vksdl does not set it).
class Allocator {
  public:
    [[nodiscard]] static Result<Allocator> create(const Instance& instance, const Device& device);

    ~Allocator();
    Allocator(Allocator&&) noexcept;
    Allocator& operator=(Allocator&&) noexcept;
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    [[nodiscard]] VmaAllocator native() const {
        return allocator_;
    }
    [[nodiscard]] VmaAllocator vmaAllocator() const {
        return native();
    }
    [[nodiscard]] VkDevice vkDevice() const {
        return device_;
    }

    // Returns true when VK_EXT_memory_budget was enabled at allocator creation.
    [[nodiscard]] bool hasMemoryBudget() const {
        return hasMemoryBudget_;
    }

    // Returns per-heap budget snapshots. Element count equals the physical
    // device heap count. When hasMemoryBudget() is false, values fall back
    // to VMA statistics (no OS query).
    [[nodiscard]] std::vector<HeapBudget> queryBudget() const;

    // Returns usage/budget ratio across all DEVICE_LOCAL heaps as a
    // percentage. Returns 0 when no device-local heap exists. Values above
    // 100 indicate over-budget (other processes competing for VRAM).
    // Sums usage and budget across all device-local heaps (handles ReBAR
    // systems where VRAM and host-visible BAR are separate device-local heaps).
    [[nodiscard]] float gpuMemoryUsagePercent() const;

  private:
    Allocator() = default;

    VmaAllocator allocator_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    bool hasMemoryBudget_ = false;
};

} // namespace vksdl

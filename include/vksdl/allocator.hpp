#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

// Forward-declare VMA handle to avoid pulling vk_mem_alloc.h into user code.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace vksdl {

class Instance;
class Device;

class Allocator {
public:
    [[nodiscard]] static Result<Allocator> create(const Instance& instance, const Device& device);

    ~Allocator();
    Allocator(Allocator&&) noexcept;
    Allocator& operator=(Allocator&&) noexcept;
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    [[nodiscard]] VmaAllocator native()       const { return allocator_; }
    [[nodiscard]] VmaAllocator vmaAllocator() const { return native(); }
    [[nodiscard]] VkDevice     vkDevice()     const { return device_; }

private:
    Allocator() = default;

    VmaAllocator allocator_ = nullptr;
    VkDevice     device_    = VK_NULL_HANDLE;
};

} // namespace vksdl

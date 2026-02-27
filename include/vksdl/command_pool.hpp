#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// RAII wrapper for VkCommandPool. Created with TRANSIENT + RESET bits
// for per-frame command buffer reuse.
//
// Thread safety: thread-confined. Each thread needs its own pool.
class CommandPool {
public:
    [[nodiscard]] static Result<CommandPool> create(
        const Device& device, std::uint32_t queueFamily);

    ~CommandPool();
    CommandPool(CommandPool&&) noexcept;
    CommandPool& operator=(CommandPool&&) noexcept;
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    [[nodiscard]] VkCommandPool native()        const { return pool_; }
    [[nodiscard]] VkCommandPool vkCommandPool() const { return native(); }

    // Allocate a single primary command buffer.
    [[nodiscard]] Result<VkCommandBuffer> allocate();

    // Allocate multiple primary command buffers.
    [[nodiscard]] Result<std::vector<VkCommandBuffer>> allocate(std::uint32_t count);

    // Reset the pool, recycling all command buffers.
    void reset();

private:
    friend class CommandPoolFactory;

    CommandPool() = default;
    void destroy();

    VkDevice      device_ = VK_NULL_HANDLE;
    VkCommandPool pool_   = VK_NULL_HANDLE;
};

} // namespace vksdl

#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace vksdl {

class CommandPool;
class Device;

// Per-thread command pool factory. Owns all pools and provides zero-overhead
// pool access after the first call per thread via a thread_local cache.
//
// Thread safety: getForCurrentThread() acquires a lock only on first call per
// thread (pool creation). Subsequent calls on the same thread are lock-free.
// resetAll() is synchronized and must not race with getForCurrentThread() on
// new threads.
class CommandPoolFactory {
public:
    [[nodiscard]] static Result<CommandPoolFactory> create(
        const Device& device, std::uint32_t queueFamily);

    ~CommandPoolFactory();
    CommandPoolFactory(CommandPoolFactory&&) noexcept;
    CommandPoolFactory& operator=(CommandPoolFactory&&) noexcept;
    CommandPoolFactory(const CommandPoolFactory&) = delete;
    CommandPoolFactory& operator=(const CommandPoolFactory&) = delete;

    // Get the command pool for the current thread. Creates one on first call
    // per thread. Lock-free after first call.
    [[nodiscard]] CommandPool& getForCurrentThread();

    // Reset all registered pools. Call between frames when all worker threads
    // have finished recording and no thread is allocating from any pool.
    void resetAll();

    // Implementation detail: exposed for thread_local cache in .cpp only.
    struct Impl;

private:
    CommandPoolFactory() = default;

    std::unique_ptr<Impl> impl_;
};

} // namespace vksdl

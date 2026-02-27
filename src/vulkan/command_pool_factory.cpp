#include <vksdl/command_pool_factory.hpp>
#include <vksdl/command_pool.hpp>
#include <vksdl/device.hpp>

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vksdl {

// Monotonic generation counter. Each factory gets a unique ID to prevent
// ABA problems when a new factory's Impl happens to land at the same address
// as a previously destroyed factory.
static std::atomic<std::uint64_t> g_factoryGen{0};

struct CommandPoolFactory::Impl {
    VkDevice        device = VK_NULL_HANDLE;
    std::uint32_t   family = 0;
    std::uint64_t   generation = 0;

    std::mutex                                         mutex;
    std::vector<std::unique_ptr<CommandPool>>          pools;
    std::unordered_map<std::thread::id, CommandPool*>  threadMap;
};

// Thread-local map keyed by generation (not address) to the per-thread
// CommandPool for that factory. Using a monotonic generation prevents
// stale entries from matching a new factory at the same address.
namespace {
thread_local std::unordered_map<std::uint64_t, CommandPool*> tl_poolMap;
} // namespace

Result<CommandPoolFactory> CommandPoolFactory::create(const Device& device,
                                                       std::uint32_t queueFamily) {
    CommandPoolFactory f;
    f.impl_ = std::make_unique<Impl>();
    f.impl_->device     = device.vkDevice();
    f.impl_->family     = queueFamily;
    f.impl_->generation = g_factoryGen.fetch_add(1, std::memory_order_relaxed);
    return f;
}

CommandPoolFactory::~CommandPoolFactory() {
    if (!impl_) return;
    // Erase this thread's cache entry. Worker threads with stale entries
    // are safe: their keys use a unique generation that will never match
    // any future factory.
    tl_poolMap.erase(impl_->generation);
}

CommandPoolFactory::CommandPoolFactory(CommandPoolFactory&&) noexcept = default;
CommandPoolFactory& CommandPoolFactory::operator=(CommandPoolFactory&&) noexcept = default;

CommandPool& CommandPoolFactory::getForCurrentThread() {
    std::uint64_t key = impl_->generation;

    // Fast path: thread already has a pool for this factory instance.
    auto it = tl_poolMap.find(key);
    if (it != tl_poolMap.end()) {
        return *it->second;
    }

    // Slow path: first call from this thread. Create and register under lock.
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Re-check the factory's threadMap in case another code path already
    // registered this thread (handles TID reuse edge case).
    std::thread::id tid = std::this_thread::get_id();
    auto fit = impl_->threadMap.find(tid);
    if (fit != impl_->threadMap.end()) {
        tl_poolMap.emplace(key, fit->second);
        return *fit->second;
    }

    // CommandPoolFactory is a friend of CommandPool, so we can default-
    // construct directly with new (std::make_unique cannot reach private ctors
    // from a friend function -- it invokes new via a non-friend template).
    CommandPool* raw = new CommandPool(); // NOLINT: wrapped immediately below
    raw->device_ = impl_->device;

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = impl_->family;

    VkResult vr = vkCreateCommandPool(impl_->device, &poolCI, nullptr, &raw->pool_);
    if (vr != VK_SUCCESS) {
        delete raw;
        // No Result to propagate from a reference-returning function. A failed
        // command pool allocation is unrecoverable (device lost or OOM).
        std::abort();
    }

    impl_->pools.emplace_back(raw);
    impl_->threadMap.emplace(tid, raw);
    tl_poolMap.emplace(key, raw);

    return *raw;
}

void CommandPoolFactory::resetAll() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& p : impl_->pools) {
        p->reset();
    }
}

} // namespace vksdl

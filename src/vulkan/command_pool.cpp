#include <vksdl/command_pool.hpp>
#include <vksdl/device.hpp>

namespace vksdl {

void CommandPool::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, pool_, nullptr);

    device_ = VK_NULL_HANDLE;
    pool_   = VK_NULL_HANDLE;
}

CommandPool::~CommandPool() { destroy(); }

CommandPool::CommandPool(CommandPool&& o) noexcept
    : device_(o.device_), pool_(o.pool_) {
    o.device_ = VK_NULL_HANDLE;
    o.pool_   = VK_NULL_HANDLE;
}

CommandPool& CommandPool::operator=(CommandPool&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        pool_   = o.pool_;
        o.device_ = VK_NULL_HANDLE;
        o.pool_   = VK_NULL_HANDLE;
    }
    return *this;
}

Result<CommandPool> CommandPool::create(const Device& device,
                                         std::uint32_t queueFamily) {
    CommandPool cp;
    cp.device_ = device.vkDevice();

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = queueFamily;

    VkResult vr = vkCreateCommandPool(cp.device_, &poolCI, nullptr, &cp.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create command pool", static_cast<std::int32_t>(vr),
                     "vkCreateCommandPool failed"};
    }

    return cp;
}

Result<VkCommandBuffer> CommandPool::allocate() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = pool_;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult vr = vkAllocateCommandBuffers(device_, &allocInfo, &cmd);
    if (vr != VK_SUCCESS) {
        return Error{"allocate command buffer", static_cast<std::int32_t>(vr),
                     "vkAllocateCommandBuffers failed"};
    }

    return cmd;
}

Result<std::vector<VkCommandBuffer>> CommandPool::allocate(std::uint32_t count) {
    if (count == 0) return std::vector<VkCommandBuffer>{};

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = pool_;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = count;

    std::vector<VkCommandBuffer> cmds(count, VK_NULL_HANDLE);
    VkResult vr = vkAllocateCommandBuffers(device_, &allocInfo, cmds.data());
    if (vr != VK_SUCCESS) {
        return Error{"allocate command buffers", static_cast<std::int32_t>(vr),
                     "vkAllocateCommandBuffers failed"};
    }

    return cmds;
}

void CommandPool::reset() {
    if (pool_ == VK_NULL_HANDLE) return;
    vkResetCommandPool(device_, pool_, 0);
}

} // namespace vksdl

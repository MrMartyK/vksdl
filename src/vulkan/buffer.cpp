#include <vksdl/buffer.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4189) // local variable initialized but not referenced
#pragma warning(disable : 4244) // conversion, possible loss of data
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstring>

namespace vksdl {

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

Buffer::Buffer(Buffer&& o) noexcept
    : device_(o.device_), allocator_(o.allocator_), buffer_(o.buffer_),
      allocation_(o.allocation_), size_(o.size_), mapped_(o.mapped_) {
    o.device_     = VK_NULL_HANDLE;
    o.allocator_  = nullptr;
    o.buffer_     = VK_NULL_HANDLE;
    o.allocation_ = nullptr;
    o.size_       = 0;
    o.mapped_     = nullptr;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        if (buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }
        device_       = o.device_;
        allocator_    = o.allocator_;
        buffer_       = o.buffer_;
        allocation_   = o.allocation_;
        size_         = o.size_;
        mapped_       = o.mapped_;
        o.device_     = VK_NULL_HANDLE;
        o.allocator_  = nullptr;
        o.buffer_     = VK_NULL_HANDLE;
        o.allocation_ = nullptr;
        o.size_       = 0;
        o.mapped_     = nullptr;
    }
    return *this;
}

VkDeviceAddress Buffer::deviceAddress() const {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer_;
    return vkGetBufferDeviceAddress(device_, &info);
}

BufferBuilder::BufferBuilder(const Allocator& allocator)
    : allocator_(allocator.vmaAllocator()), device_(allocator.vkDevice()) {}

BufferBuilder& BufferBuilder::size(VkDeviceSize bytes) {
    size_ = bytes;
    return *this;
}

// Vertex and index buffers include SHADER_DEVICE_ADDRESS_BIT (Vulkan 1.2 core,
// always enabled) so device addresses are available for any use case.
// Chain .accelerationStructureInput() when the buffer feeds into BLAS builds.
BufferBuilder& BufferBuilder::vertexBuffer() {
    usage_ = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::indexBuffer() {
    usage_ = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::uniformBuffer() {
    usage_  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    mapped_ = true;
    return *this;
}

BufferBuilder& BufferBuilder::storageBuffer() {
    usage_ = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::stagingBuffer() {
    usage_  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    mapped_ = true;
    return *this;
}

BufferBuilder& BufferBuilder::indirectBuffer() {
    usage_ = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::scratchBuffer() {
    usage_ = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::accelerationStructureStorage() {
    usage_ = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::shaderBindingTable() {
    usage_  = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    mapped_ = true;
    return *this;
}

BufferBuilder& BufferBuilder::usage(VkBufferUsageFlags flags) {
    usage_ = flags;
    return *this;
}

BufferBuilder& BufferBuilder::deviceAddressable() {
    usage_ |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return *this;
}

BufferBuilder& BufferBuilder::accelerationStructureInput() {
    usage_ |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    return *this;
}

BufferBuilder& BufferBuilder::mapped() {
    mapped_ = true;
    return *this;
}

BufferBuilder& BufferBuilder::memoryPriority(float p) {
    priority_ = p;
    return *this;
}

Result<Buffer> BufferBuilder::build() {
    if (size_ == 0) {
        return Error{"create buffer", 0,
                     "buffer size is 0 -- call size(bytes)"};
    }
    if (usage_ == 0) {
        return Error{"create buffer", 0,
                     "no usage flags -- call vertexBuffer(), stagingBuffer(), etc."};
    }

    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = size_;
    bufCI.usage = usage_;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage    = VMA_MEMORY_USAGE_AUTO;
    allocCI.priority = priority_;

    if (mapped_) {
        allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    Buffer buf;
    buf.device_    = device_;
    buf.allocator_ = allocator_;
    buf.size_      = size_;

    VmaAllocationInfo allocInfo{};
    VkResult vr = vmaCreateBuffer(allocator_, &bufCI, &allocCI,
                                   &buf.buffer_, &buf.allocation_, &allocInfo);
    if (vr != VK_SUCCESS) {
        return Error{"create buffer", static_cast<std::int32_t>(vr),
                     "vmaCreateBuffer failed"};
    }

    if (mapped_) {
        buf.mapped_ = allocInfo.pMappedData;
    }

    return buf;
}

Result<void> uploadToBuffer(
    const Allocator& allocator,
    const Device& device,
    const Buffer& dst,
    const void* data,
    VkDeviceSize size) {

    VkBufferCreateInfo stagingCI{};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size  = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI{};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuf    = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc  = nullptr;
    VmaAllocationInfo stagingInfo{};

    VkResult vr = vmaCreateBuffer(allocator.vmaAllocator(), &stagingCI, &stagingAllocCI,
                                   &stagingBuf, &stagingAlloc, &stagingInfo);
    if (vr != VK_SUCCESS) {
        return Error{"upload to buffer", static_cast<std::int32_t>(vr),
                     "failed to create staging buffer"};
    }

    std::memcpy(stagingInfo.pMappedData, data, static_cast<std::size_t>(size));

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.queueFamilies().graphics;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(device.vkDevice(), &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to buffer", static_cast<std::int32_t>(vr),
                     "failed to create command pool for transfer"};
    }

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = cmdPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(device.vkDevice(), &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to buffer", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer for transfer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, stagingBuf, dst.vkBuffer(), 1, &region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vr = vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to buffer", static_cast<std::int32_t>(vr),
                     "vkQueueSubmit failed for transfer"};
    }
    // VKSDL_BLOCKING_WAIT: init-time staging upload waits for transfer completion.
    vkQueueWaitIdle(device.graphicsQueue());

    vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
    vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);

    return {};
}

Result<Buffer> uploadBuffer(
    const Allocator& allocator,
    const Device& device,
    VkBufferUsageFlags usage,
    const void* data,
    VkDeviceSize size) {

    // Ensure the buffer can receive a transfer.
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    auto buf = BufferBuilder(allocator)
        .usage(usage)
        .size(size)
        .build();

    if (!buf.ok()) return buf.error();

    auto uploadRes = uploadToBuffer(allocator, device, buf.value(), data, size);
    if (!uploadRes.ok()) return uploadRes.error();

    return buf;
}

Result<Buffer> uploadVertexBuffer(
    const Allocator& allocator, const Device& device,
    const void* data, VkDeviceSize size) {
    auto buf = BufferBuilder(allocator).vertexBuffer().size(size).build();
    if (!buf.ok()) return buf.error();
    auto r = uploadToBuffer(allocator, device, buf.value(), data, size);
    if (!r.ok()) return r.error();
    return buf;
}

Result<Buffer> uploadIndexBuffer(
    const Allocator& allocator, const Device& device,
    const void* data, VkDeviceSize size) {
    auto buf = BufferBuilder(allocator).indexBuffer().size(size).build();
    if (!buf.ok()) return buf.error();
    auto r = uploadToBuffer(allocator, device, buf.value(), data, size);
    if (!r.ok()) return r.error();
    return buf;
}

Result<Buffer> uploadStorageBuffer(
    const Allocator& allocator, const Device& device,
    const void* data, VkDeviceSize size) {
    auto buf = BufferBuilder(allocator).storageBuffer().size(size).build();
    if (!buf.ok()) return buf.error();
    auto r = uploadToBuffer(allocator, device, buf.value(), data, size);
    if (!r.ok()) return r.error();
    return buf;
}

Result<Buffer> uploadIndirectBuffer(
    const Allocator& allocator, const Device& device,
    const void* data, VkDeviceSize size) {
    auto buf = BufferBuilder(allocator).indirectBuffer().size(size).build();
    if (!buf.ok()) return buf.error();
    auto r = uploadToBuffer(allocator, device, buf.value(), data, size);
    if (!r.ok()) return r.error();
    return buf;
}

} // namespace vksdl

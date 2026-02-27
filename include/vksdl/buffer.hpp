#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/vma_fwd.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>

namespace vksdl {

class Allocator;
class Device;

// Thread safety: immutable after construction.
class Buffer {
  public:
    ~Buffer();
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] VkBuffer native() const {
        return buffer_;
    }
    [[nodiscard]] VkBuffer vkBuffer() const {
        return native();
    }
    [[nodiscard]] VkDeviceSize size() const {
        return size_;
    }
    [[nodiscard]] void* mappedData() const {
        return mapped_;
    }

    // Returns the device address of this buffer. Only valid when the buffer was
    // created with SHADER_DEVICE_ADDRESS_BIT (all vertex/index/scratch/AS/SBT
    // convenience methods include it).
    [[nodiscard]] VkDeviceAddress deviceAddress() const;

  private:
    friend class BufferBuilder;
    friend Result<void> uploadToBuffer(const Allocator&, const Device&, const Buffer&, const void*,
                                       VkDeviceSize);
    Buffer() = default;

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
};

class BufferBuilder {
  public:
    explicit BufferBuilder(const Allocator& allocator);

    BufferBuilder& size(VkDeviceSize bytes);

    // Convenience methods -- set usage + VMA flags for common patterns.
    BufferBuilder& vertexBuffer();  // VERTEX_BUFFER | TRANSFER_DST | SHADER_DEVICE_ADDRESS
    BufferBuilder& indexBuffer();   // INDEX_BUFFER  | TRANSFER_DST | SHADER_DEVICE_ADDRESS
    BufferBuilder& uniformBuffer(); // UNIFORM_BUFFER, host-mapped
    BufferBuilder& storageBuffer(); // STORAGE_BUFFER | TRANSFER_DST, device-local
    BufferBuilder& stagingBuffer(); // TRANSFER_SRC, host-mapped

    // GPU-driven rendering
    BufferBuilder&
    indirectBuffer(); // INDIRECT_BUFFER | STORAGE_BUFFER | TRANSFER_DST | SHADER_DEVICE_ADDRESS

    // RT convenience methods
    BufferBuilder& scratchBuffer();                // STORAGE_BUFFER | SHADER_DEVICE_ADDRESS
    BufferBuilder& accelerationStructureStorage(); // AS_STORAGE | SHADER_DEVICE_ADDRESS
    BufferBuilder& shaderBindingTable(); // SBT | SHADER_DEVICE_ADDRESS | TRANSFER_DST, mapped

    // Escape hatches
    BufferBuilder& usage(VkBufferUsageFlags flags);
    BufferBuilder& deviceAddressable(); // adds SHADER_DEVICE_ADDRESS_BIT
    BufferBuilder&
    accelerationStructureInput(); // adds AS_BUILD_INPUT_READ_ONLY_BIT (requires VK_KHR_acceleration_structure)
    BufferBuilder& mapped();

    // Memory priority hint [0, 1]. Higher values are evicted last under pressure.
    // Effective only when the device reports hasMemoryPriority() == true.
    // Default 0.5 (mid-priority, VMA default).
    BufferBuilder& memoryPriority(float p);

    [[nodiscard]] Result<Buffer> build();

  private:
    VmaAllocator allocator_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkBufferUsageFlags usage_ = 0;
    float priority_ = 0.5f;
    bool mapped_ = false;
};

// Staged upload: creates a temporary staging buffer + command pool, copies data
// to the destination buffer via the GPU, waits for completion, then cleans up.
// Blocking -- suitable for init-time uploads only.
[[nodiscard]] Result<void> uploadToBuffer(const Allocator& allocator, const Device& device,
                                          const Buffer& dst, const void* data, VkDeviceSize size);

// One-liner: creates a device-local buffer and uploads data in a single call.
// Combines BufferBuilder + uploadToBuffer. Blocking -- init-time only.
// Usage flag selects the buffer type (vertex, index, storage).
[[nodiscard]] Result<Buffer> uploadBuffer(const Allocator& allocator, const Device& device,
                                          VkBufferUsageFlags usage, const void* data,
                                          VkDeviceSize size);

// Named convenience wrappers around uploadBuffer. Each uses the corresponding
// BufferBuilder convenience method, so RT-required flags (SHADER_DEVICE_ADDRESS,
// AS_BUILD_INPUT_READ_ONLY) are included automatically for vertex/index buffers.
[[nodiscard]] Result<Buffer> uploadVertexBuffer(const Allocator& allocator, const Device& device,
                                                const void* data, VkDeviceSize size);

[[nodiscard]] Result<Buffer> uploadIndexBuffer(const Allocator& allocator, const Device& device,
                                               const void* data, VkDeviceSize size);

[[nodiscard]] Result<Buffer> uploadStorageBuffer(const Allocator& allocator, const Device& device,
                                                 const void* data, VkDeviceSize size);

[[nodiscard]] Result<Buffer> uploadIndirectBuffer(const Allocator& allocator, const Device& device,
                                                  const void* data, VkDeviceSize size);

} // namespace vksdl

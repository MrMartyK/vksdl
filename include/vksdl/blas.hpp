#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Allocator;
class Buffer;
class Device;
class Mesh;

struct BlasTriangleGeometry {
    VkDeviceAddress vertexBufferAddress = 0;
    VkDeviceAddress indexBufferAddress = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t vertexStride = 0;
    VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    bool opaque = true;

    // Convenience: fills in addresses, counts, and stride from Buffers.
    [[nodiscard]] static BlasTriangleGeometry
    fromBuffers(const Buffer& vertexBuffer, const Buffer& indexBuffer, std::uint32_t vertexCount,
                std::uint32_t indexCount, std::uint32_t vertexStride);
};

struct BlasBuildSizes {
    VkDeviceSize accelerationStructureSize;
    VkDeviceSize buildScratchSize;
    VkDeviceSize updateScratchSize;
};

// Thread safety: immutable after construction.
class Blas {
  public:
    ~Blas();
    Blas(Blas&&) noexcept;
    Blas& operator=(Blas&&) noexcept;
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    [[nodiscard]] VkAccelerationStructureKHR native() const {
        return as_;
    }
    [[nodiscard]] VkAccelerationStructureKHR vkAccelerationStructure() const {
        return native();
    }
    [[nodiscard]] VkDeviceAddress deviceAddress() const {
        return address_;
    }

  private:
    friend class BlasBuilder;
    friend Result<void> compactBlas(const Device&, const Allocator&, Blas&);
    Blas() = default;

    VkDevice device_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR as_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;

    // Backing buffer (raw VMA handles to avoid pulling in Buffer header).
    // Forward-declared VMA types are already in buffer.hpp pattern.
    struct BackingBuffer;
    BackingBuffer* backing_ = nullptr;
};

class BlasBuilder {
  public:
    BlasBuilder(const Device& device, const Allocator& allocator);

    BlasBuilder& addTriangles(const BlasTriangleGeometry& geometry);
    BlasBuilder& addMesh(const Mesh& mesh);

    BlasBuilder& preferFastTrace();
    BlasBuilder& preferFastBuild();
    BlasBuilder& allowCompaction();

    [[nodiscard]] Result<BlasBuildSizes> sizes() const;

    // Sync build: allocates scratch internally, one-shot command pool, waits.
    // If allowCompaction() was called, automatically compacts after the build.
    // This causes two full pipeline stalls (one for build, one for compaction copy).
    [[nodiscard]] Result<Blas> build();

    // Async build: records into user's command buffer, user provides scratch.
    // User must fence before using the Blas. Does not compact.
    [[nodiscard]] Result<Blas> cmdBuild(VkCommandBuffer cmd, const Buffer& scratch);

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamily_ = 0;
    VkDeviceSize scratchAlignment_ = 0;
    const Allocator* allocator_ = nullptr;
    std::vector<BlasTriangleGeometry> geometries_;
    VkBuildAccelerationStructureFlagsKHR buildFlags_ =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bool allowCompaction_ = false;
};

// Compact a BLAS that was built with allowCompaction().
// Sync: queries compacted size, allocates smaller buffer, copies, waits, swaps.
// Two pipeline stalls (property query + compaction copy).
[[nodiscard]] Result<void> compactBlas(const Device& device, const Allocator& allocator,
                                       Blas& blas);

} // namespace vksdl

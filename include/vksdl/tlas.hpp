#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Allocator;
class Blas;
class Buffer;
class Device;

struct TlasInstance {
    VkDeviceAddress blasAddress = 0;
    float           transform[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};
    std::uint32_t   customIndex = 0;
    std::uint8_t    mask        = 0xFF;
    std::uint32_t   sbtOffset   = 0;
    VkGeometryInstanceFlagsKHR flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
};

class Tlas {
public:
    ~Tlas();
    Tlas(Tlas&&) noexcept;
    Tlas& operator=(Tlas&&) noexcept;
    Tlas(const Tlas&) = delete;
    Tlas& operator=(const Tlas&) = delete;

    [[nodiscard]] VkAccelerationStructureKHR native()                  const { return as_; }
    [[nodiscard]] VkAccelerationStructureKHR vkAccelerationStructure() const { return native(); }
    [[nodiscard]] bool supportsUpdate() const { return allowUpdate_; }
    [[nodiscard]] std::uint32_t maxInstanceCount() const { return maxInstanceCount_; }
    [[nodiscard]] VkDeviceSize updateScratchSize() const { return updateScratchSize_; }

    // Record an in-place TLAS update for dynamic per-frame transforms.
    // Requires ALLOW_UPDATE at build time and instanceCount <= maxInstanceCount().
    [[nodiscard]] Result<void> cmdUpdate(VkCommandBuffer cmd,
                                         const Buffer& scratch,
                                         const Buffer& instanceBuffer,
                                         std::uint32_t instanceCount);

private:
    friend class TlasBuilder;
    Tlas() = default;

    VkDevice                     device_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR   as_     = VK_NULL_HANDLE;
    bool                         allowUpdate_ = false;
    std::uint32_t                maxInstanceCount_ = 0;
    VkDeviceSize                 scratchAlignment_ = 0;
    VkBuildAccelerationStructureFlagsKHR buildFlags_ =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    VkDeviceSize                 updateScratchSize_ = 0;

    // Backing buffer + instance buffer (raw VMA handles).
    struct BackingBuffer;
    BackingBuffer* backing_  = nullptr;
    BackingBuffer* instance_ = nullptr;
};

class TlasBuilder {
public:
    TlasBuilder(const Device& device, const Allocator& allocator);

    TlasBuilder& addInstance(const Blas& blas, const float transform[3][4],
                             std::uint32_t customIndex = 0,
                             std::uint8_t mask = 0xFF,
                             std::uint32_t sbtOffset = 0);
    TlasBuilder& addInstance(const TlasInstance& instance);

    TlasBuilder& preferFastTrace();
    TlasBuilder& preferFastBuild();
    TlasBuilder& allowUpdate();

    // Sync build: allocates instance buffer + scratch internally, one-shot
    // command pool, waits. One pipeline stall.
    [[nodiscard]] Result<Tlas> build();

    // Async build: records into user's command buffer, user provides scratch.
    // Instance data must already be uploaded to instanceBuffer.
    // User must fence before using the Tlas.
    [[nodiscard]] Result<Tlas> cmdBuild(VkCommandBuffer cmd,
                                        const Buffer& scratch,
                                        const Buffer& instanceBuffer,
                                        std::uint32_t instanceCount);

private:
    VkDevice                              device_       = VK_NULL_HANDLE;
    VkPhysicalDevice                      physDevice_   = VK_NULL_HANDLE;
    VkQueue                               queue_        = VK_NULL_HANDLE;
    std::uint32_t                         queueFamily_  = 0;
    VkDeviceSize                          scratchAlignment_ = 0;
    const Allocator*                      allocator_    = nullptr;
    std::vector<TlasInstance>             instances_;
    VkBuildAccelerationStructureFlagsKHR  buildFlags_   =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bool                                  allowUpdate_  = false;
};

} // namespace vksdl

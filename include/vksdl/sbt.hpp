#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vksdl {

class Allocator;
class Buffer;
class Device;
class Pipeline;

// Thread safety: immutable after construction.
class ShaderBindingTable {
public:
    [[nodiscard]] static Result<ShaderBindingTable> create(
        const Device& device,
        const Pipeline& rtPipeline,
        const Allocator& allocator,
        std::uint32_t missCount,
        std::uint32_t hitCount,
        std::uint32_t callableCount = 0);

    ~ShaderBindingTable();
    ShaderBindingTable(ShaderBindingTable&&) noexcept;
    ShaderBindingTable& operator=(ShaderBindingTable&&) noexcept;
    ShaderBindingTable(const ShaderBindingTable&) = delete;
    ShaderBindingTable& operator=(const ShaderBindingTable&) = delete;

    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& raygenRegion()   const { return raygen_; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& missRegion()     const { return miss_; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& hitRegion()      const { return hit_; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& callableRegion() const { return callable_; }

    void traceRays(VkCommandBuffer cmd,
                   std::uint32_t width, std::uint32_t height,
                   std::uint32_t depth = 1) const;

private:
    ShaderBindingTable() = default;

    struct BackingBuffer;
    BackingBuffer*          backing_      = nullptr;
    VkDevice                device_       = VK_NULL_HANDLE;
    PFN_vkCmdTraceRaysKHR   pfnTraceRays_ = nullptr;

    VkStridedDeviceAddressRegionKHR raygen_   = {};
    VkStridedDeviceAddressRegionKHR miss_     = {};
    VkStridedDeviceAddressRegionKHR hit_      = {};
    VkStridedDeviceAddressRegionKHR callable_ = {};
};

} // namespace vksdl

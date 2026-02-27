#include <vksdl/sbt.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>
#include <vksdl/pipeline.hpp>
#include <vksdl/util.hpp>

#include "rt_functions.hpp"

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
#include <vector>

namespace vksdl {

struct ShaderBindingTable::BackingBuffer {
    VmaAllocator  allocator  = nullptr;
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

ShaderBindingTable::~ShaderBindingTable() {
    if (backing_) {
        if (backing_->buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(backing_->allocator, backing_->buffer,
                             backing_->allocation);
        }
        delete backing_;
    }
}

ShaderBindingTable::ShaderBindingTable(ShaderBindingTable&& o) noexcept
    : backing_(o.backing_), device_(o.device_), pfnTraceRays_(o.pfnTraceRays_),
      raygen_(o.raygen_), miss_(o.miss_), hit_(o.hit_), callable_(o.callable_) {
    o.backing_      = nullptr;
    o.device_       = VK_NULL_HANDLE;
    o.pfnTraceRays_ = nullptr;
    o.raygen_       = {};
    o.miss_         = {};
    o.hit_          = {};
    o.callable_     = {};
}

ShaderBindingTable& ShaderBindingTable::operator=(
    ShaderBindingTable&& o) noexcept {
    if (this != &o) {
        if (backing_) {
            if (backing_->buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(backing_->allocator, backing_->buffer,
                                 backing_->allocation);
            }
            delete backing_;
        }

        backing_      = o.backing_;
        device_       = o.device_;
        pfnTraceRays_ = o.pfnTraceRays_;
        raygen_       = o.raygen_;
        miss_         = o.miss_;
        hit_          = o.hit_;
        callable_     = o.callable_;

        o.backing_      = nullptr;
        o.device_       = VK_NULL_HANDLE;
        o.pfnTraceRays_ = nullptr;
        o.raygen_       = {};
        o.miss_         = {};
        o.hit_          = {};
        o.callable_     = {};
    }
    return *this;
}

Result<ShaderBindingTable> ShaderBindingTable::create(
    const Device& device,
    const Pipeline& rtPipeline,
    const Allocator& allocator,
    std::uint32_t missCount,
    std::uint32_t hitCount,
    std::uint32_t callableCount) {

    auto fn = detail::loadRtFunctions(device.vkDevice());
    if (!fn.getShaderGroupHandles) {
        return Error{"create SBT", 0,
                     "vkGetRayTracingShaderGroupHandlesKHR not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    std::uint32_t handleSize      = device.shaderGroupHandleSize();
    std::uint32_t baseAlignment   = device.shaderGroupBaseAlignment();
    std::uint32_t handleAlignment = device.shaderGroupHandleAlignment();

    if (handleSize == 0 || baseAlignment == 0) {
        return Error{"create SBT", 0,
                     "RT properties not available (handleSize or baseAlignment is 0)"};
    }

    auto handleSizeAligned = static_cast<VkDeviceSize>(
        alignUp(static_cast<VkDeviceSize>(handleSize),
                static_cast<VkDeviceSize>(handleAlignment)));

    std::uint32_t totalGroups = 1 + missCount + hitCount + callableCount;

    std::vector<std::uint8_t> handleData(
        static_cast<std::size_t>(totalGroups) * handleSize);
    VkResult vr = fn.getShaderGroupHandles(
        device.vkDevice(), rtPipeline.vkPipeline(),
        0, totalGroups,
        handleData.size(), handleData.data());
    if (vr != VK_SUCCESS) {
        return Error{"create SBT", static_cast<std::int32_t>(vr),
                     "vkGetRayTracingShaderGroupHandlesKHR failed"};
    }

    // Each region's start must be aligned to baseAlignment.
    // Raygen: stride == size (Vulkan spec requirement: exactly one raygen entry per region).
    VkDeviceSize raygenStride = alignUp(handleSizeAligned,
                                        static_cast<VkDeviceSize>(baseAlignment));
    VkDeviceSize raygenSize   = raygenStride;

    VkDeviceSize missStride = missCount > 0 ? handleSizeAligned : 0;
    VkDeviceSize missSize   = missCount > 0
        ? alignUp(handleSizeAligned * missCount,
                  static_cast<VkDeviceSize>(baseAlignment))
        : 0;

    VkDeviceSize hitStride = hitCount > 0 ? handleSizeAligned : 0;
    VkDeviceSize hitSize   = hitCount > 0
        ? alignUp(handleSizeAligned * hitCount,
                  static_cast<VkDeviceSize>(baseAlignment))
        : 0;

    VkDeviceSize callableStride = handleSizeAligned;
    VkDeviceSize callableSize   = callableCount > 0
        ? alignUp(callableStride * callableCount,
                  static_cast<VkDeviceSize>(baseAlignment))
        : 0;

    VkDeviceSize totalSize = raygenSize + missSize + hitSize + callableSize;

    VmaAllocator vma = allocator.vmaAllocator();

    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = totalSize;
    bufCI.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

    auto* backing = new ShaderBindingTable::BackingBuffer{};
    backing->allocator = vma;

    VmaAllocationInfo allocInfo{};
    vr = vmaCreateBuffer(vma, &bufCI, &allocCI,
                          &backing->buffer, &backing->allocation, &allocInfo);
    if (vr != VK_SUCCESS) {
        delete backing;
        return Error{"create SBT", static_cast<std::int32_t>(vr),
                     "failed to create SBT buffer"};
    }

    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = backing->buffer;
    VkDeviceAddress baseAddr = vkGetBufferDeviceAddress(device.vkDevice(), &addrInfo);

    auto* dst = static_cast<std::uint8_t*>(allocInfo.pMappedData);
    std::memset(dst, 0, static_cast<std::size_t>(totalSize));

    std::uint32_t handleIdx = 0;
    VkDeviceSize  offset    = 0;

    std::memcpy(dst + offset,
                handleData.data() + static_cast<std::size_t>(handleIdx) * handleSize,
                handleSize);
    handleIdx++;
    offset += raygenSize;

    for (std::uint32_t i = 0; i < missCount; ++i) {
        std::memcpy(dst + offset + i * missStride,
                    handleData.data() + static_cast<std::size_t>(handleIdx) * handleSize,
                    handleSize);
        handleIdx++;
    }
    offset += missSize;

    for (std::uint32_t i = 0; i < hitCount; ++i) {
        std::memcpy(dst + offset + i * hitStride,
                    handleData.data() + static_cast<std::size_t>(handleIdx) * handleSize,
                    handleSize);
        handleIdx++;
    }
    offset += hitSize;

    for (std::uint32_t i = 0; i < callableCount; ++i) {
        std::memcpy(dst + offset + i * callableStride,
                    handleData.data() + static_cast<std::size_t>(handleIdx) * handleSize,
                    handleSize);
        handleIdx++;
    }

    ShaderBindingTable sbt;
    sbt.device_       = device.vkDevice();
    sbt.backing_      = backing;
    sbt.pfnTraceRays_ = device.traceRaysFn();

    VkDeviceSize regionOffset = 0;

    sbt.raygen_.deviceAddress = baseAddr + regionOffset;
    sbt.raygen_.stride        = raygenStride;
    sbt.raygen_.size          = raygenSize;
    regionOffset += raygenSize;

    sbt.miss_.deviceAddress = missCount > 0 ? baseAddr + regionOffset : 0;
    sbt.miss_.stride        = missStride;
    sbt.miss_.size          = missSize;
    regionOffset += missSize;

    sbt.hit_.deviceAddress = hitCount > 0 ? baseAddr + regionOffset : 0;
    sbt.hit_.stride        = hitStride;
    sbt.hit_.size          = hitSize;
    regionOffset += hitSize;

    sbt.callable_.deviceAddress = callableCount > 0
        ? baseAddr + regionOffset
        : 0;
    sbt.callable_.stride = callableCount > 0 ? callableStride : 0;
    sbt.callable_.size   = callableSize;

    return sbt;
}

void ShaderBindingTable::traceRays(VkCommandBuffer cmd,
                                   std::uint32_t width, std::uint32_t height,
                                   std::uint32_t depth) const {
    pfnTraceRays_(cmd, &raygen_, &miss_, &hit_, &callable_,
                  width, height, depth);
}

} // namespace vksdl

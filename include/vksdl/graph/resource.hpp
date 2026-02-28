#pragma once

#include <vksdl/graph/resource_state.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace vksdl::graph {

// Opaque handle to a graph resource. Index into the graph's resource table.
struct ResourceHandle {
    std::uint32_t index = UINT32_MAX;

    [[nodiscard]] bool valid() const {
        return index != UINT32_MAX;
    }
    [[nodiscard]] bool operator==(const ResourceHandle&) const = default;
};

enum class ResourceTag : std::uint8_t {
    External,  // User-owned vksdl::Image or vksdl::Buffer imported into the graph.
    Transient, // Graph-allocated, lifetime bounded by first and last use.
};

enum class ResourceKind : std::uint8_t {
    Image,
    Buffer,
};

// Description of a transient image to be allocated by the graph.
struct ImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    std::uint32_t mipLevels = 1;
    std::uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    [[nodiscard]] bool operator==(const ImageDesc&) const = default;
};

// Description of a transient buffer to be allocated by the graph.
struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;

    [[nodiscard]] bool operator==(const BufferDesc&) const = default;
};

// Internal resource entry in the graph's resource table.
struct ResourceEntry {
    ResourceTag tag = ResourceTag::External;
    ResourceKind kind = ResourceKind::Image;

    std::string name;

    VkImage vkImage = VK_NULL_HANDLE;
    VkImageView vkImageView = VK_NULL_HANDLE;
    VkBuffer vkBuffer = VK_NULL_HANDLE;
    VkDeviceSize bufferSize = 0;

    ImageDesc imageDesc;
    BufferDesc bufferDesc;

    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    ResourceState initialState;

    // Lifetime span (set during compile, indices into sorted pass order).
    std::uint32_t firstPass = UINT32_MAX;
    std::uint32_t lastPass = UINT32_MAX;
};

} // namespace vksdl::graph

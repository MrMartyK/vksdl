#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/vma_fwd.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vksdl {

class Allocator;

class Image {
public:
    ~Image();
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    [[nodiscard]] VkImage                native()      const { return image_; }
    [[nodiscard]] VkImage                vkImage()     const { return native(); }
    [[nodiscard]] VkImageView            vkImageView() const { return view_; }
    [[nodiscard]] VkFormat               format()      const { return format_; }
    [[nodiscard]] VkExtent2D             extent()      const { return extent_; }
    [[nodiscard]] std::uint32_t          mipLevels()   const { return mipLevels_; }
    [[nodiscard]] VkSampleCountFlagBits  samples()     const { return samples_; }

private:
    friend class ImageBuilder;
    Image() = default;

    VmaAllocator  allocator_  = nullptr;
    VkDevice      device_     = VK_NULL_HANDLE;
    VkImage       image_      = VK_NULL_HANDLE;
    VkImageView   view_       = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VkFormat              format_     = VK_FORMAT_UNDEFINED;
    VkExtent2D            extent_     = {0, 0};
    std::uint32_t         mipLevels_  = 1;
    VkSampleCountFlagBits samples_    = VK_SAMPLE_COUNT_1_BIT;
};

class ImageBuilder {
public:
    explicit ImageBuilder(const Allocator& allocator);

    ImageBuilder& size(std::uint32_t width, std::uint32_t height);
    ImageBuilder& format(VkFormat fmt);

    // Convenience methods -- set usage + aspect flags for common patterns.
    ImageBuilder& colorAttachment();       // COLOR_ATTACHMENT | SAMPLED
    ImageBuilder& depthAttachment();       // DEPTH_STENCIL_ATTACHMENT, defaults D32_SFLOAT
    ImageBuilder& sampled();               // SAMPLED | TRANSFER_DST
    ImageBuilder& storage();               // STORAGE | SAMPLED
    ImageBuilder& mipmapped();             // adds TRANSFER_SRC, auto-calculates mip levels in build()
    ImageBuilder& msaaColorAttachment();   // COLOR_ATTACHMENT only (no SAMPLED -- MSAA targets are transient)

    // Escape hatches
    ImageBuilder& usage(VkImageUsageFlags flags);
    ImageBuilder& addUsage(VkImageUsageFlags flags);
    ImageBuilder& aspect(VkImageAspectFlags flags);
    ImageBuilder& samples(VkSampleCountFlagBits s);
    ImageBuilder& mipLevels(std::uint32_t levels);

    [[nodiscard]] Result<Image> build();

private:
    VmaAllocator          allocator_ = nullptr;
    VkDevice              device_    = VK_NULL_HANDLE;
    std::uint32_t         width_     = 0;
    std::uint32_t         height_    = 0;
    VkFormat              format_    = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags     usage_     = 0;
    VkImageAspectFlags    aspect_    = VK_IMAGE_ASPECT_COLOR_BIT;
    VkSampleCountFlagBits samples_   = VK_SAMPLE_COUNT_1_BIT;
    std::uint32_t         mipLevels_ = 1;
    bool                  mipmapped_ = false;
};

} // namespace vksdl

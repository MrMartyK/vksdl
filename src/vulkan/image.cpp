#include <vksdl/image.hpp>
#include <vksdl/allocator.hpp>

#include <algorithm>
#include <cmath>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

namespace vksdl {

Image::~Image() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
    }
}

Image::Image(Image&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_), image_(o.image_),
      view_(o.view_), allocation_(o.allocation_),
      format_(o.format_), extent_(o.extent_), mipLevels_(o.mipLevels_),
      samples_(o.samples_) {
    o.allocator_  = nullptr;
    o.device_     = VK_NULL_HANDLE;
    o.image_      = VK_NULL_HANDLE;
    o.view_       = VK_NULL_HANDLE;
    o.allocation_ = nullptr;
}

Image& Image::operator=(Image&& o) noexcept {
    if (this != &o) {
        if (view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, view_, nullptr);
        }
        if (image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_, image_, allocation_);
        }
        allocator_  = o.allocator_;
        device_     = o.device_;
        image_      = o.image_;
        view_       = o.view_;
        allocation_ = o.allocation_;
        format_     = o.format_;
        extent_     = o.extent_;
        mipLevels_  = o.mipLevels_;
        samples_    = o.samples_;
        o.allocator_  = nullptr;
        o.device_     = VK_NULL_HANDLE;
        o.image_      = VK_NULL_HANDLE;
        o.view_       = VK_NULL_HANDLE;
        o.allocation_ = nullptr;
    }
    return *this;
}

ImageBuilder::ImageBuilder(const Allocator& allocator)
    : allocator_(allocator.vmaAllocator()), device_(allocator.vkDevice()) {}

ImageBuilder& ImageBuilder::size(std::uint32_t width, std::uint32_t height) {
    width_  = width;
    height_ = height;
    return *this;
}

ImageBuilder& ImageBuilder::format(VkFormat fmt) {
    format_ = fmt;
    return *this;
}

ImageBuilder& ImageBuilder::colorAttachment() {
    usage_  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    return *this;
}

ImageBuilder& ImageBuilder::depthAttachment() {
    usage_  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    aspect_ = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (format_ == VK_FORMAT_UNDEFINED) {
        format_ = VK_FORMAT_D32_SFLOAT;
    }
    return *this;
}

ImageBuilder& ImageBuilder::sampled() {
    usage_  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    return *this;
}

ImageBuilder& ImageBuilder::storage() {
    usage_  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    return *this;
}

ImageBuilder& ImageBuilder::usage(VkImageUsageFlags flags) {
    usage_ = flags;
    return *this;
}

ImageBuilder& ImageBuilder::addUsage(VkImageUsageFlags flags) {
    usage_ |= flags;
    return *this;
}

ImageBuilder& ImageBuilder::aspect(VkImageAspectFlags flags) {
    aspect_ = flags;
    return *this;
}

ImageBuilder& ImageBuilder::samples(VkSampleCountFlagBits s) {
    samples_ = s;
    return *this;
}

ImageBuilder& ImageBuilder::mipLevels(std::uint32_t levels) {
    mipLevels_ = levels;
    return *this;
}

ImageBuilder& ImageBuilder::mipmapped() {
    mipmapped_ = true;
    return *this;
}

ImageBuilder& ImageBuilder::msaaColorAttachment() {
    usage_  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    return *this;
}

Result<Image> ImageBuilder::build() {
    if (width_ == 0 || height_ == 0) {
        return Error{"create image", 0,
                     "image size is 0 -- call size(width, height)"};
    }
    if (format_ == VK_FORMAT_UNDEFINED) {
        return Error{"create image", 0,
                     "no format set -- call format(VkFormat) or a convenience method"};
    }
    if (usage_ == 0) {
        return Error{"create image", 0,
                     "no usage flags -- call colorAttachment(), depthAttachment(), etc."};
    }

    if (mipmapped_) {
        usage_ |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        mipLevels_ = static_cast<std::uint32_t>(
            std::floor(std::log2(std::max(width_, height_)))) + 1;
    }

    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = format_;
    imageCI.extent        = {width_, height_, 1};
    imageCI.mipLevels     = mipLevels_;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = samples_;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = usage_;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;

    Image img;
    img.allocator_ = allocator_;
    img.device_    = device_;
    img.format_    = format_;
    img.extent_    = {width_, height_};
    img.mipLevels_ = mipLevels_;
    img.samples_   = samples_;

    VkResult vr = vmaCreateImage(allocator_, &imageCI, &allocCI,
                                  &img.image_, &img.allocation_, nullptr);
    if (vr != VK_SUCCESS) {
        return Error{"create image", static_cast<std::int32_t>(vr),
                     "vmaCreateImage failed"};
    }

    VkImageViewCreateInfo viewCI{};
    viewCI.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image    = img.image_;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format   = format_;
    viewCI.subresourceRange.aspectMask     = aspect_;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = mipLevels_;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;

    vr = vkCreateImageView(device_, &viewCI, nullptr, &img.view_);
    if (vr != VK_SUCCESS) {
        vmaDestroyImage(allocator_, img.image_, img.allocation_);
        img.image_ = VK_NULL_HANDLE;
        return Error{"create image view", static_cast<std::int32_t>(vr),
                     "vkCreateImageView failed for VMA-allocated image"};
    }

    return img;
}

} // namespace vksdl

#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

namespace vksdl {

class Allocator;
class Device;
class Image;

// Pixel data loaded from an image file. Owns the pixel memory (freed by destructor).
// Always RGBA, 4 channels, one byte per channel.
struct ImageData {
    ~ImageData();
    ImageData(ImageData&&) noexcept;
    ImageData& operator=(ImageData&&) noexcept;
    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;

    unsigned char* pixels   = nullptr;
    std::uint32_t  width    = 0;
    std::uint32_t  height   = 0;
    std::uint32_t  channels = 0;

    [[nodiscard]] VkDeviceSize sizeBytes() const {
        return static_cast<VkDeviceSize>(width) * height * channels;
    }

private:
    friend Result<ImageData> loadImage(const std::filesystem::path&);
    ImageData() = default;
};

// Load an image file (PNG, JPG, BMP, etc.) via stb_image.
// Always loads as RGBA (4 channels) to match VK_FORMAT_R8G8B8A8_SRGB.
[[nodiscard]] Result<ImageData> loadImage(const std::filesystem::path& path);

// Staged upload from CPU pixels to a GPU Image.
// Creates a staging buffer, records layout transitions, copies data, waits for completion.
// Blocking -- suitable for init-time uploads only.
// When dst.mipLevels() == 1: transitions to SHADER_READ_ONLY_OPTIMAL (ready to sample).
// When dst.mipLevels() > 1:  leaves level 0 in TRANSFER_DST_OPTIMAL -- call
//   generateMipmaps() next to fill remaining levels and transition to SHADER_READ_ONLY.
[[nodiscard]] Result<void> uploadToImage(
    const Allocator& allocator,
    const Device& device,
    const Image& dst,
    const void* pixels,
    VkDeviceSize size);

// Compute full mip chain count for given dimensions.
// Returns floor(log2(max(width, height))) + 1.
[[nodiscard]] std::uint32_t calculateMipLevels(std::uint32_t width, std::uint32_t height);

// Recording-time: records a blit chain that fills mip levels 1..N-1 from level 0.
// Precondition: level 0 in TRANSFER_DST_OPTIMAL (has data from uploadToImage),
//               levels 1..N-1 in UNDEFINED.
// Postcondition: all levels in SHADER_READ_ONLY_OPTIMAL.
// Format must support VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT (true
// for R8G8B8A8_SRGB/UNORM and all common color formats; validation layers will
// report unsupported formats).
void generateMipmaps(VkCommandBuffer cmd, const Image& image);

// Raw-handle overload (escape hatch). Same preconditions and postconditions.
void generateMipmaps(VkCommandBuffer cmd, VkImage image, VkFormat format,
                     std::uint32_t width, std::uint32_t height,
                     std::uint32_t mipLevels);

} // namespace vksdl

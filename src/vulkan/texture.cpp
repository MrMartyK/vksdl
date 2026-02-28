#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>
#include <vksdl/image.hpp>
#include <vksdl/texture.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4189) // local variable initialized but not referenced
#pragma warning(disable : 4244) // conversion, possible loss of data
#pragma warning(disable : 4245) // signed/unsigned mismatch in initialization
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if VKSDL_HAS_LOADERS
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4244) // conversion, possible loss of data
#pragma warning(disable : 4245) // signed/unsigned mismatch in initialization
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif // VKSDL_HAS_LOADERS

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace vksdl {

#if VKSDL_HAS_LOADERS

ImageData::~ImageData() {
    if (pixels) {
        stbi_image_free(pixels);
    }
}

ImageData::ImageData(ImageData&& o) noexcept
    : pixels(o.pixels), width(o.width), height(o.height), channels(o.channels) {
    o.pixels = nullptr;
    o.width = 0;
    o.height = 0;
    o.channels = 0;
}

ImageData& ImageData::operator=(ImageData&& o) noexcept {
    if (this != &o) {
        if (pixels) {
            stbi_image_free(pixels);
        }
        pixels = o.pixels;
        width = o.width;
        height = o.height;
        channels = o.channels;
        o.pixels = nullptr;
        o.width = 0;
        o.height = 0;
        o.channels = 0;
    }
    return *this;
}

Result<ImageData> loadImage(const std::filesystem::path& path) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load(path.string().c_str(), &w, &h, &ch, 4);
    if (!pixels) {
        const char* reason = stbi_failure_reason();
        return Error{"load image", 0,
                     "stbi_load failed: " + std::string(reason ? reason : "unknown error") +
                         " -- path: " + path.string()};
    }

    ImageData data;
    data.pixels = pixels;
    data.width = static_cast<std::uint32_t>(w);
    data.height = static_cast<std::uint32_t>(h);
    data.channels = 4;

    return data;
}

#endif // VKSDL_HAS_LOADERS

Result<void> uploadToImage(const Allocator& allocator, const Device& device, const Image& dst,
                           const void* pixels, VkDeviceSize size) {

    VkBufferCreateInfo stagingCI{};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI{};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingInfo{};

    VkResult vr = vmaCreateBuffer(allocator.vmaAllocator(), &stagingCI, &stagingAllocCI,
                                  &stagingBuf, &stagingAlloc, &stagingInfo);
    if (vr != VK_SUCCESS) {
        return Error{"upload to image", static_cast<std::int32_t>(vr),
                     "failed to create staging buffer"};
    }

    std::memcpy(stagingInfo.pMappedData, pixels, static_cast<std::size_t>(size));

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.queueFamilies().graphics;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(device.vkDevice(), &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to image", static_cast<std::int32_t>(vr),
                     "failed to create command pool for transfer"};
    }

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool = cmdPool;
    cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(device.vkDevice(), &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to image", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer for transfer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition: UNDEFINED -> TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier2 toTransferDst{};
    toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toTransferDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    toTransferDst.srcAccessMask = VK_ACCESS_2_NONE;
    toTransferDst.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toTransferDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDst.image = dst.vkImage();
    toTransferDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo1{};
    depInfo1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo1.imageMemoryBarrierCount = 1;
    depInfo1.pImageMemoryBarriers = &toTransferDst;
    vkCmdPipelineBarrier2(cmd, &depInfo1);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {dst.extent().width, dst.extent().height, 1};

    vkCmdCopyBufferToImage(cmd, stagingBuf, dst.vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);

    // For single-mip images: transition to SHADER_READ_ONLY (ready to sample).
    // For mipmapped images: leave in TRANSFER_DST so generateMipmaps() can start.
    if (dst.mipLevels() <= 1) {
        VkImageMemoryBarrier2 toShaderRead{};
        toShaderRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toShaderRead.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toShaderRead.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toShaderRead.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toShaderRead.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toShaderRead.image = dst.vkImage();
        toShaderRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo depInfo2{};
        depInfo2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo2.imageMemoryBarrierCount = 1;
        depInfo2.pImageMemoryBarriers = &toShaderRead;
        vkCmdPipelineBarrier2(cmd, &depInfo2);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vr = vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);
        return Error{"upload to image", static_cast<std::int32_t>(vr), "vkQueueSubmit failed"};
    }
    // VKSDL_BLOCKING_WAIT: init-time texture upload waits for copy completion.
    vkQueueWaitIdle(device.graphicsQueue());

    vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
    vmaDestroyBuffer(allocator.vmaAllocator(), stagingBuf, stagingAlloc);

    return {};
}

std::uint32_t calculateMipLevels(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0)
        return 1;
    return static_cast<std::uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

void generateMipmaps(VkCommandBuffer cmd, VkImage image, VkFormat format, std::uint32_t width,
                     std::uint32_t height, std::uint32_t mipLevels) {
    (void) format; // reserved for future format validation

    if (mipLevels <= 1)
        return;

    auto mipWidth = static_cast<std::int32_t>(width);
    auto mipHeight = static_cast<std::int32_t>(height);

    for (std::uint32_t i = 1; i < mipLevels; ++i) {
        // Transition level i-1: TRANSFER_DST -> TRANSFER_SRC (source for blit)
        // Transition level i:   UNDEFINED -> TRANSFER_DST (destination for blit)
        VkImageMemoryBarrier2 barriers[2]{};

        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].image = image;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barriers[1].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].image = image;
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1};

        VkDependencyInfo dep1{};
        dep1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep1.imageMemoryBarrierCount = 2;
        dep1.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep1);

        // Blit from level i-1 to level i
        std::int32_t nextWidth = std::max(mipWidth / 2, 1);
        std::int32_t nextHeight = std::max(mipHeight / 2, 1);

        VkImageBlit2 blit{};
        blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextWidth, nextHeight, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.layerCount = 1;

        VkBlitImageInfo2 blitInfo{};
        blitInfo.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blitInfo.srcImage = image;
        blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blitInfo.dstImage = image;
        blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blitInfo.regionCount = 1;
        blitInfo.pRegions = &blit;
        blitInfo.filter = VK_FILTER_LINEAR;

        vkCmdBlitImage2(cmd, &blitInfo);

        // Transition level i-1: TRANSFER_SRC -> SHADER_READ_ONLY
        VkImageMemoryBarrier2 toRead{};
        toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toRead.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toRead.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        toRead.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toRead.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.image = image;
        toRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.imageMemoryBarrierCount = 1;
        dep2.pImageMemoryBarriers = &toRead;
        vkCmdPipelineBarrier2(cmd, &dep2);

        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    // Transition last level: TRANSFER_DST -> SHADER_READ_ONLY
    VkImageMemoryBarrier2 lastToRead{};
    lastToRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    lastToRead.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    lastToRead.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    lastToRead.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    lastToRead.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    lastToRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    lastToRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lastToRead.image = image;
    lastToRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 1};

    VkDependencyInfo dep3{};
    dep3.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep3.imageMemoryBarrierCount = 1;
    dep3.pImageMemoryBarriers = &lastToRead;
    vkCmdPipelineBarrier2(cmd, &dep3);
}

void generateMipmaps(VkCommandBuffer cmd, const Image& image) {
    generateMipmaps(cmd, image.vkImage(), image.format(), image.extent().width,
                    image.extent().height, image.mipLevels());
}

} // namespace vksdl

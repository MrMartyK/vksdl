#include <vksdl/barriers.hpp>
#include <vksdl/image.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>

namespace vksdl {

void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
#ifndef NDEBUG
    if (srcStage == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT ||
        dstStage == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) {
        std::fprintf(stderr,
            "[vksdl perf] transitionImage uses ALL_COMMANDS stage "
            "-- this creates a full pipeline bubble. "
            "Use a narrower stage mask for better performance.\n");
    }
#endif
    VkImageMemoryBarrier2 barrier{};
    barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask     = srcStage;
    barrier.srcAccessMask    = srcAccess;
    barrier.dstStageMask     = dstStage;
    barrier.dstAccessMask    = dstAccess;
    barrier.oldLayout        = oldLayout;
    barrier.newLayout        = newLayout;
    barrier.image            = image;
    // Derive aspect from target layout: depth/stencil layouts use depth aspect,
    // everything else uses color.
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    auto isDepthStencil = [](VkImageLayout l) {
        return l == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || l == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            || l == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
            || l == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
    };
    auto isDepthOnly = [](VkImageLayout l) {
        return l == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
            || l == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    };
    if (isDepthStencil(newLayout) || isDepthStencil(oldLayout)) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    } else if (isDepthOnly(newLayout) || isDepthOnly(oldLayout)) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS,
                                        0, VK_REMAINING_ARRAY_LAYERS};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void transitionToColorAttachment(VkCommandBuffer cmd, VkImage image) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

void transitionToPresent(VkCommandBuffer cmd, VkImage image) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

void transitionToComputeWrite(VkCommandBuffer cmd, VkImage image) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void transitionToTransferSrc(VkCommandBuffer cmd, VkImage image) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);
}

void transitionToTransferDst(VkCommandBuffer cmd, VkImage image) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

void transitionToDepthAttachment(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
    barrier.srcAccessMask    = 0;
    barrier.dstStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                             | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barrier.dstAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    barrier.image            = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierComputeToIndirectRead(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierComputeToVertexRead(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierAsBuildToRead(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount   = 1;
    dep.pMemoryBarriers      = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierAsBuildToTlasBuild(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount   = 1;
    dep.pMemoryBarriers      = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void blitToSwapchain(VkCommandBuffer cmd,
                     VkImage src, VkExtent2D srcExtent,
                     VkImageLayout srcCurrentLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkImage dst, VkExtent2D dstExtent) {
    transitionImage(cmd, src,
                    srcCurrentLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    srcStage, srcAccess,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    transitionImage(cmd, dst,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageBlit2 region{};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1]  = {static_cast<int32_t>(srcExtent.width),
                             static_cast<int32_t>(srcExtent.height), 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[1]  = {static_cast<int32_t>(dstExtent.width),
                             static_cast<int32_t>(dstExtent.height), 1};

    VkBlitImageInfo2 blitInfo{};
    blitInfo.sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blitInfo.srcImage       = src;
    blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blitInfo.dstImage       = dst;
    blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blitInfo.regionCount    = 1;
    blitInfo.pRegions       = &region;
    blitInfo.filter         = VK_FILTER_LINEAR;

    vkCmdBlitImage2(cmd, &blitInfo);

    transitionImage(cmd, dst,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

void blitToSwapchain(VkCommandBuffer cmd,
                     const Image& src,
                     VkImageLayout srcCurrentLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkImage dst, VkExtent2D dstExtent) {
    blitToSwapchain(cmd, src.vkImage(), src.extent(),
                    srcCurrentLayout, srcStage, srcAccess,
                    dst, dstExtent);
}

void clearImage(VkCommandBuffer cmd, VkImage image,
                VkClearColorValue clearValue, VkImageLayout targetLayout) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageSubresourceRange range{};
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = 0;
    range.levelCount     = 1;
    range.baseArrayLayer = 0;
    range.layerCount     = 1;

    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clearValue, 1, &range);

    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, targetLayout,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
}

void clearImage(VkCommandBuffer cmd, const Image& image,
                VkClearColorValue clearValue, VkImageLayout targetLayout) {
    clearImage(cmd, image.vkImage(), clearValue, targetLayout);
}

void transitionFromRTWrite(VkCommandBuffer cmd, VkImage image,
                           VkImageLayout targetLayout,
                           VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_GENERAL, targetLayout,
                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    dstStage, dstAccess);
}

void barrierQueueRelease(VkCommandBuffer cmd,
                         VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                         VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                         std::uint32_t srcFamily, std::uint32_t dstFamily) {
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask       = 0;
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.buffer              = buffer;
    barrier.offset              = offset;
    barrier.size                = size;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierQueueAcquire(VkCommandBuffer cmd,
                         VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                         std::uint32_t srcFamily, std::uint32_t dstFamily) {
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    barrier.srcAccessMask       = 0;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.buffer              = buffer;
    barrier.offset              = offset;
    barrier.size                = size;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierQueueRelease(VkCommandBuffer cmd,
                         VkImage image, VkImageLayout layout,
                         VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                         std::uint32_t srcFamily, std::uint32_t dstFamily) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask       = 0;
    barrier.oldLayout           = layout;
    barrier.newLayout           = layout; // preserve layout across ownership transfer
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.image               = image;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

void barrierQueueAcquire(VkCommandBuffer cmd,
                         VkImage image, VkImageLayout layout,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                         std::uint32_t srcFamily, std::uint32_t dstFamily) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    barrier.srcAccessMask       = 0;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.oldLayout           = layout;
    barrier.newLayout           = layout; // preserve layout across ownership transfer
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.image               = image;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace vksdl

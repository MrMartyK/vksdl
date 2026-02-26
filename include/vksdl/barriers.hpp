#pragma once

#include <vulkan/vulkan.h>

namespace vksdl {

class Image;

// Full-control image layout transition using VkImageMemoryBarrier2.
// Inserts a pipeline barrier with the exact stages and access masks you specify.
void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

// Convenience: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL.
// Use before rendering to a swapchain image.
void transitionToColorAttachment(VkCommandBuffer cmd, VkImage image);

// Convenience: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR.
// Use after rendering, before presenting a swapchain image.
void transitionToPresent(VkCommandBuffer cmd, VkImage image);

// Convenience: UNDEFINED -> GENERAL.
// Use before a compute shader writes via imageStore().
// Uses UNDEFINED as oldLayout -- discards previous contents.
void transitionToComputeWrite(VkCommandBuffer cmd, VkImage image);

// Convenience: GENERAL -> TRANSFER_SRC_OPTIMAL.
// Use after compute write, before vkCmdBlitImage as source.
void transitionToTransferSrc(VkCommandBuffer cmd, VkImage image);

// Convenience: UNDEFINED -> TRANSFER_DST_OPTIMAL.
// Use on swapchain image before vkCmdBlitImage as destination.
void transitionToTransferDst(VkCommandBuffer cmd, VkImage image);

// Convenience: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL.
// Use before rendering with a depth attachment. Safe to call every frame
// when loadOp=CLEAR (discards previous contents via UNDEFINED oldLayout).
void transitionToDepthAttachment(VkCommandBuffer cmd, VkImage image);

// GPU-driven rendering barriers (VkMemoryBarrier2, no layout transitions).

// After compute shader fills indirect/count buffer, before vkCmdDrawIndirect.
void barrierComputeToIndirectRead(VkCommandBuffer cmd);

// After compute shader writes vertex data, before vertex input reads it.
void barrierComputeToVertexRead(VkCommandBuffer cmd);

// Acceleration structure barriers (VkMemoryBarrier2, no layout transitions).

// After BLAS/TLAS build, before RT shader reads the AS.
void barrierAsBuildToRead(VkCommandBuffer cmd);

// After BLAS build, before TLAS build that references the BLAS.
void barrierAsBuildToTlasBuild(VkCommandBuffer cmd);

// Blit an offscreen image to a swapchain image, with barriers.
// Transitions src from srcCurrentLayout -> TRANSFER_SRC, dst from UNDEFINED -> TRANSFER_DST,
// performs the blit (LINEAR filter), then transitions dst -> PRESENT_SRC.
void blitToSwapchain(VkCommandBuffer cmd,
                     VkImage src, VkExtent2D srcExtent,
                     VkImageLayout srcCurrentLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkImage dst, VkExtent2D dstExtent);

void blitToSwapchain(VkCommandBuffer cmd,
                     const Image& src,
                     VkImageLayout srcCurrentLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkImage dst, VkExtent2D dstExtent);

// Clear a color image: transitions to TRANSFER_DST, clears, transitions to targetLayout.
void clearImage(VkCommandBuffer cmd, VkImage image,
                VkClearColorValue clearValue, VkImageLayout targetLayout);

void clearImage(VkCommandBuffer cmd, const Image& image,
                VkClearColorValue clearValue, VkImageLayout targetLayout);

// Convenience: GENERAL -> targetLayout after RT shader writes to a storage image.
// srcStage = RAY_TRACING_SHADER, srcAccess = SHADER_STORAGE_WRITE.
void transitionFromRTWrite(VkCommandBuffer cmd, VkImage image,
                           VkImageLayout targetLayout,
                           VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

} // namespace vksdl

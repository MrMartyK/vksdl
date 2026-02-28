#include <vksdl/graph/barrier_compiler.hpp>

namespace vksdl::graph {

VkDependencyInfo BarrierBatch::dependencyInfo() const {
    VkDependencyInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    info.imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size());
    info.pImageMemoryBarriers = imageBarriers.data();
    info.bufferMemoryBarrierCount = static_cast<std::uint32_t>(bufferBarriers.size());
    info.pBufferMemoryBarriers = bufferBarriers.data();
    info.memoryBarrierCount = static_cast<std::uint32_t>(memoryBarriers.size());
    info.pMemoryBarriers = memoryBarriers.data();
    return info;
}

void BarrierBatch::clear() {
    imageBarriers.clear();
    bufferBarriers.clear();
    memoryBarriers.clear();
}

bool isWriteAccess(VkAccessFlags2 access) {
    constexpr VkAccessFlags2 writeBits =
        VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    return (access & writeBits) != 0;
}

// Compute the source and destination stage/access for a barrier given the
// resource's current state (src) and the desired access (dst).
//
// The dst ResourceState is interpreted as:
//   - lastWriteStage/lastWriteAccess: the stage/access of the new operation
//   - currentLayout: the desired layout
//   - isRead: whether the new operation is a read or write
//
// Returns false if no barrier is needed.
struct BarrierParams {
    VkPipelineStageFlags2 srcStage;
    VkAccessFlags2 srcAccess;
    VkPipelineStageFlags2 dstStage;
    VkAccessFlags2 dstAccess;
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    bool needed;
};

static BarrierParams computeBarrier(const ResourceState& src, const ResourceState& dst,
                                    bool dstIsRead) {
    BarrierParams p{};
    p.oldLayout = src.currentLayout;
    p.newLayout = dst.currentLayout;
    p.needed = false;

    // The new operation's stage and access (stored in dst's write fields
    // by convention from the caller).
    p.dstStage = dst.lastWriteStage;
    p.dstAccess = dstIsRead ? dst.readAccessSinceWrite : dst.lastWriteAccess;

    // If dstAccess is zero but stored in the other field, pick the right one.
    // The caller encodes the desired access in either lastWriteAccess (for writes)
    // or readAccessSinceWrite (for reads) of the dst state.
    if (p.dstAccess == VK_ACCESS_2_NONE) {
        p.dstAccess = dstIsRead ? dst.lastWriteAccess : dst.readAccessSinceWrite;
    }

    bool hasWrite = isWriteAccess(src.lastWriteAccess);
    bool layoutChange = (p.oldLayout != p.newLayout);

    if (dstIsRead) {
        // New operation is a read.
        if (hasWrite) {
            // There was a prior write. Check if this read's stage is already
            // covered by previous readers since that write.
            if ((src.readStagesSinceWrite & p.dstStage) == p.dstStage && !layoutChange) {
                // This read's stage is fully covered by prior barriers.
                return p; // no barrier needed
            }

            if (src.readStagesSinceWrite != VK_PIPELINE_STAGE_2_NONE && !layoutChange) {
                // Prior readers already made the write visible. We only need
                // an execution dependency from the write stage (no memory dep).
                p.srcStage = src.lastWriteStage;
                p.srcAccess = VK_ACCESS_2_NONE;
                p.needed = true;
            } else {
                // First reader after write, or layout change. Need full
                // memory dependency to make write visible.
                p.srcStage = src.lastWriteStage;
                p.srcAccess = src.lastWriteAccess;
                p.needed = true;
            }
        } else {
            // No prior write. Only barrier needed if layout changes.
            if (layoutChange) {
                p.srcStage = (src.lastWriteStage != VK_PIPELINE_STAGE_2_NONE)
                                 ? src.lastWriteStage
                                 : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                p.srcAccess = VK_ACCESS_2_NONE;
                p.needed = true;
            }
            // Same layout, no write: no barrier.
        }
    } else {
        // New operation is a write.
        if (hasWrite || src.readStagesSinceWrite != VK_PIPELINE_STAGE_2_NONE) {
            // Must wait for prior writer AND all readers.
            p.srcStage = src.lastWriteStage | src.readStagesSinceWrite;
            p.srcAccess = src.lastWriteAccess | src.readAccessSinceWrite;
            if (p.srcStage == VK_PIPELINE_STAGE_2_NONE)
                p.srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            p.needed = true;
        } else if (layoutChange) {
            // No prior usage, but layout transition needed.
            p.srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            p.srcAccess = VK_ACCESS_2_NONE;
            p.needed = true;
        }
    }

    return p;
}

void appendImageBarrier(BarrierBatch& batch, const ImageBarrierRequest& req) {
    auto p = computeBarrier(req.src, req.dst, req.isRead);
    if (!p.needed)
        return;

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = p.srcStage;
    barrier.srcAccessMask = p.srcAccess;
    barrier.dstStageMask = p.dstStage;
    barrier.dstAccessMask = p.dstAccess;
    barrier.oldLayout = p.oldLayout;
    barrier.newLayout = p.newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = req.image;
    barrier.subresourceRange = VkImageSubresourceRange{
        req.aspect,           req.range.baseMipLevel,
        req.range.levelCount, req.range.baseArrayLayer,
        req.range.layerCount,
    };

    batch.imageBarriers.push_back(barrier);
}

void appendBufferBarrier(BarrierBatch& batch, const BufferBarrierRequest& req) {
    auto p = computeBarrier(req.src, req.dst, req.isRead);
    if (!p.needed)
        return;

    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = p.srcStage;
    barrier.srcAccessMask = p.srcAccess;
    barrier.dstStageMask = p.dstStage;
    barrier.dstAccessMask = p.dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = req.buffer;
    barrier.offset = req.offset;
    barrier.size = req.size;

    batch.bufferBarriers.push_back(barrier);
}

} // namespace vksdl::graph

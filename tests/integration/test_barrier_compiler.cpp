#include <vksdl/graph/barrier_compiler.hpp>

#include <cassert>
#include <cstdio>

using namespace vksdl::graph;

int main() {
    // 1. No-op: same src/dst state, same layout -> empty batch.
    {
        BarrierBatch batch;
        ResourceState state{};
        state.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = state,
                                      .dst = state,
                                      .isRead = true,
                                  });

        assert(batch.empty());
        std::printf("  no-op barrier: ok\n");
    }

    // 2. Read-to-read same layout, no prior write -> no barrier.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // No write recorded (lastWriteAccess = NONE).

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dst.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = true,
                                  });

        assert(batch.empty());
        std::printf("  read-to-read no write: ok\n");
    }

    // 3. Write-to-read -> execution + memory dep + layout transition.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.lastWriteStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        src.lastWriteAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        src.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dst.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = true,
                                  });

        assert(batch.imageBarriers.size() == 1);
        auto& b = batch.imageBarriers[0];
        assert(b.srcStageMask == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        assert(b.srcAccessMask == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        assert(b.dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        assert(b.dstAccessMask == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        assert(b.oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        assert(b.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        std::printf("  write-to-read: ok\n");
    }

    // 4. Write-to-write -> execution + memory dep.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.lastWriteStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src.lastWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        src.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        dst.lastWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = false,
                                  });

        assert(batch.imageBarriers.size() == 1);
        auto& b = batch.imageBarriers[0];
        assert(b.srcStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        assert(b.srcAccessMask == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        assert(b.dstStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        assert(b.dstAccessMask == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        std::printf("  write-to-write: ok\n");
    }

    // 5. UNDEFINED to color attachment -> layout transition with TOP_OF_PIPE.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dst.lastWriteAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = false,
                                  });

        assert(batch.imageBarriers.size() == 1);
        auto& b = batch.imageBarriers[0];
        assert(b.srcStageMask == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        assert(b.srcAccessMask == VK_ACCESS_2_NONE);
        assert(b.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        assert(b.newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        std::printf("  UNDEFINED to color attachment: ok\n");
    }

    // 6. Buffer barrier.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.lastWriteStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src.lastWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        dst.readAccessSinceWrite = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;

        appendBufferBarrier(batch, BufferBarrierRequest{
                                       .buffer = VK_NULL_HANDLE,
                                       .offset = 0,
                                       .size = VK_WHOLE_SIZE,
                                       .src = src,
                                       .dst = dst,
                                       .isRead = true,
                                   });

        assert(batch.bufferBarriers.size() == 1);
        auto& b = batch.bufferBarriers[0];
        assert(b.srcStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        assert(b.srcAccessMask == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        assert(b.dstStageMask == VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT);
        assert(b.dstAccessMask == VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
        std::printf("  buffer barrier: ok\n");
    }

    // 7. Batch accumulation.
    {
        BarrierBatch batch;

        // Add two image barriers.
        ResourceState src{};
        src.lastWriteStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        src.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        src.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dst.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = true,
                                  });
        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = true,
                                  });

        assert(batch.imageBarriers.size() == 2);
        assert(!batch.empty());

        auto dep = batch.dependencyInfo();
        assert(dep.sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        assert(dep.imageMemoryBarrierCount == 2);
        assert(dep.pImageMemoryBarriers == batch.imageBarriers.data());

        batch.clear();
        assert(batch.empty());

        std::printf("  batch accumulation: ok\n");
    }

    // 8. Multi-reader fan-out correctness.
    //
    // Scenario: pass A writes (fragment), pass B reads (compute),
    // pass C reads (fragment).
    //
    // After A->B barrier, readStagesSinceWrite = COMPUTE_SHADER.
    // C should still get an execution dep from A's write stage because
    // FRAGMENT_SHADER is not covered by readStagesSinceWrite.
    {
        BarrierBatch batch;

        // State after write (pass A) and first reader barrier (pass B).
        ResourceState afterBBarrier{};
        afterBBarrier.lastWriteStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        afterBBarrier.lastWriteAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        afterBBarrier.readStagesSinceWrite = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        afterBBarrier.readAccessSinceWrite = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        afterBBarrier.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Pass C wants to read in fragment shader.
        ResourceState dstC{};
        dstC.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dstC.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        dstC.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = afterBBarrier,
                                      .dst = dstC,
                                      .isRead = true,
                                  });

        // A barrier MUST be emitted (execution dep from write stage).
        assert(batch.imageBarriers.size() == 1);
        auto& b = batch.imageBarriers[0];
        assert(b.srcStageMask == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        // srcAccess should be 0 -- data already visible from A->B barrier.
        assert(b.srcAccessMask == VK_ACCESS_2_NONE);
        assert(b.dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        assert(b.dstAccessMask == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        std::printf("  multi-reader fan-out: ok\n");
    }

    // 9. isWriteAccess.
    {
        assert(isWriteAccess(VK_ACCESS_2_SHADER_WRITE_BIT));
        assert(isWriteAccess(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
        assert(isWriteAccess(VK_ACCESS_2_TRANSFER_WRITE_BIT));
        assert(isWriteAccess(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));
        assert(!isWriteAccess(VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
        assert(!isWriteAccess(VK_ACCESS_2_UNIFORM_READ_BIT));
        assert(!isWriteAccess(VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT));
        assert(!isWriteAccess(VK_ACCESS_2_NONE));
        std::printf("  isWriteAccess: ok\n");
    }

    // 10. Write-after-read: must wait for readers + writer.
    {
        BarrierBatch batch;
        ResourceState src{};
        src.lastWriteStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        src.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        src.readStagesSinceWrite =
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src.readAccessSinceWrite =
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        src.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        ResourceState dst{};
        dst.lastWriteStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        dst.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dst.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        appendImageBarrier(batch, ImageBarrierRequest{
                                      .image = VK_NULL_HANDLE,
                                      .range = {0, 1, 0, 1},
                                      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .src = src,
                                      .dst = dst,
                                      .isRead = false,
                                  });

        assert(batch.imageBarriers.size() == 1);
        auto& b = batch.imageBarriers[0];
        // srcStage should include writer AND all readers.
        assert(b.srcStageMask ==
               (VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
        assert(b.srcAccessMask ==
               (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
        std::printf("  write-after-read: ok\n");
    }

    std::printf("barrier compiler test passed\n");
    return 0;
}

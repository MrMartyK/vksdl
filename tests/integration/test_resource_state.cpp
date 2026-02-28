#include <vksdl/graph/resource_state.hpp>

#include <cassert>
#include <cstdio>

using namespace vksdl::graph;

// Helper to make state construction readable.
static ResourceState makeState(VkPipelineStageFlags2 writeStage, VkAccessFlags2 writeAccess,
                               VkImageLayout layout) {
    return ResourceState{
        .lastWriteStage = writeStage,
        .lastWriteAccess = writeAccess,
        .readStagesSinceWrite = VK_PIPELINE_STAGE_2_NONE,
        .readAccessSinceWrite = VK_ACCESS_2_NONE,
        .currentLayout = layout,
        .queueFamily = VK_QUEUE_FAMILY_IGNORED,
    };
}

int main() {
    // 1. Initial single slice.
    {
        ImageSubresourceMap map(5, 1);
        assert(map.sliceCount() == 1);

        auto state = map.queryState({0, 5, 0, 1});
        assert(state.lastWriteStage == VK_PIPELINE_STAGE_2_NONE);
        assert(state.lastWriteAccess == VK_ACCESS_2_NONE);
        assert(state.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        std::printf("  initial single slice: ok\n");
    }

    // 2. setState full range.
    {
        ImageSubresourceMap map(5, 1);
        auto newState = makeState(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        map.setState({0, 5, 0, 1}, newState);
        assert(map.sliceCount() == 1);

        auto queried = map.queryState({0, 5, 0, 1});
        assert(queried.lastWriteStage == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        assert(queried.lastWriteAccess == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        assert(queried.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        std::printf("  setState full range: ok\n");
    }

    // 3. setState single mip -> splits into 3 slices.
    {
        ImageSubresourceMap map(5, 1);
        auto newState = makeState(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        map.setState({2, 1, 0, 1}, newState);
        assert(map.sliceCount() == 3);

        // Mip 2 should have the new state.
        auto mip2 = map.queryState({2, 1, 0, 1});
        assert(mip2.lastWriteStage == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
        assert(mip2.currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // Mip 0 should have the original state.
        auto mip0 = map.queryState({0, 1, 0, 1});
        assert(mip0.lastWriteStage == VK_PIPELINE_STAGE_2_NONE);
        assert(mip0.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        // Mip 4 should have the original state.
        auto mip4 = map.queryState({4, 1, 0, 1});
        assert(mip4.lastWriteStage == VK_PIPELINE_STAGE_2_NONE);
        assert(mip4.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        std::printf("  setState single mip splits: ok\n");
    }

    // 4. Two disjoint ranges.
    {
        ImageSubresourceMap map(5, 1);
        auto stateA = makeState(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        auto stateB = makeState(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        map.setState({1, 1, 0, 1}, stateA);
        map.setState({3, 1, 0, 1}, stateB);

        auto q1 = map.queryState({1, 1, 0, 1});
        assert(q1.currentLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        auto q3 = map.queryState({3, 1, 0, 1});
        assert(q3.currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        auto q0 = map.queryState({0, 1, 0, 1});
        assert(q0.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        std::printf("  two disjoint ranges: ok\n");
    }

    // 5. queryState over mixed states -> union.
    {
        ImageSubresourceMap map(4, 1);

        ResourceState stateA{};
        stateA.lastWriteStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        stateA.lastWriteAccess = VK_ACCESS_2_SHADER_WRITE_BIT;
        stateA.currentLayout = VK_IMAGE_LAYOUT_GENERAL;

        ResourceState stateB{};
        stateB.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        stateB.lastWriteAccess = VK_ACCESS_2_SHADER_WRITE_BIT;
        stateB.readStagesSinceWrite = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        stateB.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        stateB.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        map.setState({0, 2, 0, 1}, stateA);
        map.setState({2, 2, 0, 1}, stateB);

        // Query spanning both.
        auto merged = map.queryState({0, 4, 0, 1});
        assert(merged.lastWriteStage ==
               (VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
        assert(merged.readStagesSinceWrite == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        assert(merged.readAccessSinceWrite == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        // Mixed layouts -> UNDEFINED.
        assert(merged.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        std::printf("  queryState union: ok\n");
    }

    // 6. setState overlapping existing split.
    {
        ImageSubresourceMap map(5, 1);

        // Split into mip0, mip1, mip2, mip3, mip4 with different states.
        auto stateA = makeState(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        map.setState({1, 1, 0, 1}, stateA); // mip1 -> stateA
        assert(map.sliceCount() == 3);

        // Now set a range that overlaps the split boundary.
        auto stateB =
            makeState(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        map.setState({0, 3, 0, 1}, stateB); // mips 0-2 -> stateB

        // Mips 0-2 should all be stateB.
        auto q0 = map.queryState({0, 1, 0, 1});
        assert(q0.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        auto q1 = map.queryState({1, 1, 0, 1});
        assert(q1.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        auto q2 = map.queryState({2, 1, 0, 1});
        assert(q2.currentLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Mips 3-4 should still be original.
        auto q3 = map.queryState({3, 1, 0, 1});
        assert(q3.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        std::printf("  setState overlapping split: ok\n");
    }

    // 7. Array layer splits.
    {
        ImageSubresourceMap map(1, 6); // 1 mip, 6 layers (cubemap).

        auto faceState = makeState(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        map.setState({0, 1, 2, 1}, faceState); // layer 2 only.
        assert(map.sliceCount() == 3);

        auto q2 = map.queryState({0, 1, 2, 1});
        assert(q2.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        auto q0 = map.queryState({0, 1, 0, 1});
        assert(q0.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        auto q5 = map.queryState({0, 1, 5, 1});
        assert(q5.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        // 2D split: mip + layer on a multi-mip cubemap.
        ImageSubresourceMap map2(4, 6);
        map2.setState({1, 2, 2, 3}, faceState); // mips 1-2, layers 2-4.

        auto qInside = map2.queryState({1, 1, 3, 1});
        assert(qInside.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        auto qOutside = map2.queryState({0, 1, 0, 1});
        assert(qOutside.currentLayout == VK_IMAGE_LAYOUT_UNDEFINED);

        std::printf("  array layer splits: ok\n");
    }

    // 8. aspectFromFormat.
    {
        assert(aspectFromFormat(VK_FORMAT_R8G8B8A8_UNORM) == VK_IMAGE_ASPECT_COLOR_BIT);
        assert(aspectFromFormat(VK_FORMAT_R8G8B8A8_SRGB) == VK_IMAGE_ASPECT_COLOR_BIT);
        assert(aspectFromFormat(VK_FORMAT_B8G8R8A8_UNORM) == VK_IMAGE_ASPECT_COLOR_BIT);
        assert(aspectFromFormat(VK_FORMAT_R16G16B16A16_SFLOAT) == VK_IMAGE_ASPECT_COLOR_BIT);

        assert(aspectFromFormat(VK_FORMAT_D32_SFLOAT) == VK_IMAGE_ASPECT_DEPTH_BIT);
        assert(aspectFromFormat(VK_FORMAT_D16_UNORM) == VK_IMAGE_ASPECT_DEPTH_BIT);
        assert(aspectFromFormat(VK_FORMAT_X8_D24_UNORM_PACK32) == VK_IMAGE_ASPECT_DEPTH_BIT);

        assert(aspectFromFormat(VK_FORMAT_D24_UNORM_S8_UINT) ==
               (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
        assert(aspectFromFormat(VK_FORMAT_D32_SFLOAT_S8_UINT) ==
               (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));

        assert(aspectFromFormat(VK_FORMAT_S8_UINT) == VK_IMAGE_ASPECT_STENCIL_BIT);

        std::printf("  aspectFromFormat: ok\n");
    }

    std::printf("resource state test passed\n");
    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>

// Demonstrates VK_KHR_unified_image_layouts benefit.
//
// A 4-pass compute chain processes each frame:
//   Pass 0: gen_pattern  -- animated colorful pattern -> image A
//   Pass 1: blur         -- box blur: image A -> image B
//   Pass 2: color_shift  -- animated hue rotation: image B -> image A
//   Pass 3: vignette     -- chromatic aberration + vignette: image A -> image B
//
// image B is then blitted to the swapchain, which leaves it in TRANSFER_SRC.
// After the blit, image B is transitioned back to GENERAL for the next frame.
//
// Without unified layouts: 2 extra UNDEFINED->GENERAL transitions per frame
// (one per image before their first compute use), because the spec requires
// explicit layout transitions when oldLayout=UNDEFINED.
//
// With unified layouts: those 2 transitions are skipped -- images stay in
// GENERAL permanently. Memory barriers between passes are still required
// (they flush shader caches, not layout changes), and the post-blit transition
// back to GENERAL is always needed (blitToSwapchain changes the tracked layout).

namespace {

// Helper: memory barrier between consecutive compute passes.
// Flushes SHADER_STORAGE_WRITE from the previous dispatch before the next
// dispatch reads via SHADER_STORAGE_READ. No layout transitions involved.
void computeToCompute(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// Build a storage image for the ping-pong chain.
vksdl::Image makeStorageImage(const vksdl::Allocator& allocator,
                               std::uint32_t w, std::uint32_t h) {
    return vksdl::ImageBuilder(allocator)
        .size(w, h)
        .format(VK_FORMAT_R8G8B8A8_UNORM)
        .storage()
        .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .build().value();
}

} // namespace

int main() {
    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Unified Layouts", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_unified_layouts")
        .requireVulkan(1, 3)
        .enableWindowSupport()
        .build().value();

    auto surface = vksdl::Surface::create(instance, window).value();

    auto device = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .build().value();

    bool unified = device.hasUnifiedImageLayouts();
    std::printf("GPU: %s\n", device.gpuName());
    std::printf("Unified image layouts: %s\n",
                unified ? "SUPPORTED -- layout transitions eliminated"
                        : "not supported -- conservative barriers active");

    auto swapchain = vksdl::SwapchainBuilder(device, surface)
        .size(window.pixelSize())
        .build().value();

    auto frames    = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    // Two ping-pong storage images.
    // With unified layouts: stay GENERAL permanently after initial transition.
    // Without unified layouts: transitioned UNDEFINED->GENERAL each frame.
    auto imageA = makeStorageImage(allocator,
                                   swapchain.extent().width,
                                   swapchain.extent().height);
    auto imageB = makeStorageImage(allocator,
                                   swapchain.extent().width,
                                   swapchain.extent().height);
    // Fresh images start UNDEFINED -- need initial transition regardless.
    bool needsInitialTransition = true;

    // Descriptor set for gen_pattern: single writeonly storage image (image A).
    auto dsGen = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // Descriptor sets for post-process passes:
    //   dsAB: binding 0 = image A (read), binding 1 = image B (write)
    //   dsBA: binding 0 = image B (read), binding 1 = image A (write)
    //   dsAB2: binding 0 = image A (read again), binding 1 = image B (write)
    //
    // Each DescriptorSetBuilder owns its own pool (sized for exactly one set),
    // so three separate builders are required.
    auto dsAB = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .addStorageImage(1, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    auto dsBA = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .addStorageImage(1, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // Pass 3 (vignette) reads A and writes B -- same layout as dsAB.
    // It needs its own descriptor set instance.
    auto dsAB2 = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .addStorageImage(1, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    auto updateDescriptors = [&]() {
        dsGen.updateImage(0, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        // blur: read A, write B
        dsAB.updateImage(0, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        dsAB.updateImage(1, imageB.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        // color_shift: read B, write A
        dsBA.updateImage(0, imageB.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        dsBA.updateImage(1, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        // vignette: read A, write B
        dsAB2.updateImage(0, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        dsAB2.updateImage(1, imageB.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
    };
    updateDescriptors();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    // gen_pattern pipeline: single storage image descriptor, writes image A.
    auto pipelineGen = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "gen_pattern.comp.spv")
        .descriptorSetLayout(dsGen.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // blur pipeline: two storage image descriptors.
    auto pipelineBlur = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "blur.comp.spv")
        .descriptorSetLayout(dsAB.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // color_shift pipeline: two storage image descriptors (same layout as blur).
    auto pipelineColorShift = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "color_shift.comp.spv")
        .descriptorSetLayout(dsBA.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // vignette pipeline: two storage image descriptors.
    auto pipelineVignette = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "vignette.comp.spv")
        .descriptorSetLayout(dsAB2.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    bool running  = true;
    vksdl::Event event;
    auto startTime = std::chrono::steady_clock::now();

    // Track layout transitions per frame for the window title.
    // Both paths have 1 mandatory post-blit transition (TRANSFER_SRC->GENERAL).
    // Without unified: +2 UNDEFINED->GENERAL transitions per frame = 3 total.
    // With unified: those 2 are eliminated = 1 total (post-blit only).
    int frameCount = 0;

    char titleBuf[256];

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
            imageA = makeStorageImage(allocator,
                                      swapchain.extent().width,
                                      swapchain.extent().height);
            imageB = makeStorageImage(allocator,
                                      swapchain.extent().width,
                                      swapchain.extent().height);
            updateDescriptors();
            needsInitialTransition = true;  // new images start UNDEFINED
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto   now  = std::chrono::steady_clock::now();
        float  time = std::chrono::duration<float>(now - startTime).count();

        std::uint32_t groupsX = (swapchain.extent().width  + 15) / 16;
        std::uint32_t groupsY = (swapchain.extent().height + 15) / 16;
        VkCommandBuffer cmd = frame.cmd;

        vksdl::beginOneTimeCommands(cmd);

        // First frame or without unified: UNDEFINED->GENERAL (discards contents,
        // fine because gen_pattern overwrites every pixel).
        // With unified (after first frame): already GENERAL -- skip.
        if (needsInitialTransition || !unified) {
            vksdl::transitionToComputeWrite(cmd, imageA.vkImage());
        }

        pipelineGen.bind(cmd, dsGen);
        pipelineGen.pushConstants(cmd, time);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Flush image A write before blur reads it.
        computeToCompute(cmd);

        // First frame or without unified: UNDEFINED->GENERAL.
        // With unified (after first frame): already GENERAL from post-blit
        // transition at the end of the previous frame.
        if (needsInitialTransition || !unified) {
            vksdl::transitionToComputeWrite(cmd, imageB.vkImage());
        }

        pipelineBlur.bind(cmd, dsAB);
        pipelineBlur.pushConstants(cmd, time);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Flush image B write before color_shift reads it.
        computeToCompute(cmd);

        pipelineColorShift.bind(cmd, dsBA);
        pipelineColorShift.pushConstants(cmd, time);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Flush image A write before vignette reads it.
        computeToCompute(cmd);

        pipelineVignette.bind(cmd, dsAB2);
        pipelineVignette.pushConstants(cmd, time);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // blitToSwapchain transitions image B to TRANSFER_SRC_OPTIMAL internally.
        vksdl::blitToSwapchain(cmd, imageB,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               img.image, swapchain.extent());

        // Transition image B back to GENERAL for next frame's compute passes.
        // This is always needed because blitToSwapchain leaves it in TRANSFER_SRC.
        vksdl::transitionImage(cmd, imageB.vkImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_READ_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        needsInitialTransition = false;

        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Update window title once per second with transition stats.
        ++frameCount;

        if (frameCount % 60 == 0) {
            if (unified) {
                std::printf("[frame %d] unified=yes, 1 transition/frame (2 eliminated), 4 compute passes\n",
                            frameCount);
                std::snprintf(titleBuf, sizeof(titleBuf),
                    "Unified Layouts | yes | "
                    "1 transition/frame (2 eliminated) | 4 compute passes");
            } else {
                std::printf("[frame %d] unified=no, 3 transitions/frame, 4 compute passes\n",
                            frameCount);
                std::snprintf(titleBuf, sizeof(titleBuf),
                    "Unified Layouts | no | "
                    "3 transitions/frame | 4 compute passes");
            }
            SDL_SetWindowTitle(window.sdlWindow(), titleBuf);
        }
    }

    device.waitIdle();

    return 0;
}

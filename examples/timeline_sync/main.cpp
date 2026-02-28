// Demonstrates TimelineSync with two GPU passes per frame synchronized by one
// timeline semaphore -- no fences needed.
//
// Pass 1 (compute): animated plasma pattern written to a storage image.
// Pass 2 (graphics): blit compute result to swapchain, then draw a spinning
//                    semi-transparent triangle overlay on top.
//
// Window title reports: Frame N | Timeline: V | 2 passes/frame, 0 fences

#include <SDL3/SDL.h>
#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("Timeline Sync", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_timeline_sync")
                        .requireVulkan(1, 3)
                        .enableWindowSupport()
                        .build()
                        .value();

    auto surface = vksdl::Surface::create(instance, window).value();

    auto device = vksdl::DeviceBuilder(instance, surface)
                      .needSwapchain()
                      .needDynamicRendering()
                      .needSync2()
                      .preferDiscreteGpu()
                      .build()
                      .value();

    auto swapchain =
        vksdl::SwapchainBuilder(device, surface).size(window.pixelSize()).build().value();

    // One timeline semaphore replaces N fences for CPU-GPU sync.
    auto sync = vksdl::TimelineSync::create(device, swapchain.imageCount()).value();

    auto allocator = vksdl::Allocator::create(instance, device).value();

    // Storage image that the compute shader writes to each frame.
    // TRANSFER_SRC_BIT needed for blitToSwapchain.
    auto buildStorageImage = [&]() {
        return vksdl::ImageBuilder(allocator)
            .size(swapchain.extent().width, swapchain.extent().height)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .storage()
            .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            .build()
            .value();
    };

    auto storageImage = buildStorageImage();

    auto descriptors = vksdl::DescriptorSetBuilder(device)
                           .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
                           .build()
                           .value();
    descriptors.updateImage(0, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto computePipeline = vksdl::ComputePipelineBuilder(device)
                               .shader(shaderDir / "timeline_pattern.comp.spv")
                               .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
                               .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
                               .build()
                               .value();

    // No vertex input -- positions are hardcoded in the vertex shader.
    auto overlayPipeline =
        vksdl::PipelineBuilder(device)
            .vertexShader(shaderDir / "overlay.vert.spv")
            .fragmentShader(shaderDir / "overlay.frag.spv")
            .colorFormat(swapchain)
            .enableBlending()
            .pushConstants<float>(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .build()
            .value();

    std::printf("TimelineSync: 1 semaphore, 0 fences, 2 passes/frame\n");
    std::printf("Pass 1: compute plasma  Pass 2: blit + triangle overlay\n");

    bool running = true;
    vksdl::Event event;
    std::uint32_t frameCount = 0;

    auto startTime = std::chrono::steady_clock::now();

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void) swapchain.recreate(device, window);

            storageImage = buildStorageImage();
            descriptors.updateImage(0, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        }

        auto [frame, img] = vksdl::acquireTimelineFrame(swapchain, sync, device, window).value();

        auto now = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        VkCommandBuffer cmd = frame.cmd;
        vksdl::beginOneTimeCommands(cmd);

        vksdl::transitionToComputeWrite(cmd, storageImage.vkImage());

        computePipeline.bind(cmd, descriptors);
        computePipeline.pushConstants(cmd, time);

        std::uint32_t groupsX = (swapchain.extent().width + 15) / 16;
        std::uint32_t groupsY = (swapchain.extent().height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Blit: storage image (GENERAL) -> swapchain (ends in PRESENT_SRC_KHR).
        vksdl::blitToSwapchain(cmd, storageImage, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, img.image, swapchain.extent());

        // blitToSwapchain leaves dst in PRESENT_SRC_KHR.
        // Transition to COLOR_ATTACHMENT_OPTIMAL to draw the overlay on top.
        // Must use PRESENT_SRC_KHR as oldLayout to preserve the blitted pixels.
        vksdl::transitionImage(cmd, img.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkExtent2D extent = swapchain.extent();

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = img.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // preserve blit
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderInfo{};
        renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea = {{0, 0}, extent};
        renderInfo.layerCount = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, extent};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        overlayPipeline.bind(cmd);
        overlayPipeline.pushConstants(cmd, time);
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRendering(cmd);

        vksdl::transitionToPresent(cmd, img.image);

        vksdl::endCommands(cmd);

        vksdl::presentTimelineFrame(device, swapchain, window, sync, frame, img,
                                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        ++frameCount;

        if (frameCount % 60 == 0) {
            std::printf("[frame %u] timeline=%llu  2 passes/frame, 0 fences\n", frameCount,
                        static_cast<unsigned long long>(sync.currentValue()));
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Timeline Sync | Frame %u | Timeline: %llu | "
                          "2 passes/frame, 0 fences",
                          frameCount, static_cast<unsigned long long>(sync.currentValue()));
            SDL_SetWindowTitle(window.sdlWindow(), title);
        }
    }

    device.waitIdle();

    return 0;
}

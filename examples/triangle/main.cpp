#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

// This is "real Vulkan" -- the part beginners are actually learning.

static void recordTriangle(VkCommandBuffer cmd, VkExtent2D extent,
                           VkImage swapImage, VkImageView swapView,
                           const vksdl::Pipeline& pipeline) {
    vksdl::transitionToColorAttachment(cmd, swapImage);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = swapView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = {{0, 0}, extent};
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    pipeline.bind(cmd);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    vksdl::transitionToPresent(cmd, swapImage);
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Triangle", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_triangle")
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

    auto swapchain = vksdl::SwapchainBuilder(device, surface)
        .size(window.pixelSize())
        .build().value();

    auto frames = vksdl::FrameSync::create(device, swapchain.imageCount()).value();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "triangle.vert.spv")
        .fragmentShader(shaderDir / "triangle.frag.spv")
        .colorFormat(swapchain)
        .build().value();

    bool running = true;
    vksdl::Event event;

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        vksdl::beginOneTimeCommands(frame.cmd);

        recordTriangle(frame.cmd, swapchain.extent(),
                       img.image, img.view,
                       pipeline);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    device.waitIdle();

    return 0;
}

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <filesystem>

// Demonstrates VK_EXT_device_fault detection and the recommended
// error-handling pattern for VK_ERROR_DEVICE_LOST. No actual fault
// is triggered -- the example prints whether the extension is supported
// and shows how to query fault info in error handlers.
//
// If your GPU supports VK_EXT_device_fault, the fault query function
// is automatically available. When a real device lost occurs, call
// device.queryDeviceFault() before destroying the device for useful
// diagnostic output.

static void recordTriangle(VkCommandBuffer cmd, VkExtent2D extent, VkImage swapImage,
                           VkImageView swapView, const vksdl::Pipeline& pipeline) {
    vksdl::transitionToColorAttachment(cmd, swapImage);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

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

    pipeline.bind(cmd);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    vksdl::transitionToPresent(cmd, swapImage);
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Device Fault", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_device_fault")
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

    auto frames = vksdl::FrameSync::create(device, swapchain.imageCount()).value();

    // Print device fault capability.
    std::printf("GPU: %s\n", device.gpuName());
    std::printf("VK_EXT_device_fault: %s\n",
                device.hasDeviceFault() ? "supported" : "not available");

    if (device.hasDeviceFault()) {
        // Query now (no fault has occurred, so empty).
        auto info = device.queryDeviceFault();
        std::printf("Current fault status: %s\n", info.empty() ? "(none)" : info.c_str());
    }

    std::printf("\nIn a real application, use this pattern:\n"
                "  VkResult vr = vkQueueSubmit(...);\n"
                "  if (vr == VK_ERROR_DEVICE_LOST) {\n"
                "      auto fault = device.queryDeviceFault();\n"
                "      fprintf(stderr, \"%%s\\n\", fault.c_str());\n"
                "  }\n\n");

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
                        .vertexShader(shaderDir / "triangle.vert.spv")
                        .fragmentShader(shaderDir / "triangle.frag.spv")
                        .colorFormat(swapchain)
                        .build()
                        .value();

    // Render loop with device-fault-aware error handling.
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
            (void) swapchain.recreate(device, window);
        }

        auto acquired = vksdl::acquireFrame(swapchain, frames, device, window);
        if (!acquired.ok())
            continue;

        auto [frame, img] = acquired.value();

        vksdl::beginOneTimeCommands(frame.cmd);

        recordTriangle(frame.cmd, swapchain.extent(), img.image, img.view, pipeline);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    device.waitIdle();

    return 0;
}

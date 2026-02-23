#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("dynamic state test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_dynamic_state")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .build();
    assert(device.ok());

    auto swapchain = vksdl::SwapchainBuilder(device.value(), surface.value())
        .size(window.value().pixelSize())
        .build();
    assert(swapchain.ok());

    auto frames = vksdl::FrameSync::create(device.value(),
                                            swapchain.value().imageCount());
    assert(frames.ok());

    std::filesystem::path shaderDir =
        std::filesystem::path(SDL_GetBasePath()) / "shaders";

    {
        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .dynamicCullMode()
            .dynamicTopology()
            .build();

        assert(pipeline.ok() && "pipeline with dynamic cull+topology failed");
        std::printf("  dynamic cull+topology pipeline: ok\n");
    }

    {
        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .dynamicCullMode()
            .dynamicDepthTest()
            .dynamicTopology()
            .dynamicFrontFace()
            .build();

        assert(pipeline.ok() && "pipeline with all dynamic states failed");
        std::printf("  all dynamic state pipeline: ok\n");
    }

    {
        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .dynamicCullMode()
            .dynamicTopology()
            .dynamicFrontFace()
            .build();
        assert(pipeline.ok());

        auto [frame, img] = vksdl::acquireFrame(swapchain.value(),
                                                 frames.value(),
                                                 device.value(),
                                                 window.value()).value();

        vksdl::beginOneTimeCommands(frame.cmd);

        vksdl::transitionToColorAttachment(frame.cmd, img.image);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView   = img.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderingInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea           = {{0, 0}, swapchain.value().extent()};
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;

        vkCmdBeginRendering(frame.cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapchain.value().extent().width);
        viewport.height   = static_cast<float>(swapchain.value().extent().height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapchain.value().extent()};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        pipeline.value().bind(frame.cmd);

        // Set dynamic states.
        vksdl::Pipeline::setCullMode(frame.cmd, VK_CULL_MODE_BACK_BIT);
        vksdl::Pipeline::setTopology(frame.cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vksdl::Pipeline::setFrontFace(frame.cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);

        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        // Switch states and draw again.
        vksdl::Pipeline::setCullMode(frame.cmd, VK_CULL_MODE_NONE);
        vksdl::Pipeline::setTopology(frame.cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);

        vksdl::transitionToPresent(frame.cmd, img.image);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device.value(), swapchain.value(), window.value(),
                            frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        std::printf("  dynamic state setters in cmd buffer: ok\n");
    }

    device.value().waitIdle();
    std::printf("dynamic state test passed\n");
    return 0;
}

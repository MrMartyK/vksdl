#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("timeline sync test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_timeline_sync")
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

    {
        auto sync = vksdl::TimelineSync::create(device.value(), 2);
        assert(sync.ok() && "TimelineSync creation failed");
        assert(sync.value().count() == 2);
        assert(sync.value().currentValue() == 0);
        assert(sync.value().vkTimelineSemaphore() != VK_NULL_HANDLE);
        std::printf("  create TimelineSync: ok\n");
    }

    {
        auto sync = vksdl::TimelineSync::create(device.value(), 2);
        assert(sync.ok());

        VkSemaphore original = sync.value().vkTimelineSemaphore();
        vksdl::TimelineSync moved = std::move(sync.value());
        assert(moved.vkTimelineSemaphore() == original);
        assert(moved.count() == 2);
        std::printf("  move semantics: ok\n");
    }

    {
        auto sync = vksdl::TimelineSync::create(device.value(), 2);
        assert(sync.ok());

        // First two frames should not wait (counter < count).
        auto f1 = sync.value().nextFrame();
        assert(f1.ok());
        assert(f1.value().value == 1);
        assert(f1.value().index == 0);
        assert(f1.value().cmd != VK_NULL_HANDLE);

        auto f2 = sync.value().nextFrame();
        assert(f2.ok());
        assert(f2.value().value == 2);
        assert(f2.value().index == 1);

        assert(sync.value().currentValue() == 2);
        std::printf("  nextFrame advances timeline: ok\n");
    }

    {
        auto sync = vksdl::TimelineSync::create(device.value(),
                                                  swapchain.value().imageCount());
        assert(sync.ok());

        std::filesystem::path shaderDir =
            std::filesystem::path(SDL_GetBasePath()) / "shaders";

        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .build();
        assert(pipeline.ok());

        auto [frame, img] = vksdl::acquireTimelineFrame(
            swapchain.value(), sync.value(),
            device.value(), window.value()).value();

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
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);

        vksdl::transitionToPresent(frame.cmd, img.image);

        vksdl::endCommands(frame.cmd);

        vksdl::presentTimelineFrame(device.value(), swapchain.value(),
                                     window.value(), sync.value(),
                                     frame, img,
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        std::printf("  full render cycle with timeline: ok\n");
    }

    device.value().waitIdle();
    std::printf("timeline sync test passed\n");
    return 0;
}
